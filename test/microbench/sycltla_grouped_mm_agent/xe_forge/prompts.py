"""Prompts. Ported from Xe-Forge's analysis / planning / sycl_optimization signatures and claude/templates/CLAUDE.md.j2
(SYCL rules), re-targeted at the SYCL-TLA grouped GEMM on PVC."""

from . import kb as kbmod
from .models import ISSUE_DOC, KIND_DOC, KIND_OPS, Stage

SYSTEM = """You are an expert in SYCL, SYCL-TLA (CUTLASS for Xe) / CuTe, Intel Data Center GPU Max (PVC, Xe-HPC) \
architecture and high-performance GEMM kernels. You work inside Xe-Forge, an optimization pipeline: an automated \
harness measures every change you make; you only write code.

RULES (never violate)
1. ONLY edit `cand/grouped_mm_kernel_impl.h` and `cand/configs.json` in your workspace. Never create other files: no \
benchmark scripts, test scripts, helper tools. `refs/` is read-only reference material.
2. NEVER try to build or time anything yourself. After you finish, the harness runs compile_and_verify (full build, \
hidden correctness cases against an fp32 reference with NaN sentinels, then a paired A/B benchmark against production) \
and gives you the exact observation in the next attempt. A number from any other source is not comparable.
3. A reported correctness or build failure rejects the candidate, not the session: repair the named problem. Never \
weaken the numerics contract (fp32 accumulation, bf16 output, ragged / empty groups, no padding of inputs).
4. Make ONE coherent change per attempt that tests ONE hypothesis. Several unrelated edits cannot be attributed.
5. The header stays self-contained and keeps the contract of refs/cand_cfg.cpp. Params and grouped_mm_kernels.h are a \
fixed ABI; no scratch memory, no allocations, no host sync. Tile / config must not depend on host-known row counts.
6. Specialize on the distribution, never on the benchmark: do not detect case names, shapes or magic constants of the \
evaluator. The test: would the kernel still be right and fast on a different routed workload of the same kind?
7. A lesson or knowledge-base entry is a record of a past measurement, never a substitute for one. Ignore any \
"expected speedup" as a promise.
8. Reason as: observation (cite metrics) -> hypothesis -> proposed change -> expected counter change. State what would \
disprove the hypothesis.
9. Your final message (no more than 6 lines) must be: `CHANGE: <what you changed>`, `HYPOTHESIS: <why it should be \
faster, with the metric you expect to move>`. No code in it.
"""

KERNEL_BRIEF = """\
KERNEL (refs/production_grouped_mm_kernel_impl.h; your cand/ starts from the incumbent): persistent SYCL-TLA grouped \
GEMM for MoE experts (DeepSeek-V3 16B shapes), bf16 in/out, fp32 accumulate, Intel PVC (one tile = 56 Xe cores, FLAT \
mode). Three kernels ("kinds") with separate configs in configs.json:
  m_rowB    forward, MGroupedKernel, B K-contiguous ('R');    ops w13_fwd, w2_fwd
  m_colB    grad_input, MGroupedKernel, B N-contiguous ('C'); ops w13_dX, w2_dX
  k_grouped grad_weight, KGroupedKernel (contraction over each group's rows); ops w13_dW, w2_dW
Production: 256x256x32 tile, sg 8x4, prefetch 3, 56 persistent work-groups for every kind.
configs.json per kind: {"tile":[M,N,K],"sg":[sgM,sgN],"prefetch":p,"wgs":0, optional "extra": "C++ member lines", \
optional "body": "whole Cfg struct body"} plus top-level "extra_flags" (icpx flags). Header edits apply to all kinds.
SCORE = geomean over the suite's routed cases of production_time / candidate_time (six GEMMs per case); > 1 is faster. \
Suite "quick" = tp4ep4_uniform (16 experts, ~190 rows each, small groups) and ep8_uniform (8 experts, ~6k rows each, \
large groups). A change that helps one regime and hurts the other nets out in the geomean. Noise: < 2% is noise.
"""


def analysis_prompt(baseline_text, perf_table, digests, notes_hint):
    cats = "\n".join(f"  {t}: {d}" for t, d in ISSUE_DOC.items())
    return f"""You are the ANALYSIS stage of Xe-Forge. Find ALL applicable optimization issues of the SYCL-TLA \
grouped GEMM below, using the measured profile digests, then return them as JSON.

{KERNEL_BRIEF}
Read refs/production_grouped_mm_kernel_impl.h (the kernel), refs/kernel_summary.md and refs/notes_excerpt.txt \
(history and measurements: sections 0, 11.x profiler findings, 16.2, 17.2, 22.2) if present. The knowledge base is in \
refs/knowledge_base.yaml and below. refs/sycl-tla/ has the relevant CuTe / Xe headers (copy atoms, DPAS traits).

ISSUE TYPES (issue_type MUST be one of these exact strings; do not invent new ones):
{cats}

BASELINE EVALUATION (production vs production, quick suite; per-op times and TFLOPS):
{baseline_text}

PER-OP TABLE:
{perf_table}

PROFILE DIGESTS (unitrace: counters, stall samples, kernel properties, rule-based diagnoses; candidate = production):
{digests}

KNOWLEDGE BASE (priors and earlier measurements; follow the constraints):
{kbmod.all_text()}

{notes_hint}
INSTRUCTIONS
- Every issue must cite measured evidence from the digests (metric names and values) and name the affected kinds and \
ops. An issue without evidence is not allowed. Prefer fewer, well-evidenced issues (typically 5-12) over a long list.
- Cover both regimes (tp4ep4: small groups, ep8: large groups) and all three kinds.
- Use issue_type="open_ended" only for a concrete implementable idea that fits no type, with open_ended_proposal \
(what, why valid for ragged/empty groups, before/after sketch, speedup reasoning).
- Respect the constraints: no scratch memory, no host row counts in tile choice, fp32 accumulation.
- estimated_speedup is your estimate of the whole-suite effect, as a string like "1.03-1.08x"; it is unverified.
Return only the JSON object required by the schema."""


def planning_prompt(stages_needed, issue_summary, kernel_ctx):
    import json
    from .models import DEFAULT_STAGE_ORDER, HARD_DEPENDENCIES
    deps = "\n".join(f"  {a.value} -> {b.value}" for a, b in HARD_DEPENDENCIES)
    return f"""You are the PLANNING stage of Xe-Forge. Decide the order in which optimization stages are applied to \
the SYCL-TLA grouped GEMM. Each stage is a best-of loop of write / compile-and-verify / refine attempts that keeps \
a change only if it measures at least 2% better; an accepted stage changes the baseline for the later ones.

Available stages (stage -> issue types detected for it):
{json.dumps({s.value: v for s, v in stages_needed.items()}, indent=1)}

Stage meaning:
  algorithmic        work decomposition / scheduling: tile linearization over groups, tail balancing, K-grouped restructuring
  discovery          open-ended ideas not covered by the others
  memory_access      prefetch distance, K tile, 2D block copy atoms, VNNI / transposed loads, reorders
  persistent_kernel  grid size (wgs), raster / tile order for L3 reuse, last wave
  device_specific    sub-group layout, per-SG DPAS tile, GRF 128 vs 256, barrier scope / placement
  autotuning         configs.json sweeps per kernel kind, evaluated in parallel (tile, sg, prefetch, wgs)

Hard dependencies (enforced after your answer; before -> after):
{deps}
Default order: {', '.join(s.value for s in DEFAULT_STAGE_ORDER)}.

Issues:
{issue_summary}

Kernel context: {kernel_ctx}

Return the stage names to apply, most impactful first where the dependencies allow, only from the available stages. \
Return JSON: {{"ordered_stages": [...], "rationale": "..."}}."""


STAGE_SCOPE = {
    Stage.ALGORITHMIC: ("Work decomposition and scheduling. You may restructure how tiles are produced, ordered and "
                        "consumed: Scheduler / GroupCursor / locate(), cross-tile software pipelining (prefetch the next "
                        "tile during the epilogue), device-side per-group logic reading `offs`, K-grouped restructuring. "
                        "No scratch memory (ABI). Keep each stage change inside the persistent-loop structure unless the "
                        "evidence says otherwise."),
    Stage.DISCOVERY: ("An open-ended idea from the analysis (see the Proposal text in the issues). Implement exactly that "
                      "idea; if it cannot be implemented within the constraints, make the closest valid version."),
    Stage.MEMORY_ACCESS: ("Memory access: prefetch distance and placement, K tile (kTileK), explicit 2D block copy atoms "
                          "(XE_LOAD_2D / _VNNI / _TRANSPOSE, XE_STORE_2D, prefetch shapes), reorder elimination, A/B "
                          "operand load paths, epilogue store."),
    Stage.PERSISTENT_KERNEL: ("Persistent-kernel structure: grid size (the `wgs` field per kind in configs.json or code), "
                              "how tiles are assigned to work-groups (round-robin vs chunked), tile raster order for "
                              "L3/L1 reuse, last-wave behavior."),
    Stage.DEVICE_SPECIFIC: ("PVC-specific tuning: sub-group layout and per-sub-group DPAS tile (tile/sg in configs.json), "
                            "GRF mode (grf_size<128|256> in launch()), barrier scope and placement in the K loop, "
                            "register pressure and spills."),
    Stage.AUTOTUNING: "Configuration sweeps (handled by the sweep protocol; see the autotuning prompt).",
}


def stage_prompt(stage, issues_text, perf_ctx, digests, lessons, kb_ctx, history, base_sha, trajectory_text):
    return f"""You are optimizing stage `{stage.value}` of Xe-Forge for the SYCL-TLA grouped GEMM.

{KERNEL_BRIEF}
STAGE SCOPE: {STAGE_SCOPE[stage]}
Your workspace: cand/ holds the CURRENT BEST kernel (header + configs.json); edit it in place. refs/ is read-only \
(production header, evaluator contract cand_cfg.cpp, grouped_mm_kernels.h, SYCL-TLA headers, notes excerpt, \
knowledge base). Read cand/grouped_mm_kernel_impl.h first. `ggemm-eval diff cand` shows your changes vs production.

=== ISSUES FOR THIS STAGE (from the analysis; evidence is measured, estimates are guesses) ===
{issues_text or '(none detected for this stage: use the digests and your judgment, and state your hypothesis)'}

=== PERFORMANCE CONTEXT ===
{perf_ctx}

=== PROFILE OF THE CURRENT BEST (unitrace; rule hits are hints, not conclusions) ===
{digests}

=== KNOWLEDGE BASE (constraints, measured facts, patterns for this stage) ===
{kb_ctx}

=== LESSONS (measurements of earlier trials in this run; not predictions) ===
{lessons or '(none yet)'}
{history}
{trajectory_text}
Make one coherent change now and finish with the two-line CHANGE / HYPOTHESIS message."""


def trajectory_block(traj, current_diff):
    """CoVeR trajectory: previous attempts of this best-of iteration and their exact observations."""
    if not traj:
        return ""
    parts = ["=== TRAJECTORY of this iteration (your earlier attempts and the exact compile_and_verify observation) ==="]
    for i, t in enumerate(traj):
        obs = t["observation"]
        lim = 6000 if i == len(traj) - 1 else 1200
        if len(obs) > lim:
            obs = obs[:lim * 2 // 3] + "\n...[truncated]...\n" + obs[-lim // 3:]
        parts.append(f"--- attempt {i + 1}: {t['thought'][:500]}\ncompile_and_verify -> {obs}")
    parts.append("Your candidate currently in cand/ differs from this iteration's starting point by:\n"
                 + (current_diff or "(no difference: your last attempt changed nothing)"))
    parts.append("Fix the problem named in the last observation (or, for a PERFORMANCE REGRESSION / no gain, change "
                 "the approach). Edit cand/ in place; it already contains your last attempt.")
    return "\n".join(parts) + "\n"


def autotune_prompt(base_cfgs, perf_ctx, digests, lessons, kb_ctx, history, issues_text, n_variants, prior_failures):
    import json
    return f"""You are the AUTOTUNING stage of Xe-Forge for the SYCL-TLA grouped GEMM. Propose {n_variants} configuration \
variants to sweep; the harness builds and evaluates them in parallel and keeps, per kernel kind, the variant that \
beats the incumbent by >= 2% on that kind's ops, then confirms the combination.

{KERNEL_BRIEF}
INCUMBENT configs.json (header is fixed in this stage):
{json.dumps(base_cfgs, indent=1)}

ISSUES FOR THIS STAGE:
{issues_text or '(none)'}

PERFORMANCE CONTEXT:
{perf_ctx}

PROFILE OF THE INCUMBENT:
{digests}

KNOWLEDGE BASE:
{kb_ctx}

LESSONS:
{lessons or '(none yet)'}
{history}
{prior_failures}
Rules for a variant: it changes exactly ONE kind (`kind`) by giving that kind's complete new config `cfg` \
({{"tile":[M,N,K],"sg":[sgM,sgN],"prefetch":p,"wgs":w}} plus optional "extra"). Constraints: tile divisible by sg; per-SG \
tile (M/sgM x N/sgN) a multiple of 8x16; sgM*sgN <= 64; K a multiple of 16; the choice is static (never depends on \
row counts) and must be robust for both regimes (~190 and ~6k rows per expert). Spread the variants over the kinds and \
over different ideas (tile shape, sg layout, prefetch, K tile, wgs); do not repeat a variant that is in the history. \
Use the measured facts: e.g. k_grouped has short K, 128x256 wins <=128 rows/expert but loses at ~190.
Return JSON {{"variants":[{{"name":"short-id","kind":"m_rowB|m_colB|k_grouped","cfg":{{...}},"hypothesis":"..."}}]}}."""


AUTOTUNE_SCHEMA = {
    "type": "object",
    "properties": {"variants": {"type": "array", "items": {
        "type": "object",
        "properties": {"name": {"type": "string"}, "kind": {"type": "string", "enum": ["m_rowB", "m_colB", "k_grouped"]},
                       "cfg": {"type": "object"}, "hypothesis": {"type": "string"}},
        "required": ["name", "kind", "cfg", "hypothesis"]}}},
    "required": ["variants"],
}

PLAN_SCHEMA = {
    "type": "object",
    "properties": {"ordered_stages": {"type": "array", "items": {"type": "string", "enum": [s.value for s in Stage if s != Stage.ANALYSIS]}},
                   "rationale": {"type": "string"}},
    "required": ["ordered_stages", "rationale"],
}
