"""Prompts of the LLM revision action: operator card, hardware card, per-expansion context."""

OPERATOR_CARD = """\
# Operator card: SYCL-TLA grouped GEMM behind torch._grouped_mm on Intel PVC

One persistent kernel launch per call. `offs` (int32, cumulative group ends) is read ON THE DEVICE, so there
is no host sync. bf16 inputs, fp32 accumulation (DPAS atom XE_DPAS_TT<8, float, bf16, bf16>), bf16 output.
Ragged and empty groups must work without padding (out-of-bounds loads zero-fill, stores are masked).

Three kernel kinds, each with its own tile config in configs.json:
| kind        | used for                | what it is |
| m_rowB      | forward (w13_fwd, w2_fwd)   | M-grouped: offs partitions the ROWS of A and D; B[e] is [N, K] read K-contiguous ('R') |
| m_colB      | grad_input (w13_dX, w2_dX)  | M-grouped, B[e] read N-contiguous ('C') |
| k_grouped   | grad_weight (w13_dW, w2_dW) | K-grouped: offs partitions the CONTRACTION dim; D[e] is [M, N]; A is read transposed |

The six GEMMs of one DeepSeek-V3 16B MoE layer per EP rank (hidden 2048, expert inner 1408):
  w13_fwd  x[M,2048] . w13[E,2816,2048]^T -> [M,2816]      w2_fwd  h[M,1408] . w2[E,2048,1408]^T -> [M,2048]
  w13_dX   dg[M,2816] . w13[E] -> [M,2048]                   w2_dX   dy[M,2048] . w2[E] -> [M,1408]
  w13_dW   dg^T[2816,M] . x[M,2048] -> [E,2816,2048]         w2_dW   dy^T[2048,M] . h[M,1408] -> [E,2048,1408]
Cases: TP4/EP4 = 16 local experts x ~190 rows (sparse / hot / balanced variants, probes at 8 and 64 rows);
EP8 = 8 local experts x ~6k rows. The score is the geomean speedup vs production over the suite's routed cases.

Production: 256x256x32 tile, sub-groups 8x4 (32 sub-groups = 512 work-items), each sub-group owns a 32x64
accumulator block, prefetch distance 3, persistent grid of 56 work-groups (one per Xe core), same for all kinds.
Measured on one PVC tile (unitrace): dX 216 us, XMX 70%, SBID stall 7%; forward 223 us, XMX 65%, SBID 14%;
grad_weight 299 us, XMX 36%, stalled 33% (SBID 31%): the weakest op (80-90 TFLOPS at ~190 rows/expert, ~6 K tiles
of 32 per expert, so the prologue and the 256x256 output write are not hidden). HBM and L3 are far from their
limits (15-18% of GTI): the kernels are bound by latency, tile fill and wave quantization, not bandwidth.
Known tuner facts: 128x256 tiles win by up to 40% at <=128 rows/expert but lose ~10% at ~190 rows; prefetch 2
wins grad_weight; at ~6k rows/expert a dense per-group GEMM is 4-13% faster than 256x256 (a better large-group tile
is open). Tile configs that compile: tile M/sgM a multiple of 8, tile N/sgN a multiple of 16; tuner configs
include 256x128 sg8x2, 128x256 sg4x4, 128x128 sg4x2, 512x128 sg16x2, 256x256 sg4x8, tile_k 64 with prefetch 2.

# Hardware card (Intel Data Center GPU Max 1550 "PVC", one tile in FLAT mode)
- 56 Xe cores usable (448 XVEs; max_compute_units 448). Each Xe core: 8 XVEs, 8 XMX engines, shared L1/SLM.
- bf16 DPAS (XMX) matrix instructions (atom XE_DPAS_TT<8, ...>); practical dense ceiling ~220-250 TFLOPS per tile (production kernels
  reach 130-150 TF on fwd/dX, 80-90 TF on dW).
- Sub-group size 16, GRF 256 (large register file, grf_size<256>): half the hardware threads per XVE, so one
  512-work-item work-group fills an Xe core. Register spills are reported by the build and are very costly.
- Data movement: Xe 2D block loads (plain, VNNI or transposed, chosen from the tensor strides), 2D block
  prefetch into L1/L3, barrier_arrive / barrier_wait split barriers. Scoreboard (SBID) stalls mean a
  sub-group waits for its own loads: deepen/shorten prefetch, move loads earlier, or give each sub-group more
  independent DPAS work per load.
- 128 KB SLM per Xe core; L3 ~2.8 TB/s and HBM ~1.2 TB/s per tile (Advisor roofline measurements).

# The kernel header and what is fixed
`grouped_mm_kernel_impl.h` (in the working dir) is the production file: `gemm_tile` (2D block loads + prefetch +
DPAS mainloop), `GroupCursor` / `Scheduler` (linearize tiles over groups from `offs`), `MGroupedKernel`,
`KGroupedKernel`, `launch`. Contract the evaluator relies on (do not break): `sycltla::grouped_gemm::
run_m_grouped_cfg<Cfg, DType, LayoutB>(queue, params, wgs)`, `run_k_grouped_cfg<Cfg, DType>(queue, params, wgs)`
and `TiledMmaFor<T, Cfg>` (or Cfg::kThreads) must still exist; `Params` and grouped_mm_kernels.h are a FIXED ABI
(no workspace pointer: no scratch memory, no allocations in the launch path). The Cfg struct (WGTile, SGLayout,
kTileM/N/K, kPrefetchDist, optional extra members) is generated from configs.json; use "extra" (C++ member lines)
for new compile-time knobs your header reads from Cfg, or "body" to replace the struct body.
Rules: tile/config choice must NOT depend on host-known routed row counts (all EP ranks must run the same
kernel); device-side logic reading `offs` is allowed. fp32 accumulation, bf16 output. Must stay correct for
ragged / empty groups, single rows, tile-edge sizes (the evaluator verifies 22 hidden cases against an fp32
reference and checks every timed output).

# configs.json
{"m_rowB": {"tile":[M,N,K], "sg":[sgM,sgN], "prefetch":p, "wgs":0, optional "extra":"...", "body":"..."},
 "m_colB": {...}, "k_grouped": {...}, "extra_flags": []}   ("wgs" 0 = Xe cores x 512 / work-group threads)
Kinds that share an identical config share one build. The evaluator is the ONLY judge: you cannot run it; the
harness builds, verifies and benchmarks your edit after you finish.

# Ideas worth trying (non-exhaustive; prefer ONE focused, well-motivated change)
grad_weight: split the contraction across sub-group rows inside the work-group (SLM reduction), smaller
prefetch, K-contiguous copy atom for the transposed activation, 128-row tiles, better tile ordering.
fwd/dX: tile order / swizzle for L2 reuse of B, scheduler that interleaves groups' tails (reduce the empty last
wave), different tile for the three kinds, prologue prefetch before the first tile's group lookup, fewer
barriers, larger per-sub-group blocks, hoisting index math. Large groups: larger-N or 512-row tiles.
"""

SYSTEM = ("You are a GPU kernel optimization engineer improving one SYCL-TLA grouped GEMM header for Intel "
          "PVC. You work in the current directory only: edit grouped_mm_kernel_impl.h and/or configs.json "
          "(the two files that make up the candidate). You cannot run the benchmark; the harness evaluates "
          "your edit afterwards. Make ONE focused, well-motivated change per revision; do not rewrite the file "
          "wholesale. Never create other files that the build would need.\n\n" + OPERATOR_CARD)

NOTE_FORMAT = ("When you are done, your final message must be a short note (2-4 sentences, no code fences) "
               "starting with 'CHANGE:' that says what you changed and why you expect it to be faster.")


FOCI = [
    "grad_weight (K-grouped) short-K mainloop: prologue / prefetch structure, load vs barrier ordering, hiding "
    "the 256x256 output write",
    "tile scheduling: tile order / swizzle across groups (L2 reuse of B and A), balancing the last wave over "
    "the work-groups",
    "forward / dX (M-grouped) mainloop: placement of loads, prefetch and reorder, barrier spacing, copy atoms "
    "for the 'R' versus 'C' B views",
    "per-kind specialization: compile-time branches driven by Cfg members (configs.json 'extra') so each kind "
    "gets its own mainloop variant",
    "large groups (EP8, ~6k rows/expert): mainloop structure or K step better suited to long K",
    "free choice: whatever the profile digest and evaluation show as the biggest inefficiency",
]


def revision_prompt(*, node, focus, lineage_text, parent_eval_text, profile_text, profile_target, siblings_text,
                    recent_failures_text, cur_cfg_text):
    return f"""\
You are revising candidate {node.id} of a revision tree (Monte-Carlo tree search over kernel versions). The
working directory holds its header and configs; your edit becomes a NEW child candidate.

## Lineage of the parent (root = production kernel)
{lineage_text}

## Parent's current configs.json
{cur_cfg_text}

## Parent's last evaluation (search suite; SCORE > 1 is faster than production, noise < 2%)
{parent_eval_text}

## Profile digest of the parent's weakest op ({profile_target})
{profile_text}

## Sibling attempts under this parent (do not repeat these; learn from them)
{siblings_text}

## Recent failures elsewhere in the tree
{recent_failures_text}

## Task
Propose and implement one change to the parent that you expect to raise the score. Pure knob changes
(tile size, sub-group layout, prefetch distance, tile_k, grid size) are explored separately by a cheap
mutation operator, and parallel revisions of this parent tend to pick the same obvious knob: your revision
should be a CODE change to grouped_mm_kernel_impl.h (plus the configs.json edits it needs).
Suggested focus for this revision (not mandatory if the evidence points elsewhere): {focus}.
Edit the files in place (use Read/Edit/Write). Prefer
changes that compile first time; use only constructs already used in the header or in the sycl-tla sources
(read-only copy available through --add-dir) if you need examples.
{NOTE_FORMAT}"""


def repair_prompt(*, status, error_text, prior_note, attempt, max_attempts, diff_stat):
    return f"""\
Your previous edit to this candidate did not pass the harness (repair attempt {attempt} of {max_attempts}).

STATUS: {status}
{error_text}

Your earlier note: {prior_note}
Diff stat vs the parent: {diff_stat}

Fix the problem with the smallest edit that keeps the intent of the change (or revert the part that cannot
work). Read the current files first. Correctness is non-negotiable: verification compares the full output
against an fp32 reference including empty groups, ragged sizes and tile-edge sizes.
{NOTE_FORMAT}"""
