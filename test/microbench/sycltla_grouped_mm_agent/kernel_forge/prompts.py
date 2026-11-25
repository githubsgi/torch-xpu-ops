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

Production (v4, torch-xpu-ops PR 2358f4c1; itself the output of an earlier agent league): 256x256x32 tile,
sub-groups 8x4 (32 sub-groups = 512 work-items), each sub-group owns a 32x64 accumulator block, prefetch
distance 3, persistent grid of 56 work-groups (one per Xe core). M-grouped kinds: split barrier every 2 K
tiles (kBarrierEvery 2) and idle sub-groups skip their loads/DPAS (kSkipIdleSg). grad_weight: same tile with
K step 16, barrier every K tile, no idle skip. All kinds: prefetch pipeline carried across persistent tiles,
8x32 bf16 output store atom (XE_STORE_2D<16,8,32>), device-side wave-tail split of the last partial wave into
4 (or 2) narrower N sub-tiles.
Measured on one PVC tile (unitrace, 16 experts x ~190 rows): w13_fwd 184-210 us, XMX 69%, XVE 77% active /
14% stalled, occupancy 46%; w13_dW 274 us, XMX 53%, SBID stall 13%, ~23% of samples are register shuffles that
re-pair the transposed A operand, ~18% per-tile setup, ~15% serialized epilogue stores. HBM and L3 are far
from their limits: the kernels are bound by latency, per-tile fixed cost, tile fill and wave quantization.
Already measured and NOT worth repeating on this baseline: tile sweep (256x256 beats 256x128 / 128x128 /
128x256 on every m-grouped op; 16B dW gains <= 5% from 256x128 or 128x256), prefetch distance (3 optimal;
2 and 4 within noise, 6-8 slower), wider 2D loads (slower and incorrect), payload-per-message tricks (IGC
merges them back), a 224-row tile, 4x8 sub-group layout, K step 64 (spills).
Not yet tried on this baseline: LSC cache hints on the A / B loads and prefetches (Cfg kCacheA / kCacheB /
kCachePf in the extended header), GRF-128 mode with a 16x64 or 32x32 slab and 64 sub-groups per work-group
(kGrf), barrier span / idle skip on grad_weight, tile order or swizzle for L3 reuse, split-K for grad_weight,
explicit load block shapes, a cheaper dW epilogue store chain.
Tile configs that compile: tile M/sgM a multiple of 8, tile N/sgN a multiple of 16; <= 32 sub-groups in
GRF-256, <= 64 in GRF-128.

# Hardware card (Intel Data Center GPU Max 1550 "PVC", one tile in FLAT mode)
- 56 Xe cores usable (448 XVEs; max_compute_units 448). Each Xe core: 8 XVEs, 8 XMX engines, shared L1/SLM.
- bf16 DPAS (XMX) matrix instructions (atom XE_DPAS_TT<8, ...>); practical dense ceiling ~220-250 TFLOPS per tile (production kernels
  reach 130-150 TF on fwd/dX, 80-90 TF on dW).
- Sub-group size 16, GRF 256 (large register file, grf_size<256>): half the hardware threads per XVE, so one
  512-work-item work-group fills an Xe core. GRF 128 (Cfg kGrf = 128) allows 1024 work-items per core but
  halves the registers: the 32x64 slab (128 accumulator registers) spills, use 16x64 or 32x32. Register spills
  are reported by the build and are very costly.
- LSC cache hints (vISA): loads default to the driver policy; .ca.ca caches in L1 and L3, .st.ca streams through
  L1 (data read once per core), .uc.ca bypasses L1, .uc.uc bypasses both. Prefetch is .ca.ca; .uc.ca
  prefetches into L3 only. The extended header exposes them as Cfg kCacheA / kCacheB / kCachePf (0 default,
  1 ca.ca, 2 st.ca, 3 uc.ca, 4 uc.uc; kCachePf 0/1).
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
kTileM/N/K, kPrefetchDist, optional members) is generated from configs.json. Optional per-kind keys become Cfg
members: "barrier_every" -> kBarrierEvery, "skip_idle" -> kSkipIdleSg, "cache_a" / "cache_b" -> kCacheA / kCacheB,
"cache_pf" -> kCachePf, "grf" -> kGrf (see the hardware card). Use "extra" (C++ member lines) for new
compile-time knobs your header reads from Cfg, or "body" to replace the struct body.
Rules: tile/config choice must NOT depend on host-known routed row counts (all EP ranks must run the same
kernel); device-side logic reading `offs` is allowed. fp32 accumulation, bf16 output. Must stay correct for
ragged / empty groups, single rows, tile-edge sizes (the evaluator verifies 28 hidden cases against an fp32
reference and checks every timed output).

# configs.json
{"m_rowB": {"tile":[M,N,K], "sg":[sgM,sgN], "prefetch":p, "wgs":0, "barrier_every":2, "skip_idle":true,
            optional "cache_a":0-4, "cache_b":0-4, "cache_pf":0-1, "grf":128|256, "extra":"...", "body":"..."},
 "m_colB": {...}, "k_grouped": {...}, "extra_flags": []}   ("wgs" 0 = Xe cores x 512 / work-group threads;
 in GRF-128 mode set it explicitly, e.g. 112 for 512-thread work-groups)
Kinds that share an identical config share one build. The evaluator is the ONLY judge: you cannot run it; the
harness builds, verifies and benchmarks your edit after you finish.

# Ideas worth trying (non-exhaustive; prefer ONE focused, well-motivated change)
grad_weight: a cheaper epilogue store chain (the 8x32 stores wait on one shared header register), a cheaper
re-pairing of the transposed A operand than the generic reorder(), split-K across sub-group rows (SLM
reduction), barrier span 2 and idle skip, better tile ordering. fwd/dX: tile order / swizzle for L3 reuse of B,
a scheduler that interleaves groups' tails, cache hints that keep the shared operand in L1 and stream the
other, prologue prefetch before the first tile's group lookup, GRF-128 with 64 sub-groups, hoisting index
math. Large groups: larger-N or 512-row tiles, cache hints (the weights no longer fit L1 per core).
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
(tile size, sub-group layout, prefetch distance, tile_k, grid size, barrier span, idle skip, cache hints,
GRF mode) are explored separately by a cheap mutation operator, and parallel revisions of this parent tend to
pick the same obvious knob: your revision should be a CODE change to grouped_mm_kernel_impl.h (plus the
configs.json edits it needs).
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
