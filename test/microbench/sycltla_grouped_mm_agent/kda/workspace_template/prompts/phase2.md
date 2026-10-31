# Phase 2 - profile-guided exploration

Develop a faster SYCL-TLA grouped GEMM for Intel PVC while preserving numerical correctness. Target speedup: **{{TARGET_SPEEDUP}}x** SCORE on the `full` suite (no op regressing > 3%, no new spills).

{{COMMON}}

## Phase 2 goal
Start from the best correct implementation (`best/`; `cand/` is initialised from it). Phase 2 is an exploration phase: use unitrace profiling (xpu-profile), the kernel-wiki, the SYCL-TLA headers in `/mnt/aimod/subratag/sycl-tla` and the notes facts to identify as many plausible optimization directions as possible, then explore them systematically.

The draft must list the candidate directions, rank them by expected benefit and implementation risk, and split each into concrete subtasks. Consider at least: prefetch distance per kind; per-kind tile shape and sub-group layout (128-row tiles for dW, 256x128 / 128x256 / 128x128 for M-grouped, N-tile choice for N=1408); wave quantization and tile order (m-major vs n-major linearization, heavy-first, finer last wave); per-tile overhead (descriptor hoisting, epilogue simplification, cross-tile prefetch); register double buffering and K-step size; intra-work-group K slicing for grad_weight; scheduler cost (offs read once per work-group); work-group count (`wgs`); GRF/occupancy; device-side per-group specialization (rank-independent); compiler flags. Skip directions the constraints rule out (scratch memory, host-row-count-dependent choices) unless you can show a scratch-free variant.

Explore each direction for **at most five iterations**. If a direction cannot be implemented cleanly, fails correctness, or shows no credible path to improvement after those iterations, record the evidence (numbers, profile) in `candidates.jsonl` as rejected and move to the next ranked direction. For each explored direction collect before/after `ggemm-eval run` results on the relevant suites (at least `quick`, and `small`/`large` for anything that touches that regime) and enough profile evidence (same case/op before and after) to justify keep, revise or reject.

Combine winners only after each has been measured alone. Keep `cand/` equal to the best measured combination at the end of a round.
