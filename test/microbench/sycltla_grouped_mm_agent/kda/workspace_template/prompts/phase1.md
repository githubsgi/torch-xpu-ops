# Phase 1 - baseline, correctness and understanding

Develop a faster SYCL-TLA grouped GEMM for Intel PVC while preserving numerical correctness. Target speedup for the whole effort: **{{TARGET_SPEEDUP}}x** SCORE on the `full` suite (no op regressing > 3%).

{{COMMON}}

## Phase 1 goal
Start from the production kernel (`cand/` is initialised from it). Phase 1 produces a trustworthy baseline and the understanding that later phases need; performance matters, but correctness and a clean measured baseline are the priority.
1. Read `cand/grouped_mm_kernel_impl.h`, `reference/grouped_mm_kernels.h`, `reference/cand_cfg.cpp`, the wiki pages `ker-ggemm-production`, `ker-ggemm-constraints`, `lang-sycl-tla-cute-xe-api`, `hw-pvc-*`.
2. Verify the untouched candidate: `ggemm-eval run cand --suite quick --verify-only`, then `--suite full`. Record it as candidate `production` / `c0` (SCORE ~1.0 is the noise floor; note the spread you see per case: that is your noise estimate).
3. Profile the three kernel kinds at the small-group regime (`tp4ep4_uniform`: one fwd, one dX, one dW op) and the large-group regime (`ep8_uniform`, at least dW) with the xpu-profile skill; write `profile/phase1_baseline/REPORT.md` with 3-5 ranked priorities.
4. Write `docs/draft.md`: baseline numbers, risks/unknowns, candidate directions ranked by expected benefit and risk (each cited to wiki ids and the report's evidence), the first concrete steps, exact validation/evaluation commands, and the evidence needed to keep/revise/reject each direction.
Only low-risk config experiments (one or two `configs.json` knob changes, e.g. prefetch per kind) are expected to be tried in this phase, to calibrate the harness; the kernel body is changed from phase 2.
