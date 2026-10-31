## Kernel information
- Kernel: persistent SYCL-TLA/CuTe grouped GEMM for `torch._grouped_mm` on Intel PVC (one tile of a Data Center GPU Max 1550), bf16 operands, fp32 accumulation, bf16 output. Candidate: `cand/` (editable `grouped_mm_kernel_impl.h` + `configs.json`).
- Operations (DeepSeek-V3 16B MoE layer; hidden 2048, expert FFN 1408): `w13_fwd`, `w2_fwd` (M-grouped, B K-contiguous), `w13_dX`, `w2_dX` (M-grouped, B N-contiguous), `w13_dW`, `w2_dW` (K-grouped).
- Baseline: the production kernel at its production config (256x256x32 tile, sg 8x4, prefetch 3, 56 work-groups for all kinds), timed in the same process as the candidate by `ggemm-eval run`.
- Known facts (notes_ggemm.txt, wiki `ker-ggemm-*`): TP4/EP4 (~190 rows/expert) forward/dX 126-150 TFLOPS, grad_weight 84-88 TFLOPS (XMX 34%, SBID stalls 37%, latency-bound: ~6 K tiles per output tile); at ~6k rows/expert the per-group dense fallback is 4-13% faster than the kernel and dW reaches 129-138 TFLOPS; 128x256 M tiles win by up to 42% at <= 128 rows/expert but lose ~10% at ~190 rows; HBM/L3 are never the limit.

## Acceptance (the shared evaluator decides)
- Correctness: `ggemm-eval run cand --suite quick --verify-only` -> `CORRECTNESS: PASS` (22 hidden cases, fresh seed per run). Do not weaken or work around it.
- Development workloads: `quick` (tp4ep4_uniform + ep8_uniform) while iterating; `small` for TP4/EP4 changes, `large` for EP8; `full` after any promising change and before claiming a win. SCORE = geomean of production/candidate time over the scored cases. Less than 2% is noise.

## Workflow requirements
- Write the plan draft to `docs/draft.md` first; implementation starts only from the executable plan (`docs/plan.md`).
- Record every performance-related change in `benchmark.csv` (`kda-record bench`) and every candidate in `candidates.jsonl` as a DAG (`kda-record cand`, parents = the candidate it was derived from, `production` is the root).
- Keep profile records (`profile/<run>/REPORT.md` from the xpu-profile skill) for each major optimization direction.
- Use the kernel-wiki skill for PVC hardware, SYCL-TLA API, CuTe-to-SYCL-TLA migration, techniques and patterns; cite page ids. Use the xpu-profile skill for every profiling step (Profile -> Diagnose -> Plan; never guess).
- Look at the constraints in CLAUDE.md: no scratch memory / workspace, no host-row-count-dependent choices, fp32 accumulation, only edit `cand/`.
- Do not silently redefine the target or the baseline. If the target cannot be reached, produce the benchmark and profile evidence that explains why.
{{GUIDANCE}}
