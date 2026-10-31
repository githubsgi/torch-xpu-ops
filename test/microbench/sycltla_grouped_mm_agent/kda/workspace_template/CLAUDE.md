# Task contract: optimize the SYCL-TLA grouped GEMM on Intel PVC

You are a kernel design agent (KDA flow: draft -> plan -> implement/validate/profile -> review). This workspace is the task workspace; the evaluator and the production kernel live outside it. Work here, keep every claim backed by a measurement.

## Objective
Make `torch._grouped_mm` on Intel PVC faster for the DeepSeek-V3 16B MoE layer by editing the candidate in `cand/`:
`cand/grouped_mm_kernel_impl.h` (the persistent SYCL-TLA/CuTe grouped GEMM: `gemm_tile`, `GroupCursor`/`Scheduler`, `MGroupedKernel` for forward ('R' B K-contiguous) and grad_input ('C' B N-contiguous), `KGroupedKernel` for grad_weight, `launch` with sub_group_size 16 and grf_size<256>) and `cand/configs.json` (per kind `m_rowB` / `m_colB` / `k_grouped`: `{"tile":[M,N,K],"sg":[sgM,sgN],"prefetch":p,"wgs":0}`, optional `"extra"` (extra Cfg members) / `"body"` (whole Cfg struct body), top-level `"extra_flags"`). `reference/` holds the production header, `grouped_mm_kernels.h` (the fixed ABI) and `cand_cfg.cpp` (the contract). `best/` is the best promoted candidate so far (starts as production); `cand/` is your working copy.

## Inputs and outputs
bf16 (fp16 also compiled by torch, evaluated in bf16). M-grouped: A [R,K], B_g, D [R,N], `offs` int32 cumulative on device partitions the rows. K-grouped (grad_weight): A [R,M] (M-contiguous), B [R,N], D [E,M,N], `offs` partitions the contraction. fp32 accumulation, bf16 output. Six GEMMs per layer: w13_fwd, w2_fwd, w13_dX, w2_dX, w13_dW, w2_dW (hidden 2048, expert FFN 1408; w13 fuses 2x1408). Regimes: TP4/EP4 = 16 experts x ~190 rows (`small` suite), EP8 = 8 experts x ~6k rows (`large`).

## Correctness (non-negotiable)
`ggemm-eval run` verifies 22 hidden cases with a fresh seed each run (ragged, empty and all-empty groups, single rows, tile-edge sizes, non-DSv3 shapes) against a naive fp32 device reference, inside NaN sentinels (out-of-bounds writes and stale outputs fail), and checks every timed output against the baseline. A candidate with `CORRECTNESS: FAIL` or `STATUS != ok` has no score. Run `--verify-only` after every edit before benchmarking.

## Constraints
- `Params` and `grouped_mm_kernels.h` are the ABI torch uses: no workspace pointer, no new fields. Designs needing scratch memory (classic split-K/stream-K reductions, global atomic queues, prepare kernels) are out of scope; no allocations in the launch path.
- Tile/config choice must NOT depend on host-known routed row counts: every EP rank must run the same kernel. A static config per kind is fine; device-side logic reading `offs` is fine.
- Keep the contract in `reference/cand_cfg.cpp`: `run_m_grouped_cfg<Cfg,DType,LayoutB>`, `run_k_grouped_cfg<Cfg,DType>`, `TiledMmaFor<T,Cfg>` (or `Cfg::kThreads`) in `sycltla::grouped_gemm`.
- Only edit files in `cand/` (and the records below). Never touch `reference/`, `best/`, `src/`, the evaluator, or other candidates' snapshots. Do not run `git commit`/`git push` (the driver commits each round).
- Do not silently redefine the target or the baseline: SCORE is production/candidate time (geomean over scored cases) from the shared evaluator.

## Commands
```bash
ggemm-eval run cand --suite quick --verify-only       # correctness only (~20-60 s)
ggemm-eval run cand --suite quick                     # tp4ep4_uniform + ep8_uniform
ggemm-eval run cand --suite small|large|full [--ops w13_dW,w2_dW]
ggemm-eval profile cand --case tp4ep4_uniform --op w13_dW   # unitrace digest + raw logs (JOB dir)
ggemm-eval diff cand                                  # candidate vs production
ggemm-eval init DIR [--from cand]                     # scratch copies for A/B experiments
ggemm-eval slots                                      # GPU/build slot usage (shared with other jobs)
kw query|get|grep ...                                 # kernel-wiki skill CLI
xpu-report make|check ...                             # xpu-profile skill CLI
kda-record bench|cand|show ...                        # benchmark.csv / candidates.jsonl records
```
Cold build ~45-60 s, cached ~0 s; quick ~15-30 s; small/full 1-2 min; profile 1-2 min. Long commands are fine: run them in the foreground. Noise: production vs itself is 0.98-1.03 per TP4/EP4 case and +-0.2% for EP8, so **less than 2% is noise** - repeat or use more cases before believing a small win.

## Evidence rules
- Every perf-related change gets a row in `benchmark.csv`: `kda-record bench --id <cand-id> --direction "<what>" [--parent <id>] --note "<result in words>"` right after the `ggemm-eval run` that measured it (it reads `cand/last_eval.json`).
- Every candidate is a node of the DAG in `candidates.jsonl`: `kda-record cand --id cN --parent cM|production --status proposed|kept|rejected --direction "..." --note "..." [--snapshot cand]`. A rejected candidate needs the reason with numbers. At most 5 iterations per direction, then keep/revise/reject with before/after numbers and profile evidence.
- Profile before you decide (xpu-profile skill: Profile -> Diagnose -> Plan, REPORT.md with 3-5 ranked priorities in `profile/<run>/`). Look things up in the kernel-wiki skill before guessing hardware facts; cite page ids.
- Lessons: when something non-obvious cost you time or a rule proved right/wrong, add a lesson to `.humanize/bitlesson.md` (strict template there) and report it in `## BitLesson Delta` of the round summary.

## Promotion criteria (applied by the driver, not by you)
A candidate is promoted to `best/` only when: `STATUS: ok` with `CORRECTNESS: PASS` on the `full` suite, **SCORE >= 1.02**, and **no op regresses by more than 3%** (per-op geomean >= 0.97), and no new register spills. An improvement below 2% is noise and is reported as NEUTRAL. A round that ends with no promotable candidate is still a valid outcome if the summary contains the profile and benchmark evidence for why the ranked directions do not pay off.

## Layout
`docs/draft.md`, `docs/plan.md`, `docs/wiki-feedback.md` - plans and wiki write-backs. `profile/<run>/` - profiling runs. `benchmark.csv`, `candidates.jsonl`, `candidates/<id>/` - evidence. `.humanize/` - RLCR loop state (read-only for you except `bitlesson.md`, and `goal-tracker.md` mutable section as instructed). `prompts/` - phase prompts. `.claude/skills/` - kernel-wiki, xpu-profile.
