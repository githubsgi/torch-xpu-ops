---
name: xpu-profile
description: Use before deciding what to optimize in the PVC grouped GEMM, when an op is slower than expected or a change has an effect you do not understand, and whenever you are about to run `ggemm-eval profile` or read unitrace output. Enforces Profile -> Diagnose -> Plan with evidence (metric values), maps unitrace counters and stall classes to a diagnosis playbook, and generates REPORT.md with 3-5 ranked priorities (evidence, expected gain, effort, why now) via `xpu-report`. XPU port of ncu-report-skill. Never guess the bottleneck.
---

# Skill: xpu-profile (unitrace in place of Nsight Compute)

**Golden rule: Profile -> Diagnose -> Plan, in that order. Never guess.** Do not hypothesize before you have the profile, do not code a fix before you matched the observed pattern to a diagnosis, do not write a wall of suggestions - rank by evidence and estimated impact (3-5 priorities).

## Quickstart

0. **One directory per profiling run**: `profile/<run-id>/` (e.g. `profile/r2_dW_base/`); never reuse one.
1. **Decide what you profile.** Pick (case, op) from the weakest rows of the latest `last_eval.txt` per-op table. Cases: `tp4ep4_uniform` (16 experts x ~190 rows, the small-group regime), `ep8_uniform` (8 x ~6k), also `tp4ep4_hot|sparse|balanced|rows64|rows8`, `ep8_hot|balanced|rows1024`. Ops: `w13_fwd w2_fwd w13_dX w2_dX w13_dW w2_dW`. Profile the candidate AND the production baseline (`production/` = an untouched `ggemm-eval init` copy) when you need a before/after.
2. **Collect** (1-2 min each; the candidate must build, run `ggemm-eval run cand --suite quick` first):
   ```bash
   ggemm-eval profile cand --case tp4ep4_uniform --op w13_dW | tee profile/<run>/profile_tp4ep4_uniform_w13_dW.txt
   ```
   It prints counters (XVE active/stall/idle, occupancy, XMX utilization, HBM/L3 GB/s, clock), the stall-sample mix, kernel properties (spills, GRF) and rule hits, plus `JOB: <dir>` with the raw unitrace logs (`ut_metrics.log`, `ut_stalls.log`, `profile_result.json`).
3. **Report**:
   ```bash
   xpu-report make --profile profile/<run>/profile_*.txt --cand cand --out profile/<run>/REPORT.md --title "w13_dW at 190 rows/expert"
   xpu-report check profile/<run>/REPORT.md
   ```
   The script adds what the digest lacks: tiles per launch, waves, wave efficiency, row fill, K tiles per tile, and ranked rule hits with evidence. It leaves `TODO(agent)` placeholders - **replace them with your judgement** (overrule a heuristic gain when the counters say otherwise, and say so). `check` must pass before you cite the report.
4. **Match to the playbook** (`reference/playbook.md`) and read the linked wiki pages (`kw get <id>`).
5. **Plan**: the report's priorities become the candidate directions in `docs/draft.md` / the round contract. Each tried direction gets a before/after profile (same case/op) so keep/revise/reject rests on counters, not only on the score.

## Critical lessons

1. Counters are medians over 20 launches of ONE (case, op); stall shares are from a separate pass. Compare profiles taken in the same session; check the `clock MHz` row (a throttled run is not comparable).
2. Name the 2-3 metric values behind a conclusion. "Latency-bound on 2D loads: SBID 35%, XMX 34%, HBM 297 GB/s of 1180" is a diagnosis; "it looks memory-bound" is not.
3. Check the input regime before profiling: rows per expert decide tile fill, waves and K tiles per tile. The report computes them from the case's counts.
4. The `ggemm-eval run` table decides promotion; the profile explains. A profile cannot override a failed correctness check.
5. HBM/L3 ceilings (1180 / 2840 GB/s measured) were never reached by any op: do not propose bandwidth fixes unless the report shows > 800 GB/s HBM or > 2 TB/s L3 (`pat-memory-bound-expected`).
6. Do not delegate understanding: open the JOB logs when a number looks odd.

## Files
- `reference/playbook.md` - pattern -> signals -> why -> first-line fix -> deeper fixes -> exceptions, in unitrace terms with reference values.
- `reference/report-template.md` - REPORT.md structure and style rules (what `xpu-report` generates and what you add).
- `reference/metrics.md` - metric and stall-class glossary (also `kw get hw-unitrace-metrics`).
- `scripts/make_report.py` - `xpu-report make|check`.
