# BitLesson Knowledge Base

This file is project-specific. Keep entries precise and reusable for future rounds.

## Entry Template (Strict)

Use this exact field order for every entry:

```markdown
## Lesson: <unique-id>
Lesson ID: <BL-YYYYMMDD-short-name>
Scope: <component/subsystem/files>
Problem Description: <specific failure mode with trigger conditions>
Root Cause: <direct technical cause>
Solution: <exact fix that resolved the problem>
Constraints: <limits, assumptions, non-goals>
Validation Evidence: <tests/commands/logs/PR evidence>
Source Rounds: <round numbers where problem appeared and was solved>
```

## Entries

<!-- Add lessons below using the strict template. -->

## Lesson: bl-seed-noise
Lesson ID: BL-20261030-noise-floor
Scope: ggemm-eval scores, any config change
Problem Description: A 1-2% SCORE change gets reported as a win or a regression.
Root Cause: Production vs itself gives 0.98-1.03 per TP4/EP4 case (+-0.2% EP8); the tuner sweeps in the notes had ~10% run-to-run noise.
Solution: Treat < 2% as noise; repeat the run and use `small` or `full` (more cases) before keeping or rejecting; record the spread in benchmark.csv notes.
Constraints: applies to the shared evaluator on one PVC tile.
Validation Evidence: sycltla_grouped_mm_agent/README.md "SCORE"; notes_ggemm.txt 16.2.
Source Rounds: seed

## Lesson: bl-seed-rowcount
Lesson ID: BL-20261030-no-host-rowcount
Scope: tile/config selection, dispatch
Problem Description: A tile or kernel is selected from host-known routed row counts.
Root Cause: Different EP ranks then run different kernels in the same step; 3 of 4 mixed-mode EP8 runs crashed (notes 19.2).
Solution: Use a static config per kind, or device-side logic that reads `offs`.
Constraints: the Params ABI has no workspace pointer either (no scratch-based designs).
Validation Evidence: notes_ggemm.txt 19.2.
Source Rounds: seed

## Lesson: bl-seed-verify-first
Lesson ID: BL-20261030-verify-before-bench
Scope: every kernel edit
Problem Description: Time is spent benchmarking an edit that is incorrect on ragged/empty groups.
Root Cause: Verification uses 22 hidden cases with NaN sentinels; K==0 groups in the K-grouped kernel must store zeros, empty M groups must write nothing.
Solution: `ggemm-eval run cand --suite quick --verify-only` after every edit, before any benchmark.
Constraints: none.
Validation Evidence: sycltla_grouped_mm_agent/README.md "verify".
Source Rounds: seed
