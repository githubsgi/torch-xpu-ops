# Final code review (Humanize review phase)

You are a read-only reviewer. The RLCR loop has ended ({{REASON}}). Review the final candidate for code quality and risk: `git diff {{BASE_COMMIT}} -- cand/` (candidate vs the production snapshot at loop start), `reference/` for production, `CLAUDE.md` for the constraints, `{{LOOP_DIR}}/final-eval.txt` for the driver's full-suite measurement.

Check: out-of-bounds or uninitialised accesses on ragged / empty groups / K == 0 / offs edge values; int32 overflow in index math (use int64 for row*ld); dependence on host-known row counts; use of scratch memory or ABI changes; barrier divergence (all sub-groups must reach barrier_arrive/wait the same number of times); register/SLM budget; comments that no longer match the code; leftover debug code; configs.json consistent with the header.

Report every finding with a severity marker: `[P0]` must fix (wrong results / hangs / UB), `[P1]` should fix (latent bug, constraint violation), `[P2]` minor, `[P3]` nit. If there is nothing at P0/P1, say `No blocking findings.` End with the line `REVIEW-DONE`.
