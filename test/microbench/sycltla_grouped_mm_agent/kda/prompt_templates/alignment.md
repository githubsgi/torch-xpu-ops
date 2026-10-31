
## Part 2b: FULL GOAL ALIGNMENT AUDIT (mandatory checkpoint, round {{ROUND}})
Completed iterations: {{COMPLETED}} (rounds 0..{{ROUND}}). Audit:
1. Acceptance criteria table for EVERY AC in the immutable section (status, evidence, blocker/justification).
2. Forgotten items: plan tasks neither Active, Completed nor Deferred; tasks claimed complete but unverified.
3. Deferred items: still justified? contradicting the Ultimate Goal?
4. Direction ledger: are ranked directions explored in order, at most 5 iterations each, with decisions and evidence? Any direction silently dropped?
5. Mainline drift: is the current objective singular and plan-derived? `Blocking Side Issues: N | Queued Side Issues: N`.
6. Progress stagnation check: read the previous review results and summaries (`{{LOOP_DIR}}/round-*-review-result.md`, `round-*-summary.md`) and decide if development has stalled (same issues repeating, no AC progress, no new code changes, circular discussion). If stalled, the last line must be STOP.
7. Has the target been silently redefined (different baseline, different suite, different threshold)? That is a defect.
