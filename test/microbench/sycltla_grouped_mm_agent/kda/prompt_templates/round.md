{{HEADER}}

## Original Implementation Plan
Read `{{PLAN_FILE}}` (the plan you implement) and the goal tracker `{{GOAL_TRACKER_FILE}}` before anything else.

{{PHASE_PROMPT_BLOCK}}

---

## Round {{ROUND}} contract (REQUIRED FIRST STEP)
Before writing code or running evals, write `{{CONTRACT_FILE}}` containing:
- exactly one **mainline objective** for this round (one optimization direction or one analysis task from the plan),
- the 1-2 target ACs,
- which issues are truly **blocking** for that objective, and which are **queued** (explicitly out of scope),
- concrete success criteria (numbers: which suite, which op, what ratio, which correctness gate).
Do not start implementation until the contract exists.

## Task lane rules
Tag every task in the goal tracker "Active Tasks" with one lane: `[mainline]` (advances this round's objective), `[blocking]` (prevents the objective from succeeding), `[queued]` (non-blocking; document, do not do). Do not let queued work replace the objective.

## BitLesson selection (REQUIRED)
Read `.humanize/bitlesson.md` before each task. In your summary, list the lesson IDs that applied (or NONE).

## Skills (REQUIRED, they are the research and diagnosis path; the driver audits the hook log and the reviewer fails rounds that skip them)
Order of work: (1) call the `Skill` tool with `kernel-wiki`, then `kw query/get` for this round's direction; (2) call the `Skill` tool with `xpu-profile` and follow Profile -> Diagnose -> Plan; (3) only then edit and measure. Shell notes: only the allowed commands run (no python, no loops, no variables): edit `cand/configs.json` with the Edit/Write tools, copy scratch candidates with `ggemm-eval init DIR --from cand`.
- Invoke the `kernel-wiki` skill (Skill tool) and look up pages with `kw` for every direction you plan or revise; cite ids.
- Invoke the `xpu-profile` skill before deciding what to change next in any non-trivial step: Profile (ggemm-eval profile) -> Diagnose (xpu-report, playbook) -> Plan. Keep `profile/<run>/REPORT.md` for each major direction.
- Record every perf-related change: `kda-record bench ...` after each measuring `ggemm-eval run`; every candidate in the DAG: `kda-record cand ...` (rejections need the reason and numbers).

## Budget and pacing
Remaining LLM budget for the whole run is limited; this round has a hard cap of ${{WORKER_BUDGET}}. Prefer the `quick` suite while iterating and `small`/`large`/`full` for decisions. At most five iterations per direction. Finish by writing the summary (below) even if the direction failed.

{{BODY}}

---

## Summary (REQUIRED, the reviewer reads it)
When done, write `{{SUMMARY_FILE}}` with these sections:
```markdown
# Round {{ROUND}} summary
## Mainline objective and outcome
## What was done (files changed in cand/, configs, commands run)
## Evidence
(table: candidate id | change | suite | SCORE | min per-op | correctness | spills | verdict; quote ggemm-eval output lines; benchmark.csv ids)
## Profile evidence
(profile run dirs, REPORT.md priorities used, before/after counters for the same case/op)
## Decisions
(per direction: keep / revise / reject, with the numbers; candidates.jsonl ids)
## Skills used
(wiki page ids read and how they changed the plan; xpu-profile runs; or "none" with the reason)
## Goal tracker updates
(what you changed in the mutable section; or an "Optional Goal Tracker Update Request" if you could not)
## BitLesson Delta
- Action: none|add|update
- Lesson ID(s): <BL-YYYYMMDD-short-name ids that exist in .humanize/bitlesson.md, or NONE>
- Notes: <what changed and why>
## Next step suggestion
```
Rules for the BitLesson Delta: `Action: none` must use `Lesson ID(s): NONE`; `add`/`update` must reference ids that you added/changed in `.humanize/bitlesson.md` using the strict template. Leave `cand/` equal to the best measured candidate when you finish. Do not run git commit/push.
