You are running the equivalent of Humanize `/humanize:gen-plan` for a KDA run. Input: `docs/draft.md` (plus `CLAUDE.md`, `docs/` and any `profile/*/REPORT.md`). Output: `docs/plan.md`, a structured, executable implementation plan. Do not run GPU commands; you may read the wiki with `kw get`.

Analyse the draft for clarity, consistency, completeness and feasibility under the constraints in CLAUDE.md (no scratch memory, no host-row-count-dependent choices, only `cand/` is editable). Resolve ambiguities yourself by choosing the safest reading and recording it under "Assumptions". Then write `docs/plan.md` with exactly these sections:

# <title>
## Goal Description
(one paragraph: the user-facing objective, the target speedup {{TARGET_SPEEDUP}}x SCORE on the `full` suite, no op regressing > 3%.)
## Acceptance Criteria
(AC-1 ... AC-n; each with "Positive Tests" (expected to pass) and "Negative Tests" (expected to fail/be rejected). They must be measurable with ggemm-eval, e.g. correctness PASS on every kept candidate; every explored direction has before/after benchmark rows + profile evidence and a keep/revise/reject decision recorded in candidates.jsonl; a promoted candidate meets the promotion criteria or evidence explains why none can; benchmark.csv has a row for every perf-related change; candidates.jsonl is a consistent DAG; ranked directions in the draft were explored in order with at most 5 iterations each.)
## Path Boundaries
(Upper bound: the most ambitious acceptable result. Lower bound: the minimum acceptable result. Allowed choices: can use / cannot use.)
## Feasibility Hints and Suggestions
(conceptual approach and pointers: wiki ids, file names, config keys; marked as hints, not requirements.)
## Dependencies and Sequence
(milestones and ordering.)
## Task Breakdown
(table: Task ID | Description | Target AC | Tag (`coding` = worker executes; `analyze` = worker only profiles/analyses) | Depends On.)
## Assumptions and Pending Decisions

Be singular and directive. Do not invent measurements. Write the file, then reply with one line.
