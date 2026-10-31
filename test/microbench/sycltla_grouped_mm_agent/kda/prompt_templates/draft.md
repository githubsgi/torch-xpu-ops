You are the planning agent of a Kernel Design Agents (KDA) run, working in the task workspace (your current directory). Read `CLAUDE.md` first: it is the task contract.

{{PHASE_PROMPT}}

## What to do now (draft stage; no kernel edits yet unless the phase prompt allows config experiments)
1. Skills: invoke the `kernel-wiki` skill (Skill tool) and use `kw query|get` for the pages relevant to this phase; invoke the `xpu-profile` skill and follow it for any profiling. Cite wiki page ids in the draft.
2. Establish the baseline with the evaluator exactly as the phase prompt says (budget awareness: this whole stage should cost no more than a few minutes of GPU time; prefer the `quick` suite plus ONE profile of the weakest op unless the phase prompt demands more).
3. Write the plan draft to `docs/draft.md`. It MUST contain: (a) baseline numbers and how they are validated; (b) main risks/unknowns; (c) candidate optimization directions ranked by expected benefit and risk, each with wiki ids, profile evidence, a measurable expected effect, subtasks, and the keep/revise/reject evidence needed (at most 5 iterations per direction); (d) first concrete implementation steps; (e) exact validation and evaluation commands; (f) the promotion criteria from CLAUDE.md.
4. Record the baseline: `kda-record bench --id production --direction baseline` after the baseline `ggemm-eval run`, and `kda-record cand --id production --parent production --status kept --direction baseline --note "..."` is NOT needed (production is the DAG root).
5. Do not start implementing directions. Finish with a short message stating the file you wrote.
