# kda: Kernel Design Agent port for the SYCL-TLA grouped GEMM

A port of NVlabs **KDA** and the MIT HAN Lab **MLSys-2026 FlashInfer contest** flow to the
`torch._grouped_mm` XPU kernel (Intel PVC). It drives headless Claude Code (`claude -p`, through
`common/ggemm_agent/llm.py`) in Humanize-style rounds, with a wiki of PVC knowledge and an `xpu-profile`
skill in place of Nsight Compute. Evaluation, correctness and GPU/build slots come from `../common`
(`ggemm-eval`); nothing here edits `common/` or `src/`.

## Mapping to the sources

| source | what it does | here |
|---|---|---|
| **KDA** `CLAUDE.md`, `docs/agent-flow.md`, `prompts/basic-flow.md` | task contract, draft -> plan -> implement -> validate -> record evidence -> promote | `workspace_template/CLAUDE.md` (objective, I/O, correctness, constraints, commands, promotion criteria), `docs/draft.md` / `docs/plan.md`, `benchmark.csv`, `candidates.jsonl` (DAG), `profile/<run>/` |
| **Humanize** RLCR (`start-rlcr-loop`, stop hook, `gen-plan`) | draft.md -> plan.md with acceptance criteria; `.humanize/rlcr/<ts>/` with `plan.md`, `state.md`, `goal-tracker.md`, `round-N-prompt.md`, `round-N-summary.md` (`## BitLesson Delta`), `round-N-review-result.md`; Codex reviews every summary; COMPLETE -> code-review phase; full alignment audit every 5 rounds; max 42 rounds; `bitlesson.md` | `rlcr.py`: same file layout and state fields. Reviewer = read-only `claude -p` (no Codex); verdict on the last line (`COMPLETE` / `STOP` / `CONTINUE`) plus the mandatory `Mainline Progress Verdict: ADVANCED/STALLED/REGRESSED` line (the driver re-runs a review without it). Round contract + lanes (`[mainline]`/`[blocking]`/`[queued]`), frozen immutable goal-tracker section (the driver restores it if a later round edits it), BitLesson Delta validation (invalid -> normalized to `none` and flagged), full alignment audit at rounds 4, 9, ... (`--full-review-round`), final code-review phase with `[P0]-[P3]` markers and one fix pass, `.humanize/bitlesson.md` seeded with three lessons |
| **KernelWiki** (`SKILL.md`, `wiki/{hardware,techniques,kernels,patterns,languages,migration}`, `query.py`, `get_page.py`, `grep_wiki.py`) | cited, confidence-graded pages, pattern pages with symptoms + candidate techniques | `workspace_template/.claude/skills/kernel-wiki/`: 39 pages seeded from the SYCL-TLA v0.9 headers and `notes_ggemm.txt` (each cites its sections), `kw query/get/grep/validate`, same frontmatter (`id title type tags architectures confidence sources related`, `symptoms`/`candidate_techniques` on patterns), confidence `verified > source-reported > inferred > experimental`, `[claim]` lines with gpu/dtype/shape/metric/value/source |
| **ncu-report-skill** | Profile -> Diagnose -> Plan, never guess; REPORT.md with 3-5 ranked priorities (Evidence, impact, effort, why now); playbook (signals / why / first-line fix / deeper fixes / exceptions) | `.claude/skills/xpu-profile/`: unitrace (`ggemm-eval profile`) instead of ncu; `reference/playbook.md` in unitrace terms with reference values; `scripts/make_report.py` (`xpu-report make/check`) adds tiles/waves/wave-efficiency/row fill/K-tiles-per-tile and ranks rule hits with evidence |
| **Contest prompts** `prompts/moe-fp8/phase{1,2,3}.md` | phase 1 baseline + correctness, phase 2 explore ranked directions (<= 5 iterations each, before/after + profile evidence), phase 3 specialize by regime where measured wins justify it; benchmark.csv per perf change; DAG of candidates; never silently redefine the target | `workspace_template/prompts/phase{1,2,3}.md` + `common.md`, adapted to this kernel (regimes TP4/EP4 vs EP8, three kernel kinds with independent configs, rank-independence constraint) |

Differences from the originals: the reviewer is a Claude session instead of Codex; Humanize's stop hook
is the Python driver (`rlcr.py` decides, between sessions, instead of intercepting the stop); the driver adds
what a hook cannot: an independent `ggemm-eval` run of `cand/` after every round, automatic promotion to
`best/` on the full suite, and hard $ / stall / failure stops.

## Run

```bash
cd test/microbench/sycltla_grouped_mm_agent          # so that `python -m kda.rlcr` finds the package
export PATH=$PWD/common/bin:$PATH                     # ggemm-eval (the driver also passes this on)
~/miniforge3/bin/python3 -m kda.rlcr \
    --workdir $GGEMM_AGENT_RUNS/kda/run1 --phase 2 --max-rounds 8 --model opus --max-llm-usd 60
```

Run it on the login node (internet for `claude`; compiles and GPU runs go to compute nodes over ssh).
`--phase 1|2|3` selects the phase prompt; re-running the same command **resumes** (every artifact is a file;
a loop that stopped on max rounds or the $ cap is extended when you raise `--max-rounds` / `--max-llm-usd`).
Running a later phase on a finished workspace archives the old draft/plan, copies `best/` into `cand/` and
starts a new loop. Useful options:

| option | default | |
|---|---|---|
| `--target-speedup` | 1.05 | stated in the phase prompt / plan (promotion itself needs >= 1.02, no op < 0.97) |
| `--guidance`, `--guidance-file` | | human hints appended to the prompts (hardware features to try, directions to avoid) |
| `--eval-suite` | quick | suite of the driver's per-round independent evaluation (>= 1.01 triggers a full-suite check, promotion at >= 1.02) |
| `--full-review-round` | 5 | alignment audit at rounds 4, 9, 14, ... |
| `--max-stall` | 3 | stop after this many consecutive STALLED/REGRESSED reviews |
| `--worker-budget`, `--reviewer-budget`, `--draft-budget`, `--plan-budget` | 8 / 3 / 5 / 2.5 | per-session `--max-budget-usd`; the total is capped by `--max-llm-usd` (default 60) |
| `--model`, `--reviewer-model`, `--plan-model` | `$GGEMM_LLM_MODEL` or opus | |
| `--final-review complete\|always\|never` | complete | code-review phase after COMPLETE (`always` also after other stops) |
| `GGEMM_KDA_FAKE_LLM=1` | | canned LLM stub: tests the control flow with no cost |

Stops: reviewer `COMPLETE` (accepted only if `benchmark.csv` has rows and the driver's evaluation is `STATUS: ok`),
reviewer `STOP`, `--max-rounds`, `--max-llm-usd`, `--max-stall`, two consecutive failed worker sessions.

## Outputs (everything under `--workdir`, which is the task workspace)

```
CLAUDE.md  prompts/  .claude/skills/{kernel-wiki,xpu-profile}/  bin/{kw,xpu-report,kda-record}  tools/
reference/        production header, ABI header, evaluator contract (read-only)
production/ best/ cand/   untouched copy / best promoted candidate / working candidate (header + configs.json)
docs/draft.md docs/plan.md docs/task_prompt.md docs/wiki-feedback.md
benchmark.csv     one row per perf-related change (worker rows and driver rows)
candidates.jsonl  candidate DAG (parents, status kept/rejected/promoted, reasons); candidates/<id>/ snapshots
profile/<run>/    profile texts + REPORT.md (3-5 ranked priorities)
.humanize/bitlesson.md  .humanize/tool_log.jsonl (hook log: Skill/Bash/Read uses per round)
.humanize/costs.jsonl  .humanize/llm_calls.jsonl (full prompts and replies)
.humanize/rlcr/<ts>/  plan.md state.md goal-tracker.md goal-tracker.immutable.md
                      round-N-{prompt,contract,summary,eval,driver,review-prompt,review-result}.md
                      final-eval.txt code-review-result.md final-report.md
FINAL_REPORT.md   rounds, verdicts, scores, benchmark rows, candidate DAG, spend, skill activity
```
The workspace is a git repository (the driver commits after each round, so `git diff <base_commit> -- cand/`
is the candidate's total change). A promoted winner is `best/` (a drop-in for
`src/ATen/native/xpu/grouped_mm/sycltla/grouped_mm_kernel_impl.h`, see `../README.md`).

## Tools and permissions of the sessions

| session | tools |
|---|---|
| draft / worker | `llm.EDIT_TOOLS` + `Skill`, `kw`, `xpu-report`, `kda-record`, `tee`, `find`, read-only git; read access to the sycl-tla checkout and the notes directory |
| gen-plan | Read/Write/Edit/Glob/Grep, `kw`, `ls`, `cat` (no GPU) |
| reviewer / code review | Read/Glob/Grep, `cat ls diff head tail wc`, read-only git, `kw`, `ggemm-eval diff`, `xpu-report check` (cannot edit, cannot run evals) |

`.claude/settings.json` in the workspace installs a PostToolUse hook (`tools/log_tool.py`) that logs Skill / Bash /
Read / Edit / Write uses to `.humanize/tool_log.jsonl`; the driver summarizes it per round for the reviewer
(proof that the skills and CLIs were used). `BASH_DEFAULT_TIMEOUT_MS` is raised so that `ggemm-eval` runs
(minutes) are not cut by Claude Code's 2-minute Bash default.

## Extending the wiki

Add `wiki/<dir>/<id>.md` (copy a neighbour's frontmatter; ids are unique, file name = id) and run
`bin/kw validate`. The worker records confirmations/refutations in `docs/wiki-feedback.md`; promote them by hand
between runs (the template is the source of truth: `kda/workspace_template/.claude/skills/kernel-wiki/`).

## Smoke run (2026-10-30, `GGEMM_LLM_MODEL=sonnet`, phase 2, 3 rounds)

Run directory `~/aimod/ggemm_agent_runs/kda/smoke-1030-2341` (resumed once: 2 rounds, then `--max-rounds 3`).
Total LLM spend $1.6. The draft session invoked both skills through the `Skill` tool (they load headless from
the workspace `.claude/skills`), ran the baseline and a dW profile, and wrote `xpu-report` output; workers used
`kw`, `kda-record`, `ggemm-eval run/profile`; the reviewer caught a worker that skipped the skills and corrupted
`benchmark.csv`. Result: no promotable improvement. `k_grouped` prefetch 2 gave dW +2-4% per op, but the driver's
independent full-suite evaluation gave SCORE 1.005 (worker claimed 1.013); cross-tile prefetch (a kernel edit)
verified correct but was slower and was rejected with a profile (SBID 34% -> 24%, XMX unchanged at 36%).
The loop stopped on the stall counter (3 consecutive STALLED reviews).
