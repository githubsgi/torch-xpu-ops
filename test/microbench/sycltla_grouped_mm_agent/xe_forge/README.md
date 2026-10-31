# Xe-Forge port for the SYCL-TLA grouped GEMM

Port of Intel Labs' [Xe-Forge](https://github.com/IntelLabs/Xe-Forge) (analysis -> planning -> dependency-ordered
optimization stages, each a best-of loop around a CoVeR write / compile-and-verify / refine loop) to the persistent
grouped GEMM in `src/ATen/native/xpu/grouped_mm/sycltla/grouped_mm_kernel_impl.h` (Intel PVC). It uses the shared
evaluator in `../common` as `compile_and_verify` and benchmark, and headless Claude Code (`ggemm_agent.llm.run_claude`)
as the LLM. Python standard library only (no DSPy / litellm / pydantic); the knowledge base is read with PyYAML if it is
installed, else with a small subset parser (`kb.py`).

## Run

From the login node (it needs the internet for `claude -p`; builds and GPU runs go to the compute nodes by ssh):

```bash
cd test/microbench/sycltla_grouped_mm_agent          # the package dir's parent
export GGEMM_LLM_MODEL=opus                          # or sonnet for a cheap smoke test
~/miniforge3/bin/python3 -m xe_forge.run --max-llm-usd 80                  # all stages with detected issues
~/miniforge3/bin/python3 -m xe_forge.run --stages memory_access,autotuning --model sonnet \
    --max-iterations 2 --max-attempts 3 --max-llm-usd 20 --run-id smoke  # a short run
```

Useful flags: `--workdir DIR` (resumes when `DIR/pipeline.json` exists), `--suite quick` (stage gate and
compile_and_verify), `--final-suite full`, `--max-iterations N` (best-of, env `AGENT_MAX_ITERATIONS`, default 3),
`--max-attempts N` (CoVeR attempts per iteration, env `MAX_ATTEMPTS_PER_STAGE`, default 5), `--max-llm-usd`,
`--per-call-usd`, `--effort`, `--stop-on-no-improvement`, `--no-profile`, `--profile-matrix small`,
`--lessons-dir` (shared ledger), `--parallel` (concurrent evaluations of an autotuning sweep). Default run directory:
`$GGEMM_AGENT_RUNS/xe_forge/<run-id>/` (default `~/aimod/ggemm_agent_runs`). GPU/build slots are shared with the other
harnesses through the evaluator's locks.

## Pipeline

1. **Baseline**: production evaluated against itself on the stage suite -> trial `t0` (score ~1.00 = the noise level).
2. **Profiles**: unitrace digests (`ggemm-eval profile`, 6 in parallel) of `w13_fwd`, `w13_dX`, `w13_dW` at
   `tp4ep4_uniform` and `ep8_uniform` for the three kernel kinds in both regimes. Re-collected after every accepted stage.
3. **Analysis** (`analysis.py`, `analysis_signature` equivalent): one read-only Claude session sees the kernel, notes
   excerpts, knowledge base, per-op table and digests and returns JSON (`--json-schema`): a list of `DetectedIssue`
   {issue_type from the enum in `models.py`, severity 1-5, kinds, ops, evidence (measured metrics), suggested_fix,
   estimated_speedup, open_ended_proposal}. Each issue type maps to one stage (`ISSUE_STAGE`).
4. **Planning**: Claude orders the stages that have issues (JSON schema); `HARD_DEPENDENCIES` are then enforced
   regardless of its answer; any failure falls back to `DEFAULT_STAGE_ORDER`.
5. **Stages** (`optimizer.py`, `autotune.py`), in order. Per stage:
   * best-of loop (`--max-iterations`): each iteration starts from a fresh workspace copy of the best candidate so far
     (stage prompt + stage-specific knowledge-base rules + issues + performance context + profile digests + lessons +
     the last three attempt summaries and "Try a DIFFERENT approach to beat the current best");
   * inside it a CoVeR loop (`--max-attempts`): the LLM edits `cand/` in place; the harness runs `compile_and_verify`
     (static checks, build, hidden correctness cases, paired benchmark) and feeds the exact observation back, with the
     diff of what the LLM currently has, until the observation is `Success!` (builds, correct, not below 0.98x of the
     incumbent: `PERFORMANCE REGRESSION` otherwise);
   * keep gate: score >= 1.02 x the score of the candidate the iteration started from (`trials.keepable`); else the
     stage is reverted. The accepted candidate becomes the incumbent and the profiles are refreshed.
   * `autotuning` is different, as in the original (the harness is the autotuner): Claude proposes configs.json
     variants (each changes one kind), they are built and evaluated in parallel threads, the best variant per kind
     (>= 1.02x on that kind's ops) wins, the combination is evaluated and gated like any trial.
6. **Final**: the incumbent is re-evaluated on `--final-suite` (default `full`) and copied to `best/`; `report.md`.

Deviation from the original: when an iteration is neutral or fails, the original ends the stage; here the remaining
best-of iterations are used (`--stop-on-no-improvement` restores the original behavior).

## Mapping of the original stages

| Xe-Forge stage | here | meaning for the grouped GEMM |
|---|---|---|
| analysis | analysis | LLM + profile digests -> issues |
| algorithmic | `algorithmic` | work decomposition / scheduling: tile linearization over groups, cross-tile pipelining, K-grouped restructuring (no scratch memory: the ABI has no workspace) |
| discovery | `discovery` | open_ended issues |
| dtype_fix | dropped | fp32 accumulation + bf16 output is the contract |
| fusion | dropped | no epilogue to fuse; Params / ABI fixed |
| memory_access | `memory_access` | prefetch distance, K tile, 2D block copy atoms, VNNI / transposed loads, reorders |
| tensor_descriptors | merged into `memory_access` | the CuTe 2D block descriptors are already in use |
| persistent_kernel | `persistent_kernel` | grid size (`wgs`), tile order / raster for L3 reuse, last wave |
| device_specific | `device_specific` | sub-group layout, per-SG DPAS tile, GRF 128 vs 256, barriers |
| autotuning | `autotuning` | configs.json sweeps per kind, parallel |

Hard dependencies: algorithmic, discovery -> memory_access; algorithmic, memory_access -> persistent_kernel;
memory_access, persistent_kernel -> device_specific; device_specific -> autotuning (the original's
`memory_access -> device_specific -> autotuning` kept, plus the new ordering among the persistent/algorithmic stages).

## Files

| file | role (original) |
|---|---|
| `run.py` | CLI (`cli.py`) |
| `pipeline.py` | `pipeline.py:optimize`; resumable `pipeline.json` |
| `models.py` | `models.py`: stages, issue enum, `DetectedIssue`, issue -> stage, dependencies |
| `analysis.py` | `analyzer_agent.py` + `planner.py` |
| `optimizer.py` | `optimizer_agent.optimize_stage` (best-of) + `cover.py` (CoVeR), harness-driven |
| `autotune.py` | autotuning stage with parallel sweeps |
| `trials.py` | `core/trial_manager.py`: trial tree `trials/state.json`, `_rank`, `keepable` |
| `lessons.py` | `core/lessons.py`: append-only measured lessons |
| `measure.py` | compile_and_verify (evaluator), performance context, profile collection |
| `workspace.py` | per-iteration workspaces, read-only `refs/`, static checks (INVALID: ... returns) |
| `prompts.py` | analysis / planning / stage / autotune prompts and the SYCL rules (`CLAUDE.md.j2`) |
| `kb.py`, `knowledge_base/pvc_sycltla_grouped_gemm.yaml` | PVC / SYCL-TLA knowledge base for this kernel |
| `report.py` | `report.md` |

## Outputs (in the run directory)

`report.md` (stage-by-stage table, accepted diffs, final per-case scores, LLM cost by purpose, trial tree),
`pipeline.json` (phase and stage records), `trials/state.json` (trial tree: parent/child, stage, strategy, score,
per-op, `measured_against` baseline digest, `source: measured`) with each trial's `grouped_mm_kernel_impl.h`,
`configs.json`, `result.json` under `trials/tN/`, `lessons.md` (append-only, measurements only), `analysis.json`,
`plan.json`, `profiles/<tag>/*.txt`, `stages/<stage>/iter*/ws/` (workspaces), `best/` (the winning candidate, a drop-in
candidate dir), `final_eval.{txt,json}`, `cost.json`, `llm_log.jsonl`, `run.log`.

## Knowledge base

`knowledge_base/pvc_sycltla_grouped_gemm.yaml`: constraints (fixed ABI, no scratch, rank-independent choice, numerics,
candidate contract, DPAS atom, GRF/residency, Xe 2D block limits, barriers), measured facts with citations to the notes
(sections 0, 11.1-11.4, 16.2, 17.2, 22.2), API notes (names verified against `/mnt/aimod/subratag/sycl-tla` v0.9) and
per-stage patterns ported from Xe-Forge's `sycl_tla_patterns.yaml` / `cutlass_sycl_framework.yaml` with PVC numbers.
Entries are priors and measurements of other runs; the evaluator decides.
