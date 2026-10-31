# kernel_forge: MCTS over kernel versions (port of Kernel Forge)

Port of Kernel Forge (U. Michigan, arXiv 2607.24762) to the SYCL-TLA grouped GEMM on Intel PVC.
One persistent revision tree for the operator (the three kernel kinds are tuned together, one
candidate = `grouped_mm_kernel_impl.h` + `configs.json`). Stdlib Python only; the evaluator, build
cache, slots and LLM wrapper come from `../common`.

## Mapping to the paper

| Kernel Forge | here |
|---|---|
| revision tree, node = kernel + parent / children / visits / notes / latency | `tree_store.py`: `Node`, `Tree`; `tree.json` (resumable) + `nodes/<id>/` (header, configs, last eval) |
| latency L of a node | `L = 1 / score`, score = geomean speedup vs production on the search suite (production has L = 1) |
| expansion by progressive widening: a node with N visits takes a new child while `|children| <= floor(N^alpha)` | `Tree.select`; in-flight children count toward `|children|` |
| alpha annealed 0.5 -> 0.3 over the first 1000 root visits | `Tree.alpha` (`--alpha0/--alpha1/--anneal-steps`). With a 50-iteration budget alpha stays ~0.5 |
| selection: child minimizing `L_c - C sqrt(ln N_p / N_c)`, L_c = best-subtree latency | `Tree.lcb` |
| no rollout; the child is evaluated directly | `Forge.expand` -> `evaluate()` |
| backprop pushes best-subtree latency up | `Tree.backprop` adds the visits; the best-subtree latency is derived from the nodes on demand (`best_sub_L`) so it can never go stale |
| valid-but-slower candidates stay in the tree | yes (they are the siblings the LLM is told about) |
| incorrect candidates never admitted; compile / runtime failures go through a bounded repair loop | a node is admitted only with status `ok` (22 hidden verify cases + every timed output checked). LLM revisions get `--max-repairs` (2) repair sessions that receive the build / verify error; knob mutations are not repaired (nothing to repair) |
| parallel workers | `--workers W` threads, virtual loss on in-flight paths (`--vloss`, plus virtual visits and pending children), at most 16 |
| LLM gets operator card, hardware config, parent source and lineage | `prompts.py`: operator card (three kinds, six GEMMs, DSv3 shapes, ABI rules), PVC hardware card, parent lineage with per-step notes and scores, parent's last evaluation, a `unitrace` profile digest of the parent's weakest op (lowest absolute TFLOPS, cached per node), summaries of the siblings (slower and failed ones) and of recent failures elsewhere. The parent source is the file the session edits (a fresh child copy) |
| 50 iterations per kernel | `--iterations` (total target; resuming continues up to it) |

### Our extension: two action types

Each expansion is, with probability `--knob-ratio`, a **knob mutation** (no LLM, `knobs.py`), else an
**LLM revision** (`claude -p`, the paper's action). Once the LLM budget (`--max-llm-usd`) is spent, only
knob mutations are drawn.

* **knob**: mutates one aspect of one kind (grad_weight weighted 0.4): tile M x N from a valid set with
  a re-derived sub-group layout (sub-group tile in {16,32,64} x {32,64}, 4-32 sub-groups, preferring
  the production 32x64), sub-group layout, prefetch 1-4, tile_k 32 <-> 64 (64 caps the prefetch at 2),
  `wgs` in {0, 56, 112, 168, 224} (multiples of the 56 Xe cores; 0 = production rule). Duplicates
  (by content hash, including in-flight and failed ones) are re-sampled. Notes = the mutation text.
* **LLM revision**: the session runs in `attempts/<id>/` (copy of the parent) with Read / Edit / Write /
  Grep and read-only shell, plus the sycl-tla sources via `--add-dir`. It cannot run the evaluator;
  the harness evaluates after it finishes (so the evaluator slots are shared fairly). Because the knob
  operator already covers config changes, the prompt asks for a code change in the header and gives each
  parallel session a different suggested focus. A no-op edit or a duplicate of an existing candidate
  counts as a failed round and goes through the repair loop too. Notes = the session's `CHANGE:` text;
  each node also stores an automatic diff (configs changed, header +/- lines).

### Choice of C

The paper uses C = 1.0 on latencies in ms. Here L is normalized to production, so the useful candidates
sit at L = 0.7-1.1 and a real improvement is 0.02-0.15 in L. With C = 1.0 the exploration term
(`sqrt(ln N_p / N_c)` is 1.0-1.5 for a child with one visit) is 10x the whole signal and selection
degenerates to visit balancing. The default **C = 0.1** makes the bonus of a once-visited child under a
parent with 10 visits (0.15) comparable to a 15% latency gap, and a ten-times-visited child's bonus
(0.05) comparable to the 2% measurement noise. Raise it for long runs (hundreds of iterations) or noisy
suites. Virtual loss adds `--vloss` (0.05) to a child's L for every in-flight pass through it.

## Run

```bash
export PATH=$PWD/../common/bin:$PATH                      # ggemm-eval (the LLM sessions do not need it)
cd test/microbench/sycltla_grouped_mm_agent               # parent of this package
GGEMM_LLM_MODEL=opus ~/miniforge3/bin/python3 -u -m kernel_forge.mcts \
    --workdir $GGEMM_AGENT_RUNS/kernel_forge/run1 --iterations 50 --workers 8 \
    --knob-ratio 0.3 --model opus --max-llm-usd 50
```

Run on the login node (internet for `claude`, ssh to the compute nodes for builds and GPU runs).
Re-running with the same `--workdir` resumes from `tree.json` (`--iterations` is the total).
`--report-only` rebuilds `report.md`; `--skip-final` stops after the search.
Other options: `--suite quick|small` (search suite), `--final-suite full`, `--top-k 3`, `--c`, `--vloss`,
`--alpha0/--alpha1/--anneal-steps`, `--max-repairs`, `--llm-call-usd` (cap per session), `--llm-timeout-min`,
`--max-wall-min`, `--no-profile`, `--seed`, `--llm-allow-config-only`.

`python -m kernel_forge.test_tree` is an offline check of widening / selection / virtual loss.

## Outputs (in `--workdir`)

| file | content |
|---|---|
| `tree.json` | the tree: nodes (visits, score, per-op, notes, auto diff, cached profile, final full eval), failed attempts, LLM spend, run arguments |
| `iterations.jsonl` | one line per iteration: selection trace (widening / LCB decisions), action, outcome, score, vs parent, cost, repairs; plus `repair` events |
| `nodes/<id>/` | admitted candidates (`grouped_mm_kernel_impl.h`, `configs.json`, `last_eval.txt`) |
| `attempts/<id>/` | every attempt's working dir (failed ones too) |
| `logs/claude.jsonl` | every `claude -p` prompt and answer with cost |
| `report.md` | run summary, result and verdict, lineage of the best node (change + score per step), top candidates (search vs full score), per-op table, config and header diff vs production, failed attempts, tree summary |
| `best/` | the best candidate (drop-in for the production header; see `../README.md`, "Moving a winner into torch") |

At the end the production root and the top-k nodes are re-evaluated on the `full` suite; the best node
is the one with the highest full score. Search-suite scores (2 cases) have a noise of a few percent, so
treat a node as an improvement only if the full-suite score beats the re-run production by more than 2%.
