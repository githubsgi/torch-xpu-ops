---
name: kernel-wiki
description: Use BEFORE proposing or implementing any optimization of the SYCL-TLA grouped GEMM on Intel PVC (tile shapes, prefetch, scheduling, split-K, GRF/occupancy, 2D block loads, DPAS), when a profile shows a symptom (SBID stalls, idle XVEs, low XMX utilization, spills), when porting a CUDA/CuTe idea to SYCL-TLA, or when you need measured facts about this kernel (what was tried, regimes, constraints). A seeded wiki of PVC hardware, SYCL-TLA API, techniques, symptom patterns and kernel facts with confidence levels and sources. Do not guess hardware numbers - look them up here.
---

# Skill: kernel-wiki (PVC / SYCL-TLA grouped GEMM)

A small, curated knowledge base (the XPU counterpart of KernelWiki). Pages are markdown with YAML frontmatter under `wiki/`:

| dir | type | content |
|---|---|---|
| `wiki/hardware/` | hardware | Xe core, XVE/XMX, GRF modes, DPAS shapes, 2D block I/O limits, memory hierarchy, PVC-vs-NVIDIA map, unitrace metrics |
| `wiki/languages/` | language | SYCL-TLA v0.9 CuTe-on-Xe API (`XE_DPAS_TT`, `TiledMMAHelper`, `get_block_2d_copy_*`, `make_block_2d_prefetch`, `reorder`, `barrier_arrive/wait`) |
| `wiki/migration/` | migration | CUDA CuTe/CUTLASS concept -> SYCL-TLA mapping |
| `wiki/techniques/` | technique | prefetch distance, tile shape, persistent scheduling, grouped scheduling, split-K family (scratch-free variants), double buffering, epilogue/descriptor overhead, GRF/occupancy, device-side dispatch, compiler flags |
| `wiki/patterns/` | pattern | symptom -> candidate techniques (`symptoms:` and `candidate_techniques:` in frontmatter), phrased in unitrace terms |
| `wiki/kernels/` | kernel | this kernel: structure, per-op facts (grad_weight, forward/dX), regimes, constraints, history of tried ideas |

Frontmatter: `id`, `title`, `type`, `tags`, `architectures`, `confidence` (`verified` > `source-reported` > `inferred` > `experimental`), `sources` (notes sections `notes_ggemm.txt#16.2`, sycl-tla files, kernel code), `related`. Performance claims are written as `[claim] gpu=.. dtype=.. shape=.. metric=.. value=.. source=..`.

## Commands (from the workspace root; `bin/` is on PATH)

```bash
kw query --type pattern --q "sbid short k"      # ranked keyword search; also --tag T (repeatable), --arch pvc, --confidence source-reported
kw query --symptom "idle"                        # pattern pages by symptom text
kw query --list-tags
kw get pat-short-k-prologue-bound                # print a page (several ids allowed); --frontmatter-only; --follow-related
kw grep 'prefetch|SBID' --type technique -C 1    # regex over bodies
kw validate                                      # wiki integrity (when you add pages)
```

## How to use it (output contract)

1. Start from the *symptom* (unitrace digest, `last_eval.txt`): `kw query --type pattern --q "<symptom words>"`, read the matching pattern page, then the technique pages in its `candidate_techniques`.
2. Before touching hardware numbers, API names or constraints, `kw get` the hardware/language/kernel page. Never invent register budgets, message limits or measured TFLOPS.
3. **Cite page ids** in `docs/draft.md`, `docs/plan.md`, round summaries and `candidates.jsonl` notes (e.g. "per tech-prefetch-distance (source-reported), try pf 2 on k_grouped").
4. Respect confidence: `experimental`/`inferred` pages are hypotheses - say so in the plan and test them; `source-reported` come from measured notes; `verified` from headers/code.
5. **Write back**: when a measurement confirms or refutes a page, add a short entry to `docs/wiki-feedback.md` (page id, claim, new evidence with shape/metric/value and the benchmark.csv row). A final phase may promote it into the wiki. If you add a page, copy an existing one's frontmatter and run `kw validate`.
6. The wiki is guidance, not evidence for promotion: promotion evidence is `ggemm-eval` results.
