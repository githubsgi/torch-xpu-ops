# Agent loops for the SYCL-TLA grouped GEMM

Agentic kernel optimization of the `torch._grouped_mm` XPU kernels in
`src/ATen/native/xpu/grouped_mm/sycltla/` (persistent SYCL-TLA grouped GEMM,
Intel PVC), without PyTorch. Three harnesses, ported from published GPU-kernel
agents, share one evaluator:

| harness | branch | idea | ported from |
|---|---|---|---|
| `xe_forge/` | `ggemm-agent/xe-forge` | analysis -> plan -> dependency-ordered stages, each a CoVeR (write, compile-and-verify, refine) loop with a keep gate | Intel Labs Xe-Forge |
| `kda/` | `ggemm-agent/kda` | Claude Code rounds (draft -> plan -> execute -> review), KernelWiki of PVC rules, `xpu-profile` skill (unitrace in place of ncu) | NVlabs KDA / MIT HAN Lab |
| `kernel_forge/` | `ggemm-agent/kernel-forge` | MCTS over kernel versions: progressive widening, LCB selection on latency, LLM edits plus tile-knob mutations | U. Michigan Kernel Forge |

`common/` (branch `ggemm-agent/common`, the base of the other three) is the
evaluator and the LLM wrapper.

## The evaluator (`common/`)

A **candidate** is a directory with an editable copy of
`grouped_mm_kernel_impl.h` and a `configs.json` with one tile config per kernel
kind (`m_rowB` = forward, `m_colB` = grad_input, `k_grouped` = grad_weight):

```bash
export PATH=$PWD/common/bin:$PATH          # ggemm-eval
ggemm-eval init /path/cand                 # copy of production
ggemm-eval run /path/cand --suite quick    # build, verify, A/B benchmark
ggemm-eval profile /path/cand --case tp4ep4_uniform --op w13_dW   # unitrace digest
ggemm-eval diff /path/cand
ggemm-eval slots                           # who uses which node / tile
```

What `run` does:

1. Snapshots the candidate into `$GGEMM_AGENT_RUNS/jobs/<id>/`.
2. Builds one `.so` per distinct config with icpx (AOT for PVC, the
   torch-xpu-ops-sycltla flags), on any compute node holding a build slot.
   Builds are cached by content hash and shared by all harnesses. A config
   struct named `cand_<hash>` keeps the SYCL kernel names unique, so the
   baseline and the candidate load into one process.
3. Runs `eval/agent_eval` on a free PVC tile (`ZE_AFFINITY_MASK`):
   * **verify**: 22 hidden cases with a new seed on every run. They cover
     ragged, empty and all-empty groups, single rows, sizes around the tile
     edges, and non-DSv3 shapes. The full output is compared with a naive
     fp32 device reference, inside NaN sentinels: an out-of-bounds or missing
     write fails the case, and so does a result left over from an earlier call.
   * **bench**: the six DSv3 16B MoE GEMMs over the suite's routed cases, timed
     against the production kernel at its production config in the same
     process. Batches alternate A/B, B/A for 11 rounds of at least 10 ms each;
     the speedup is the median of the paired ratios. Every timed candidate
     output is checked in full against the baseline output.
4. Prints a summary in the shape of Xe-Forge's external-benchmark contract
   (`STATUS`, `CORRECTNESS`, `SCORE`, per case and op, `VERDICT`, `DONE`), and
   keeps `last_eval.txt` / `last_eval.json` / `evals/` in the candidate.

**SCORE** is the geomean, over the suite's cases, of production time divided by
candidate time for the case's six GEMMs (above 1 means faster). Production
against itself gives 0.98-1.03 per TP4/EP4 case and +-0.2% per EP8 case, so
treat less than 2% as noise.

| suite | cases |
|---|---|
| `quick` | `tp4ep4_uniform`, `ep8_uniform` |
| `small` | TP4/EP4 (16 experts, ~190 rows each): uniform, balanced, hot, sparse, rows64, rows8 |
| `large` | EP8 (8 experts, ~6k rows each): uniform, balanced, hot, rows1024 |
| `full` | small + large |

Contract of a candidate header (see `common/eval/cand_cfg.cpp`):
`run_m_grouped_cfg<Cfg, DType, LayoutB>(queue, params, wgs)`,
`run_k_grouped_cfg<Cfg, DType>(queue, params, wgs)` and `TiledMmaFor<T, Cfg>`
(or `Cfg::kThreads`) in `sycltla::grouped_gemm`; `Params` and
`grouped_mm_kernels.h` are fixed (they are the ABI torch uses). Tile choice
must not depend on routed row counts (all EP ranks must run the same kernel;
notes 19.2).

Settings (env): `GGEMM_AGENT_RUNS` (shared work dir, default
`~/aimod/ggemm_agent_runs`), `GGEMM_NODES` / `GGEMM_NODEFILE` (default
`~/PBS_NODEFILE`), `GGEMM_TILES_PER_NODE` (12), `GGEMM_SYCL_TLA_DIR`,
`GGEMM_LLM_MODEL` (`opus`). See `common/ggemm_agent/config.py`.

Run the harnesses on the login node: it has the internet access `claude -p`
needs, and it reaches the compute nodes over ssh. Compiles and GPU runs happen
on the nodes.

## Moving a winner into torch

The candidate header is a drop-in for
`src/ATen/native/xpu/grouped_mm/sycltla/grouped_mm_kernel_impl.h`; its configs
become `TileCfg*` structs dispatched from `run_m_grouped` / `run_k_grouped`
(see `test/microbench/sycltla_grouped_mm_tune/README.md`, "Move a winner into
torch"). Then run the torch tests and `grouped_mm.dsv3_16b.py` (notes,
section 0).
