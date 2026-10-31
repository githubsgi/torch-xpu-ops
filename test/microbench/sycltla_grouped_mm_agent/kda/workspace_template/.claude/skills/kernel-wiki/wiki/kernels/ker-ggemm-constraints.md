---
id: ker-ggemm-constraints
title: Hard constraints on any candidate (ABI, rank independence, numerics)
type: kernel
tags: [constraints, abi, rank-independence, workspace, numerics]
architectures: [pvc]
confidence: verified
sources: [notes_ggemm.txt#19.2, notes_ggemm.txt#0, common/eval/cand_cfg.cpp, grouped_mm_kernels.h]
related: [ker-ggemm-production, tech-device-side-dispatch, tech-split-k-stream-k, tech-grouped-gemm-scheduling]
---
# Constraints

1. `Params` and `grouped_mm_kernels.h` are the ABI torch uses: no workspace pointer, no new fields. Designs needing scratch memory (classic split-K, stream-K reductions, global atomic work queues, prepare kernels) are out of scope. No allocations in the launch path.
2. Tile/config choice must not depend on host-known routed row counts: every EP rank must run the same kernel (notes 19.2: mixed kernels across ranks crashed 3 of 4 EP8 runs; root cause unknown). Per-kind static configs are fine; device-side branching on `offs` is fine.
3. fp32 accumulation, bf16 output; verified against an fp32 reference on 22 hidden cases (ragged, empty groups, all-empty, single rows, tile-edge sizes, non-DSv3 shapes) with NaN sentinels around the output: OOB writes and stale outputs fail.
4. The header must keep the contract in `common/eval/cand_cfg.cpp`: `run_m_grouped_cfg<Cfg,DType,LayoutB>(q,p,wgs)`, `run_k_grouped_cfg<Cfg,DType>(q,p,wgs)`, `TiledMmaFor<T,Cfg>` (or `Cfg::kThreads`).
5. Launch: sub_group_size 16 and the persistent grid; kernels must handle `num_groups` up to 64, empty groups, `offs` entries equal (zero rows), K == 0 in K-grouped.
6. Leading dims are multiples of 32 elements and base pointers 64-byte aligned (checked on the host before the kernel is used).
