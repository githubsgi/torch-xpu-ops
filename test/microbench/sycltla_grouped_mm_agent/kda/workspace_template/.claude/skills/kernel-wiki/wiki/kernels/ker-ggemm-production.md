---
id: ker-ggemm-production
title: Production persistent SYCL-TLA grouped GEMM (structure, configs, facts)
type: kernel
tags: [grouped-gemm, production, sycl-tla, persistent, m-grouped, k-grouped]
architectures: [pvc]
confidence: verified
sources: [kernel:grouped_mm_kernel_impl.h, sycltla_ggemm_kernel_summary.md, notes_ggemm.txt#0, notes_ggemm.txt#16.3]
related: [ker-ggemm-constraints, ker-ggemm-grad-weight, ker-ggemm-forward-dx, ker-ggemm-regime-map, tech-grouped-gemm-scheduling, lang-sycl-tla-cute-xe-api]
---
# What the kernel is

`torch._grouped_mm` XPU path for MoE expert layers: one persistent launch per call, `offs` (cumulative int32) read on device, bf16/fp16, fp32 accumulation, bf16 output.
- **M-grouped** (`MGroupedKernel<T, LayoutB, Cfg>`): forward (B is [E,N,K] -> K-contiguous, 'R') and grad_input (B is [E,K,N] -> N-contiguous, 'C'). `offs` partitions rows of A and D.
- **K-grouped** (`KGroupedKernel<T, Cfg>`): grad_weight, A = activation^T viewed column-major (M-contiguous), B row-major [rows,N] viewed column-major, D = [E,M,N]; `offs` partitions the contraction.
- Per tile: `gemm_tile(A,B,D,m_idx,n_idx,mma)`: prefetch prologue (dist), loop {barrier_arrive; load A,B; prefetch k+dist; reorder; DPAS; barrier_wait}, then reorder+store D.
- Production config on PVC for ALL kinds: tile 256x256x32, sg 8x4 (32 sub-groups = 512 work-items), 32x64 per sub-group, prefetch 3, 56 work-groups, sub_group_size 16, grf_size 256.
- Other tiles built in: 256x128 (sg 8x2), 128x256 (sg 4x4), 128x128 (sg 4x2). Auto rule picks only the device default (256x256 on PVC); `TORCH_XPU_GROUPED_MM_TILE` forces another.

Workload (agent_eval): DeepSeek-V3 16B MoE layer, hidden 2048, expert FFN 1408; six GEMMs: w13_fwd (N=2816,K=2048), w2_fwd (N=2048,K=1408), w13_dX (N=2048,K=2816), w2_dX (N=1408,K=2048), w13_dW (M=2816,N=2048), w2_dW (M=2048,N=1408). 64 experts, top-6; TP4/EP4 = 16 local experts x ~190 rows; EP8 = 8 local experts x ~6k rows.

Measured throughput, uniform routing, static split, PVC tile (notes 22.2):
- [claim] gpu=PVC-tile dtype=bf16 shape=TP4EP4 ~190 rows/expert metric=TFLOPS value=w13_fwd 137, w2_fwd 126, w13_dX 127, w2_dX 131, w13_dW 84, w2_dW 88 source=notes_ggemm.txt#22.2
- [claim] gpu=PVC-tile dtype=bf16 shape=EP8 ~6k rows/expert metric=TFLOPS value=w13_fwd 142, w2_fwd 137, w13_dX 155, w2_dX 143, w13_dW 138, w2_dW 129 source=notes_ggemm.txt#22.2
- [claim] metric=layer time TP4EP4 uniform value=1.43 ms (fallback 2.54, oneDNN 2.55) source=notes_ggemm.txt#22.2
- unitrace/VTune per kernel (TP4/EP4, notes 11.2): dW 300 us XVE 53% active/30% stalled/17% idle, occupancy 41%; dX 203 us 66%/6%/28%/36%; fwd 235 us 56%/12%/33%/34%.
