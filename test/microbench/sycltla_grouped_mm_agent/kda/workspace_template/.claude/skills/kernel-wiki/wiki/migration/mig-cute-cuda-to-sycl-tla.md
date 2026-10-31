---
id: mig-cute-cuda-to-sycl-tla
title: Migrating CUTLASS/CuTe (CUDA) kernel ideas to SYCL-TLA on PVC
type: migration
tags: [migration, cutlass, cute, cuda, sycl-tla, tma, wgmma, persistent]
architectures: [pvc, a100, h100]
confidence: inferred
sources: [notes_ggemm.txt#14, notes_ggemm.txt#18, sycl-tla:include/cute, sycl-tla:examples/12_xe20_moe_gemm_cute_interface]
related: [hw-pvc-vs-a100-model, lang-sycl-tla-cute-xe-api, tech-persistent-scheduling, tech-split-k-stream-k, ker-ggemm-measured-history]
---
# CUDA/CuTe concept -> SYCL-TLA on PVC

| CUDA / CuTe idea | on PVC with SYCL-TLA | caveat |
|---|---|---|
| CTA / thread block | work-group (one per Xe core ideally) | 512 work-items = 32 sub-groups |
| warp (32) | sub-group (16 lanes here) | DPAS and 2D blocks are per sub-group |
| `mma.sync` / wgmma / tcgen05 | `XE_DPAS_TT` through `TiledMMAHelper` | no async warpgroup MMA, no TMEM: accumulators live in the GRF |
| shared-memory staging via `cp.async` / TMA | **no staging**: 2D block load straight to registers + `prefetch` into cache | there is no SMEM pipeline to deepen; the "stages" are `kPrefetchDist` and register double buffering |
| TMA descriptor | block-2d payload built per message (`.with(tensor)`) | descriptor rebuild cost is paid in registers/instructions |
| mbarrier / pipeline stages | `barrier_arrive/wait` split barrier, scope = work-group | coarse; no producer/consumer phases |
| warp specialization (producer/consumer) | not used: every sub-group loads and computes | possible but untested; loads are cheap issue-wise, prefetch plays the producer role |
| cluster / multicast | none | operand sharing happens through L1/L3 only |
| persistent tile scheduler (CUTLASS `PersistentTileScheduler`, group scheduler) | `GroupCursor` + `Scheduler::locate` | scheduler walk cost ~0.37 us per group (notes 14.7) |
| CUTLASS grouped GEMM with pointer arrays + prepare kernel | single kernel reading `offs` on device | prepare kernel + scratch is out of scope (ABI has no workspace) |
| epilogue visitor tree (alpha/beta, bias) | plain `reorder` (fp32 -> bf16) + `copy` D | the CUTLASS-style LinearCombination epilogue measured 2x more instructions for dW (notes 14.6) |
| stream-K / split-K with workspace | needs scratch: unavailable. Intra-work-group K slicing is the SLM-based substitute | see [[tech-split-k-stream-k]] |
| occupancy via registers/SMEM | two discrete GRF modes (128/256) | see [[hw-pvc-grf-threads]] |

What carried over unchanged from CUDA wisdom: tile quantization hurts small problems; grouped scheduling should be device-side to avoid host syncs and CUDA-graph-hostile copies; fixed per-call cost matters because H100-class kernels are 2.4x shorter.
What does not: bandwidth-oriented fixes (TMA multicast, 2-CTA MMA) do not apply because the kernels are not bandwidth-bound; swizzled SMEM layouts are irrelevant without SMEM.
