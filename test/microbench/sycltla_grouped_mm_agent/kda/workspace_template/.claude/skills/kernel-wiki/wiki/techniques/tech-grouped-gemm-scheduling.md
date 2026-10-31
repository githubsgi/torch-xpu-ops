---
id: tech-grouped-gemm-scheduling
title: Grouped GEMM scheduling from device-side offsets (M-grouped, K-grouped)
type: technique
tags: [grouped-gemm, scheduler, offs, k-grouped, m-grouped, moe]
architectures: [pvc, h100, b200]
confidence: source-reported
sources: [notes_ggemm.txt#14.7, notes_ggemm.txt#14.8, kernel:grouped_mm_kernel_impl.h:Scheduler, sycl-tla:examples/12_xe20_moe_gemm_cute_interface/moe_tile_scheduler.hpp]
related: [tech-persistent-scheduling, ker-ggemm-production, ker-ggemm-constraints, pat-tail-wave-quantization]
---
# How this kernel schedules groups

- `offs` (int32 cumulative, device) -> `Scheduler::describe(cursor)` computes (row_start, rows, tiles_m, tiles) for one group; `locate(linear)` advances the cursor group by group until `linear` falls inside it, then `m_idx = local % tiles_m`, `n_idx = local / tiles_m`. A work-group's linear index only increases, so the cursor never rewinds; each work-group walks all groups once over its lifetime.
- Cost: the walk is a chain of dependent global loads of `offs[g]`: ~0.37 us per group for the old kernel (notes 14.7), CUTLASS's group scheduler ~0.7 us/group. At 16 groups this is ~6 us per launch of a 100-500 us kernel (1-5%); for 8-row calls it dominates. Possible fix: load all `num_groups` offsets once into registers/SLM at kernel start (they are only 16-64 ints) and binary-search / ballot within the sub-group.
- K-grouped (grad_weight): `m = fixed_m`, groups split K; each group has tiles_m x tiles_n output tiles, each tile reduces over the group's rows (~190). Empty groups still own a full set of tiles whose output must be ZEROED (K==0 -> clear accumulators, store zeros; the host path zero-fills only when the whole contraction is empty).
- M-grouped: groups with 0 rows contribute 0 tiles.

Variants on other stacks: CUTLASS pointer-array grouped GEMM needs a prepare kernel + scratch (out of scope); vLLM XPU and sycl-tla example 12 use the same on-device offs walk as this kernel.

Safe design rules: never read `offs` out of range (`g >= num_groups` terminates), treat negative start as 0 (`start < 0 ? 0 : start`), 64-bit tile indices.
