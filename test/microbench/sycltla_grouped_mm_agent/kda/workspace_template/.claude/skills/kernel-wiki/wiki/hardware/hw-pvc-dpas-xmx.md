---
id: hw-pvc-dpas-xmx
title: DPAS / XMX shapes for bf16 and how the TiledMMA tiles them
type: hardware
tags: [pvc, dpas, xmx, mma-atom, bf16]
architectures: [pvc]
confidence: verified
sources: [sycl-tla:include/cute/arch/mma_xe.hpp, sycl-tla:include/cute/atom/mma_traits_xe.hpp, notes_ggemm.txt#18]
related: [hw-pvc-grf-threads, lang-sycl-tla-mma-tiling, hw-pvc-2d-block-io, pat-xmx-underfed]
---
# DPAS (systolic dot-product-accumulate) on XMX

- Atom used by this kernel: `MMA_Atom<XE_DPAS_TT<8, float, T, T>>` with T = `bfloat16_t` / `half_t`. Template args: `XE_DPAS_TT<M, TypeD, TypeA, TypeB, TypeC>`; `M` is the repeat count (rows, up to 8).
- Atom shape `Shape_MNK = <M, 16, K>` with **K = 256 / max(bits(A), bits(B)) = 16 for 16-bit types** (from `XE_DPAS_TT_Base::K`). So one bf16 DPAS is 8x16x16 per sub-group (SIMD16: N is the 16 lanes), 4096 FLOP.
- Operand registers: A = `vector_t<TA, (M*K+15)/16>` (M=8: 8 elements per lane), B = `vector_t<TB, K>` (16 per lane, VNNI-packed pairs), C/D = `vector_t<TC, M>` (8 floats per lane).
- Throughput: 4096 bf16 FLOP/clock per Xe core = 512 FLOP/clock per XMX, so one 8x16x16 DPAS keeps an XMX busy ~8 cycles (inferred). Back-to-back dependent DPAS on the same accumulator are chained in the systolic array; independent accumulators let the XVE interleave.
- A sub-group tile of 32x64 = 4 (M) x 4 (N) DPAS output blocks per K=16 slice => 16 independent accumulators and 32 DPAS per K=32 step, which is what hides DPAS latency in the mainloop.
- Inline asm in `mma_xe.hpp` emits `dpas.<TB>.<TA>.8.<M>`; the accumulate form reuses `d` as `c`.

Per-sub-group tile constraint enforced by the evaluator (`candidate.normalize`): `(tile_m/sg_m) % 8 == 0` and `(tile_n/sg_n) % 16 == 0`.

fp32 accumulation is required by the task contract: do not switch the atom to a bf16-accumulate variant (`XE_DPAS_TT<M, bf, bf, bf, bf>` exists in `mma_xe.hpp`; it would fail verification tolerances).
