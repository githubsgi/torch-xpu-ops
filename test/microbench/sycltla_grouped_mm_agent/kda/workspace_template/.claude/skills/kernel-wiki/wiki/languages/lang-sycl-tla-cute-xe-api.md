---
id: lang-sycl-tla-cute-xe-api
title: SYCL-TLA v0.9 CuTe-on-Xe API cheat sheet (names as used by the production kernel)
type: language
tags: [sycl-tla, cute, xe, api, tiled-mma, block-2d, reorder, barrier]
architectures: [pvc, bmg]
confidence: verified
sources: [sycl-tla:include/cute/atom/mma_atom.hpp, sycl-tla:include/cute/atom/copy_traits_xe_2d.hpp, sycl-tla:include/cute/util/xe_split_barrier.hpp, sycl-tla:examples/12_xe20_moe_gemm_cute_interface, kernel:grouped_mm_kernel_impl.h]
related: [lang-sycl-tla-mma-tiling, lang-sycl-tla-block-2d-copy, lang-sycl-tla-split-barrier, mig-cute-cuda-to-sycl-tla, ker-ggemm-production]
---
# The calls `gemm_tile` makes, in order (header: `grouped_mm_kernel_impl.h`)

```cpp
// MMA: atom x sub-group layout over the work-group tile
using TiledMma = typename TiledMMAHelper<MMA_Atom<XE_DPAS_TT<8, float, T, T>>,
                                         Layout<WGTile /*Shape<M,N,K>*/>, SGLayout>::TiledMMA;
auto thr_mma = mma.get_slice(local_id);            // local_id = this work-item
auto wg_tile = mma.tile_mnk();                     // WG tile (M,N,K)

// global tiles from identity tensors (shape only, no data)
Tensor cA = make_identity_tensor(A.shape());       // (M,K)
Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(m_idx, _));   // (BLK_M,BLK_K,k)

// copy atoms chosen from the tensor STRIDES (plain / VNNI / transpose)
auto copy_a = get_block_2d_copy_A<void>(mma, A);   // also _B, and get_block_2d_copy_D for the store
auto thr_copy_a = copy_a.get_slice(local_id);

// register fragments: MMA-layout (tCr*) and copy-layout (tAr*)
auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));
auto tCrD = thr_mma.partition_sg_fragment_C(gD);
auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
auto tCrD_final = thr_copy_d.partition_sg_fragment_S(gD);   // bf16 store fragment
Tensor tAgA = thr_copy_a.partition_S(gA);          // source view per K tile: tAgA(_,_,_,k)

// prefetch: derived from the load copy (its own tiling, 64 B rows)
auto prefetch_a = make_block_2d_prefetch(copy_a);
auto pAgA = prefetch_a.get_slice(local_id).partition_S(gA);
prefetch(prefetch_a, pAgA(_,_,_,k));

// mainloop body
barrier_arrive(2 /*ScopeWorkgroup*/);
copy(copy_a, tAgA(_,_,_,k), tArA);                 // 2D block load into registers
reorder(tArA, tCrA);                               // copy layout -> DPAS layout (register shuffle)
cute::gemm(mma, tCrA, tCrB, tCrD);                 // DPAS
barrier_wait(2);

reorder(tCrD, tCrD_final);                         // fp32 -> bf16 convert
copy(copy_d, tCrD_final, tCgD);                    // 2D block store (height <= 8)
```

Launch: `queue.parallel_for<KernelT>(nd_range<3>, properties{sub_group_size<16>, intelex::grf_size<256>}, kernel)`; the work-group size must equal `size(TiledMma)` (sg count x 16).

Other names worth knowing: `XE_LOAD_2D`, `XE_LOAD_2D_VNNI`, `XE_LOAD_2D_TRANSPOSE`, `XE_PREFETCH_2D`, `XE_STORE_2D` (explicit atoms; pass them to `get_block_2d_copy_A<Op>` instead of `void`), `make_block_2d_copy_A/B/C/D`, `make_block_2d_prefetch<ValType, SGCount>(shape, strides)`, `Copy_Atom`, `TiledCopy`, `cute::gemm`, `clear`, `ceil_div`, `CUTE_UNROLL`, `compat::` utilities in `cute/util/compat.hpp`. Epilogue/collective-level classes live under `include/cutlass/gemm/collective/xe_mma.hpp`, `.../epilogue/` and the persistent group scheduler `xe_tile_scheduler_group.hpp`.

Reference implementation to read before inventing: `examples/12_xe20_moe_gemm_cute_interface` (`moe_grouped_gemm.hpp`, `moe_tile_scheduler.hpp`), the origin of this kernel's scheduler.
