---
id: lang-sycl-tla-block-2d-copy
title: Block 2D copy atoms and prefetch in SYCL-TLA
type: language
tags: [sycl-tla, block-2d, copy-atom, prefetch, vnni, transpose, strides]
architectures: [pvc, bmg]
confidence: verified
sources: [sycl-tla:include/cute/atom/copy_traits_xe_2d.hpp, sycl-tla:include/cute/arch/copy_xe_2d.hpp, kernel:grouped_mm_kernel_impl.h:make_2d_tensor]
related: [hw-pvc-2d-block-io, lang-sycl-tla-cute-xe-api, tech-prefetch-distance, ker-ggemm-grad-weight]
---
# Block-2D copies

- The copy atom is chosen at compile time from the global tensor's **strides** (`find_x_mode`, `find_y_mode`: which mode has stride 1). The kernel's `make_2d_tensor<'R'|'C'>` builds tensors with strides `(ld, 1)` or `(1, ld)`; dynamic extents (rows, cols) carry the group's real size so out-of-range lanes are zero-filled/masked by the hardware.
- Call forms: `make_block_2d_copy_A(mma, gmem_tensor)` / `get_block_2d_copy_A<void>(mma, tensor)` (default atom), `get_block_2d_copy_A<XE_LOAD_2D_VNNI<16,32,16>>(...)` (explicit op), `.with(tensor)` rebinds base pointer/extents for a new tile or group without recomputing the tiling (cheap descriptor update).
- Per-tile setup cost: building the three tiled copies + two prefetches per tile (`gemm_tile`) is paid for every tile of the persistent loop. With ~6 K steps per dW tile it is visible (notes 14.6: ~2.3x more code and fewer DPAS per instruction than the old hand-written kernel). A hoisted-descriptor design builds copies once per group (or once per kernel with `.with()` per tile).
- Prefetch: `make_block_2d_prefetch(tiled_copy)` -> `prefetch_a.get_slice(id).partition_S(gA)`; `prefetch(pf, pAgA(_,_,_,k))` fetches K tile k into cache (no registers). The load and the prefetch have independent tilings: prefetch uses full 64 B rows and splits the whole tile across all sub-groups.
- Stores: `get_block_2d_copy_D<void>(mma, D)` with `partition_sg_fragment_S`; `XE_STORE_2D` height <= 8.
- Pitfall: a `Count`/`BlockWidth` combination violating the limits in [[hw-pvc-2d-block-io]] fails with a static_assert deep in templates; read the first assertion.
