---
id: hw-pvc-2d-block-io
title: Xe 2D block load/store/prefetch: shapes, limits, VNNI and transpose
type: hardware
tags: [pvc, block-2d, load, store, prefetch, vnni, transpose]
architectures: [pvc, bmg]
confidence: verified
sources: [sycl-tla:include/cute/arch/copy_xe_2d.hpp, sycl-tla:include/cute/atom/copy_traits_xe_2d.hpp, kernel:grouped_mm_kernel_impl.h:gemm_tile]
related: [lang-sycl-tla-block-2d-copy, hw-pvc-memory-hierarchy, tech-prefetch-distance, ker-ggemm-grad-weight]
---
# 2D block messages (`lsc_load_block2d`, `lsc_store_block2d`)

Limits encoded in `XE_Copy_Op_2D_Base` (static_asserts, so violating them is a compile error):
- Height <= 32 rows (loads, prefetch). **Stores: Height <= 8**.
- Bits x Width <= 512 bits = **64 bytes per row** (32 bf16 elements).
- Block count (`Count = Width/BlockWidth`): Bits x Count <= 64, Count <= 4, Count != 3.
- Flavors: `XE_LOAD_2D<Bits,H,W,BlockW>` (plain), `XE_LOAD_2D_VNNI` (8/16-bit only; packs pairs for the DPAS B operand), `XE_LOAD_2D_TRANSPOSE` (32/64-bit only, Width <= 8; gives K-contiguous data for operands whose memory is transposed), `XE_PREFETCH_2D` (`lsc_load_block2d ... %null`, brings lines into cache without registers), `XE_STORE_2D`.
- Out-of-bounds rows/columns are zero-filled on load and masked on store: this is why ragged group sizes need no padding in this kernel.
- The base address and pitch must satisfy the descriptor alignment rules; production `build_plan()` rejects calls unless base pointers are 64-byte aligned and leading dims are multiples of 32 elements (otherwise the per-group fallback runs).

How `get_block_2d_copy_A/B/D<void>(mma, tensor)` picks atoms: from the tensor strides it finds the contiguous (x) mode and the strided (y) mode, then chooses plain / VNNI / transpose and a height/width that tile the sub-group's fragment. So:
- forward B (K-contiguous 'R'), dX B (N-contiguous 'C') and the dW operands (M-contiguous 'C') use different atoms: they have different load efficiency. Notes sec. 11.4: forward (K-contig B) stalls ~2x dX (14% vs 7% SBID).
- `make_block_2d_prefetch(tiled_copy)` derives a prefetch tiling with cache-line-wide (64 B) rows and height up to 32; it can be a different shape than the load (see sycl-tla `make_block_2d_prefetch`).

Each message is issued per sub-group and costs descriptor/address setup; fewer, larger messages (taller blocks, `Count` 2) reduce `send` instruction count (see [[pat-xmx-underfed]]).
