---
id: lang-sycl-tla-mma-tiling
title: TiledMMAHelper: work-group tile, sub-group layout and what each sub-group owns
type: language
tags: [sycl-tla, tiled-mma, sub-group, tile-shape, fragment]
architectures: [pvc, bmg]
confidence: verified
sources: [sycl-tla:include/cute/atom/mma_atom.hpp#TiledMMAHelper, kernel:grouped_mm_kernel_impl.h:TileCfg*, ggemm_agent/candidate.py:normalize]
related: [lang-sycl-tla-cute-xe-api, hw-pvc-dpas-xmx, hw-pvc-grf-threads, tech-tile-shape-selection]
---
# `TiledMMAHelper<MMA_Atom, Layout<WGTile>, SGLayout>`

- `WGTile = Shape<_M,_N,_K>` is the per-work-group tile (K = K-step per mainloop iteration, 32 in all production configs = two DPAS K=16 slices).
- `SGLayout = Layout<Shape<_sgM,_sgN,_1>, Stride<_sgN,_1,_0>>` arranges sub-groups over (M, N); N-fastest ("n-major"). Sub-group tile = (M/sgM) x (N/sgN) and must be a multiple of the atom (8 x 16).
- Work-group size = sgM x sgN x 16 work-items (<= 1024).
- Production: all configs use a **32x64 per-sub-group block** (256x256: sg 8x4; 256x128: 8x2; 128x256: 4x4; 128x128: 4x2). Bigger blocks (e.g. 64x64, 32x128) mean more DPAS per loaded byte but ~2x the accumulator registers; see [[hw-pvc-grf-threads]] for why 32x64 is the register-budget edge in GRF-256 mode.
- A and B fragments are loaded per sub-group for ITS sub-tile: with sg 8x4 each A row-slab (32 rows) is loaded by the 4 sub-groups in the same sg row, and each B slab by the 8 sub-groups in the same sg column (redundant L1 hits, not redundant HBM reads). Sub-group layouts that put more sub-groups along the longer dimension change this duplication. `make_block_2d_prefetch` instead splits the tile among all sub-groups without duplication.
- In `configs.json` the evaluator generates `SGLayout` with N-fastest strides. For a different arrangement (m-major) use the `body` field to supply a custom struct body.

Extra per-kind knobs: `extra` (appended `static constexpr` members, e.g. `kSplitK`), which your edited header must read; `body` (whole struct body).
