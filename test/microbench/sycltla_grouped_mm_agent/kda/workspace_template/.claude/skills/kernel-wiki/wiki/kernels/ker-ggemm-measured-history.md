---
id: ker-ggemm-measured-history
title: What has already been tried (and what it measured)
type: kernel
tags: [history, negative-results, tuner, cuda-style, 128x128, 128x256, onednn]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#14, notes_ggemm.txt#16, notes_ggemm.txt#17, notes_ggemm.txt#19.2, notes_ggemm.txt#22]
related: [ker-ggemm-regime-map, ker-ggemm-grad-weight, tech-tile-shape-selection, mig-cute-cuda-to-sycl-tla]
---
# Do not re-discover these

| attempt | result | source |
|---|---|---|
| CUDA-style two-kernel design (prepare kernel + CUTLASS group GEMM) | 21% slower layer (1.793 vs 1.481 ms): +15-19 us per call floor, dW +60% of the gap from epilogue/descriptor instructions | notes 14.6 |
| back to one persistent kernel | 1.498 ms layer at 190 rows (baseline of this project) | 16.4 |
| 128x128 tile for <= 64 rows | 11-97% faster at <= 64 rows, 3-60% slower from 128 rows | 16.2 |
| 128x256 for <= 128 rows | best from 16 to 128 rows (up to 42% over 256x256), then removed because the choice used host row counts | 17, 19.2 |
| auto tile rule on routed row counts | different kernels per rank -> crashes; removed | 19.2 |
| oneDNN grouped matmul | 1.6-2.0x slower than this kernel, fails on all-empty input | 22 |
| per-group fallback | 4-13% faster at 6k rows static split, 8.7x slower when split changes | 22.2 |
| prefetch distance | tuner: 2 beats 3 for dW | 11.4 |
| tuner noise | ~10% between runs in sweeps; evaluator noise <2% (paired A/B) | 16.2, README |

Ideas listed but not done (notes open items 3-5): split-K for dW, smaller prefetch, 128-row tile for dW, c256x256_sg4x8 / c512x128 large-group tile, stream-K, model-level tile hint, 128x128 for N <= 128.
