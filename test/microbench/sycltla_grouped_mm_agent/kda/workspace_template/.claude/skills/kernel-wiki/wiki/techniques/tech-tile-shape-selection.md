---
id: tech-tile-shape-selection
title: Work-group tile shape per kernel kind
type: technique
tags: [tile-shape, wave-quantization, padding, configs, small-groups]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#16.2, notes_ggemm.txt#17.2, notes_ggemm.txt#19.2, notes_ggemm.txt#0]
related: [ker-ggemm-regime-map, pat-tile-padding-waste, pat-tail-wave-quantization, tech-device-side-dispatch, lang-sycl-tla-mma-tiling]
---
# Choosing the tile (`tile`, `sg` in configs.json)

Measured (tuner, ms, one PVC tile; notes 16.2/17.2, noise ~10% in sweeps):
- forward / dX at **<= 128 rows per expert**: 128x256 beats 256x256 by up to 42% (and 128x128 by 5-43% from 16 to 128 rows); 128x128 wins only at 8 rows with 16 experts, and with 8 experts for w2_dX it ties.
- from **~160 rows** the 256-row tiles win; at ~190 rows forcing 128x128 gave 2.166 ms vs 1.498 ms for the layer (notes 16.4).
- dW: all four tiles within a few %; 128x128 won w2_dW at routed shapes by ~10% but lost elsewhere; not enabled.
- ~6k rows (EP8): the per-group dense fallback is 4-13% faster than 256x256; tuner shows c256x256_sg4x8 and c512x128 as candidates (open item 4).

Constraint (notes 19.2): the tile must NOT depend on host-known routed row counts; all EP ranks run the same kernel. Allowed levers: a fixed tile per kind (m_rowB, m_colB, k_grouped are separate configs), device-side branching on `offs` ([[tech-device-side-dispatch]]).

Arithmetic helpers (w13: N=2816, w2: N=2048/1408, K=2048/1408):
- tiles per group = ceil(rows/tile_m) x ceil(N/tile_n); waves = tiles x groups / work-groups. 256x256 at 16 experts x 190 rows: w13_fwd 16x11 = 176 tiles / 56 WGs = 3.14 waves -> 79% wave efficiency; w2_fwd (N=2048) 128 tiles = 2.29 waves -> 76%.
- row fill: 190/256 = 74% of the M tile does useful work, 190/128 -> 2 tiles = 74% as well but more tiles per wave.
- N tile 256 vs N=1408 (w2_dX): 1408 / 256 = 5.5 -> 6 tiles, 8% padding; vs 128: 11 tiles, 0%.

Validate changes with `small` + `large` suites: a tile that wins small can lose large.
