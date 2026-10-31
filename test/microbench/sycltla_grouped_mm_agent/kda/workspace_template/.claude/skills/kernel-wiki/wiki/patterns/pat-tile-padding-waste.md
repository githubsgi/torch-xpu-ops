---
id: pat-tile-padding-waste
title: Padding waste: rows % tile_m and N % tile_n
type: pattern
tags: [padding, small-groups, tile-fill, rows64, rows8]
architectures: [pvc]
confidence: source-reported
symptoms: [rows per expert much smaller than tile_m, rows8 or rows64 probes far below uniform TFLOPS, tile fill rows/(ceil(rows/tile_m)*tile_m) < 0.75, N=1408 with a 256-wide N tile]
candidate_techniques: [tech-tile-shape-selection, tech-device-side-dispatch]
sources: [notes_ggemm.txt#16.2, notes_ggemm.txt#17.2]
related: [pat-tail-wave-quantization, ker-ggemm-regime-map]
---
# Pattern: useless FLOPs in edge tiles

Zero-filled loads still issue DPAS: a 190-row group in a 256-row tile does 26% wasted math; 8 rows waste 97%. 128x256 halves M padding; 128x128 halves both. Gains seen: up to 42% (<=128 rows) and losses ~10% at ~190 rows because operand reuse and tile count trade off.
First-line: per-kind tile in `configs.json` (static). Deeper: device-side specialization by group size ([[tech-device-side-dispatch]]).
Exceptions: the score weights routed cases (uniform/balanced/hot/sparse at ~190 rows); the 8/64-row probes are not scored - do not trade scored cases for probes.
