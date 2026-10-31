---
id: ker-ggemm-forward-dx
title: Forward and grad_input (M-grouped): behaviour and limits
type: kernel
tags: [forward, grad-input, m-grouped, wave-quantization, k-contiguous, n-contiguous]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#11.2, notes_ggemm.txt#11.4, notes_ggemm.txt#22.2, notes_ggemm.txt#17.2]
related: [pat-tail-wave-quantization, pat-tile-padding-waste, tech-tile-shape-selection, ker-ggemm-regime-map, hw-pvc-2d-block-io]
---
# M-grouped ops

- dX (B N-contiguous, 'C'): XMX 65-70%, stalls 6-7% (SBID), but 28% idle; 127-155 TFLOPS.
- forward (B K-contiguous, 'R'): same shape of problem, SBID stall 14% (2x dX), 8-18% slower in earlier measurements; 126-142 TFLOPS. The K-contiguous B uses a different (transposing/VNNI-less) path, see [[hw-pvc-2d-block-io]].
- Idle share comes from (a) 74% row fill of a 256-row tile at ~190 rows, (b) wave quantization: 16 x 11 = 176 tiles (w13) / 56 = 3.14 waves; (c) skewed routing (`hot`): one expert's 5x tiles dominate the tail.
- 128x256 tile: up to 42% faster at <= 128 rows/expert, ~10% slower at ~190 rows (notes 17.2, 16.4); tile cannot be picked from host row counts.
- Long K means prefetch distance is not binding for these ops; look at the tail and the tile fill instead.
- Hot spots for ideas: n-major vs m-major linearization (L3 reuse of B for tiles_m>1), last-wave fine-grain tiles, row-tail handling.
