---
id: pat-short-k-prologue-bound
title: Short contraction: prologue and epilogue dominate the tile (grad_weight)
type: pattern
tags: [short-k, prologue, epilogue, sbid, xmx-low, dW]
architectures: [pvc]
confidence: source-reported
symptoms: [SBID stall samples >= 20%, XMX utilization < 45% with XVE active ~55%, K tiles per output tile <= 12 (rows/expert/32), HBM and L3 far below ceiling, op is k_grouped]
candidate_techniques: [tech-prefetch-distance, tech-split-k-stream-k, tech-epilogue-simplification, tech-tile-shape-selection, tech-register-double-buffering]
sources: [notes_ggemm.txt#11.4, notes_ggemm.txt#14.6, notes_ggemm.txt#22.2]
related: [ker-ggemm-grad-weight, pat-sbid-load-latency]
---
# Pattern: short K

**Signals (unitrace):** SBID >= 20% (dW production: 31-37%), XMX util 24-36%, XVE active ~53%, stalled ~30%, idle ~17%; HBM 170-185 GB/s, L3 ~720 GB/s. K tiles per tile = group rows / 32 (about 6 at 190 rows).
**Why:** with 6 K steps the prefetch prologue (dist 3) is half the loop, load-to-use latency is exposed in each of the first iterations, and the 256x256 bf16 store (128 KB) is ~40% of the tile's input bytes.
**First-line fix:** `prefetch` 2 (or 1) for `k_grouped` (config only). Then a `128x128` or `256x128` dW tile (more, smaller tiles in flight).
**Deeper fixes:** prefetch the next tile's first K steps before the current epilogue store (cross-tile pipelining); hoist per-tile descriptor setup; intra-work-group K slicing with SLM reduction.
**Exceptions:** at ~6k rows/expert (EP8) dW reaches 129-138 TFLOPS and this pattern does not apply.
**Verify:** SBID% and XMX% of the dW kernel in `ggemm-eval profile ... --op w13_dW`, `tp4ep4_uniform`; score on `small` suite, probe `tp4ep4_rows8`.
