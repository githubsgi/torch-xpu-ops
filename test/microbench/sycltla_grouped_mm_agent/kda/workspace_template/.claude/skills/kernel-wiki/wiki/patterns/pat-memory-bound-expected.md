---
id: pat-memory-bound-expected
title: When HBM or L3 bandwidth is actually the limit (not the case today)
type: pattern
tags: [memory-bound, hbm, l3, roofline, expected-negative]
architectures: [pvc]
confidence: source-reported
symptoms: [HBM read+write > 800 GB/s, L3 read > 2 TB/s, FLOP per byte below 100]
candidate_techniques: [tech-tile-shape-selection, tech-persistent-scheduling]
sources: [notes_ggemm.txt#11.3]
related: [hw-pvc-memory-hierarchy]
---
# Pattern: memory bound (expected NOT to apply)

All three production kernels use 15-18% of GTI bandwidth and 36-52% of L3 (notes 11.3). If a candidate drives HBM > 800 GB/s or L3 > 2 TB/s (small tiles, lost reuse), then bytes are the new bottleneck: go back to larger tiles or better tile order for L3 reuse. Otherwise do NOT spend iterations on bandwidth ideas.
