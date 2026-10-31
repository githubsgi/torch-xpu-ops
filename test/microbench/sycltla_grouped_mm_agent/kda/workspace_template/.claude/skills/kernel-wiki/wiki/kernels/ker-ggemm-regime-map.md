---
id: ker-ggemm-regime-map
title: Workload regimes: rows per expert and which configuration wins
type: kernel
tags: [regimes, small-groups, large-groups, tile-choice, fallback, ep4, ep8]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#16.2, notes_ggemm.txt#17.2, notes_ggemm.txt#22.2, notes_ggemm.txt#22.4, notes_ggemm.txt#0]
related: [tech-tile-shape-selection, tech-device-side-dispatch, ker-ggemm-production, ker-ggemm-constraints, ker-ggemm-measured-history]
---
# Regimes (evaluator suites: small = TP4/EP4 16 experts, large = EP8 8 experts)

| regime | rows/expert | best known | notes |
|---|---|---|---|
| tiny / decode-like | 8-64 | 128x256 (fwd, dX) up to 42% over 256x256; 128x128 at 8 rows | scored only through the `tp4ep4_rows8/rows64` probes (not scored) |
| TP4/EP4 routed | ~190 (skewed in `hot`) | production 256x256 for M-grouped; dW weakest | `small` suite, scored: uniform, balanced, hot, sparse |
| EP8 routed | ~6k | production; the per-group dense fallback is 4-13% faster with a static split but is outside this harness (candidates are only compared with production) | `large` suite: ep8_uniform, balanced, hot, rows1024 probe |

SCORE = geomean over scored cases of production/candidate time (probes rows8/rows64/rows1024 are reported but not scored). Phase 3 specialization = different `configs.json` per kind (m_rowB / m_colB / k_grouped) or device-side per-group paths; never host-side row-count branching.

Performance claims
- [claim] gpu=PVC-tile dtype=bf16 shape=16 experts x 190 rows metric=layer ms value=1.498 (auto 256x256) vs 2.166 (forced 128x128) source=notes_ggemm.txt#16.4
- [claim] gpu=PVC-tile dtype=bf16 shape=8 experts x 6k rows metric=layer ms value=SYCL-TLA 17.72 vs fallback 16.92 (6144 rows) source=notes_ggemm.txt#22.2
