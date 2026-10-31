---
id: tech-prefetch-distance
title: Prefetch distance (kPrefetchDist) tuning
type: technique
tags: [prefetch, latency, mainloop, tuning, short-k]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#11.4, notes_ggemm.txt#0, kernel:grouped_mm_kernel_impl.h:gemm_tile, ggemm_agent/profile.py:DIAGNOSES]
related: [pat-short-k-prologue-bound, pat-sbid-load-latency, ker-ggemm-grad-weight, lang-sycl-tla-block-2d-copy]
---
# Prefetch distance

What it is: `gemm_tile` prefetches the first `kPrefetchDist` K tiles of A and B into cache before the loop (prologue), then each iteration issues the prefetch of tile `k + dist`. `prefetch_k < k_tile_count` bounds it (an empty K issues none).

Trade-off:
- Larger distance hides more HBM/L3 latency per iteration - helps long-K tiles (forward/dX: 44-88 K steps).
- Short-K tiles (dW: ~6 K steps at ~190 rows/expert) spend a large fraction of their life in the prologue; extra distance = extra prefetch messages that are unused or late. The tuner found distance 2 wins grad_weight (notes 11.4).
- Prefetch payloads also cost registers; at the 16 KB budget limit distance can trigger spills ([[pat-register-spills]]).

Config: `"prefetch": p` per kind in `configs.json` (no header edit needed). Try p in {1,2,3,4} per kind; the three kinds are independent.
Data-dependent distance (`min(dist, k_tile_count)` already) could be extended: distance = f(k_tile_count) computed on device from group rows - allowed because it reads `offs`, not host row counts.

Expected: single-digit % for dW (0.98-1.03 noise band, so confirm with `small` + the `rows64` probe). Cheap: first thing to try, one config line.

Evidence to record: per-op ratio for the three prefetch values; SBID% before/after from `ggemm-eval profile`.
