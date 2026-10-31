---
id: tech-epilogue-simplification
title: Cheaper epilogue and per-tile setup (descriptor hoisting)
type: technique
tags: [epilogue, descriptors, per-tile-overhead, short-k, instruction-count]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#14.6, kernel:grouped_mm_kernel_impl.h:gemm_tile]
related: [pat-xmx-underfed, pat-short-k-prologue-bound, lang-sycl-tla-block-2d-copy, ker-ggemm-grad-weight]
---
# Per-tile overhead

Evidence (notes 14.6, CUTLASS-style group path vs old hand kernel, dW, 190 rows/expert): same XVE busy time (53-56%), but XMX util 24-26% vs 35-36%, code 2047 vs 885 IPs, InstrFetch stalls 7% vs 2% => "the new kernel runs more non-matrix instructions per DPAS, not more waiting". It scaled with K steps per tile: +30% slower at 190 rows, +2% at ~6100.

The current production kernel is the lean rewrite, but `gemm_tile` still, per tile: builds A/B/D tiled copies and two prefetches, three identity tensors and local_tiles, partitions fragments, clears the accumulator, runs a prologue, then `reorder(tCrD, tCrD_final)` + store (fp32 -> bf16 convert through the reorder path).

Ideas (rank by effort):
1. Build the MMA/copy objects outside the persistent loop (they are type-level constants); only rebind the tensors/coords per tile (`.with(...)`).
2. Hoist group-invariant pieces (per-group base pointers, tensor shapes) out of the per-tile path; the cursor already caches group data.
3. Store D as bf16 directly from the accumulator layout (avoid an extra reorder when the layout allows).
4. Unroll count: `CUTE_UNROLL` on the prefetch prologue and mainloop body influences code size and InstrFetch stalls (>= 5% is a flag).

Measure: instruction-level facts come from `ggemm-eval profile` (XMX util, stall mix, InstrFetch). Expect a visible effect only on dW and small-K shapes.
