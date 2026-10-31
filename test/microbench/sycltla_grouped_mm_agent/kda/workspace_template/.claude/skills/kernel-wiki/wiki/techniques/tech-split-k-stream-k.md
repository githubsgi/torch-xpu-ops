---
id: tech-split-k-stream-k
title: Split-K, stream-K and intra-work-group K slicing (without scratch memory)
type: technique
tags: [split-k, stream-k, k-grouped, short-k, reduction, slm]
architectures: [pvc, h100, b200]
confidence: experimental
sources: [notes_ggemm.txt#0, notes_ggemm.txt#11.4, sycl-tla:examples/03_bmg_gemm_streamk, ker-ggemm-constraints]
related: [pat-short-k-prologue-bound, ker-ggemm-grad-weight, tech-persistent-scheduling, mig-cute-cuda-to-sycl-tla]
---
# Split-K family and what is possible here

Standard split-K / stream-K (sycl-tla `examples/03_bmg_gemm_streamk`) partitions the K loop across work-groups and reduces partial accumulators through a **global workspace**. This kernel's `Params` ABI has no workspace pointer and the launch path may not allocate (task constraint), and output is bf16 so partial sums cannot be accumulated in the output buffer without precision loss. => classic split-K is **out of scope**.

In-scope substitutes (all experimental, none tried yet):
1. **Intra-work-group K slicing**: split the sub-groups of a work-group into S slices; each slice runs the K loop over every S-th K tile of the SAME output tile, then partial accumulators are added through SLM (up to 128 KB per work-group; a 256x256 fp32 tile is 256 KB, so reduce in chunks or use a smaller tile such as 128x128 fp32 = 64 KB). Targets dW's short K: more independent load streams in flight per tile. Costs: SLM traffic, an extra barrier, and fewer DPAS per sub-group tile if the sub-group tile shrinks.
2. **Tile pipelining across persistent iterations**: issue the next tile's prologue prefetches before the current tile's epilogue store (works for any K, no reduction needed). Cheapest relative of the stream-K idea: it removes the exposed prologue that makes short-K tiles latency bound.
3. **Smaller output tile for dW** (128x128: 4x the tiles, each with the same K): no reduction needed, more tiles in flight, but less operand reuse; the tuner saw it win w2_dW by ~10% and lose elsewhere (noise ~10%).

Expected gain order of magnitude: dW is ~43% of the layer time (0.41+0.20 of 1.43 ms, derived from the TFLOPS in notes 22.2); closing half the gap to fwd/dX efficiency (84 -> ~110 TFLOPS) would be ~10% of the layer.
