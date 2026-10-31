---
id: ker-ggemm-grad-weight
title: grad_weight (K-grouped) - the weakest op
type: kernel
tags: [grad-weight, k-grouped, short-k, weakest-op, dW]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#0, notes_ggemm.txt#11.4, notes_ggemm.txt#14.6, notes_ggemm.txt#16.2, notes_ggemm.txt#22.2]
related: [pat-short-k-prologue-bound, pat-sbid-load-latency, tech-prefetch-distance, tech-split-k-stream-k, tech-epilogue-simplification, ker-ggemm-production]
---
# grad_weight at small groups

Facts:
- 84-88 TFLOPS at TP4/EP4 (~190 rows/expert) vs 126-137 fwd and 127-131 dX; at ~6k rows 129-138 TFLOPS: **the problem is short K**, not the kernel body.
- Each expert contributes ~190 rows of contraction = ~6 K tiles of 32. Output per tile is 256x256 bf16 = 128 KB vs ~195 KB of operands read: the epilogue store is ~40% of the per-tile traffic and the prologue (prefetch dist 3 of 6) is half the loop.
- Tiles: w13_dW 11 (M=2816) x 8 (N=2048) = 88 per expert, 1408 per launch over 56 work-groups = 25 waves -> wave quantization is small here; the loss is per-tile latency (XVE 53% active, 30% stalled; SBID 31-37%, XMX 34-36%; HBM 170-185 GB/s, L3 ~720 GB/s, far from limits).
- Empty experts still own tiles that store zeros (K=0).
- Operands: A is M-contiguous (transposed activation) and B N-contiguous viewed as column-major -> different copy atoms than fwd/dX.
- Tuner: prefetch distance 2 wins dW; the four tiles are within a few % (noise ~10%); 128x128 +10% on w2_dW at routed shapes only.
- Old hand kernel vs CUTLASS-style path: +30% dW time at 190 rows, +2% at 6k rows - per-tile overhead (epilogue, descriptors) is the lever (notes 14.6).

Candidate directions, with the stated expected effect (all unverified): prefetch 1-2 for k_grouped ([[tech-prefetch-distance]]); cross-tile prefetch ([[tech-split-k-stream-k]] item 2); intra-work-group K slicing; per-tile setup hoisting ([[tech-epilogue-simplification]]); 128x128 tile for k_grouped only.
