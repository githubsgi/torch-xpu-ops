---
id: tech-register-double-buffering
title: Register double buffering / software pipelining the K loop
type: technique
tags: [mainloop, pipelining, registers, latency, dpas]
architectures: [pvc]
confidence: experimental
sources: [notes_ggemm.txt#14.6, kernel:grouped_mm_kernel_impl.h:gemm_tile]
related: [pat-sbid-load-latency, hw-pvc-grf-threads, pat-register-spills, tech-prefetch-distance]
---
# Overlap the load of K step k+1 with the DPAS of step k

Current loop: `copy(A,B tile k) -> prefetch(k+dist) -> reorder -> gemm -> barrier_wait`. The loads for step k are consumed immediately (`reorder`, DPAS), so a load-to-use latency is exposed unless the hardware thread count hides it (4 threads/XVE in GRF-256 mode, so hiding is limited). A pipelined loop issues the load of step k+1 into a second fragment pair before running DPAS on step k.

Budget: second fragment pair = +6 KB per thread (A 2 KB + B 4 KB for the 32x64 sub-group tile) on top of ~14 KB => over the 16 KB register file (inferred). Options: halve the K step (K=16 with two buffers), shrink the sub-group tile to 32x32 (accumulator 4 KB) at the cost of more loads per FLOP, or double buffer only B.

Verify with the build's spill report (`register spills:` line in the eval BUILD line, `Spill Memory Per Thread` in profile properties). If it spills, reject and record.
Notes 14.6 lists this as an idea for dW (old kernel had 28% SBID vs new 20%, but the issue there was instruction count).
