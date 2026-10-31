---
id: lang-sycl-tla-split-barrier
title: Split barriers (barrier_arrive / barrier_wait) in the mainloop
type: language
tags: [sycl-tla, barrier, split-barrier, synchronization, mainloop]
architectures: [pvc]
confidence: verified
sources: [sycl-tla:include/cute/util/xe_split_barrier.hpp, kernel:grouped_mm_kernel_impl.h:gemm_tile]
related: [pat-barrier-sync-stalls, lang-sycl-tla-cute-xe-api, tech-prefetch-distance]
---
# `barrier_arrive(scope)` / `barrier_wait(scope)`

- Implemented with `__spirv_ControlBarrierArriveINTEL` / `...WaitINTEL`: the arrive half signals "I reached the barrier" without blocking; the wait half blocks until all peers arrived. The production loop does `barrier_arrive(2)` at the top of each K iteration and `barrier_wait(2)` at the bottom, so a sub-group's loads/DPAS of iteration k overlap with slower peers finishing iteration k-1 (scope 2 = `ScopeWorkgroup`; 3 = sub-group).
- Purpose here: keep the sub-groups of a work-group within roughly one K step of each other so they hit the same prefetched lines in L1/L3 (no SLM is used, so no data hazard requires it).
- Costs: every barrier is a work-group-wide rendezvous; unitrace reports `XVE_BARRIER_MESSAGE_COUNT` and the stall-sample classes `Sync`/`Dist`. With 32 sub-groups per work-group the slowest sub-group sets the pace.
- Experiments worth measuring (experimental): barrier every 2 K steps; arrive earlier (before the reorder/DPAS of the previous step); remove the barrier entirely for short-K (dW) tiles where the loop has only ~6 iterations.
