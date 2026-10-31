---
id: hw-pvc-grf-threads
title: GRF modes (128 vs 256 registers) and hardware threads per XVE
type: hardware
tags: [pvc, grf, registers, occupancy, spills]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#18, kernel:grouped_mm_kernel_impl.h:launch, unitrace kernel properties]
related: [hw-pvc-xe-core, pat-register-spills, tech-grf-mode-and-occupancy, hw-pvc-dpas-xmx]
---
# GRF modes and threads per XVE

- Default mode: 128 registers x 64 B per hardware thread = 8 KB; **8 threads per XVE** (64 per Xe core).
- Large-GRF mode (`intelex::grf_size<256>`, what `launch()` sets): 256 registers x 64 B = **16 KB per thread**; **4 threads per XVE** (32 sub-groups per Xe core).
- A sub-group (16 lanes) owns one hardware thread: its register budget is 16 KB, i.e. 1 KB per lane.
- Occupancy ceiling in 256 mode: 32 sub-groups per Xe core = one 512-thread work-group (256x256 tile, sg 8x4) or two 256-thread work-groups.

Why the production sub-group tile is 32x64 (inferred arithmetic, check with a build):
- fp32 accumulator 32x64 = 2048 floats = 8 KB (half of the 16 KB budget).
- bf16 A fragment 32x32 = 2 KB, bf16 B fragment 32x64 = 4 KB per K step (more if the load is double buffered).
- => ~14 KB of 16 KB before address registers and prefetch payloads: any bigger per-sub-group tile, a second buffered K step in registers, or `kPrefetchDist` payload growth tends to spill. The evaluator prints `register spills: ...` in every BUILD line; `ggemm-eval profile` shows `Spill Memory Per Thread`.
- Changing to GRF-128 (smaller footprint, 8 threads/XVE) requires a sub-group accumulator of at most ~16x64 or 32x32; this doubles B/A reload per FLOP.

Compiler/driver flag for register mode exists (`-ftarget-register-alloc-mode=pvc:large|auto` in icpx) but the kernel sets it per-kernel through the `grf_size` property; do not combine both without checking the build log (confidence: experimental).
