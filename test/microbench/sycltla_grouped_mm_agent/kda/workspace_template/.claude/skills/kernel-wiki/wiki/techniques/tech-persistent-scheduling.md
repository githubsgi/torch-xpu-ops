---
id: tech-persistent-scheduling
title: Persistent work-groups and the wgs (grid size) knob
type: technique
tags: [persistent, scheduler, grid, work-groups, wave-quantization]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#16.1, notes_ggemm.txt#14.7, kernel:grouped_mm_kernel_impl.h:MGroupedKernel, ggemm_agent/candidate.py]
related: [tech-grouped-gemm-scheduling, pat-tail-wave-quantization, tech-grf-mode-and-occupancy, hw-pvc-xe-core]
---
# Persistent kernel

`MGroupedKernel`/`KGroupedKernel`: `for (lin = wg; ; lin += nwg) { locate(lin) -> (group, m, n); gemm_tile(...) }`. Work-groups stride the linearized tile space; there is no atomic counter, so assignment is static round-robin (tile i goes to work-group i mod nwg).

Knob: `"wgs"` per kind (0 = Xe cores x 512 / threads = 56 / 112 / 224). Default is one resident work-group per 512 threads of an Xe core. Fewer than the default leaves cores idle; more than 1 resident per core is impossible in GRF-256 mode for 512-thread groups (they would queue, not co-run).

Static round-robin ⇒ tail effect: with T tiles and W work-groups the last "wave" has T mod W tiles; every group's tiles are equal-cost only if rows are equal. Skewed routing (the `hot` case) leaves the work-group that draws the heavy expert's tiles late. Fixes without scratch memory:
- reorder the linearization so heavy-to-light tile order is visited first (needs per-group tile counts, which are in `offs`);
- finer tiles at the end (device-side dispatch of a smaller tile for the last wave);
- N-major vs M-major linearization changes which tiles share operands in L3 (current: m fastest within a group, so consecutive tiles share the same n index, i.e. the same B tile when tiles_m > 1, or the same A tile when the group fits one M tile, as at ~190 rows).
Dynamic work stealing needs a global atomic counter = scratch memory that is zeroed per launch; the ABI has no workspace pointer (out of scope), see [[tech-grouped-gemm-scheduling]].
