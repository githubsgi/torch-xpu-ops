---
id: tech-grf-mode-and-occupancy
title: GRF mode, work-group size and resident work-groups per Xe core
type: technique
tags: [grf, occupancy, work-group-size, spills, configs]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#18, notes_ggemm.txt#16.1, notes_ggemm.txt#11.2, kernel:grouped_mm_kernel_impl.h:launch]
related: [hw-pvc-grf-threads, pat-register-spills, pat-low-occupancy, tech-tile-shape-selection]
---
# Occupancy levers

| tile | threads | WGs/Xe core | WGs/tile | threads per XVE |
|---|---|---|---|---|
| 256x256 sg8x4 | 512 | 1 | 56 | 4 |
| 256x128 / 128x256 | 256 | 2 | 112 | 2 per WG, 4 total |
| 128x128 sg4x2 | 128 | 4 | 224 | 4 total |

All fill the 4 hardware threads per XVE of GRF-256 mode; the difference is how many independent work-groups (barrier domains, tile streams) share a core. Smaller work-groups mean (a) a barrier rendezvous of 8-16 instead of 32 sub-groups, (b) several tiles in different phases per core (one in prologue, another in DPAS) which covers the exposed prologue of short-K tiles, (c) lower operand reuse.
Measured occupancy (VTune, 256x256): 34-41% thread occupancy, XVE idle 17-33%.

Levers: `tile`/`sg` (changes WG size), `wgs` (grid), `extra_flags` (compiler register mode, experimental). Unexplored: 256 threads with a 256x256 tile (sg 4x4, 64x64 per sub-group - needs 2x accumulator registers -> certain spills in GRF-256), or 4 sub-group rows with 16x... Check the spill line in the BUILD summary first.
