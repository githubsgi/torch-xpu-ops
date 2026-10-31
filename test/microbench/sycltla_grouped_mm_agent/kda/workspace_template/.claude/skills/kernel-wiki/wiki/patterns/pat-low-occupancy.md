---
id: pat-low-occupancy
title: Low thread occupancy / few resident work-groups
type: pattern
tags: [occupancy, work-groups, grid, threads]
architectures: [pvc]
confidence: source-reported
symptoms: [XVE_THREADS_OCCUPANCY < 35%, grid has fewer work-groups than Xe cores, wgs set below default]
candidate_techniques: [tech-grf-mode-and-occupancy, tech-persistent-scheduling, tech-tile-shape-selection]
sources: [notes_ggemm.txt#11.2, notes_ggemm.txt#16.1]
related: [hw-pvc-xe-core, pat-tail-wave-quantization]
---
# Pattern: occupancy

Reference: 34-41% thread occupancy for the production 256x256 kernels (28,672 global work-items = 56 x 512 = 1 work-group per Xe core). Occupancy is capped by 4 threads/XVE in GRF-256 mode, so 100% is not reachable with 512-thread groups in 128-GRF-sized tiles.
Fix candidates: more work-groups of a smaller tile (224 x 128 threads), or `wgs` changes; verify the XVE idle share changes, not just occupancy.
Exception: occupancy by itself is not a target; a high-XMX kernel at 36% occupancy is fine (dX).
