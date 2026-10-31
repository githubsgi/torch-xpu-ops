---
id: tech-device-side-dispatch
title: Device-side per-group specialization that keeps one kernel for all ranks
type: technique
tags: [dispatch, specialization, device-side, offs, rank-independence, small-groups]
architectures: [pvc]
confidence: experimental
sources: [notes_ggemm.txt#19.2, notes_ggemm.txt#17.2, ker-ggemm-constraints]
related: [tech-tile-shape-selection, ker-ggemm-regime-map, ker-ggemm-constraints, tech-persistent-scheduling]
---
# Specializing by the data that only the device knows

Rule from notes 19.2: the host may not pick tiles from routed row counts; different ranks running different kernels crashed 3 of 4 EP8 runs. But **one kernel that branches on `offs` on the device is rank-independent** by construction.

Design sketch (experimental, never built):
- Two tile paths in one kernel, e.g. 256-row path and 128-row path instantiated as two `gemm_tile` template calls. For each group, `rows <= 128` (or last partial tile with `rows % 256 <= 128`) uses the 128x256 path, else 256x256.
- The 128x256 tile uses 16 sub-groups: in a 32-sub-group work-group, half the sub-groups idle for those tiles, or the work-group splits into two independent halves processing two 128x256 tiles (needs split barriers per half; the barrier scope is work-group-wide).
- Costs: code size (InstrFetch stalls), register allocation = max of both paths, extra scheduler complexity.
Simplest slice: M-grouped only, last-partial-tile specialization. Wins only where rows % 256 is small and many groups exist.

Evidence required before keeping: `small` AND `large` suites (rows64, rows1024 probes), no op regression > 3%, spills none.
