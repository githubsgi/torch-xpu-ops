---
id: pat-barrier-sync-stalls
title: Barrier and dependency (Sync/Dist) stalls
type: pattern
tags: [barrier, sync, dist, stalls, split-barrier]
architectures: [pvc]
confidence: inferred
symptoms: [Sync + Dist stall samples >= 15%, barrier message count high relative to K tiles, short loops with a work-group barrier every iteration]
candidate_techniques: [tech-grf-mode-and-occupancy, tech-tile-shape-selection, tech-prefetch-distance]
sources: [ggemm_agent/profile.py:DIAGNOSES, sycl-tla:include/cute/util/xe_split_barrier.hpp]
related: [lang-sycl-tla-split-barrier]
---
# Pattern: barrier-bound

**Signals:** unitrace stall mix `Sync`, `Dist`; `barrier msgs` in the digest.
**Why:** 32 sub-groups rendezvous per K step; one slow sub-group delays all. With ~6 K steps per dW tile the barrier fraction rises.
**First-line fix:** smaller work-groups (256x128 / 128x256 -> 16 sub-groups) or fewer barriers (every 2 steps).
**Exceptions:** production shows no large Sync share today (check before spending effort).
