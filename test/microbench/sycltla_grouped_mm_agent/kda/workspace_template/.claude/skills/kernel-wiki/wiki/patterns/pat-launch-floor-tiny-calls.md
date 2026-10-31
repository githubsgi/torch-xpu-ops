---
id: pat-launch-floor-tiny-calls
title: Per-call floor and the offs walk dominating tiny problems
type: pattern
tags: [launch, floor, offs-walk, tiny, rows8]
architectures: [pvc]
confidence: source-reported
symptoms: [rows8 probe ms barely changes with problem size, time grows with number of groups not with FLOPs, work-groups mostly walk offs without tiles]
candidate_techniques: [tech-grouped-gemm-scheduling, tech-persistent-scheduling]
sources: [notes_ggemm.txt#14.7, notes_ggemm.txt#16.4]
related: [ker-ggemm-regime-map]
---
# Pattern: fixed cost per call

Tiny call (16 x 8 rows, K=N=128): 8-11 us (one-kernel) vs 17-28 us (CUDA-style). The per-group walk costs ~0.37 us/group (64 groups: 26 us). Fix: read offs once per work-group into registers/SLM instead of one dependent load per step. Relevant only to the unscored rows8 probe at current shapes.
