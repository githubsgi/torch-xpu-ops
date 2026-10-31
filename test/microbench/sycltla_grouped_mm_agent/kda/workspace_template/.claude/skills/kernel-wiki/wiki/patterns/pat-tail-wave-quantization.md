---
id: pat-tail-wave-quantization
title: Idle XVEs from wave quantization and heavy-expert tails
type: pattern
tags: [idle, tail, waves, quantization, scheduler, hot-expert]
architectures: [pvc]
confidence: source-reported
symptoms: [XVE idle >= 25%, occupancy < 40% with XMX util > 60% while active, tiles / work-groups has a fractional part < 0.5, hot or sparse routing cases score worse than uniform]
candidate_techniques: [tech-tile-shape-selection, tech-persistent-scheduling, tech-device-side-dispatch, tech-grouped-gemm-scheduling]
sources: [notes_ggemm.txt#11.4, notes_ggemm.txt#16.2]
related: [ker-ggemm-forward-dx, pat-tile-padding-waste]
---
# Pattern: partial last wave

**Signals:** fwd/dX idle 28-33% (VTune) with healthy XMX (65-70%) when busy; the profile digest prints `idle`. Compute waves = tiles / work-groups (profile script prints it).
**Why:** each group is rounded up to whole tiles, tiles are dealt round-robin statically; with 16 groups x 11 N-tiles over 56 work-groups the last wave is 14% full. Skewed routing makes one group's tiles the long tail.
**First-line fix:** smaller M tile for small groups (but see the ~190 row loss) or order the tile list so equal-cost tiles end together.
**Deeper fixes:** device-side finer tiles for the last wave; heavy-first ordering from `offs`; N tile 128 for N where `ceil(N/256)` wastes (w2_dX: N=1408).
**Exceptions:** large-group EP8 (60+ waves): negligible.
