---
id: pat-register-spills
title: Register spills in GRF-256 mode
type: pattern
tags: [spills, grf, registers, private-memory]
architectures: [pvc]
confidence: verified
symptoms: [build summary lists register spills, unitrace Spill Memory Per Thread > 0, scratch/private memory traffic, perf collapses after enlarging the sub-group tile or prefetch distance]
candidate_techniques: [tech-grf-mode-and-occupancy, tech-register-double-buffering, tech-tile-shape-selection]
sources: [ggemm_agent/profile.py:DIAGNOSES, ggemm_agent/evaluate.py:summarize, notes_ggemm.txt#11.3]
related: [hw-pvc-grf-threads]
---
# Pattern: spills

**Signals:** `BUILD: ok ...; register spills: <kernels>` in `ggemm-eval run`, `Spill Memory Per Thread` in the profile "properties" line. Production: none.
**Why:** 32x64 fp32 accumulators already take 8 KB of the 16 KB per thread; extra fragments or buffering exceed it.
**First-line fix:** revert the last register-growing change; reduce prefetch distance; 32x32 sub-group tile.
**Exceptions:** none - a spilling candidate is almost always slower; reject and record the numbers.
