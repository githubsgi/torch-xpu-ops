---
id: pat-xmx-underfed
title: XVEs busy but XMX underfed (instruction overhead per DPAS)
type: pattern
tags: [xmx, instruction-overhead, send, reorder, ifetch, address-math]
architectures: [pvc]
confidence: source-reported
symptoms: [XVE active >= 60% but XMX util < 50%, Instruction-fetch stalls >= 5%, many more code IPs than a lean kernel, XMX share falls as K tiles per tile fall]
candidate_techniques: [tech-epilogue-simplification, tech-register-double-buffering, tech-compiler-flags, tech-tile-shape-selection]
sources: [notes_ggemm.txt#14.6, ggemm_agent/profile.py:DIAGNOSES]
related: [pat-short-k-prologue-bound, pat-instruction-fetch]
---
# Pattern: non-DPAS instructions compete with DPAS

**Evidence (14.6):** CUTLASS-style path ran XVE 53-56% active but XMX 24-26% vs 35-36% for the hand kernel; code size 2047 vs 885 IPs.
**First-line fix:** remove per-tile setup from the hot path (hoist descriptors/copies), avoid alpha/beta epilogues, keep reorders minimal.
**Deeper fixes:** larger sub-group tile (more DPAS per 2D message) within the register budget; fewer, taller 2D messages.
**Exceptions:** XMX 65-70% (dX) is near what this design reaches; do not chase it further.
