---
id: pat-instruction-fetch
title: Instruction-fetch stalls from code bloat
type: pattern
tags: [ifetch, code-size, unroll, templates]
architectures: [pvc]
confidence: source-reported
symptoms: [InstrFetch stall samples >= 5%, code IP count in the thousands, new variants merged into one kernel]
candidate_techniques: [tech-compiler-flags, tech-epilogue-simplification, tech-device-side-dispatch]
sources: [notes_ggemm.txt#14.6, ggemm_agent/profile.py:DIAGNOSES]
related: [pat-xmx-underfed]
---
# Pattern: code too large

Reference: lean kernel 885 IPs / 2% InstrFetch vs CUTLASS-style 2047 IPs / 7%. Beware when merging specializations into one kernel ([[tech-device-side-dispatch]]): both paths share the instruction cache.
Fix: reduce `CUTE_UNROLL`, share one mainloop between paths, drop per-tile template instantiations.
