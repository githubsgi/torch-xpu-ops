---
id: pat-sbid-load-latency
title: Scoreboard (SBID) stalls: sub-groups wait for 2D block loads
type: pattern
tags: [sbid, scoreboard, load-latency, prefetch, forward]
architectures: [pvc]
confidence: source-reported
symptoms: [SBID stall samples >= 12% with long K, forward (K-contiguous B) stalls about 2x dX, XMX util below 65% with idle XVE threads, prefetch distance already >= 3]
candidate_techniques: [tech-prefetch-distance, tech-register-double-buffering, tech-epilogue-simplification, tech-grf-mode-and-occupancy]
sources: [notes_ggemm.txt#11.4, ggemm_agent/profile.py:DIAGNOSES]
related: [pat-short-k-prologue-bound, hw-pvc-2d-block-io, ker-ggemm-forward-dx]
---
# Pattern: load latency (SBID)

**Signals:** `Sbid` share of IP samples (profile "stall samples" line); reference: fwd 14%, dX 7%, dW 31-37%.
**Why:** DPAS needs the registers the 2D load is still filling; with 4 hardware threads per XVE (GRF-256) there are few other threads to switch to. K-contiguous B (forward) needs a less efficient load path than N-contiguous (dX).
**First-line fix:** prefetch distance up (long-K ops) or down (short-K): config `prefetch`; compare per-op.
**Deeper fixes:** issue the next K step's loads before the current DPAS (register double buffering, check spills); change the K step (16 vs 32); taller load blocks / fewer messages.
**Exceptions:** SBID is partly unavoidable in a DPAS loop; a kernel with XMX > 65% and SBID < 10% (dX) should be left alone.
