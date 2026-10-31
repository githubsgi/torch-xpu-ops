---
id: hw-pvc-memory-hierarchy
title: PVC memory hierarchy: L1/SLM, L3 ("L2" in Intel specs), HBM, measured bandwidth
type: hardware
tags: [pvc, l1, slm, l3, hbm, bandwidth, roofline]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#11.3, notes_ggemm.txt#18, notes_ggemm.txt#11.4]
related: [hw-pvc-xe-core, hw-pvc-2d-block-io, pat-memory-bound-expected, tech-tile-shape-selection]
---
# Memory levels (one tile)

| level | size | bandwidth | note |
|---|---|---|---|
| GRF | 512 KB / Xe core | - | the kernel stages A/B fragments straight in registers, no SLM |
| L1 / SLM | 512 KB / Xe core | SLM ceiling 23 TB/s (Advisor) | this kernel uses 0 B SLM |
| L3 (unitrace "L3", Intel spec "L2") | 204 MB / tile | 2.84 TB/s measured ceiling | operand tiles shared by work-groups hit here |
| HBM | per tile | 1.6 TB/s spec, **1.18 TB/s measured** | |

Measured utilisation of the production kernel (Advisor/VTune, TP4/EP4, notes 11.2-11.3): 15-18% of GTI, 36-52% of L3, HBM read 170-225 GB/s. **No op is memory-bound**; all three kernels sit far right of the roofline ridge (300-550 FLOP/byte of HBM traffic).

=> bytes-reduction ideas (compression, fp8 operands, more reuse) do not help here. The headroom is latency hiding, tile fill, waves and instruction overhead.

Operand reuse arithmetic (inferred): a 256x256 tile with K-step 32 loads (256+256) x 32 x 2 B = 32 KB per K step for 2 x 256 x 256 x 32 = 4.2 MFLOP => 128 FLOP/B from L3/L1. A 128x256 tile: 24 KB per 2.1 MFLOP = 87 FLOP/B. Smaller tiles raise L3 traffic per FLOP but remain far below the L3 ceiling at current speeds; their loss at ~190 rows is (inferred, not yet isolated) per-tile overhead and lost operand reuse rather than bandwidth (see [[ker-ggemm-regime-map]]).
