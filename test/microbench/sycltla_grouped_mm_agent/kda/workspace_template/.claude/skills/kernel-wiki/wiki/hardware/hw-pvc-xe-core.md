---
id: hw-pvc-xe-core
title: PVC Xe core, XVE, XMX and the tile as the unit this kernel runs on
type: hardware
tags: [pvc, xe-core, xve, xmx, execution-model]
architectures: [pvc]
confidence: source-reported
sources: [notes_ggemm.txt#18, notes_ggemm.txt#11.2, kernel:grouped_mm_kernel_impl.h:launch]
related: [hw-pvc-grf-threads, hw-pvc-dpas-xmx, hw-pvc-memory-hierarchy, hw-pvc-vs-a100-model, ker-ggemm-production]
---
# Xe core / XVE / XMX (Data Center GPU Max 1550, one tile in FLAT mode)

What the harness runs on: ONE PVC tile (`ZE_AFFINITY_MASK=<tile>`, FLAT hierarchy), 12 tiles per node.

| unit | count | note |
|---|---|---|
| Xe core | 56 per tile here (spec 64) | a work-group lives on exactly one Xe core |
| XVE (vector engine) | 8 per Xe core, 448 per tile (= SYCL `max_compute_units`) | issues one hardware thread per cycle |
| XMX (matrix engine, runs DPAS) | 8 per Xe core, one per XVE | 4096 bf16 FLOP/clock per Xe core |
| register file | 512 KB per Xe core (64 KB per XVE) | see [[hw-pvc-grf-threads]] |
| L1 + SLM | 512 KB per Xe core | SLM up to 128 KB per work-group |

Consequences for the persistent grouped GEMM:
- Grid = Xe cores x 512 / work-group threads: 56 work-groups of 512 work-items for 256x256; 112 for 256-thread tiles; 224 for 128-thread tiles (`default_num_workgroups`).
- A 512-thread work-group in GRF-256 mode fills the Xe core (32 sub-groups = 4 threads x 8 XVEs). Two 256-thread work-groups also fill it but then share the L1 and cannot share operand tiles.
- Clock under load is 1.2-1.3 GHz (unitrace `AvgGpuCoreFrequencyMHz`), well below the 1.6 GHz peak spec, so quote TFLOPS against the practical ceiling (~220-250 TFLOPS, large dense GEMM), not the 420 TFLOPS spec peak.

Performance claims
- [claim] gpu=PVC-1550-tile dtype=bf16 shape=large dense GEMM metric=practical TFLOPS value=220-250 source=notes_ggemm.txt#11.3
- [claim] gpu=PVC-1550-tile dtype=bf16 shape=dsv3 TP4/EP4 metric=fraction of practical ceiling value=dX ~55-60%, fwd ~50%, dW ~40% source=notes_ggemm.txt#11.3
