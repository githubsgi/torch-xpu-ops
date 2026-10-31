---
id: hw-pvc-vs-a100-model
title: PVC vs NVIDIA execution model (vocabulary map and numbers)
type: hardware
tags: [pvc, a100, execution-model, migration, vocabulary]
architectures: [pvc, a100]
confidence: source-reported
sources: [notes_ggemm.txt#18]
related: [mig-cute-cuda-to-sycl-tla, hw-pvc-xe-core, hw-pvc-grf-threads]
---
# Vocabulary and numbers

| concept | PVC (SYCL) | A100 (CUDA) |
|---|---|---|
| block home | Xe core (56/tile here) | SM (108) |
| scheduler sub-unit | XVE, 8 per core | SMSP, 4 per SM |
| matrix unit | XMX, 8 per core | tensor core, 4 per SM |
| SIMD group | sub-group (16 or 32 lanes; DPAS and 2D blocks need 16 here) | warp = 32 |
| block | work-group (<= 1024 work-items) | CTA (<= 1024 threads) |
| resident SIMD groups per core | 32 (256-GRF mode) or 64 (128-GRF) | up to 64 warps |
| register file | 512 KB per Xe core; modes 128/256 regs per thread | 256 KB per SM, any count |
| shared memory | SLM inside the 512 KB L1 | up to 164 KB |
| last-level cache | 204 MB per tile | 40 MB |
| dense bf16 peak | ~420 TFLOPS per tile (220-250 practical) | 312 TFLOPS |
| tiny dependent kernel gap | 0.5 us (end to start), 1.8-1.9 us start to start | ~1-4 us |

Mapping of our tile configs: 256x256 -> 32 sub-groups -> 1 work-group per Xe core -> 56 work-groups per tile; 256x128 / 128x256 -> 16 sub-groups -> 2 per core -> 112; 128x128 -> 8 sub-groups -> 4 per core -> 224.
