---
id: hw-unitrace-metrics
title: Reading the unitrace digest (metrics, stall classes, reference values)
type: hardware
tags: [unitrace, profiling, metrics, stalls, xve, xmx]
architectures: [pvc]
confidence: source-reported
sources: [ggemm_agent/profile.py, notes_ggemm.txt#11.2, notes_ggemm.txt#11.4, notes_ggemm.txt#14.6]
related: [pat-short-k-prologue-bound, pat-xmx-underfed, pat-tail-wave-quantization, pat-barrier-sync-stalls]
---
# Metrics printed by `ggemm-eval profile`

| field | unitrace source | meaning |
|---|---|---|
| XVE active / stall / idle % | `XVE_ACTIVE`, `XVE_STALL`, 100-sum | idle = no resident thread to issue (tail, occupancy); stall = threads waiting |
| occupancy % | `XVE_THREADS_OCCUPANCY_ALL` | resident hardware threads vs max |
| XMX util % | `XVE_INST_EXECUTED_XMX_ALL_UTILIZATION` | DPAS pipe busy; production dX 65-70, fwd ~65, dW 34-36 |
| send util % | `XVE_INST_EXECUTED_SEND_ALL_UTILIZATION` | memory messages issue rate |
| HBM / L3 GB/s | `GPU_MEMORY_BYTE_*`, `L3_BYTE_READ` | far below ceilings for all ops |
| MHz | `AvgGpuCoreFrequencyMHz` | 1.2-1.3 GHz under load; a drop means throttling |
| barrier msgs | `XVE_BARRIER_MESSAGE_COUNT` | work-group barrier traffic |
| stall samples | `--stall-sampling` IP classes | Sbid (load/scoreboard), Pipe, Dist, Sync, InstrFetch, Other |
| kernel properties | driver table | SIMD, SLM, private/spill memory per thread, GRF count |

Reference points at TP4/EP4, production: dX XMX 70% stall 7%; fwd XMX 65% SBID 14%; dW XMX 36% SBID 31%, 33% stalled; HBM never the limit.
Stall percentages are shares of all sampled instruction pointers (`Active` = issuing, the rest are stall classes), matching the notes figure "SBID 37%". Counters come from one `ComputeBasic` pass (instance medians over 20 launches) and one stall-sampling pass: they do not run in the same pass, so do not expect them to sum to a consistent timeline.
