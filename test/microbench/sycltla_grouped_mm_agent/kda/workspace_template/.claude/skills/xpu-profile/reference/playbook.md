# Diagnosis playbook - unitrace signal -> cause -> fix (PVC grouped GEMM)

How to use: for each observation read **Signals** (metric thresholds; reference = production kernel), **Why**, **First-line fix** (cheapest change), **Deeper fixes**, **Exceptions** (where the pattern is expected). Most kernels match 2-3 patterns; rank by the share of time they explain and fix the biggest first. Wiki pages carry the detail (`kw get <id>`).

Reference rows (production 256x256, TP4/EP4, one tile): dX XMX 65-70%, stall ~7%, idle 28%; forward XMX ~65%, SBID 14%, idle 33%; grad_weight XMX 34-36%, SBID 31-37%, stalled 30-33%, idle 17%, occupancy 41%; HBM 170-225 GB/s, L3 ~720 GB/s; clock 1.2-1.3 GHz.

## A. Short contraction (prologue/epilogue-bound)  [pat-short-k-prologue-bound]
- **Signals:** K tiles per output tile <= 12 (rows/expert/32); SBID >= 20%; XMX < 45% with XVE active ~55%; HBM/L3 low; op = k_grouped.
- **Why:** with ~6 K steps, the prefetch prologue and the 256x256 bf16 store (128 KB) are a large part of each tile; loads are consumed immediately after issue.
- **First-line fix:** `prefetch` 1-2 for `k_grouped` (configs.json). Then dW tile 128x128 / 256x128 (more, smaller tiles).
- **Deeper fixes:** issue the next tile's prologue prefetches before the current epilogue store; hoist per-tile descriptors; intra-work-group K slicing + SLM reduction (no scratch). Classic split-K needs a workspace: not allowed.
- **Exceptions:** large groups (~6k rows): dW is fine (129-138 TFLOPS).

## B. Load-latency (SBID) with long K  [pat-sbid-load-latency]
- **Signals:** SBID >= 12% of IP samples and K tiles per tile > 12 (forward 14%, dX 7%).
- **Why:** DPAS waits for 2D loads; only 4 hardware threads per XVE in GRF-256 mode. K-contiguous B (forward) uses a less efficient load path.
- **First-line fix:** prefetch distance 4 (or 2); compare per op.
- **Deeper fixes:** register double buffering (check spills first), larger K step, taller load blocks.
- **Exceptions:** dX-like XMX >= 65% / SBID < 10%: leave alone.

## C. Wave quantization / tail  [pat-tail-wave-quantization]
- **Signals:** XVE idle >= 25%; wave efficiency < 85% (script row "tiles, waves, wave eff."); hot/sparse cases worse than uniform.
- **Why:** groups rounded up to whole tiles, static round-robin over 56 work-groups; last wave nearly empty; one heavy expert is a tail.
- **First-line fix:** smaller M tile for the kind (static config), check the ~190-row loss on `small`.
- **Deeper fixes:** device-side finer tiles for the last wave / heavy-first ordering (reads offs only), N tile 128 where N % 256 wastes (w2_dX N=1408).
- **Exceptions:** EP8 (60+ waves) is not tail-limited.

## D. Padding waste  [pat-tile-padding-waste]
- **Signals:** row fill < 78% (script), probes rows8/rows64 far below uniform.
- **Why:** zero-filled rows still run DPAS.
- **First-line fix:** 128-row tile for m_rowB/m_colB. **Deeper:** device-side specialization. **Exceptions:** probes are not scored.

## E. XMX underfed  [pat-xmx-underfed]
- **Signals:** XVE active >= 60% and XMX util < 50%; InstrFetch >= 5%; send util high.
- **Why:** address math, descriptor building, reorders and epilogue instructions compete with DPAS (notes 14.6).
- **First-line fix:** hoist per-tile setup out of the persistent loop; plain convert+store epilogue.
- **Deeper:** bigger sub-group tile within the register budget; fewer, taller 2D messages; unroll limits.

## F. Register spills  [pat-register-spills]
- **Signals:** BUILD line lists spills; `Spill Memory Per Thread` > 0.
- **First-line fix:** revert the register-growing edit. **Exceptions:** none.

## G. Barrier / dependency stalls  [pat-barrier-sync-stalls]
- **Signals:** Sync + Dist >= 15% of IP samples.
- **First-line fix:** 16-sub-group tiles or fewer barriers. **Exceptions:** production shows no large share today.

## H. Low occupancy / grid  [pat-low-occupancy]
- **Signals:** occupancy < 35% AND idle high; work-groups < Xe cores.
- **Fix:** more, smaller work-groups; `wgs`. **Exception:** occupancy alone is not a goal (dX at 36% is fine).

## I. Instruction fetch  [pat-instruction-fetch]
- **Signals:** InstrFetch >= 5%. **Fix:** less unrolling, one shared mainloop. **Note:** grows when merging specializations.

## J. Memory-bound (expected NOT to apply)  [pat-memory-bound-expected]
- **Signals:** HBM > 800 GB/s or L3 > 2 TB/s. If absent, do not propose bandwidth fixes.

## K. Clock / measurement validity
- **Signals:** clock < 1100 MHz, GPU kernel us varies > 10% across the 20 instances. Rerun on another tile.

## L. Per-call floor  [pat-launch-floor-tiny-calls]
- **Signals:** rows8 probe time ~ independent of size; time grows with number of groups. Fix: read offs once per work-group.
