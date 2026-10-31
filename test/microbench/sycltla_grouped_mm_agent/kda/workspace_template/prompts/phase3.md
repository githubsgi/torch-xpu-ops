# Phase 3 - specialize by workload regime

Develop a faster SYCL-TLA grouped GEMM for Intel PVC while preserving numerical correctness. Target speedup: **{{TARGET_SPEEDUP}}x** SCORE on the `full` suite (no op regressing > 3%, no new spills).

{{COMMON}}

## Phase 3 goal
Analyze the full workload distribution and specialize the kernel configuration for the regimes in the evaluator: small groups (TP4/EP4, ~190 rows/expert: uniform, balanced, hot, sparse; probes rows64, rows8) and large groups (EP8, ~6k rows/expert: uniform, balanced, hot; probe rows1024), and for the three kernel kinds (`m_rowB` forward, `m_colB` grad_input, `k_grouped` grad_weight), which have independent configs. Group-size dependent behaviour must come from device-side logic that reads `offs` (identical kernel on every rank), never from host-known row counts.

Design specialized configs/paths only where the measured win justifies the added complexity (code size, registers, instruction fetch). Start from the per-kind table of measurements gathered in phase 2 (`benchmark.csv`, `candidates.jsonl`): for each (kind, regime) record the best known config and its margin over production; specialise a kind only when the gain is >= 2% on the scored cases of that regime and does not cost > 3% on the other regime. The final candidate must be evaluated on the **full** suite (all ten cases), not only on the quick cases.
