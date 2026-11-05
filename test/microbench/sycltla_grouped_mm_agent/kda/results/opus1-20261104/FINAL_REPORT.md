# KDA / RLCR run report

- run directory: `/mnt/aimod/subratag/ggemm_agent_runs/kda/opus1-20261104`
- loop: `2026-11-04_15-55-56`, phase 2, status **stopped_review** (reviewer verdict STOP at round 9)
- rounds executed: 10 (max 10)
- LLM spend: $17.19 of $60.00 (draft $0.83, gen-plan $0.29, reviewer $2.65, reviewer-audit $0.80, worker $12.61)
- best promoted candidate: `r6`, full-suite SCORE 1.0514 (promotion needs >= 1.02 and no op < 0.97)

## Rounds

| round | driver eval | reviewer verdict | mainline progress |
|---|---|---|---|
| 0 | score 1.0126 | CONTINUE | ADVANCED |
| 1 | score 1.0136 | CONTINUE | ADVANCED |
| 2 | score 1.0052 | CONTINUE | ADVANCED |
| 3 | score 1.0093 | CONTINUE | ADVANCED |
| 4 | score 1.0153 | CONTINUE | ADVANCED |
| 5 | score 1.0287 | CONTINUE | ADVANCED |
| 6 | score 1.0211 | CONTINUE | ADVANCED |
| 7 | unchanged: cand/ is identical to best/ (no evaluation run) | CONTINUE | ADVANCED |
| 8 | score 1.0332 | CONTINUE | ADVANCED |
| 9 | score 1.0270 | STOP | ADVANCED |

## benchmark.csv (42 rows)

| round | source | candidate | suite | status | score | note |
|---|---|---|---|---|---|---|
|  | worker | production | quick | ok | 1.0027 | quick: cand==production, SCORE 1.0027 (noise); tp4ep4 dW 77-78 TF, fwd/dX 106-12 |
| 0 | worker | c1 | quick | ok | 1.0672 | quick dW ops: SCORE 1.067; w13_dW 1.033 (tp4 1.051 IQR .085, ep8 1.015), w2_dW 1 |
| 0 | worker | c2 | quick | ok | 1.0152 | quick dW ops: SCORE 1.015 (NEUTRAL); w13_dW 1.012 (tp4 1.017 IQR .086, ep8 1.008 |
| 0 | worker | c3 | quick | ok | 0.9360 | quick dW ops: SCORE 0.936 REGRESSED; w13_dW 0.941, w2_dW 0.925; BUILD reports sp |
| 0 | worker | c1 | small | ok | 1.0196 | small: SCORE 1.0196; w13_dW 1.031, w2_dW 1.026, fwd/dX 0.999-1.001; cases 1.012- |
| 0 | worker | c1 | large | ok | 1.0089 | large: SCORE 1.0089; w13_dW 1.029, w2_dW 1.038, fwd/dX 0.999-1.000; PASS no spil |
| 0 | worker | c4 | quick | ok | 1.0016 | quick fwd/dX ops: SCORE 1.0016 NEUTRAL; w13_fwd 1.015, w2_fwd 1.004, w13_dX 1.00 |
| 0 | worker | c5 | quick | ok | 0.9875 | quick fwd/dX ops: SCORE 0.9875 NEUTRAL/slightly worse; w13_fwd 0.990, w2_fwd 0.9 |
| 0 | worker | c1 | full | ok | 1.0136 | full: SCORE 1.0136 (below 1.02 promotion bar); w13_dW 1.033, w2_dW 1.034, fwd/dX |
| 1 | worker | c6 | quick | ok | 0.9145 | quick dW ops: SCORE 0.9145 REGRESSED; w13_dW 0.898 (tp4 0.943, ep8 0.856), w2_dW |
| 1 | worker | c7 | quick | ok | 0.9133 | quick dW ops: SCORE 0.9133 REGRESSED; w13_dW 0.912 (tp4 0.970, ep8 0.858), w2_dW |
| 1 | worker | c8 | quick | ok | 0.8630 | quick dW ops: SCORE 0.8630 REGRESSED; w13_dW 0.861 (tp4 0.974 IQR .156, ep8 0.76 |
| 1 | worker | c9 | quick | ok | 0.9755 | quick dW ops: SCORE 0.9755 REGRESSED; w13_dW 0.976 (tp4 1.028, ep8 0.926), w2_dW |
| 2 | worker | c10 | quick | ok | 0.8961 | quick dX ops: SCORE 0.896 REGRESSED; w13_dX 0.898 (tp4 0.939, ep8 0.859), w2_dX  |
| 2 | worker | c12 | quick | ok | 0.9756 | quick dW: SCORE 0.976 REGRESSED; w13_dW 0.965 (tp4 0.980 IQR .062, ep8 0.950), w |
| 2 | worker | c14 | quick | ok | 1.0513 | quick dW: SCORE 1.051 (TP4 base swing); w13_dW 1.009 (tp4 1.019, ep8 0.999), w2_ |
| 2 | worker | c15 | quick | ok | 1.0207 | quick dW: SCORE 1.021 (TP4 base swing); w13_dW 1.010 (tp4 1.007, ep8 1.012), w2_ |
| 2 | worker | c16 | quick | ok | 1.0080 | quick dW: SCORE 1.008 NEUTRAL; w13_dW 0.985 (tp4 1.025, ep8 0.948), w2_dW 0.970  |
| 3 | worker | c17 | quick | ok | 1.0229 | quick SCORE 1.0229; dW per-op 1.036/1.032 (c1 1.018/1.024); tp4 w13_dW 1.061 w2_ |
| 3 | worker | c17 | full | ok | 1.0125 | full SCORE 1.0125 PASS no spills; dW 1.033/1.037, fwd/dX 0.999-1.000; vs c1 full |
| 3 | worker | c1 | full | ok | 1.0107 | full 1.0107 PASS no spills; dW 1.014/1.039, fwd/dX 0.998-1.001 |
| 3 | worker | c17 | full | ok | 1.0090 | full repeat 1.0090 PASS; dW 1.029/1.031; mean of 2 full runs 1.011 vs c1 same-se |
| 4 | worker | c18 | quick | ok | 0.9937 | quick SCORE 0.9937 NEUTRAL/worse vs c1 quick 1.009; per-op fwd/dX 0.990-0.997, d |
| 4 | worker | c19 | quick | ok | 1.0146 | quick SCORE 1.0146 PASS no spills; dW 1.025/1.038 (tp4 1.038/1.058, ep8 1.011/1. |
| 4 | worker | c19 | full | ok | 1.0130 | full SCORE 1.0130 PASS no spills; dW 1.025/1.038, fwd/dX 0.999-1.003; c1 full ra |
| 4 | worker | c1 | full | ok | 1.0121 | full 1.0121 PASS no spills; dW 1.032/1.033, fwd/dX 0.999-1.001 -> c19 (1.0130, d |
| 5 | worker | c22 | quick | ok | 1.0282 | quick SCORE 1.0282; tp4 uniform 1.051 (w13_fwd 1.052, w2_fwd 1.068, w13_dX 1.091 |
| 5 | worker | c22 | full | ok | 1.0411 | full SCORE 1.0411, PASS, spills none; PER-OP w13_fwd 1.009 w2_fwd 1.071 w13_dX 1 |
| 5 | worker | c1 | full | ok | 1.0110 | full SCORE 1.0110; PER-OP w13_fwd 0.997 w2_fwd 1.000 w13_dX 1.000 w2_dX 0.999 w1 |
| 5 | worker | c23 | full | ok | 0.9436 | full SCORE 0.9436; fwd/dX per-op 0.862-0.925, dW 1.034 unchanged: oversubscribin |
| 5 | worker | c24 | full | ok | 1.0201 | full SCORE 1.0201 vs c22 1.0411; tp4 w2_dX 0.813/0.827, tp8 w13_fwd 0.816, w13_f |
| 5 | worker | c22 | full | ok | 1.0415 | full repeat SCORE 1.0415 (first 1.0411); PER-OP w13_fwd 1.011 w2_fwd 1.069 w13_d |
| 6 | worker | c25 | full | ok | 1.0494 | full SCORE 1.0494 PASS spills none; per-op w13_fwd 1.012 w2_fwd 1.090 w13_dX 1.1 |
| 6 | worker | c22 | full | ok | 1.0437 | full SCORE 1.0437 PASS; per-op w13_fwd 1.013 w2_fwd 1.072 w13_dX 1.109 w2_dX 1.0 |
| 6 | worker | c25 | full | ok | 1.0481 | full repeat SCORE 1.0481 PASS; per-op w13_fwd 1.018 w2_fwd 1.087 w13_dX 1.121 w2 |
| 6 | worker | c26 | full | ok | 1.0424 | full SCORE 1.0424 PASS spills none; cases tp4 1.039/1.045/1.038/1.064 tp8 1.077/ |
| 6 | worker | c25 | full | ok | 1.0450 | full SCORE 1.0450 PASS; cases tp4 1.033/1.034/1.051/1.046 tp8 1.089/1.087/1.069/ |
| 7 | worker | c28 | full | ok | 1.0481 | full run 1 (see SCORE line); PER-OP w13_fwd 1.018 w2_fwd 1.087 w13_dX 1.116 w2_d |
| 7 | worker | c25 | full | ok | 1.0487 | c25 control, full; compare with c28 1.0481 |
| 7 | worker | c25 | full | ok | 1.0444 | full 1.0444, per-op min w13_dW 1.012; spills none; PASS |
| 8 | worker | c29 | full | ok | 1.0470 | full 1.0470 (c25 range 1.0444-1.0514); per-op 1.017/1.088/1.121/1.049/1.017/1.03 |
| 9 | worker | c29 | full | ok | 1.0468 | full 1.0468 PASS spills none; within c25/c29 range 1.0444-1.0514; c30 not benchm |

## Candidate DAG (candidates.jsonl)

- `c1` <- production: **kept** D1 k_grouped prefetch 2 | dW per-op small 1.031/1.026, large 1.029/1.038; SCORE small 1.0196 large 1.0089; pf1 (c2) quick 1.015, pf4 (c3) spills and 0.936; per tech-p
- `c2` <- production: **rejected** D1 k_grouped prefetch 1 | quick dW per-op 1.012/1.043, SCORE 1.015 vs pf2 1.067; ep8 dW only 1.005-1.008
- `c3` <- production: **rejected** D1 k_grouped prefetch 4 | BUILD spills ~1; quick dW 0.941/0.925, SCORE 0.936
- `c4` <- c1: **rejected** D1 m_rowB/m_colB prefetch 2 | quick fwd/dX SCORE 1.0016, per-op 1.003-1.015: NEUTRAL (<2%), long K (44-88 steps) hides latency at pf3
- `c5` <- c1: **rejected** D1 m_rowB/m_colB prefetch 4 | quick fwd/dX SCORE 0.9875, per-op 0.990-0.997, tp4 case 0.979: NEUTRAL-to-worse
- `c6` <- c1: **rejected** D2 k_grouped tile 256x128x32 sg8x2 | quick: w13_dW 0.898, w2_dW 0.963 vs production (c1: 1.033/1.031); EP8 0.856/0.909; new spills ~1. Halving N tile halves A reuse; per tech-ti
- `c7` <- c1: **rejected** D2 k_grouped tile 128x256x32 sg4x4 | quick: w13_dW 0.912, w2_dW 0.952 vs production (c1 1.033/1.031); EP8 0.858/0.879; new spills ~2
- `c8` <- c1: **rejected** D2 k_grouped tile 128x128x32 sg4x2 | quick: w13_dW 0.861, w2_dW 0.913 vs production (c1 1.033/1.031); EP8 0.760/0.810 (106 TF vs 140); TP4 no better than c1; no spills
- `c9` <- c1: **rejected** D2 k_grouped sg layout 4x8 (256x256x32) | quick: w13_dW 0.976, w2_dW 0.949 vs production (c1 1.033/1.031); EP8 0.926/0.933; no spills; 64x32 SG tile worse than 32x64
- `c10` <- c1: **rejected** D2 m_colB 256x128x32 sg8x2 | quick dX SCORE 0.896: w2_dX 0.903 (tp4 0.911 ep8 0.896), w13_dX 0.898; below the 1.02 gate, D2 closed (5/5 iterations); c1 m_colB 256x256 sg
- `c11` <- c1: **rejected** D9 k_grouped tile K64 256x256x64 sg8x4 pf2 | BUILD register spills: spilled around 191 (c1 none); rejected on the spill line without benchmarking per AC-1; D9 closed (1 iteration)
- `c12` <- c1: **rejected** D3 cross-tile head prefetch v1 (before D store) | quick dW 0.965/0.963 (ep8 0.950/0.943, tp4 0.980/0.984); TP4 profile at 1199 MHz SBID 31->26%, kernel 375->364us, XMX 39->39%; EP8 profile X
- `c13` <- c12: **rejected** D3 v2: next-tile locate inside hook | BUILD spills ~3 (c1 none); quick dW 0.973/0.965 -> reject
- `c14` <- c12: **rejected** D3 control: hook compiled out | quick dW 1.009/1.008 (=c1 within noise): diagnostic only, no gain; shows the loop restructure is free
- `c15` <- c12: **rejected** D3 v4: hook after D store | BUILD spills ~5 (c1 none); quick dW 1.010/1.012 (noise) -> reject
- `c16` <- c12: **rejected** D3 cross-tile prefetch v5 (device-gated) | quick dW 0.985/0.970, EP8 0.948/0.937 although the hook is runtime-skipped at EP8 -> static code-gen cost; D3 closed after 5 iterations
- `c17` <- c1: **rejected** D4 hoisting: copy/prefetch atoms + identity tensors built on | NEUTRAL: quick 1.0229 (dW 1.036/1.032, ep8 1.011/1.027), full 1.0125 (dW 1.033/1.037) vs c1 full 1.009-1.014; profile r3_d4_v1_tp4 vs r2_c1_
- `c18` <- c1: **rejected** D6 v1: offs cached in sub-group registers + select_from_grou | quick 0.9937 vs c1 quick 1.009 (same day); tp4 w13_dW 0.990 w2_dW 0.973 vs c1 ~1.04/1.03; ep8 dW 1.007/1.010; PASS, spills none. Shuffle loo
- `c19` <- c1: **rejected** D6 v2: software-pipelined offs walk (offs[g+1] loaded at des | NEUTRAL: full 1.0130 (dW 1.025/1.038) vs c1 same-session full 1.0121 (dW 1.032/1.033); quick 1.0146; PASS no spills; profile r4_d6_v2_tp4 37
- `c20` <- c1: **rejected** D9 M-grouped K64: m_rowB 256x256x64 sg8x4 pf3 (verify-only) | verify-only PASS but BUILD spills ~108 (c1 none) -> rejected on the AC-1 spill line, not benchmarked
- `c21` <- c1: **rejected** D9 M-grouped K64: m_colB 256x256x64 sg8x4 pf3 (verify-only) | verify-only PASS but BUILD spills ~95 (c1 none) -> rejected on the AC-1 spill line, not benchmarked; D9 closed for all kinds (k_grouped c11 
- `c22` <- c1: **proposed** D5(b) tail-wave N-half split in MGroupedKernel | quick 1.0282 vs c1 ~1.015; tp4 w13_fwd 1.052 w13_dX 1.091; w2_dX 0.972 to check
- `c23` <- c22: **rejected** D5(d) wgs=112 control (M-grouped) | full 0.9436 vs c22 1.0411; w2_dX 0.862, w13_fwd 0.894 -> rejected; wgs=0 (56) kept
- `c24` <- c22: **rejected** D5(b) quarter split 256x64 tail | full 1.0201 vs c22 1.0411; w13_fwd per-op 0.966, tp4 w2_dX 0.813; rejected
- `c22` <- c1: **kept** D5(b) tail-wave N-half split in MGroupedKernel | full 1.0411/1.0415 vs c1 same-session 1.0110; min per-op w13_fwd 1.009; spills none; profile r5_c22_tp4_w13_fwd idle 20->12%, XMX 68->73%
- `r5` <- production: **promoted** driver promotion after full-suite check | full SCORE 1.0395, per-op {'w13_fwd': 1.009849654631004, 'w2_fwd': 1.0700893797751132, 'w13_dX': 1.107235596027608, 'w2_dX': 1.0335831267418
- `c25` <- c22: **kept** D5(b) follow-up: sub-group-parallel offs walk for tail-split | full 1.0494/1.0481 vs c22 same-session 1.0437; w2_dX 1.050/1.051 vs 1.035; profile r6_c25 vs r6_c22 tp4 w2_dX GPU 112.1 vs 115.8 us (c1 112.
- `c26` <- c25: **rejected** D7 masked-half 128x256 sg4x8 for last M tiles <=128 rows | NEUTRAL: full 1.0424 vs c25 same-session 1.0450 (-0.25%); tp8 uniform/balanced 1.077/1.074 vs 1.089/1.087, ep8 1.003 vs 1.007; gains only on
- `r6` <- production: **promoted** driver promotion after full-suite check | full SCORE 1.0514, per-op {'w13_fwd': 1.018093706535237, 'w2_fwd': 1.0877508368313609, 'w13_dX': 1.121848007559296, 'w2_dX': 1.0500988597259
- `c27` <- c25: **rejected** D8 intra-WG K slicing (skipped on paper, not built) | Not feasible. (1) 256x256 split 2 ways needs a 64x64 fp32 accumulator per SG = 16 KB = the whole GRF (hw-pvc-grf-threads), or 64 SGs > 32 ma
- `c28` <- c25: **rejected** D10 extra_flags -ftarget-register-alloc-mode=pvc:large | NEUTRAL: full 1.0481 (per-op 1.018/1.087/1.116/1.051/1.020/1.035) vs same-session c25 control 1.0487 (1.018/1.087/1.119/1.052/1.026/1.037), 
- `c29` <- c25: **kept** T21 (AC-6 robustness on D5 c25): offs-walk lane stride from  | NEUTRAL robustness fix: full 1.0470 (c25 6-run mean 1.0478, range 1.0444-1.0514); per-op 1.017/1.088/1.121/1.049/1.017/1.035, min 1.017; spi
- `c30` <- c29: **rejected** D3 re-try (T13b): next-tile A-only head prefetch dist 1 afte | verify-only PASS but BUILD spills ~2 (c29 none) -> rejected on spill line (AC-1), not benchmarked; c15 same placement spilled ~5

## Final evaluation

```
CANDIDATE: /mnt/aimod/subratag/ggemm_agent_runs/kda/opus1-20261104/cand
STATUS: ok
JOB: /home/subratag/aimod/ggemm_agent_runs/jobs/20261104-174053-685ede-kda-final
BUILD: ok in 0.0 s; register spills: none
CORRECTNESS: PASS (22 hidden verification cases, full output vs fp32 reference, NaN sentinels around the output)
CONFIG: m_rowB=256x256x32 sg8x4 pf3 wgs=56; m_colB=256x256x32 sg8x4 pf3 wgs=56; k_grouped=256x256x32 sg8x4 pf2 wgs=56
SCORE: 1.0496x  (geomean over the routed cases of production/candidate time of the six GEMMs, suite full, on x1002c6s0b0n0.0; rowsN probe cases are not scored)
case tp4ep4_uniform: six GEMMs base 1.6568 ms, cand 1.5683 ms -> 1.056x
   w13_fwd  base 0.3320 ms (106 TF)  cand 0.3002 ms (117 TF)  1.066x  (noise: IQR of paired ratios 0.095)
   w2_fwd   base 0.1686 ms (105 TF)  cand 0.1547 ms (114 TF)  1.089x  (noise: IQR of paired ratios 0.008)
   w13_dX   base 0.3127 ms (113 TF)  cand 0.2832 ms (125 TF)  1.085x  (noise: IQR of paired ratios 0.028)
   w2_dX    base 0.1566 ms (113 TF)  cand 0.1582 ms (111 TF)  0.999x  (noise: IQR of paired ratios 0.033)
   w13_dW   base 0.4603 ms (77 TF)  cand 0.4516 ms (78 TF)  1.018x  (noise: IQR of paired ratios 0.016)
   w2_dW    base 0.2266 ms (78 TF)  cand 0.2205 ms (80 TF)  1.035x  (noise: IQR of paired ratios 0.035)
case tp4ep4_balanced: six GEMMs base 1.6342 ms, cand 1.5666 ms -> 1.043x
   w13_fwd  base 0.3353 ms (106 TF)  cand 0.3239 ms (109 TF)  1.021x  (noise: IQR of paired ratios 0.021)
   w2_fwd   base 0.1686 ms (105 TF)  cand 0.1554 ms (114 TF)  1.087x  (noise: IQR of paired ratios 0.014)
   w13_dX   base 0.3104 ms (114 TF)  cand 0.2825 ms (125 TF)  1.099x  (noise: IQR of paired ratios 0.017)
   w2_dX    base 0.1551 ms (114 TF)  cand 0.1564 ms (113 TF)  0.991x  (noise: IQR of paired ratios 0.001)
   w13_dW   base 0.4440 ms (80 TF)  cand 0.4349 ms (81 TF)  1.021x  (noise: IQR of paired ratios 0.041)
   w2_dW    base 0.2208 ms (80 TF)  cand 0.2134 ms (83 TF)  1.035x  (noise: IQR of paired ratios 0.027)
case tp4ep4_hot: six GEMMs base 2.2366 ms, cand 2.1306 ms -> 1.050x
   w13_fwd  base 0.4547 ms (114 TF)  cand 0.4558 ms (114 TF)  0.990x  (noise: IQR of paired ratios 0.017)
   w2_fwd   base 0.2320 ms (112 TF)  cand 0.2242 ms (115 TF)  1.067x  (noise: IQR of paired ratios 0.019)
   w13_dX   base 0.4418 ms (117 TF)  cand 0.4122 ms (126 TF)  1.060x  (noise: IQR of paired ratios 0.031)
   w2_dX    base 0.2306 ms (112 TF)  cand 0.1992 ms (130 TF)  1.171x  (noise: IQR of paired ratios 0.026)
   w13_dW   base 0.5876 ms (88 TF)  cand 0.5705 ms (91 TF)  1.044x  (noise: IQR of paired ratios 0.052)
   w2_dW    base 0.2899 ms (89 TF)  cand 0.2687 ms (96 TF)  1.085x  (noise: IQR of paired ratios 0.016)
case tp4ep4_sparse: six GEMMs base 0.9982 ms, cand 0.9174 ms -> 1.088x
   w13_fwd  base 0.1588 ms (109 TF)  cand 0.1601 ms (108 TF)  0.994x  (noise: IQR of paired ratios 0.013)
   w2_fwd   base 0.1018 ms (85 TF)  cand 0.0844 ms (103 TF)  1.196x  (noise: IQR of paired ratios 0.021)
   w13_dX   
```

## Skill/tool activity (hook log)

Skill calls: ['-:kernel-wiki', '-:xpu-profile', '0:kernel-wiki', '0:xpu-profile', '1:kernel-wiki', '1:xpu-profile', '2:kernel-wiki', '2:xpu-profile', '3:kernel-wiki', '3:xpu-profile', '4:kernel-wiki', '4:xpu-profile', '5:kernel-wiki', '5:xpu-profile', '6:kernel-wiki', '6:xpu-profile', '7:kernel-wiki', '7:xpu-profile', '8:kernel-wiki', '8:xpu-profile', '9:kernel-wiki', '9:xpu-profile']
tool uses: 516 (Bash 267, Edit 162, Read 39, Skill 22, Write 26)
