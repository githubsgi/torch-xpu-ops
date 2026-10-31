---
id: tech-compiler-flags
title: Compiler and driver flags through configs.json extra_flags
type: technique
tags: [compiler, flags, icpx, ocloc, register-allocation, unroll]
architectures: [pvc]
confidence: experimental
sources: [ggemm_agent/candidate.py, kernel:grouped_mm_kernel_impl.h:launch, ggemm_agent/evaluate.py]
related: [tech-grf-mode-and-occupancy, pat-instruction-fetch, pat-register-spills]
---
# `extra_flags`

`configs.json: "extra_flags": ["..."]` is appended to the icpx command for the candidate .so (AOT for PVC). Examples to try one at a time (each is part of the build hash, so cached separately):
- `-Xs "-options -ze-opt-large-register-file"` style driver options (name varies by driver; check the build log for rejection),
- `-ftarget-register-alloc-mode=pvc:large` (icpx register mode; the kernel already requests `grf_size<256>`),
- `-mllvm -unroll-threshold=...`/`#pragma unroll N` in the header for loop unrolling control.
Treat every flag as an experiment: a build error is reported as `STATUS: build_error`, a silent perf change shows in the per-op table. Keep only flags that win on `small` and `large`, and note them in candidates.jsonl.
