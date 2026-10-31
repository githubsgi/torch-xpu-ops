/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// One candidate config of an agent-edited kernel header, built as a shared
// object for agent_eval (same entry point as the tuner's libtune_cfg_*.so).
//
//   -I<candidate dir>    first, so its grouped_mm_kernel_impl.h wins
//   -DCAND_CFG_HEADER="cfg_<hash>.h"   generated: struct tune::CAND_CFG_NAME
//   -DCAND_CFG_NAME=cand_<hash>        unique per (header, config): SYCL
//                                      kernel names never collide across .so
//   -DCAND_MROW=1 / -DCAND_MCOL=1 / -DCAND_K=1   kernels to instantiate
//
// Contract with the candidate header (same as the production one):
//   sycltla::grouped_gemm::run_m_grouped_cfg<Cfg, DType, char LayoutB>(q, p, wgs)
//   sycltla::grouped_gemm::run_k_grouped_cfg<Cfg, DType>(q, p, wgs)
//   sycltla::grouped_gemm::TiledMmaFor<T, Cfg>  (work-group size = its size),
//   unless Cfg defines `static constexpr int kThreads`.

#include "grouped_mm_kernel_impl.h"
#include "tune_registry.h"

#include CAND_CFG_HEADER

#define CAND_STR2(x) #x
#define CAND_STR(x) CAND_STR2(x)

namespace {
using namespace sycltla::grouped_gemm;
using Cfg = tune::CAND_CFG_NAME;
using T = cute::bfloat16_t;

template <class C>
constexpr int threads() {
  if constexpr (requires { C::kThreads; }) {
    return C::kThreads;
  } else {
    return int(cute::size(TiledMmaFor<T, C>{}));
  }
}

tune::Candidate make() {
  tune::Candidate c;
  c.name = CAND_STR(CAND_CFG_NAME);
  c.desc = Cfg::kDesc;
  c.threads = threads<Cfg>();
#if CAND_MROW
  c.m_rowB = [](sycl::queue& q, const Params& p, int w) {
    run_m_grouped_cfg<Cfg, DType::BF16, 'R'>(q, p, w);
  };
#endif
#if CAND_MCOL
  c.m_colB = [](sycl::queue& q, const Params& p, int w) {
    run_m_grouped_cfg<Cfg, DType::BF16, 'C'>(q, p, w);
  };
#endif
#if CAND_K
  c.k_grouped = [](sycl::queue& q, const Params& p, int w) {
    run_k_grouped_cfg<Cfg, DType::BF16>(q, p, w);
  };
#endif
  return c;
}

const tune::Candidate kCandidate = make();
} // namespace

extern "C" const tune::Candidate* tune_grouped_mm_candidate() {
  return &kCandidate;
}
