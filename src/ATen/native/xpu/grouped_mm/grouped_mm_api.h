/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#pragma once

#include <ATen/core/Tensor.h>

namespace sycltla {

// True when torch-xpu-ops was built with SYCL-TLA (USE_SYCLTLA).
bool is_grouped_mm_available();

// Whether the SYCL-TLA grouped GEMM can serve this at::_grouped_mm call.
// Supported today (bf16 / fp16, same dtype for all operands, int32 offs):
//   * mat_a [M, K] row-major, mat_b [E, K, N] (K- or N-contiguous per group),
//     out [M, N] row-major  -> groups partition rows of mat_a / out.
//   * mat_a [K, M] M-contiguous (a transposed row-major activation), mat_b
//     [M, N] row-major, out [E, K, N] -> groups partition the contraction dim.
// Leading dimensions must be multiples of 64 bytes and base pointers 64-byte
// aligned (Xe 2D block message requirements). Anything else returns false so
// the caller can fall back.
bool can_use_grouped_mm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    const at::Tensor& out);

// Runs the grouped GEMM on the current XPU stream. Requires
// can_use_grouped_mm() to hold; `out` is fully overwritten.
void grouped_mm(
    const at::Tensor& mat_a,
    const at::Tensor& mat_b,
    const at::Tensor& offs,
    at::Tensor& out);

} // namespace sycltla
