/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

#include "grouped_mm_kernels.h"

#include <algorithm>

namespace sycltla {
namespace grouped_gemm {

namespace {
bool is_pvc(const sycl::device& dev) {
  namespace ex = sycl::ext::oneapi::experimental;
  try {
    const auto arch = dev.get_info<ex::info::device::architecture>();
    return arch == ex::architecture::intel_gpu_pvc ||
        arch == ex::architecture::intel_gpu_pvc_vg;
  } catch (...) {
    return false;
  }
}

int xe_core_count(const sycl::device& dev) {
  int n = 0;
  try {
    n = int(dev.get_info<sycl::ext::intel::info::device::gpu_slices>()) *
        int(dev.get_info<
            sycl::ext::intel::info::device::gpu_subslices_per_slice>());
  } catch (...) {
    n = 0;
  }
  if (n <= 0) {
    // 8 EUs per Xe core on PVC/BMG.
    n = std::max(
        1, int(dev.get_info<sycl::info::device::max_compute_units>()) / 8);
  }
  return n;
}
} // namespace

int default_tile(const sycl::device& dev) {
  return is_pvc(dev) ? 2 : 1;
}

int default_num_workgroups(const sycl::device& dev) {
  // One persistent work-group per Xe core. A 256-thread work-group fills half
  // of a core's thread slots in GRF-256 mode, so launch two per core for the
  // 16-sub-group tile; the 32-sub-group tile fills a core on its own.
  const int cores = xe_core_count(dev);
  return default_tile(dev) == 2 ? cores : 2 * cores;
}

} // namespace grouped_gemm
} // namespace sycltla
