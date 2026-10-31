/*
 * Copyright 2020-2026 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 */

// Evaluator for agent-generated SYCL-TLA grouped GEMM candidates (no PyTorch).
//
// Two "arms" are loaded from libtune_cfg-style shared objects: the baseline
// (the production kernel at its production config) and a candidate. Each arm
// has one launcher per kernel kind (M-grouped with K-contiguous B, M-grouped
// with N-contiguous B, K-grouped), possibly from different .so files. Then:
//
//   verify  every kind on hidden, seeded cases (ragged, empty and all-empty
//           groups, odd row counts, non-DSv3 shapes) against a naive fp32
//           device reference, full output, with NaN sentinels around the
//           output so out-of-bounds or missing writes are caught.
//   bench   the six DSv3 16B MoE GEMMs over a suite of routed cases. Baseline
//           and candidate are timed alternately (A/B, then B/A, ...) in the
//           same process on the same tile; the result is the median of the
//           rounds. Every timed candidate output is compared in full against
//           the baseline output of the same call.
//   profile run one (case, op) of the candidate in a loop, for unitrace.
//
// Results go to a JSON file (--json) and a short text summary on stdout.

#include <sycl/sycl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "tune_registry.h"

using bf16 = sycl::ext::oneapi::bfloat16;
using sycltla::grouped_gemm::Params;

namespace {

// ---------------------------------------------------------------- model ---
constexpr int kDim = 2048;       // DeepSeek-V3 16B hidden size
constexpr int kMoeHidden = 1408; // expert FFN width (w13 fuses 2 x 1408)
constexpr int kExperts = 64;
constexpr int kTopK = 6;
constexpr int kGuard = 256;      // sentinel elements after every output
constexpr uint16_t kSentinel = 0x7fc1; // a quiet NaN bit pattern

enum Kind { kMRow = 0, kMCol = 1, kK = 2 };
const char* kKindName[3] = {"m_rowB", "m_colB", "k_grouped"};

// One grouped GEMM problem. M-grouped: groups split the rows of A / D,
// A [R, K] row-major, B_g (n, k) at n * K + k ('R', weight [E, N, K]) or
// k * N + n ('C', weight [E, K, N]), D [R, N]. K-grouped: groups split the
// contraction, A (m, r) at m + r * M (activation [R, M]), B [R, N], D [E, M, N].
struct Problem {
  std::string name;
  Kind kind;
  int k_or_m;  // M-grouped: K. K-grouped: the fixed output M.
  int n;
};

const std::vector<Problem>& dsv3_ops() {
  static const std::vector<Problem> d = {
      {"w13_fwd", kMRow, kDim, 2 * kMoeHidden},
      {"w2_fwd", kMRow, kMoeHidden, kDim},
      {"w13_dX", kMCol, 2 * kMoeHidden, kDim},
      {"w2_dX", kMCol, kDim, kMoeHidden},
      {"w13_dW", kK, 2 * kMoeHidden, kDim},
      {"w2_dW", kK, kDim, kMoeHidden},
  };
  return d;
}

// -------------------------------------------------------------- routing ---
// Per-local-expert row counts for EP rank 0 of a top-k router (as the tuner).
std::vector<int> topk_counts(int g_tokens, int e_local, std::mt19937& rng,
                             const std::vector<float>& bias, int force_expert) {
  std::uniform_real_distribution<float> u(0.f, 1.f);
  std::vector<int> counts(kExperts, 0);
  std::vector<std::pair<float, int>> s(kExperts);
  for (int t = 0; t < g_tokens; ++t) {
    for (int e = 0; e < kExperts; ++e) {
      float g = -std::log(-std::log(std::max(u(rng), 1e-12f)));
      s[e] = {bias.empty() ? u(rng) : bias[e] + g, e};
    }
    if (force_expert >= 0) s[force_expert].first = 1e9f;
    std::partial_sort(s.begin(), s.begin() + kTopK, s.end(),
                      [](auto& a, auto& b) { return a.first > b.first; });
    for (int k = 0; k < kTopK; ++k) counts[s[k].second]++;
  }
  return std::vector<int>(counts.begin(), counts.begin() + e_local);
}

std::vector<int> route(const std::string& kind, int ep, int tokens, std::mt19937& rng) {
  const int e_local = kExperts / ep, g_tokens = ep * tokens;
  if (kind.rfind("rows", 0) == 0) return std::vector<int>(e_local, std::stoi(kind.substr(4)));
  if (kind == "balanced") return std::vector<int>(e_local, g_tokens * kTopK / kExperts);
  if (kind == "uniform") return topk_counts(g_tokens, e_local, rng, {}, -1);
  if (kind == "skewed") {
    std::vector<int> rank(kExperts);
    std::iota(rank.begin(), rank.end(), 1);
    std::shuffle(rank.begin(), rank.end(), rng);
    std::vector<float> bias(kExperts);
    for (int e = 0; e < kExperts; ++e) bias[e] = -1.2f * std::log(float(rank[e]));
    return topk_counts(g_tokens, e_local, rng, bias, -1);
  }
  if (kind == "hot") return topk_counts(g_tokens, e_local, rng, {}, 0);
  if (kind == "sparse") {
    auto c = topk_counts(g_tokens, e_local, rng, {}, -1);
    std::vector<int> idx(e_local);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), rng);
    for (int i = 0; i < e_local / 2; ++i) c[idx[i]] = 0;
    return c;
  }
  std::fprintf(stderr, "unknown routing %s\n", kind.c_str());
  std::exit(2);
}

// Bench suites: (case name, ep, tokens/rank, routing). tp4ep4 = 16 local
// experts x ~190 rows (torchtitan TP4 EP4); ep8 = 8 experts x ~6k rows.
struct BenchCase { std::string name; int ep, tokens; std::string routing; };

std::vector<BenchCase> suite_cases(const std::string& s) {
  std::vector<BenchCase> small = {
      {"tp4ep4_uniform", 4, 512, "uniform"}, {"tp4ep4_balanced", 4, 512, "balanced"},
      {"tp4ep4_hot", 4, 512, "hot"},         {"tp4ep4_sparse", 4, 512, "sparse"},
      {"tp4ep4_rows64", 4, 512, "rows64"},   {"tp4ep4_rows8", 4, 512, "rows8"}};
  std::vector<BenchCase> large = {
      {"ep8_uniform", 8, 8192, "uniform"}, {"ep8_balanced", 8, 8192, "balanced"},
      {"ep8_hot", 8, 8192, "hot"},         {"ep8_rows1024", 8, 8192, "rows1024"}};
  if (s == "quick") return {small[0], large[0]};
  if (s == "small") return small;
  if (s == "large") return large;
  if (s == "full") { auto v = small; v.insert(v.end(), large.begin(), large.end()); return v; }
  if (s == "none") return {};
  // a single case, e.g. --suite case:tp4ep4_hot
  if (s.rfind("case:", 0) == 0) {
    auto v = small; v.insert(v.end(), large.begin(), large.end());
    for (auto& c : v) if (c.name == s.substr(5)) return {c};
  }
  std::fprintf(stderr, "unknown suite %s (quick|small|large|full|none|case:<name>)\n", s.c_str());
  std::exit(2);
}

// ------------------------------------------------------------------- arms ---
struct Arm {
  tune::Launch launch[3];
  int wgs[3] = {0, 0, 0};
  std::string desc[3];
  std::string so[3];
};

// Manifest: one line per kind, "<kind> <path.so> <wgs>" (wgs 0 => Xe cores x
// 512 / work-group threads, i.e. the production rule).
Arm load_arm(const std::string& manifest, int cores) {
  Arm arm;
  std::ifstream f(manifest);
  if (!f) { std::fprintf(stderr, "cannot read %s\n", manifest.c_str()); std::exit(2); }
  std::string kind, path;
  int wgs;
  while (f >> kind >> path >> wgs) {
    int k = kind == "m_rowB" ? kMRow : kind == "m_colB" ? kMCol : kind == "k_grouped" ? kK : -1;
    if (k < 0) { std::fprintf(stderr, "bad kind %s in %s\n", kind.c_str(), manifest.c_str()); std::exit(2); }
    void* h = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "dlopen %s: %s\n", path.c_str(), dlerror()); std::exit(3); }
    auto fn = reinterpret_cast<tune::CandidateFn>(dlsym(h, tune::kCandidateSymbol));
    if (!fn) { std::fprintf(stderr, "%s: no %s\n", path.c_str(), tune::kCandidateSymbol); std::exit(3); }
    const tune::Candidate* c = fn();
    arm.launch[k] = k == kMRow ? c->m_rowB : k == kMCol ? c->m_colB : c->k_grouped;
    if (!arm.launch[k]) { std::fprintf(stderr, "%s has no %s launcher\n", path.c_str(), kind.c_str()); std::exit(3); }
    arm.wgs[k] = wgs > 0 ? wgs : std::max(1, cores * 512 / std::max(1, c->threads));
    arm.desc[k] = c->name + " " + c->desc;
    arm.so[k] = path;
  }
  for (int k = 0; k < 3; ++k)
    if (!arm.launch[k]) { std::fprintf(stderr, "%s: missing %s\n", manifest.c_str(), kKindName[k]); std::exit(2); }
  return arm;
}

// ------------------------------------------------------------- buffers ---
// Device buffers for one (problem, row counts) instance.
struct Instance {
  sycl::queue& q;
  Problem pr;
  std::vector<int> counts;
  int total = 0, groups = 0;
  size_t a_elems = 0, b_elems = 0, d_elems = 0;
  bf16 *a = nullptr, *b = nullptr, *d = nullptr, *d2 = nullptr;
  int32_t* offs = nullptr;

  Instance(sycl::queue& q_, const Problem& p, const std::vector<int>& c, uint32_t seed)
      : q(q_), pr(p), counts(c), groups(int(c.size())) {
    total = std::accumulate(c.begin(), c.end(), 0);
    const size_t R = size_t(std::max(total, 1));
    if (pr.kind == kK) {
      a_elems = R * pr.k_or_m;
      b_elems = R * pr.n;
      d_elems = size_t(groups) * pr.k_or_m * pr.n;
    } else {
      a_elems = R * pr.k_or_m;
      b_elems = size_t(groups) * pr.k_or_m * pr.n;
      d_elems = R * pr.n;
    }
    a = sycl::malloc_device<bf16>(a_elems + kGuard, q);
    b = sycl::malloc_device<bf16>(b_elems + kGuard, q);
    d = sycl::malloc_device<bf16>(d_elems + kGuard, q);
    d2 = sycl::malloc_device<bf16>(d_elems + kGuard, q);
    offs = sycl::malloc_device<int32_t>(std::max(groups, 1), q);
    fill_random(a, a_elems, seed * 2654435761u + 1);
    fill_random(b, b_elems, seed * 2246822519u + 7);
    std::vector<int32_t> o(groups);
    int acc = 0;
    for (int g = 0; g < groups; ++g) { acc += c[g]; o[g] = acc; }
    if (groups) q.memcpy(offs, o.data(), groups * sizeof(int32_t));
    q.wait();
  }
  ~Instance() {
    sycl::free(a, q); sycl::free(b, q); sycl::free(d, q); sycl::free(d2, q); sycl::free(offs, q);
  }
  void fill_random(bf16* p, size_t n, uint32_t seed) {
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      uint32_t x = uint32_t(i[0]) * 0x9e3779b9u ^ seed;
      x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
      p[i] = bf16(float(x >> 8) * (1.0f / 16777216.0f) - 0.5f);
    });
  }
  void fill_sentinel(bf16* p) {
    const size_t n = d_elems + kGuard;
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      reinterpret_cast<uint16_t*>(p)[i] = kSentinel;
    }).wait();
  }
  Params params(bf16* out, int wgs) const {
    Params p;
    p.a = a; p.b = b; p.d = out; p.offs = offs; p.num_groups = groups; p.N = pr.n;
    p.num_workgroups = wgs;
    if (pr.kind == kK) {
      p.M = pr.k_or_m; p.K = total; p.lda = pr.k_or_m; p.ldb = pr.n; p.ldd = pr.n;
      p.group_stride_d = int64_t(pr.k_or_m) * pr.n;
    } else {
      p.M = total; p.K = pr.k_or_m; p.lda = pr.k_or_m; p.ldd = pr.n;
      p.ldb = pr.kind == kMRow ? pr.k_or_m : pr.n;
      p.group_stride_b = int64_t(pr.k_or_m) * pr.n;
    }
    return p;
  }
  double flops() const { return 2.0 * double(total) * pr.k_or_m * pr.n; }

  // Naive fp32 reference into `out` (bf16).
  void reference(bf16* out) {
    const bf16 *A = a, *B = b;
    const int32_t* O = offs;
    const int G = groups, N = pr.n, KM = pr.k_or_m;
    const Kind kind = pr.kind;
    if (kind == kK) {
      q.parallel_for(sycl::range<2>(size_t(G) * KM, N), [=](sycl::id<2> id) {
        const int g = int(id[0] / KM), m = int(id[0] % KM), n = int(id[1]);
        const int r0 = g ? O[g - 1] : 0, r1 = O[g];
        float acc = 0.f;
        for (int r = r0; r < r1; ++r) acc += float(A[size_t(r) * KM + m]) * float(B[size_t(r) * N + n]);
        out[(size_t(g) * KM + m) * N + n] = bf16(acc);
      }).wait();
    } else if (total > 0) {
      q.parallel_for(sycl::range<2>(size_t(total), N), [=](sycl::id<2> id) {
        const int r = int(id[0]), n = int(id[1]);
        int g = 0;
        while (g < G - 1 && O[g] <= r) ++g;
        const bf16* w = B + size_t(g) * KM * N;
        float acc = 0.f;
        for (int k = 0; k < KM; ++k) {
          float bv = kind == kMRow ? float(w[size_t(n) * KM + k]) : float(w[size_t(k) * N + n]);
          acc += float(A[size_t(r) * KM + k]) * bv;
        }
        out[size_t(r) * N + n] = bf16(acc);
      }).wait();
    }
  }
};

// Full comparison of `got` against `ref` over the d_elems outputs plus the
// sentinel guard after them (must be untouched). Tolerance per element:
// |got - ref| <= 0.02 |ref| + atol, plus relative L2 error <= 1e-2.
struct Cmp {
  uint64_t bad = 0, nan = 0, guard_bad = 0;
  double max_err = 0, rel_l2 = 0;
  bool ok() const { return bad == 0 && nan == 0 && guard_bad == 0 && rel_l2 <= 1e-2; }
  std::string json() const {
    char s[256];
    std::snprintf(s, sizeof(s),
                  "{\"ok\": %s, \"bad\": %llu, \"nan\": %llu, \"guard_bad\": %llu, \"max_err\": %.4g, \"rel_l2\": %.3g}",
                  ok() ? "true" : "false", (unsigned long long)bad, (unsigned long long)nan,
                  (unsigned long long)guard_bad, max_err, rel_l2);
    return s;
  }
};

Cmp compare(sycl::queue& q, const bf16* got, const bf16* ref, size_t n, float atol) {
  struct Acc { uint64_t bad, nan, guard; float max_err; double e2, r2; };
  Acc* acc = sycl::malloc_shared<Acc>(1, q);
  *acc = {0, 0, 0, 0.f, 0.0, 0.0};
  const size_t total = n + kGuard;
  q.submit([&](sycl::handler& h) {
    auto rb = sycl::reduction(&acc->bad, sycl::plus<uint64_t>());
    auto rn = sycl::reduction(&acc->nan, sycl::plus<uint64_t>());
    auto rg = sycl::reduction(&acc->guard, sycl::plus<uint64_t>());
    auto rm = sycl::reduction(&acc->max_err, sycl::maximum<float>());
    auto re = sycl::reduction(&acc->e2, sycl::plus<double>());
    auto rr = sycl::reduction(&acc->r2, sycl::plus<double>());
    h.parallel_for(sycl::range<1>(total), rb, rn, rg, rm, re, rr,
                   [=](sycl::id<1> i, auto& b, auto& nn, auto& g, auto& m, auto& e, auto& r) {
      const size_t k = i[0];
      const uint16_t bits = reinterpret_cast<const uint16_t*>(got)[k];
      if (k >= n) { if (bits != kSentinel) g += uint64_t(1); return; }
      const float x = float(got[k]), y = float(ref[k]);
      if (sycl::isnan(x) || sycl::isinf(x)) { nn += uint64_t(1); return; }
      const float err = sycl::fabs(x - y);
      if (!(err <= 0.02f * sycl::fabs(y) + atol)) b += uint64_t(1);
      m.combine(err);
      e += double(err) * err;
      r += double(y) * y;
    });
  }).wait();
  Cmp c;
  c.bad = acc->bad; c.nan = acc->nan; c.guard_bad = acc->guard; c.max_err = acc->max_err;
  c.rel_l2 = acc->r2 > 0 ? std::sqrt(acc->e2 / acc->r2) : (acc->e2 > 0 ? 1.0 : 0.0);
  sycl::free(acc, q);
  return c;
}

// --------------------------------------------------------------- verify ---
struct VerifyCase { std::string name; Problem pr; std::vector<int> counts; };

std::vector<VerifyCase> verify_cases(uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<VerifyCase> v;
  // DSv3 shapes, small routed calls (hidden seed: different every evaluation).
  auto routed = route("uniform", 4, 128 + int(rng() % 256), rng);  // 16 experts, ~50-140 rows
  auto hot = route("hot", 8, 512 + int(rng() % 512), rng);         // 8 experts, one hot
  // Ragged: empty groups, single rows, sizes around the tile edges.
  const int edge_pool[] = {0, 1, 7, 15, 16, 31, 33, 127, 128, 129, 255, 257, 300, 0, 513};
  std::vector<int> edge(16);
  for (auto& e : edge) e = edge_pool[rng() % (sizeof(edge_pool) / sizeof(int))];
  edge[rng() % 16] = 0;
  for (auto& op : dsv3_ops()) {
    v.push_back({op.name + "/routed", op, routed});
    v.push_back({op.name + "/edge", op, edge});
  }
  v.push_back({"w13_fwd/hot", dsv3_ops()[0], hot});
  v.push_back({"w2_dX/hot", dsv3_ops()[3], hot});
  v.push_back({"w2_dW/hot", dsv3_ops()[5], hot});
  // Shapes that are not DSv3 (catch hard-coded dims). Leading dims stay
  // multiples of 32, which the production host checks require.
  std::vector<int> ragged5 = {37, 0, 96, 1, 210};
  v.push_back({"gen_mrow_k192_n320", {"gen", kMRow, 192, 320}, ragged5});
  v.push_back({"gen_mcol_k96_n544", {"gen", kMCol, 96, 544}, ragged5});
  v.push_back({"gen_k_m96_n160", {"gen", kK, 96, 160}, ragged5});
  v.push_back({"gen_k_m288_n64", {"gen", kK, 288, 64}, {0, 5, 64, 0, 33}});
  v.push_back({"gen_mrow_one_group", {"gen", kMRow, 512, 128}, {77}});
  // All groups empty: M-grouped writes nothing, K-grouped writes zeros.
  v.push_back({"w2_fwd/all_empty", dsv3_ops()[1], std::vector<int>(8, 0)});
  v.push_back({"w2_dW/all_empty", dsv3_ops()[5], std::vector<int>(8, 0)});
  return v;
}

// --------------------------------------------------------------- timing ---
double time_arm(sycl::queue& q, const tune::Launch& L, const Params& p, int wgs, int warmup, int iters) {
  for (int i = 0; i < warmup; ++i) L(q, p, wgs);
  q.wait();
  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; ++i) L(q, p, wgs);
  q.wait();
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / iters;
}

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  const size_t n = v.size();
  return n == 0 ? 0 : (n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]));
}

std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') { o += '\\'; o += c; }
    else if (c == '\n') o += "\\n";
    else o += c;
  }
  return o + "\"";
}

std::string counts_str(const std::vector<int>& c) {
  std::string s = "[";
  for (size_t i = 0; i < c.size(); ++i) s += (i ? "," : "") + std::to_string(c[i]);
  return s + "]";
}

struct Opts {
  std::string base, cand, mode = "both", suite = "quick", json, profile_case = "tp4ep4_uniform",
              profile_op = "w13_fwd", ops;
  uint32_t verify_seed = 0;
  int rounds = 11, warmup = 3, profile_iters = 50;
  double min_batch_ms = 10.0;
};

void usage() {
  std::printf(
      "agent_eval --base base.manifest --cand cand.manifest [options]\n"
      "  --mode verify|bench|both|profile   (default both)\n"
      "  --suite quick|small|large|full|none|case:<name>   bench cases (default quick)\n"
      "  --ops a,b            subset of w13_fwd,w2_fwd,w13_dX,w2_dX,w13_dW,w2_dW\n"
      "  --verify-seed N      hidden verification seed (default 0)\n"
      "  --rounds N           A/B rounds per (case, op) (default 11)\n"
      "  --min-batch-ms X     launches per timed batch chosen to take >= X ms (default 10)\n"
      "  --profile-case C --profile-op O --profile-iters N   (mode profile; candidate only)\n"
      "  --json FILE\n"
      "manifest: lines '<m_rowB|m_colB|k_grouped> <lib.so> <wgs>' (wgs 0 = production rule)\n");
}

} // namespace

int main(int argc, char** argv) {
  Opts o;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) { usage(); std::exit(2); }
      return argv[++i];
    };
    if (a == "--base") o.base = next();
    else if (a == "--cand") o.cand = next();
    else if (a == "--mode") o.mode = next();
    else if (a == "--suite") o.suite = next();
    else if (a == "--ops") o.ops = next();
    else if (a == "--json") o.json = next();
    else if (a == "--verify-seed") o.verify_seed = uint32_t(std::stoul(next()));
    else if (a == "--rounds") o.rounds = std::stoi(next());
    else if (a == "--warmup") o.warmup = std::stoi(next());
    else if (a == "--min-batch-ms") o.min_batch_ms = std::stod(next());
    else if (a == "--profile-case") o.profile_case = next();
    else if (a == "--profile-op") o.profile_op = next();
    else if (a == "--profile-iters") o.profile_iters = std::stoi(next());
    else { usage(); return a == "-h" || a == "--help" ? 0 : 2; }
  }
  if (o.cand.empty() || (o.base.empty() && o.mode != "profile" && o.mode != "verify")) { usage(); return 2; }

  sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order()};
  auto dev = q.get_device();
  const int cores = sycltla::grouped_gemm::xe_core_count(dev);
  Arm cand = load_arm(o.cand, cores);
  Arm base;
  if (!o.base.empty()) base = load_arm(o.base, cores);

  std::ostringstream js;
  js << "{\"device\": " << jstr(dev.get_info<sycl::info::device::name>()) << ", \"xe_cores\": " << cores;
  js << ", \"cand\": {";
  for (int k = 0; k < 3; ++k)
    js << (k ? ", " : "") << jstr(kKindName[k]) << ": {\"desc\": " << jstr(cand.desc[k]) << ", \"wgs\": " << cand.wgs[k] << "}";
  js << "}";
  if (!o.base.empty()) {
    js << ", \"base\": {";
    for (int k = 0; k < 3; ++k)
      js << (k ? ", " : "") << jstr(kKindName[k]) << ": {\"desc\": " << jstr(base.desc[k]) << ", \"wgs\": " << base.wgs[k] << "}";
    js << "}";
  }
  std::printf("device: %s, %d Xe cores\n", dev.get_info<sycl::info::device::name>().c_str(), cores);
  for (int k = 0; k < 3; ++k)
    std::printf("  cand %-9s %s wgs=%d\n", kKindName[k], cand.desc[k].c_str(), cand.wgs[k]);
  std::fflush(stdout);

  std::vector<std::string> op_filter;
  for (size_t s = 0; s <= o.ops.size() && !o.ops.empty();) {
    size_t e = o.ops.find(',', s);
    if (e == std::string::npos) e = o.ops.size();
    if (e > s) op_filter.push_back(o.ops.substr(s, e - s));
    s = e + 1;
  }
  auto op_selected = [&](const std::string& n) {
    return op_filter.empty() || std::find(op_filter.begin(), op_filter.end(), n) != op_filter.end();
  };

  // ---- profile: candidate only, one (case, op), many launches
  if (o.mode == "profile") {
    auto cases = suite_cases("full");
    BenchCase bc{"", 0, 0, ""};
    for (auto& c : cases) if (c.name == o.profile_case) bc = c;
    if (bc.name.empty()) { std::fprintf(stderr, "unknown --profile-case %s\n", o.profile_case.c_str()); return 2; }
    const Problem* pr = nullptr;
    for (auto& p : dsv3_ops()) if (p.name == o.profile_op) pr = &p;
    if (!pr) { std::fprintf(stderr, "unknown --profile-op %s\n", o.profile_op.c_str()); return 2; }
    std::mt19937 rng(0);
    Instance in(q, *pr, route(bc.routing, bc.ep, bc.tokens, rng), 1);
    const auto& L = cand.launch[pr->kind];
    const int w = cand.wgs[pr->kind];
    Params p = in.params(in.d, w);
    double ms = time_arm(q, L, p, w, o.warmup, o.profile_iters);
    std::printf("profile %s %s: %d launches, %.4f ms/call, %.1f TFLOPS\n", bc.name.c_str(), pr->name.c_str(),
                o.profile_iters, ms, in.flops() / (ms * 1e-3) / 1e12);
    js << ", \"profile\": {\"case\": " << jstr(bc.name) << ", \"op\": " << jstr(pr->name) << ", \"ms\": " << ms
       << ", \"tflops\": " << in.flops() / (ms * 1e-3) / 1e12 << ", \"counts\": " << counts_str(in.counts) << "}}";
    if (!o.json.empty()) std::ofstream(o.json) << js.str() << "\n";
    return 0;
  }

  bool verify_ok = true;
  // ---- verify: every kind against the naive reference, hidden cases
  if (o.mode == "verify" || o.mode == "both") {
    js << ", \"verify\": [";
    std::printf("\n== verify (seed %u)\n", o.verify_seed);
    bool first = true;
    for (auto& vc : verify_cases(o.verify_seed)) {
      Instance in(q, vc.pr, vc.counts, o.verify_seed + 11);
      in.fill_sentinel(in.d);
      in.fill_sentinel(in.d2);
      in.reference(in.d2);
      // run twice: the second call must not depend on the first (stale state)
      const auto& L = cand.launch[vc.pr.kind];
      const int w = cand.wgs[vc.pr.kind];
      Params p = in.params(in.d, w);
      std::string err;
      try {
        L(q, p, w);
        q.wait_and_throw();
        in.fill_sentinel(in.d);
        L(q, p, w);
        q.wait_and_throw();
      } catch (const std::exception& e) {
        err = e.what();
      }
      Cmp c;
      if (err.empty()) {
        if (in.pr.kind != kK && in.total == 0) {
          // M-grouped with all-empty input: nothing may be written at all.
          std::vector<uint16_t> h(in.d_elems + kGuard);
          q.memcpy(h.data(), in.d, h.size() * 2).wait();
          for (auto x : h) if (x != kSentinel) c.guard_bad++;
        } else {
          c = compare(q, in.d, in.d2, in.d_elems, 0.05f);
        }
      }
      const bool ok = err.empty() && c.ok();
      verify_ok &= ok;
      std::printf("  %-28s %-9s rows=%-6d %s %s\n", vc.name.c_str(), kKindName[vc.pr.kind], in.total,
                  ok ? "ok" : "FAIL", err.empty() ? c.json().c_str() : err.c_str());
      std::fflush(stdout);
      js << (first ? "" : ", ") << "{\"case\": " << jstr(vc.name) << ", \"kind\": " << jstr(kKindName[vc.pr.kind])
         << ", \"k_or_m\": " << vc.pr.k_or_m << ", \"n\": " << vc.pr.n << ", \"counts\": " << counts_str(vc.counts)
         << ", \"ok\": " << (ok ? "true" : "false") << ", \"cmp\": " << (err.empty() ? c.json() : "null")
         << ", \"error\": " << jstr(err) << "}";
      first = false;
    }
    js << "], \"verify_ok\": " << (verify_ok ? "true" : "false");
    std::printf("verify: %s\n", verify_ok ? "PASS" : "FAIL");
  }

  // ---- bench: A/B against the baseline on the suite
  bool bench_ok = true;
  if ((o.mode == "bench" || o.mode == "both") && (verify_ok || o.mode == "bench")) {
    js << ", \"bench\": [";
    bool first = true;
    for (auto& bc : suite_cases(o.suite)) {
      std::mt19937 rng(0);  // fixed routing: comparable across evaluations
      auto counts = route(bc.routing, bc.ep, bc.tokens, rng);
      const int total = std::accumulate(counts.begin(), counts.end(), 0);
      std::printf("\n== bench %s: %zu experts, rows %d..%d, total %d\n", bc.name.c_str(), counts.size(),
                  *std::min_element(counts.begin(), counts.end()), *std::max_element(counts.begin(), counts.end()),
                  total);
      std::printf("  %-8s %9s %9s %8s %8s %8s  %s\n", "op", "base ms", "cand ms", "speedup", "base TF", "cand TF",
                  "full-output check vs baseline");
      for (auto& pr : dsv3_ops()) {
        if (!op_selected(pr.name)) continue;
        Instance in(q, pr, counts, 1);
        const int kd = pr.kind;
        Params pb = in.params(in.d2, base.wgs[kd]);
        Params pc = in.params(in.d, cand.wgs[kd]);
        in.fill_sentinel(in.d);
        in.fill_sentinel(in.d2);
        std::string err;
        std::vector<double> tb, tc;
        try {
          // size the timed batch from one call of the baseline
          double est = time_arm(q, base.launch[kd], pb, base.wgs[kd], 1, 1);
          int iters = std::clamp(int(std::ceil(o.min_batch_ms / std::max(est, 1e-3))), 5, 400);
          for (int r = 0; r < o.rounds; ++r) {
            if (r % 2 == 0) {
              tb.push_back(time_arm(q, base.launch[kd], pb, base.wgs[kd], o.warmup, iters));
              tc.push_back(time_arm(q, cand.launch[kd], pc, cand.wgs[kd], o.warmup, iters));
            } else {
              tc.push_back(time_arm(q, cand.launch[kd], pc, cand.wgs[kd], o.warmup, iters));
              tb.push_back(time_arm(q, base.launch[kd], pb, base.wgs[kd], o.warmup, iters));
            }
          }
          q.wait_and_throw();
        } catch (const std::exception& e) {
          err = e.what();
        }
        Cmp c;
        if (err.empty()) c = compare(q, in.d, in.d2, in.d_elems, 0.05f);
        const bool ok = err.empty() && c.ok();
        bench_ok &= ok;
        const double mb = median(tb), mc = median(tc);
        // speedup = median of the per-round paired ratios (adjacent A/B
        // batches share the drift); noise = their interquartile range
        std::vector<double> ratio;
        for (size_t i = 0; i < tb.size() && i < tc.size(); ++i) ratio.push_back(tb[i] / tc[i]);
        std::sort(ratio.begin(), ratio.end());
        const double sp = (ok && !ratio.empty()) ? median(ratio) : 0.0;
        const double iqr = ratio.size() >= 4 ? ratio[ratio.size() * 3 / 4] - ratio[ratio.size() / 4] : 0.0;
        std::printf("  %-8s %9.4f %9.4f %7.3fx %8.1f %8.1f  %s\n", pr.name.c_str(), mb, mc, sp,
                    in.flops() / (mb * 1e-3) / 1e12, mc > 0 ? in.flops() / (mc * 1e-3) / 1e12 : 0.0,
                    err.empty() ? (ok ? "ok" : ("MISMATCH " + c.json()).c_str()) : err.c_str());
        std::fflush(stdout);
        js << (first ? "" : ", ") << "{\"case\": " << jstr(bc.name) << ", \"op\": " << jstr(pr.name)
           << ", \"kind\": " << jstr(kKindName[kd]) << ", \"total_rows\": " << total
           << ", \"counts\": " << counts_str(counts) << ", \"base_ms\": " << mb << ", \"cand_ms\": " << mc
           << ", \"speedup\": " << sp << ", \"speedup_iqr\": " << iqr << ", \"base_tflops\": " << in.flops() / (mb * 1e-3) / 1e12
           << ", \"cand_tflops\": " << (mc > 0 ? in.flops() / (mc * 1e-3) / 1e12 : 0.0)
           << ", \"base_rounds\": [";
        for (size_t i = 0; i < tb.size(); ++i) js << (i ? ", " : "") << tb[i];
        js << "], \"cand_rounds\": [";
        for (size_t i = 0; i < tc.size(); ++i) js << (i ? ", " : "") << tc[i];
        js << "], \"ok\": " << (ok ? "true" : "false") << ", \"cmp\": " << (err.empty() ? c.json() : "null")
           << ", \"error\": " << jstr(err) << "}";
        first = false;
      }
    }
    js << "], \"bench_ok\": " << (bench_ok ? "true" : "false");
  }
  js << "}";
  if (!o.json.empty()) std::ofstream(o.json) << js.str() << "\n";
  return (verify_ok && bench_ok) ? 0 : 1;
}
