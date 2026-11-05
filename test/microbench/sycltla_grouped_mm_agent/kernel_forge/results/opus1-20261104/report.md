# Kernel Forge (MCTS) run report

## Run

- workdir: `/mnt/aimod/subratag/ggemm_agent_runs/kernel_forge/opus1-20261104`
- iterations: 50 (target 50); nodes admitted: 50; failed attempts: 0
- actions: knob 10, llm 40; outcomes: ok 50
- LLM revisions: 40 claude sessions (incl. repairs), 0 repair rounds, spend $18.65 (budget $80, model opus)
- search suite `quick`, final suite `full`, C=0.1, vloss=0.05, alpha 0.5->0.3 over 1000 root visits, workers 8, knob ratio 0.3
- production vs itself measured on `quick`: 1.0016x; on `full`: 0.9969x

## Result

Best node **n046**: 1.0872x on `quick`, **1.1401x on `full`** (production re-run: 0.9969x).

Verdict: IMPROVED (+14.4%).

## Lineage of the best node

| step | node | action | change | quick score | vs parent |
|---|---|---|---|---|---|
| 0 | n000 | root | production kernel | 1.0000 |  |
| 1 | n007 | llm | I added cross-tile prefetch to `gemm_tile`, in `grouped_mm_kernel_impl.h` only (configs.json is unchanged). Both persistent kernels now look up their next tile before computing the current one. In the last few K iterations, where the prefetch slot is otherwise _[configs unchanged; header +77/-10 lines]_ | 1.0062 | +0.6% |
| 2 | n013 | knob | knob prefetch on m_colB: 256x256x32 sg8x4 pf3 wgs0 -> 256x256x32 sg8x4 pf1 wgs0 _[m_colB: prefetch 3->1; header +0/-0 lines]_ | 0.9873 | -1.9% |
| 3 | n024 | llm | In `gemm_tile`, the output now uses an explicit 8x32 bf16 store atom (`XE_STORE_2D<16, 8, 32>`, the same one SYCL-TLA's own MoE example uses) instead of the auto-selected 8x16. That halves the store messages each sub-group sends per 256x256 tile, which helps m _[m_colB: prefetch 1->3; header +6/-1 lines]_ | 1.0651 | +7.9% |
| 4 | n046 | llm | I added a device-side wave-tail split to both persistent kernels in `grouped_mm_kernel_impl.h`; configs.json is unchanged. It has not been compiled or run here; the harness will be the first build and test. Each work-group now counts the total tiles from `offs _[configs unchanged; header +129/-10 lines]_ | 1.0872 | +2.1% |

## Top candidates (search suite, then full re-evaluation)

| node | action | quick | full | parent | note |
|---|---|---|---|---|---|
| n046 | llm | 1.0872 | 1.1401 | n024 | I added a device-side wave-tail split to both persistent kernels in `grouped_mm_kernel_impl.h`; configs.json i |
| n037 | llm | 1.0725 | 1.1086 | n024 | Only the forward and dX kernels (`MGroupedKernel`) changed. Their A operand is row-major, so they now use an e |
| n024 | llm | 1.0651 | 1.0943 | n013 | In `gemm_tile`, the output now uses an explicit 8x32 bf16 store atom (`XE_STORE_2D<16, 8, 32>`, the same one S |
| n031 | llm | 1.0511 | - | n021 | In the forward and dX kernels (M-grouped only), sub-groups whose whole 32-row block lies past the group's last |
| n038 | llm | 1.0473 | - | n017 | I made sub-groups whose whole 32x64 accumulator block lies outside D skip their A/B block loads, reorder, DPAS |
| n018 | llm | 1.0358 | - | n002 | In the M-grouped kernel (forward and grad_input), each work-group now counts the total tiles across all groups |
| n022 | llm | 1.0315 | - | n015 | I added cross-tile prefetch pipelining to `gemm_tile`, used by the forward, dX and grad_weight kernels. Each w |
| n021 | llm | 1.0279 | - | n005 | I removed the parent's cross-tile prefetch, which cost about 3% on every op, and added a scheduler that splits |

## Per-op speedup of the best node vs production

| op | quick | full |
|---|---|---|
| w13_dW | 1.135x | 1.303x |
| w13_dX | 1.075x | 1.135x |
| w13_fwd | 1.053x | 1.037x |
| w2_dW | 1.116x | 1.257x |
| w2_dX | 1.029x | 1.066x |
| w2_fwd | 1.080x | 1.104x |

Per case (full):

- tp4ep4_uniform: 1.161x
- tp4ep4_balanced: 1.168x
- tp4ep4_hot: 1.123x
- tp4ep4_sparse: 1.326x
- tp4ep4_rows64 (probe, not scored): 1.279x
- tp4ep4_rows8 (probe, not scored): 1.237x
- tp8ep8_uniform: 1.178x
- tp8ep8_balanced: 1.185x
- tp8ep8_hot: 1.104x
- tp8ep8_sparse: 1.269x
- tp8ep8_rows64 (probe, not scored): 1.261x
- ep8_uniform: 1.023x
- ep8_balanced: 1.023x
- ep8_hot: 1.023x
- ep8_rows1024 (probe, not scored): 1.040x

## Configs and diff vs production

```
configs: configs unchanged
```

Header diff:

```diff
--- production/grouped_mm_kernel_impl.h
+++ candidate/grouped_mm_kernel_impl.h
@@ -111,6 +111,27 @@
     MMA_Atom<XE_DPAS_TT<8, float, T, T>>,
     Layout<typename Cfg::WGTile>,
     typename Cfg::SGLayout>::TiledMMA;
+
+// Half-N variant of a tile config (same sub-group layout, so the same
+// work-group size). Used for the persistent grid's last, partially filled
+// wave: each remaining tile is split into two N halves computed by two
+// work-groups, which shortens the exposed tail wave.
+template <class Cfg>
+struct HalfNCfg {
+  using WGTile =
+      Shape<Int<Cfg::kTileM>, Int<Cfg::kTileN / 2>, Int<Cfg::kTileK>>;
+  using SGLayout = typename Cfg::SGLayout;
+  static constexpr int kTileM = Cfg::kTileM;
+  static constexpr int kTileN = Cfg::kTileN / 2;
+  static constexpr int kTileK = Cfg::kTileK;
+  static constexpr int kPrefetchDist = Cfg::kPrefetchDist;
+};
+
+template <class Cfg>
+constexpr bool kCanSplitTail = (Cfg::kTileN % 2 == 0) &&
+    ((Cfg::kTileN / 2) %
+         (decltype(cute::size<1>(typename Cfg::SGLayout{}))::value * 16) ==
+     0);
 
 // (rows, cols) matrix view. 'R': element (r, c) at r * ld + c.
 //                           'C': element (r, c) at r + c * ld.
@@ -144,6 +165,11 @@
     DTensor& D,
     int m_idx,
     int n_idx,
+    ATensor const& A_next,
+    BTensor const& B_next,
+    int m_next,
+    int n_next,
+    bool has_next,
     TiledMMA const& mma) {
   auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
   auto local_id = item.get_local_linear_id();
@@ -163,7 +189,12 @@
 
   auto tiled_copy_a = get_block_2d_copy_A<void>(mma, A);
   auto tiled_copy_b = get_block_2d_copy_B<void>(mma, B);
-  auto tiled_copy_d = get_block_2d_copy_D<void>(mma, D);
+  // Wide 8x32 bf16 store atom (as in the SYCL-TLA MoE example) instead of
+  // the auto-selected 8x16: halves the epilogue's 2D store messages, which
+  // matters for short-K tiles (grad_weight) whose 256x256 write is exposed.
+  // The fp32->bf16 reorder below already rearranges data, so the wider
+  // register layout comes for free. Hardware clips at the surface bounds.
+  auto tiled_copy_d = get_block_2d_copy_D<XE_STORE_2D<16, 8, 32>>(mma, D);
 
   auto thr_copy_a = tiled_copy_a.get_slice(local_id);
   auto thr_copy_b = tiled_copy_b.get_slice(local_id);
@@ -189,6 +220,31 @@
 
   auto pAgA = thr_prefetch_A.partition_S(gA);
   auto pBgB = thr_prefetch_B.partition_S(gB);
+
+  // Cross-tile prefetch: the work-group's NEXT tile (from the persistent
+  // scheduler) gets its first K blocks prefetched in the otherwise idle
+  // prefetch slots of this tile's last iterations, so its prologue loads
+  // hit L1/L3 instead of paying the full memory latency (short-K
+  // grad_weight tiles have only ~6 K blocks, so this latency is not hidden).
+  Tensor cAn = make_identity_tensor(A_next.shape());
+  Tensor cBn = make_identity_tensor(B_next.shape());
+  Tensor gAn =
+      local_tile(cAn, select<0, 2>(wg_tile), make_coord(m_next, _));
+  Tensor gBn =
+      local_tile(cBn, select<1, 2>(wg_tile), make_coord(n_next, _));
+  auto prefetch_an =
+      make_block_2d_prefetch(get_block_2d_copy_A<void>(mma, A_next));
+  auto prefetch_bn =
+      make_block_2d_prefetch(get_block_2d_copy_B<void>(mma, B_next));
+  auto thr_prefetch_An = prefetch_an.get_slice(local_id);
+  auto thr_prefetch_Bn = prefetch_bn.get_slice(local_id);
+  auto pAgAn = thr_prefetch_An.partition_S(gAn);
+  auto pBgBn = thr_prefetch_Bn.partition_S(gBn);
+  const int next_k_count =
+      has_next ? int(ceil_div(shape<1>(A_next), get<2>(wg_tile))) : 0;
+  const int next_pf_limit =
+      next_k_count < PrefetchDist ? next_k_count : PrefetchDist;
+  int next_pk = 0;
 
   constexpr int barrier_scope = 2;
   constexpr int prefetch_dist = PrefetchDist;
@@ -215,6 +271,10 @@
     if (prefetch_k < k_tile_count) {
       prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
       prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
+    } else if (next_pk < next_pf_limit) {
+      prefetch(prefetch_an, pAgAn(_, _, _, next_pk));
+      prefetch(prefetch_bn, pBgBn(_, _, _, next_pk));
+      next_pk++;
     }
 
     reorder(tArA, tCrA);
@@ -222,6 +282,13 @@
 
     cute::gemm(mma, tCrA, tCrB, tCrD);
     barrier_wait(barrier_scope);
+  }
+
+  // Tiles shorter than the prefetch distance: issue the rest of the next
+  // tile's prologue before the epilogue store.
+  for (; next_pk < next_pf_limit; next_pk++) {
+    prefetch(prefetch_an, pAgAn(_, _, _, next_pk));
+    prefetch(prefetch_bn, pBgBn(_, _, _, next_pk));
   }
 
   reorder(tCrD, tCrD_final);
@@ -256,6 +323,24 @@
     const int32_t m = KGrouped ? fixed_m : c.rows;
     c.tiles_m = (m + TileM - 1) / TileM;
     c.tiles = int64_t(c.tiles_m) * int64_t((n + TileN - 1) / TileN);
+  }
+
+  // Total tile count over all groups (same per-group counts as describe()).
+  CUTE_DEVICE int64_t total_tiles() const {
+    const int64_t tn = int64_t((n + TileN - 1) / TileN);
+    if constexpr (KGrouped) {
+      return int64_t(num_groups) * int64_t((fixed_m + TileM - 1) / TileM) * tn;
+    } else {
+      int64_t t = 0;
+      for (int32_t g = 0; g < num_groups; ++g) {
+        int32_t start = (g == 0) ? 0 : offs[g - 1];
+        int32_t end = offs[g];
+        start = start < 0 ? 0 : start;
+        const int32_t rows = (end > start) ? (end - start) : 0;
+        t += (rows + TileM - 1) / TileM;
+      }
+      return t * tn;
+    }
   }
 
   // Resolve linear tile index -> (group, m_idx, n_idx). Returns false once
@@ -299,10 +384,33 @@
     T* b = const_cast<T*>(static_cast<const T*>(p.b));
     T* d = static_cast<T*>(p.d);
 
-    for (int64_t lin = wg;; lin += nwg) {
-      int32_t m_idx, n_idx;
-      if (!sched.locate(lin, cur, m_idx, n_idx)) {
-        break;
+    // Wave-tail split: if the last wave would occupy at most half of the
+    // work-groups, its tiles are computed as two N halves each.
+    const int64_t total = sched.total_tiles();
+    const int64_t rem = total % nwg;
+    const bool split_tail =
+        kCanSplitTail<Cfg> && rem > 0 && 2 * rem <= nwg;
+    const int64_t full_end = split_tail ? total - rem : total;
+    const bool my_tail = split_tail && wg < 2 * rem;
+    const int64_t tail_lin = full_end + (wg >> 1);
+
+    int32_t m_idx = 0, n_idx = 0;
+    bool have = (wg < full_end) && sched.locate(wg, cur, m_idx, n_idx);
+    for (int64_t lin = wg; have; lin += nwg) {
+      GroupCursor nxt = cur;
+      int32_t m_nx = 0, n_nx = 0;
+      const bool next_full =
+          (lin + nwg < full_end) && sched.locate(lin + nwg, nxt, m_nx, n_nx);
+      bool pf_next = next_full;
+      if (!next_full) {
+        nxt = cur;
+        if (my_tail) {
+          // Prefetch the tail tile's first K blocks (full-tile footprint).
+          pf_next = sched.locate(tail_lin, nxt, m_nx, n_nx);
+          if (!pf_next) {
+            nxt = cur;
+          }
+        }
       }
       auto A = make_2d_tensor<'R'>(
           a + int64_t(cur.row_start) * p.lda, cur.rows, p.K, int(p.lda));
@@ -310,7 +418,38 @@
           b + int64_t(cur.g) * p.group_stride_b, p.N, p.K, int(p.ldb));
       auto D = make_2d_tensor<'R'>(
           d + int64_t(cur.row_start) * p.ldd, cur.rows, p.N, int(p.ldd));
-      gemm_tile<Cfg::kPrefetchDist>(A, B, D, m_idx, n_idx, mma);
+      auto An = make_2d_tensor<'R'>(
+          a + int64_t(nxt.row_start) * p.lda, nxt.rows, p.K, int(p.lda));
+      auto Bn = make_2d_tensor<LayoutB>(
+          b + int64_t(nxt.g) * p.group_stride_b, p.N, p.K, int(p.ldb));
+      gemm_tile<Cfg::kPrefetchDist>(
+          A, B, D, m_idx, n_idx, An, Bn, m_nx, n_nx, pf_next, mma);
+      cur = nxt;
+      m_idx = m_nx;
+      n_idx = n_nx;
+      have = next_full;
+    }
+
+    if constexpr (kCanSplitTail<Cfg>) {
+      if (my_tail) {
+        GroupCursor tc = cur; // positioned at or before tail_lin
+        int32_t tm = 0, tn = 0;
+        if (sched.locate(tail_lin, tc, tm, tn)) {
+          constexpr int kHalfN = Cfg::kTileN / 2;
+          const int32_t hn = 2 * tn + int32_t(wg & 1);
+          if (int64_t(hn) * kHalfN < int64_t(p.N)) {
+            const auto mma_h = TiledMmaFor<T, HalfNCfg<Cfg>>{};
+            auto A = make_2d_tensor<'R'>(
+                a + int64_t(tc.row_start) * p.lda, tc.rows, p.K, int(p.lda));
+            auto B = make_2d_tensor<LayoutB>(
+                b + int64_t(tc.g) * p.group_stride_b, p.N, p.K, int(p.ldb));
+            auto D = make_2d_tensor<'R'>(
+                d + int64_t(tc.row_start) * p.ldd, tc.rows, p.N, int(p.ldd));
+            gemm_tile<Cfg::kPrefetchDist>(
+                A, B, D, tm, hn, A, B, tm, hn, false, mma_h);
+          }
+        }
+      }
     }
   }
 };
@@ -334,10 +473,31 @@
     T* b = const_cast<T*>(static_cast<const T*>(p.b));
     T* d = static_cast<T*>(p.d);
 
-    for (int64_t lin = wg;; lin += nwg) {
-      int32_t m_idx, n_idx;
-      if (!sched.locate(lin, cur, m_idx, n_idx)) {
-        break;
+    // Wave-tail split (see MGroupedKernel).
+    const int64_t total = sched.total_tiles();
+    const int64_t rem = total % nwg;
+    const bool split_tail =
+        kCanSplitTail<Cfg> && rem > 0 && 2 * rem <= nwg;
+    const int64_t full_end = split_tail ? total - rem : total;
+    const bool my_tail = split_tail && wg < 2 * rem;
+    const int64_t tail_lin = full_end + (wg >> 1);
+
+    int32_t m_idx = 0, n_idx = 0;
+    bool have = (wg < full_end) && sched.locate(wg, cur, m_idx, n_idx);
+    for (int64_t lin = wg; have; lin += nwg) {
+      GroupCursor nxt = cur;
+      int32_t m_nx = 0, n_nx = 0;
+      const bool next_full =
+          (lin + nwg < full_end) && sched.locate(lin + nwg, nxt, m_nx, n_nx);
+      bool pf_next = next_full;
+      if (!next_full) {
+        nxt = cur;
+        if (my_tail) {
+          pf_next = sched.locate(tail_lin, nxt, m_nx, n_nx);
+          if (!pf_next) {
+            nxt = cur;
+          }
+        }
       }
       // A: (M, rows) with element (m, r) at m + r * lda  -> column-major.
       auto A = make_2d_tensor<'C'>(
@@ -348,7 +508,38 @@
           b + int64_t(cur.row_start) * p.ldb, p.N, cur.rows, int(p.ldb));
       auto D = make_2d_tensor<'R'>(
           d + int64_t(cur.g) * p.group_stride_d, p.M, p.N, int(p.ldd));
-      gemm_tile<Cfg::kPrefetchDist>(A, B, D, m_idx, n_idx, mma);
+      auto An = make_2d_tensor<'C'>(
+          a + int64_t(nxt.row_start) * p.lda, p.M, nxt.rows, int(p.lda));
+      auto Bn = make_2d_tensor<'C'>(
+          b + int64_t(nxt.row_start) * p.ldb, p.N, nxt.rows, int(p.ldb));
+      gemm_tile<Cfg::kPrefetchDist>(
+          A, B, D, m_idx, n_idx, An, Bn, m_nx, n_nx, pf_next, mma);
+      cur = nxt;
+      m_idx = m_nx;
+      n_idx = n_nx;
+      have = next_full;
+    }
+
+    if constexpr (kCanSplitTail<Cfg>) {
+      if (my_tail) {
+        GroupCursor tc = cur; // positioned at or before tail_lin
+        int32_t tm = 0, tn = 0;
+        if (sched.locate(tail_lin, tc, tm, tn)) {
+          constexpr int kHalfN = Cfg::kTileN / 2;
+          const int32_t hn = 2 * tn + int32_t(wg & 1);
+          if (int64_t(hn) * kHalfN < int64_t(p.N)) {
+            const auto mma_h = TiledMmaFor<T, HalfNCfg<Cfg>>{};
+            auto A = make_2d_tensor<'C'>(
+                a + int64_t(tc.row_start) * p.lda, p.M, tc.rows, int(p.lda));
+            auto B = make_2d_tensor<'C'>(
+                b + int64_t(tc.row_start) * p.ldb, p.N, tc.rows, int(p.ldb));
+            auto D = make_2d_tensor<'R'>(
+                d + int64_t(tc.g) * p.group_stride_d, p.M, p.N, int(p.ldd));
+            gemm_tile<Cfg::kPrefetchDist>(
+                A, B, D, tm, hn, A, B, tm, hn, false, mma_h);
+          }
+        }
+      }
     }
   }
 };

```

The best candidate directory is copied to `/mnt/aimod/subratag/ggemm_agent_runs/kernel_forge/opus1-20261104/best` (header + configs.json).

## Tree summary

(N = visits, score = own search-suite score, best-subtree = best score in the subtree; the selection value is L = 1/score)

```
n000 [root] N=51 score=1.0000 best-subtree=1.0872 production kernel
  n004 [knob] N=8 score=0.9113 best-subtree=1.0315 knob tile on m_rowB: 256x256x32 sg8x4 pf3 wgs0 -> 512x128x32 sg16x2 pf3 wgs0
    n009 [llm] N=2 score=0.9845 best-subtree=0.9979 In the K-grouped (grad_weight) kernel, I added a hook that runs after the mainloop and bef
      n044 [knob] N=1 score=0.9979 best-subtree=0.9979 knob prefetch on k_grouped: 256x256x32 sg8x4 pf3 wgs0 -> 256x256x32 sg8x4 pf2 wgs0
    n015 [llm] N=4 score=1.0090 best-subtree=1.0315 I added tail balancing to the M-grouped kernel (forward and dX). At launch, each work-grou
      n022 [llm] N=2 score=1.0315 best-subtree=1.0315 I added cross-tile prefetch pipelining to `gemm_tile`, used by the forward, dX and grad_we
        n048 [llm] N=1 score=1.0021 best-subtree=1.0021 Edge tiles now run as half-size tiles. In both kernels, an output tile whose valid columns
      n032 [llm] N=1 score=1.0089 best-subtree=1.0089 Some tiles in the last N column are at most half real columns. For N = 1408 with 256-wide 
    n041 [knob] N=1 score=0.8787 best-subtree=0.8787 knob prefetch on m_rowB: 512x128x32 sg16x2 pf3 wgs0 -> 512x128x32 sg16x2 pf1 wgs0
  n005 [llm] N=6 score=0.9708 best-subtree=1.0511 I added a cross-tile prefetch to both kernels. It hasn't been compiled or benchmarked; the
    n010 [llm] N=1 score=0.4531 best-subtree=0.4531 Sub-groups whose whole 32x64 accumulator block sits outside D now skip their A/B block loa
    n021 [llm] N=3 score=1.0279 best-subtree=1.0511 I removed the parent's cross-tile prefetch, which cost about 3% on every op, and added a s
      n031 [llm] N=1 score=1.0511 best-subtree=1.0511 In the forward and dX kernels (M-grouped only), sub-groups whose whole 32-row block lies p
      n040 [llm] N=1 score=0.4080 best-subtree=0.4080 In the shared `gemm_tile` mainloop, each sub-group now works out once per tile (from the M
    n045 [llm] N=1 score=1.0062 best-subtree=1.0062 I removed the parent's cross-tile prefetch, which cost about 3% on every op. In the M-grou
  n006 [llm] N=5 score=0.9403 best-subtree=1.0118 I added a cross-tile prefetch pipeline to `grouped_mm_kernel_impl.h`; it has not been comp
    n011 [llm] N=1 score=0.9357 best-subtree=0.9357 I moved the parent's prefetch of the next tile out of the DPAS mainloop. It now runs once,
    n028 [llm] N=3 score=1.0118 best-subtree=1.0118 I made the parent's cross-tile prefetch pipeline a per-kind compile-time option, `Cfg::kCr
      n034 [llm] N=1 score=0.9842 best-subtree=0.9842 I added a low-register cross-tile prefetch for grad_weight only. It is a new `gemm_tile_ta
      n043 [llm] N=1 score=0.4139 best-subtree=0.4139 I changed only the mainloop path that is actually built. Each sub-group now checks whether
  n007 [llm] N=9 score=1.0062 best-subtree=1.0872 I added cross-tile prefetch to `gemm_tile`, in `grouped_mm_kernel_impl.h` only (configs.js
    n013 [knob] N=5 score=0.9873 best-subtree=1.0872 knob prefetch on m_colB: 256x256x32 sg8x4 pf3 wgs0 -> 256x256x32 sg8x4 pf1 wgs0
      n024 [llm] N=3 score=1.0651 best-subtree=1.0872 In `gemm_tile`, the output now uses an explicit 8x32 bf16 store atom (`XE_STORE_2D<16, 8, 
        n037 [llm] N=1 score=1.0725 best-subtree=1.0725 Only the forward and dX kernels (`MGroupedKernel`) changed. Their A operand is row-major, 
        n046 [llm] N=1 score=1.0872 best-subtree=1.0872 I added a device-side wave-tail split to both persistent kernels in `grouped_mm_kernel_imp
      n033 [knob] N=1 score=0.9707 best-subtree=0.9707 knob tile on k_grouped: 256x256x32 sg8x4 pf3 wgs0 -> 128x256x32 sg4x4 pf3 wgs0
    n012 [llm] N=2 score=0.9490 best-subtree=0.9490 I added an opt-in register-pipelined mainloop to `gemm_tile`, switched on by a new Cfg kno
      n049 [knob] N=1 score=0.9336 best-subtree=0.9336 knob prefetch on m_colB: 256x256x32 sg8x4 pf3 wgs0 -> 256x256x32 sg8x4 pf1 wgs0
    n035 [llm] N=1 score=0.4494 best-subtree=0.4494 In `gemm_tile` (`grouped_mm_kernel_impl.h`), sub-groups whose whole accumulator block lies
  n008 [llm] N=1 score=0.4693 best-subtree=0.4693 Sub-groups whose 32x64 accumulator slab lies entirely outside D now skip their A/B block l
  n002 [llm] N=7 score=1.0216 best-subtree=1.0358 I added cross-tile prefetch to `gemm_tile` and to both persistent kernels. Each work-group
    n018 [llm] N=3 score=1.0358 best-subtree=1.0358 In the M-grouped kernel (forward and grad_input), each work-group now counts the total til
      n026 [knob] N=1 score=1.0175 best-subtree=1.0175 knob tile on k_grouped: 256x256x32 sg8x4 pf3 wgs0 -> 512x128x32 sg16x2 pf3 wgs0
      n042 [llm] N=1 score=0.4177 best-subtree=0.4177 In `gemm_tile`, which all three kernel kinds share, a sub-group now skips its block loads,
    n014 [llm] N=2 score=1.0097 best-subtree=1.0097 I changed only `grouped_mm_kernel_impl.h`. Both persistent kernels now count the total til
      n027 [llm] N=1 score=0.4089 best-subtree=0.4089 In `gemm_tile`, each sub-group now checks once per tile whether its whole accumulator bloc
    n030 [llm] N=1 score=0.8199 best-subtree=0.8199 I added a per-kind switch, `Cfg::kSyncMainloop`. It is read through a `SyncMainloopOf` tra
  n001 [llm] N=8 score=0.9890 best-subtree=1.0073 I added a cross-tile prefetch pipeline to `gemm_tile`, used by both `MGroupedKernel` and `
    n016 [knob] N=2 score=0.9860 best-subtree=0.9928 knob prefetch on k_grouped: 256x256x32 sg8x4 pf3 wgs0 -> 256x256x32 sg8x4 pf4 wgs0
      n029 [llm] N=1 score=0.9928 best-subtree=0.9928 In `gemm_tile`, the prefetch state for the next tile (the tensors and prefetch atoms for A
    n019 [llm] N=3 score=1.0073 best-subtree=1.0073 In `gemm_tile` I moved the next-tile prefetch setup (tensors, prefetch descriptors and par
      n023 [llm] N=1 score=1.0052 best-subtree=1.0052 In `gemm_tile` I moved the fp32-to-bf16 accumulator conversion (`reorder(tCrD, tCrD_final)
      n039 [llm] N=1 score=0.9756 best-subtree=0.9756 I replaced the serial group walk in `Scheduler::locate` with a lookup table spread across 
    n036 [knob] N=2 score=0.9651 best-subtree=0.9908 knob wgs on m_colB: 256x256x32 sg8x4 pf3 wgs0 -> 256x256x32 sg8x4 pf3 wgs112
      n050 [llm] N=1 score=0.9908 best-subtree=0.9908 I added a compile-time `CrossTile` flag to `gemm_tile` in grouped_mm_kernel_impl.h. The cr
  n003 [llm] N=6 score=1.0085 best-subtree=1.0473 I added cross-tile prefetch pipelining to the persistent loop. Each work-group now works o
    n017 [llm] N=3 score=1.0166 best-subtree=1.0473 I added wave-tail splitting to the M-grouped (fwd/dX) persistent scheduler; it has not bee
      n025 [llm] N=1 score=1.0087 best-subtree=1.0087 I extended the wave-tail splitting to the grad_weight kernel (it had only covered fwd/dX),
      n038 [llm] N=1 score=1.0473 best-subtree=1.0473 I made sub-groups whose whole 32x64 accumulator block lies outside D skip their A/B block 
    n020 [llm] N=1 score=0.8469 best-subtree=0.8469 I added a per-kind mainloop variant behind a new Cfg knob, `kLoadAheadA`, and turned it on
    n047 [knob] N=1 score=0.4523 best-subtree=0.4523 knob sg on k_grouped: 256x256x32 sg8x4 pf3 wgs0 -> 256x256x32 sg4x4 pf3 wgs0
```

Widening at the end: alpha = 0.490 (root visits 51).
