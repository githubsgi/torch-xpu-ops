# Xe-Forge run opus1-20261104

- workdir: `/home/subratag/aimod/ggemm_agent_runs/xe_forge/opus1-20261104`  |  phase: done  |  wall time: 2391 s
- model `opus`, suite `quick` (stage gate), final suite `full`, best-of 3 x CoVeR 5, keep gate 1.02x, LLM cap $80.0
- LLM cost: **$5.09** total; by purpose: analysis $0.34, planning $0.09, algorithmic $1.85, memory_access $1.14, persistent_kernel $1.19, autotuning $0.47
- baseline t0 (production vs production, quick): 1.004x (this is the noise level)
- **final: trial t17 on suite `full`: status ok, score 1.184x vs production**

## Analysis

Small-group regime (tp4ep4, ~190 rows/expert): the M-grouped kinds are tile-fill and wave limited, not latency limited. XVE idle is 21% (fwd) and 25% (dX), XMX is 68% and SBID only 6-7%. w13 gives 16x11=176 tiles for 56 WGs, about 3.1 waves, so the last wave is mostly empty, and each tile is only ~74% full of rows. k_grouped (w13_dW+w2_dW = 43% of the tp4ep4 case time) is latency/prologue bound: stall 38%, SBID 34% of samples, XMX 37%, 94 TF, with ~6 K tiles per output tile and a 256x256 store per tile (HBM wr 172 GB/s). Large-group regime (ep8, ~6k rows): all kinds are mostly busy (XVE active 84-92%) and run throttled at ~940 MHz versus 1298 MHz at tp4ep4. The remaining gaps are forward's B-operand load path (159 TF/XMX 84% versus dX 177 TF/XMX 90%), K-grouped transposed A loads (XMX 77%), and a nearly empty 25th M-tile per group (6158 rows = 24x256+14), which wastes ~4% of tiles.

| severity | issue | stage | kinds | evidence (measured) | est. (unmeasured) |
|---|---|---|---|---|---|
| 5 | short_k_mainloop | algorithmic | k_grouped | tp4ep4 w13_dW: 94.1 TF, XVE active 52% stall 38%, XMX util 37%, stall samples Sbid 34%, rule hit load_latency. ep8 w13_dW of the same kernel: XMX 77%, Sbid 3%,  | 1.02-1.05x |
| 4 | last_wave_underfill | persistent_kernel | m_rowB,m_colB | tp4ep4 fwd XVE idle 21%, dX idle 25%, XMX 68%, Sbid only 6-7%, i.e. idle time rather than stalls. w13: 16 groups x 11 n-tiles = 176 tiles / 56 WGs = 3.14 waves  | 1.02-1.05x |
| 3 | prefetch_distance | memory_access | k_grouped | tp4ep4 dW: Sbid 34% with prefetch 3 and only ~6 K tiles per tile, so half the K loop is prologue. ep8 dW (190+ K tiles) has Sbid 3%, so a shorter distance shoul | 1.00-1.02x |
| 3 | copy_atom_selection | memory_access | k_grouped | The K-grouped A/B are col-major views (contraction over rows), so A goes through the 32-bit-pair transpose path. ep8 dW reaches XMX 77% / 164 TF versus dX XMX 9 | 1.01-1.04x |
| 3 | tile_padding_waste | algorithmic | m_rowB,m_colB | tp4ep4: 171-208 rows per group fill a 256-row tile 67-81% (fwd 125-131 TF versus ep8 159 TF). ep8: 6048-6169 rows = 23-24 full tiles plus a remainder of 160 dow | 1.01-1.04x |
| 3 | transposed_or_vnni_load | memory_access | m_rowB | ep8 w13_fwd 159 TF, XMX 84%, Sbid 9%, versus w13_dX 177 TF, XMX 90%, Sbid 6% on the same FLOPs. In tp4ep4 fwd Sbid is 7% versus dX 6%. The K-contiguous B ('R')  | 1.01-1.03x |
| 3 | tile_config_per_kind | autotuning | k_grouped | tp4ep4 dW occupancy 45%, stall 38% with one 512-thread WG per Xe core, so when the WG is in its prologue or epilogue the core has nothing else to run. 1408 outp | 1.00-1.04x |
| 2 | k_tile_size | memory_access | m_rowB,m_colB | ep8 fwd: barrier msgs 4.37M, Sbid 9%, XMX 84%. K=2048/1408 gives 64/44 K iterations of 32 per tile, each with a WG barrier plus 2 copies and 2 prefetches. | 1.00-1.02x |
| 2 | raster_order_l3_reuse | persistent_kernel | m_rowB,m_colB,k_grouped | ep8: L3 rd 1534-1609 GB/s, HBM rd ~200 GB/s, clock throttled to 937-949 MHz (versus 1298 MHz at tp4ep4), so the kernel is power-limited and moving fewer bytes m | 1.00-1.02x |
| 2 | scheduler_overhead | algorithmic | k_grouped,m_rowB,m_colB | locate() uses int64 % and / by runtime tiles_m per tile. In tp4ep4 dW there are 1408 tiles (w13) of only ~6 K iterations each, and XVE active 52% versus XMX 37% | 1.00-1.01x |
| 2 | epilogue_store | memory_access | k_grouped | tp4ep4 w13_dW: HBM wr 172 GB/s (versus 1-19 GB/s for M-grouped), because every ~6-K-tile mainloop is followed by a full 256x256 bf16 store with height-8 store b | 1.00-1.02x |

## Plan

Order after dependency enforcement: `algorithmic -> memory_access -> persistent_kernel -> autotuning`

Algorithmic goes first. The top issue (short_k_mainloop, 5/5) and tile_padding_waste are both in this stage. k_grouped dW takes 43% of tp4ep4 time and is bound by prologue and latency: Sbid 34%, XMX 37%, 94 TF. Fixing tile fill (about 74% at ~190 rows) and scheduler overhead also changes how work is split up, which the later stages depend on. Memory_access comes next: prefetch distance for short K loops, the transposed-A and VNNI load path for K-grouped, the forward B-operand gap (159 vs 177 TF), and the dW epilogue store. Persistent_kernel then fixes the last-wave underfill (176 tiles on 56 WGs, ~3.1 waves) and the raster order for L3 reuse. The tile count only settles once the algorithmic tiling is done, so this stage must come after it. Autotuning goes last to sweep per-kind tile, sg, prefetch and wgs settings on the final code. Discovery and device_specific are left out because they are not in the available stages and no issues were detected for them. device_specific would only have been a soft link between persistent_kernel and autotuning.

## Stages

| stage | result | start | trial | score (start -> end) | iterations | evaluated attempts | cost | time |
|---|---|---|---|---|---|---|---|---|
| algorithmic | accepted | t0 | t4 | 1.004 -> 1.028 | 3 | 4 | $1.85 | 724s |
| memory_access | accepted | t4 | t6 | 1.028 -> 1.056 | 3 | 3 | $1.14 | 373s |
| persistent_kernel | accepted | t6 | t10 | 1.056 -> 1.094 | 3 | 3 | $1.19 | 538s |
| autotuning | accepted | t10 | t17 | 1.094 -> 1.132 | 3 | 18 | $0.47 | 547s |

### algorithmic: accepted
The prefetch pipeline now runs across tiles in both persistent loops. Each loop finds the next tile (`locate(lin+nwg)` on a copy of the cursor) before running the current one. In the last `prefetch_dist` mainloop iterations, where the curre
- iter 1: t1 0.994x < 1.02 x 1.004x: not kept
- iter 2: t3 1.007x < 1.02 x 1.004x: not kept
- iter 3: new stage best t4 1.028x (was 1.004x)

### memory_access: accepted
In `gemm_tile`, the epilogue store now uses an explicit `XE_STORE_2D<16, 8, 32>` atom (8 rows by 64 bytes, the widest legal 2D store) instead of the auto-selected store. The selector most likely picks 8x16, but I worked that out from the se
- iter 1: t5 1.019x < 1.02 x 1.028x: not kept
- iter 2: new stage best t6 1.056x (was 1.028x)
- iter 3: t7 1.062x < 1.02 x 1.056x: not kept

### persistent_kernel: accepted
In MGroupedKernel (fwd and dX), each work-group now counts all tiles from `offs` on the device. If the last partial wave has r tiles, those r tiles are split along N into 4r sub-tiles of 256x64 when 4r ≤ nwg, or into 2r sub-tiles of 256x128
- iter 1: t8 1.076x < 1.02 x 1.056x: not kept
- iter 2: t9 1.058x < 1.02 x 1.056x: not kept
- iter 3: new stage best t10 1.094x (was 1.056x)

### autotuning: accepted
k_grouped {"tile":[256,256,16],"sg":[8,4],"prefetch":4,"wgs":0} (k_grouped spills 256 B/thread, and K=64 collapsed it to 0.40x, so A/B fragment size is what drives the spill. Halving th)
- round 1: no variant beat the base by 1.02x on its kind
- round 2: new stage best t17 1.132x (was 1.094x)
- round 3: no variant beat the base by 1.02x on its kind

## Accepted diffs

### algorithmic: t0 -> t4

```diff
--- a/grouped_mm_kernel_impl.h
+++ b/grouped_mm_kernel_impl.h
@@ -138,13 +138,19 @@
     class BTensor,
     class DTensor,
     class TiledMMA>
-CUTE_DEVICE void gemm_tile(
+CUTE_DEVICE int gemm_tile(
     ATensor const& A,
     BTensor const& B,
     DTensor& D,
     int m_idx,
     int n_idx,
-    TiledMMA const& mma) {
+    TiledMMA const& mma,
+    int pre_issued, // leading K tiles already prefetched by the previous tile
+    ATensor const& An, // operands of this work-group's next tile (prefetch)
+    BTensor const& Bn,
+    int mn_idx,
+    int nn_idx,
+    bool has_next) {
   auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
   auto local_id = item.get_local_linear_id();
 
@@ -190,10 +196,30 @@
   auto pAgA = thr_prefetch_A.partition_S(gA);
   auto pBgB = thr_prefetch_B.partition_S(gB);
 
+  // Prefetch view of the next tile: the prefetch pipeline continues across
+  // tile boundaries, so the next tile's first K tiles are requested during
+  // this tile's last mainloop iterations (not after its store).
+  Tensor cAn = make_identity_tensor(An.shape());
+  Tensor cBn = make_identity_tensor(Bn.shape());
+  Tensor gAn = local_tile(cAn, select<0, 2>(wg_tile), make_coord(mn_idx, _));
+  Tensor gBn = local_tile(cBn, select<1, 2>(wg_tile), make_coord(nn_idx, _));
+  auto prefetch_an =
+      make_block_2d_prefetch(get_block_2d_copy_A<void>(mma, An));
+  auto prefetch_bn =
+      make_block_2d_prefetch(get_block_2d_copy_B<void>(mma, Bn));
+  auto pAgAn = prefetch_an.get_slice(local_id).partition_S(gAn);
+  auto pBgBn = prefetch_bn.get_slice(local_id).partition_S(gBn);
+
   constexpr int barrier_scope = 2;
   constexpr int prefetch_dist = PrefetchDist;
   const int k_tile_count = ceil_div(shape<1>(A), get<2>(wg_tile));
-  int prefetch_k = 0;
+  int next_limit = 0;
+  if (has_next) {
+    const int next_k_count = ceil_div(shape<1>(An), get<2>(wg_tile));
+    next_limit = next_k_count < prefetch_dist ? next_k_count : prefetch_dist;
+  }
+  int next_issued = 0;
+  int prefetch_k = pre_issued < k_tile_count ? pre_issued : k_tile_count;
 
   clear(tCrD);
 
@@ -215,6 +241,10 @@
     if (prefetch_k < k_tile_count) {
       prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
       prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
+    } else if (next_issued < next_limit) {
+      prefetch(prefetch_an, pAgAn(_, _, _, next_issued));
+      prefetch(prefetch_bn, pBgBn(_, _, _, next_issued));
+      ++next_issued;
     }
 
     reorder(tArA, tCrA);
@@ -226,6 +256,7 @@
 
   reorder(tCrD, tCrD_final);
   copy(tiled_copy_d, tCrD_final, tCgD);
+  return next_issued;
 }
 
 // Linearized, device-side tile scheduler over all groups. Every work-group
@@ -299,18 +330,32 @@
     T* b = const_cast<T*>(static_cast<const T*>(p.b));
     T* d = static_cast<T*>(p.d);
 
-    for (int64_t lin = wg;; lin += nwg) {
-      int32_t m_idx, n_idx;
-      if (!sched.locate(lin, cur, m_idx, n_idx)) {
-        break;
-      }
-      auto A = make_2d_tensor<'R'>(
-          a + int64_t(cur.row_start) * p.lda, cur.rows, p.K, int(p.lda));
-      auto B = make_2d_tensor<LayoutB>(
-          b + int64_t(cur.g) * p.group_stride_b, p.N, p.K, int(p.ldb));
+    auto make_a = [&](const GroupCursor& c) {
+      return make_2d_tensor<'R'>(
+          a + int64_t(c.row_start) * p.lda, c.rows, p.K, int(p.lda));
+    };
+    auto make_b = [&](const GroupCursor& c) {
+      return make_2d_tensor<LayoutB>(
+          b + int64_t(c.g) * p.group_stride_b, p.N, p.K, int(p.ldb));
+    };
+
+    int32_t m_idx = 0, n_idx = 0;
+    bool valid = sched.locate(wg, cur, m_idx, n_idx);
+    int pre_issued = 0;
+    for (int64_t lin = wg; valid; lin += nwg) {
+      GroupCursor nxt = cur;
+      int32_t mn_idx = 0, nn_idx = 0;
+      const bool has_next = sched.locate(lin + nwg, nxt, mn_idx, nn_idx);
+      const GroupCursor& pc = has_next ? nxt : cur;
       auto D = make_2d_tensor<'R'>(
           d + int64_t(cur.row_start) * p.ldd, cur.rows, p.N, int(p.ldd));
-      gemm_tile<Cfg::kPrefetchDist>(A, B, D, m_idx, n_idx, mma);
+      pre_issued = gemm_tile<Cfg::kPrefetchDist>(
+          make_a(cur), make_b(cur), D, m_idx, n_idx, mma, pre_issued,
+          make_a(pc), make_b(pc), mn_idx, nn_idx, has_next);
+      cur = nxt;
+      m_idx = mn_idx;
+      n_idx = nn_idx;
+      valid = has_next;
     }
   }
 };
@@ -334,21 +379,35 @@
     T* b = const_cast<T*>(static_cast<const T*>(p.b));
     T* d = static_cast<T*>(p.d);
 
-    for (int64_t lin = wg;; lin += nwg) {
-      int32_t m_idx, n_idx;
-      if (!sched.locate(lin, cur, m_idx, n_idx)) {
-        break;
-      }
-      // A: (M, rows) with element (m, r) at m + r * lda  -> column-major.
-      auto A = make_2d_tensor<'C'>(
-          a + int64_t(cur.row_start) * p.lda, p.M, cur.rows, int(p.lda));
-      // B as (N, rows) with element (n, r) at n + r * ldb -> column-major
-      // view of the row-major [rows, N] operand.
-      auto B = make_2d_tensor<'C'>(
-          b + int64_t(cur.row_start) * p.ldb, p.N, cur.rows, int(p.ldb));
+    // A: (M, rows) with element (m, r) at m + r * lda  -> column-major.
+    auto make_a = [&](const GroupCursor& c) {
+      return make_2d_tensor<'C'>(
+          a + int64_t(c.row_start) * p.lda, p.M, c.rows, int(p.lda));
+    };
+    // B as (N, rows) with element (n, r) at n + r * ldb -> column-major
+    // view of the row-major [rows, N] operand.
+    auto make_b = [&](const GroupCursor& c) {
+      return make_2d_tensor<'C'>(
+          b + int64_t(c.row_start) * p.ldb, p.N, c.rows, int(p.ldb));
+    };
+
+    int32_t m_idx = 0, n_idx = 0;
+    bool valid = sched.locate(wg, cur, m_idx, n_idx);
+    int pre_issued = 0;
+    for (int64_t lin = wg; valid; lin += nwg) {
+      GroupCursor nxt = cur;
+      int32_t mn_idx = 0, nn_idx = 0;
+      const bool has_next = sched.locate(lin + nwg, nxt, mn_idx, nn_idx);
+      const GroupCursor& pc = has_next ? nxt : cur;
       auto D = make_2d_tensor<'R'>(
           d + int64_t(cur.g) * p.group_stride_d, p.M, p.N, int(p.ldd));
-      gemm_tile<Cfg::kPrefetchDist>(A, B, D, m_idx, n_idx, mma);
+      pre_issued = gemm_tile<Cfg::kPrefetchDist>(
+          make_a(cur), make_b(cur), D, m_idx, n_idx, mma, pre_issued,
+          make_a(pc), make_b(pc), mn_idx, nn_idx, has_next);
+      cur = nxt;
+      m_idx = mn_idx;
+      n_idx = nn_idx;
+      valid = has_next;
     }
   }
 };

```

### memory_access: t4 -> t6

```diff
--- a/grouped_mm_kernel_impl.h
+++ b/grouped_mm_kernel_impl.h
@@ -169,7 +169,24 @@
 
   auto tiled_copy_a = get_block_2d_copy_A<void>(mma, A);
   auto tiled_copy_b = get_block_2d_copy_B<void>(mma, B);
-  auto tiled_copy_d = get_block_2d_copy_D<void>(mma, D);
+  // Epilogue store: the auto selector picks block width = one DPAS N atom
+  // (16 elements), i.e. 8x16 messages. A 2D store may be 64 bytes wide, so
+  // use 8 x (512 / bits) blocks when each sub-group's N slab allows it:
+  // half the store messages per tile (the reorder into the store fragment
+  // happens anyway for the fp32 -> bf16 conversion).
+  using DElem = typename DTensor::value_type;
+  constexpr int kDBits = int(sizeof_bits_v<DElem>);
+  constexpr int kStoreW = 512 / kDBits;
+  using MmaTileN = std::decay_t<decltype(get<1>(
+      std::declval<TiledMMA const&>().tile_mnk()))>;
+  using MmaThrN = std::decay_t<decltype(size<2>(
+      std::declval<TiledMMA const&>().get_thr_layout_vmnk()))>;
+  constexpr int kSgN = int(MmaTileN::value) / int(MmaThrN::value);
+  using StoreOp = std::conditional_t<
+      (kSgN % kStoreW == 0),
+      XE_STORE_2D<kDBits, 8, kStoreW>,
+      void>;
+  auto tiled_copy_d = get_block_2d_copy_D<StoreOp>(mma, D);
 
   auto thr_copy_a = tiled_copy_a.get_slice(local_id);
   auto thr_copy_b = tiled_copy_b.get_slice(local_id);

```

### persistent_kernel: t6 -> t10

```diff
--- a/grouped_mm_kernel_impl.h
+++ b/grouped_mm_kernel_impl.h
@@ -356,13 +356,37 @@
           b + int64_t(c.g) * p.group_stride_b, p.N, p.K, int(p.ldb));
     };
 
+    // Device-side wave-tail balancing (no scratch, only offs): count all
+    // tiles; if the last partial wave holds r tiles and S*r <= nwg, those r
+    // tiles are run as S*r narrower (TileN/S) sub-tiles, one per work-group,
+    // so the final wave takes ~1/S of a full tile instead of a full tile
+    // with most of the GPU idle. S = 4 when it fits, else 2.
+    int64_t total = 0;
+    for (int32_t g = 0; g < p.num_groups; ++g) {
+      GroupCursor c;
+      c.g = g;
+      sched.describe(c);
+      total += c.tiles;
+    }
+    const int64_t rem = total % nwg;
+    int split = 1;
+    if (rem > 0) {
+      if (kSplit4Ok && 4 * rem <= nwg) {
+        split = 4;
+      } else if (kSplit2Ok && 2 * rem <= nwg) {
+        split = 2;
+      }
+    }
+    const int64_t full_end = (split > 1) ? (total - rem) : total;
+
     int32_t m_idx = 0, n_idx = 0;
-    bool valid = sched.locate(wg, cur, m_idx, n_idx);
+    bool valid = (int64_t(wg) < full_end) && sched.locate(wg, cur, m_idx, n_idx);
     int pre_issued = 0;
     for (int64_t lin = wg; valid; lin += nwg) {
       GroupCursor nxt = cur;
       int32_t mn_idx = 0, nn_idx = 0;
-      const bool has_next = sched.locate(lin + nwg, nxt, mn_idx, nn_idx);
+      const bool has_next = (lin + nwg < full_end) &&
+          sched.locate(lin + nwg, nxt, mn_idx, nn_idx);
       const GroupCursor& pc = has_next ? nxt : cur;
       auto D = make_2d_tensor<'R'>(
           d + int64_t(cur.row_start) * p.ldd, cur.rows, p.N, int(p.ldd));
@@ -374,6 +398,52 @@
       n_idx = nn_idx;
       valid = has_next;
     }
+
+    if (split > 1 && int64_t(wg) < int64_t(split) * rem) {
+      GroupCursor c;
+      sched.describe(c);
+      int32_t tm = 0, tn = 0;
+      const int64_t lin = full_end + int64_t(wg) / split;
+      const int q = int(int64_t(wg) % split);
+      if (sched.locate(lin, c, tm, tn)) {
+        auto A = make_a(c);
+        auto B = make_b(c);
+        auto D = make_2d_tensor<'R'>(
+            d + int64_t(c.row_start) * p.ldd, c.rows, p.N, int(p.ldd));
+        if constexpr (kSplit4Ok) {
+          if (split == 4) {
+            run_sub<4>(A, B, D, tm, tn * 4 + q);
+            return;
+          }
+        }
+        if constexpr (kSplit2Ok) {
+          run_sub<2>(A, B, D, tm, tn * 2 + q);
+        }
+      }
+    }
+  }
+
+  static constexpr int kThrN =
+      int(decltype(size<1>(typename Cfg::SGLayout{}))::value);
+  // A sub-tile needs >= one 16-wide DPAS N atom per sub-group.
+  static constexpr bool kSplit2Ok = (Cfg::kTileN % (2 * 16 * kThrN)) == 0;
+  static constexpr bool kSplit4Ok = (Cfg::kTileN % (4 * 16 * kThrN)) == 0;
+
+  template <int S>
+  using SubMma = typename TiledMMAHelper<
+      MMA_Atom<XE_DPAS_TT<8, float, T, T>>,
+      Layout<Shape<Int<Cfg::kTileM>, Int<Cfg::kTileN / S>, Int<Cfg::kTileK>>>,
+      typename Cfg::SGLayout>::TiledMMA;
+
+  template <int S, class AT, class BT, class DT>
+  CUTE_DEVICE void run_sub(
+      AT const& A, BT const& B, DT& D, int m_idx, int sub_n) const {
+    if (int64_t(sub_n) * (Cfg::kTileN / S) >= int64_t(p.N)) {
+      return; // sub-tile fully past N (ragged N): nothing to compute/store
+    }
+    const auto smma = SubMma<S>{};
+    gemm_tile<Cfg::kPrefetchDist>(
+        A, B, D, m_idx, sub_n, smma, 0, A, B, m_idx, sub_n, false);
   }
 };
 

```

### autotuning: t10 -> t17

```diff
--- a/configs.json
+++ b/configs.json
@@ -1,4 +1,5 @@
 {
+  "extra_flags": [],
   "m_rowB": {
     "tile": [
       256,
@@ -29,14 +30,13 @@
     "tile": [
       256,
       256,
-      32
+      16
     ],
     "sg": [
       8,
       4
     ],
-    "prefetch": 3,
+    "prefetch": 4,
     "wgs": 0
-  },
-  "extra_flags": []
+  }
 }

```

## Final evaluation

Suite `full`, trial t17 (production = 1.000x):

| case | speedup |
|---|---|
| tp4ep4_uniform | 1.203x |
| tp4ep4_balanced | 1.194x |
| tp4ep4_hot | 1.179x |
| tp4ep4_sparse | 1.374x |
| tp4ep4_rows64 | 1.250x (probe, not scored) |
| tp4ep4_rows8 | 1.266x (probe, not scored) |
| tp8ep8_uniform | 1.279x |
| tp8ep8_balanced | 1.263x |
| tp8ep8_hot | 1.172x |
| tp8ep8_sparse | 1.312x |
| tp8ep8_rows64 | 1.313x (probe, not scored) |
| ep8_uniform | 1.038x |
| ep8_balanced | 1.037x |
| ep8_hot | 1.035x |
| ep8_rows1024 | 1.083x (probe, not scored) |

Per op (geomean): w13_fwd 1.039x, w2_fwd 1.137x, w13_dX 1.143x, w2_dX 1.061x, w13_dW 1.412x, w2_dW 1.391x

## Trial tree

```
t0 [baseline i0a0] 1.004x production kernel
  t1 [algorithmic i0a1] 0.994x The persistent loop now works out the next tile one step early by call
  t2 [algorithmic i1a1] 0.450x In `gemm_tile`, each sub-group checks once per tile, using its own D-f
  t3 [algorithmic i1a2] 1.007x I took out last attempt's per-sub-group skip and put the original `gem
  t4 [algorithmic i2a1] 1.028x The prefetch pipeline now runs across tiles in both persistent loops. 
    t5 [memory_access i0a1] 1.019x In `cand/configs.json`, raised the k_grouped (grad_weight) prefetch di
    t6 [memory_access i1a1] 1.056x In `gemm_tile`, the epilogue store now uses an explicit `XE_STORE_2D<1
      t7 [memory_access i2a1] 1.062x In `cand/configs.json`, lowered the k_grouped (grad_weight) prefetch d
      t8 [persistent_kernel i0a1] 1.076x In `MGroupedKernel` (fwd and dX), each work-group now counts the total
      t9 [persistent_kernel i1a1] 1.058x Within each group, `Scheduler::locate` now splits the n tiles into nea
      t10 [persistent_kernel i2a1] 1.094x In MGroupedKernel (fwd and dX), each work-group now counts all tiles f
        t11 [autotuning i0a1] 1.070x k_grouped {"tile":[128,256,32],"sg":[4,4],"prefetch":3,"wgs":0} (In tp
        t12 [autotuning i0a2] 1.070x k_grouped {"tile":[256,128,32],"sg":[8,2],"prefetch":3,"wgs":0} (Same 
        t13 [autotuning i0a3] 1.059x k_grouped {"tile":[128,256,32],"sg":[4,4],"prefetch":2,"wgs":0} (At ~1
        t14 [autotuning i0a4] 0.685x k_grouped {"tile":[256,256,64],"sg":[8,4],"prefetch":2,"wgs":0} (K=64 
        t15 [autotuning i0a5] 1.074x m_rowB {"tile":[256,256,32],"sg":[8,4],"prefetch":4,"wgs":0} (The forw
        t16 [autotuning i0a6] 0.766x m_colB {"tile":[256,256,64],"sg":[8,4],"prefetch":2,"wgs":0} (dX has 0
        t17 [autotuning i1a1] 1.132x k_grouped {"tile":[256,256,16],"sg":[8,4],"prefetch":4,"wgs":0} (k_gro
          t23 [autotuning i2a1] 1.102x k_grouped {"tile":[256,256,16],"sg":[8,4],"prefetch":5,"wgs":0} (K16 w
          t24 [autotuning i2a2] 1.104x k_grouped {"tile":[256,256,16],"sg":[8,4],"prefetch":3,"wgs":0} (Short
          t25 [autotuning i2a3] 1.087x k_grouped {"tile":[128,256,16],"sg":[4,4],"prefetch":4,"wgs":0} (128x2
          t26 [autotuning i2a4] 1.065x m_rowB {"tile":[256,256,16],"sg":[8,4],"prefetch":4,"wgs":0} (The K16 
          t27 [autotuning i2a5] 1.056x m_rowB {"tile":[256,256,16],"sg":[8,4],"prefetch":5,"wgs":0} (Same as 
          t28 [autotuning i2a6] 1.078x m_colB {"tile":[256,256,16],"sg":[8,4],"prefetch":4,"wgs":0} (The K16 
        t18 [autotuning i1a2] 1.075x k_grouped {"tile":[256,256,32],"sg":[4,8],"prefetch":3,"wgs":0} (A 64x
        t19 [autotuning i1a3] 1.084x k_grouped {"tile":[256,256,32],"sg":[8,4],"prefetch":2,"wgs":0} (tp4ep
        t20 [autotuning i1a4] 1.083x m_rowB {"tile":[256,256,32],"sg":[8,4],"prefetch":2,"wgs":0} (Prefetch
        t21 [autotuning i1a5] 1.092x m_colB {"tile":[256,256,32],"sg":[8,4],"prefetch":4,"wgs":0} (dX is th
        t22 [autotuning i1a6] 1.081x m_rowB {"tile":[256,256,32],"sg":[4,8],"prefetch":3,"wgs":0} (Forward 
```

Trial state: `/home/subratag/aimod/ggemm_agent_runs/xe_forge/opus1-20261104/trials/state.json`; lessons: `/home/subratag/aimod/ggemm_agent_runs/xe_forge/opus1-20261104/lessons.md`; LLM log: `/home/subratag/aimod/ggemm_agent_runs/xe_forge/opus1-20261104/llm_log.jsonl`.
