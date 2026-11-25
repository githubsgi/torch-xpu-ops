# Kernel Forge (MCTS) run report

## Run

- workdir: `/mnt/aimod/subratag/ggemm_agent_runs/kernel_forge/v4knobs-r2-20261125`
- iterations: 150 (target 150); nodes admitted: 150; failed attempts: 0
- actions: knob 64, llm 86; outcomes: ok 150
- LLM revisions: 91 claude sessions (incl. repairs), 5 repair rounds, spend $49.39 (budget $400, model opus)
- search suite `search`, final suite `all`, C=0.1, vloss=0.05, alpha 0.5->0.3 over 1000 root visits, workers 8, knob ratio 0.5
- production vs itself measured on `search`: 1.0103x; on `all`: 1.0110x

## Result

Best node **n027**: 1.0156x on `search`, **1.0157x on `all`** (production re-run: 1.0110x).

Verdict: no significant improvement (< 2% is noise) (+0.5%).

## Lineage of the best node

| step | node | action | change | quick score | vs parent |
|---|---|---|---|---|---|
| 0 | n000 | root | production kernel | 1.0103 |  |
| 1 | n004 | knob | knob sg on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg4x8 pf3 wgs0 be2 si _[m_rowB: sg [8, 4]->[4, 8]; header +0/-0 lines]_ | 1.0133 | +0.3% |
| 2 | n012 | llm | The forward (m_rowB) kernel can split its last partial wave 4 ways again. The parent's 4x8 sub-group layout gives each sub-group 32 columns of a 256-wide tile. A quarter-width (64-column) tail sub-tile would leave each sub-group only 8 columns, below the 16-co _[no config change; header +30/-4 lines]_ | 1.0140 | +0.1% |
| 3 | n027 | knob | knob cache_a on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 ca1 _[k_grouped: cache_a None->1; header +0/-0 lines]_ | 1.0156 | +0.2% |

## Top candidates (search suite, then full re-evaluation)

| node | action | quick | full | parent | note |
|---|---|---|---|---|---|
| n116 | llm | 1.0209 | 0.9141 | n067 | In the grad_weight kernel, the chunked tile order (kChunkOrder) is now chosen on the device. It turns on only  |
| n020 | llm | 1.0189 | 0.9691 | n011 | I added a `kChunkTiles` option to the chunked tile order in `KGroupedKernel` and set it to 4 for k_grouped (gr |
| n034 | llm | 1.0162 | 0.9597 | n020 | In `KGroupedKernel` (grad_weight) I added a `kMergeTailN` option and turned it on for k_grouped. It only kicks |
| n027 | knob | 1.0156 | 1.0157 | n012 | knob cache_a on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 ca1 |
| n131 | llm | 1.0156 | - | n126 | I changed how the M-grouped kernel (forward and dX) finds its tiles; I could not build or benchmark it here. E |
| n037 | llm | 1.0153 | - | n019 | In the grad_weight (K-grouped) kernel, the ragged last N column is now split out when N % TileN is at most Til |
| n030 | llm | 1.0150 | - | n017 | In `KGroupedKernel`'s persistent loop (grad_weight), a tile at the right edge of N with at most half its colum |
| n123 | knob | 1.0146 | - | n102 | knob cache_b on m_rowB: 256x256x32 sg4x8 pf3 wgs0 be2 si -> 256x256x32 sg4x8 pf3 wgs0 be2 si cb2 |

## Per-op speedup of the best node vs production

| op | quick | full |
|---|---|---|
| w13_dW | 1.005x | 1.035x |
| w13_dX | 1.012x | 1.02x |
| w13_fwd | 1.014x | 1.036x |
| w2_dW | 1.045x | 1.063x |
| w2_dX | 1.013x | 1.02x |
| w2_fwd | 1.002x | 1.021x |

Per case (full):

- tp4ep4_uniform: 1.014x
- tp4ep4_balanced: 1.022x
- tp4ep4_hot: 1.014x
- tp4ep4_sparse: 1.012x
- tp4ep4_rows64 (probe, not scored): 1.114x
- tp4ep4_rows8 (probe, not scored): 1.173x
- tp8ep8_uniform: 1.010x
- tp8ep8_balanced: 1.034x
- tp8ep8_hot: 1.017x
- tp8ep8_sparse: 1.013x
- tp8ep8_rows64 (probe, not scored): 1.131x
- ep8_uniform: 1.009x
- ep8_balanced: 1.012x
- ep8_hot: 1.014x
- ep8_rows1024 (probe, not scored): 0.998x
- 671b_tp8ep32_uniform: 1.014x
- 671b_tp8ep32_balanced: 1.012x
- 671b_tp8ep32_hot: 1.024x
- 671b_tp8ep32_sparse: 1.017x
- 671b_tp8ep32_rows128 (probe, not scored): 1.012x
- 671b_tp8ep32_rows2048 (probe, not scored): 1.019x

## Configs and diff vs production

```
configs: m_rowB: sg [8, 4]->[4, 8]; m_colB: barrier_every 2->1; k_grouped: extra None->static constexpr bool kKTailSplit = true; static constexpr bool kNoBarrier = true;, cache_a None->1
```

Header diff:

```diff
--- production/grouped_mm_kernel_impl.h
+++ candidate/grouped_mm_kernel_impl.h
@@ -36,6 +36,162 @@
 
 #pragma clang diagnostic ignored "-Wpass-failed"
 #pragma clang diagnostic ignored "-Wdeprecated-declarations"
+
+// ---------------------------------------------------------------------------
+// LSC cache-hint variants of the Xe 2D block loads / prefetches. sycl-tla's
+// atoms carry no cache modifier (loads use the driver default, prefetches
+// .ca.ca). These wrappers keep the auto-selected atom's shape and register
+// layout and only change the vISA cache modifier:
+//   hint 0 = unchanged, 1 = .ca.ca, 2 = .st.ca (L1 streaming), 3 = .uc.ca
+//   (L1 uncached), 4 = .uc.uc; prefetch hint 0 = .ca.ca, 1 = .uc.ca (L3 only).
+// ---------------------------------------------------------------------------
+namespace cute {
+
+namespace gg_cc {
+template <class Base>
+struct op_kind {
+  static constexpr int value = 0; // plain (nn)
+};
+template <int B, int H, int W, int BW>
+struct op_kind<XE_LOAD_2D_VNNI<B, H, W, BW>> {
+  static constexpr int value = 1; // VNNI (nt)
+};
+template <int B, int H, int W>
+struct op_kind<XE_LOAD_2D_TRANSPOSE<B, H, W>> {
+  static constexpr int value = 2; // transposed (tn)
+};
+template <class Traits>
+struct op_of;
+template <class Op, class X, class Y, class V, class S>
+struct op_of<Copy_Traits<Op, X, Y, V, S>> {
+  using type = Op;
+};
+} // namespace gg_cc
+
+#define GG_CC_LOAD_NN(MOD)                                            \
+  asm("lsc_load_block2d.ugm" MOD                                      \
+      " (M1, 1)  %0:d%2.%3x%4x%5nn flat[%1+(0,0)]"                    \
+      : "=rw"(dv)                                                     \
+      : "rw.u"(payload), "P"(Bits), "P"(Width / BlockWidth),          \
+        "P"(BlockWidth), "P"(Height))
+#define GG_CC_LOAD_NT(MOD)                                            \
+  asm("lsc_load_block2d.ugm" MOD                                      \
+      " (M1, 1)  %0:d%2.%3x%4x%5nt flat[%1+(0,0)]"                    \
+      : "=rw"(dv)                                                     \
+      : "rw.u"(payload), "P"(Bits), "P"(Width / BlockWidth),          \
+        "P"(BlockWidth), "P"(Height))
+#define GG_CC_LOAD_TN(MOD)                                            \
+  asm("lsc_load_block2d.ugm" MOD " (M1, 1)  %0:d%2.%3x%4tn flat[%1+(0,0)]" \
+      : "=rw"(dv)                                                     \
+      : "rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height))
+#define GG_CC_LOAD_ALL(KIND)                                          \
+  if constexpr (Hint == 1) {                                          \
+    GG_CC_LOAD_##KIND(".ca.ca");                                      \
+  } else if constexpr (Hint == 2) {                                   \
+    GG_CC_LOAD_##KIND(".st.ca");                                      \
+  } else if constexpr (Hint == 3) {                                   \
+    GG_CC_LOAD_##KIND(".uc.ca");                                      \
+  } else {                                                            \
+    GG_CC_LOAD_##KIND(".uc.uc");                                      \
+  }
+
+template <class Base, int Hint>
+struct XE_LOAD_2D_CC : Base {
+  static_assert(Hint >= 1 && Hint <= 4, "cache hint 1..4");
+  static constexpr int Bits = Base::CopyBits;
+  static constexpr int Width = Base::AtomWidth;
+  static constexpr int Height = Base::AtomHeight;
+  static constexpr int BlockWidth = Width / Base::BlockCount;
+
+  template <typename T>
+  CUTE_HOST_DEVICE static void copy(const int* payload, T* dst) {
+#ifdef CUTE_ARCH_COPY_XE_ENABLED
+    using namespace intel;
+    auto& dv = *reinterpret_cast<
+        storage_vector_t<T, Width * Height * Bits / sg_size>*>(dst);
+    constexpr int K = gg_cc::op_kind<Base>::value;
+    if constexpr (K == 0) {
+      GG_CC_LOAD_ALL(NN)
+    } else if constexpr (K == 1) {
+      GG_CC_LOAD_ALL(NT)
+    } else {
+      GG_CC_LOAD_ALL(TN)
+    }
+#else
+    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
+#endif
+  }
+
+  using PREFETCH = typename Base::PREFETCH;
+};
+
+template <class Base, int Hint>
+struct XE_PREFETCH_2D_CC : Base {
+  static_assert(Hint >= 0 && Hint <= 1, "prefetch hint 0..1");
+  static constexpr int Bits = Base::CopyBits;
+  static constexpr int Width = Base::AtomWidth;
+  static constexpr int Height = Base::AtomHeight;
+
+  CUTE_HOST_DEVICE static void copy(const int* payload) {
+#ifdef CUTE_ARCH_COPY_XE_ENABLED
+    if constexpr (Hint == 1) {
+      asm("lsc_load_block2d.ugm.uc.ca (M1, 1)  %%null:d%1.%2x%3nn flat[%0+(0,0)]"
+          ::"rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height));
+    } else {
+      asm("lsc_load_block2d.ugm.ca.ca (M1, 1)  %%null:d%1.%2x%3nn flat[%0+(0,0)]"
+          ::"rw.u"(payload), "P"(Bits), "P"(Width), "P"(Height));
+    }
+#else
+    CUTE_INVALID_CONTROL_PATH("Cannot use Xe block 2D copy atom on non-Xe hardware");
+#endif
+  }
+
+  using PREFETCH = XE_PREFETCH_2D_CC;
+};
+
+template <class Base, int Hint>
+struct is_xe_block_2d_atom<XE_LOAD_2D_CC<Base, Hint>> : std::true_type {};
+
+// Traits: same register layouts as the wrapped atom, our copy() routine.
+template <class XMode, class YMode, class ValType, class TiledStrides, class Base, int Hint>
+struct Copy_Traits<XE_LOAD_2D_CC<Base, Hint>, XMode, YMode, ValType, TiledStrides>
+    : Xe2DLoadTraitsBase<XE_LOAD_2D_CC<Base, Hint>, XMode, YMode, ValType, TiledStrides> {
+  using Super = Xe2DLoadTraitsBase<XE_LOAD_2D_CC<Base, Hint>, XMode, YMode, ValType, TiledStrides>;
+  using Super::Super;
+  using BaseTraits = Copy_Traits<Base, XMode, YMode, ValType, TiledStrides>;
+  using DstLayout = typename BaseTraits::DstLayout;
+  using RefLayout = typename BaseTraits::RefLayout;
+  using SrcLayout = typename BaseTraits::SrcLayout;
+};
+
+template <class XMode, class YMode, class ValType, class TiledStrides, class Base, int Hint>
+struct Copy_Traits<XE_PREFETCH_2D_CC<Base, Hint>, XMode, YMode, ValType, TiledStrides>
+    : Xe2DTraitsBase<XE_PREFETCH_2D_CC<Base, Hint>, XMode, YMode, ValType, TiledStrides> {
+  using Op = XE_PREFETCH_2D_CC<Base, Hint>;
+  using BaseTraits = Copy_Traits<Base, XMode, YMode, ValType, TiledStrides>;
+  using DstLayout = typename BaseTraits::DstLayout;
+  using RefLayout = typename BaseTraits::RefLayout;
+  using SrcLayout = typename BaseTraits::SrcLayout;
+  using Super = Xe2DTraitsBase<Op, XMode, YMode, ValType, TiledStrides>;
+  using Traits = typename Super::Traits;
+  using ThrID = typename Super::ThrID;
+  using Super::Super;
+
+  template <class SEngine, class SLayout, class DEngine, class DLayout>
+  CUTE_DEVICE friend constexpr void copy_unpack(
+      Traits const& traits,
+      Tensor<SEngine, SLayout> const& src,
+      Tensor<DEngine, DLayout>& dst) {
+    using SrcLayout = typename Traits::SrcLayout;
+    static_assert(is_counting_layout_v<SLayout>, "Source tensor must be a coordinate tensor.");
+    static_assert(size(SLayout{}) * Super::ValBits == size<1>(SrcLayout{}),
+                  "Source tensor size does not match copy atom size.");
+    traits.template update_payload<Super::ValBits>(src.data().coord_);
+    Op::copy(traits.payload);
+  }
+};
+
+} // namespace cute
 
 namespace sycltla {
 namespace grouped_gemm {
@@ -138,6 +294,23 @@
   static constexpr int value = Cfg::kBarrierEvery;
 };
 
+// Optional Cfg::kNoBarrier (default false): the mainloop issues no
+// work-group barriers at all (gemm_tile BarrierEvery = 0). The barriers only
+// keep the sub-groups in lockstep for L1 reuse (no SLM is shared), so this
+// is purely a scheduling choice: on short K loops (grad_weight, ~12 K tiles
+// per tile) lockstep makes all 32 sub-groups hit the serialized epilogue
+// stores and the next tile's setup at the same moment with no DPAS left to
+// overlap them; free-running sub-groups overlap one's epilogue with
+// another's mainloop.
+template <class Cfg, class = void>
+struct NoBarrierOf {
+  static constexpr bool value = false;
+};
+template <class Cfg>
+struct NoBarrierOf<Cfg, std::void_t<decltype(Cfg::kNoBarrier)>> {
+  static constexpr bool value = bool(Cfg::kNoBarrier);
+};
+
 // Optional Cfg::kSkipIdleSg (default false): a sub-group whose whole
 // accumulator slab lies outside D (ragged last row tile of a group, or a
 // partial last N tile) skips its A/B block loads and DPAS work; it still
@@ -153,6 +326,95 @@
 struct SkipIdleSgOf<Cfg, std::void_t<decltype(Cfg::kSkipIdleSg)>> {
   static constexpr bool value = bool(Cfg::kSkipIdleSg);
 };
+
+// Optional Cfg::kKTailSplit (default false): the K-grouped kernel splits the
+// last partial wave of tiles into 2 or 4 narrower N sub-tiles.
+template <class Cfg, class = void>
+struct KTailSplitOf {
+  static constexpr bool value = false;
+};
+template <class Cfg>
+struct KTailSplitOf<Cfg, std::void_t<decltype(Cfg::kKTailSplit)>> {
+  static constexpr bool value = bool(Cfg::kKTailSplit);
+};
+
+// Optional Cfg::kCacheA / kCacheB (default 0): LSC cache hint of the A / B 2D
+// block loads (1 .ca.ca, 2 .st.ca, 3 .uc.ca, 4 .uc.uc; see the cute block
+// above). Optional Cfg::kCachePf (default 0 = .ca.ca): 1 prefetches into L3
+// only (.uc.ca). Optional Cfg::kGrf (default 256): register file of the launch.
+template <class Cfg, class = void>
+struct CacheAOf {
+  static constexpr int value = 0;
+};
+template <class Cfg>
+struct CacheAOf<Cfg, std::void_t<decltype(Cfg::kCacheA)>> {
+  static constexpr int value = Cfg::kCacheA;
+};
+template <class Cfg, class = void>
+struct CacheBOf {
+  static constexpr int value = 0;
+};
+template <class Cfg>
+struct CacheBOf<Cfg, std::void_t<decltype(Cfg::kCacheB)>> {
+  static constexpr int value = Cfg::kCacheB;
+};
+template <class Cfg, class = void>
+struct CachePfOf {
+  static constexpr int value = 0;
+};
+template <class Cfg>
+struct CachePfOf<Cfg, std::void_t<decltype(Cfg::kCachePf)>> {
+  static constexpr int value = Cfg::kCachePf;
+};
+template <class Cfg, class = void>
+struct GrfOf {
+  static constexpr int value = 256;
+};
+template <class Cfg>
+struct GrfOf<Cfg, std::void_t<decltype(Cfg::kGrf)>> {
+  static constexpr int value = Cfg::kGrf;
+};
+
+// Operand copies / prefetches with the Cfg cache hints; hint 0 is exactly the
+// auto-selected sycl-tla atom.
+template <int Hint, class TiledMMA, class ATensor>
+CUTE_DEVICE auto make_copy_a_cc(TiledMMA const& mma, ATensor const& A) {
+  if constexpr (Hint == 0) {
+    return get_block_2d_copy_A<void>(mma, A);
+  } else {
+    using Auto = decltype(make_block_2d_copy_A(mma, A));
+    using Op = typename cute::gg_cc::op_of<typename Auto::Traits>::type;
+    return make_block_2d_copy_A(cute::XE_LOAD_2D_CC<Op, Hint>{}, mma, A);
+  }
+}
+template <int Hint, class TiledMMA, class BTensor>
+CUTE_DEVICE auto make_copy_b_cc(TiledMMA const& mma, BTensor const& B) {
+  if constexpr (Hint == 0) {
+    return get_block_2d_copy_B<void>(mma, B);
+  } else {
+    using Auto = decltype(make_block_2d_copy_B(mma, B));
+    using Op = typename cute::gg_cc::op_of<typename Auto::Traits>::type;
+    return make_block_2d_copy_B(cute::XE_LOAD_2D_CC<Op, Hint>{}, mma, B);
+  }
+}
+template <int Hint, class Copy_Atom, class LayoutCopy_TV, class ShapeTiler_MN>
+CUTE_DEVICE auto make_prefetch_cc(
+    TiledCopy<Copy_Atom, LayoutCopy_TV, ShapeTiler_MN> const& tc) {
+  if constexpr (Hint == 0) {
+    return make_block_2d_prefetch(tc);
+  } else {
+    using TCopy = TiledCopy<Copy_Atom, LayoutCopy_TV, ShapeTiler_MN>;
+    constexpr auto sg_count =
+        typename TCopy::TiledNumThr{} / typename TCopy::AtomNumThr{};
+    auto& traits = cute::detail::as_block_2d_traits(tc);
+    using AutoPf = decltype(make_block_2d_prefetch(tc));
+    using PfOp = typename cute::gg_cc::op_of<typename AutoPf::Traits>::type;
+    return make_block_2d_prefetch<typename Copy_Atom::ValType, sg_count()>(
+               cute::XE_PREFETCH_2D_CC<PfOp, Hint>{}, ShapeTiler_MN{},
+               traits.tiled_strides, traits.get_x_mode(), tra
```

The best candidate directory is copied to `/mnt/aimod/subratag/ggemm_agent_runs/kernel_forge/v4knobs-r2-20261125/best` (header + configs.json).

## Tree summary

(N = visits, score = own search-suite score, best-subtree = best score in the subtree; the selection value is L = 1/score)

```
n000 [root] N=151 score=1.0103 best-subtree=1.0209 seed kernel /mnt/aimod/subratag/ggemm_agent_runs/kernel_forge/v4knobs-20261125/nodes/n120
  n006 [knob] N=2 score=0.7616 best-subtree=0.7616 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 64x128x16 sg4x4 pf3 wgs0
    n009 [knob] N=1 score=0.6136 best-subtree=0.6136 knob tile_k on k_grouped: 64x128x16 sg4x4 pf3 wgs0 -> 64x128x64 sg4x4 pf2 wgs0
  n004 [knob] N=27 score=1.0133 best-subtree=1.0156 knob sg on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg4x8 pf3 wgs0 be2 si
    n010 [llm] N=5 score=0.9278 best-subtree=1.0050 I added a `kRegPipe` option to `gemm_tile` and turned it on only for grad_weight (k_groupe
      n018 [llm] N=1 score=0.9692 best-subtree=0.9692 I changed `kRegPipe` from on/off to a mode setting (0 off, 1 = A and B as before, 2 = A on
      n028 [llm] N=3 score=1.0037 best-subtree=1.0050 I turned off `kRegPipe` for grad_weight, because the parent's own evaluation shows it cost
        n068 [knob] N=1 score=1.0050 best-subtree=1.0050 knob prefetch on k_grouped: 256x256x16 sg8x4 pf3 wgs0 si -> 256x256x16 sg8x4 pf4 wgs0 si
        n089 [llm] N=1 score=1.0042 best-subtree=1.0042 I cached `offs` in sub-group registers for the K-grouped (grad_weight) kernel; this compil
    n012 [llm] N=7 score=1.0140 best-subtree=1.0156 The forward (m_rowB) kernel can split its last partial wave 4 ways again. The parent's 4x8
      n019 [llm] N=2 score=1.0104 best-subtree=1.0153 When the last partial wave is split into narrower N sub-tiles (in the M-grouped forward/dX
        n037 [llm] N=1 score=1.0153 best-subtree=1.0153 In the grad_weight (K-grouped) kernel, the ragged last N column is now split out when N % 
      n027 [knob] N=2 score=1.0156 best-subtree=1.0156 knob cache_a on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 ca1
        n033 [knob] N=1 score=0.9648 best-subtree=0.9648 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 ca1 -> 128x256x16 sg4x4 pf3 wgs0 ca1
      n081 [llm] N=2 score=0.9771 best-subtree=0.9771 I added an opt-in `Cfg::kEarlyLoadA` mainloop variant to `gemm_tile` and turned it on only
        n140 [knob] N=1 score=0.4830 best-subtree=0.4830 knob tile_k on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x64 sg8x4 pf2 wgs0
    n014 [llm] N=1 score=0.5303 best-subtree=0.5303 I added an opt-in register double-buffered mainloop to `gemm_tile` (Cfg `kRegPipe`, off by
    n047 [knob] N=6 score=1.0089 best-subtree=1.0139 knob prefetch on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf4 wgs0
      n052 [knob] N=2 score=1.0139 best-subtree=1.0139 knob cache_a on m_rowB: 256x256x32 sg4x8 pf3 wgs0 be2 si -> 256x256x32 sg4x8 pf3 wgs0 be2 
        n063 [llm] N=1 score=0.9667 best-subtree=0.9667 I added a `kPipeA` option to `gemm_tile` in `grouped_mm_kernel_impl.h` and turned it on fo
      n056 [llm] N=2 score=1.0096 best-subtree=1.0096 When the kernel splits the last partial wave of tiles, it can now split them along M if th
        n073 [llm] N=1 score=0.9806 best-subtree=0.9806 I generalized the wave-tail split in MGroupedKernel so the last partial wave can be split 
      n135 [knob] N=1 score=1.0082 best-subtree=1.0082 knob cache_a on m_rowB: 256x256x32 sg4x8 pf3 wgs0 be2 si -> 256x256x32 sg4x8 pf3 wgs0 be2 
    n098 [knob] N=7 score=1.0141 best-subtree=1.0146 knob cache_a on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg8x4 pf3 wgs0 be1 
      n102 [llm] N=3 score=1.0103 best-subtree=1.0146 I changed which accumulator block each sub-group works on. In gemm_tile, when a work-group
        n117 [llm] N=1 score=0.9927 best-subtree=0.9927 In the `gemm_tile` mainloop, which all three kinds share, each K step now issues its coope
        n123 [knob] N=1 score=1.0146 best-subtree=1.0146 knob cache_b on m_rowB: 256x256x32 sg4x8 pf3 wgs0 be2 si -> 256x256x32 sg4x8 pf3 wgs0 be2 
      n113 [knob] N=1 score=0.9224 best-subtree=0.9224 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 512x128x16 sg8x4 pf3 wgs0
      n130 [knob] N=2 score=0.9884 best-subtree=0.9884 knob prefetch on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si ca2 -> 256x256x32 sg8x4 pf1 wgs0
        n146 [knob] N=1 score=0.9323 best-subtree=0.9323 knob tile on m_colB: 256x256x32 sg8x4 pf1 wgs0 be1 si ca2 -> 512x128x32 sg16x2 pf1 wgs0 be
  n008 [knob] N=26 score=0.9547 best-subtree=1.0189 knob tile on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x128x32 sg8x2 pf3 wgs0 be1 si
    n013 [knob] N=2 score=0.8607 best-subtree=0.8607 knob wgs on m_colB: 256x128x32 sg8x2 pf3 wgs0 be1 si -> 256x128x32 sg8x2 pf3 wgs56 be1 si
      n016 [knob] N=1 score=0.7034 best-subtree=0.7034 knob tile_k on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x32 sg8x4 pf3 wgs0
    n011 [llm] N=12 score=0.9951 best-subtree=1.0189 I added a `kChunkedOrder` option to `KGroupedKernel` in `grouped_mm_kernel_impl.h` and tur
      n020 [llm] N=4 score=1.0189 best-subtree=1.0189 I added a `kChunkTiles` option to the chunked tile order in `KGroupedKernel` and set it to
        n034 [llm] N=2 score=1.0162 best-subtree=1.0162 In `KGroupedKernel` (grad_weight) I added a `kMergeTailN` option and turned it on for k_gr
          n099 [llm] N=1 score=0.9034 best-subtree=0.9034 I added a `kCarryLoad` option to `KGroupedKernel` and turned it on for k_grouped (grad_wei
        n071 [knob] N=1 score=1.0098 best-subtree=1.0098 knob prefetch on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf2 wgs0 be2
      n031 [knob] N=3 score=0.9906 best-subtree=1.0111 knob prefetch on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf2 wgs0
        n050 [llm] N=1 score=1.0111 best-subtree=1.0111 The K-grouped (grad_weight) kernel now chooses its tile order on the device. It uses the c
        n076 [knob] N=1 score=0.9882 best-subtree=0.9882 knob barrier on k_grouped: 256x256x16 sg8x4 pf2 wgs0 -> 256x256x16 sg8x4 pf2 wgs0 be2
      n059 [llm] N=3 score=0.9664 best-subtree=0.9917 I generalized the device-side wave-tail split in MGroupedKernel, which handles the forward
        n090 [llm] N=1 score=0.9917 best-subtree=0.9917 I replaced the cost model the parent's wave-tail split in MGroupedKernel uses to choose S.
        n122 [knob] N=1 score=0.9769 best-subtree=0.9769 knob cache_a on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be2 
      n150 [knob] N=1 score=0.9237 best-subtree=0.9237 knob tile on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x128x32 sg8x2 pf3 wgs0 be2 si
    n029 [llm] N=9 score=1.0053 best-subtree=1.0095 In `MGroupedKernel` (forward and dX), each work-group now looks up its wave-tail sub-tile 
      n041 [knob] N=1 score=0.7272 best-subtree=0.7272 knob tile_k on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x64 sg8x4 pf2 wgs0 be2 s
      n044 [knob] N=4 score=0.9372 best-subtree=1.0095 knob wgs on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs168
        n048 [llm] N=1 score=0.9942 best-subtree=0.9942 I added an optional `kKChunked` mode to `KGroupedKernel` and switched it on for k_grouped 
        n060 [llm] N=2 score=1.0095 best-subtree=1.0095 `launch()` in grouped_mm_kernel_impl.h now caps the persistent grid at what fits on the GP
          n127 [llm] N=1 score=1.0084 best-subtree=1.0084 In `gemm_tile`, the M-grouped kernels (forward and dX, 8x4 layout with skip-idle) now rela
      n064 [llm] N=3 score=1.0044 best-subtree=1.0044 I ported the parent's wave-tail prefetch hand-off from the forward/dX kernel to the grad_w
        n086 [knob] N=1 score=0.9952 best-subtree=0.9952 knob cache_a on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 ca2
        n114 [llm] N=1 score=0.9288 best-subtree=0.9288 I added a software-pipelined mainloop behind a new compile-time knob, `kLoadAhead`, and tu
    n045 [knob] N=1 score=0.7578 best-subtree=0.7578 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 64x256x16 sg4x8 pf3 wgs0
    n080 [llm] N=1 score=0.3868 best-subtree=0.3868 I added a compile-time `kPipeLoads` option to the `gemm_tile` mainloop and turned it on fo
  n003 [knob] N=1 score=0.8068 best-subtree=0.8068 knob tile_k on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x32 sg8x4 pf3 wgs0
  n005 [llm] N=24 score=0.9277 best-subtree=1.0209 I added an optional software-pipelined mainloop to `gemm_tile`, switched on only for grad_
    n015 [llm] N=5 score=0.9730 best-subtree=1.0072 Both kernels now split the last partial wave more flexibly. The leftover tiles' S×r narrow
      n026 [knob] N=1 score=0.9388 best-subtree=0.9388 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 512x128x16 sg16x2 pf3 wgs0
      n043 [llm] N=3 score=1.0072 best-subtree=1.0072 The forward and grad_input kernels go back to production's wave-tail rule: they only split
        n051 [llm] N=1 score=1.0043 best-subtree=1.0043 In the grad_weight (K-grouped) kernel, when the last N tile column has at most TileN/2 (or
        n093 [knob] N=1 score=0.9910 best-subtree=0.9910 knob skip_idle on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg8x4 pf3 wgs0 be
    n017 [llm] N=7 score=0.9584 best-subtree=1.0150 I replaced the parent's full load-ahead on grad_weight, which cost 17% on dW, with a B-onl
      n030 [llm] N=3 score=1.0150 best-subtree=1.0150 In `KGroupedKernel`'s persistent loop (grad_weight), a tile at the right edge of N with at
        n055 [llm] N=1 score=1.0032 best-subtree=1.0032 I ported the parent's ragged-N edge-tile trick from grad_weight to the M-grouped forward a
        n065 [llm] N=1 score=0.9931 best-subtree=0.9931 In `gemm_tile` I replaced the A-operand `reorder(tArA, tCrA)` with a new `reorder_operand(
      n039 [llm] N=1 score=0.9576 best-subtree=0.9576 In `MGroupedKernel`'s persistent loop (forward and grad_input), a group's last row tile no
      n095 [llm] N=2 score=1.0077 best-subtree=1.0077 I added a code change to `grouped_mm_kernel_impl.h` and left `configs.json` as it was (the
        n143 [knob] N=1 score=0.9655 best-subtree=0.9655 knob sg on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg4x8 pf3 wgs0
    n038 [knob] N=2 score=0.9178 best-subtree=0.9178 knob prefetch on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf2 wgs0 be2
      n145 [knob] N=1 score=0.9010 best-subtree=0.9010 knob prefetch on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg8x4 pf1 wgs0 be1
    n061 [llm] N=8 score=1.0115 best-subtree=1.0209 On grad_weight I replaced the parent's hand-written load-ahead pipeline (`kLoadAhead`), wh
      n067 [llm] N=3 score=0.9996 best-subtree=1.0209 I fixed the bug that left three grad_weight output tiles unwritten: those were the NaN reg
        n116 [llm] N=1 score=1.0209 best-subtree=1.0209 In the grad_weight kernel, the chunked tile order (kChunkOrder) is now chosen on the devic
        n137 [knob] N=1 score=0.4925 best-subtree=0.4925 knob tile_k on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x64 sg8x4 pf2 wgs0
      n085 [llm] N=2 score=1.0111 best-subtree=1.0111 I changed how both kernels schedule an output whose last tile column is half width or less
        n111 [knob] N=1 score=0.9926 best-subtree=0.9926 knob barrier on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be4 
      n119 [llm] N=2 score=1.0026 best-subtree=1.0026 I added a `kChunkK` option to `gemm_tile` and switched it on for the forward kind only (m_
        n134 [knob] N=1 score=0.9426 best-subtree=0.9426 knob wgs on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs168
    n129 [knob] N=1 score=0.9153 best-subtree=0.9153 knob prefetch on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf2 wgs0
  n001 [llm] N=1 score=0.6782 best-subtree=0.6782 I added an opt-in software-pipelined mainloop (Cfg::kEarlyLoad) to gemm_tile and turned it
  n002 [llm] N=21 score=0.9835 best-subtree=0.9989 For grad_weight only, I software-pipelined the transposed A operand by one K tile, control
    n021 [knob] N=5 score=0.9609 best-subtree=0.9988 knob sg on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg4x8 pf3 wgs0 be1 si
      n025 [knob] N=1 score=0.9632 best-subtree=0.9632 knob barrier on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be1 
      n092 [llm] N=3 score=0.9988 best-subtree=0.9988 The M-grouped kernel (forward and grad_input) now calls a new `Scheduler::preload()` once 
        n100 [knob] N=1 score=0.7662 best-subtree=0.7662 knob tile on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x64x32 sg8x1 pf3 wgs0 be1 si
        n103 [knob] N=1 score=0.9628 best-subtree=0.9628 knob prefetch on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf1 wgs0 be2
    n022 [llm] N=6 score=0.9770 best-subtree=0.9989 I changed how both kernels split the last partial wave of tiles: they now pick the split (
      n032 [llm] N=2 score=0.9972 best-subtree=0.9978 I replaced the tail-split cost model with a per-kernel one, `pick_tail_split<C2, C4>`, whi
        n058 [llm] N=1 score=0.9978 best-subtree=0.9978 When N leaves a last tile column that is at most half used (N=1408 = 5.5 × 256 gives one t
      n053 [knob] N=2 score=0.9740 best-subtree=0.9989 knob cache_b on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be2 
        n082 [llm] N=1 score=0.9989 best-subtree=0.9989 I made the M-grouped (fwd/dX) wave-tail split account for a fixed cost per pass. It now ch
      n109 [llm] N=1 score=0.9796 best-subtree=0.9796 The last full tile now prefetches the first K tiles of the work-group's first tail sub-til
    n042 [knob] N=3 score=0.9684 best-subtree=0.9684 knob barrier on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 be4
      n046 [llm] N=1 score=0.9651 best-subtree=0.9651 In the grad_weight kPipeA path, the mainloop no longer issues the extra A load past the la
      n066 [knob] N=1 score=0.8809 best-subtree=0.8809 knob sg on k_grouped: 256x256x16 sg8x4 pf3 wgs0 be4 -> 256x256x16 sg4x8 pf3 wgs0 be4
    n070 [knob] N=4 score=0.9780 best-subtree=0.9780 knob cache_a on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 ca1
      n075 [knob] N=2 score=0.9696 best-subtree=0.9696 knob prefetch on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf4 wgs0 be2
        n105 [knob] N=1 score=0.7034 best-subtree=0.7034 knob tile_k on m_rowB: 256x256x32 sg8x4 pf4 wgs0 be2 si -> 256x256x64 sg8x4 pf2 wgs0 be2 s
      n079 [llm] N=1 score=0.9695 best-subtree=0.9695 In the software-pipelined A path that grad_weight uses (kPipeA), I added a guard so the lo
    n118 [knob] N=2 score=0.9763 best-subtree=0.9763 knob cache_a on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg8x4 pf3 wgs0 be1 
      n125 [knob] N=1 score=0.9726 best-subtree=0.9726 knob cache_a on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be2 
  n007 [llm] N=23 score=1.0016 best-subtree=1.0132 I changed the epilogue store in `gemm_tile` so it alternates between two 2D-block address 
    n023 [llm] N=6 score=1.0066 best-subtree=1.0132 I added a `kEdgeHalfN` option to the grad_weight (K-grouped) kernel and turned it on in co
      n035 [llm] N=2 score=0.9969 best-subtree=1.0132 I ported the `kEdgeHalfN` edge-tile scheduling from the grad_weight kernel to the M-groupe
        n088 [llm] N=1 score=1.0132 best-subtree=1.0132 I made a small header change and kept the configs.json edit, which leaves `kEdgeHalfN` off
      n054 [knob] N=2 score=1.0112 best-subtree=1.0112 knob barrier on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 be4
        n057 [knob] N=1 score=0.7625 best-subtree=0.7625 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 be4 -> 256x64x16 sg8x1 pf3 wgs0 be4
      n115 [llm] N=1 score=0.9167 best-subtree=0.9167 I added an optional register double-buffered mainloop to `gemm_tile` (`Cfg::kRegDB`) and t
    n024 [llm] N=6 score=1.0069 best-subtree=1.0128 I added an optional tile-chunking scheduler to `KGroupedKernel` (`Cfg::kKChunk`, set to 2 
      n036 [knob] N=2 score=1.0081 best-subtree=1.0128 knob cache_b on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg8x4 pf3 wgs0 be1 
        n062 [llm] N=1 score=1.0128 best-subtree=1.0128 The M-grouped kernel (forward and dX) now loads the group offsets once per work-group into
      n049 [knob] N=2 score=0.9897 best-subtree=1.0103 knob prefetch on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg8x4 pf1 wgs0 be1
        n074 [llm] N=1 score=1.0103 best-subtree=1.0103 I added N-remainder pairing to the grad_weight (K-grouped) kernel, switched on by a new `C
      n128 [knob] N=1 score=0.8053 best-subtree=0.8053 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x64x16 sg8x1 pf3 wgs0
    n040 [llm] N=1 score=0.9224 best-subtree=0.9224 I added an opt-in slab-packed scheduler for the M-grouped kernels (`Cfg::kPackM`, a new `g
    n069 [llm] N=5 score=0.9658 best-subtree=1.0088 I added an optional `Cfg::kPipeA` knob that changes the grad_weight mainloop so A is loade
      n083 [llm] N=2 score=1.0065 best-subtree=1.0065 I split the generic mainloop in `gemm_tile` into two phases, and I also turned `kPipeA` of
        n108 [knob] N=1 score=0.9677 best-subtree=0.9677 knob wgs on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs224 be2 si
      n097 [llm] N=2 score=1.0018 best-subtree=1.0088 In grad_weight (k_grouped), a tile on the ragged right edge with at most TileN/2 (or TileN
        n110 [knob] N=1 score=1.0088 best-subtree=1.0088 knob cache_b on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg8x4 pf3 wgs0 be1 
    n133 [knob] N=4 score=1.0005 best-subtree=1.0005 knob barrier on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg8x4 pf3 wgs0 be3 
      n139 [knob] N=2 score=0.9367 best-subtree=0.9367 knob wgs on m_colB: 256x256x32 sg8x4 pf3 wgs0 be3 si -> 256x256x32 sg8x4 pf3 wgs112 be3 si
        n149 [llm] N=1 score=0.9320 best-subtree=0.9320 I added a `Cfg::kLoadAhead` knob to `gemm_tile` and turned it on for k_grouped. It pipelin
      n141 [llm] N=1 score=0.9299 best-subtree=0.9299 I added software-pipelined operand loads to `gemm_tile`, behind a new optional `Cfg::kLoad
  n072 [knob] N=13 score=0.9753 best-subtree=1.0075 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x128x16 sg8x2 pf3 wgs0
    n077 [llm] N=2 score=0.9748 best-subtree=0.9748 In the M-grouped (forward / dX) kernel, the group table now gets built once per work-group
      n084 [knob] N=1 score=0.9529 best-subtree=0.9529 knob cache_b on k_grouped: 256x128x16 sg8x2 pf3 wgs0 -> 256x128x16 sg8x2 pf3 wgs0 cb2
    n078 [llm] N=5 score=0.9364 best-subtree=1.0075 I added a software-pipelined mainloop for the grad_weight kernel. Each sub-group now issue
      n087 [llm] N=3 score=1.0071 best-subtree=1.0075 In the M-grouped (fwd/dX) mainloop, which uses barriers, each sub-group now issues its dis
        n104 [llm] N=1 score=0.8413 best-subtree=0.8413 I fixed the build error by taking the row-block and K/N counts in the masked-DPAS path fro
        n148 [llm] N=1 score=1.0075 best-subtree=1.0075 I added a compile-time flag `kPfFirst` (template parameter `PfFirst` in `gemm_tile`) and s
      n094 [llm] N=1 score=0.6716 best-subtree=0.6716 In the grad_weight kernel's software-pipelined (kRegPipe) mainloop, B now loads into two a
    n091 [llm] N=4 score=0.9923 best-subtree=1.0057 I made a small code change to the header (`grouped_mm_kernel_impl.h`) and kept the earlier
      n101 [llm] N=2 score=1.0057 best-subtree=1.0057 In the grad_weight kernel, a last N tile with at most half its columns in range now runs a
        n136 [knob] N=1 score=0.9942 best-subtree=0.9942 knob cache_a on k_grouped: 256x256x16 sg16x2 pf3 wgs0 -> 256x256x16 sg16x2 pf3 wgs0 ca2
      n121 [llm] N=1 score=0.9951 best-subtree=0.9951 I changed the grad_weight (K-grouped) scheduler: when all the operand data (total routed r
    n142 [knob] N=1 score=0.7817 best-subtree=0.7817 knob tile on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x64x32 sg8x1 pf3 wgs0 be2 si
  n096 [llm] N=7 score=1.0096 best-subtree=1.0139 In the grad_weight (K-grouped) kernel, I added an opt-in `kHalfNTail` knob and turned it o
    n106 [knob] N=3 score=0.9887 best-subtree=1.0089 knob prefetch on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg8x4 pf1 wgs0 be1
      n112 [llm] N=1 score=0.9310 best-subtree=0.9310 I added an opt-in `kPipeLoads` knob to `gemm_tile` and turned it on for k_grouped only (gr
      n120 [llm] N=1 score=1.0089 best-subtree=1.0089 I added an opt-in `kHalfNTail` path to the M-grouped kernel and turned it on for m_colB (g
    n107 [llm] N=2 score=1.0139 best-subtree=1.0139 I changed the M-grouped (fwd/dX) kernel so that, whenever there are at most 16 groups (tru
      n138 [llm] N=1 score=0.9891 best-subtree=0.9891 In the M-grouped (fwd/dX) kernel I added a "fit-M" tile path, switched on with a new Cfg k
    n124 [llm] N=1 score=1.0031 best-subtree=1.0031 I ported the half-width N-tail path to the M-grouped kernel and turned it on only for m_co
  n126 [knob] N=5 score=0.9953 best-subtree=1.0156 knob sg on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x32 sg4x8 pf3 wgs0 be1 si
    n131 [llm] N=2 score=1.0156 best-subtree=1.0156 I changed how the M-grouped kernel (forward and dX) finds its tiles; I could not build or 
      n144 [llm] N=1 score=1.0011 best-subtree=1.0011 I generalised the M-grouped (forward and dX) wave-tail split so it balances the last parti
    n132 [llm] N=2 score=1.0037 best-subtree=1.0037 In MGroupedKernel, the prologue used to scan offs twice before the first tile: once to cou
      n147 [llm] N=1 score=0.6944 best-subtree=0.6944 Forward and dX now double-buffer the A/B operand registers in the M-grouped mainloop (new
```

Widening at the end: alpha = 0.470 (root visits 151).
