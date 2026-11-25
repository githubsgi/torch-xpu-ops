# Kernel Forge (MCTS) run report

## Run

- workdir: `/mnt/aimod/subratag/ggemm_agent_runs/kernel_forge/v4knobs-20261125`
- iterations: 120 (target 120); nodes admitted: 120; failed attempts: 0
- actions: knob 45, llm 75; outcomes: ok 120
- LLM revisions: 75 claude sessions (incl. repairs), 0 repair rounds, spend $37.57 (budget $400, model opus)
- search suite `search`, final suite `all`, C=0.1, vloss=0.05, alpha 0.5->0.3 over 1000 root visits, workers 8, knob ratio 0.5
- production vs itself measured on `search`: 0.9994x; on `all`: 1.0002x

## Result

Best node **n120**: 1.0126x on `search`, **1.0104x on `all`** (production re-run: 1.0002x).

Verdict: no significant improvement (< 2% is noise) (+1.0%).

## Lineage of the best node

| step | node | action | change | quick score | vs parent |
|---|---|---|---|---|---|
| 0 | n000 | root | production kernel | 0.9994 |  |
| 1 | n001 | llm | I gave the K-grouped (grad_weight) kernel the same device-side split of the last partial wave that the forward and grad_input kernels already use, turned on through a new Cfg knob `kKTailSplit` (set via "extra" for k_grouped in configs.json). The tile count is _[k_grouped: extra None->static constexpr bool kKTailSplit = true;; header +96/-2 lines]_ | 0.9997 | +0.0% |
| 2 | n066 | knob | knob barrier on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be1 si _[m_colB: barrier_every 2->1; header +0/-0 lines]_ | 1.0081 | +0.8% |
| 3 | n099 | llm | In the K-grouped (grad_weight) kernel I replaced the sub-group-scan tile lookup (`locate_par`) with a new `Scheduler::locate_direct`. Every group has the same tile count (tiles_m × tiles_n), so the owning group is just `linear / tiles_per_group`, and the two ` _[no config change; header +35/-3 lines]_ | 1.0044 | -0.4% |
| 4 | n120 | llm | I added a `kNoBarrier` Cfg knob, read through a `NoBarrierOf` trait. It sets the `gemm_tile` barrier span to 0, which removes every split-barrier arrive/wait from the mainloop. I turned it on only for k_grouped (grad_weight), through "extra" in configs.json; f _[k_grouped: extra static constexpr bool kKTailSplit = true;->static constexpr bool kKTailSplit = true; static constexpr bool kNoBarrier = true;; header +28/-5 lines]_ | 1.0126 | +0.8% |

## Top candidates (search suite, then full re-evaluation)

| node | action | quick | full | parent | note |
|---|---|---|---|---|---|
| n120 | llm | 1.0126 | 1.0104 | n099 | I added a `kNoBarrier` Cfg knob, read through a `NoBarrierOf` trait. It sets the `gemm_tile` barrier span to 0 |
| n035 | llm | 1.0087 | 1.0061 | n019 | I narrowed when the grad_weight kernel (`KGroupedKernel`) splits its last partial wave. It now splits only whe |
| n066 | knob | 1.0081 | 1.0023 | n001 | knob barrier on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be1 si |
| n078 | llm | 1.0081 | 1.0058 | n059 | I added the wave-tail split to the grad_weight kernel (KGroupedKernel). The forward and dX kernels already had |
| n087 | llm | 1.0080 | - | n072 | I split the `gemm_tile` mainloop into two phases. The steady-state loop issues the current tile's K+dist prefe |
| n079 | llm | 1.0069 | - | n066 | In the M-grouped kernel (forward and grad_input), each work-group now looks up its first tile before it counts |
| n090 | llm | 1.0068 | - | n079 | I changed how the K-grouped (grad_weight) kernel finds its tiles. It now calls a new `locate_k` helper that wo |
| n048 | llm | 1.0065 | - | n012 | I gave the K-grouped grad_weight kernel the same wave-tail split the M-grouped kernels already use. If the las |

## Per-op speedup of the best node vs production

| op | quick | full |
|---|---|---|
| w13_dW | 1.018x | 1.041x |
| w13_dX | 1.007x | 1.024x |
| w13_fwd | 1.000x | 1.002x |
| w2_dW | 1.043x | 1.068x |
| w2_dX | 1.009x | 1.02x |
| w2_fwd | 0.996x | 1.002x |

Per case (full):

- tp4ep4_uniform: 1.014x
- tp4ep4_balanced: 1.012x
- tp4ep4_hot: 1.014x
- tp4ep4_sparse: 1.006x
- tp4ep4_rows64 (probe, not scored): 1.114x
- tp4ep4_rows8 (probe, not scored): 1.124x
- tp8ep8_uniform: 1.019x
- tp8ep8_balanced: 1.023x
- tp8ep8_hot: 1.016x
- tp8ep8_sparse: 1.023x
- tp8ep8_rows64 (probe, not scored): 1.111x
- ep8_uniform: 1.006x
- ep8_balanced: 1.002x
- ep8_hot: 1.008x
- ep8_rows1024 (probe, not scored): 1.002x
- 671b_tp8ep32_uniform: 1.004x
- 671b_tp8ep32_balanced: 1.000x
- 671b_tp8ep32_hot: 1.004x
- 671b_tp8ep32_sparse: 1.003x
- 671b_tp8ep32_rows128 (probe, not scored): 1.013x
- 671b_tp8ep32_rows2048 (probe, not scored): 1.001x

## Configs and diff vs production

```
configs: m_colB: barrier_every 2->1; k_grouped: extra None->static constexpr bool kKTailSplit = true; static constexpr bool kNoBarrier = true;
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

The best candidate directory is copied to `/mnt/aimod/subratag/ggemm_agent_runs/kernel_forge/v4knobs-20261125/best` (header + configs.json).

## Tree summary

(N = visits, score = own search-suite score, best-subtree = best score in the subtree; the selection value is L = 1/score)

```
n000 [root] N=121 score=0.9994 best-subtree=1.0126 seed kernel /mnt/aimod/subratag/ggemm_agent_runs/v4knobs/seeds/s0_v4ext
  n003 [knob] N=16 score=0.9052 best-subtree=1.0050 knob wgs on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs112
    n009 [knob] N=5 score=0.9420 best-subtree=1.0044 knob wgs on k_grouped: 256x256x16 sg8x4 pf3 wgs112 -> 256x256x16 sg8x4 pf3 wgs168
      n011 [llm] N=2 score=1.0039 best-subtree=1.0044 In the forward/grad_input (M-grouped) mainloop of `gemm_tile`, each K tile now sends its f
        n050 [llm] N=1 score=1.0044 best-subtree=1.0044 The grad_weight kernel (`KGroupedKernel`) now uses the same device-side wave-tail split th
      n029 [llm] N=2 score=0.9739 best-subtree=1.0021 I added an optional compile-time knob, `Cfg::kEarlyA` (through the `EarlyAOf` trait), to `
        n116 [llm] N=1 score=1.0021 best-subtree=1.0021 I added wave-tail balancing to grad_weight (`KGroupedKernel`), copying the device-side spl
    n010 [llm] N=4 score=0.9943 best-subtree=0.9943 In the shared `gemm_tile` mainloop, the prefetch for K tile k+3 (or for the next tile's fi
      n017 [llm] N=1 score=0.9113 best-subtree=0.9113 I added an opt-in `kLoadAhead` knob to the shared `gemm_tile` mainloop and turned it on on
      n039 [llm] N=2 score=0.9899 best-subtree=0.9938 I added the device-side wave-tail split to `KGroupedKernel` (grad_weight), which so far on
        n062 [llm] N=1 score=0.9938 best-subtree=0.9938 In `MGroupedKernel` (forward and dX), each work-group now finds its first tile before anyt
    n026 [knob] N=1 score=0.8568 best-subtree=0.8568 knob sg on k_grouped: 256x256x16 sg8x4 pf3 wgs112 -> 256x256x16 sg4x8 pf3 wgs112
    n069 [llm] N=5 score=1.0050 best-subtree=1.0050 I ported the device-side wave-tail split from the M-grouped kernel to `KGroupedKernel` in 
      n082 [llm] N=2 score=0.9991 best-subtree=0.9991 In `gemm_tile` I added a fixed bijection from each sub-group's hardware id to the 32x64 sl
        n107 [llm] N=1 score=0.9033 best-subtree=0.9033 I added an opt-in `kLoadAhead` software pipeline to the `gemm_tile` mainloop in `grouped_m
      n093 [knob] N=2 score=1.0033 best-subtree=1.0033 knob cache_b on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be2 
        n095 [llm] N=1 score=0.6864 best-subtree=0.6864 I added an optional `Cfg::kRegPipe` to `gemm_tile` and turned it on for forward and grad_i
  n006 [knob] N=1 score=0.7316 best-subtree=0.7316 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 64x128x16 sg4x4 pf3 wgs0
  n004 [knob] N=18 score=0.9944 best-subtree=1.0065 knob prefetch on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf4 wgs0 be2
    n014 [llm] N=4 score=0.9937 best-subtree=0.9999 I ported the wave-tail split, which already exists in MGroupedKernel, to KGroupedKernel (g
      n030 [llm] N=2 score=0.9999 best-subtree=0.9999 I changed the grad_weight loop in KGroupedKernel so that when a group's last N column is a
        n089 [llm] N=1 score=0.9916 best-subtree=0.9916 In MGroupedKernel (forward and grad_input), when a group's last N column is at most half i
      n061 [llm] N=1 score=0.9955 best-subtree=0.9955 I changed the forward and grad_input kernel (MGroupedKernel) so that the last full tile pr
    n012 [llm] N=5 score=1.0033 best-subtree=1.0065 In `gemm_tile`, I moved the split-barrier wait that closes each tile's last K span so it n
      n024 [llm] N=2 score=1.0043 best-subtree=1.0043 In `gemm_tile`, each sub-group's role (which 32x64 slab it owns, plus its share of the A/B
        n112 [llm] N=1 score=0.6980 best-subtree=0.6980 I added an optional knob, Cfg kPipeRegs, that double-buffers the A/B operand registers in 
      n048 [llm] N=2 score=1.0065 best-subtree=1.0065 I gave the K-grouped grad_weight kernel the same wave-tail split the M-grouped kernels alr
        n070 [knob] N=1 score=0.8343 best-subtree=0.8343 knob cache_pf on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 cp1
    n038 [knob] N=4 score=0.9916 best-subtree=1.0055 knob prefetch on m_rowB: 256x256x32 sg8x4 pf4 wgs0 be2 si -> 256x256x32 sg8x4 pf2 wgs0 be2
      n046 [llm] N=2 score=1.0055 best-subtree=1.0055 I added the device-side wave-tail split to KGroupedKernel (grad_weight), which until now o
        n080 [llm] N=1 score=0.8927 best-subtree=0.8927 In the grad_weight kernel (KGroupedKernel), each tile now loads the next tile's first K-ti
      n056 [llm] N=1 score=0.9982 best-subtree=0.9982 The M-grouped kernel used to scan `offs` twice in a row at startup: once to count all tile
    n068 [knob] N=4 score=0.9936 best-subtree=0.9988 knob prefetch on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf4 wgs0
      n075 [knob] N=2 score=0.9988 best-subtree=0.9988 knob prefetch on m_rowB: 256x256x32 sg8x4 pf4 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be2
        n101 [llm] N=1 score=0.3594 best-subtree=0.3594 I added a compile-time knob, Cfg::kRegPipe, that double-buffers the A/B loads in registers
      n076 [knob] N=1 score=0.9945 best-subtree=0.9945 knob cache_a on k_grouped: 256x256x16 sg8x4 pf4 wgs0 -> 256x256x16 sg8x4 pf4 wgs0 ca1
  n008 [knob] N=18 score=0.9349 best-subtree=1.0048 knob tile on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x128x32 sg8x2 pf3 wgs0 be2 si
    n013 [knob] N=2 score=0.9431 best-subtree=0.9502 knob barrier on m_colB: 256x128x32 sg8x2 pf3 wgs0 be2 si -> 256x128x32 sg8x2 pf3 wgs0 be1 
      n063 [llm] N=1 score=0.9502 best-subtree=0.9502 I made a code-only change to the grad_weight (K-grouped) kernel and did not touch configs.
    n015 [llm] N=5 score=1.0024 best-subtree=1.0024 I added `Scheduler::locate_prologue`, which makes one pass over `offs` per sub-group to ge
      n025 [knob] N=3 score=0.9867 best-subtree=0.9867 knob sg on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg4x8 pf3 wgs0 be2 si
        n053 [knob] N=1 score=0.6885 best-subtree=0.6885 knob grf on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg16x4 pf3 wgs56 be2 si
        n111 [knob] N=1 score=0.5334 best-subtree=0.5334 knob cache_a on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be2 
      n031 [knob] N=1 score=0.5827 best-subtree=0.5827 knob tile_k on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x64 sg8x4 pf2 wgs0
    n034 [llm] N=4 score=0.9937 best-subtree=0.9945 In the gemm_tile mainloop, the cooperative L1 prefetches for K tile k+dist (and the next t
      n040 [llm] N=2 score=0.9945 best-subtree=0.9945 I added the device-side wave-tail split to `KGroupedKernel` (grad_weight). It already exis
        n100 [knob] N=1 score=0.7208 best-subtree=0.7208 knob tile on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x64x32 sg8x1 pf3 wgs0 be2 si
      n057 [knob] N=1 score=0.9567 best-subtree=0.9567 knob sg on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg4x8 pf3 wgs0
    n065 [llm] N=6 score=1.0048 best-subtree=1.0048 I added the device-side wave-tail split to the K-grouped (grad_weight) kernel. It already 
      n077 [llm] N=2 score=1.0012 best-subtree=1.0012 In the K-grouped (grad_weight) persistent loop, a tile in the ragged last N column with at
        n098 [knob] N=1 score=0.3731 best-subtree=0.3731 knob cache_b on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be2 
      n088 [llm] N=2 score=1.0003 best-subtree=1.0003 When idle-sub-group skipping is on (the M-grouped forward and dX kernels), `gemm_tile` now
        n103 [knob] N=1 score=0.9658 best-subtree=0.9658 knob prefetch on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf1 wgs0 be2
      n115 [llm] N=1 score=0.9135 best-subtree=0.9135 I added a compile-time knob, `Cfg::kLoadAhead`, and turned it on only for k_grouped (grad_
  n002 [llm] N=17 score=1.0020 best-subtree=1.0087 I changed the device-side tile scheduler so each lane loads its group's row range, tile co
    n016 [knob] N=1 score=0.6260 best-subtree=0.6260 knob tile_k on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x64 sg8x4 pf2 wgs0
    n019 [llm] N=8 score=0.9958 best-subtree=1.0087 The grad_weight kernel (`KGroupedKernel`) now handles its last partial wave the way the M-
      n035 [llm] N=3 score=1.0087 best-subtree=1.0087 I narrowed when the grad_weight kernel (`KGroupedKernel`) splits its last partial wave. It
        n044 [knob] N=1 score=0.9870 best-subtree=0.9870 knob barrier on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 be3
        n054 [knob] N=1 score=0.9294 best-subtree=0.9294 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 128x128x16 sg4x2 pf3 wgs0
      n028 [llm] N=3 score=0.9963 best-subtree=0.9963 In the grad_weight kernel (`KGroupedKernel`), if the last N column of the output is at mos
        n047 [knob] N=1 score=0.6765 best-subtree=0.6765 knob tile_k on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x64 sg8x4 pf2 wgs0
        n064 [llm] N=1 score=0.9383 best-subtree=0.9383 In `gemm_tile`, a tile no longer opens a barrier span that would cover only its last K ste
      n049 [knob] N=1 score=0.7347 best-subtree=0.7347 knob tile_k on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x64 sg8x4 pf2 wgs0 be2 s
    n042 [knob] N=1 score=0.3688 best-subtree=0.3688 knob cache_a on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 ca4
    n059 [llm] N=6 score=1.0035 best-subtree=1.0081 In the M-grouped kernels (fwd and dX), I changed which accumulator slab each sub-group own
      n078 [llm] N=3 score=1.0081 best-subtree=1.0081 I added the wave-tail split to the grad_weight kernel (KGroupedKernel). The forward and dX
        n097 [llm] N=1 score=0.9791 best-subtree=0.9791 In the forward and dX kernels, a tile whose group has at most 192 rows left now runs on a 
        n113 [knob] N=1 score=0.9584 best-subtree=0.9584 knob sg on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg4x8 pf3 wgs0
      n085 [llm] N=1 score=0.7300 best-subtree=0.7300 In the forward and dX kernels, each K tile's operand loads are now split into two halves o
      n118 [knob] N=1 score=1.0018 best-subtree=1.0018 knob cache_b on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be2 
  n007 [llm] N=15 score=1.0004 best-subtree=1.0046 I ported the wave-tail split from the forward/grad_input kernels into `KGroupedKernel` (gr
    n018 [llm] N=4 score=1.0032 best-subtree=1.0032 In `gemm_tile`, the split-barrier wait for the last K tile now happens after the sub-group
      n027 [knob] N=2 score=1.0022 best-subtree=1.0022 knob cache_b on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 cb2
        n102 [llm] N=1 score=0.6560 best-subtree=0.6560 I added an optional register double-buffer to the M-grouped mainloop in `gemm_tile` (`Cfg:
      n055 [llm] N=1 score=0.9987 best-subtree=0.9987 I moved the first tile's prologue prefetch earlier in `MGroupedKernel` (forward and grad_i
    n020 [llm] N=4 score=0.9893 best-subtree=1.0036 In both persistent loops (`MGroupedKernel` and `KGroupedKernel`), a tile in the ragged las
      n032 [llm] N=2 score=1.0036 best-subtree=1.0036 I removed the half-width ragged-N edge path from `MGroupedKernel` (forward and grad_input)
        n092 [llm] N=1 score=1.0007 best-subtree=1.0007 In `KGroupedKernel` (grad_weight), I replaced the `Scheduler::locate_par` lookup with a di
      n045 [knob] N=1 score=0.7674 best-subtree=0.7674 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 64x256x16 sg4x8 pf3 wgs0
    n037 [llm] N=5 score=0.9865 best-subtree=1.0046 In the `gemm_tile` mainloop, each sub-group now waits on the split barrier for the previou
      n051 [llm] N=3 score=1.0045 best-subtree=1.0046 The early-wait barrier placement from the parent now applies only when the barrier spans o
        n081 [llm] N=1 score=1.0046 best-subtree=1.0046 In the M-grouped kernel (forward and grad_input), the group table is now read from `offs` 
        n117 [llm] N=1 score=0.8990 best-subtree=0.8990 I added a pipelined mainloop variant to `gemm_tile`, switched on by a new optional Cfg kno
      n060 [llm] N=1 score=0.9601 best-subtree=0.9601 In both persistent loops (`MGroupedKernel` and `KGroupedKernel`), a tile in the last N col
    n067 [llm] N=1 score=0.9094 best-subtree=0.9094 I added a `kLoadAhead` option to the `gemm_tile` mainloop and turned it on only for grad_w
  n005 [llm] N=1 score=0.6145 best-subtree=0.6145 I added a register double-buffered mainloop to `gemm_tile`, behind a new `Cfg::kRegPipe` k
  n001 [llm] N=18 score=0.9997 best-subtree=1.0126 I gave the K-grouped (grad_weight) kernel the same device-side split of the last partial w
    n021 [knob] N=7 score=1.0024 best-subtree=1.0024 knob prefetch on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf4 wgs0 be2
      n023 [llm] N=2 score=0.9773 best-subtree=0.9773 I added a mixed wave-tail mode to the M-grouped kernel (forward and grad_input). It covers
        n041 [knob] N=1 score=0.9373 best-subtree=0.9373 knob wgs on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs112 be2 si
      n033 [knob] N=3 score=0.9671 best-subtree=1.0021 knob tile on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 128x256x16 sg4x4 pf3 wgs0
        n043 [llm] N=1 score=1.0021 best-subtree=1.0021 In `gemm_tile`, I moved the split-barrier wait that ends each tile's last barrier span to 
        n058 [llm] N=1 score=0.9100 best-subtree=0.9100 I added a compile-time mainloop variant to `gemm_tile`, switched on only for k_grouped thr
      n052 [knob] N=1 score=0.2145 best-subtree=0.2145 knob cache_b on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be2 
    n022 [llm] N=1 score=0.9098 best-subtree=0.9098 In grad_weight I made the operand loads run one K tile ahead. The first tile's A/B copy no
    n036 [knob] N=1 score=0.8706 best-subtree=0.8706 knob cache_pf on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be2
    n066 [knob] N=8 score=1.0081 best-subtree=1.0126 knob barrier on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs0 be1 
      n071 [knob] N=1 score=0.7757 best-subtree=0.7757 knob tile_k on m_colB: 256x256x32 sg8x4 pf3 wgs0 be1 si -> 256x256x64 sg8x4 pf2 wgs0 be1 s
      n079 [llm] N=3 score=1.0069 best-subtree=1.0069 In the M-grouped kernel (forward and grad_input), each work-group now looks up its first t
        n086 [knob] N=1 score=0.6952 best-subtree=0.6952 knob cache_b on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg8x4 pf3 wgs0 cb3
        n090 [llm] N=1 score=1.0068 best-subtree=1.0068 I changed how the K-grouped (grad_weight) kernel finds its tiles. It now calls a new `loca
      n099 [llm] N=3 score=1.0044 best-subtree=1.0126 In the K-grouped (grad_weight) kernel I replaced the sub-group-scan tile lookup (`locate_p
        n109 [llm] N=1 score=0.6109 best-subtree=0.6109 I added a register-pipelined mainloop to `gemm_tile` behind a new Cfg knob `kRegPipe`, and
        n120 [llm] N=1 score=1.0126 best-subtree=1.0126 I added a `kNoBarrier` Cfg knob, read through a `NoBarrierOf` trait. It sets the `gemm_til
  n072 [knob] N=11 score=0.9599 best-subtree=1.0080 knob sg on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg4x8 pf3 wgs0
    n073 [llm] N=3 score=0.9957 best-subtree=0.9957 In the `gemm_tile` mainloop, the cooperative L1/L3 prefetch for K tile k+3 (or for the nex
      n083 [llm] N=1 score=0.9176 best-subtree=0.9176 I made the grad_weight (K-grouped) mainloop load one K tile ahead. The A/B block loads for
      n110 [knob] N=1 score=0.7327 best-subtree=0.7327 knob grf on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg16x4 pf3 wgs56 be2 si
    n074 [llm] N=3 score=1.0018 best-subtree=1.0018 I added the device-side wave-tail split to KGroupedKernel, which runs grad_weight. Until n
      n084 [knob] N=1 score=0.9212 best-subtree=0.9212 knob grf on k_grouped: 256x256x16 sg8x4 pf3 wgs0 -> 256x256x16 sg16x4 pf3 wgs56 g128
      n091 [llm] N=1 score=1.0017 best-subtree=1.0017 In grad_weight (KGroupedKernel) I now merge tiles in the ragged last N column. When N % 25
    n087 [llm] N=3 score=1.0080 best-subtree=1.0080 I split the `gemm_tile` mainloop into two phases. The steady-state loop issues the current
      n094 [llm] N=1 score=0.6208 best-subtree=0.6208 I added a register double-buffered mainloop to `gemm_tile`, behind a new opt-in Cfg flag `
      n106 [knob] N=1 score=0.7010 best-subtree=0.7010 knob tile_k on m_colB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x64 sg8x4 pf2 wgs0 be2 s
    n114 [llm] N=1 score=0.9932 best-subtree=0.9932 grad_weight (k_grouped) now uses sg 16x2, so each sub-group owns a 16x128 slab instead of 
  n096 [llm] N=5 score=0.9977 best-subtree=0.9990 I added a `Cfg::kTailNoBarrier` knob to `gemm_tile` and set it to 2 for `k_grouped` (grad_
    n105 [knob] N=2 score=0.9579 best-subtree=0.9579 knob wgs on m_rowB: 256x256x32 sg8x4 pf3 wgs0 be2 si -> 256x256x32 sg8x4 pf3 wgs112 be2 si
      n108 [knob] N=1 score=0.9428 best-subtree=0.9428 knob skip_idle on m_rowB: 256x256x32 sg8x4 pf3 wgs112 be2 si -> 256x256x32 sg8x4 pf3 wgs11
    n104 [llm] N=2 score=0.9990 best-subtree=0.9990 I added a sub-group swizzle to `gemm_tile`. It applies to the 8x4 sub-group layout, which 
      n119 [llm] N=1 score=0.9969 best-subtree=0.9969 I added the wave-tail balancing that the forward and grad_input kernels already use to `KG
```

Widening at the end: alpha = 0.476 (root visits 121).
