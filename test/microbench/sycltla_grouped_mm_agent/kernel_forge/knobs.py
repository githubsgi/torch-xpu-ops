"""Knob mutations: cheap, no LLM. Mutate ONE aspect of ONE kind's tile config.

Aspects: tile (M x N, sub-group layout re-derived), sg (layout only), prefetch (1-4),
tile_k (32 <-> 64; 16 <-> 32 for grad_weight), wgs (persistent grid, multiples of the 56 Xe
cores of a PVC tile), and the Cfg-member knobs of the v4 / extended header: barrier
(kBarrierEvery), skip_idle (kSkipIdleSg), cache_a / cache_b (LSC cache hint of the A / B
2D block loads), cache_pf (prefetch hint), grf (128 / 256 register file; 128 doubles the
threads per Xe core, so it is paired with a 16x64 or 32x32 sub-group slab and up to 64
sub-groups).
Validity is the evaluator's `candidate.normalize` plus the sub-group-tile limits below
(a sub-group owns a multiple of the 8x16 DPAS tile; all production tiles use 32x64).
Configs that still fail to compile come back as ordinary build errors.
"""

import copy
import json

from . import _paths  # noqa: F401
from ggemm_agent import candidate

TILES = [(128, 128), (128, 256), (256, 128), (256, 256), (64, 128), (64, 256), (512, 128), (256, 64)]
SG_TILE_M = (16, 32, 64)
SG_TILE_N = (32, 64)
PREFETCH = (1, 2, 3, 4)
TILE_K = (16, 32, 64)
BARRIER = (1, 2, 3, 4)
# Measured 2026-11-25 on the v4 baseline (notes_ggemm_kernel.md section 15): .uc.* hints (0.48x),
# L3-only prefetch (0.63x) and GRF-128 (0.46-0.65x) are far below production, so the mutation
# operator leaves them out; .ca.ca / .st.ca and GRF-256 stay. The LLM may still set them via
# configs.json if it has a reason.
CACHE = (0, 1, 2)                # default, ca.ca, st.ca
CACHE_PF = (0,)                  # ca.ca only
GRF = (256,)
XE_CORES = 56
WGS = (0, XE_CORES, 2 * XE_CORES, 3 * XE_CORES, 4 * XE_CORES)       # 0 = production rule
KIND_WEIGHT = {"k_grouped": 0.4, "m_rowB": 0.3, "m_colB": 0.3}      # grad_weight is the weakest op
ASPECT_WEIGHT = {"tile": 0.22, "sg": 0.10, "prefetch": 0.16, "tile_k": 0.10, "wgs": 0.10,
                 "barrier": 0.12, "skip_idle": 0.04, "cache_a": 0.08, "cache_b": 0.08, "cache_pf": 0.0,
                 "grf": 0.0}
# Only the kinds where the knob can matter: skip_idle needs partial tiles (M-grouped ragged rows).
ASPECT_KINDS = {"skip_idle": ("m_rowB", "m_colB")}


def max_sgs(grf):
    """Hardware threads per Xe core: 32 in GRF-256 mode, 64 in GRF-128 mode."""
    return 32 if grf == 256 else 64


def max_slab_elems(grf):
    """fp32 accumulator budget per lane: 128 regs x 16 lanes = 2048 elements (32x64) in GRF-256;
    half that in GRF-128 (64 of the 128 regs for the accumulator, the rest for A/B fragments)."""
    return 2048 if grf == 256 else 1024


def sg_layouts(tm, tn, grf=256):
    """(sgM, sgN) layouts whose sub-group tile is a legal DPAS multiple within the thread and
    register budget of `grf`."""
    out = []
    for stm in SG_TILE_M:
        for stn in SG_TILE_N:
            if tm % stm or tn % stn or stm * stn > max_slab_elems(grf):
                continue
            sm, sn = tm // stm, tn // stn
            if 4 <= sm * sn <= max_sgs(grf):
                out.append((sm, sn))
    return out


def _pick_sg(rng, tm, tn, grf=256, exclude=None):
    opts = [s for s in sg_layouts(tm, tn, grf) if s != exclude]
    if not opts:
        return None
    pref_slab = (32, 64) if grf == 256 else (16, 64)
    pref = [s for s in opts if (tm // s[0], tn // s[1]) == pref_slab]
    return rng.choice(pref) if pref and rng.random() < 0.6 else rng.choice(opts)


def _choice_ne(rng, opts, cur):
    opts = [o for o in opts if o != cur]
    return rng.choice(opts) if opts else None


def mutate(cfgs, rng, tries=30):
    """Return (new_cfgs, description) or (None, reason). `cfgs` is a normalized configs dict."""
    kinds = [k for k in candidate.KINDS if "body" not in cfgs[k]]
    if not kinds:
        return None, "all kinds use a custom body"
    for _ in range(tries):
        new = copy.deepcopy(cfgs)
        kind = rng.choices(kinds, [KIND_WEIGHT[k] for k in kinds])[0]
        c = new[kind]
        aspect = rng.choices(list(ASPECT_WEIGHT), list(ASPECT_WEIGHT.values()))[0]
        if aspect in ASPECT_KINDS and kind not in ASPECT_KINDS[aspect]:
            continue
        (tm, tn, tk), (sm, sn) = c["tile"], c["sg"]
        grf = c.get("grf", 256)
        before = _fmt(c)
        if aspect == "tile":
            t = _choice_ne(rng, TILES, (tm, tn))
            sg = _pick_sg(rng, *t, grf) if t else None
            if not sg:
                continue
            c["tile"], c["sg"] = [t[0], t[1], tk], list(sg)
        elif aspect == "sg":
            sg = _pick_sg(rng, tm, tn, grf, exclude=(sm, sn))
            if not sg:
                continue
            c["sg"] = list(sg)
        elif aspect == "prefetch":
            c["prefetch"] = _choice_ne(rng, PREFETCH, c["prefetch"])
        elif aspect == "tile_k":
            opts = TILE_K if kind == "k_grouped" else (32, 64)
            nk = _choice_ne(rng, opts, tk)
            c["tile"] = [tm, tn, nk]
            if nk == 64 and c["prefetch"] > 2:
                c["prefetch"] = 2                    # tuner: k64 wants a shorter prefetch
        elif aspect == "wgs":
            c["wgs"] = _choice_ne(rng, WGS, c["wgs"])
        elif aspect == "barrier":
            c["barrier_every"] = _choice_ne(rng, BARRIER, c.get("barrier_every", 1))
        elif aspect == "skip_idle":
            c["skip_idle"] = not c.get("skip_idle", False)
        elif aspect in ("cache_a", "cache_b"):
            v = _choice_ne(rng, CACHE, c.get(aspect, 0))
            if v is None:
                continue
            c[aspect] = v
        elif aspect == "cache_pf":
            v = _choice_ne(rng, CACHE_PF, c.get("cache_pf", 0))
            if v is None:
                continue
            c["cache_pf"] = v
        elif aspect == "grf":
            ng = 128 if grf == 256 else 256
            if ng not in GRF:
                continue
            sg = _pick_sg(rng, tm, tn, ng)          # the slab must fit the new register budget
            if not sg:
                continue
            c["grf"], c["sg"] = ng, list(sg)
            if ng == 128 and c["wgs"] == 0:
                # the production grid rule assumes 512 threads per core; fill the 1024 of GRF-128
                c["wgs"] = XE_CORES * 1024 // (sg[0] * sg[1] * 16)
        if c["wgs"] == XE_CORES * 512 // (c["sg"][0] * c["sg"][1] * 16):
            c["wgs"] = 0                             # explicit value equal to the production rule
        _drop_defaults(c)
        try:
            new = candidate.normalize(new)
        except candidate.CandidateError:
            continue
        if json.dumps(new, sort_keys=True) == json.dumps(cfgs, sort_keys=True):
            continue
        return new, f"knob {aspect} on {kind}: {before} -> {_fmt(new[kind])}"
    return None, "no valid mutation found"


def _drop_defaults(c):
    """Knobs at their header default need no Cfg member (keeps the generated struct minimal)."""
    for key, default in (("cache_a", 0), ("cache_b", 0), ("cache_pf", 0), ("grf", 256)):
        if c.get(key) == default:
            del c[key]


def _fmt(c):
    (tm, tn, tk), (sm, sn) = c["tile"], c["sg"]
    s = f"{tm}x{tn}x{tk} sg{sm}x{sn} pf{c['prefetch']} wgs{c['wgs']}"
    tags = candidate.knob_desc(c)
    return s + (" " + tags if tags else "")


def config_diff(old, new):
    """One-line description of what changed between two normalized configs."""
    parts = []
    for k in candidate.KINDS:
        a, b = old.get(k, {}), new.get(k, {})
        keys = ("tile", "sg", "prefetch", "wgs", "extra", "body") + tuple(candidate.KNOB_MEMBERS)
        ch = [f"{f} {a.get(f)}->{b.get(f)}" for f in keys if a.get(f) != b.get(f)]
        if ch:
            parts.append(f"{k}: " + ", ".join(ch))
    if old.get("extra_flags") != new.get("extra_flags"):
        parts.append(f"extra_flags {old.get('extra_flags')}->{new.get('extra_flags')}")
    return "; ".join(parts) if parts else "no config change"
