"""Knob mutations: cheap, no LLM. Mutate ONE aspect of ONE kind's tile config.

Aspects: tile (M x N, sub-group layout re-derived), sg (layout only), prefetch (1-4),
tile_k (32 <-> 64), wgs (persistent grid, multiples of the 56 Xe cores of a PVC tile).
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
TILE_K = (32, 64)
XE_CORES = 56
WGS = (0, XE_CORES, 2 * XE_CORES, 3 * XE_CORES, 4 * XE_CORES)       # 0 = production rule
KIND_WEIGHT = {"k_grouped": 0.4, "m_rowB": 0.3, "m_colB": 0.3}      # grad_weight is the weakest op
ASPECT_WEIGHT = {"tile": 0.3, "sg": 0.15, "prefetch": 0.25, "tile_k": 0.15, "wgs": 0.15}


def sg_layouts(tm, tn):
    """All (sgM, sgN) with sub-group tile in SG_TILE_M x SG_TILE_N and 4..32 sub-groups."""
    out = []
    for stm in SG_TILE_M:
        for stn in SG_TILE_N:
            if tm % stm == 0 and tn % stn == 0:
                sm, sn = tm // stm, tn // stn
                if 4 <= sm * sn <= 32:
                    out.append((sm, sn))
    return out


def _pick_sg(rng, tm, tn, exclude=None):
    opts = [s for s in sg_layouts(tm, tn) if s != exclude]
    if not opts:
        return None
    pref = [s for s in opts if (tm // s[0], tn // s[1]) == (32, 64)]
    return rng.choice(pref if pref and rng.random() < 0.6 else opts)


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
        (tm, tn, tk), (sm, sn) = c["tile"], c["sg"]
        before = f"{tm}x{tn}x{tk} sg{sm}x{sn} pf{c['prefetch']} wgs{c['wgs']}"
        if aspect == "tile":
            t = _choice_ne(rng, TILES, (tm, tn))
            sg = _pick_sg(rng, *t) if t else None
            if not sg:
                continue
            c["tile"], c["sg"] = [t[0], t[1], tk], list(sg)
        elif aspect == "sg":
            sg = _pick_sg(rng, tm, tn, exclude=(sm, sn))
            if not sg:
                continue
            c["sg"] = list(sg)
        elif aspect == "prefetch":
            c["prefetch"] = _choice_ne(rng, PREFETCH, c["prefetch"])
        elif aspect == "tile_k":
            nk = 64 if tk == 32 else 32
            c["tile"] = [tm, tn, nk]
            if nk == 64 and c["prefetch"] > 2:
                c["prefetch"] = 2                    # tuner: k64 wants a shorter prefetch
        elif aspect == "wgs":
            c["wgs"] = _choice_ne(rng, WGS, c["wgs"])
        if c["wgs"] == XE_CORES * 512 // (c["sg"][0] * c["sg"][1] * 16):
            c["wgs"] = 0                             # explicit value equal to the production rule
        try:
            new = candidate.normalize(new)
        except candidate.CandidateError:
            continue
        if json.dumps(new, sort_keys=True) == json.dumps(cfgs, sort_keys=True):
            continue
        a = new[kind]
        after = f"{a['tile'][0]}x{a['tile'][1]}x{a['tile'][2]} sg{a['sg'][0]}x{a['sg'][1]} pf{a['prefetch']} wgs{a['wgs']}"
        return new, f"knob {aspect} on {kind}: {before} -> {after}"
    return None, "no valid mutation found"


def config_diff(old, new):
    """One-line diff of two normalized configs dicts."""
    parts = []
    for k in candidate.KINDS:
        a, b = old.get(k, {}), new.get(k, {})
        ch = [f"{f} {a.get(f)}->{b.get(f)}" for f in ("tile", "sg", "prefetch", "wgs", "extra", "body")
              if a.get(f) != b.get(f)]
        if ch:
            parts.append(f"{k}: " + ", ".join(str(x)[:120] for x in ch))
    if old.get("extra_flags") != new.get("extra_flags"):
        parts.append(f"extra_flags {old.get('extra_flags')}->{new.get('extra_flags')}")
    return "; ".join(parts) or "configs unchanged"
