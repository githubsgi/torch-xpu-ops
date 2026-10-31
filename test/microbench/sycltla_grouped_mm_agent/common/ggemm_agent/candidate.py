"""Candidate kernels: a directory an agent edits.

    <cand>/grouped_mm_kernel_impl.h   the kernel header (starts as a copy of
                                      production; may be edited freely as long
                                      as the contract in eval/cand_cfg.cpp holds)
    <cand>/configs.json               one tile config per kernel kind

configs.json:
    {
      "m_rowB":    {"tile": [256, 256, 32], "sg": [8, 4], "prefetch": 3, "wgs": 0},
      "m_colB":    {...},          # forward reads B K-contiguous ('R'), dX N-contiguous ('C')
      "k_grouped": {...},          # grad_weight
      "extra_flags": []            # extra icpx flags, e.g. ["-Xs", "-options -ze-opt-..."]
    }
Per kind, optional: "extra" (C++ member lines appended to the generated Cfg
struct, e.g. "static constexpr int kSplitK = 2;") or "body" (the whole struct
body, replacing the generated WGTile / SGLayout / k* members). "wgs" 0 means
the production rule (Xe cores x 512 / work-group threads).
"""

import hashlib
import json
import shutil
from pathlib import Path

from . import config

KINDS = ("m_rowB", "m_colB", "k_grouped")
KIND_MACRO = {"m_rowB": "CAND_MROW", "m_colB": "CAND_MCOL", "k_grouped": "CAND_K"}
# Bump when the build recipe changes so cached .so files are not reused.
BUILD_RECIPE = "v1"

PRODUCTION_CFG = {"tile": [256, 256, 32], "sg": [8, 4], "prefetch": 3, "wgs": 0}


def production_configs():
    """Production on PVC: choose_tile() = 256x256 for every call, default grid."""
    return {k: dict(PRODUCTION_CFG) for k in KINDS} | {"extra_flags": []}


class CandidateError(ValueError):
    pass


def init_dir(path, src=None):
    """Create a candidate dir from production (src=None) or another candidate."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if src is None:
        shutil.copy(config.PROD_HEADER, path / "grouped_mm_kernel_impl.h")
        (path / "configs.json").write_text(json.dumps(production_configs(), indent=2) + "\n")
    else:
        src = Path(src)
        for f in ("grouped_mm_kernel_impl.h", "configs.json"):
            shutil.copy(src / f, path / f)
    return path


def load(path):
    path = Path(path)
    hdr = path / "grouped_mm_kernel_impl.h"
    cfgf = path / "configs.json"
    if not hdr.exists():
        raise CandidateError(f"{hdr} missing")
    cfgs = json.loads(cfgf.read_text()) if cfgf.exists() else production_configs()
    return hdr.read_text(), normalize(cfgs)


def normalize(cfgs):
    out = {"extra_flags": list(cfgs.get("extra_flags", []))}
    for k in KINDS:
        c = dict(PRODUCTION_CFG) | dict(cfgs.get(k, {}))
        if "body" not in c:
            tm, tn, tk = (int(x) for x in c["tile"])
            sm, sn = (int(x) for x in c["sg"])
            if tm % sm or tn % sn:
                raise CandidateError(f"{k}: tile {tm}x{tn} not divisible by sg {sm}x{sn}")
            if (tm // sm) % 8 or (tn // sn) % 16:
                raise CandidateError(f"{k}: sub-group tile {tm // sm}x{tn // sn} must be a multiple of 8x16 (DPAS)")
            if sm * sn > 64:
                raise CandidateError(f"{k}: {sm * sn} sub-groups > 64 (1024 work-items)")
            c["tile"], c["sg"], c["prefetch"] = [tm, tn, tk], [sm, sn], int(c["prefetch"])
        c["wgs"] = int(c.get("wgs", 0))
        out[k] = c
    return out


def cfg_struct(name, c):
    if "body" in c:
        body = c["body"]
    else:
        (tm, tn, tk), (sm, sn), pf = c["tile"], c["sg"], c["prefetch"]
        body = (f"  using WGTile = Shape<Int<{tm}>, Int<{tn}>, Int<{tk}>>;\n"
                f"  using SGLayout = Layout<Shape<Int<{sm}>, Int<{sn}>, _1>, Stride<Int<{sn}>, _1, _0>>;\n"
                f"  static constexpr int kTileM = {tm};\n"
                f"  static constexpr int kTileN = {tn};\n"
                f"  static constexpr int kTileK = {tk};\n"
                f"  static constexpr int kPrefetchDist = {pf};\n"
                f"  static constexpr const char* kDesc = \"{tm}x{tn}x{tk} sg{sm}x{sn} pf{pf}\";\n")
    if c.get("extra"):
        body += "  " + c["extra"].strip() + "\n"
    return ("#pragma once\n#include <cute/tensor.hpp>\nnamespace tune {\nusing namespace cute;\n"
            f"struct {name} {{\n{body}}};\n}} // namespace tune\n")


def build_units(header_text, cfgs):
    """Distinct .so files needed: kinds sharing a config share one .so.

    Returns [{"hash", "name", "kinds", "cfg_header", "flags"}] plus the
    kind -> (unit hash, wgs) map."""
    groups = {}
    for k in KINDS:
        c = {x: v for x, v in cfgs[k].items() if x != "wgs"}
        groups.setdefault(json.dumps(c, sort_keys=True), []).append(k)
    units, kind_map = [], {}
    hdr_hash = hashlib.sha256(header_text.encode() + b"".join(
        f.read_bytes() for f in (config.EVAL_DIR / "cand_cfg.cpp", config.TUNER_DIR / "tune_registry.h",
                                 config.SRC_DIR / "grouped_mm_kernels.h"))).hexdigest()
    for key, kinds in groups.items():
        c = json.loads(key)
        h = hashlib.sha256("\0".join(
            [BUILD_RECIPE, hdr_hash, key, ",".join(kinds), json.dumps(cfgs["extra_flags"]),
             config.AOT_DEVICE]).encode()).hexdigest()[:16]
        name = f"cand_{h}"
        units.append({"hash": h, "name": name, "kinds": kinds, "cfg_header": cfg_struct(name, c),
                      "flags": cfgs["extra_flags"]})
        for k in kinds:
            kind_map[k] = (h, cfgs[k]["wgs"])
    return units, kind_map


def diff_vs_production(path):
    import difflib
    a = config.PROD_HEADER.read_text().splitlines(keepends=True)
    b = (Path(path) / "grouped_mm_kernel_impl.h").read_text().splitlines(keepends=True)
    return "".join(difflib.unified_diff(a, b, "production/grouped_mm_kernel_impl.h",
                                        "candidate/grouped_mm_kernel_impl.h"))
