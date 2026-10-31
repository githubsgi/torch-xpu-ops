#!/usr/bin/env python3
"""Turn `ggemm-eval profile` output into a REPORT.md with ranked, evidence-backed priorities.

  xpu-report make  --profile P [--profile P2 ...] --cand cand [--eval cand/last_eval.json]
                   --out profile/<run>/REPORT.md [--title "..."]
  xpu-report check profile/<run>/REPORT.md          # 3-5 priorities, each with Evidence / Expected impact / Effort / Why now

P is either the profile text saved from `ggemm-eval profile` (its `JOB: <dir>` line is followed to
<dir>/profile_result.json) or a profile_result.json itself.  Stdlib only (python >= 3.6).

What it computes beyond the digest
  * launch geometry: tiles per launch from the case's rows/expert and the candidate's tile
    (M-grouped: sum_g ceil(rows_g/tile_m)*ceil(N/tile_n); K-grouped: groups * ceil(M/tile_m)*ceil(N/tile_n)),
    waves = tiles / work-groups, wave efficiency = tiles / (ceil(waves) * work-groups), row fill,
    K tiles per output tile.
  * rule hits from the playbook (reference/playbook.md): each with the metric values that triggered it,
    a heuristic upper bound on the gain, effort, why-now, and the wiki pages to read.
  Gain heuristics are upper bounds, not predictions; the agent must overrule them with judgement
  (say so in the report when you do).
"""
import argparse
import json
import math
import os
import re
import sys

OPS = {  # op -> (kind, a, b)   M-grouped: a=K, b=N.  K-grouped: a=M (fixed output rows), b=N
    "w13_fwd": ("m_rowB", 2048, 2816), "w2_fwd": ("m_rowB", 1408, 2048),
    "w13_dX": ("m_colB", 2816, 2048), "w2_dX": ("m_colB", 2048, 1408),
    "w13_dW": ("k_grouped", 2816, 2048), "w2_dW": ("k_grouped", 2048, 1408),
}
REF = {"dX": "XMX 65-70%, stall ~7%", "fwd": "XMX ~65%, SBID 14%", "dW": "XMX 34-36%, SBID 31-37%"}


def load_profile(path):
    if path.endswith(".json"):
        return json.load(open(path)), path
    txt = open(path).read()
    m = re.search(r"^JOB:\s*(\S+)", txt, re.M)
    if not m:
        sys.exit("%s: no 'JOB: <dir>' line (is it the output of `ggemm-eval profile`?)" % path)
    jp = os.path.join(m.group(1), "profile_result.json")
    if not os.path.exists(jp):
        sys.exit("%s not found" % jp)
    return json.load(open(jp)), jp


def geometry(res, cfgs):
    t = res.get("timing") or {}
    op = t.get("op")
    if op not in OPS:
        return {}
    kind, a, b = OPS[op]
    cfg = (cfgs or {}).get(kind) or {}
    tile = cfg.get("tile") or [256, 256, 32]
    tm, tn, tk = tile
    counts = t.get("counts") or "[]"
    if isinstance(counts, str):
        counts = json.loads(counts)
    g = {"op": op, "kind": kind, "tile": tile, "prefetch": cfg.get("prefetch"), "groups": len(counts),
         "rows": sum(counts), "rows_mean": (sum(counts) / float(len(counts))) if counts else 0}
    n_t = int(math.ceil(b / float(tn)))
    if kind == "k_grouped":
        m_t = int(math.ceil(a / float(tm)))
        g["tiles"] = len(counts) * m_t * n_t
        ks = [int(math.ceil(c / float(tk))) for c in counts if c > 0]
        g["k_tiles_per_tile"] = sum(ks) / float(len(ks)) if ks else 0
        g["fill"] = 1.0
        g["empty_groups"] = sum(1 for c in counts if c == 0)
    else:
        g["tiles"] = sum(int(math.ceil(c / float(tm))) * n_t for c in counts)
        padded = sum(int(math.ceil(c / float(tm))) * tm for c in counts)
        g["fill"] = (sum(counts) / float(padded)) if padded else 1.0
        g["k_tiles_per_tile"] = int(math.ceil(a / float(tk)))
        g["empty_groups"] = sum(1 for c in counts if c == 0)
    return g


def wgs_of(m):
    """unitrace names the kernel `[SIMD16 {56; 1; 1} {512; 1; 1}]` = {work-groups} {work-items per group}.
    (If the first number is a global size, i.e. a large multiple of the second, divide.)"""
    mm = re.match(r"(\d+) x (\d+)", m.get("grid", "") or "")
    if not mm:
        return 0
    a, b = int(mm.group(1)), int(mm.group(2))
    return a // b if b and a % b == 0 and a // b >= 8 else a


def analyse(res, cfgs):
    """-> list of dicts, one per kernel in the profile, with metrics, geometry and rule hits."""
    out = []
    geo = geometry(res, cfgs)
    for k, m in (res.get("metrics") or {}).items():
        s = (res.get("stalls") or {}).get(k, {})
        p = (res.get("props") or {}).get(k, {})
        ctx = {"kernel": k, "m": m, "s": s, "p": p, "g": dict(geo), "timing": res.get("timing") or {}}
        w = wgs_of(m)
        g = ctx["g"]
        if w and g.get("tiles"):
            g["wgs"] = w
            g["waves"] = g["tiles"] / float(w)
            g["wave_eff"] = g["tiles"] / (math.ceil(g["waves"]) * w)
        ctx["hits"] = sorted([h for h in (r(ctx) for r in RULES) if h], key=lambda h: -h["gain"])
        out.append(ctx)
    return out


def num(x):
    try:
        return float(str(x).split()[0])
    except (ValueError, IndexError, TypeError):
        return 0.0


def _hit(rid, name, gain, ev, fix, effort, why, wiki):
    return {"id": rid, "name": name, "gain": gain, "evidence": ev, "fix": fix, "effort": effort, "why": why,
            "wiki": wiki}


def r_spill(c):
    sp = num(c["p"].get("Spill Memory Per Thread"))
    if sp > 0:
        return _hit("spill", "Register spills", 25.0, ["Spill Memory Per Thread = %.0f" % sp],
                    "revert the last register-growing change; smaller sub-group tile or prefetch distance",
                    "low", "spills usually dominate everything else", ["pat-register-spills", "hw-pvc-grf-threads"])


def r_short_k(c):
    g, m, s = c["g"], c["m"], c["s"]
    if g.get("kind") == "k_grouped" and g.get("k_tiles_per_tile", 99) <= 12 and (s.get("Sbid", 0) >= 15 or m["xmx_util"] < 45):
        gain = max(0.0, (65.0 - m["xmx_util"]) / 65.0 * 100.0 * 0.5)
        return _hit("short_k", "Short contraction: prologue/epilogue dominate the tile", gain,
                    ["K tiles per output tile = %.1f (rows/expert mean %.0f)" % (g["k_tiles_per_tile"], g["rows_mean"]),
                     "SBID %.0f%% of IP samples, XMX util %.0f%% (dX reference ~70%%), XVE active %.0f%% stall %.0f%%"
                     % (s.get("Sbid", 0), m["xmx_util"], m["xve_active"], m["xve_stall"]),
                     "HBM %.0f+%.0f GB/s, L3 %.0f GB/s: far from ceilings (1180 / 2840 GB/s)" %
                     (m["hbm_rd_gbs"], m["hbm_wr_gbs"], m["l3_rd_gbs"])],
                    "prefetch 1-2 for k_grouped (configs.json); then cross-tile prefetch / 128-row dW tile / "
                    "per-tile setup hoisting", "low (config) to high (kernel edit)",
                    "weakest op (~43% of the layer time at ~190 rows: 0.41+0.20 ms of 1.43 ms, from TFLOPS)",
                    ["pat-short-k-prologue-bound", "tech-prefetch-distance", "tech-split-k-stream-k",
                     "tech-epilogue-simplification"])


def r_load_latency(c):
    g, m, s = c["g"], c["m"], c["s"]
    if s.get("Sbid", 0) >= 12 and not (g.get("kind") == "k_grouped" and g.get("k_tiles_per_tile", 99) <= 12):
        gain = s.get("Sbid", 0) * 0.5
        return _hit("sbid", "Load-latency (SBID) stalls", gain,
                    ["SBID %.0f%% of IP samples; K tiles per tile %s" % (s["Sbid"], g.get("k_tiles_per_tile", "?")),
                     "XMX util %.0f%%, occupancy %.0f%%" % (m["xmx_util"], m["occupancy"])],
                    "retune prefetch distance for this kind; register double buffering if the build has no spills",
                    "low-medium", "SBID is the largest controllable stall class here",
                    ["pat-sbid-load-latency", "tech-prefetch-distance", "tech-register-double-buffering"])


def r_tail(c):
    g, m = c["g"], c["m"]
    idle = m["idle"]
    weff = g.get("wave_eff", 1.0)
    if idle >= 25 or (g.get("waves") and weff < 0.85 and idle >= 15):
        gain = max(idle - 12.0, (1 - weff) * 100.0 * 0.7)
        ev = ["XVE idle %.0f%%, occupancy %.0f%%" % (idle, m["occupancy"])]
        if "waves" in g:
            ev.append("tiles %d over %d work-groups = %.2f waves, wave efficiency %.0f%%; row fill %.0f%%; "
                      "empty groups %s" % (g["tiles"], g["wgs"], g["waves"], 100 * weff, 100 * g["fill"],
                                           g.get("empty_groups", "?")))
        return _hit("tail", "Idle XVEs: wave quantization / tail", gain, ev,
                    "smaller M tile for this kind (static), finer tiles at the end of the tile list, heavy-first "
                    "ordering from offs", "medium", "idle share is the second large loss for fwd/dX",
                    ["pat-tail-wave-quantization", "tech-tile-shape-selection", "tech-persistent-scheduling",
                     "tech-device-side-dispatch"])


def r_padding(c):
    g = c["g"]
    if g.get("kind") in ("m_rowB", "m_colB") and g.get("fill", 1) < 0.78:
        return _hit("padding", "Row-padding waste in edge tiles", (1 - g["fill"]) * 100.0 * 0.4,
                    ["row fill %.0f%% with tile_m=%d at mean %.0f rows/expert" % (100 * g["fill"], g["tile"][0], g["rows_mean"])],
                    "per-kind tile with smaller M; check the 'large' suite does not regress",
                    "low (config)", "directly measurable on the small suite",
                    ["pat-tile-padding-waste", "tech-tile-shape-selection", "ker-ggemm-regime-map"])


def r_xmx_underfed(c):
    m, s = c["m"], c["s"]
    if m["xve_active"] >= 60 and m["xmx_util"] < 50:
        return _hit("xmx_underfed", "XVEs busy, XMX underfed (non-DPAS instructions)", (65 - m["xmx_util"]) * 0.5,
                    ["XVE active %.0f%% but XMX util %.0f%%; send util %.1f%%; InstrFetch %.0f%% of IP samples" %
                     (m["xve_active"], m["xmx_util"], m["send_util"], s.get("InstrFetch", 0))],
                    "hoist per-tile setup, simplify epilogue, fewer/taller 2D messages", "medium",
                    "instruction overhead scales with tiles",
                    ["pat-xmx-underfed", "tech-epilogue-simplification", "tech-compiler-flags"])


def r_sync(c):
    s = c["s"]
    if s.get("Sync", 0) + s.get("Dist", 0) >= 15:
        return _hit("sync", "Barrier / dependency stalls", (s.get("Sync", 0) + s.get("Dist", 0)) * 0.4,
                    ["Sync %.0f%% + Dist %.0f%% of IP samples; barrier msgs %.0f" %
                     (s.get("Sync", 0), s.get("Dist", 0), c["m"]["barrier_msgs"])],
                    "smaller work-groups (16 sub-groups) or fewer barriers", "medium",
                    "only if the share is large", ["pat-barrier-sync-stalls", "lang-sycl-tla-split-barrier"])


def r_ifetch(c):
    s = c["s"]
    if s.get("InstrFetch", 0) >= 5:
        return _hit("ifetch", "Instruction-fetch stalls (code size)", s["InstrFetch"],
                    ["InstrFetch %.0f%% of IP samples" % s["InstrFetch"]],
                    "less unrolling, share the mainloop between paths", "low-medium", "code bloat grows with variants",
                    ["pat-instruction-fetch", "tech-compiler-flags"])


def r_clock(c):
    if 0 < c["m"]["mhz"] < 1100:
        return _hit("clock", "Clock below 1.1 GHz (throttling)", 5.0, ["AvgGpuCoreFrequencyMHz = %.0f" % c["m"]["mhz"]],
                    "rerun; compare on another tile; do not compare profiles taken at different clocks", "none",
                    "invalidates comparisons", ["hw-pvc-xe-core"])


def r_membound(c):
    m = c["m"]
    if m["hbm_rd_gbs"] + m["hbm_wr_gbs"] > 800 or m["l3_rd_gbs"] > 2000:
        return _hit("membound", "Memory bandwidth is near a ceiling", 15.0,
                    ["HBM %.0f GB/s, L3 %.0f GB/s" % (m["hbm_rd_gbs"] + m["hbm_wr_gbs"], m["l3_rd_gbs"])],
                    "larger tiles / better tile order for L3 reuse", "medium", "new bottleneck for small tiles",
                    ["pat-memory-bound-expected", "hw-pvc-memory-hierarchy"])


RULES = [r_spill, r_short_k, r_load_latency, r_tail, r_padding, r_xmx_underfed, r_sync, r_ifetch, r_clock, r_membound]


def fmt_report(title, items, cand, evalj, profile_paths):
    L = ["# %s" % title, ""]
    L += ["**Target hardware:** Intel Data Center GPU Max (PVC), one tile (FLAT), bf16 -> fp32 accumulate -> bf16",
          "**Profiler:** unitrace (`-g ComputeBasic` per-instance counters + `--stall-sampling`), via `ggemm-eval profile`",
          "**Candidate:** `%s`" % cand, "**Profiles:** " + ", ".join("`%s`" % p for p in profile_paths), "", "---", ""]
    L += ["## 0. Profiling setup", "",
          "- Harness: `agent_eval --mode profile` (candidate only, 20 launches of one (case, op)); the raw unitrace logs "
          "(`ut_metrics.log`, `ut_stalls.log`) are in the JOB directories printed by each profile.",
          "- Cases/ops: " + "; ".join("%s / %s" % (it["timing"].get("case"), it["timing"].get("op")) for it in items),
          "- Counter medians over instances; stall shares are fractions of stall *samples*, not of time.", "",
          "## 1. Headline numbers", ""]
    hdr = "| Metric | " + " | ".join("%s %s" % (it["timing"].get("op"), it["timing"].get("case", "")[:12]) for it in items) + " |"
    L += [hdr, "|---|" + "---:|" * len(items)]

    def row(name, f):
        L.append("| %s | %s |" % (name, " | ".join(f(it) for it in items)))
    row("ms / call", lambda it: "%.4f" % it["timing"].get("ms", 0))
    row("TFLOPS", lambda it: "%.0f" % it["timing"].get("tflops", 0))
    row("GPU kernel us (median)", lambda it: "%.0f" % it["m"]["gpu_us"])
    row("XVE active / stall / idle %", lambda it: "%.0f / %.0f / %.0f" % (it["m"]["xve_active"], it["m"]["xve_stall"], it["m"]["idle"]))
    row("occupancy %", lambda it: "%.0f" % it["m"]["occupancy"])
    row("XMX util %", lambda it: "%.0f" % it["m"]["xmx_util"])
    row("SBID % of IP samples", lambda it: "%.0f" % it["s"].get("Sbid", 0))
    row("HBM rd+wr / L3 rd GB/s", lambda it: "%.0f / %.0f" % (it["m"]["hbm_rd_gbs"] + it["m"]["hbm_wr_gbs"], it["m"]["l3_rd_gbs"]))
    row("clock MHz", lambda it: "%.0f" % it["m"]["mhz"])
    row("tile / work-groups / prefetch", lambda it: "%s / %s / %s" % ("x".join(str(x) for x in it["g"].get("tile", [])), it["g"].get("wgs", "?"), it["g"].get("prefetch", "?")))
    row("tiles, waves, wave eff.", lambda it: ("%d, %.2f, %.0f%%" % (it["g"]["tiles"], it["g"]["waves"], 100 * it["g"]["wave_eff"])) if "waves" in it["g"] else "?")
    row("K tiles per tile / row fill", lambda it: "%s / %.0f%%" % (("%.1f" % it["g"]["k_tiles_per_tile"]) if it["g"].get("k_tiles_per_tile") is not None else "?", 100 * it["g"].get("fill", 1)))
    row("spill mem / thread", lambda it: str(it["p"].get("Spill Memory Per Thread", "?")))
    L += ["", "Reference (production, TP4/EP4): " + "; ".join("%s: %s" % kv for kv in REF.items()), ""]
    if evalj:
        L += ["Latest evaluation of this candidate: suite `%s`, status `%s`, score %.4f, per-op %s" %
              (evalj.get("suite"), evalj.get("status"), evalj.get("score", 0),
               ", ".join("%s %.3f" % kv for kv in (evalj.get("per_op") or {}).items())), ""]
    L += ["**One-line read:** TODO(agent): one sentence naming the two or three metric values that decide the diagnosis "
          "(e.g. 'latency-bound on 2D loads: SBID x%, XMX y%, HBM z% of ceiling', not 'looks memory-bound').", "", "---", "",
          "## 2. Per-dimension findings (auto-generated; refine)", ""]
    for it in items:
        g = it["g"]
        L += ["### %s / %s" % (it["timing"].get("op"), it["timing"].get("case")),
              "- launch geometry: %s tiles, %s work-groups, waves %s, wave efficiency %s" % (
                  g.get("tiles"), g.get("wgs"), "%.2f" % g["waves"] if "waves" in g else "?",
                  "%.0f%%" % (100 * g["wave_eff"]) if "wave_eff" in g else "?"),
              "- stalls: " + ", ".join("%s %.0f%%" % kv for kv in list(it["s"].items())[:6]),
              "- tensor: XMX %.0f%% (XVE active %.0f%%)" % (it["m"]["xmx_util"], it["m"]["xve_active"]),
              "- memory: HBM %.0f GB/s, L3 %.0f GB/s; spills %s" % (it["m"]["hbm_rd_gbs"] + it["m"]["hbm_wr_gbs"], it["m"]["l3_rd_gbs"], it["p"].get("Spill Memory Per Thread", "?")), ""]
    L += ["---", "", "## 3. Summary diagnosis", "", "| Factor | " + " | ".join("%s" % it["timing"].get("op") for it in items) + " |",
          "|---|" + "---|" * len(items)]
    ids = []
    for it in items:
        for h in it["hits"]:
            if h["id"] not in ids:
                ids.append(h["id"])
    for i in ids:
        cells = []
        for it in items:
            h = next((x for x in it["hits"] if x["id"] == i), None)
            cells.append("%.0f%% (upper bound)" % h["gain"] if h else "-")
        nm = next(h["name"] for it in items for h in it["hits"] if h["id"] == i)
        L.append("| %s | %s |" % (nm, " | ".join(cells)))
    L += ["", "---", "", "## 4. Optimization directions (ranked by evidence-weighted impact)", ""]
    # merge hits across items, keep the highest gain per id, cite which op
    best = {}
    for it in items:
        for h in it["hits"]:
            cur = best.get(h["id"])
            if cur is None or h["gain"] > cur[0]["gain"]:
                best[h["id"]] = (h, it)
    ranked = sorted(best.values(), key=lambda x: -x[0]["gain"])[:5]
    if not ranked:
        L += ["No playbook rule fired. Look at tile fill and the per-op TFLOPS in the latest eval; a balanced kernel "
              "(dX-like: XMX 65-70%, stall 7%) is at the ceiling of this design.", ""]
    for n, (h, it) in enumerate(ranked, 1):
        L += ["### Priority %d - %s" % (n, h["name"]), "",
              "First-line change: %s." % h["fix"], "",
              "**Evidence:** (%s / %s)" % (it["timing"].get("op"), it["timing"].get("case"))] + ["- " + e for e in h["evidence"]] + [
              "", "**Expected impact:** upper bound ~%.0f%% on %s (heuristic; TODO(agent): overrule with your own estimate and say why)" % (h["gain"], it["timing"].get("op")),
              "", "**Effort:** %s" % h["effort"], "", "**Why now:** %s" % h["why"], "",
              "**Wiki:** " + ", ".join("`%s`" % w for w in h["wiki"]), ""]
    L += ["---", "", "## 5. Confidence & caveats", "",
          "- Sure about: the counter values above (instance medians, 20 launches).",
          "- Uncertain: gain bounds are heuristics (see make_report.py docstring); stall shares come from a separate pass.",
          "- TODO(agent): what the profile could not answer and which experiment resolves it.", "",
          "## 6. Reproduction", "", "    ggemm-eval profile %s --case <case> --op <op> | tee profile/<run>/profile_<case>_<op>.txt" % cand,
          "    xpu-report make --profile profile/<run>/profile_<case>_<op>.txt --cand %s --out profile/<run>/REPORT.md" % cand, ""]
    return "\n".join(L)


def check_report(path):
    txt = open(path).read()
    pr = re.split(r"^### Priority \d+", txt, flags=re.M)[1:]
    errs = []
    if not 1 <= len(pr) <= 5:
        errs.append("%d priorities (need 3-5; fewer only if fewer rules fired)" % len(pr))
    for i, p in enumerate(pr, 1):
        for key in ("**Evidence:**", "**Expected impact:**", "**Effort:**", "**Why now:**"):
            if key not in p:
                errs.append("priority %d lacks %s" % (i, key))
        if not re.search(r"\d", p.split("**Effort:**")[0]):
            errs.append("priority %d has no numbers in its evidence" % i)
    if "TODO(agent)" in txt:
        errs.append("TODO(agent) placeholders remain: replace them with your judgement")
    for e in errs:
        print("REPORT CHECK:", e)
    print("REPORT CHECK:", "FAILED" if errs else "ok", "(%s)" % path)
    return 1 if errs else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--profile", action="append", required=True)
    p.add_argument("--cand", default="cand")
    p.add_argument("--eval")
    p.add_argument("--out", required=True)
    p.add_argument("--title", default="unitrace profile report")
    p = sub.add_parser("check")
    p.add_argument("report")
    a = ap.parse_args()
    if a.cmd == "check":
        return check_report(a.report)
    cfgs = {}
    cf = os.path.join(a.cand, "configs.json")
    if os.path.exists(cf):
        cfgs = json.load(open(cf))
    ev = a.eval or os.path.join(a.cand, "last_eval.json")
    evalj = json.load(open(ev)) if os.path.exists(ev) else None
    items = []
    for pth in a.profile:
        res, _ = load_profile(pth)
        items += analyse(res, cfgs)
    if not items:
        sys.exit("no candidate kernel metrics in the profile(s): the profile failed (see the JOB dir logs)")
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    open(a.out, "w").write(fmt_report(a.title, items, a.cand, evalj, a.profile))
    print("wrote %s: %d kernel(s); top rules: %s" % (a.out, len(items),
          ", ".join("%s(%.0f%%)" % (h["id"], h["gain"]) for it in items for h in it["hits"][:2])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
