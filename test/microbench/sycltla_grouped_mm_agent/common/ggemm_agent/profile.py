"""Hardware profile of one candidate (case, op) with unitrace: the XPU stand-in for ncu.

    res = profile("path/to/cand", case="tp4ep4_uniform", op="w13_dW")
    res["text"]    digest for an LLM: counters, stall mix, kernel properties, and
                   ranked diagnoses from the rules in DIAGNOSES

Three unitrace passes over `agent_eval --mode profile` (candidate only):
  -d -h -q -g ComputeBasic   per-instance counters (XVE active/stall, occupancy,
                             XMX utilization, HBM / L3 bytes) and the driver's kernel
                             properties table (SIMD, SLM, private / spill memory, GRF)
  --stall-sampling           per-instruction stall reasons (SBID, Pipe, Dist, Sync, ...)
"""

import csv
import io
import json
import re
import statistics
import time
from collections import defaultdict
from pathlib import Path

from . import config


# ----------------------------------------------------------- node side ---
def _unitrace(job_dir, tile, args, tag, driver_args):
    from .node_eval import run_driver
    out_dir = Path(job_dir) / f"ut_{tag}"
    out_dir.mkdir(exist_ok=True)
    rc, out, err = run_driver(job_dir, tile, [config.UNITRACE] + args + ["--output-dir-path", str(out_dir)]
                              + driver_args, config.RUN_TIMEOUT_S)
    (Path(job_dir) / f"ut_{tag}.log").write_text(out + "\n--- stderr ---\n" + err)
    return rc, out + "\n" + err


def phase_profile(job_dir, tile):
    t0 = time.time()
    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    b = json.loads((job_dir / "build.json").read_text())
    (job_dir / "cand.manifest").write_text(b["cand_manifest"])
    drv = [b["driver"], "--cand", "cand.manifest", "--mode", "profile", "--profile-case", job["case"],
           "--profile-op", job["op"], "--profile-iters", str(job.get("iters", 20)), "--json", "profile.json"]
    rc1, log1 = _unitrace(job_dir, tile, ["-d", "-h", "-q", "-g", "ComputeBasic"], "metrics", drv)
    rc2, log2 = _unitrace(job_dir, tile, ["--stall-sampling"], "stalls", drv)
    res = {"ok": rc1 == 0, "seconds": round(time.time() - t0, 1), "rc": [rc1, rc2],
           "metrics": parse_metrics(log1), "props": parse_props(log1), "stalls": parse_stalls(log2)}
    try:
        res["timing"] = json.loads((job_dir / "profile.json").read_text()).get("profile")
    except (FileNotFoundError, json.JSONDecodeError):
        res["timing"] = None
    if not res["metrics"]:
        res["ok"] = False
        res["log_tail"] = log1[-3000:]
    (job_dir / "profile_result.json").write_text(json.dumps(res, indent=1))
    return res


def _is_cand_kernel(name):
    return "cand_" in name or "Grouped" in name


def _short(name):
    m = re.search(r"(\w+Kernel)<[^,]*, *\(char\)(\d+)", name)
    if m:
        return f"{m[1]}<B={'K-contig' if m[2] == '82' else 'N-contig'}>"
    m = re.search(r"(\w+Kernel)<", name)
    return m[1] if m else name[:80]


def parse_metrics(log):
    lines = log.splitlines()
    try:
        start = next(i for i, l in enumerate(lines) if l.startswith("Kernel,GlobalInstanceId"))
    except StopIteration:
        return {}
    body = [lines[start]]
    for l in lines[start + 1:]:
        if not l.strip():
            continue
        if not l.startswith('"'):
            break
        body.append(l)
    groups = defaultdict(list)
    for r in csv.DictReader(io.StringIO("\n".join(body))):
        try:
            float(r["GpuTime[ns]"])
        except (KeyError, ValueError, TypeError):
            continue
        if _is_cand_kernel(r["Kernel"]):
            groups[_short(r["Kernel"])].append(r)
    f = lambda r, k: float(r.get(k) or 0)  # noqa: E731
    med = lambda rs, k: statistics.median(f(r, k) for r in rs)  # noqa: E731
    out = {}
    for k, rs in groups.items():
        bw = lambda key: statistics.median(f(r, key) / max(f(r, "GpuTime[ns]"), 1) for r in rs)  # noqa: E731
        launch = re.search(r"\{(\d+); 1; 1\} \{(\d+); 1; 1\}", rs[0]["Kernel"])
        out[k] = {"n": len(rs), "gpu_us": med(rs, "GpuTime[ns]") / 1e3,
                  "xve_active": med(rs, "XVE_ACTIVE[%]"), "xve_stall": med(rs, "XVE_STALL[%]"),
                  "occupancy": med(rs, "XVE_THREADS_OCCUPANCY_ALL[%]"),
                  "xmx_util": med(rs, "XVE_INST_EXECUTED_XMX_ALL_UTILIZATION[%]"),
                  "send_util": med(rs, "XVE_INST_EXECUTED_SEND_ALL_UTILIZATION[%]"),
                  "hbm_rd_gbs": bw("GPU_MEMORY_BYTE_READ[bytes]"), "hbm_wr_gbs": bw("GPU_MEMORY_BYTE_WRITE[bytes]"),
                  "l3_rd_gbs": bw("L3_BYTE_READ[bytes]"), "mhz": med(rs, "AvgGpuCoreFrequencyMHz[MHz]"),
                  "barrier_msgs": med(rs, "XVE_BARRIER_MESSAGE_COUNT[messages]"),
                  "grid": f"{launch[1]} x {launch[2]}" if launch else ""}
        out[k]["idle"] = max(0.0, 100 - out[k]["xve_active"] - out[k]["xve_stall"])
    return out


def parse_props(log):
    lines = log.splitlines()
    try:
        i = next(i for i, l in enumerate(lines) if l.strip().startswith("Kernel, Compiled, SIMD"))
    except StopIteration:
        return {}
    hdr = [h.strip() for h in lines[i].split(",")]
    out = {}
    for l in lines[i + 1:]:
        if not l.strip():
            break
        row = next(csv.reader([l.strip()], skipinitialspace=True))
        if row and _is_cand_kernel(row[0]):
            out[_short(row[0])] = {h: v.strip() for h, v in zip(hdr[1:], row[1:])}
    return out


def parse_stalls(log):
    lines = log.splitlines()
    try:
        i = next(i for i, l in enumerate(lines) if l.strip().startswith("Kernel, IP[Address]"))
    except StopIteration:
        return {}
    hdr = [h.strip() for h in lines[i].split(",")]
    tot = defaultdict(lambda: defaultdict(float))
    for l in lines[i + 1:]:
        l = l.strip()
        if not l.startswith('"'):
            continue
        name, rest = l[1:].split('",', 1)
        if not _is_cand_kernel(name):
            continue
        for h, v in zip(hdr[1:], (x.strip() for x in rest.split(","))):
            if h.endswith("[Events]"):
                try:
                    tot[_short(name)][h.replace("[Events]", "").replace("Stall", "")] += float(v)
                except ValueError:
                    pass
    out = {}
    for k, d in tot.items():
        s = sum(d.values()) or 1.0
        out[k] = {n: round(100 * v / s, 1) for n, v in sorted(d.items(), key=lambda x: -x[1]) if v > 0}
    return out


# Diagnosis playbook (PVC, from the SYCL-TLA grouped GEMM measurements in the
# notes and Xe-Forge's VTune rules). Each: (id, test(metrics, stalls, props),
# finding, first-line fixes).
DIAGNOSES = [
    ("spill", lambda m, s, p: _num(p.get("Spill Memory Per Thread")) > 0,
     "register spills (Spill Memory Per Thread > 0)",
     "smaller per-sub-group accumulator (more sub-groups or smaller tile), keep grf_size<256>, fewer live copy "
     "fragments; check the build log's 'spilled around' line"),
    ("load_latency", lambda m, s, p: s.get("Sbid", 0) >= 20,
     "scoreboard (SBID) stalls: sub-groups wait on their own 2D block loads",
     "longer prefetch distance or a larger K tile so more loads are in flight; for short K (grad_weight with "
     "~190 rows/expert) the opposite: prefetch 1-2 so the prologue does not dominate; split-K / more tiles in flight"),
    ("tail_or_occupancy", lambda m, s, p: m["idle"] >= 25,
     "XVEs idle a large share of the time: partial last wave, too few work-groups or tiles, or launch tail",
     "smaller M tile for small groups (128-row tiles), more persistent work-groups, a different linearization "
     "(group-major vs n-major), stream-K / split-K so tiles even out"),
    ("low_occupancy", lambda m, s, p: m["occupancy"] < 35,
     "low thread occupancy",
     "work-group size vs 8 threads/XVE in GRF-256 mode: 512-thread work-groups give one per Xe core; check SLM "
     "per work-group and the persistent grid size"),
    ("xmx_underfed", lambda m, s, p: m["xmx_util"] < 50 and m["xve_active"] >= 60,
     "XVEs active but XMX (DPAS) utilization low: instruction overhead (address math, reorders, scheduler) "
     "competes with DPAS",
     "hoist index math out of the k-loop, avoid per-tile reorder() copies (load directly in the MMA layout), "
     "larger sub-group tiles (more DPAS per load)"),
    ("sync", lambda m, s, p: s.get("Sync", 0) + s.get("Dist", 0) >= 15,
     "barrier / dependency (Sync + Dist) stalls",
     "fewer or split barriers (barrier_arrive / barrier_wait spacing), sub-group-scope instead of work-group "
     "barriers where SLM is not shared"),
    ("pipe", lambda m, s, p: s.get("Pipe", 0) >= 20,
     "pipe stalls (XMX / ALU pipeline busy, usually fine for a GEMM)",
     "interleave independent DPAS chains; nothing to fix if XMX utilization is already high"),
    ("ifetch", lambda m, s, p: s.get("InstrFetch", 0) >= 5,
     "instruction fetch stalls: code too large",
     "less unrolling, fewer template-instantiated variants in one kernel"),
]


def _num(x):
    try:
        return float(str(x).split()[0])
    except (ValueError, IndexError, TypeError):
        return 0.0


def digest(res):
    L = []
    t = res.get("timing") or {}
    if t:
        L.append(f"PROFILE {t.get('case')} {t.get('op')}: {t.get('ms', 0):.4f} ms/call, {t.get('tflops', 0):.1f} TFLOPS "
                 f"(candidate only, rows/expert {t.get('counts')})")
    for k, m in (res.get("metrics") or {}).items():
        L.append(f"kernel {k}  grid {m['grid']}  ({m['n']} instances, medians)")
        L.append(f"  GPU {m['gpu_us']:.1f} us | XVE active {m['xve_active']:.0f}% stall {m['xve_stall']:.0f}% "
                 f"idle {m['idle']:.0f}% | occupancy {m['occupancy']:.0f}% | XMX util {m['xmx_util']:.0f}% | "
                 f"send util {m['send_util']:.1f}%")
        L.append(f"  HBM rd {m['hbm_rd_gbs']:.0f} wr {m['hbm_wr_gbs']:.0f} GB/s | L3 rd {m['l3_rd_gbs']:.0f} GB/s | "
                 f"{m['mhz']:.0f} MHz | barrier msgs {m['barrier_msgs']:.0f}")
        s = (res.get("stalls") or {}).get(k, {})
        if s:
            L.append("  stall samples: " + ", ".join(f"{n} {v:.0f}%" for n, v in list(s.items())[:8]))
        p = (res.get("props") or {}).get(k, {})
        if p:
            L.append("  properties: " + ", ".join(f"{n}={v}" for n, v in p.items()))
        hits = [(i, f, fix) for i, test, f, fix in DIAGNOSES if _safe(test, m, s, p)]
        if hits:
            L.append("  DIAGNOSES (rule hits, most specific first):")
            for i, f, fix in hits:
                L.append(f"   - [{i}] {f}\n       try: {fix}")
        else:
            L.append("  DIAGNOSES: no rule hit (balanced kernel; look at tile fill and the per-op TFLOPS)")
    if not res.get("metrics"):
        L.append("PROFILE FAILED: no unitrace metrics for the candidate kernels")
        if res.get("log_tail"):
            L.append(res["log_tail"][-1500:])
    L.append("Reference points (production 256x256, TP4/EP4, unitrace): dX XMX 70% stall 7%; fwd XMX 65% "
             "SBID 14%; grad_weight XMX 36% SBID 31%, 33% stalled. HBM is never the limit (<20% of GTI).")
    return "\n".join(L)


def _safe(test, m, s, p):
    try:
        return bool(test(m, s, p))
    except Exception:  # noqa: BLE001
        return False


# ---------------------------------------------------------- login side ---
def profile(cand_dir, case="tp4ep4_uniform", op="w13_dW", label="", iters=20):
    from . import candidate, evaluate, slots
    t0 = time.time()
    hdr, cfgs = candidate.load(cand_dir)
    job = evaluate._job_dir("prof-" + (label or Path(cand_dir).name))
    (job / "cand").mkdir()
    (job / "cand/grouped_mm_kernel_impl.h").write_text(hdr)
    (job / "cand/configs.json").write_text(json.dumps(cfgs, indent=2))
    (job / "job.json").write_text(json.dumps({"case": case, "op": op, "iters": iters, "label": label}))
    build_node = None
    if evaluate._all_cached(job):
        from . import node_eval
        node_eval.phase_build(job)
    else:
        with slots.acquire("build", label=f"build {job.name}") as (build_node, _):
            evaluate.ssh_phase(build_node, job, "build", timeout=config.BUILD_TIMEOUT_S + 600)
    b = json.loads((job / "build.json").read_text()) if (job / "build.json").exists() else {"ok": False}
    if not b.get("ok"):
        return {"ok": False, "job": str(job), "text": "PROFILE: candidate does not build; run `ggemm-eval run` first"}
    with slots.acquire("gpu", label=f"profile {job.name}", prefer_node=build_node) as (node, tile):
        evaluate.ssh_phase(node, job, "profile", extra=str(tile), timeout=3 * config.RUN_TIMEOUT_S)
    try:
        res = json.loads((job / "profile_result.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        res = {"ok": False, "metrics": {}, "log_tail": "profile phase produced no result"}
    res["job"] = str(job)
    res["text"] = digest(res) + f"\nJOB: {job}"
    res["seconds"] = round(time.time() - t0, 1)
    d = Path(cand_dir) / "profiles"
    d.mkdir(exist_ok=True)
    (d / f"{time.strftime('%Y%m%d-%H%M%S')}_{case}_{op}.txt").write_text(res["text"] + "\n")
    return res
