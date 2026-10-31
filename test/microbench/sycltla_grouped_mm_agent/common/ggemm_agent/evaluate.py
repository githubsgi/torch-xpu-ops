"""Evaluate a candidate from the login node (or anywhere with ssh to the nodes).

    res = evaluate("path/to/cand", suite="quick")
    res["status"]   ok | invalid | build_error | runtime_error | verify_failed | bench_mismatch
    res["score"]    geomean over bench cases of (baseline / candidate) time of the
                    case's six GEMMs (> 1 is faster than production); 0 unless ok
    res["per_op"]   geomean speedup per op over the cases
    res["text"]     summary for an LLM (also printed by `ggemm-eval run`)

Flow: snapshot the candidate into RUNS/jobs/<id>/, build its .so files (and
the baseline's) on any node holding a build slot, then run agent_eval on a
free GPU tile. Builds are content-cached, so re-evaluating or sharing configs
between candidates is cheap. Thread-safe: run many evaluate() calls at once.
"""

import json
import math
import os
import random
import shutil
import subprocess
import time
from pathlib import Path

from . import candidate, config, slots

JOBS = config.RUNS / "jobs"
LEDGER = config.RUNS / "ledger.jsonl"
SUITES = ("quick", "small", "large", "full")


def _job_dir(label):
    JOBS.mkdir(parents=True, exist_ok=True)
    tag = "".join(c if c.isalnum() or c in "-_" else "_" for c in label)[:40]
    d = JOBS / f"{time.strftime('%Y%m%d-%H%M%S')}-{random.randrange(16 ** 6):06x}{'-' + tag if tag else ''}"
    d.mkdir()
    return d


def _node_script(job_dir, phase, extra=""):
    use, mod = config.ONEAPI_MODULES.split(":", 1)
    env = "".join(f"export {k}={json.dumps(v)}\n" for k, v in os.environ.items() if k.startswith("GGEMM_"))
    s = Path(job_dir) / f"{phase}.sh"
    s.write_text(
        "#!/bin/bash\n"
        f"module use {use} >/dev/null 2>&1\nmodule load {mod} >/dev/null 2>&1\n"
        f"{env}export PYTHONPATH={config.COMMON_DIR}${{PYTHONPATH:+:$PYTHONPATH}}\n"
        f"exec {config.NODE_PYTHON} -m ggemm_agent.node_eval {phase} {job_dir} {extra}\n")
    return s


def ssh_phase(node, job_dir, phase, extra="", timeout=None):
    s = _node_script(job_dir, phase, extra)
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", node, f"bash -l {s}"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return -9, "", f"ssh {phase} timed out after {timeout} s"


def _all_cached(job_dir):
    from . import node_eval
    ch, cc = candidate.load(Path(job_dir) / "cand")
    units = candidate.build_units(ch, cc)[0] + candidate.build_units(
        config.PROD_HEADER.read_text(), candidate.normalize(candidate.production_configs()))[0]
    return node_eval.driver_exe().exists() and all(
        (node_eval.SO_CACHE / u["hash"] / "meta.json").exists() for u in units)


def _geomean(xs):
    xs = [x for x in xs if x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def evaluate(cand_dir, suite="quick", mode="both", ops="", rounds=11, verify_seed=None, label="",
             timeout=None):
    t0 = time.time()
    cand_dir = Path(cand_dir)
    res = {"cand": str(cand_dir), "suite": suite, "label": label, "status": None, "score": 0.0}
    try:
        hdr, cfgs = candidate.load(cand_dir)
    except (candidate.CandidateError, ValueError, KeyError, json.JSONDecodeError) as e:
        res.update(status="invalid", error=str(e))
        return _finish(res, t0)
    job = _job_dir(label)
    (job / "cand").mkdir()
    (job / "cand/grouped_mm_kernel_impl.h").write_text(hdr)
    (job / "cand/configs.json").write_text(json.dumps(cfgs, indent=2))
    seed = random.randrange(1, 2 ** 31) if verify_seed is None else int(verify_seed)
    (job / "job.json").write_text(json.dumps({"mode": mode, "suite": suite, "ops": ops, "rounds": rounds,
                                              "verify_seed": seed, "label": label,
                                              "timeout": timeout or config.RUN_TIMEOUT_S}))
    res["job"] = str(job)
    res["configs"] = cfgs

    # build (on the login node if everything is cached: no compiler needed)
    build_node = None
    if _all_cached(job):
        from . import node_eval
        node_eval.phase_build(job)
    else:
        with slots.acquire("build", label=f"build {job.name}") as (node, _):
            build_node = node
            rc, out, err = ssh_phase(node, job, "build", timeout=config.BUILD_TIMEOUT_S + 600)
        if not (job / "build.json").exists():
            res.update(status="runtime_error", error=f"build phase failed on {node}: rc={rc}\n{err[-2000:]}")
            return _finish(res, t0)
    b = json.loads((job / "build.json").read_text())
    res["build"] = {"seconds": b["seconds"],
                    "spills": sorted({s for u in b["cand_units"] for s in u["spills"]}),
                    "units": [{k: u[k] for k in ("hash", "kinds", "ok", "seconds")} for u in b["cand_units"]]}
    if not b["ok"]:
        errs = [f"[{','.join(u['kinds'])}] {u['errors']}" for u in b["cand_units"] if not u["ok"]]
        if b.get("driver_error"):
            errs.append(b["driver_error"])
        if any(not u["ok"] for u in b["base_units"]):
            errs.append("BASELINE build failed: environment problem, not the candidate")
        res.update(status="build_error", error="\n".join(errs))
        return _finish(res, t0)

    # run on a GPU tile
    with slots.acquire("gpu", label=f"run {job.name}", prefer_node=build_node) as (node, tile):
        rc, out, err = ssh_phase(node, job, "run", extra=str(tile),
                                 timeout=(timeout or config.RUN_TIMEOUT_S) + 300)
    try:
        run = json.loads((job / "run.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        res.update(status="runtime_error", error=f"run phase failed on {node}.{tile}: rc={rc}\n{err[-2000:]}")
        return _finish(res, t0)
    res["node"], res["tile"] = config.short(run["node"]), run["tile"]
    d = run.get("driver")
    if d is None:
        res.update(status="runtime_error",
                   error=f"agent_eval rc={run['rc']} (crash, hang or GPU fault)\n"
                         f"{run['stdout_tail'][-1500:]}\n{run['stderr_tail'][-1500:]}")
        return _finish(res, t0)
    res["device"] = d.get("device")
    res["cand_desc"] = d.get("cand")
    v = d.get("verify", [])
    res["verify"] = {"ok": d.get("verify_ok", True), "cases": len(v),
                     "failed": [x for x in v if not x["ok"]][:8]}
    bench = d.get("bench", [])
    res["bench"] = bench
    if mode in ("both", "verify") and not d.get("verify_ok", True):
        res["status"] = "verify_failed"
        return _finish(res, t0)
    if bench and not d.get("bench_ok", True):
        res["status"] = "bench_mismatch"
        res["mismatch"] = [x for x in bench if not x["ok"]][:8]
        return _finish(res, t0)
    if run["rc"] not in (0, 1):
        res.update(status="runtime_error", error=run["stderr_tail"][-1500:])
        return _finish(res, t0)
    res["status"] = "ok"
    cases = {}
    for x in bench:
        c = cases.setdefault(x["case"], {"base_ms": 0.0, "cand_ms": 0.0})
        c["base_ms"] += x["base_ms"]
        c["cand_ms"] += x["cand_ms"]
    for c in cases.values():
        c["speedup"] = c["base_ms"] / c["cand_ms"] if c["cand_ms"] > 0 else 0.0
    res["per_case"] = cases
    per_op = {}
    for x in bench:
        per_op.setdefault(x["op"], []).append(x["speedup"])
    res["per_op"] = {k: _geomean(v) for k, v in per_op.items()}
    # Probe cases (equal rows on every expert: rows8, rows64, rows1024) are
    # reported but not scored: the score is what routed MoE calls see.
    for name, c in cases.items():
        c["probe"] = is_probe(name)
    scored = [c["speedup"] for c in cases.values() if not c["probe"]] or [c["speedup"] for c in cases.values()]
    res["score"] = _geomean(scored) if cases else 0.0
    return _finish(res, t0)


def is_probe(case):
    return "_rows" in case


def _finish(res, t0):
    res["seconds"] = round(time.time() - t0, 1)
    res["text"] = summarize(res)
    if res.get("job"):
        Path(res["job"], "result.json").write_text(json.dumps(res, indent=1))
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({k: res.get(k) for k in ("label", "cand", "job", "suite", "status", "score",
                                                    "per_op", "seconds")}) + "\n")
    return res


def summarize(res):
    """Plain-text result in the shape of Xe-Forge's external-benchmark contract."""
    L = [f"CANDIDATE: {res['cand']}", f"STATUS: {res['status']}"]
    if res.get("job"):
        L.append(f"JOB: {res['job']}")
    if res["status"] == "invalid":
        L.append(f"ERROR: {res.get('error')}")
    if res.get("build"):
        b = res["build"]
        L.append(f"BUILD: {'ok' if res['status'] != 'build_error' else 'FAILED'} in {b['seconds']} s; "
                 f"register spills: {', '.join(b['spills']) or 'none'}")
    if res["status"] in ("build_error", "runtime_error"):
        L.append("ERRORS:\n" + str(res.get("error", ""))[:6000])
    v = res.get("verify")
    if v:
        L.append(f"CORRECTNESS: {'PASS' if v['ok'] else 'FAIL'} ({v['cases']} hidden verification cases, full "
                 f"output vs fp32 reference, NaN sentinels around the output)")
        for f in v["failed"]:
            L.append(f"  FAILED {f['case']} kind={f['kind']} K_or_M={f['k_or_m']} N={f['n']} counts={f['counts']}"
                     f" cmp={f.get('cmp')} {f.get('error') or ''}")
    for m in res.get("mismatch", []):
        L.append(f"  TIMED OUTPUT MISMATCH vs baseline: {m['case']} {m['op']} cmp={m.get('cmp')} {m.get('error')}")
    if res["status"] == "ok":
        cd = res.get("cand_desc") or {}
        L.append("CONFIG: " + "; ".join(f"{k}={v.get('desc', '').split(' ', 1)[-1]} wgs={v.get('wgs')}"
                                       for k, v in cd.items()))
        L.append(f"SCORE: {res['score']:.4f}x  (geomean over the routed cases of production/candidate time of the "
                 f"six GEMMs, suite {res['suite']}, on {res.get('node')}.{res.get('tile')}; rowsN probe cases are "
                 f"not scored)")
        for name, c in res["per_case"].items():
            L.append(f"{'probe' if c.get('probe') else 'case'} {name}: six GEMMs base {c['base_ms']:.4f} ms, "
                     f"cand {c['cand_ms']:.4f} ms -> {c['speedup']:.3f}x")
            for x in res["bench"]:
                if x["case"] == name:
                    L.append(f"   {x['op']:8s} base {x['base_ms']:.4f} ms ({x['base_tflops']:.0f} TF)  "
                             f"cand {x['cand_ms']:.4f} ms ({x['cand_tflops']:.0f} TF)  {x['speedup']:.3f}x  "
                             f"(noise: IQR of paired ratios {x.get('speedup_iqr', 0):.3f})")
        L.append("PER-OP: " + ", ".join(f"{k} {v:.3f}x" for k, v in res["per_op"].items()))
        s = res["score"]
        L.append("VERDICT: " + ("IMPROVED" if s >= 1.01 else "REGRESSED" if s <= 0.99 else "NEUTRAL (within 1%)"))
    L.append("DONE")
    return "\n".join(L)


def save_to_cand(res):
    """Keep the result next to the candidate (agents read it from there)."""
    d = Path(res["cand"]) / "evals"
    d.mkdir(exist_ok=True)
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{res['suite']}_{res['status']}"
    (d / f"{name}.json").write_text(json.dumps({k: v for k, v in res.items() if k != "text"}, indent=1))
    (Path(res["cand"]) / "last_eval.txt").write_text(res["text"] + "\n")
    shutil.copy(d / f"{name}.json", Path(res["cand"]) / "last_eval.json")
