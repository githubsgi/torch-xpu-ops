"""Runs on a compute node (oneAPI loaded): build candidate .so files, run agent_eval.

    python -m ggemm_agent.node_eval build   <job_dir>
    python -m ggemm_agent.node_eval run     <job_dir> <tile>
    python -m ggemm_agent.node_eval profile <job_dir> <tile>

<job_dir>/job.json holds the request (see evaluate.py); cand/ the candidate
snapshot. Builds are cached by content hash under RUNS/cache/so/<hash>/ and
shared by every job and harness.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import candidate, config

SO_CACHE = config.RUNS / "cache/so"
DRIVER_CACHE = config.RUNS / "cache/driver"


def icpx_common():
    tla = config.SYCL_TLA
    return (["icpx", "-fsycl", "-std=c++20", "-O3", "-fPIC", "-DCUTLASS_ENABLE_SYCL", "-DSYCL_INTEL_TARGET",
             f"-I{tla}/include", f"-I{tla}/tools/util/include", f"-I{tla}/applications"],
            ["-fsycl-targets=spir64_gen", "-Xs", f"-device {config.AOT_DEVICE}", "-Xspirv-translator",
             "-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,"
             "+SPV_INTEL_subgroup_matrix_multiply_accumulate"])


class Lock:
    """mkdir lock on the shared file system; stale after `stale_s`."""

    def __init__(self, path, stale_s=config.BUILD_TIMEOUT_S + 300):
        self.path, self.stale_s = Path(str(path) + ".lock"), stale_s

    def __enter__(self):
        while True:
            try:
                self.path.mkdir(parents=True)
                (self.path / "owner").write_text(f"{os.uname().nodename} {os.getpid()} {time.time()}\n")
                return self
            except FileExistsError:
                try:
                    if time.time() - self.path.stat().st_mtime > self.stale_s:
                        shutil.rmtree(self.path, ignore_errors=True)
                        continue
                except FileNotFoundError:
                    continue
                time.sleep(3)

    def __exit__(self, *a):
        shutil.rmtree(self.path, ignore_errors=True)


def error_digest(log, limit=60):
    """The compiler errors an agent needs, not the whole log."""
    lines = log.splitlines()
    out = []
    for i, l in enumerate(lines):
        if re.search(r"\berror\b", l) and "warning" not in l.lower()[:40]:
            out.extend(lines[i:i + 4])
            if len(out) >= limit:
                break
    if not out:
        out = lines[-limit:]
    return "\n".join(out[:limit])


def spills(log):
    return sorted(set(re.findall(r"spilled around \d+|spill size \d+|private memory: \d+ bytes", log)))


def build_unit(header_text, unit):
    """Build one candidate .so (cached). Returns a status dict."""
    d = SO_CACHE / unit["hash"]
    so = d / "lib.so"
    meta = d / "meta.json"
    if meta.exists():
        return json.loads(meta.read_text())
    with Lock(d):
        if meta.exists():
            return json.loads(meta.read_text())
        srcd = d / "src"
        srcd.mkdir(parents=True, exist_ok=True)
        (srcd / "grouped_mm_kernel_impl.h").write_text(header_text)
        (srcd / f"cfg_{unit['hash']}.h").write_text(unit["cfg_header"])
        common, device = icpx_common()
        cmd = (common + [f"-I{srcd}", f"-I{config.SRC_DIR}", f"-I{config.REPO_DIR}/src", f"-I{config.TUNER_DIR}"]
               + device + list(unit["flags"])
               + ["-shared", f"-DCAND_CFG_NAME={unit['name']}", f'-DCAND_CFG_HEADER="cfg_{unit["hash"]}.h"']
               + [f"-D{candidate.KIND_MACRO[k]}=1" for k in unit["kinds"]]
               + [str(config.EVAL_DIR / "cand_cfg.cpp"), "-o", str(so)])
        t0 = time.time()
        try:
            p = subprocess.run(cmd, capture_output=True, text=True, timeout=config.BUILD_TIMEOUT_S)
            log, rc = p.stdout + p.stderr, p.returncode
        except subprocess.TimeoutExpired as e:
            log, rc = f"build timed out after {config.BUILD_TIMEOUT_S} s\n{e.stdout or ''}{e.stderr or ''}", -1
        (d / "build.log").write_text(" ".join(cmd) + "\n\n" + log)
        ok = rc == 0 and so.exists()
        res = {"hash": unit["hash"], "kinds": unit["kinds"], "ok": ok, "so": str(so) if ok else None,
               "seconds": round(time.time() - t0, 1), "spills": spills(log),
               "errors": "" if ok else error_digest(log).replace(f"{srcd}/", "candidate/"),
               "log": str(d / "build.log"),
               "node": os.uname().nodename}
        # Failed builds are cached too (same input, same failure) except timeouts.
        if rc != -1:
            meta.write_text(json.dumps(res, indent=1))
        return res


DRIVER_SRCS = [config.EVAL_DIR / "agent_eval.cpp", config.SRC_DIR / "grouped_mm_util.cpp",
               config.SRC_DIR / "grouped_mm_kernels.h", config.TUNER_DIR / "tune_registry.h"]


def driver_exe():
    h = hashlib.sha256(b"".join(p.read_bytes() for p in DRIVER_SRCS) + config.AOT_DEVICE.encode()).hexdigest()[:16]
    return DRIVER_CACHE / h / "agent_eval"


def build_driver():
    srcs = DRIVER_SRCS
    exe = driver_exe()
    d = exe.parent
    if exe.exists():
        return str(exe)
    with Lock(d):
        if exe.exists():
            return str(exe)
        d.mkdir(parents=True, exist_ok=True)
        common, device = icpx_common()
        cmd = (common + [f"-I{config.SRC_DIR}", f"-I{config.REPO_DIR}/src", f"-I{config.TUNER_DIR}"] + device
               + [str(srcs[0]), str(srcs[1]), "-ldl", "-rdynamic", "-o", str(exe) + ".tmp"])
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=config.BUILD_TIMEOUT_S)
        (d / "build.log").write_text(" ".join(cmd) + "\n\n" + p.stdout + p.stderr)
        if p.returncode != 0:
            raise RuntimeError(f"agent_eval build failed, see {d / 'build.log'}:\n{error_digest(p.stdout + p.stderr)}")
        os.replace(str(exe) + ".tmp", exe)
        return str(exe)


def arm_units(header_text, cfgs):
    units, kind_map = candidate.build_units(header_text, cfgs)
    return units, kind_map


def phase_build(job_dir):
    job_dir = Path(job_dir)
    cand_hdr, cand_cfgs = candidate.load(job_dir / "cand")
    base_hdr, base_cfgs = config.PROD_HEADER.read_text(), candidate.normalize(candidate.production_configs())
    cu, cmap = arm_units(cand_hdr, cand_cfgs)
    bu, bmap = arm_units(base_hdr, base_cfgs)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        fut_drv = ex.submit(build_driver)
        fc = [ex.submit(build_unit, cand_hdr, u) for u in cu]
        fb = [ex.submit(build_unit, base_hdr, u) for u in bu]
        cres = [f.result() for f in fc]
        bres = [f.result() for f in fb]
        try:
            driver, drv_err = fut_drv.result(), ""
        except Exception as e:  # noqa: BLE001
            driver, drv_err = None, str(e)
    ok = all(r["ok"] for r in cres + bres) and driver is not None

    def manifest(results, kmap):
        by = {r["hash"]: r for r in results}
        return "".join(f"{k} {by[h]['so']} {w}\n" for k, (h, w) in kmap.items()) if all(
            by[h]["ok"] for h, _ in kmap.values()) else ""

    out = {"ok": ok, "seconds": round(time.time() - t0, 1), "cand_units": cres, "base_units": bres,
           "driver": driver, "driver_error": drv_err,
           "cand_manifest": manifest(cres, cmap), "base_manifest": manifest(bres, bmap)}
    (job_dir / "build.json").write_text(json.dumps(out, indent=1))
    return out


def run_driver(job_dir, tile, args, timeout):
    env = dict(os.environ, ZE_FLAT_DEVICE_HIERARCHY="FLAT", ZE_AFFINITY_MASK=str(tile))
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env, cwd=job_dir)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        partial = e.stdout or ""
        if isinstance(partial, bytes):
            partial = partial.decode(errors="replace")
        return -9, partial, f"timed out after {timeout} s"


def phase_run(job_dir, tile):
    job_dir = Path(job_dir)
    job = json.loads((job_dir / "job.json").read_text())
    b = json.loads((job_dir / "build.json").read_text())
    (job_dir / "cand.manifest").write_text(b["cand_manifest"])
    (job_dir / "base.manifest").write_text(b["base_manifest"])
    args = [b["driver"], "--base", "base.manifest", "--cand", "cand.manifest", "--mode", job.get("mode", "both"),
            "--suite", job.get("suite", "quick"), "--verify-seed", str(job.get("verify_seed", 0)),
            "--rounds", str(job.get("rounds", 5)), "--json", "driver.json"]
    if job.get("ops"):
        args += ["--ops", job["ops"]]
    t0 = time.time()
    rc, out, err = run_driver(job_dir, tile, args, job.get("timeout", config.RUN_TIMEOUT_S))
    (job_dir / "driver.log").write_text(out + "\n--- stderr ---\n" + err)
    res = {"rc": rc, "seconds": round(time.time() - t0, 1), "node": os.uname().nodename, "tile": tile,
           "stdout_tail": out[-4000:], "stderr_tail": err[-3000:]}
    try:
        res["driver"] = json.loads((job_dir / "driver.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        res["driver"] = None
    (job_dir / "run.json").write_text(json.dumps(res, indent=1))
    return res


def main(argv):
    phase, job_dir = argv[0], argv[1]
    if phase == "build":
        r = phase_build(job_dir)
        print(json.dumps({"ok": r["ok"], "seconds": r["seconds"]}))
    elif phase == "run":
        r = phase_run(job_dir, int(argv[2]))
        print(json.dumps({"rc": r["rc"], "seconds": r["seconds"]}))
    elif phase == "profile":
        from . import profile
        r = profile.phase_profile(job_dir, int(argv[2]))
        print(json.dumps({"ok": r.get("ok"), "seconds": r.get("seconds")}))
    else:
        raise SystemExit(f"unknown phase {phase}")


if __name__ == "__main__":
    main(sys.argv[1:])
