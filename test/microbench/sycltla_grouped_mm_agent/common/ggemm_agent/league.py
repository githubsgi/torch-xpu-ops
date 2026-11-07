"""League: run the three harnesses in periods, all starting from the global best.

    python -m ggemm_agent.league --root RUNS/league --start <cand dir> --start-name v3_t17 \
        --harness-budget-usd 550 --caps xe_forge=15,kda=20,kernel_forge=25 \
        --stop-at 2026-11-09T00:00Z [--push]

Each period:
  1. start Xe-Forge, KDA and Kernel Forge (each from its own worktree, on its
     own node) from the current global best, each with its period $ cap;
  2. wait until all three exit, or stop them after --period-max-hours;
  3. re-verify the incumbent and every harness's best on suite `all`
     (--verify-runs runs, new hidden seed each), all on the same nodes;
  4. a challenger replaces the incumbent only if every one of its runs beats
     the incumbent's mean by --improve (default 1%), and torch's four kernel
     files (m/k x bf16/fp16) compile against its header;
  5. archive the period (each harness's best, the global best, verification
     logs, summary) under <root>/archive/pNN/, optionally commit the global
     best to the ggemm-agent/league branch and push.
Stops at --periods, the harness budget (sum of the harnesses' own spend
records), --stop-at, or a file <root>/STOP. Restarting with the same --root
resumes: harness workdirs resume themselves, finished periods are skipped.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import candidate, config, evaluate

HARNESSES = ("xe_forge", "kda", "kernel_forge")
WORKTREE = {"xe_forge": "ggemm-agent-xe-forge", "kda": "ggemm-agent-kda", "kernel_forge": "ggemm-agent-kernel-forge"}
PY = os.environ.get("GGEMM_LOGIN_PYTHON", str(config.HOME / "miniforge3/bin/python3"))


def log(msg):
    print(f"[league {time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def cand_hash(d):
    d = Path(d)
    return hashlib.sha256((d / "grouped_mm_kernel_impl.h").read_bytes()
                          + (d / "configs.json").read_bytes()).hexdigest()[:16]


def copy_cand(src, dst):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    for f in ("grouped_mm_kernel_impl.h", "configs.json", "last_eval.txt"):
        if (Path(src) / f).exists():
            shutil.copy(Path(src) / f, dst / f)
    return dst


# ------------------------------------------------------------ harnesses ---
def harness_cmd(h, a, wd, start, score):
    common = ["--model", a.model]
    if h == "xe_forge":
        return [PY, "-u", "-m", "xe_forge.run", "--workdir", str(wd), "--start-from", str(start),
                "--max-llm-usd", str(a.cap[h]), "--suite", "search", "--final-suite", "all"] + common
    if h == "kda":
        guidance = (f"This run starts from the current best kernel, not production: cand/ and best/ hold it, "
                    f"and it scores {score:.4f}x vs production on suite `all` (DSv3 16B TP4/EP4, TP8/EP8, EP8 "
                    f"and DSv3 671B TP8/EP32). Promotion is checked on suite `all` (not `full`) and needs a "
                    f"SCORE above {score:.4f} with no op below 0.97x. Use `--suite search` (one uniform case per "
                    f"regime) for quick checks and `--suite all` before claiming a win. Optimize for both models.")
        return [PY, "-u", "-m", "kda.rlcr", "--workdir", str(wd), "--phase", "2", "--start-from", str(start),
                "--start-score", f"{score:.4f}", "--max-rounds", "42", "--max-llm-usd", str(a.cap[h]),
                "--eval-suite", "search", "--promote-suite", "all", "--guidance", guidance] + common
    return [PY, "-u", "-m", "kernel_forge.mcts", "--workdir", str(wd), "--start-from", str(start),
            "--iterations", str(a.kf_iterations), "--workers", str(a.kf_workers), "--knob-ratio", "0.3",
            "--max-llm-usd", str(a.cap[h]), "--suite", "search", "--final-suite", "all",
            "--max-wall-min", str(int(a.period_max_hours * 60 - 30))] + common


def spend(h, wd):
    wd = Path(wd)
    try:
        if h == "xe_forge":
            return sum(e.get("usd", 0.0) for e in json.loads((wd / "cost.json").read_text()))
        if h == "kda":
            return sum(json.loads(l).get("cost_usd", 0.0)
                       for l in (wd / ".humanize/costs.jsonl").read_text().splitlines() if l.strip())
        return float(json.loads((wd / "tree.json").read_text()).get("llm_cost", 0.0))
    except (FileNotFoundError, json.JSONDecodeError, ValueError, TypeError):
        return 0.0


def run_harnesses(a, pdir, start, score, deadline):
    procs = {}
    for h in HARNESSES:
        if a.cap[h] <= 0:
            continue
        wd = pdir / h
        env = dict(os.environ, GGEMM_LLM_MODEL=a.model, GGEMM_NODES=",".join(a.nodes[h]))
        env.pop("GGEMM_TILES", None)
        cwd = a.worktrees / WORKTREE[h] / "test/microbench/sycltla_grouped_mm_agent"
        cmd = harness_cmd(h, a, wd, start, score)
        out = open(pdir / f"{h}.log", "a")
        procs[h] = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=subprocess.STDOUT,
                                    stdin=subprocess.DEVNULL, start_new_session=True)
        log(f"  started {h} pid {procs[h].pid} on {','.join(config.short(n) for n in a.nodes[h])} "
            f"(cap ${a.cap[h]})")
    t_end = min(time.time() + a.period_max_hours * 3600, deadline)
    while any(p.poll() is None for p in procs.values()):
        if time.time() > t_end or (a.root / "STOP").exists():
            log("  period time limit / STOP: stopping the harnesses")
            for p in procs.values():
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
            for _ in range(24):
                if all(p.poll() is not None for p in procs.values()):
                    break
                time.sleep(5)
            for p in procs.values():
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGKILL)
            break
        time.sleep(60)
    return {h: p.poll() for h, p in procs.items()}


# --------------------------------------------------------- verification ---
def verify(a, pdir, cands):
    """cands: {name: dir}. Returns {name: {"runs": [score...], "ok": bool, "mean", "min", "texts"}}."""
    vdir = pdir / "verify"
    vdir.mkdir(exist_ok=True)
    jobs = [(name, d, i) for name, d in cands.items() for i in range(a.verify_runs)]

    def one(job):
        name, d, i = job
        work = copy_cand(d, vdir / f"{name}_v{i}")
        res = evaluate.evaluate(work, suite="all", label=f"league-{pdir.name}-{name}-v{i}")
        (vdir / f"{name}_v{i}.txt").write_text(res["text"] + "\n")
        return name, res

    out = {n: {"runs": [], "ok": True, "per_op": []} for n in cands}
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        for name, res in ex.map(one, jobs):
            out[name]["runs"].append(res["score"] if res["status"] == "ok" else 0.0)
            out[name]["per_op"].append(res.get("per_op", {}))
            out[name]["ok"] &= res["status"] == "ok"
    for v in out.values():
        v["mean"] = sum(v["runs"]) / len(v["runs"])
        v["min"] = min(v["runs"])
    return out


def torch_compile_check(a, pdir, cand_dir):
    """Compile torch-xpu-ops' four kernel files (m/k x bf16/fp16) against the candidate header."""
    ov = pdir / "torch_compile" / "overlay" / "ATen/native/xpu/grouped_mm/sycltla"
    ov.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(cand_dir) / "grouped_mm_kernel_impl.h", ov / "grouped_mm_kernel_impl.h")
    shutil.copy(config.SRC_DIR / "grouped_mm_kernels.h", ov / "grouped_mm_kernels.h")
    use, mod = config.ONEAPI_MODULES.split(":", 1)
    tla, outd = config.SYCL_TLA, pdir / "torch_compile"
    script = outd / "compile.sh"
    script.write_text(
        f"module use {use} >/dev/null 2>&1; module load {mod} >/dev/null 2>&1\ncd {outd}\nrc=0\n"
        "for f in m_bf16 m_fp16 k_bf16 k_fp16; do\n"
        f"  icpx -fsycl -std=c++20 -O3 -fPIC -DCUTLASS_ENABLE_SYCL -DSYCL_INTEL_TARGET -I{tla}/include "
        f"-I{tla}/tools/util/include -I{tla}/applications -I{outd / 'overlay'} -I{config.REPO_DIR}/src "
        f"-fsycl-targets=spir64_gen -Xs \"-device {config.AOT_DEVICE}\" -Xspirv-translator "
        "\"-spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate\" "
        f"-shared {config.SRC_DIR}/grouped_mm_$f.cpp -o lib_$f.so > $f.log 2>&1 &\n"
        "done\nfor j in $(jobs -p); do wait $j || rc=1; done\n"
        "for f in m_bf16 m_fp16 k_bf16 k_fp16; do [ -f lib_$f.so ] || rc=1; echo \"$f $([ -f lib_$f.so ] && echo ok "
        "|| echo FAIL) $(grep -o 'spilled around [0-9]*' $f.log | sort -u | tr '\\n' ' ')\"; done\nexit $rc\n")
    node = a.nodes["kernel_forge"][0]
    p = subprocess.run(["ssh", "-o", "BatchMode=yes", node, f"bash -l {script}"], capture_output=True,
                       text=True, timeout=1800)
    return p.returncode == 0, p.stdout.strip()


# ---------------------------------------------------------------- archive ---
def archive(a, p, pdir, best_dirs, ver, glob_name, glob_dir, summary):
    ad = a.root / "archive" / f"p{p:02d}"
    if ad.exists():
        shutil.rmtree(ad)
    for h, d in best_dirs.items():
        if d:
            copy_cand(d, ad / f"{h}_best")
    copy_cand(glob_dir, ad / "global_best")
    shutil.copytree(pdir / "verify", ad / "verify")
    (ad / "summary.json").write_text(json.dumps(summary, indent=1))
    L = [f"# League period {p:02d}  ({summary['started']} -> {summary['ended']})", "",
         f"Incumbent at start: {summary['incumbent']} ({summary['incumbent_score_before']:.4f}x)", "",
         "| candidate | runs on `all` (vs production) | mean | spend | note |", "|---|---|---|---|---|"]
    for n, v in ver.items():
        L.append(f"| {n} | {', '.join(f'{x:.4f}' for x in v['runs'])} | {v['mean']:.4f} | "
                 f"{'$%.2f' % summary['spend'].get(n, 0) if n in summary['spend'] else '-'} | "
                 f"{summary['notes'].get(n, '')} |")
    L += ["", f"**Global best after period {p:02d}: {glob_name} ({summary['global_score']:.4f}x)**", "",
          f"torch compile check: {summary.get('torch_compile', '-')}", "",
          f"Harness spend this period ${sum(summary['spend'].values()):.2f}; league total "
          f"${summary['spend_total']:.2f} of ${a.harness_budget_usd:.0f}"]
    (ad / "summary.md").write_text("\n".join(L) + "\n")
    return ad


def git_push(a, p, ad):
    wt = a.worktrees / "ggemm-agent-league"
    repo = config.REPO_DIR
    if not wt.exists():
        subprocess.run(["git", "-C", str(repo), "worktree", "add", "-q", "-B", "ggemm-agent/league", str(wt),
                        "ggemm-agent/common"], check=True)
    dst = wt / "test/microbench/sycltla_grouped_mm_agent/league/archive" / f"p{p:02d}"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    copy_cand(ad / "global_best", dst / "global_best")
    shutil.copy(ad / "summary.md", dst / "summary.md")
    shutil.copy(ad / "summary.json", dst / "summary.json")
    g = lambda *c: subprocess.run(["git", "-C", str(wt)] + list(c), capture_output=True, text=True)  # noqa: E731
    g("add", "-A", "test/microbench/sycltla_grouped_mm_agent/league")
    msg = (f"[XPU] grouped GEMM agent league: period {p:02d} global best\n\n"
           + (ad / "summary.md").read_text()[:3000] + "\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\n")
    c = g("commit", "-q", "-m", msg)
    r = g("push", "-q", "-u", "origin", "ggemm-agent/league")
    log(f"  git: commit rc={c.returncode} push rc={r.returncode} {r.stderr.strip()[-200:]}")


# ------------------------------------------------------------------- main ---
def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(config.RUNS / "league"))
    ap.add_argument("--start", help="initial global best (candidate dir); needed for a new league")
    ap.add_argument("--start-name", default="start")
    ap.add_argument("--periods", type=int, default=20)
    ap.add_argument("--harness-budget-usd", type=float, default=550.0)
    ap.add_argument("--caps", default="xe_forge=15,kda=20,kernel_forge=25", help="$ per harness per period")
    ap.add_argument("--period-max-hours", type=float, default=8.0)
    ap.add_argument("--stop-at", default="", help="UTC time, e.g. 2026-11-09T00:00Z: no period runs past it")
    ap.add_argument("--nodes", default="", help="kda=<n>,xe_forge=<n>,kernel_forge=<n>+<n> (short names); "
                                                "default: node file order kda, xe_forge, kernel_forge x2")
    ap.add_argument("--worktrees", default=str(config.REPO_DIR.parent))
    ap.add_argument("--model", default=os.environ.get("GGEMM_LLM_MODEL", "opus"))
    ap.add_argument("--kf-workers", type=int, default=4)
    ap.add_argument("--kf-iterations", type=int, default=150)
    ap.add_argument("--verify-runs", type=int, default=2)
    ap.add_argument("--improve", type=float, default=1.01)
    ap.add_argument("--push", action="store_true", help="commit each period's global best to ggemm-agent/league")
    a = ap.parse_args(argv)
    a.root, a.worktrees = Path(a.root), Path(a.worktrees)
    a.cap = {h: 0.0 for h in HARNESSES}
    for kv in a.caps.split(","):
        k, v = kv.split("=")
        a.cap[k.strip()] = float(v)
    full = {config.short(n): n for n in config.nodes()}
    if a.nodes:
        a.nodes = {k: [full.get(x, x) for x in v.split("+")] for k, v in
                   (kv.split("=") for kv in a.nodes.split(","))}
    else:
        n = list(full.values())
        a.nodes = {"kda": n[:1], "xe_forge": n[1:2] or n[:1], "kernel_forge": n[2:4] or n[:1]}
    a.deadline = (dt.datetime.strptime(a.stop_at, "%Y-%m-%dT%H:%MZ").replace(tzinfo=dt.timezone.utc).timestamp()
                  if a.stop_at else float("inf"))
    return a


def main(argv=None):
    a = parse(argv if argv is not None else sys.argv[1:])
    a.root.mkdir(parents=True, exist_ok=True)
    sf = a.root / "state.json"
    st = json.loads(sf.read_text()) if sf.exists() else None
    if st is None:
        if not a.start:
            raise SystemExit("new league: --start <candidate dir> is required")
        g = copy_cand(a.start, a.root / "global" / "p00_start")
        st = {"global": {"name": a.start_name, "dir": str(g), "score": None}, "periods": [], "spend_total": 0.0}
        sf.write_text(json.dumps(st, indent=1))
    log(f"league {a.root}: global best {st['global']['name']}, {len(st['periods'])} periods done, "
        f"harness spend ${st['spend_total']:.2f} of ${a.harness_budget_usd:.0f}; nodes "
        + "; ".join(f"{h}={'+'.join(config.short(n) for n in v)}" for h, v in a.nodes.items()))

    while True:
        p = len(st["periods"]) + 1
        per_period = sum(a.cap.values())
        if p > a.periods:
            log("stop: period limit"); break
        if st["spend_total"] + per_period > a.harness_budget_usd:
            log(f"stop: budget (${st['spend_total']:.2f} spent, a period may cost up to ${per_period:.0f})"); break
        if time.time() + 3600 > a.deadline:
            log("stop: less than 1 h before --stop-at"); break
        if (a.root / "STOP").exists():
            log("stop: STOP file"); break

        pdir = a.root / "periods" / f"p{p:02d}"
        pdir.mkdir(parents=True, exist_ok=True)
        G, gname = Path(st["global"]["dir"]), st["global"]["name"]
        # the incumbent's score: last verified mean, or verify it now (first period)
        if st["global"]["score"] is None:
            log(f"p{p:02d}: verifying the start kernel on `all`")
            v0 = verify(a, pdir, {gname: G})[gname]
            if not v0["ok"]:
                raise SystemExit(f"start kernel fails verification: {v0}")
            st["global"]["score"] = v0["mean"]
            sf.write_text(json.dumps(st, indent=1))
        started = time.strftime("%F %T")
        log(f"p{p:02d}: starting all harnesses from {gname} ({st['global']['score']:.4f}x)")
        rcs = run_harnesses(a, pdir, G, st["global"]["score"], a.deadline)
        sp = {h: round(spend(h, pdir / h), 2) for h in HARNESSES if a.cap[h] > 0}
        log(f"p{p:02d}: harnesses done rc={rcs} spend={sp}")

        # candidates: incumbent + each harness's best (skip missing / identical to the incumbent)
        cands, notes, best_dirs = {"incumbent": G}, {"incumbent": gname}, {}
        ghash = cand_hash(G)
        for h in HARNESSES:
            b = pdir / h / "best"
            ok = (b / "grouped_mm_kernel_impl.h").exists() and (b / "configs.json").exists()
            best_dirs[h] = b if ok else None
            if not ok:
                notes[h] = "no best/ produced"
            elif cand_hash(b) == ghash:
                notes[h] = "best == incumbent (no improvement kept)"
            else:
                cands[h] = b
        log(f"p{p:02d}: verifying {list(cands)} on `all` x{a.verify_runs}")
        ver = verify(a, pdir, cands)
        inc = ver["incumbent"]["mean"]
        ranked = sorted((n for n in ver if n != "incumbent" and ver[n]["ok"]
                         and ver[n]["min"] > inc * a.improve), key=lambda n: -ver[n]["mean"])
        new_name, new_dir, tc = gname, G, "-"
        for n in ranked:
            ok, out = torch_compile_check(a, pdir, cands[n])
            tc = f"{n}: {'ok' if ok else 'FAILED'} ({out.replace(chr(10), '; ')})"
            log(f"p{p:02d}: torch compile check {tc}")
            if ok:
                new_name, new_dir = f"p{p:02d}_{n}", cands[n]
                break
            notes[n] = "rejected: torch kernel files do not compile"
        for n in ver:
            if n != "incumbent" and n not in notes:
                notes[n] = ("NEW GLOBAL BEST" if f"p{p:02d}_{n}" == new_name else
                            "not ok" if not ver[n]["ok"] else
                            f"min {ver[n]['min']:.4f} not > {a.improve:.2f} x incumbent {inc:.4f}")
        if new_dir != G:
            new_dir = copy_cand(new_dir, a.root / "global" / new_name)
        st["spend_total"] = round(st["spend_total"] + sum(sp.values()), 2)
        summary = {"period": p, "started": started, "ended": time.strftime("%F %T"), "incumbent": gname,
                   "incumbent_score_before": st["global"]["score"], "incumbent_remeasured": inc,
                   "verify": {n: {k: v[k] for k in ("runs", "mean", "min", "ok")} for n, v in ver.items()},
                   "notes": notes, "spend": sp, "spend_total": st["spend_total"], "harness_rc": rcs,
                   "global": new_name,
                   "global_score": ver[new_name.split("_", 1)[1]]["mean"] if new_name != gname else inc,
                   "torch_compile": tc}
        st["global"] = {"name": new_name, "dir": str(new_dir), "score": summary["global_score"]}
        st["periods"].append({k: summary[k] for k in ("period", "started", "ended", "incumbent", "global",
                                                       "global_score", "spend", "spend_total")})
        sf.write_text(json.dumps(st, indent=1))
        ad = archive(a, p, pdir, best_dirs, ver, new_name, new_dir, summary)
        log(f"p{p:02d}: global best {new_name} ({summary['global_score']:.4f}x); archive {ad}")
        if a.push:
            git_push(a, p, ad)
    log(f"league done: global best {st['global']['name']} ({st['global']['score']:.4f}x), "
        f"harness spend ${st['spend_total']:.2f}")


if __name__ == "__main__":
    main()
