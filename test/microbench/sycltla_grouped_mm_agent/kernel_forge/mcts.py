"""Kernel Forge port: MCTS over kernel versions for the SYCL-TLA grouped GEMM.

    python -m kernel_forge.mcts --workdir RUNDIR --iterations 50 --workers 8 \
        --knob-ratio 0.3 --model opus --max-llm-usd 50

Run from the sycltla_grouped_mm_agent/ directory (or with it on PYTHONPATH). Re-running
with the same --workdir resumes from RUNDIR/tree.json; --iterations is the TOTAL target.
See README.md for the mapping to the paper.
"""

import argparse
import concurrent.futures as cf
import difflib
import json
import os
import random
import shutil
import sys
import threading
import time
import traceback
from pathlib import Path

from . import _paths  # noqa: F401
from ggemm_agent import candidate, config, evaluate, llm, profile

from . import knobs, prompts
from .tree_store import Node, Tree, content_hash

REPAIRABLE = ("invalid", "build_error", "runtime_error", "verify_failed", "bench_mismatch")
KF_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep", "Bash(ls:*)", "Bash(cat:*)", "Bash(diff:*)",
            "Bash(grep:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(wc:*)", "Bash(sed -n:*)")
INFRA_MARKERS = ("build phase failed", "run phase failed", "ssh ", "Connection", "timed out")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Forge:
    def __init__(self, a):
        self.a = a
        self.wd = Path(a.workdir).resolve()
        self.wd.mkdir(parents=True, exist_ok=True)
        (self.wd / "logs").mkdir(exist_ok=True)
        self.t0 = time.time()
        self.io_lock = threading.Lock()
        self.llm_inflight = 0
        self.cost_avg = []
        self.focus_ctr = {}
        if (self.wd / "tree.json").exists():
            self.tree = Tree.load(self.wd)
            log(f"resumed {self.wd}: {len(self.tree.nodes)} nodes, {self.tree.iterations_done} iterations, "
                f"LLM ${self.tree.llm_cost:.2f}")
            self.resumed = True
        else:
            self.tree = Tree(self.wd)
            self.resumed = False
        self.tree.meta.setdefault("runs", []).append(
            {"start": time.strftime("%F %T"), "argv": sys.argv[1:], "args": {k: v for k, v in vars(a).items()}})

    # ------------------------------------------------------------------ io
    def jlog(self, rec):
        rec = {"time": time.strftime("%F %T"), **rec}
        with self.io_lock, open(self.wd / "iterations.jsonl", "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    def wall_exceeded(self):
        return self.a.max_wall_min and (time.time() - self.t0) / 60 > self.a.max_wall_min

    # ---------------------------------------------------------------- root
    def ensure_root(self):
        t = self.tree
        if t.root:
            return
        d = t.node_dir("n000")
        candidate.init_dir(d)
        hdr, cfgs = candidate.load(d)
        log(f"evaluating production root on '{self.a.suite}'")
        res = evaluate.evaluate(d, suite=self.a.suite, label="kf-root")
        evaluate.save_to_cand(res)
        if res["status"] != "ok":
            raise SystemExit(f"production root failed to evaluate: {res['text']}")
        # The root is the reference: L = 1 by definition; the measured self-score is the noise estimate.
        n = Node(id="n000", parent=None, depth=0, visits=1, action="root", notes="production kernel",
                 chash=content_hash(hdr, cfgs), score=1.0, eval=self.pack_eval(res))
        n.eval["measured_self_score"] = res["score"]
        t.add_node(n)
        t.save()
        log(f"root ready: production vs itself measured {res['score']:.4f}x (noise estimate)")

    @staticmethod
    def pack_eval(res):
        return {"status": res["status"], "score": res.get("score", 0.0), "suite": res.get("suite"),
                "per_op": res.get("per_op", {}),
                "per_case": {k: {"speedup": round(v["speedup"], 4), "probe": v.get("probe", False)}
                             for k, v in res.get("per_case", {}).items()},
                "tflops": {f"{x['case']}/{x['op']}": round(x["cand_tflops"], 1) for x in res.get("bench", [])},
                "seconds": res.get("seconds"), "job": res.get("job"), "node": res.get("node"),
                "spills": (res.get("build") or {}).get("spills", [])}

    # ----------------------------------------------------------- main loop
    def search(self):
        t, a = self.tree, self.a
        rng = random.Random(a.seed + t.iterations_done)
        pending = {}
        ex = cf.ThreadPoolExecutor(max_workers=a.workers)
        log(f"search: target {a.iterations} iterations, {a.workers} workers, knob ratio {a.knob_ratio}, "
            f"C={a.c}, suite {a.suite}")
        try:
            while True:
                stop = self.wall_exceeded()
                while (not stop and len(pending) < a.workers
                       and t.iterations_done + len(pending) < a.iterations):
                    fut, info = self.launch(ex, rng)
                    pending[fut] = info
                if not pending:
                    break
                done, _ = cf.wait(list(pending), timeout=60, return_when=cf.FIRST_COMPLETED)
                for f in done:
                    self.finish(f, pending.pop(f))
                if stop and not done:
                    log("wall-time budget reached: waiting for in-flight expansions")
        except KeyboardInterrupt:
            log("interrupted: saving tree (in-flight expansions are forgotten)")
            for f in pending:
                f.cancel()
            t.save()
            raise
        finally:
            ex.shutdown(wait=False, cancel_futures=True)
            t.save()

    def llm_allowed(self):
        t, a = self.tree, self.a
        if a.knob_ratio >= 1.0:
            return False
        est = max(0.3, sum(self.cost_avg) / len(self.cost_avg)) if self.cost_avg else a.llm_call_usd / 3
        return t.llm_cost + (self.llm_inflight + 1) * est <= a.max_llm_usd

    def launch(self, ex, rng):
        t, a = self.tree, self.a
        path, sel = t.select(rng, a.c, a.vloss, a.alpha0, a.alpha1, a.anneal_steps)
        parent = t.nodes[path[-1]]
        aid = t.new_attempt_id()
        want_knob = rng.random() < a.knob_ratio
        action = "knob" if (want_knob or not self.llm_allowed()) else "llm"
        if action == "llm":
            self.llm_inflight += 1
        info = {"aid": aid, "path": path, "parent": parent.id, "action": action, "sel": sel,
                "t0": time.time(), "llm_forced_knob": (action == "knob" and not want_knob)}
        log(f"{aid}: expand {parent.id} (depth {parent.depth}, N={parent.visits}, "
            f"{len(parent.children)} children) via {action}; alpha={sel['alpha']}; path {'>'.join(path)}")
        t.save()
        fut = ex.submit(self.expand, aid, parent.id, action, random.Random(f"{a.seed}-{aid}"))
        return fut, info

    def finish(self, fut, info):
        t, a = self.tree, self.a
        if info["action"] == "llm":
            self.llm_inflight -= 1
        try:
            out = fut.result()
        except Exception as e:  # noqa: BLE001  worker bug: record as a failed iteration, keep going
            out = {"ok": False, "status": "harness_error", "error": f"{e!r}\n{traceback.format_exc()[-1500:]}",
                   "notes": "", "cost_usd": 0.0}
        parent = t.nodes[info["parent"]]
        t.release(info["path"])
        t.llm_cost += out.get("cost_usd", 0.0)
        if out.get("llm_calls"):
            t.llm_calls += out["llm_calls"]
            self.cost_avg.append(out["cost_usd"])
        rec = {"iter": t.iterations_done + 1, "aid": info["aid"], "parent": parent.id, "action": info["action"],
               "forced_knob": info["llm_forced_knob"], "path": info["path"], "alpha": info["sel"]["alpha"],
               "selection": info["sel"]["trace"], "seconds": round(time.time() - info["t0"], 1),
               "cost_usd": round(out.get("cost_usd", 0.0), 4), "repairs": out.get("repairs", 0),
               "notes": out.get("notes", "")}
        if out["ok"]:
            nid = "n" + info["aid"][1:]
            shutil.copytree(out["dir"], t.node_dir(nid), dirs_exist_ok=True)
            node = Node(id=nid, parent=parent.id, depth=parent.depth + 1, visits=0, action=info["action"],
                        notes=out["notes"], auto_change=out["auto_change"], chash=out["chash"],
                        score=out["score"], eval=out["eval"], cost_usd=out.get("cost_usd", 0.0),
                        repairs=out.get("repairs", 0), seconds=rec["seconds"])
            t.add_node(node)
            t.backprop(info["path"], nid)
            best_sub = 1 / t.best_sub_L(parent.id)
            rec.update(status="ok", node=nid, score=round(out["score"], 4),
                       vs_parent=round(out["score"] / parent.score, 4), parent_best_subtree=round(best_sub, 4))
            log(f"{info['aid']}: ADMITTED {nid} score {out['score']:.4f}x ({info['action']}, "
                f"repairs {out.get('repairs', 0)}, ${out.get('cost_usd', 0):.2f}) {out['notes'][:110]}")
        else:
            t.backprop(info["path"], None)      # the attempt was a visit; no child to admit
            t.add_failure(id=info["aid"], parent=parent.id, action=info["action"], status=out["status"],
                          notes=out.get("notes", ""), error=str(out.get("error", ""))[:700],
                          chash=out.get("chash"), cost_usd=out.get("cost_usd", 0.0),
                          repairs=out.get("repairs", 0))
            rec.update(status=out["status"], error=str(out.get("error", ""))[:300])
            log(f"{info['aid']}: failed ({out['status']}, repairs {out.get('repairs', 0)}, "
                f"${out.get('cost_usd', 0):.2f}) {str(out.get('error', ''))[:100]!r}")
        t.iterations_done += 1
        self.jlog(rec)
        t.save()

    # --------------------------------------------------------------- actions
    def expand(self, aid, parent_id, action, rng):
        t = self.tree
        parent = t.nodes[parent_id]
        d = self.wd / "attempts" / aid
        candidate.init_dir(d, src=t.node_dir(parent_id))
        phdr, pcfg = candidate.load(t.node_dir(parent_id))
        if action == "knob":
            return self.knob_expand(aid, d, parent, phdr, pcfg, rng)
        return self.llm_expand(aid, d, parent, phdr, pcfg)

    def evaluate_node(self, d, aid):
        res = evaluate.evaluate(d, suite=self.a.suite, label=f"kf-{aid}")
        if res["status"] == "runtime_error" and any(m in str(res.get("error", "")) for m in INFRA_MARKERS):
            time.sleep(20)
            res = evaluate.evaluate(d, suite=self.a.suite, label=f"kf-{aid}-retry")
        evaluate.save_to_cand(res)
        return res

    def pack_ok(self, aid, d, parent, phdr, pcfg, res, notes, cost, repairs, llm_calls=0):
        hdr, cfgs = candidate.load(d)
        return {"ok": True, "status": "ok", "dir": str(d), "score": res["score"], "eval": self.pack_eval(res),
                "chash": content_hash(hdr, cfgs), "notes": notes, "cost_usd": cost, "repairs": repairs,
                "llm_calls": llm_calls, "auto_change": self.auto_change(phdr, pcfg, hdr, cfgs)}

    @staticmethod
    def auto_change(phdr, pcfg, hdr, cfgs):
        plus = minus = 0
        for l in difflib.unified_diff(phdr.splitlines(), hdr.splitlines(), lineterm="", n=0):
            if l.startswith("+") and not l.startswith("+++"):
                plus += 1
            elif l.startswith("-") and not l.startswith("---"):
                minus += 1
        return f"{knobs.config_diff(pcfg, cfgs)}; header +{plus}/-{minus} lines"

    def knob_expand(self, aid, d, parent, phdr, pcfg, rng):
        t = self.tree
        for _ in range(12):
            new, desc = knobs.mutate(pcfg, rng)
            if new is None:
                return {"ok": False, "status": "no_mutation", "error": desc, "notes": desc}
            h = content_hash(phdr, new)
            if t.reserve_hash(h, aid):
                break
        else:
            return {"ok": False, "status": "duplicate", "error": "all sampled mutations already known",
                    "notes": desc}
        (d / "configs.json").write_text(json.dumps(new, indent=2) + "\n")
        res = self.evaluate_node(d, aid)
        if res["status"] != "ok":
            t.mark_failed_hash(h, aid)
            return {"ok": False, "status": res["status"], "error": self.err_text(res), "notes": desc, "chash": h}
        return self.pack_ok(aid, d, parent, phdr, pcfg, res, desc, 0.0, 0)

    @staticmethod
    def err_text(res, limit=3000):
        if res["status"] in ("verify_failed", "bench_mismatch"):
            lines = [l for l in res["text"].splitlines() if "FAILED" in l or "MISMATCH" in l or "CORRECTNESS" in l]
            return "\n".join(lines)[:limit] or res["text"][:limit]
        return str(res.get("error", res["text"]))[:limit]

    def llm_expand(self, aid, d, parent, phdr, pcfg):
        t, a = self.tree, self.a
        cost, calls, repairs = 0.0, 0, 0
        prof_text, prof_target = self.parent_profile(parent)
        with t.lock:
            k = self.focus_ctr.setdefault(parent.id, sum(1 for c in parent.children if t.nodes[c].action == "llm")
                                          + sum(1 for f in t.failures if f["parent"] == parent.id
                                                and f["action"] == "llm"))
            self.focus_ctr[parent.id] = k + 1
        prompt = prompts.revision_prompt(
            focus=prompts.FOCI[(k + sum(map(ord, parent.id)) % 3) % len(prompts.FOCI)], node=parent, lineage_text=t.lineage_text(parent.id), parent_eval_text=self.eval_digest(parent),
            profile_text=prof_text, profile_target=prof_target, siblings_text=self.siblings_text(parent),
            recent_failures_text=self.recent_failures(parent), cur_cfg_text=json.dumps(pcfg, indent=1))
        notes = ""
        last = {"status": "llm_error", "error": ""}
        while True:
            r = llm.run_claude(prompt, d, tools=KF_TOOLS, system=prompts.SYSTEM, add_dirs=(config.SYCL_TLA,),
                               model=a.model, budget_usd=a.llm_call_usd, timeout_s=int(a.llm_timeout_min * 60),
                               log_path=self.wd / "logs" / "claude.jsonl")
            cost += r["cost_usd"]
            calls += 1
            if not r["ok"] and not r["text"]:
                return {"ok": False, "status": "llm_error", "error": r["error"], "notes": notes,
                        "cost_usd": cost, "llm_calls": calls, "repairs": repairs}
            notes = self.parse_note(r["text"]) or notes
            problem = None
            try:
                hdr, cfgs = candidate.load(d)
            except (candidate.CandidateError, ValueError, KeyError, json.JSONDecodeError) as e:
                problem = ("invalid", f"candidate.load/normalize failed: {e}")
            if problem is None and a.llm_code_only and hdr == phdr and repairs < a.max_repairs:
                problem = ("config_only", "You only edited configs.json. Tile / sub-group / prefetch / tile_k / grid "
                                          "changes are explored by a separate cheap knob operator; make a CODE change "
                                          "to grouped_mm_kernel_impl.h (keep the config edits it needs).")
            if problem is None:
                h = content_hash(hdr, cfgs)
                if h == parent.chash:
                    problem = ("no_change", "The files are identical to the parent: make your edit.")
                elif not t.reserve_hash(h, aid):
                    problem = ("duplicate", f"Identical to an existing candidate ({t.owner_of(h)}); try a "
                                            f"different change.")
            if problem is None:
                res = self.evaluate_node(d, aid)
                if res["status"] == "ok":
                    out = self.pack_ok(aid, d, parent, phdr, pcfg, res, notes or "(no note)", cost, repairs, calls)
                    return out
                t.mark_failed_hash(h, aid)
                problem = (res["status"], self.err_text(res))
                if "BASELINE build failed" in problem[1] or (res["status"] == "runtime_error" and any(
                        m in problem[1] for m in INFRA_MARKERS)):
                    return {"ok": False, "status": "env_error", "error": problem[1], "notes": notes,
                            "cost_usd": cost, "llm_calls": calls, "repairs": repairs, "chash": h}
            status, err = problem
            if repairs >= a.max_repairs or (status not in REPAIRABLE and status not in ("no_change", "duplicate", "config_only")):
                return {"ok": False, "status": status, "error": err, "notes": notes, "cost_usd": cost,
                        "llm_calls": calls, "repairs": repairs}
            repairs += 1
            log(f"{aid}: repair {repairs}/{a.max_repairs} after {status}")
            self.jlog({"aid": aid, "event": "repair", "attempt": repairs, "status": status, "error": err[:400]})
            hdr_now = (d / "grouped_mm_kernel_impl.h").read_text()
            cfg_now = json.loads((d / "configs.json").read_text()) if (d / "configs.json").exists() else {}
            prompt = prompts.repair_prompt(
                status=status, error_text=err, prior_note=notes or "(none)", attempt=repairs,
                max_attempts=a.max_repairs, diff_stat=self.auto_change(phdr, pcfg, hdr_now, cfg_now)
                if isinstance(cfg_now, dict) and cfg_now else "n/a")

    @staticmethod
    def parse_note(text):
        text = (text or "").strip()
        i = text.rfind("CHANGE:")
        return (text[i + 7:] if i >= 0 else text[-500:]).strip()[:700]

    # ---------------------------------------------------- prompt ingredients
    def eval_digest(self, node):
        e = node.eval
        lines = [f"score {node.score:.4f}x on suite {e.get('suite')}; per-op " +
                 ", ".join(f"{k} {v:.3f}x" for k, v in e.get("per_op", {}).items())]
        lines += [f"case {k}: {v['speedup']:.3f}x" for k, v in e.get("per_case", {}).items()]
        lines.append("candidate TFLOPS: " + ", ".join(f"{k} {v}" for k, v in e.get("tflops", {}).items()))
        if e.get("spills"):
            lines.append(f"register spills: {e['spills']}")
        return "\n".join(lines)

    def siblings_text(self, parent):
        t = self.tree
        L = []
        for c in parent.children:
            n = t.nodes[c]
            L.append(f"- {n.id} [{n.action}] {n.notes[:200]} => score {n.score:.4f}x "
                     f"({(n.score / parent.score - 1) * 100:+.1f}% vs parent); {n.auto_change[:160]}")
        for f in t.failures:
            if f["parent"] == parent.id:
                L.append(f"- {f['id']} [{f['action']}] {f['notes'][:160]} => FAILED {f['status']}: "
                         f"{f['error'][:200].replace(chr(10), ' ')}")
        return "\n".join(L) or "(none yet: this is the first child)"

    def recent_failures(self, parent):
        t = self.tree
        fs = [f for f in t.failures if f["parent"] != parent.id and f["status"] in
              ("build_error", "verify_failed", "bench_mismatch", "runtime_error")][-5:]
        return "\n".join(f"- [{f['action']}] {f['notes'][:120]} => {f['status']}: "
                         f"{f['error'][:180].replace(chr(10), ' ')}" for f in fs) or "(none)"

    def parent_profile(self, parent):
        """Profile digest of the parent's weakest op by absolute TFLOPS (cached per node)."""
        t = self.tree
        tf = parent.eval.get("tflops", {})
        pref = {k: v for k, v in tf.items() if k.startswith("tp4ep4_uniform/")} or tf
        if not pref or not self.a.profile:
            return "(profiling disabled)", "n/a"
        target = min(pref, key=pref.get)
        case, op = target.split("/")
        label = f"{op} in {case} at {pref[target]} TFLOPS"
        with t.lock:
            lock = t.profile_locks.setdefault(parent.id, threading.Lock())
        with lock:
            if parent.profile is None:
                log(f"profiling {parent.id}: {label}")
                try:
                    pr = profile.profile(t.node_dir(parent.id), case=case, op=op, label=f"kf-{parent.id}")
                    parent.profile = pr["text"][:4500] if pr.get("ok") else "(profile unavailable) " + pr["text"][:300]
                except Exception as e:  # noqa: BLE001
                    parent.profile = f"(profile failed: {e!r})"
                t.save()
        return parent.profile, label

    # ------------------------------------------------------------ final phase
    def final_eval(self):
        from .report import write_report
        t, a = self.tree, self.a
        ranked = sorted((n for n in t.nodes.values() if n.id != t.root), key=lambda n: -n.score)
        top = ranked[:a.top_k]
        todo = [t.nodes[t.root]] + top
        log(f"final: re-evaluating root + top {len(top)} on '{a.final_suite}'")
        def one(n):
            res = evaluate.evaluate(t.node_dir(n.id), suite=a.final_suite, label=f"kf-final-{n.id}")
            evaluate.save_to_cand(res)
            return n.id, res
        with cf.ThreadPoolExecutor(max_workers=min(len(todo), 6)) as ex:
            for nid, res in ex.map(one, todo):
                t.nodes[nid].full_eval = {**self.pack_eval(res), "suite": a.final_suite, "text": res["text"]}
                log(f"final {nid}: {res['status']} {res.get('score', 0):.4f}x on {a.final_suite}")
        t.save()
        write_report(self)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--iterations", type=int, default=50, help="total expansion attempts (resume: total target)")
    ap.add_argument("--workers", type=int, default=4, help="parallel expansions (<= 16 please)")
    ap.add_argument("--knob-ratio", type=float, default=0.3, help="fraction of expansions that are knob mutations")
    ap.add_argument("--model", default=os.environ.get("GGEMM_LLM_MODEL", "opus"))
    ap.add_argument("--max-llm-usd", type=float, default=50.0, help="LLM budget; knobs only afterwards")
    ap.add_argument("--llm-call-usd", type=float, default=4.0, help="cap of one claude session")
    ap.add_argument("--llm-timeout-min", type=float, default=40.0)
    ap.add_argument("--max-wall-min", type=float, default=0.0, help="0 = unlimited")
    ap.add_argument("--max-repairs", type=int, default=2)
    ap.add_argument("--llm-allow-config-only", dest="llm_code_only", action="store_false",
                    help="accept LLM revisions that only edit configs.json (default: ask for a header change "
                         "while repair rounds remain; the knob operator covers config changes)")
    ap.add_argument("--suite", default="quick", help="search suite (quick|small)")
    ap.add_argument("--final-suite", default="full")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--c", type=float, default=0.1,
                    help="LCB exploration constant on L=1/score (paper: 1.0 on ms latencies; see README)")
    ap.add_argument("--vloss", type=float, default=0.05, help="virtual loss added to L per in-flight pass")
    ap.add_argument("--alpha0", type=float, default=0.5)
    ap.add_argument("--alpha1", type=float, default=0.3)
    ap.add_argument("--anneal-steps", type=int, default=1000, help="root visits over which alpha anneals")
    ap.add_argument("--no-profile", dest="profile", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-only", action="store_true", help="skip the search, rebuild report.md")
    ap.add_argument("--skip-final", action="store_true", help="search only, no full re-evaluation")
    a = ap.parse_args(argv)
    if a.workers > 16:
        ap.error("--workers must be <= 16 (tiles are shared with the other harnesses)")
    f = Forge(a)
    if a.report_only:
        from .report import write_report
        write_report(f)
        return 0
    f.ensure_root()
    f.search()
    if not a.skip_final:
        f.final_eval()
    else:
        from .report import write_report
        write_report(f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
