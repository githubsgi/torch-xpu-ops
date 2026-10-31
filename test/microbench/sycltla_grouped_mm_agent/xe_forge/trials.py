"""Trial tree (port of Xe-Forge core/trial_manager.py): parent/child records of every evaluated candidate,
persisted as state.json; ranking and keep gate.

A trial record:
  id, parent, stage, iteration, attempt, strategy, dir (saved copy of header + configs.json),
  status (evaluator status), correctness ("pass" | "fail" | "unknown"), score (geomean speedup vs production on
  `suite`), per_op, suite, measured_against (digest of the baseline everything is compared to), source
  ("measured": numbers written by the evaluator, never by an LLM), diff_sha, cost_usd.
"""

import hashlib
import json
import shutil
import threading
import time
from pathlib import Path

MIN_IMPROVEMENT = 1.02  # original optimizer_agent._MIN_IMPROVEMENT


def digest_dir(path):
    h = hashlib.sha256()
    for f in ("grouped_mm_kernel_impl.h", "configs.json"):
        p = Path(path) / f
        if p.exists():
            h.update(f.encode() + b"\0" + p.read_bytes())
    return h.hexdigest()


def rank(trial):
    """Ordering key of a trial, or None if it does not order (original: _rank)."""
    if trial.get("correctness") != "pass" or trial.get("status") != "ok":
        return None
    sc = trial.get("score")
    if sc is None or sc <= 0:
        return None
    return (round(float(sc), 4), 1, -float(trial.get("cand_ms") or 0.0))


def best_id(trials):
    best, bid = None, None
    for tid, t in trials.items():
        r = rank(t)
        if r is not None and (best is None or r > best):
            best, bid = r, tid
    return bid


def keepable(trial, ref_score, factor=MIN_IMPROVEMENT):
    """Gate: measured, correct, and >= factor x the incumbent's score."""
    r = rank(trial)
    return r is not None and trial["score"] >= ref_score * factor


class TrialTree:
    def __init__(self, workdir, baseline_digest=""):
        self.dir = Path(workdir) / "trials"
        self.path = self.dir / "state.json"
        self.lock = threading.RLock()
        self.dir.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            self.state = json.loads(self.path.read_text())
        else:
            self.state = {"baseline_digest": baseline_digest, "trials": {}, "best_trial": None, "next_id": 0,
                          "created": time.strftime("%F %T")}
            self._save()

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1))
        tmp.replace(self.path)

    @property
    def trials(self):
        return self.state["trials"]

    def get(self, tid):
        return self.state["trials"][tid]

    def path_of(self, tid):
        return Path(self.get(tid)["dir"])

    def add(self, cand_dir, *, parent, stage, iteration=0, attempt=0, strategy="", res=None, cost_usd=0.0):
        """Save a copy of the candidate and its measurement. Returns the trial id."""
        with self.lock:
            tid = f"t{self.state['next_id']}"
            self.state["next_id"] += 1
            dest = self.dir / tid
            dest.mkdir(parents=True, exist_ok=True)
            for f in ("grouped_mm_kernel_impl.h", "configs.json"):
                if (Path(cand_dir) / f).exists():
                    shutil.copy(Path(cand_dir) / f, dest / f)
            rec = {"id": tid, "parent": parent, "stage": stage, "iteration": iteration, "attempt": attempt,
                   "strategy": strategy[:600], "dir": str(dest), "diff_sha": digest_dir(dest)[:12],
                   "measured_against": self.state["baseline_digest"], "time": time.strftime("%F %T"),
                   "cost_usd": round(cost_usd, 4), "status": None, "correctness": "unknown", "score": None,
                   "source": "measured"}
            if res is not None:
                rec.update(self.measure_fields(res))
                try:
                    (dest / "result.json").write_text(json.dumps(res, indent=1, default=str))
                except (TypeError, ValueError):
                    pass
            self.state["trials"][tid] = rec
            self.state["best_trial"] = best_id(self.state["trials"])
            self._save()
            return tid

    @staticmethod
    def measure_fields(res):
        st = res.get("status")
        f = {"status": st, "suite": res.get("suite"), "job": res.get("job"),
             "correctness": "pass" if st == "ok" else ("fail" if st in ("verify_failed", "bench_mismatch") else "unknown")}
        if st == "ok":
            f["score"] = round(res["score"], 4)
            f["per_op"] = {k: round(v, 4) for k, v in res["per_op"].items()}
            f["per_case"] = {k: round(v["speedup"], 4) for k, v in res["per_case"].items()}
            f["cand_ms"] = round(sum(v["cand_ms"] for v in res["per_case"].values()), 5)
            f["base_ms"] = round(sum(v["base_ms"] for v in res["per_case"].values()), 5)
        else:
            f["error"] = str(res.get("error", ""))[:800]
        return f

    def update(self, tid, **kw):
        with self.lock:
            self.state["trials"][tid].update(kw)
            self.state["best_trial"] = best_id(self.state["trials"])
            self._save()

    def result(self, tid):
        """Full evaluator result of a trial (per-case bench rows), or {}."""
        p = Path(self.get(tid)["dir"]) / "result.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def children(self, tid):
        return [t for t in self.trials.values() if t["parent"] == tid]

    def lineage(self, tid):
        out = []
        while tid:
            out.append(tid)
            tid = self.get(tid).get("parent")
        return out[::-1]

    def render_tree(self):
        roots = [t["id"] for t in self.trials.values() if not t["parent"]]
        lines = []

        def walk(tid, depth):
            t = self.get(tid)
            sc = f"{t['score']:.3f}x" if t.get("score") else t.get("status") or "?"
            lines.append(f"{'  ' * depth}{tid} [{t['stage']} i{t['iteration']}a{t['attempt']}] {sc} {t['strategy'][:70]}")
            for c in self.children(tid):
                walk(c["id"], depth + 1)

        for r in roots:
            walk(r, 0)
        return "\n".join(lines)
