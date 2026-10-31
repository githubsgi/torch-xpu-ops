"""Xe-Forge pipeline (port of pipeline.py:optimize): baseline -> analysis -> planning -> stages -> final -> report.

Resumable: pipeline.json records each phase and stage; re-running with the same --workdir skips finished phases and
stages (a stage that was running when the process died is redone; its earlier trials stay in the tree).
"""

import json
import shutil
import time
from pathlib import Path

from . import analysis, measure, workspace
from .autotune import optimize_autotune
from .env import LLM, BudgetExceeded, candidate, evaluate
from .lessons import Lessons
from .models import DetectedIssue, Stage, coerce_issue, stages_needed
from .optimizer import Ctx, optimize_stage
from .report import write_report
from .trials import TrialTree, digest_dir


class Pipeline:
    def __init__(self, args):
        self.args = args
        self.wd = Path(args.workdir)
        self.wd.mkdir(parents=True, exist_ok=True)
        self.pj = self.wd / "pipeline.json"
        self.state = json.loads(self.pj.read_text()) if self.pj.exists() else {
            "run_id": self.wd.name, "created": time.strftime("%F %T"), "phase": "init", "stages": {}, "incumbent": None,
            "args": {k: str(v) for k, v in vars(args).items()}}
        self.logf = open(self.wd / "run.log", "a")

    def log(self, msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        self.logf.write(line + "\n")
        self.logf.flush()

    def save(self):
        tmp = self.pj.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, default=str))
        tmp.replace(self.pj)

    # ------------------------------------------------------------------
    def run(self):
        a, log = self.args, self.log
        t_start = time.time()
        log(f"Xe-Forge (grouped GEMM) run {self.state['run_id']} in {self.wd}")
        self.llm = LLM(self.wd, a.model, a.max_llm_usd, a.per_call_usd, a.effort)
        refs = workspace.make_refs(self.wd)

        # ---- baseline -------------------------------------------------
        base_dir = self.wd / "trials" / "base_cand"
        if not (base_dir / workspace.HEADER).exists():
            candidate.init_dir(base_dir)
        self.tree = TrialTree(self.wd, baseline_digest=digest_dir(base_dir)[:16])
        self.lessons = Lessons(self.wd, a.lessons_dir, tag=self.state["run_id"])
        if "t0" not in self.tree.trials:
            log(f"[baseline] evaluating production on suite {a.suite}")
            res = evaluate.evaluate(base_dir, suite=a.suite, label="xf-baseline")
            if res["status"] != "ok":
                log("baseline evaluation failed:\n" + res["text"])
                self.state["phase"] = "baseline_failed"
                self.save()
                raise SystemExit(2)
            self.tree.add(base_dir, parent=None, stage="baseline", res=res, strategy="production kernel")
            (self.wd / "baseline_eval.txt").write_text(res["text"] + "\n")
        base_res = self.tree.result("t0")
        log(f"[baseline] t0 score {self.tree.get('t0')['score']:.3f}x (production vs production; noise reference)")
        self.ctx = Ctx(workdir=self.wd, llm=self.llm, tree=self.tree, lessons=self.lessons, refs=refs, suite=a.suite,
                       max_iterations=a.max_iterations, max_attempts=a.max_attempts,
                       stop_on_no_improvement=a.stop_on_no_improvement, log=log, base_res=base_res,
                       parallel=a.parallel, gate=a.min_improvement)
        inc = self.state.get("incumbent") or "t0"
        self.ctx.incumbent, self.ctx.incumbent_res = inc, self.tree.result(inc)

        # ---- profiles of the incumbent ---------------------------------
        self._profile(f"inc-{inc}")

        try:
            issues = self._analysis(base_res)
            order = self._plan(issues)
            for stage in order:
                st = self.state["stages"].get(stage.value, {})
                if st.get("status") in ("accepted", "reverted", "failed"):
                    log(f"[{stage.value}] already {st['status']} (resume)")
                    continue
                self._run_stage(stage, [i for i in issues if i.stage == stage])
            self.state["phase"] = "stages_done"
        except BudgetExceeded as e:
            log(f"STOP: {e}")
            self.state["phase"] = "budget_exhausted"
        self.save()
        self._final()
        self.state["seconds_total"] = round(self.state.get("seconds_total", 0) + time.time() - t_start, 1)
        self.state["phase"] = "done" if self.state["phase"] != "budget_exhausted" else "done_budget_exhausted"
        self.save()
        write_report(self)
        log(f"report: {self.wd / 'report.md'}")
        return 0

    # ------------------------------------------------------------------
    def _profile(self, tag):
        if self.args.no_profile:
            self.ctx.profiles = {}
            return
        matrix = measure.PROFILE_MATRIX if self.args.profile_matrix == "full" else measure.PROFILE_MATRIX[2::3]
        self.log(f"[profile] {tag}: {len(matrix)} unitrace profiles in parallel")
        self.ctx.profiles = measure.collect_profiles(self.tree.path_of(self.ctx.incumbent), self.wd, tag, matrix, self.log)

    def _analysis(self, base_res):
        f = self.wd / "analysis.json"
        if f.exists():
            d = json.loads(f.read_text())
            self.log(f"[analysis] resumed: {len(d['issues'])} issues")
            return [i for i in (coerce_issue(x) for x in d["issues"]) if i]
        self.log("[analysis] LLM over kernel + profile digests")
        ws = self.wd / "analysis_ws"
        if ws.exists():
            shutil.rmtree(ws)
        ws.mkdir()
        import os
        os.symlink(os.path.relpath(self.ctx.refs, ws), ws / "refs")
        issues, summary, r = analysis.analyze(self.llm, ws, base_res.get("text", ""), measure.perf_table(base_res, "production"),
                                              measure.digests_text(self.ctx.profiles), self.log)
        f.write_text(json.dumps({"summary": summary, "issues": [i.to_dict() for i in issues],
                                 "cost_usd": r.get("cost_usd")}, indent=1))
        self.log(f"[analysis] {len(issues)} issues:\n" + analysis.issue_summary(issues))
        return issues

    def _plan(self, issues):
        f = self.wd / "plan.json"
        if f.exists():
            d = json.loads(f.read_text())
            self.log(f"[plan] resumed: {d['order']}")
            return [Stage(s) for s in d["order"]]
        needed = stages_needed(issues)
        if self.args.stages:
            wanted = [Stage(s) for s in self.args.stages]
            needed = {s: needed.get(s, []) for s in wanted}
        summary = json.loads((self.wd / "analysis.json").read_text()).get("summary", "")
        ws = self.wd / "analysis_ws"
        order, rationale = analysis.plan(self.llm, ws, needed, issues, summary, self.log)
        order = analysis.enforce_dependencies(order)
        f.write_text(json.dumps({"needed": {s.value: v for s, v in needed.items()}, "order": [s.value for s in order],
                                 "rationale": rationale}, indent=1))
        self.log(f"[plan] {[s.value for s in order]}: {rationale[:300]}")
        return order

    def _run_stage(self, stage, issues):
        ctx, log = self.ctx, self.log
        log(f"[{stage.value}] start from {ctx.incumbent} ({ctx.tree.get(ctx.incumbent)['score']:.3f}x), "
            f"{len(issues)} issues")
        self.state["stages"][stage.value] = {"status": "running", "start_trial": ctx.incumbent}
        self.save()
        if stage == Stage.AUTOTUNING:
            out = optimize_autotune(ctx, issues)
        else:
            out = optimize_stage(ctx, stage, issues)
        d = out.to_dict()
        d["status"] = "accepted" if out.accepted else ("reverted" if out.attempts else "failed")
        self.state["stages"][stage.value] = d
        if out.accepted:
            ctx.incumbent, ctx.incumbent_res = out.trial, ctx.tree.result(out.trial)
            self.state["incumbent"] = out.trial
            log(f"[{stage.value}] ACCEPTED {out.trial} {out.score:.3f}x (was {out.start_score:.3f}x), "
                f"${out.cost_usd:.2f}, {out.seconds:.0f}s")
            self.save()
            self._profile(f"after-{stage.value}-{out.trial}")   # fresh metrics for the later stages
        else:
            log(f"[{stage.value}] reverted ({d['status']}): {out.summary}; ${out.cost_usd:.2f}")
        self.save()

    def _final(self):
        a, ctx = self.args, self.ctx
        best = ctx.incumbent
        bdir = self.wd / "best"
        if bdir.exists():
            shutil.rmtree(bdir)
        candidate.init_dir(bdir, ctx.tree.path_of(best))
        self.log(f"[final] best trial {best} ({ctx.tree.get(best)['score']:.3f}x on {a.suite}); re-evaluating on {a.final_suite}")
        res = evaluate.evaluate(bdir, suite=a.final_suite, label="xf-final")
        evaluate.save_to_cand(res)
        (self.wd / "final_eval.txt").write_text(res["text"] + "\n")
        (self.wd / "final_eval.json").write_text(json.dumps({k: v for k, v in res.items() if k != "text"}, indent=1, default=str))
        self.state["final"] = {"trial": best, "suite": a.final_suite, "status": res["status"], "score": res.get("score"),
                               "per_op": res.get("per_op"), "per_case": {k: v for k, v in (res.get("per_case") or {}).items()}}
        self.log(f"[final] {res['status']} score {res.get('score', 0):.3f}x on {a.final_suite}")
        self.save()
