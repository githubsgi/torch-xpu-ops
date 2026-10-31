"""Per-stage optimization: best-of loop around a CoVeR loop (ports of agents/optimizer_agent.py:optimize_stage and
agents/cover.py).

CoVeR (write -> compile_and_verify -> refine until "Success!"): every attempt is one headless Claude Code session
that edits cand/ in place; the HARNESS then runs compile_and_verify (static checks + the evaluator's quick suite) and
the exact observation goes into the next attempt's prompt together with the diff of what the LLM currently has.
Best-of (AGENT_MAX_ITERATIONS): each iteration restarts from the best candidate so far, with the last three attempt
summaries and "Try a DIFFERENT approach to beat the current best".
Keep gate: score >= 1.02 x the score of the candidate the iteration started from.
"""

import json
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import kb as kbmod
from . import measure, prompts, workspace
from .env import BudgetExceeded
from .models import Stage
from .trials import MIN_IMPROVEMENT, keepable

EDIT_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep", "Bash(ggemm-eval diff:*)", "Bash(diff:*)", "Bash(ls:*)",
              "Bash(cat:*)", "Bash(grep:*)", "Bash(head:*)", "Bash(tail:*)", "Bash(wc:*)", "Bash(sed -n:*)")


@dataclass
class Ctx:
    """Everything a stage needs; mutated by the pipeline as the incumbent changes."""
    workdir: Path
    llm: object
    tree: object
    lessons: object
    refs: Path
    suite: str
    max_iterations: int
    max_attempts: int
    stop_on_no_improvement: bool
    log: object
    incumbent: str = "t0"
    incumbent_res: dict = field(default_factory=dict)   # full evaluator result of the incumbent
    base_res: dict = field(default_factory=dict)        # production-vs-production baseline result
    profiles: dict = field(default_factory=dict)        # (case, op) -> digest text for the incumbent
    parallel: int = 4
    gate: float = MIN_IMPROVEMENT   # keep gate (original optimizer_agent._MIN_IMPROVEMENT = 1.02)


@dataclass
class StageOutcome:
    stage: str
    accepted: bool = False
    trial: str = None
    start_trial: str = None
    start_score: float = None
    score: float = None
    iterations: int = 0
    attempts: int = 0
    trials: list = field(default_factory=list)
    summary: str = ""
    cost_usd: float = 0.0
    seconds: float = 0.0
    history: list = field(default_factory=list)

    def to_dict(self):
        return self.__dict__.copy()


def _first_line(thought, n=240):
    t = " ".join((thought or "").split())
    return t[:n]


def _change_of(thought):
    """CHANGE: ... part of the LLM's final message (what it did; the hypothesis is not recorded as a lesson)."""
    import re
    m = re.search(r"CHANGE:\s*(.*?)(?:HYPOTHESIS:|$)", " ".join((thought or "").split()), re.S)
    return (m.group(1).strip() if m else _first_line(thought))[:240]


def cover(ctx, stage, ws, base_tid, it, ref_score, prompt_kwargs, history_text):
    """One CoVeR loop. Returns (success_trial_id|None, [trial ids], last observation)."""
    base_dir = ctx.tree.path_of(base_tid)
    cand = Path(ws) / "cand"
    base_sha = workspace.cand_sha(base_dir)
    traj, tids, seen = [], [], {}
    for a in range(ctx.max_attempts):
        spent0 = ctx.llm.spent
        diff = workspace.diff_text(base_dir, cand) if a else ""
        prompt = prompts.stage_prompt(stage, trajectory_text=prompts.trajectory_block(traj, diff),
                                      history=history_text, base_sha=base_sha, **prompt_kwargs)
        r = ctx.llm.call(f"{stage.value}-i{it}a{a + 1}", prompt, ws, tools=EDIT_TOOLS, system=prompts.SYSTEM,
                         timeout_s=2400)
        thought = r["text"] or f"(llm session failed: {r['error'][:200]})"
        sha = workspace.cand_sha(cand)
        label = f"xf-{stage.value}-i{it}a{a + 1}"
        if sha == base_sha:
            obs, ok, res = ("NO CHANGE: cand/ is identical to the starting point. Make an actual change that tests "
                            "your hypothesis."), False, None
        elif sha in seen:
            obs, ok, res = f"REPEAT: cand/ is identical to attempt {seen[sha][0]}; its observation was:\n{seen[sha][1]}", False, None
        else:
            obs, ok, res = measure.compile_and_verify(cand, ctx.suite, ref_score, label)
            tid = ctx.tree.add(cand, parent=base_tid, stage=stage.value, iteration=it, attempt=a + 1,
                               strategy=_change_of(thought) or _first_line(thought), res=res or {"status": "invalid", "error": obs},
                               cost_usd=ctx.llm.spent - spent0)
            tids.append(tid)
            t = ctx.tree.get(tid)
            if res is None:
                ctx.tree.update(tid, status="invalid", error=obs[:500])
            sc = f"score {t['score']:.3f}x (parent {ctx.tree.get(base_tid).get('score') or 0:.3f}x)" if t.get("score") else \
                f"status {t['status']}"
            op = f"; per-op {t['per_op']}" if t.get("per_op") else ""
            ctx.lessons.append(f"trial {tid} [{stage.value} i{it}a{a + 1}] changed: {t['strategy'] or '?'} | measured: {sc}{op}"
                               + ("" if t.get("status") == "ok" else f" | error: {obs[:160]!r}"))
            ctx.log(f"    {stage.value} i{it} a{a + 1}: {tid} {sc if t.get('score') else t['status']}"
                    + (" [Success!]" if ok else ""))
            seen[sha] = (a + 1, obs)
            if ok:
                return tid, tids, obs
        traj.append({"thought": thought, "observation": obs})
    return None, tids, traj[-1]["observation"] if traj else ""


def optimize_stage(ctx, stage, issues):
    """Best-of loop. Does not mutate ctx.incumbent; the pipeline accepts the outcome."""
    out = StageOutcome(stage.value, start_trial=ctx.incumbent)
    t0, spent0 = time.time(), ctx.llm.spent
    inc = ctx.tree.get(ctx.incumbent)
    out.start_score = inc["score"]
    issues_text = "\n".join(i.render() for i in issues)
    kb_ctx = kbmod.stage_context(stage)
    stage_best = None   # trial id
    attempt_history = []
    digests = measure.digests_text(ctx.profiles)

    try:
        for it in range(ctx.max_iterations):
            base_tid = stage_best or ctx.incumbent
            base = ctx.tree.get(base_tid)
            ws = workspace.make_workspace(ctx.workdir / "stages" / stage.value / f"iter{it}" / "ws",
                                          ctx.tree.path_of(base_tid), ctx.refs)
            history_text = ""
            if attempt_history:
                history_text = ("\n=== Previous attempts this stage ===\n" + "\n".join(attempt_history[-3:])
                                + "\nTry a DIFFERENT approach to beat the current best.\n")
            best_score = ctx.tree.get(stage_best)["score"] if stage_best else None
            kwargs = dict(issues_text=issues_text,
                          perf_ctx=measure.perf_context(ctx.base_res, ctx.incumbent_res, best_score),
                          digests=digests, lessons=ctx.lessons.read(), kb_ctx=kb_ctx)
            ctx.log(f"  [{stage.value}] best-of iteration {it + 1}/{ctx.max_iterations} from {base_tid} "
                    f"({base['score']:.3f}x)")
            tid, tids, obs = cover(ctx, stage, ws, base_tid, it, base["score"], kwargs, history_text)
            out.trials += tids
            out.iterations = it + 1
            out.attempts += len(tids)
            ref = tid or (tids[-1] if tids else None)
            summ = f"Attempt {len(attempt_history) + 1}: " + (
                f"achieved {ctx.tree.get(tid)['score']:.3f}x (iteration started at {base['score']:.3f}x)" if tid
                else f"failed ({obs[:140]!r})") + (f" | approach: {ctx.tree.get(ref)['strategy'][:150]}" if ref else "")
            attempt_history.append(summ)
            if tid is None:
                out.history.append(f"iter {it + 1}: CoVeR exhausted {ctx.max_attempts} attempts without Success")
                if ctx.stop_on_no_improvement:
                    break
                continue
            t = ctx.tree.get(tid)
            if keepable(t, base["score"], ctx.gate):
                stage_best = tid
                out.history.append(f"iter {it + 1}: new stage best {tid} {t['score']:.3f}x (was {base['score']:.3f}x)")
                ctx.log(f"  [{stage.value}] new best {tid} {t['score']:.3f}x")
            else:
                out.history.append(f"iter {it + 1}: {tid} {t['score']:.3f}x < {ctx.gate} x {base['score']:.3f}x: not kept")
                ctx.log(f"  [{stage.value}] {tid} {t['score']:.3f}x not >= {ctx.gate}x {base['score']:.3f}x")
                if ctx.stop_on_no_improvement:
                    break
    except BudgetExceeded as e:
        out.history.append(f"stopped: {e}")
        ctx.log(f"  [{stage.value}] {e}")
    out.cost_usd = ctx.llm.spent - spent0
    out.seconds = time.time() - t0
    if stage_best:
        out.accepted, out.trial, out.score = True, stage_best, ctx.tree.get(stage_best)["score"]
        out.summary = ctx.tree.get(stage_best)["strategy"]
    else:
        out.summary = f"reverted: no attempt reached {ctx.gate}x of the incumbent"
    return out
