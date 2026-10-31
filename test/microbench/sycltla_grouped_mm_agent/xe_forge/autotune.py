"""AUTOTUNING stage: configs.json sweeps per kind, evaluated in parallel (port of the autotune stage, where the
original adds @triton.autotune configs; here the harness is the autotuner).

Round (= one best-of iteration): the LLM proposes variants (each changes ONE kind's config); invalid ones are
repaired with the exact error (CoVeR-style, up to max_attempts rounds); valid ones are built and evaluated in
parallel threads; per kind the variant with the best ratio on that kind's ops (>= 1.02 x the base's) is kept; the
combination is evaluated and gated like any other trial (>= 1.02 x the base score).
"""

import concurrent.futures as cf
import copy
import json
import math
import shutil
import time
from pathlib import Path

from . import kb as kbmod
from . import measure, prompts, workspace
from .env import BudgetExceeded, candidate
from .models import KIND_OPS, KINDS, Stage
from .optimizer import StageOutcome, _first_line
from .trials import MIN_IMPROVEMENT, keepable


def _geo(xs):
    xs = [x for x in xs if x and x > 0]
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else 0.0


def kind_ratio(var, base, kind):
    ops = KIND_OPS[kind]
    b = _geo([base["per_op"].get(o) for o in ops])
    v = _geo([var["per_op"].get(o) for o in ops])
    return v / b if b else 0.0


def validate_variant(v, base_cfgs):
    """-> (new_cfgs, error)"""
    if not isinstance(v, dict) or v.get("kind") not in KINDS or not isinstance(v.get("cfg"), dict):
        return None, "variant needs kind in {m_rowB,m_colB,k_grouped} and a cfg object"
    cfgs = copy.deepcopy(base_cfgs)
    cfgs[v["kind"]] = dict(v["cfg"])
    try:
        return candidate.normalize(cfgs), None
    except (candidate.CandidateError, KeyError, ValueError, TypeError) as e:
        return None, f"INVALID: {e}"


def _make_variant_dir(path, base_dir, cfgs):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    shutil.copy(Path(base_dir) / workspace.HEADER, path / workspace.HEADER)
    (path / workspace.CONFIGS).write_text(json.dumps(cfgs, indent=2) + "\n")
    return path


def optimize_autotune(ctx, issues):
    stage = Stage.AUTOTUNING
    out = StageOutcome(stage.value, start_trial=ctx.incumbent)
    t0, spent0 = time.time(), ctx.llm.spent
    out.start_score = ctx.tree.get(ctx.incumbent)["score"]
    issues_text = "\n".join(i.render() for i in issues)
    kb_ctx = kbmod.stage_context(stage)
    digests = measure.digests_text(ctx.profiles)
    stage_best = None
    tried = []   # one line per evaluated variant (history for the LLM)
    n_var = 6
    ws = ctx.workdir / "stages" / stage.value
    ws.mkdir(parents=True, exist_ok=True)
    if not (ws / "refs").exists():
        import os
        os.symlink(os.path.relpath(ctx.refs, ws), ws / "refs")
    try:
        for it in range(ctx.max_iterations):
            base_tid = stage_best or ctx.incumbent
            base = ctx.tree.get(base_tid)
            base_dir = ctx.tree.path_of(base_tid)
            base_cfgs = json.loads((base_dir / workspace.CONFIGS).read_text())
            ctx.log(f"  [autotuning] round {it + 1}/{ctx.max_iterations} from {base_tid} ({base['score']:.3f}x)")
            hist = ("\n=== Variants already evaluated (do not repeat) ===\n" + "\n".join(tried[-24:]) + "\n") if tried else ""
            failures = ""
            evaluated = []   # (variant, trial id)
            for rnd in range(ctx.max_attempts):
                prompt = prompts.autotune_prompt(base_cfgs, measure.perf_context(ctx.base_res, ctx.incumbent_res),
                                                 digests, ctx.lessons.read(), kb_ctx, hist, issues_text, n_var, failures)
                r = ctx.llm.call(f"autotuning-i{it}r{rnd + 1}", prompt, ctx.workdir, tools=(), system=prompts.SYSTEM,
                                 json_schema=prompts.AUTOTUNE_SCHEMA, timeout_s=900, budget_usd=4)
                variants = (r.get("structured") or {}).get("variants", [])[:n_var + 2]
                good, bad = [], []
                for k, v in enumerate(variants):
                    cfgs, err = validate_variant(v, base_cfgs)
                    key = json.dumps(cfgs, sort_keys=True) if cfgs else None
                    if err:
                        bad.append(f"{v.get('name', k)}: {err}")
                    elif cfgs == base_cfgs or key in {e[2] for e in evaluated}:
                        bad.append(f"{v.get('name', k)}: identical to the base or an earlier variant")
                    else:
                        good.append((v, cfgs, key))
                ctx.log(f"    round {it + 1}.{rnd + 1}: {len(variants)} proposed, {len(good)} valid, {len(bad)} rejected")
                # CoVeR-style refinement only when nothing valid came back
                if not good and rnd + 1 < ctx.max_attempts:
                    failures = ("\n=== Your last proposal had problems (fix them) ===\n" + "\n".join(bad or ["no variants returned"]) + "\n")
                    continue
                results = _evaluate_parallel(ctx, it, base_dir, good, stage)
                for (v, cfgs, key), (tid, res) in zip(good, results):
                    evaluated.append((v, tid, key))
                    t = ctx.tree.get(tid)
                    ratio = kind_ratio(t, base, v["kind"]) if t.get("per_op") and base.get("per_op") else 0
                    line = (f"{v.get('name')}: {v['kind']} {json.dumps(v['cfg'], separators=(',', ':'))} -> "
                            + (f"score {t['score']:.3f}x, kind ratio {ratio:.3f}" if t.get("score") else f"{t['status']}"))
                    tried.append(line)
                    ctx.lessons.append(f"trial {tid} [autotuning i{it}] changed: {v['kind']} cfg "
                                       f"{json.dumps(v['cfg'], separators=(',', ':'))} | measured: "
                                       + (f"score {t['score']:.3f}x (base {base['score']:.3f}x), {v['kind']} ops ratio {ratio:.3f}, per-op {t['per_op']}"
                                          if t.get("score") else f"status {t['status']} {t.get('error', '')[:120]!r}"))
                    ctx.log(f"    {v.get('name')}: {tid} {line.split('-> ')[-1]}")
                break
            out.trials += [tid for _, tid, _ in evaluated]
            out.attempts += len(evaluated)
            out.iterations = it + 1
            # per kind winners
            chosen = {}
            for v, tid, _ in evaluated:
                t = ctx.tree.get(tid)
                if t.get("status") != "ok" or not t.get("per_op"):
                    continue
                ratio = kind_ratio(t, base, v["kind"])
                if ratio >= ctx.gate and (v["kind"] not in chosen or ratio > chosen[v["kind"]][0]):
                    chosen[v["kind"]] = (ratio, v, tid)
            if not chosen:
                out.history.append(f"round {it + 1}: no variant beat the base by {ctx.gate}x on its kind")
                if ctx.stop_on_no_improvement:
                    break
                continue
            # a single winner already is a full trial; evaluate the combination only if >= 2 kinds changed
            if len(chosen) == 1:
                tid = next(iter(chosen.values()))[2]
                cand_tid = tid
            else:
                merged = copy.deepcopy(base_cfgs)
                for kind, (_, v, _) in chosen.items():
                    merged[kind] = dict(v["cfg"])
                merged = candidate.normalize(merged)
                d = _make_variant_dir(ws / f"iter{it}" / "combo", base_dir, merged)
                obs, ok, res = measure.compile_and_verify(d, ctx.suite, base["score"], f"xf-autotuning-i{it}-combo")
                cand_tid = ctx.tree.add(d, parent=base_tid, stage=stage.value, iteration=it, attempt=99,
                                        strategy="combination of per-kind winners: " + ", ".join(
                                            f"{k}={v['name']}" for k, (_, v, _) in chosen.items()),
                                        res=res or {"status": "invalid", "error": obs})
                out.trials.append(cand_tid)
                ct = ctx.tree.get(cand_tid)
                ctx.lessons.append(f"trial {cand_tid} [autotuning i{it}] changed: {ct['strategy']} | measured: "
                                   + (f"score {ct['score']:.3f}x per-op {ct['per_op']}" if ct.get("score") else f"status {ct['status']}"))
                ctx.log(f"    combination {cand_tid}: {ct.get('score') or ct['status']}")
            t = ctx.tree.get(cand_tid)
            if keepable(t, base["score"], ctx.gate):
                stage_best = cand_tid
                out.history.append(f"round {it + 1}: new stage best {cand_tid} {t['score']:.3f}x (was {base['score']:.3f}x)")
            else:
                out.history.append(f"round {it + 1}: {cand_tid} {t.get('score')} not >= {ctx.gate}x {base['score']:.3f}x")
                if ctx.stop_on_no_improvement:
                    break
    except BudgetExceeded as e:
        out.history.append(f"stopped: {e}")
        ctx.log(f"  [autotuning] {e}")
    out.cost_usd = ctx.llm.spent - spent0
    out.seconds = time.time() - t0
    if stage_best:
        out.accepted, out.trial, out.score = True, stage_best, ctx.tree.get(stage_best)["score"]
        out.summary = ctx.tree.get(stage_best)["strategy"]
    else:
        out.summary = f"reverted: no config sweep reached {ctx.gate}x of the incumbent"
    return out


def _evaluate_parallel(ctx, it, base_dir, good, stage):
    """Build + evaluate all valid variants concurrently (the shared slot locks bound GPU/build use)."""
    def one(k_v):
        k, (v, cfgs, key) = k_v
        d = _make_variant_dir(ctx.workdir / "stages" / stage.value / f"iter{it}" / f"v{k}_{v.get('name', k)}"[:60]
                              .replace(" ", "_").replace("/", "_"), base_dir, cfgs)
        obs, ok, res = measure.compile_and_verify(d, ctx.suite, None, f"xf-autotuning-i{it}-v{k}")
        return d, obs, res

    with cf.ThreadPoolExecutor(max_workers=max(1, ctx.parallel)) as ex:
        evals = list(ex.map(one, enumerate(good)))
    out = []
    for (v, cfgs, key), (d, obs, res) in zip(good, evals):
        base_tid = ctx.incumbent
        tid = ctx.tree.add(d, parent=_current_base(ctx, base_dir), stage=stage.value, iteration=it, attempt=len(out) + 1,
                           strategy=f"{v['kind']} {json.dumps(v['cfg'], separators=(',', ':'))} ({_first_line(v.get('hypothesis', ''), 120)})",
                           res=res or {"status": "invalid", "error": obs})
        out.append((tid, res))
    return out


def _current_base(ctx, base_dir):
    """Trial id whose saved dir is base_dir."""
    for tid, t in ctx.tree.trials.items():
        if Path(t["dir"]) == Path(base_dir):
            return tid
    return ctx.incumbent
