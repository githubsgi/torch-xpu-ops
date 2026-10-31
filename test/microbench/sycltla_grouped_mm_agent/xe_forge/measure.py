"""compile_and_verify (= evaluator), performance context and profile digests."""

import concurrent.futures as cf
import json
import re
import threading
import time
from pathlib import Path

from . import env
from .env import evaluate, profile
from .models import KIND_OPS
from .workspace import static_checks

SUCCESS = "Success!"
REGRESSION_FLOOR = 0.98   # below this fraction of the incumbent score the observation is a PERFORMANCE REGRESSION
PROFILE_MATRIX = [("tp4ep4_uniform", "w13_fwd"), ("tp4ep4_uniform", "w13_dX"), ("tp4ep4_uniform", "w13_dW"),
                  ("ep8_uniform", "w13_fwd"), ("ep8_uniform", "w13_dX"), ("ep8_uniform", "w13_dW")]


def compile_and_verify(cand, suite, incumbent_score, label):
    """Returns (observation, success, result|None). The observation is "Success!" exactly when the candidate
    builds, passes the hidden verification and timed-output checks, and is not a regression vs the incumbent."""
    msg = static_checks(cand)
    if msg:
        return msg, False, None
    res = evaluate.evaluate(cand, suite=suite, label=label)
    try:
        evaluate.save_to_cand(res)
    except Exception:  # noqa: BLE001
        pass
    st = res["status"]
    if st != "ok":
        return _failure_text(res), False, res
    if incumbent_score and res["score"] < incumbent_score * REGRESSION_FLOOR:
        return (f"PERFORMANCE REGRESSION: score {res['score']:.3f}x vs the incumbent's {incumbent_score:.3f}x "
                f"(production = 1.000x; suite {suite}). Try a different approach.\n" + _perf_lines(res)), False, res
    return SUCCESS, True, res


def _failure_text(res):
    t = res["text"]
    if len(t) > 7000:
        t = t[:4500] + "\n...[truncated]...\n" + t[-2400:]
    return t


def _perf_lines(res):
    L = [f"per-op: " + ", ".join(f"{k} {v:.3f}x" for k, v in res["per_op"].items())]
    for name, c in res["per_case"].items():
        L.append(f"case {name}: {c['speedup']:.3f}x")
    return "\n".join(L)


def perf_table(res, title="current best"):
    """Per case/op: production vs candidate ms and TFLOPS (original: _build_performance_context)."""
    if not res or res.get("status") != "ok":
        return "(not measured)"
    L = [f"{title}: score {res['score']:.3f}x vs production on suite {res['suite']} "
         f"(per-op " + ", ".join(f"{k} {v:.3f}x" for k, v in res["per_op"].items()) + ")"]
    for x in res["bench"]:
        L.append(f"  {x['case']:16s} {x['op']:8s} production {x['base_ms']:.4f} ms ({x['base_tflops']:.0f} TF)  "
                 f"{title} {x['cand_ms']:.4f} ms ({x['cand_tflops']:.0f} TF)  {x['speedup']:.3f}x")
    return "\n".join(L)


def perf_context(base_res, cur_res, stage_best=None):
    L = ["=== Performance context ===",
         "Score is production/candidate time (> 1 faster), quick suite = tp4ep4_uniform + ep8_uniform; noise < 2%."]
    L.append(perf_table(cur_res))
    if stage_best:
        L.append(f"Best achieved in this stage so far: {stage_best:.3f}x — your next attempt must beat it by 2% "
                 f"to be accepted.")
    L.append("A candidate that scores below 0.98x of the incumbent is reported as PERFORMANCE REGRESSION.")
    return "\n".join(L)


def collect_profiles(cand, workdir, tag, matrix=PROFILE_MATRIX, log=print):
    """Profile (case, op) pairs of `cand` in parallel threads; digests saved under <workdir>/profiles/<tag>/."""
    out_dir = Path(workdir) / "profiles" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    def one(case, op):
        f = out_dir / f"{case}__{op}.txt"
        if f.exists():
            return case, op, f.read_text()
        try:
            r = profile.profile(cand, case=case, op=op, label=f"{tag}-{case}-{op}")
            text = r["text"]
        except Exception as e:  # noqa: BLE001
            text = f"PROFILE FAILED: {e}"
        f.write_text(text)
        return case, op, text

    with cf.ThreadPoolExecutor(max_workers=len(matrix)) as ex:
        for case, op, text in ex.map(lambda a: one(*a), matrix):
            results[(case, op)] = text
            log(f"  profile {case} {op}: {'ok' if 'PROFILE FAILED' not in text else 'FAILED'}")
    return results


def digests_text(profiles, ops=None, max_chars_each=2600):
    parts = []
    for (case, op), text in profiles.items():
        if ops and op not in ops:
            continue
        t = text if len(text) <= max_chars_each else text[:max_chars_each] + "\n...[truncated]"
        parts.append(f"##### profile {case} / {op}\n{t}")
    return "\n\n".join(parts) if parts else "(no profile available)"
