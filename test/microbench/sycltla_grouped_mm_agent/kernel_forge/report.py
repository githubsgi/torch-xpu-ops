"""report.md: best node's lineage, diff vs production, per-op table, tree summary."""

import json
import shutil
import time
from collections import Counter
from pathlib import Path

from . import _paths  # noqa: F401
from ggemm_agent import candidate

from .knobs import config_diff


def _score_for_ranking(n):
    """Full-suite score when available (the final re-evaluation), else the search-suite score."""
    if n.full_eval and n.full_eval.get("status") == "ok":
        return n.full_eval["score"]
    return n.score if n.full_eval is None else 0.0


def write_report(forge):
    t, a = forge.tree, forge.a
    wd = forge.wd
    root = t.nodes[t.root]
    finals = [n for n in t.nodes.values() if n.full_eval and n.full_eval.get("status") == "ok" and n.id != t.root]
    best = max(finals, key=lambda n: n.full_eval["score"]) if finals else max(
        (n for n in t.nodes.values() if n.id != t.root), key=lambda n: n.score, default=root)
    full_best = best.full_eval["score"] if best.full_eval and best.full_eval.get("status") == "ok" else None
    root_full = root.full_eval["score"] if root.full_eval and root.full_eval.get("status") == "ok" else None
    L = ["# Kernel Forge (MCTS) run report", ""]

    # ---- run summary
    stat = Counter()
    act = Counter()
    for line in _iter_log(wd):
        if "iter" in line:
            stat[line.get("status")] += 1
            act[line.get("action")] += 1
    repairs = sum(n.repairs for n in t.nodes.values()) + sum(f.get("repairs", 0) for f in t.failures)
    L += ["## Run", "",
          f"- workdir: `{wd}`",
          f"- iterations: {t.iterations_done} (target {a.iterations}); nodes admitted: {len(t.nodes) - 1}; "
          f"failed attempts: {len(t.failures)}",
          f"- actions: " + ", ".join(f"{k} {v}" for k, v in act.items()) +
          f"; outcomes: " + ", ".join(f"{k} {v}" for k, v in stat.items()),
          f"- LLM revisions: {t.llm_calls} claude sessions (incl. repairs), {repairs} repair rounds, "
          f"spend ${t.llm_cost:.2f} (budget ${a.max_llm_usd:.0f}, model {a.model})",
          f"- search suite `{a.suite}`, final suite `{a.final_suite}`, C={a.c}, vloss={a.vloss}, "
          f"alpha {a.alpha0}->{a.alpha1} over {a.anneal_steps} root visits, workers {a.workers}, "
          f"knob ratio {a.knob_ratio}",
          f"- production vs itself measured on `{a.suite}`: {root.eval.get('measured_self_score', 1.0):.4f}x"
          + (f"; on `{a.final_suite}`: {root_full:.4f}x" if root_full else ""), ""]

    # ---- verdict
    q = best.score
    L += ["## Result", ""]
    if best.id == root.id:
        L += ["No admitted candidate beat production."]
    else:
        txt = f"Best node **{best.id}**: {q:.4f}x on `{a.suite}`"
        if full_best is not None:
            txt += f", **{full_best:.4f}x on `{a.final_suite}`** (production re-run: " \
                   f"{root_full:.4f}x)" if root_full else f", {full_best:.4f}x on `{a.final_suite}`"
        L.append(txt + ".")
        gain = (full_best or q) / (root_full if (full_best and root_full) else 1.0) - 1
        L.append("")
        L.append(f"Verdict: {'IMPROVED' if gain >= 0.02 else 'no significant improvement (< 2% is noise)'} "
                 f"({gain * 100:+.1f}%).")
    L.append("")

    # ---- lineage
    L += ["## Lineage of the best node", "",
          "| step | node | action | change | quick score | vs parent |", "|---|---|---|---|---|---|"]
    prev = None
    for i, n in enumerate(t.lineage(best.id)):
        d = "" if prev is None else f"{(n.score / prev.score - 1) * 100:+.1f}%"
        ch = "production kernel" if n.action == "root" else (n.notes.replace("\n", " ").replace("|", "/")[:260] +
                                                             f" _[{n.auto_change.replace('|', '/')[:200]}]_")
        L.append(f"| {i} | {n.id} | {n.action} | {ch} | {n.score:.4f} | {d} |")
        prev = n
    L.append("")

    # ---- top-k table
    L += ["## Top candidates (search suite, then full re-evaluation)", "",
          "| node | action | quick | full | parent | note |", "|---|---|---|---|---|---|"]
    for n in sorted((x for x in t.nodes.values() if x.id != t.root), key=lambda x: -x.score)[:max(8, a.top_k)]:
        fe = n.full_eval
        full = f"{fe['score']:.4f}" if fe and fe.get("status") == "ok" else (fe["status"] if fe else "-")
        L.append(f"| {n.id} | {n.action} | {n.score:.4f} | {full} | {n.parent} | "
                 f"{n.notes.replace(chr(10), ' ').replace('|', '/')[:110]} |")
    L.append("")

    # ---- per-op table
    L += ["## Per-op speedup of the best node vs production", "",
          "| op | quick | full |", "|---|---|---|"]
    qo = best.eval.get("per_op", {})
    fo = (best.full_eval or {}).get("per_op", {})
    for op in sorted(set(qo) | set(fo)):
        L.append(f"| {op} | {qo.get(op, float('nan')):.3f}x | "
                 f"{(str(round(fo[op], 3)) + 'x') if op in fo else '-'} |")
    if best.full_eval:
        L += ["", "Per case (full):", ""]
        for k, v in best.full_eval.get("per_case", {}).items():
            L.append(f"- {k}{' (probe, not scored)' if v.get('probe') else ''}: {v['speedup']:.3f}x")
    L.append("")

    # ---- diff vs production
    bdir = t.node_dir(best.id)
    _, bcfg = candidate.load(bdir)
    L += ["## Configs and diff vs production", "", "```", "configs: " + config_diff(candidate.production_configs()
                                                                                    and candidate.normalize(
        candidate.production_configs()), bcfg), "```", ""]
    diff = candidate.diff_vs_production(bdir)
    L += ["Header diff:", "", "```diff", diff[:12000] if diff else "(header identical to production)", "```", ""]
    best_out = wd / "best"
    if best_out.exists():
        shutil.rmtree(best_out)
    shutil.copytree(bdir, best_out, ignore=shutil.ignore_patterns("evals", "profiles"))
    L += [f"The best candidate directory is copied to `{best_out}` (header + configs.json).", ""]

    # ---- failures
    if t.failures:
        L += ["## Failed attempts", "", "| attempt | parent | action | status | repairs | note / error |",
              "|---|---|---|---|---|---|"]
        for f in t.failures:
            L.append(f"| {f['id']} | {f['parent']} | {f['action']} | {f['status']} | {f.get('repairs', 0)} | "
                     f"{(f.get('notes') or '')[:70].replace('|', '/')} / "
                     f"{(f.get('error') or '')[:90].replace(chr(10), ' ').replace('|', '/')} |")
        L.append("")

    # ---- tree
    L += ["## Tree summary", "", "(N = visits, score = own search-suite score, best-subtree = best score in the "
          "subtree; the selection value is L = 1/score)", "", "```", t.tree_text().rstrip(), "```", "",
          f"Widening at the end: alpha = {t.alpha(a.alpha0, a.alpha1, a.anneal_steps):.3f} "
          f"(root visits {root.visits}).", ""]
    (wd / "report.md").write_text("\n".join(L))
    return wd / "report.md"


def _iter_log(wd):
    p = Path(wd) / "iterations.jsonl"
    if not p.exists():
        return []
    out = []
    for l in p.read_text().splitlines():
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out
