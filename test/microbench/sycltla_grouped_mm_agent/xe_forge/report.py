"""report.md: stage-by-stage table, accepted diffs, scores, LLM cost."""

import difflib
import json
from pathlib import Path

from .workspace import CONFIGS, HEADER


def _diff(a_dir, b_dir, max_lines=200):
    out = []
    for f in (HEADER, CONFIGS):
        a = (Path(a_dir) / f).read_text().splitlines(keepends=True)
        b = (Path(b_dir) / f).read_text().splitlines(keepends=True)
        out += list(difflib.unified_diff(a, b, f"a/{f}", f"b/{f}", n=3))
    if len(out) > max_lines:
        out = out[:max_lines] + [f"... ({len(out) - max_lines} more lines)\n"]
    return "".join(out) or "(no difference)\n"


def write_report(p):
    st, tree, llm = p.state, p.tree, p.llm
    L = [f"# Xe-Forge run {st['run_id']}", ""]
    L.append(f"- workdir: `{p.wd}`  |  phase: {st.get('phase')}  |  wall time: {st.get('seconds_total', 0):.0f} s")
    a = p.args
    L.append(f"- model `{a.model}`, suite `{a.suite}` (stage gate), final suite `{a.final_suite}`, best-of "
             f"{a.max_iterations} x CoVeR {a.max_attempts}, keep gate {a.min_improvement}x, LLM cap ${a.max_llm_usd}")
    L.append(f"- LLM cost: **${llm.spent:.2f}** total; by purpose: " +
             ", ".join(f"{k.split('-')[0]} ${v:.2f}" for k, v in _group(llm.by_purpose()).items()))
    t0 = tree.get("t0")
    L.append(f"- baseline t0 (production vs production, {t0['suite']}): {t0['score']:.3f}x (this is the noise level)")
    fin = st.get("final")
    if fin:
        L.append(f"- **final: trial {fin['trial']} on suite `{fin['suite']}`: status {fin['status']}, score "
                 f"{(fin.get('score') or 0):.3f}x vs production**")
    L.append("")
    # analysis
    af = p.wd / "analysis.json"
    if af.exists():
        an = json.loads(af.read_text())
        L += ["## Analysis", "", an.get("summary", ""), "",
              "| severity | issue | stage | kinds | evidence (measured) | est. (unmeasured) |", "|---|---|---|---|---|---|"]
        from .models import ISSUE_STAGE
        for i in an["issues"]:
            L.append(f"| {i['severity']} | {i['issue_type']} | {ISSUE_STAGE[i['issue_type']].value} | "
                     f"{','.join(i['kinds'])} | {i['evidence'][:160].replace('|', '/')} | {i['estimated_speedup']} |")
        L.append("")
    pf = p.wd / "plan.json"
    if pf.exists():
        pl = json.loads(pf.read_text())
        L += ["## Plan", "", f"Order after dependency enforcement: `{' -> '.join(pl['order'])}`", "",
              pl.get("rationale", ""), ""]
    # stages
    L += ["## Stages", "", "| stage | result | start | trial | score (start -> end) | iterations | evaluated attempts | cost | time |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, s in st["stages"].items():
        sc = f"{s.get('start_score') or 0:.3f} -> {(s.get('score') or s.get('start_score') or 0):.3f}"
        L.append(f"| {name} | {s.get('status')} | {s.get('start_trial')} | {s.get('trial') or '-'} | {sc} | "
                 f"{s.get('iterations', 0)} | {s.get('attempts', 0)} | ${s.get('cost_usd', 0):.2f} | {s.get('seconds', 0):.0f}s |")
    L.append("")
    for name, s in st["stages"].items():
        L.append(f"### {name}: {s.get('status')}")
        L.append(s.get("summary", ""))
        for h in s.get("history", []):
            L.append(f"- {h}")
        L.append("")
    # accepted diffs
    acc = [(n, s) for n, s in st["stages"].items() if s.get("status") == "accepted"]
    if acc:
        L += ["## Accepted diffs", ""]
        for name, s in acc:
            L += [f"### {name}: {s['start_trial']} -> {s['trial']}", "", "```diff", _diff(tree.path_of(s["start_trial"]), tree.path_of(s["trial"])), "```", ""]
    # final
    if fin and fin.get("per_case"):
        L += ["## Final evaluation", "", f"Suite `{fin['suite']}`, trial {fin['trial']} (production = 1.000x):", "",
              "| case | speedup |", "|---|---|"]
        for k, v in fin["per_case"].items():
            L.append(f"| {k} | {v['speedup']:.3f}x{' (probe, not scored)' if '_rows' in k else ''} |")
        L += ["", "Per op (geomean): " + ", ".join(f"{k} {v:.3f}x" for k, v in (fin.get("per_op") or {}).items()), ""]
    L += ["## Trial tree", "", "```", tree.render_tree(), "```", "",
          f"Trial state: `{tree.path}`; lessons: `{p.wd / 'lessons.md'}`; LLM log: `{p.wd / 'llm_log.jsonl'}`.", ""]
    (p.wd / "report.md").write_text("\n".join(L))


def _group(by_purpose):
    out = {}
    for k, v in by_purpose.items():
        out[k.split("-")[0]] = out.get(k.split("-")[0], 0.0) + v
    return out
