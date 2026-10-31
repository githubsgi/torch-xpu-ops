"""Analysis stage (port of agents/analyzer_agent.py) and planning stage (planner.py)."""

import json
import re
from pathlib import Path

from . import prompts
from .models import (DEFAULT_STAGE_ORDER, HARD_DEPENDENCIES, ISSUE_SCHEMA, OPTIMIZATION_STAGES, Stage, coerce_issue,
                     stages_needed)

READ_ONLY = ("Read", "Glob", "Grep", "Bash(ls:*)", "Bash(cat:*)", "Bash(head:*)", "Bash(grep:*)", "Bash(sed -n:*)")


def analyze(llm, ws, baseline_text, perf_table, digests, log=print):
    """One read-only LLM session over the kernel + digests -> list[DetectedIssue] (JSON-schema constrained)."""
    prompt = prompts.analysis_prompt(baseline_text, perf_table, digests,
                                     "Cite sections of the notes only if you read them in refs/notes_excerpt.txt.")
    r = llm.call("analysis", prompt, ws, tools=READ_ONLY, system=prompts.SYSTEM, json_schema=ISSUE_SCHEMA,
                 timeout_s=1500)
    data = r.get("structured")
    if not data:
        log(f"  analysis: no structured output (ok={r['ok']} error={r['error'][:200]}); retrying once without tools")
        r2 = llm.call("analysis-retry", prompt + "\n\nReturn ONLY the JSON object.", ws, tools=(),
                      system=prompts.SYSTEM, json_schema=ISSUE_SCHEMA, timeout_s=900)
        data = r2.get("structured")
    if not data:
        return [], "(analysis failed)", r
    issues = [i for i in (coerce_issue(x) for x in data.get("issues", [])) if i]
    # same issue_type twice: keep the one with the higher severity
    best = {}
    for i in issues:
        k = (i.issue_type, tuple(i.kinds))
        if k not in best or i.severity > best[k].severity:
            best[k] = i
    issues = sorted(best.values(), key=lambda i: -i.severity)
    return issues, data.get("summary", ""), r


def issue_summary(issues):
    return "\n".join(f"  [{i.severity}/5] {i.issue_type} -> {i.stage.value}: {i.evidence[:110]}" for i in issues) \
        or "  (no issues)"


def default_filtered(needed):
    return [s for s in DEFAULT_STAGE_ORDER if s in needed]


def enforce_dependencies(stages):
    """Move `before` in front of `after` when both are present (original: _enforce_dependencies)."""
    stages = list(stages)
    for _ in range(len(HARD_DEPENDENCIES)):  # a few passes settle chains
        changed = False
        for before, after in HARD_DEPENDENCIES:
            if before in stages and after in stages and stages.index(before) > stages.index(after):
                stages.remove(before)
                stages.insert(stages.index(after), before)
                changed = True
        if not changed:
            break
    return stages


def parse_and_validate(raw, needed, fallback):
    valid = {s.value: s for s in OPTIMIZATION_STAGES}
    out, seen = [], set()
    for item in raw or []:
        s = valid.get(str(item).strip().lower())
        if s is None or s not in needed or s in seen:
            continue
        out.append(s)
        seen.add(s)
    for s in fallback:  # stages the LLM omitted are appended in default order
        if s not in seen:
            out.append(s)
            seen.add(s)
    return enforce_dependencies(out) or fallback


def plan(llm, ws, needed, issues, summary, log=print):
    """LLM orders the stages; dependencies are enforced afterwards. Falls back to the default order."""
    if not needed:
        return [], "no stages needed"
    fallback = default_filtered(needed)
    if len(needed) == 1:
        return list(needed), "single stage"
    prompt = prompts.planning_prompt(needed, issue_summary(issues),
                                     f"{summary[:900]} | kinds m_rowB/m_colB (fwd, dX) and k_grouped (dW); regimes "
                                     f"tp4ep4 (~190 rows/expert) and ep8 (~6k rows/expert).")
    try:
        r = llm.call("planning", prompt, ws, tools=(), system=prompts.SYSTEM, json_schema=prompts.PLAN_SCHEMA,
                     timeout_s=600, budget_usd=3)
        data = r.get("structured") or {}
        ordered = parse_and_validate(data.get("ordered_stages"), needed, fallback)
        return ordered, data.get("rationale", "")
    except Exception as e:  # noqa: BLE001
        log(f"  planner failed ({e}); default order")
        return enforce_dependencies(fallback), f"planner failed: {e}"
