"""Headless Claude Code as the LLM of every harness (`claude -p`).

Runs on the login node (the compute nodes have no internet). The agent gets
file tools in its working directory plus `ggemm-eval` on PATH, so it can edit
a candidate and evaluate / profile it itself; harnesses that only want text
(e.g. MCTS proposals) pass `tools=()` and a `json_schema`.

GGEMM_LLM_MODEL   default "opus"        GGEMM_LLM_EFFORT  default unset
GGEMM_CLAUDE      default "claude"      GGEMM_LLM_BUDGET_USD  per call, default 20
"""

import json
import os
import subprocess
import time
from pathlib import Path

from . import config

EDIT_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep",
              "Bash(ggemm-eval:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(diff:*)", "Bash(grep:*)",
              "Bash(head:*)", "Bash(tail:*)", "Bash(cp:*)", "Bash(mkdir:*)", "Bash(wc:*)", "Bash(sed -n:*)")
READ_TOOLS = ("Read", "Glob", "Grep", "Bash(ggemm-eval:*)", "Bash(ls:*)", "Bash(cat:*)", "Bash(diff:*)")


def run_claude(prompt, cwd, *, tools=EDIT_TOOLS, system=None, add_dirs=(), model=None, effort=None,
               budget_usd=None, timeout_s=5400, json_schema=None, log_path=None, extra_args=()):
    """One headless Claude Code session. Returns a dict:
    ok, text (final message), structured (parsed JSON if json_schema), cost_usd,
    session_id, seconds, error."""
    model = model or os.environ.get("GGEMM_LLM_MODEL", "opus")
    effort = effort or os.environ.get("GGEMM_LLM_EFFORT")
    budget = budget_usd or float(os.environ.get("GGEMM_LLM_BUDGET_USD", "20"))
    cmd = [os.environ.get("GGEMM_CLAUDE", "claude"), "-p", prompt, "--output-format", "json", "--model", model,
           "--max-budget-usd", str(budget), "--permission-mode", "acceptEdits", "--no-session-persistence"]
    if tools:
        cmd += ["--allowedTools", ",".join(tools)]
    else:
        cmd += ["--disallowedTools", "Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch,Agent"]
    if system:
        cmd += ["--append-system-prompt", system]
    for d in add_dirs:
        cmd += ["--add-dir", str(d)]
    if effort:
        cmd += ["--effort", effort]
    if json_schema:
        cmd += ["--json-schema", json.dumps(json_schema)]
    cmd += list(extra_args)
    env = dict(os.environ)
    env["PATH"] = f"{config.COMMON_DIR / 'bin'}:{env.get('PATH', '')}"
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True, timeout=timeout_s, env=env)
        out, err, rc = p.stdout, p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
        err, rc = f"claude timed out after {timeout_s} s", -9
    res = {"ok": False, "text": "", "structured": None, "cost_usd": 0.0, "session_id": None,
           "seconds": round(time.time() - t0, 1), "error": ""}
    try:
        j = json.loads(out)
        res.update(ok=not j.get("is_error", False) and rc == 0, text=j.get("result") or "",
                   cost_usd=j.get("total_cost_usd", 0.0), session_id=j.get("session_id"),
                   structured=j.get("structured_output"))
        if json_schema and res["structured"] is None and res["text"]:
            try:
                res["structured"] = json.loads(_strip_fence(res["text"]))
            except json.JSONDecodeError:
                pass
        if not res["ok"]:
            res["error"] = j.get("subtype") or err[-2000:]
    except json.JSONDecodeError:
        res["error"] = f"rc={rc}; {err[-2000:]}; {out[-1000:]}"
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a") as f:
            f.write(json.dumps({"time": time.strftime("%F %T"), "cwd": str(cwd), "model": model,
                                "prompt": prompt, **{k: v for k, v in res.items()}}) + "\n")
    return res


def _strip_fence(t):
    t = t.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else ""
        t = t.rsplit("```", 1)[0]
    return t.strip()
