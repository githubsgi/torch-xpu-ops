#!/usr/bin/env python3
"""PostToolUse hook: append one line per tool use to .humanize/tool_log.jsonl (evidence that the
skills/CLIs were actually used). stdlib only, python >= 3.6. Never fails the session."""
import json
import os
import sys
import time

try:
    ev = json.load(sys.stdin)
    root = os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    tool = ev.get("tool_name", "")
    inp = ev.get("tool_input") or {}
    if tool == "Skill":
        detail = str(inp.get("skill") or inp.get("name") or inp)[:120]
    elif tool == "Bash":
        detail = str(inp.get("command", ""))[:240]
    elif tool in ("Read", "Edit", "Write"):
        detail = str(inp.get("file_path", ""))[:200]
    else:
        detail = json.dumps(inp)[:160]
    try:
        rnd = open(os.path.join(root, ".kda_round")).read().strip()
    except OSError:
        rnd = ""
    os.makedirs(os.path.join(root, ".humanize"), exist_ok=True)
    with open(os.path.join(root, ".humanize", "tool_log.jsonl"), "a") as f:
        f.write(json.dumps({"t": time.strftime("%F %T"), "round": rnd, "tool": tool, "detail": detail}) + "\n")
except Exception:  # noqa: BLE001
    pass
