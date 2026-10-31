"""Path bootstrap and the LLM wrapper with a global spend cap."""

import json
import os
import sys
import threading
import time
from pathlib import Path

AGENT_DIR = Path(__file__).resolve().parent.parent
COMMON = AGENT_DIR / "common"
if str(COMMON) not in sys.path:
    sys.path.insert(0, str(COMMON))
os.environ["PATH"] = f"{COMMON / 'bin'}:{os.environ.get('PATH', '')}"

from ggemm_agent import candidate, config, evaluate, llm, profile  # noqa: E402,F401


class BudgetExceeded(RuntimeError):
    pass


class LLM:
    """run_claude with a run-wide cap (`--max-llm-usd`), a cost ledger and a call log."""

    MIN_CALL_USD = 0.75

    def __init__(self, workdir, model, max_usd, per_call_usd, effort=None):
        self.workdir = Path(workdir)
        self.model, self.max_usd, self.per_call_usd, self.effort = model, float(max_usd), float(per_call_usd), effort
        self.lock = threading.Lock()
        self.ledger_path = self.workdir / "cost.json"
        self.entries = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else []

    @property
    def spent(self):
        return sum(e["usd"] for e in self.entries)

    def remaining(self):
        return self.max_usd - self.spent

    def by_purpose(self):
        out = {}
        for e in self.entries:
            out[e["purpose"]] = out.get(e["purpose"], 0.0) + e["usd"]
        return out

    def call(self, purpose, prompt, cwd, *, tools, system=None, json_schema=None, add_dirs=(), timeout_s=1800,
             budget_usd=None):
        if self.remaining() < self.MIN_CALL_USD:
            raise BudgetExceeded(f"LLM spend cap ${self.max_usd:.2f} reached (spent ${self.spent:.2f})")
        budget = min(budget_usd or self.per_call_usd, self.remaining())
        res = llm.run_claude(prompt, cwd, tools=tools, system=system, add_dirs=add_dirs, model=self.model,
                             effort=self.effort, budget_usd=budget, timeout_s=timeout_s, json_schema=json_schema,
                             log_path=self.workdir / "llm_log.jsonl")
        with self.lock:
            self.entries.append({"time": time.strftime("%F %T"), "purpose": purpose, "usd": res["cost_usd"],
                                 "ok": res["ok"], "seconds": res["seconds"]})
            self.ledger_path.write_text(json.dumps(self.entries, indent=1))
        return res
