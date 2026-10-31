#!/usr/bin/env python3
"""KDA / Humanize-style RLCR round driver for the SYCL-TLA grouped GEMM.

    cd test/microbench/sycltla_grouped_mm_agent
    ~/miniforge3/bin/python3 -m kda.rlcr --workdir $GGEMM_AGENT_RUNS/kda/run1 --phase 2 \
        --max-rounds 8 --model opus --max-llm-usd 60

Flow (Humanize RLCR = Ralph-Loop with Codex Review; the reviewer here is a read-only `claude -p`):

    workspace from workspace_template/  (CLAUDE.md, skills kernel-wiki + xpu-profile, prompts, records)
      -> draft   worker session writes docs/draft.md (baseline, profile, ranked directions)
      -> gen-plan session turns it into docs/plan.md (goal, acceptance criteria, tasks)
      -> .humanize/rlcr/<ts>/  plan.md state.md goal-tracker.md
      -> rounds N = 0, 1, ...:
           round-N-prompt.md   -> worker (edit tools + ggemm-eval + Skill)   -> round-N-summary.md
           driver: git commit, restore the frozen immutable goal-tracker section, independent
                   `ggemm-eval run` of cand/ (round-N-eval.txt), promotion check on the full suite
           reviewer prompt     -> round-N-review-result.md  (last line COMPLETE | STOP | CONTINUE)
           review feedback becomes round N+1's prompt; every 5th round is a full goal-alignment audit
      -> stop on COMPLETE / STOP / max rounds / LLM $ cap / stall counter / repeated worker failure
      -> finalize: full-suite eval + promotion, code-review phase (on COMPLETE), FINAL_REPORT.md

Resumable: re-running the same command continues from the files present in the loop directory.
`GGEMM_KDA_FAKE_LLM=1` replaces every LLM call with a canned stub (control-flow tests, no cost).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
AGENT_DIR = HERE.parent
sys.path.insert(0, str(AGENT_DIR / "common"))

from ggemm_agent import candidate, config, evaluate, llm  # noqa: E402

TEMPLATE = HERE / "workspace_template"
PROMPTS = HERE / "prompt_templates"
NOTES_DIR = Path("/home/subratag/datavol/logs/torchtitan/outputs_deepseek_v3_16b")

WORKER_TOOLS = llm.EDIT_TOOLS + (
    "Skill", "Bash(kw:*)", "Bash(xpu-report:*)", "Bash(kda-record:*)", "Bash(tee:*)", "Bash(find:*)",
    "Bash(git diff:*)", "Bash(git log:*)", "Bash(git status:*)", "Bash(date:*)", "Bash(sort:*)", "Bash(cmp:*)")
REVIEW_TOOLS = (
    "Read", "Glob", "Grep", "Bash(cat:*)", "Bash(ls:*)", "Bash(diff:*)", "Bash(head:*)", "Bash(tail:*)",
    "Bash(wc:*)", "Bash(git diff:*)", "Bash(git log:*)", "Bash(git show:*)", "Bash(git status:*)",
    "Bash(kw:*)", "Bash(kda-record show:*)", "Bash(ggemm-eval diff:*)", "Bash(xpu-report check:*)")
PLAN_TOOLS = ("Read", "Write", "Edit", "Glob", "Grep", "Bash(kw:*)", "Bash(ls:*)", "Bash(cat:*)")

PROMOTE_SCORE = 1.02
PROMOTE_MIN_OP = 0.97
PROMOTE_TRIGGER = 1.01  # quick score that triggers the full-suite promotion check
TERMINAL = ("complete", "stopped_stall", "stopped_max_rounds", "stopped_cost", "stopped_review",
            "stopped_failures")


# ------------------------------------------------------------------ utils ---
def now():
    return time.strftime("%F %T")


def render(text, **kw):
    for k, v in kw.items():
        text = text.replace("{{%s}}" % k, str(v))
    return text


def read(p, default=""):
    try:
        return Path(p).read_text()
    except OSError:
        return default


def sh(cmd, cwd, check=False):
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    if check and p.returncode:
        raise RuntimeError("%s: %s" % (cmd, p.stderr))
    return (p.stdout + p.stderr).strip()


def sha(paths):
    h = hashlib.sha256()
    for p in paths:
        h.update(read(p).encode())
    return h.hexdigest()[:16]


def log(msg):
    print("[kda %s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


# ------------------------------------------------------------------ state ---
class State:
    """state.md: `key: value` front matter, like Humanize's state file."""
    DEFAULTS = {"current_round": 0, "max_iterations": 42, "full_review_round": 5, "phase": 2, "status": "running",
                "plan_file": "", "base_commit": "", "started_at": "", "stall_count": 0, "max_stall": 3,
                "worker_failures": 0, "last_counted_round": -1, "best_score": 1.0,
                "best_id": "production", "final_done": 0, "stop_reason": "", "cost_cap_usd": 0.0}

    def __init__(self, path):
        self.path = Path(path)
        self.d = dict(self.DEFAULTS)
        if self.path.exists():
            for line in self.path.read_text().splitlines():
                m = re.match(r"^([a-z_]+):\s*(.*)$", line)
                if m and m[1] in self.d:
                    v = m[2].strip()
                    self.d[m[1]] = type(self.DEFAULTS[m[1]])(v) if not isinstance(self.DEFAULTS[m[1]], str) else v

    def __getitem__(self, k):
        return self.d[k]

    def __setitem__(self, k, v):
        self.d[k] = v
        self.save()

    def save(self):
        self.path.write_text("---\n" + "".join("%s: %s\n" % (k, v) for k, v in self.d.items()) + "---\n"
                             "RLCR loop state (do not edit while the loop runs).\n")


# --------------------------------------------------------------- the loop ---
class Loop:
    def __init__(self, a):
        self.a = a
        self.ws = Path(a.workdir).resolve()
        self.model = a.model or os.environ.get("GGEMM_LLM_MODEL") or "opus"
        self.rmodel = a.reviewer_model or self.model
        self.fake = os.environ.get("GGEMM_KDA_FAKE_LLM") == "1"
        self.cost_file = self.ws / ".humanize" / "costs.jsonl"
        self.calls_file = self.ws / ".humanize" / "llm_calls.jsonl"
        self.loop_dir = None
        self.st = None
        self.complete_guard = ""
        # the claude sessions inherit this process' environment: skills' CLIs on PATH, long Bash timeouts
        os.environ["PATH"] = "%s:%s" % (self.ws / "bin", os.environ.get("PATH", ""))
        os.environ.setdefault("BASH_DEFAULT_TIMEOUT_MS", "600000")
        os.environ.setdefault("BASH_MAX_TIMEOUT_MS", "900000")
        os.environ["KDA_WORKSPACE"] = str(self.ws)

    # ---- workspace
    def setup_workspace(self):
        ws = self.ws
        if not (ws / "CLAUDE.md").exists():
            log("creating workspace %s" % ws)
            ws.mkdir(parents=True, exist_ok=True)
            for item in TEMPLATE.iterdir():
                dst = ws / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, symlinks=True)
                else:
                    shutil.copy2(item, dst)
            for d in ("docs", "profile", "candidates", ".humanize"):
                (ws / d).mkdir(exist_ok=True)
            (ws / "reference").mkdir(exist_ok=True)
            shutil.copy(config.PROD_HEADER, ws / "reference" / "grouped_mm_kernel_impl.production.h")
            shutil.copy(config.SRC_DIR / "grouped_mm_kernels.h", ws / "reference" / "grouped_mm_kernels.h")
            shutil.copy(config.EVAL_DIR / "cand_cfg.cpp", ws / "reference" / "cand_cfg.cpp")
            candidate.init_dir(ws / "production")
            candidate.init_dir(ws / "cand")
            candidate.init_dir(ws / "best", src=ws / "cand")
            seed = HERE / "bitlesson_seed.md"
            shutil.copy(seed, ws / ".humanize" / "bitlesson.md")
            (ws / ".kda_phase").write_text(str(self.a.phase))
            (ws / ".kda_round").write_text("")
            sh(["git", "init", "-q"], ws)
            self.commit("workspace created (phase %d)" % self.a.phase)
        else:
            self.maybe_advance_phase()
        # prompts for the chosen phase
        guidance = self.a.guidance or ""
        if self.a.guidance_file:
            guidance += "\n" + read(self.a.guidance_file)
        common = render(read(ws / "prompts" / "common.md"),
                        GUIDANCE=("\n## Human guidance\n" + guidance.strip() + "\n") if guidance.strip() else "")
        phase_txt = render(read(ws / "prompts" / ("phase%d.md" % self.a.phase)), COMMON=common,
                           TARGET_SPEEDUP=self.a.target_speedup)
        (ws / "docs" / "task_prompt.md").write_text(phase_txt)
        self.phase_prompt = phase_txt

    def maybe_advance_phase(self):
        """Re-running with a different --phase on a finished workspace archives the old plan, restarts from best/."""
        ws = self.ws
        old = read(ws / ".kda_phase", "0").strip()
        loops = sorted((ws / ".humanize" / "rlcr").glob("*")) if (ws / ".humanize" / "rlcr").exists() else []
        if loops and State(loops[-1] / "state.md")["status"] in TERMINAL and str(self.a.phase) != old:
            log("phase %s -> %d: archiving previous draft/plan, cand := best" % (old, self.a.phase))
            for f in ("draft", "plan"):
                if (ws / "docs" / (f + ".md")).exists():
                    (ws / "docs" / (f + ".md")).rename(ws / "docs" / ("%s.phase%s.md" % (f, old)))
            for f in ("grouped_mm_kernel_impl.h", "configs.json"):
                shutil.copy(ws / "best" / f, ws / "cand" / f)
            (ws / ".kda_phase").write_text(str(self.a.phase))
            self.a.new_loop = True

    def commit(self, msg):
        sh(["git", "add", "-A"], self.ws)
        sh(["git", "-c", "user.name=kda", "-c", "user.email=kda@localhost", "commit", "-q", "-m", msg,
            "--allow-empty"], self.ws)
        return sh(["git", "rev-parse", "HEAD"], self.ws)

    # ---- LLM
    def spent(self):
        t = 0.0
        for l in read(self.cost_file).splitlines():
            try:
                t += json.loads(l)["cost_usd"]
            except (ValueError, KeyError):
                pass
        return t

    def remaining(self):
        return self.a.max_llm_usd - self.spent()

    def call(self, role, prompt, *, tools, budget, rnd=None, model=None, system=None, timeout=5400):
        rem = self.remaining()
        if rem < 0.25:
            raise CostCap("LLM budget exhausted: spent $%.2f of $%.2f" % (self.spent(), self.a.max_llm_usd))
        budget = max(0.3, min(budget, rem))
        (self.ws / ".kda_round").write_text("" if rnd is None else str(rnd))
        log("%s call (model %s, cap $%.2f, spent $%.2f)" % (role, model or self.model, budget, self.spent()))
        if self.fake:
            res = fake_llm(self, role, prompt, rnd)
        else:
            res = llm.run_claude(prompt, self.ws, tools=tools, system=system, model=model or self.model,
                                 budget_usd=budget, timeout_s=timeout, log_path=self.calls_file,
                                 add_dirs=[d for d in (config.SYCL_TLA, NOTES_DIR) if Path(d).exists()])
        with open(self.cost_file, "a") as f:
            f.write(json.dumps({"t": now(), "role": role, "round": rnd, "cost_usd": res.get("cost_usd", 0.0),
                                "seconds": res.get("seconds"), "ok": res.get("ok"),
                                "session": res.get("session_id"), "error": (res.get("error") or "")[:300]}) + "\n")
        log("%s done: ok=%s cost $%.2f in %ss%s" % (role, res.get("ok"), res.get("cost_usd", 0), res.get("seconds"),
                                                   "" if res.get("ok") else " ERROR " + (res.get("error") or "")[:200]))
        return res

    # ---- stages before the loop
    def draft_stage(self):
        ws = self.ws
        if (ws / "docs" / "draft.md").exists():
            return
        prompt = render(read(PROMPTS / "draft.md"), PHASE_PROMPT=self.phase_prompt)
        res = self.call("draft", prompt, tools=WORKER_TOOLS, budget=self.a.draft_budget, rnd=None)
        if not (ws / "docs" / "draft.md").exists():
            if res.get("text"):
                (ws / "docs" / "draft.md").write_text("# Draft (driver fallback: worker did not write the file)\n\n"
                                                      + res["text"])
            else:
                raise StopLoop("failures", "draft stage produced nothing: %s" % res.get("error"))
        self.commit("draft")

    def plan_stage(self):
        ws = self.ws
        if (ws / "docs" / "plan.md").exists():
            return
        prompt = render(read(PROMPTS / "gen_plan.md"), TARGET_SPEEDUP=self.a.target_speedup)
        res = self.call("gen-plan", prompt, tools=PLAN_TOOLS, budget=self.a.plan_budget, rnd=None,
                        model=self.a.plan_model or self.model)
        if not (ws / "docs" / "plan.md").exists():
            raise StopLoop("failures", "gen-plan produced no docs/plan.md: %s" % res.get("error"))
        self.commit("plan")

    def start_loop(self):
        ws = self.ws
        base = ws / ".humanize" / "rlcr"
        base.mkdir(parents=True, exist_ok=True)
        existing = sorted(base.glob("*"))
        if existing and not self.a.new_loop:
            self.loop_dir = existing[-1]
            self.st = State(self.loop_dir / "state.md")
            if self.st["status"] in ("stopped_max_rounds", "stopped_cost") and self.st["current_round"] < self.a.max_rounds \
                    and self.remaining() > 1.0:
                log("extending a loop that stopped (%s)" % self.st["status"])
                self.st.d.update(status="running", stop_reason="", final_done=0, max_iterations=self.a.max_rounds,
                                 cost_cap_usd=self.a.max_llm_usd)
                self.st.save()
            log("resuming loop %s at round %d (status %s)" % (self.loop_dir.name, self.st["current_round"], self.st["status"]))
            return
        self.loop_dir = base / time.strftime("%Y-%m-%d_%H-%M-%S")
        self.loop_dir.mkdir()
        plan = read(ws / "docs" / "plan.md")
        (self.loop_dir / "plan.md").write_text(plan)
        goal = self.extract(plan, "Goal Description") or "See plan.md."
        acs = self.extract(plan, "Acceptance Criteria") or "See plan.md."
        (self.loop_dir / "goal-tracker.md").write_text(render(read(PROMPTS / "goal_tracker.md"), GOAL=goal, ACS=acs))
        self.st = State(self.loop_dir / "state.md")
        self.st.d.update(phase=self.a.phase, max_iterations=self.a.max_rounds, full_review_round=self.a.full_review_round,
                         plan_file=str((self.loop_dir / "plan.md").relative_to(ws)), started_at=now(),
                         base_commit=self.commit("loop start"), max_stall=self.a.max_stall,
                         cost_cap_usd=self.a.max_llm_usd)
        self.st.save()
        log("started loop %s" % self.loop_dir.name)

    @staticmethod
    def extract(plan, title):
        m = re.search(r"^##\s+%s\s*\n(.*?)(?=^##\s|\Z)" % re.escape(title), plan, re.M | re.S)
        return m.group(1).strip() if m else ""

    # ---- rounds
    def rel(self, p):
        return str(Path(p).relative_to(self.ws))

    def build_prompt(self, n):
        ws, d = self.ws, self.loop_dir
        header = "Your work is not finished. Read and execute the below." if n else \
            "You are starting round 0 of an RLCR loop. Read and execute the below."
        phase_block = ("## Phase task prompt\n" + self.phase_prompt) if n == 0 else \
            "(Phase task prompt: `docs/task_prompt.md`; contract: `CLAUDE.md`.)"
        common = dict(PLAN_FILE=self.st["plan_file"], GOAL_TRACKER_FILE=self.rel(d / "goal-tracker.md"), ROUND=n,
                      CONTRACT_FILE=self.rel(d / ("round-%d-contract.md" % n)),
                      SUMMARY_FILE=self.rel(d / ("round-%d-summary.md" % n)), WORKER_BUDGET="%.0f" % self.a.worker_budget)
        if n == 0:
            note = ""
            if (ws / "best").exists() and sha([ws / "best" / "grouped_mm_kernel_impl.h"]) != \
                    sha([config.PROD_HEADER]):
                note = "Note: `best/` already differs from production (earlier phase); cand/ starts from best/."
            body = render(read(PROMPTS / "round0_body.md"), PLAN_TEXT=read(d / "plan.md"), ROUND0_NOTE=note, **common)
        else:
            prev = n - 1
            audit = (n + 1) % self.st["full_review_round"] == 0
            body = render(read(PROMPTS / "next_round_body.md"), PREV=prev, REVIEW=read(d / ("round-%d-review-result.md" % prev)),
                          DRIVER_NOTES=self.driver_notes(prev), LOOP_DIR=self.rel(d),
                          ALIGNMENT_NOTE=("\nThis is a FULL ALIGNMENT round: re-read the whole plan and the tracker, "
                                          "reconcile every table in the mutable section, and update the Direction Ledger."
                                          if audit else ""))
        return render(read(PROMPTS / "round.md"), HEADER=header, PHASE_PROMPT_BLOCK=phase_block, BODY=body, **common)

    def driver_notes(self, prev):
        notes = ["## Driver notes (measured by the driver, not by you)",
                 "- Remaining LLM budget: $%.2f; rounds left after this one: %d." % (
                     self.remaining(), max(0, self.st["max_iterations"] - self.st["current_round"] - 1)),
                 "- Round %d independent evaluation of cand/: see `%s`." % (prev, self.rel(self.loop_dir / ("round-%d-eval.txt" % prev)))]
        f = self.loop_dir / ("round-%d-driver.md" % prev)
        if f.exists():
            notes.append(read(f).strip())
        return "\n".join(notes)

    def run_rounds(self):
        st = self.st
        while st["status"] == "running":
            n = st["current_round"]
            if n >= st["max_iterations"]:
                st["status"], st["stop_reason"] = "stopped_max_rounds", "reached max rounds (%d)" % st["max_iterations"]
                break
            verdict = self.do_round(n)
            if verdict == "COMPLETE":
                st["status"], st["stop_reason"] = "complete", "reviewer verdict COMPLETE at round %d" % n
            elif verdict == "STOP":
                st["status"], st["stop_reason"] = "stopped_review", "reviewer verdict STOP at round %d" % n
            elif st["stall_count"] >= st["max_stall"]:
                st["status"], st["stop_reason"] = "stopped_stall", "%d consecutive STALLED/REGRESSED rounds" % st["stall_count"]
            elif st["worker_failures"] >= 2:
                st["status"], st["stop_reason"] = "stopped_failures", "worker session failed twice in a row"
            else:
                st["current_round"] = n + 1
        log("loop status: %s (%s)" % (st["status"], st["stop_reason"]))

    def do_round(self, n):
        d, st = self.loop_dir, self.st
        pf = d / ("round-%d-prompt.md" % n)
        sf = d / ("round-%d-summary.md" % n)
        if not pf.exists():
            pf.write_text(self.build_prompt(n))
        if not sf.exists():
            res = self.call("worker", read(pf), tools=WORKER_TOOLS, budget=self.a.worker_budget, rnd=n,
                            timeout=self.a.worker_timeout)
            st["worker_failures"] = 0 if res.get("ok") else st["worker_failures"] + 1
            self.normalize_summary(n, res)
        self.post_worker(n)
        ef = d / ("round-%d-eval.txt" % n)
        if not ef.exists():
            self.driver_eval(n)
        rf = d / ("round-%d-review-result.md" % n)
        if not rf.exists():
            self.review(n)
        text = read(rf)
        verdict, mainline = parse_review(text)
        if st["last_counted_round"] < n:
            st["stall_count"] = st["stall_count"] + 1 if mainline in ("STALLED", "REGRESSED") else 0
            st["last_counted_round"] = n
        if verdict == "COMPLETE" and not self.complete_ok(n):
            log("COMPLETE downgraded to CONTINUE (driver guard: %s)" % self.complete_guard)
            with open(rf, "a") as f:
                f.write("\n\n[driver] COMPLETE ignored: %s\n" % self.complete_guard)
            verdict = "CONTINUE"
        log("round %d: reviewer verdict=%s mainline=%s stall_count=%d" % (n, verdict, mainline, st["stall_count"]))
        return verdict

    def complete_ok(self, n):
        rows = list(csv.DictReader(open(self.ws / "benchmark.csv"))) if (self.ws / "benchmark.csv").exists() else []
        ev = read(self.loop_dir / ("round-%d-eval.txt" % n))
        self.complete_guard = ""
        if not rows:
            self.complete_guard = "benchmark.csv has no rows"
        elif "STATUS: ok" not in ev and "unchanged" not in ev[:200]:
            self.complete_guard = "the driver's evaluation of cand/ is not STATUS: ok"
        return not self.complete_guard

    # ---- worker output hygiene
    def normalize_summary(self, n, res):
        d = self.loop_dir
        sf = d / ("round-%d-summary.md" % n)
        notes = []
        if not sf.exists():
            sf.write_text("# Round %d summary [driver fallback: the worker did not write the summary file]\n\n"
                          "Worker ok=%s error=%s\n\n## Worker final message\n%s\n\n## BitLesson Delta\n- Action: none\n"
                          "- Lesson ID(s): NONE\n- Notes: driver-generated fallback\n"
                          % (n, res.get("ok"), res.get("error"), res.get("text") or "(empty)"))
            notes.append("summary file missing -> driver fallback")
        txt = sf.read_text()
        err = bitlesson_delta_error(txt, read(self.ws / ".humanize" / "bitlesson.md"))
        if err:
            txt = re.sub(r"^##\s+BitLesson Delta.*?(?=^##\s|\Z)", "", txt, flags=re.M | re.S).rstrip() + (
                "\n\n## BitLesson Delta\n- Action: none\n- Lesson ID(s): NONE\n- Notes: [driver] original delta was invalid (%s)\n" % err)
            sf.write_text(txt)
            notes.append("BitLesson Delta invalid (%s) -> normalized" % err)
        if not (d / ("round-%d-contract.md" % n)).exists():
            notes.append("round contract file missing")
        (d / ("round-%d-hygiene.txt" % n)).write_text("\n".join(notes) + "\n")

    def post_worker(self, n):
        """Restore the frozen immutable section; commit the round."""
        d = self.loop_dir
        gt = d / "goal-tracker.md"
        frozen = d / "goal-tracker.immutable.md"
        cur = immutable_section(read(gt))
        warn = d / ("round-%d-hygiene.txt" % n)
        if n == 0:
            if not frozen.exists():
                frozen.write_text(cur)
        elif frozen.exists() and cur != read(frozen):
            t = read(gt)
            i, j = t.find("## IMMUTABLE SECTION"), t.find("## MUTABLE SECTION")
            gt.write_text(t[:i] + read(frozen) + t[j:])
            with open(warn, "a") as f:
                f.write("goal-tracker IMMUTABLE SECTION was modified after round 0 -> restored by the driver\n")
        if not (d / ("round-%d.committed" % n)).exists():
            self.commit("round %d" % n)
            (d / ("round-%d.committed" % n)).write_text(now())

    # ---- independent evaluation + promotion
    def driver_eval(self, n):
        d, ws = self.loop_dir, self.ws
        cur = sha([ws / "cand" / "grouped_mm_kernel_impl.h", ws / "cand" / "configs.json"])
        best = sha([ws / "best" / "grouped_mm_kernel_impl.h", ws / "best" / "configs.json"])
        facts = []
        if cur == best:
            txt = "unchanged: cand/ is identical to best/ (no evaluation run)"
            facts.append(txt)
            (d / ("round-%d-eval.txt" % n)).write_text(txt + "\n")
        else:
            if self.fake:
                res = {"status": "ok", "score": 1.0, "per_op": {}, "text": "STATUS: ok\nSCORE: 1.0000x (fake)\nDONE"}
            else:
                res = evaluate.evaluate(ws / "cand", suite=self.a.eval_suite, label="kda-r%d" % n)
            (d / ("round-%d-eval.txt" % n)).write_text(res["text"] + "\n")
            (d / ("round-%d-eval.json" % n)).write_text(json.dumps({k: v for k, v in res.items() if k != "text"}, indent=1,
                                                                     default=str))
            facts.append("quick/%s evaluation of cand/: status=%s score=%.4f" % (self.a.eval_suite, res.get("status"), res.get("score", 0)))
            if res.get("status") == "ok" and res.get("score", 0) >= PROMOTE_TRIGGER:
                facts.append(self.try_promote(n, "r%d" % n))
        # worker activity from the hook log
        facts.append(self.activity(n))
        (d / ("round-%d-driver.md" % n)).write_text("\n".join("- " + f for f in facts if f) + "\n")

    def try_promote(self, n, cid):
        ws, st = self.ws, self.st
        if self.fake:
            return "promotion skipped (fake)"
        res = evaluate.evaluate(ws / "cand", suite="full", label="kda-r%d-full" % n)
        (self.loop_dir / ("round-%d-full-eval.txt" % n)).write_text(res["text"] + "\n")
        ok, why = promotion_check(res, st["best_score"])
        if ok:
            for f in ("grouped_mm_kernel_impl.h", "configs.json"):
                shutil.copy(ws / "cand" / f, ws / "best" / f)
            snap = ws / "candidates" / cid
            snap.mkdir(parents=True, exist_ok=True)
            for f in ("grouped_mm_kernel_impl.h", "configs.json"):
                shutil.copy(ws / "cand" / f, snap / f)
            st["best_score"], st["best_id"] = round(res["score"], 4), cid
            with open(ws / "candidates.jsonl", "a") as f:
                f.write(json.dumps({"time": now(), "round": str(n), "id": cid, "parents": ["production"], "source": "driver",
                                    "status": "promoted", "direction": "driver promotion after full-suite check",
                                    "note": "full SCORE %.4f, per-op %s" % (res["score"], res.get("per_op")),
                                    "snapshot": "candidates/%s" % cid}) + "\n")
            return "PROMOTED to best/ as %s: full-suite %s" % (cid, why)
        return "not promoted: %s (full-suite score %.4f)" % (why, res.get("score", 0))

    def activity(self, n):
        rows = []
        for l in read(self.ws / ".humanize" / "tool_log.jsonl").splitlines():
            try:
                rows.append(json.loads(l))
            except ValueError:
                pass
        rows = [r for r in rows if r.get("round") == str(n)]
        if not rows:
            return "tool activity log: no entries for this round (hooks may not have run)"
        skills = sorted({r["detail"] for r in rows if r["tool"] == "Skill"})
        bash = [r["detail"] for r in rows if r["tool"] == "Bash"]
        cnt = lambda p: sum(1 for b in bash if re.match(r"\s*%s" % p, b))  # noqa: E731
        wiki_reads = sum(1 for r in rows if r["tool"] == "Read" and "/wiki/" in r["detail"])
        return ("tool activity (hook log): Skill calls %s; kw %d; xpu-report %d; ggemm-eval run %d, profile %d; kda-record %d; "
                "wiki page Reads %d; total tool uses %d" % (skills or "NONE", cnt("kw"), cnt("xpu-report"),
                                                            sum(1 for b in bash if "ggemm-eval run" in b),
                                                            sum(1 for b in bash if "ggemm-eval profile" in b),
                                                            cnt("kda-record"), wiki_reads, len(rows)))

    # ---- review
    def review(self, n):
        d, ws, st = self.loop_dir, self.ws, self.st
        audit = (n + 1) % st["full_review_round"] == 0
        facts = read(d / ("round-%d-eval.txt" % n))[:6000] + "\n" + read(d / ("round-%d-driver.md" % n))
        hyg = read(d / ("round-%d-hygiene.txt" % n)).strip()
        if hyg:
            facts += "\nHygiene issues: " + hyg
        facts += "\nPromotion criteria (CLAUDE.md): full-suite SCORE >= %.2f, no op below %.2f; best_score so far %.4f (%s)." % (
            PROMOTE_SCORE, PROMOTE_MIN_OP, st["best_score"], st["best_id"])
        reps = sorted((ws / "profile").glob("*/REPORT.md"))
        facts += "\nProfile reports: %s" % (", ".join(self.rel(r) for r in reps) or "NONE")
        for r in reps[-3:]:
            chk = sh(["bash", str(ws / "bin" / "xpu-report"), "check", str(r)], ws)
            facts += "\n  %s -> %s" % (self.rel(r), chk.splitlines()[-1] if chk else "?")
        facts += "\nrows in benchmark.csv: %d, candidates.jsonl lines: %d" % (
            max(0, len(read(ws / "benchmark.csv").splitlines()) - 1), len(read(ws / "candidates.jsonl").splitlines()))
        stat = sh(["git", "diff", "--stat", st["base_commit"], "--", "cand/"], ws)
        diff = sh(["git", "diff", "HEAD~1", "--", "cand/"], ws)
        facts += "\n\nCandidate change since loop start (git diff --stat):\n" + (stat or "(none)")
        facts += "\n\nCandidate change in THIS round (truncated):\n```diff\n" + "\n".join(diff.splitlines()[:250]) + "\n```"
        prompt = render(read(PROMPTS / "review.md"), ROUND=n, PLAN_FILE=st["plan_file"],
                        GOAL_TRACKER_FILE=self.rel(d / "goal-tracker.md"), SUMMARY=read(d / ("round-%d-summary.md" % n)),
                        DRIVER_FACTS=facts, LOOP_DIR=self.rel(d), BASE_COMMIT=st["base_commit"][:12],
                        ALIGNMENT_PART=render(read(PROMPTS / "alignment.md"), ROUND=n, COMPLETED=n + 1, LOOP_DIR=self.rel(d))
                        if audit else "")
        text = ""
        for attempt in range(2):
            res = self.call("reviewer" + ("-audit" if audit else ""), prompt if attempt == 0 else prompt + (
                "\n\nYour previous review lacked the mandatory `Mainline Progress Verdict:` line (ADVANCED / STALLED / REGRESSED). "
                "Produce the full review again including it.\n\nPrevious review:\n" + text),
                tools=REVIEW_TOOLS, budget=self.a.reviewer_budget, rnd=n, model=self.rmodel)
            text = res.get("text") or ("(reviewer failed: %s)" % res.get("error"))
            if parse_review(text)[1] or not res.get("ok"):
                break
        if not parse_review(text)[1]:
            text += "\n\nMainline Progress Verdict: STALLED\n(driver: reviewer produced no verdict line or failed)\nCONTINUE\n"
        (d / ("round-%d-review-prompt.md" % n)).write_text(prompt)
        (d / ("round-%d-review-result.md" % n)).write_text(text.rstrip() + "\n")

    # ---- finalize
    def finalize(self, force_review=False):
        d, ws, st = self.loop_dir, self.ws, self.st
        if st["final_done"]:
            return
        cur = sha([ws / "cand" / "grouped_mm_kernel_impl.h", ws / "cand" / "configs.json"])
        best = sha([ws / "best" / "grouped_mm_kernel_impl.h", ws / "best" / "configs.json"])
        final_txt = ""
        if cur != best and not self.fake:
            res = evaluate.evaluate(ws / "cand", suite="full", label="kda-final")
            final_txt = res["text"]
            ok, why = promotion_check(res, st["best_score"])
            if ok:
                for f in ("grouped_mm_kernel_impl.h", "configs.json"):
                    shutil.copy(ws / "cand" / f, ws / "best" / f)
                st["best_score"], st["best_id"] = round(res["score"], 4), "final"
                final_txt += "\nPROMOTED to best/ (final cand): %s\n" % why
            else:
                final_txt += "\nNot promoted: %s\n" % why
        else:
            final_txt = "cand/ is identical to best/ (best_id %s, best_score %.4f); no extra final evaluation." % (
                st["best_id"], st["best_score"])
        (d / "final-eval.txt").write_text(final_txt + "\n")
        # code-review phase
        if (st["status"] == "complete" or force_review) and self.a.final_review != "never" and self.remaining() > 1.0:
            prompt = render(read(PROMPTS / "review_code.md"), REASON=st["stop_reason"], BASE_COMMIT=st["base_commit"][:12],
                            LOOP_DIR=self.rel(d))
            res = self.call("code-review", prompt, tools=REVIEW_TOOLS, budget=self.a.reviewer_budget, rnd=None, model=self.rmodel)
            findings = res.get("text") or ""
            (d / "code-review-result.md").write_text(findings + "\n")
            if re.search(r"\[P[01]\]", findings) and self.remaining() > 2.0:
                self.call("finalize-fix", render(read(PROMPTS / "finalize.md"), FINDINGS=findings), tools=WORKER_TOOLS,
                          budget=self.a.worker_budget, rnd=None)
                self.commit("finalize: code review fixes")
        self.report()
        st["final_done"] = 1
        self.commit("final report")

    def report(self):
        d, ws, st = self.loop_dir, self.ws, self.st
        rows = list(csv.DictReader(open(ws / "benchmark.csv"))) if (ws / "benchmark.csv").exists() else []
        cands = [json.loads(l) for l in read(ws / "candidates.jsonl").splitlines() if l.strip()]
        costs = [json.loads(l) for l in read(self.cost_file).splitlines() if l.strip()]
        by_role = {}
        for c in costs:
            by_role[c["role"]] = by_role.get(c["role"], 0) + c["cost_usd"]
        L = ["# KDA / RLCR run report", "",
             "- run directory: `%s`" % ws, "- loop: `%s`, phase %s, status **%s** (%s)" % (d.name, st["phase"], st["status"], st["stop_reason"]),
             "- rounds executed: %d (max %d)" % (len(list(d.glob("round-*-review-result.md"))), st["max_iterations"]),
             "- LLM spend: $%.2f of $%.2f (%s)" % (self.spent(), self.a.max_llm_usd,
                                                    ", ".join("%s $%.2f" % kv for kv in sorted(by_role.items()))),
             "- best promoted candidate: `%s`, full-suite SCORE %.4f (promotion needs >= %.2f and no op < %.2f)" % (
                 st["best_id"], st["best_score"], PROMOTE_SCORE, PROMOTE_MIN_OP), "",
             "## Rounds", "", "| round | driver eval | reviewer verdict | mainline progress |", "|---|---|---|---|"]
        for f in sorted(d.glob("round-*-review-result.md"), key=lambda p: int(p.name.split("-")[1])):
            n = f.name.split("-")[1]
            ev = read(d / ("round-%s-eval.txt" % n))
            sc = re.search(r"SCORE: ([\d.]+)x", ev)
            v, mp = parse_review(read(f))
            L.append("| %s | %s | %s | %s |" % (n, ("score " + sc[1]) if sc else ev.split("\n")[0][:60], v, mp))
        L += ["", "## benchmark.csv (%d rows)" % len(rows), "", "| round | source | candidate | suite | status | score | note |", "|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append("| %s | %s | %s | %s | %s | %s | %s |" % (r["round"], r["source"], r["candidate"], r["suite"], r["status"],
                                                             r["score"], r["note"][:80]))
        L += ["", "## Candidate DAG (candidates.jsonl)", ""]
        for c in cands:
            L.append("- `%s` <- %s: **%s** %s | %s" % (c["id"], ",".join(c.get("parents", [])), c["status"],
                                                     c.get("direction", "")[:60], c.get("note", "")[:140]))
        L += ["", "## Final evaluation", "", "```", read(d / "final-eval.txt")[:3000], "```", "",
              "## Skill/tool activity (hook log)", ""]
        tl = [json.loads(l) for l in read(ws / ".humanize" / "tool_log.jsonl").splitlines() if l.strip()]
        L.append("Skill calls: %s" % (sorted({(r["round"] or "-") + ":" + r["detail"] for r in tl if r["tool"] == "Skill"}) or "none"))
        L.append("tool uses: %d (%s)" % (len(tl), ", ".join("%s %d" % (t, sum(1 for r in tl if r["tool"] == t)) for t in sorted({r["tool"] for r in tl}))))
        (ws / "FINAL_REPORT.md").write_text("\n".join(L) + "\n")
        shutil.copy(ws / "FINAL_REPORT.md", d / "final-report.md")

    # ---- main
    def run(self):
        self.setup_workspace()
        try:
            self.draft_stage()
            self.plan_stage()
            self.start_loop()
            try:
                self.run_rounds()
            except CostCap as e:
                self.st["status"], self.st["stop_reason"] = "stopped_cost", str(e)
                log(str(e))
            self.finalize(force_review=(self.a.final_review == "always"))
        except StopLoop as e:
            log("stopped: %s" % e)
            if self.st is not None:
                self.st["status"], self.st["stop_reason"] = "stopped_" + e.kind, str(e)
            return 2
        except CostCap as e:
            log(str(e))
            return 3
        log("done: status %s; report %s" % (self.st["status"], self.ws / "FINAL_REPORT.md"))
        return 0


class CostCap(Exception):
    pass


class StopLoop(Exception):
    def __init__(self, kind, msg):
        super().__init__(msg)
        self.kind = kind


# ---------------------------------------------------------- pure helpers ---
def immutable_section(t):
    i, j = t.find("## IMMUTABLE SECTION"), t.find("## MUTABLE SECTION")
    return t[i:j] if 0 <= i < j else ""


def parse_review(text):
    """-> (verdict COMPLETE|STOP|CONTINUE, mainline ADVANCED|STALLED|REGRESSED|None)"""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    last = lines[-1].strip("*`_ .#").upper() if lines else ""
    verdict = last if last in ("COMPLETE", "STOP") else "CONTINUE"
    m = re.search(r"Mainline Progress Verdict:\s*[*`]*\s*(ADVANCED|STALLED|REGRESSED)", text, re.I)
    return verdict, (m.group(1).upper() if m else None)


def bitlesson_delta_error(summary, kb):
    m = re.search(r"^##\s+BitLesson Delta\s*\n(.*?)(?=^##\s|\Z)", summary, re.M | re.S)
    if not m:
        return "section missing"
    sec = m.group(1)
    act = re.search(r"Action:\s*(\w+)", sec)
    ids = re.search(r"Lesson ID\(s\):\s*(.*)", sec)
    notes = re.search(r"Notes:\s*(\S.*)", sec)
    if not act or act.group(1).lower() not in ("none", "add", "update"):
        return "Action must be none|add|update"
    idtxt = (ids.group(1).strip() if ids else "")
    if act.group(1).lower() == "none":
        return "" if idtxt.upper() in ("", "NONE") else "Action none requires Lesson ID(s): NONE"
    lessons = re.findall(r"BL-\d{8}-[\w-]+", idtxt)
    if not lessons:
        return "add/update needs concrete BL-YYYYMMDD-short-name ids"
    missing = [i for i in lessons if i not in kb]
    if missing:
        return "ids not in bitlesson.md: %s" % ",".join(missing)
    if not notes:
        return "Notes missing"
    return ""


def promotion_check(res, best_score):
    """Promotion criteria of CLAUDE.md on a FULL-suite result."""
    if res.get("status") != "ok":
        return False, "status %s" % res.get("status")
    if res.get("suite") != "full":
        return False, "not a full-suite result"
    if res.get("score", 0) < PROMOTE_SCORE:
        return False, "SCORE %.4f < %.2f" % (res["score"], PROMOTE_SCORE)
    bad = {k: v for k, v in (res.get("per_op") or {}).items() if v < PROMOTE_MIN_OP}
    if bad:
        return False, "op regression > 3%%: %s" % {k: round(v, 3) for k, v in bad.items()}
    if (res.get("build") or {}).get("spills"):
        return False, "register spills: %s" % res["build"]["spills"]
    if res["score"] <= best_score:
        return False, "does not beat current best %.4f" % best_score
    return True, "SCORE %.4f, per-op %s" % (res["score"], {k: round(v, 3) for k, v in (res.get("per_op") or {}).items()})


# ------------------------------------------------------------ fake LLM ---
def fake_llm(lp, role, prompt, rnd):
    ws, d = lp.ws, lp.loop_dir
    out = {"ok": True, "text": "", "cost_usd": 0.01, "session_id": "fake", "seconds": 0.1, "error": ""}
    if role == "draft":
        (ws / "docs" / "draft.md").write_text("# Draft (fake)\n")
    elif role == "gen-plan":
        (ws / "docs" / "plan.md").write_text("# Plan (fake)\n## Goal Description\nFake goal.\n## Acceptance Criteria\n- AC-1: fake\n")
    elif role == "worker":
        (d / ("round-%d-contract.md" % rnd)).write_text("fake contract\n")
        (d / ("round-%d-summary.md" % rnd)).write_text("# Round %d summary\nfake\n## BitLesson Delta\n- Action: none\n"
                                                      "- Lesson ID(s): NONE\n- Notes: fake\n" % rnd)
        with open(ws / "cand" / "configs.json") as f:
            cfg = json.load(f)
        cfg["k_grouped"]["prefetch"] = 2 + rnd % 2
        (ws / "cand" / "configs.json").write_text(json.dumps(cfg, indent=2))
        with open(ws / "benchmark.csv", "a") as f:
            f.write("%s,%s,worker,fake,,,quick,ok,1.0,,,,,,,none,NEUTRAL,,,fake\n" % (now(), rnd))
    elif role.startswith("reviewer"):
        out["text"] = "Fake review.\nMainline Progress Verdict: ADVANCED\n" + ("COMPLETE" if rnd >= 1 else "CONTINUE")
    else:
        out["text"] = "fake"
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m kda.rlcr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, help="run directory = the task workspace (resumable)")
    ap.add_argument("--phase", type=int, choices=(1, 2, 3), default=2)
    ap.add_argument("--max-rounds", type=int, default=42)
    ap.add_argument("--model", default=None, help="worker model (default $GGEMM_LLM_MODEL or opus)")
    ap.add_argument("--reviewer-model", default=None, help="default: same as --model")
    ap.add_argument("--plan-model", default=None)
    ap.add_argument("--max-llm-usd", type=float, default=60.0, help="hard cap on total LLM spend of this run")
    ap.add_argument("--target-speedup", type=float, default=1.05)
    ap.add_argument("--guidance", default="", help="human guidance appended to the phase prompt")
    ap.add_argument("--guidance-file")
    ap.add_argument("--eval-suite", default="quick", help="suite of the driver's per-round independent evaluation")
    ap.add_argument("--full-review-round", type=int, default=5, help="alignment audit at rounds N-1, 2N-1, ...")
    ap.add_argument("--max-stall", type=int, default=3, help="stop after this many consecutive STALLED/REGRESSED reviews")
    ap.add_argument("--draft-budget", type=float, default=5.0)
    ap.add_argument("--plan-budget", type=float, default=2.5)
    ap.add_argument("--worker-budget", type=float, default=8.0, help="per worker session USD cap")
    ap.add_argument("--reviewer-budget", type=float, default=3.0)
    ap.add_argument("--worker-timeout", type=int, default=5400)
    ap.add_argument("--final-review", choices=("complete", "always", "never"), default="complete")
    ap.add_argument("--new-loop", action="store_true", help="start a new loop in an existing workspace (keeps docs/plan.md)")
    a = ap.parse_args(argv)
    return Loop(a).run()


if __name__ == "__main__":
    sys.exit(main())
