"""CLI: python -m xe_forge.run --workdir DIR [--stages memory_access,autotuning] [--model opus] ..."""

import argparse
import os
import sys
import time
from pathlib import Path


def build_parser():
    ap = argparse.ArgumentParser(prog="python -m xe_forge.run", description=__doc__)
    runs = Path(os.environ.get("GGEMM_AGENT_RUNS", str(Path.home() / "aimod/ggemm_agent_runs")))
    ap.add_argument("--workdir", help="run directory (resumes if pipeline.json exists); default "
                                      "$GGEMM_AGENT_RUNS/xe_forge/<run-id>")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%d-%H%M%S"))
    ap.add_argument("--stages", default="", help="comma list: algorithmic,discovery,memory_access,persistent_kernel,"
                                                  "device_specific,autotuning (default: stages with detected issues)")
    ap.add_argument("--model", default=os.environ.get("GGEMM_LLM_MODEL", "opus"))
    ap.add_argument("--effort", default=os.environ.get("GGEMM_LLM_EFFORT"))
    ap.add_argument("--max-llm-usd", type=float, default=float(os.environ.get("XF_MAX_LLM_USD", "80")),
                    help="cap on total LLM spend of the run")
    ap.add_argument("--per-call-usd", type=float, default=float(os.environ.get("GGEMM_LLM_BUDGET_USD", "8")),
                    help="budget of one LLM session")
    ap.add_argument("--suite", default="quick", help="suite of the stage gate / compile_and_verify")
    ap.add_argument("--final-suite", default="full")
    ap.add_argument("--max-iterations", type=int, default=int(os.environ.get("AGENT_MAX_ITERATIONS", "3")),
                    help="best-of iterations per stage (Xe-Forge AGENT_MAX_ITERATIONS)")
    ap.add_argument("--max-attempts", type=int, default=int(os.environ.get("MAX_ATTEMPTS_PER_STAGE", "5")),
                    help="CoVeR attempts (write -> compile_and_verify -> refine) per best-of iteration")
    ap.add_argument("--min-improvement", type=float, default=1.02,
                    help="keep gate: score >= this x the incumbent (Xe-Forge _MIN_IMPROVEMENT; 1.02 = the noise level)")
    ap.add_argument("--stop-on-no-improvement", action="store_true",
                    help="original Xe-Forge behavior: end the stage at the first iteration that is not a >=2%% gain "
                         "(default: use the remaining best-of iterations)")
    ap.add_argument("--no-profile", action="store_true", help="skip unitrace profiles (analysis then sees no digests)")
    ap.add_argument("--profile-matrix", choices=("full", "small"), default="full",
                    help="full: w13_fwd/dX/dW at tp4ep4_uniform and ep8_uniform; small: the 2 dW profiles")
    ap.add_argument("--lessons-dir", default=None, help="shared ledger directory (read all, write own)")
    ap.add_argument("--parallel", type=int, default=4, help="parallel evaluations in the autotuning stage")
    return ap, runs


def main(argv=None):
    ap, runs = build_parser()
    a = ap.parse_args(argv)
    a.workdir = a.workdir or str(runs / "xe_forge" / a.run_id)
    a.stages = [s for s in a.stages.split(",") if s]
    from .models import Stage
    for s in a.stages:
        Stage(s)   # validates
    from .pipeline import Pipeline
    return Pipeline(a).run()


if __name__ == "__main__":
    sys.exit(main())
