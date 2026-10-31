"""ggemm-eval: the one tool every harness (and every LLM agent) uses.

    ggemm-eval init <dir> [--from <cand>]       new candidate (production copy by default)
    ggemm-eval run <dir> [--suite quick|small|large|full] [--ops a,b] [--verify-only]
    ggemm-eval profile <dir> [--case tp4ep4_uniform] [--op w13_dW]
    ggemm-eval diff <dir>                       candidate header vs production
    ggemm-eval slots                            who is using which node / tile
    ggemm-eval nodes                            check the compute nodes respond
"""

import argparse
import json
import subprocess
import sys

from . import candidate, config, evaluate, slots


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ggemm-eval", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("init")
    p.add_argument("dir")
    p.add_argument("--from", dest="src")
    p = sub.add_parser("run")
    p.add_argument("dir")
    p.add_argument("--suite", default="quick", help="quick (2 cases) | small (TP4/EP4, ~190 rows/expert) | "
                                                    "large (EP8, ~6k rows/expert) | full")
    p.add_argument("--ops", default="", help="subset, e.g. w13_dW,w2_dW (default all six)")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--rounds", type=int, default=11)
    p.add_argument("--label", default="")
    p.add_argument("--json", action="store_true", help="print the full JSON instead of the summary")
    p = sub.add_parser("profile")
    p.add_argument("dir")
    p.add_argument("--case", default="tp4ep4_uniform")
    p.add_argument("--op", default="w13_dW")
    p.add_argument("--label", default="")
    p = sub.add_parser("diff")
    p.add_argument("dir")
    sub.add_parser("slots")
    sub.add_parser("nodes")
    a = ap.parse_args(argv)

    if a.cmd == "init":
        d = candidate.init_dir(a.dir, a.src)
        print(f"candidate at {d}: edit grouped_mm_kernel_impl.h and configs.json, then `ggemm-eval run {d}`")
    elif a.cmd == "run":
        res = evaluate.evaluate(a.dir, suite=a.suite, ops=a.ops, mode="verify" if a.verify_only else "both",
                                rounds=a.rounds, label=a.label or a.dir)
        evaluate.save_to_cand(res)
        print(json.dumps(res, indent=1) if a.json else res["text"])
        return 0 if res["status"] == "ok" else 1
    elif a.cmd == "profile":
        from . import profile
        res = profile.profile(a.dir, case=a.case, op=a.op, label=a.label or a.dir)
        print(res["text"])
        return 0 if res.get("ok") else 1
    elif a.cmd == "diff":
        print(candidate.diff_vs_production(a.dir) or "(identical to production)")
    elif a.cmd == "slots":
        print(slots.status())
    elif a.cmd == "nodes":
        for n in config.nodes():
            p = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", n,
                                "pgrep -u $USER -c -f 'agent_eval|icpx|python' ; ls /dev/dri | grep -c render"],
                               capture_output=True, text=True, timeout=30)
            print(f"{config.short(n)}: {'ok' if p.returncode in (0, 1) else 'UNREACHABLE'} "
                  f"{p.stdout.split()} {p.stderr.strip()[:100]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
