#!/usr/bin/env python3
"""Evidence records for the workspace (stdlib only, python >= 3.6).

  kda-record bench --id c3 --direction "pf2 for dW" [--suite small] [--eval cand/last_eval.json] [--note ".."]
        append one row to benchmark.csv from an eval JSON (default cand/last_eval.json).
        EVERY perf-related change gets a row (CLAUDE.md rule).
  kda-record cand --id c3 --parent c1 --status proposed|kept|rejected|promoted --direction ".." --note ".." [--snapshot cand]
        append one line to candidates.jsonl (the candidate DAG); --snapshot copies the candidate's
        grouped_mm_kernel_impl.h + configs.json into candidates/<id>/.  A rejected candidate needs a
        --note with the reason (and the numbers).
  kda-record show           print the table and the DAG
Parents are candidate ids; "production" is the root.
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BENCH = os.path.join(ROOT, "benchmark.csv")
CANDS = os.path.join(ROOT, "candidates.jsonl")
OPS = ["w13_fwd", "w2_fwd", "w13_dX", "w2_dX", "w13_dW", "w2_dW"]
COLS = ["time", "round", "source", "candidate", "parent", "direction", "suite", "status", "score"] + \
    ["op_" + o for o in OPS] + ["spills", "verdict", "per_case", "job", "note"]


def cur_round():
    try:
        return open(os.path.join(ROOT, ".kda_round")).read().strip()
    except OSError:
        return ""


def load_cands():
    out = []
    if os.path.exists(CANDS):
        for l in open(CANDS):
            l = l.strip()
            if l:
                out.append(json.loads(l))
    return out


def cmd_bench(a):
    path = a.eval or os.path.join(ROOT, "cand", "last_eval.json")
    res = json.load(open(path))
    per_op = res.get("per_op", {})
    cases = res.get("per_case", {})
    build = res.get("build", {})
    row = {"time": time.strftime("%F %T"), "round": cur_round(), "source": a.source, "candidate": a.id,
           "parent": a.parent or "", "direction": a.direction or "", "suite": res.get("suite", ""),
           "status": res.get("status", ""), "score": "%.4f" % res.get("score", 0.0),
           "spills": ";".join(build.get("spills", []) or []) or "none",
           "verdict": "", "per_case": ";".join("%s=%.3f" % (k, v.get("speedup", 0)) for k, v in cases.items()),
           "job": res.get("job", ""), "note": a.note or ""}
    for o in OPS:
        row["op_" + o] = "%.3f" % per_op[o] if o in per_op else ""
    s = res.get("score", 0.0)
    row["verdict"] = ("IMPROVED" if s >= 1.01 else "REGRESSED" if s <= 0.99 else "NEUTRAL") if res.get("status") == "ok" \
        else res.get("status", "")
    new = not os.path.exists(BENCH)
    with open(BENCH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        if new:
            w.writeheader()
        w.writerow(row)
    print("benchmark.csv += %s suite=%s status=%s score=%s %s" % (a.id, row["suite"], row["status"], row["score"],
                                                                   row["verdict"]))


def cmd_cand(a):
    cands = load_cands()
    ids = {c["id"] for c in cands} | {"production"}
    parents = [p for p in (a.parent or "production").split(",") if p]
    for p in parents:
        if p not in ids:
            sys.exit("unknown parent %s (known: %s)" % (p, ", ".join(sorted(ids))))
    if a.status == "rejected" and not (a.note and len(a.note) >= 20):
        sys.exit("a rejected candidate needs --note with the reason and the numbers (>= 20 chars)")
    rec = {"time": time.strftime("%F %T"), "round": cur_round(), "id": a.id, "parents": parents,
           "status": a.status, "direction": a.direction or "", "note": a.note or ""}
    if a.snapshot:
        d = os.path.join(ROOT, "candidates", a.id)
        os.makedirs(d, exist_ok=True)
        for f in ("grouped_mm_kernel_impl.h", "configs.json"):
            shutil.copy(os.path.join(ROOT, a.snapshot, f), os.path.join(d, f))
        rec["snapshot"] = "candidates/%s" % a.id
    with open(CANDS, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print("candidates.jsonl += %s (%s) parents=%s" % (a.id, a.status, ",".join(parents)))


def cmd_show(a):
    if os.path.exists(BENCH):
        for r in csv.DictReader(open(BENCH)):
            print("%-5s %-8s %-8s %-6s %-7s %-9s %s" % (r["round"], r["source"], r["candidate"], r["suite"],
                                                      r["score"], r["verdict"], r["direction"][:50]))
    print("--- candidates (latest record per id) ---")
    last = {}
    for c in load_cands():
        last[c["id"]] = c
    for c in last.values():
        print("%-8s <- %-14s %-9s %s | %s" % (c["id"], ",".join(c["parents"]), c["status"], c["direction"][:40],
                                             c["note"][:70]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("bench")
    p.add_argument("--id", required=True)
    p.add_argument("--parent")
    p.add_argument("--direction")
    p.add_argument("--eval")
    p.add_argument("--note")
    p.add_argument("--source", default="worker")
    p = sub.add_parser("cand")
    p.add_argument("--id", required=True)
    p.add_argument("--parent")
    p.add_argument("--status", required=True, choices=["proposed", "kept", "rejected", "promoted"])
    p.add_argument("--direction")
    p.add_argument("--note")
    p.add_argument("--snapshot")
    sub.add_parser("show")
    a = ap.parse_args()
    {"bench": cmd_bench, "cand": cmd_cand, "show": cmd_show}[a.cmd](a)


if __name__ == "__main__":
    main()
