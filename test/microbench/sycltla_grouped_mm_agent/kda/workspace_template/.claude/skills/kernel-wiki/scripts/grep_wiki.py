#!/usr/bin/env python3
"""Regex search over wiki bodies.  grep_wiki.py 'SBID|scoreboard' [--type pattern] [-C 1]"""
import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wikilib  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("regex")
    ap.add_argument("--type")
    ap.add_argument("-C", type=int, default=0, help="context lines")
    ap.add_argument("-i", action="store_true", help="ignore case")
    a = ap.parse_args()
    rx = re.compile(a.regex, re.I if a.i else 0)
    n = 0
    for p in wikilib.all_pages():
        if a.type and p.get("type") != a.type:
            continue
        lines = p["_body"].splitlines()
        for i, l in enumerate(lines):
            if rx.search(l):
                lo, hi = max(0, i - a.C), min(len(lines), i + a.C + 1)
                for j in range(lo, hi):
                    print("%s:%d: %s" % (p["id"], j + 1, lines[j]))
                if a.C:
                    print("--")
                n += 1
    if not n:
        print("(no match)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
