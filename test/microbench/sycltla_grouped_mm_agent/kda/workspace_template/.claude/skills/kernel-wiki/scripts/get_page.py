#!/usr/bin/env python3
"""Print a wiki page by id (or file name).

  get_page.py hw-pvc-dpas-xmx
  get_page.py pat-short-k-prologue-bound --frontmatter-only
  get_page.py tech-prefetch-distance --follow-related     # also prints the titles+ids of related pages
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wikilib  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("id", nargs="+")
    ap.add_argument("--frontmatter-only", action="store_true")
    ap.add_argument("--follow-related", action="store_true")
    a = ap.parse_args()
    rc = 0
    for ident in a.id:
        pg = wikilib.find_page(ident)
        if not pg:
            print("no such page: %s (use query.py)" % ident)
            rc = 1
            continue
        txt = open(pg["_path"], encoding="utf-8").read()
        if a.frontmatter_only:
            txt = txt[:txt.find("\n---", 3) + 4]
        print(txt)
        if a.follow_related:
            print("\n--- related ---")
            for r in pg.get("related", []):
                q = wikilib.find_page(r)
                print("%s | %s" % (r, q.get("title") if q else "(missing)"))
    return rc


if __name__ == "__main__":
    sys.exit(main())
