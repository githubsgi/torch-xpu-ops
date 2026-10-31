#!/usr/bin/env python3
"""Search the kernel wiki by tag / type / architecture / confidence / keywords.

  query.py --tag grouped-gemm --tag prefetch        pages having ALL given tags
  query.py --type pattern                            all pattern pages
  query.py --type pattern --q "sbid short k"         keyword match (title, tags, symptoms, body)
  query.py --list-tags                               tag vocabulary with counts
  query.py --symptom "idle"                          pattern pages whose symptoms mention it

Prints one line per page: id | type | confidence | title | tags.  Use get_page.py to read.
"""
import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wikilib  # noqa: E402


def score(pg, words):
    hay_head = " ".join([pg.get("id", ""), pg.get("title", ""), " ".join(pg.get("tags", []))]).lower()
    hay_sym = " ".join(pg.get("symptoms", [])).lower() if isinstance(pg.get("symptoms"), list) else ""
    body = pg["_body"].lower()
    s = 0
    for w in words:
        w = w.lower()
        if w in hay_head:
            s += 5
        if w in hay_sym:
            s += 3
        s += min(body.count(w), 3)
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", action="append", default=[])
    ap.add_argument("--type")
    ap.add_argument("--arch")
    ap.add_argument("--confidence", help="minimum: verified > source-reported > inferred > experimental")
    ap.add_argument("--q", help="keywords (space separated, any)")
    ap.add_argument("--symptom")
    ap.add_argument("--list-tags", action="store_true")
    a = ap.parse_args()
    pages = wikilib.all_pages()
    if a.list_tags:
        c = collections.Counter(t for p in pages for t in p.get("tags", []))
        for t, n in sorted(c.items(), key=lambda x: (-x[1], x[0])):
            print("%-28s %d" % (t, n))
        return 0
    res = []
    for p in pages:
        if a.type and p.get("type") != a.type:
            continue
        if a.arch and a.arch not in p.get("architectures", []):
            continue
        if any(t not in p.get("tags", []) for t in a.tag):
            continue
        if a.confidence:
            if wikilib.CONFIDENCE.index(p.get("confidence", "experimental")) > wikilib.CONFIDENCE.index(a.confidence):
                continue
        s = 0
        if a.q:
            s = score(p, a.q.split())
            if s == 0:
                continue
        if a.symptom:
            if a.symptom.lower() not in " ".join(p.get("symptoms", [])).lower():
                continue
        res.append((s, p))
    res.sort(key=lambda x: (-x[0], x[1].get("type", ""), x[1].get("id", "")))
    for s, p in res:
        print("%s | %s | %s | %s | %s" % (p.get("id"), p.get("type"), p.get("confidence"), p.get("title"),
                                           ",".join(p.get("tags", []))))
    if not res:
        print("(no pages match; try --list-tags or grep_wiki.py)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
