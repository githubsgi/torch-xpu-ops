#!/usr/bin/env python3
"""Validate the wiki: frontmatter fields, ids unique, related ids exist, confidence values,
pattern pages have symptoms + candidate_techniques that resolve, every page cites sources.
Exit 1 on errors.  (`kw validate`)"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wikilib  # noqa: E402

REQ = ["id", "title", "type", "tags", "architectures", "confidence", "sources", "related"]


def main():
    pages = wikilib.all_pages()
    ids = {}
    err = []
    for p in pages:
        name = os.path.relpath(p["_path"], wikilib.WIKI)
        for k in REQ:
            if k not in p or p[k] in ("", []):
                err.append("%s: missing %s" % (name, k))
        i = p.get("id")
        if i in ids:
            err.append("%s: duplicate id %s" % (name, i))
        ids[i] = p
        t = p.get("type")
        if t not in wikilib.TYPE_DIRS:
            err.append("%s: bad type %r" % (name, t))
        elif os.path.basename(os.path.dirname(p["_path"])) != wikilib.TYPE_DIRS[t]:
            err.append("%s: type %s lives in the wrong directory" % (name, t))
        if p.get("confidence") not in wikilib.CONFIDENCE:
            err.append("%s: bad confidence %r" % (name, p.get("confidence")))
        if os.path.basename(p["_path"]) != "%s.md" % i:
            err.append("%s: file name must be <id>.md" % name)
        if t == "pattern":
            for k in ("symptoms", "candidate_techniques"):
                if not p.get(k):
                    err.append("%s: pattern page needs %s" % (name, k))
    for p in pages:
        name = os.path.relpath(p["_path"], wikilib.WIKI)
        for k in ("related", "candidate_techniques"):
            for r in p.get(k, []):
                if r not in ids:
                    err.append("%s: %s -> unknown id %s" % (name, k, r))
        for m in re.finditer(r"\[\[([\w-]+)\]\]", p["_body"]):
            if m.group(1) not in ids:
                err.append("%s: [[%s]] unknown id" % (name, m.group(1)))
    for e in err:
        print("ERROR", e)
    print("%d pages, %d errors" % (len(pages), len(err)))
    return 1 if err else 0


if __name__ == "__main__":
    sys.exit(main())
