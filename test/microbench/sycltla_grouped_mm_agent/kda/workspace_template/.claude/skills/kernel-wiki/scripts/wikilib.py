"""Shared helpers for the kernel-wiki scripts (stdlib only, python >= 3.6)."""
import os
import re

WIKI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "wiki")
TYPE_DIRS = {"hardware": "hardware", "technique": "techniques", "kernel": "kernels",
             "pattern": "patterns", "language": "languages", "migration": "migration"}
CONFIDENCE = ["verified", "source-reported", "inferred", "experimental"]
LIST_KEYS = ("tags", "architectures", "sources", "related", "symptoms", "candidate_techniques")


def _parse_value(v):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [x.strip().strip("'\"") for x in re.split(r",\s*(?![^()]*\))", inner)]
    return v.strip("'\"")


def parse_page(path):
    """-> (frontmatter dict, body str). Frontmatter: flat `key: value` or `key: [a, b]`."""
    text = open(path, encoding="utf-8").read()
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end < 0:
        return {}, text
    fm = {}
    for line in text[3:end].strip().splitlines():
        if ":" in line and not line.startswith(" "):
            k, v = line.split(":", 1)
            fm[k.strip()] = _parse_value(v)
    return fm, text[end + 4:].lstrip("\n")


def all_pages():
    out = []
    for d in sorted(os.listdir(WIKI)):
        p = os.path.join(WIKI, d)
        if not os.path.isdir(p):
            continue
        for f in sorted(os.listdir(p)):
            if f.endswith(".md"):
                fm, body = parse_page(os.path.join(p, f))
                fm["_path"] = os.path.join(p, f)
                fm["_body"] = body
                out.append(fm)
    return out


def find_page(ident):
    for pg in all_pages():
        if pg.get("id") == ident or os.path.abspath(pg["_path"]) == os.path.abspath(ident) \
                or os.path.basename(pg["_path"]) == ident or os.path.basename(pg["_path"]) == ident + ".md":
            return pg
    return None
