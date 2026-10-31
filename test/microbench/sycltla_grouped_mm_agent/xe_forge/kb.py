"""Knowledge base loader (PyYAML if importable, else a small subset parser; stdlib only otherwise)."""

import re
from pathlib import Path

KB_PATH = Path(__file__).resolve().parent / "knowledge_base" / "pvc_sycltla_grouped_gemm.yaml"


def _scalar(s):
    s = s.strip()
    if s.startswith("[") and s.endswith("]"):
        return [x.strip().strip("'\"") for x in s[1:-1].split(",") if x.strip()]
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        return s[1:-1]
    return s


def parse_subset(text):
    """Subset: top-level `key:` -> list of mappings; mapping keys are scalars, inline lists, or `|` blocks."""
    out, cur_list, item = {}, None, None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip() or line.lstrip().startswith("#"):
            i += 1
            continue
        if not line.startswith(" "):
            key = line.rstrip(":").strip()
            out[key] = cur_list = []
            item = None
            i += 1
            continue
        m = re.match(r"^  - (\w+):\s*(.*)$", line)
        if m:
            item = {}
            cur_list.append(item)
            line_key, val = m.group(1), m.group(2)
            indent = 4
        else:
            m = re.match(r"^    (\w+):\s*(.*)$", line)
            if not m:
                i += 1
                continue
            line_key, val = m.group(1), m.group(2)
            indent = 4
        if val == "|":
            i += 1
            block = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith(" " * (indent + 2))):
                block.append(lines[i][indent + 2:])
                i += 1
            item[line_key] = "\n".join(block).rstrip() + "\n"
            continue
        item[line_key] = _scalar(val)
        i += 1
    return out


def load(path=KB_PATH):
    text = Path(path).read_text()
    try:
        import yaml  # noqa: PLC0415
        return yaml.safe_load(text)
    except ImportError:
        return parse_subset(text)


def _render(e, with_stage=False):
    head = f"### {e.get('name', e['id'])} [{e['id']}]"
    parts = [head]
    for k in ("description", "text", "rationale", "try", "evidence", "source"):
        if e.get(k):
            v = e[k].strip() if isinstance(e[k], str) else str(e[k])
            parts.append(f"{k.upper() if k in ('try',) else k}: {v}" if k in ("try", "evidence", "source", "description")
                         else v)
    return "\n".join(parts)


def stage_context(stage, kb=None, include_facts=True):
    """Text for a stage prompt: all constraints that apply, the stage's patterns, the facts and API notes."""
    kb = kb or load()
    stage = getattr(stage, "value", stage)
    out = ["## Constraints (must hold)"]
    for c in kb.get("constraints", []):
        if "all" in c.get("stages", []) or stage in c.get("stages", []) or stage == "analysis":
            out.append(_render(c))
    pats = [p for p in kb.get("patterns", []) if p.get("stage") == stage or stage == "analysis"]
    if pats:
        out.append("## Patterns for this stage (priors, not promises; measure)")
        out += [_render(p) for p in pats]
    if include_facts:
        out.append("## Measured facts about this kernel")
        out += [_render(f) for f in kb.get("facts", [])]
        out.append("## API and structure notes")
        out += [_render(a) for a in kb.get("api", [])]
    return "\n\n".join(out)


def all_text(kb=None):
    return stage_context("analysis", kb)
