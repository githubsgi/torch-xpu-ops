"""Workspaces the LLM edits in, and static checks (Xe-Forge's INVALID checks in compile_and_verify)."""

import difflib
import os
import re
import shutil
import stat
from pathlib import Path

from . import env
from .env import candidate, config

HEADER, CONFIGS = "grouped_mm_kernel_impl.h", "configs.json"
NOTES = Path.home() / "datavol/logs/torchtitan/outputs_deepseek_v3_16b/notes_ggemm.txt"
SUMMARY = Path.home() / "datavol/logs/torchtitan/outputs_deepseek_v3_16b/sycltla_ggemm_kernel_summary.md"
TLA_FILES = [
    "include/cute/atom/copy_traits_xe_2d.hpp", "include/cute/atom/mma_traits_xe.hpp",
    "include/cute/atom/reorder_atom_xe.hpp", "include/cute/arch/copy_xe_2d.hpp", "include/cute/arch/mma_xe.hpp",
    "include/cute/arch/reorder_xe.hpp", "include/cute/util/xe_split_barrier.hpp",
    "examples/12_xe20_moe_gemm_cute_interface/moe_grouped_gemm.hpp",
    "examples/12_xe20_moe_gemm_cute_interface/moe_tile_scheduler.hpp",
]
NOTES_SECTIONS = ("0.", "11.1", "11.2", "11.3", "11.4", "16.2", "17.2", "22.2")


def _excerpt_notes():
    """Sections 0, 11.1-11.4, 16.2, 17.2, 22.2 of the notes (the measured facts the KB cites)."""
    if not NOTES.exists():
        return "(notes file not found)\n"
    lines = NOTES.read_text().splitlines()
    head = re.compile(r"^(\d+(?:\.\d+)?)\.? ")
    out, keep = [], False
    for l in lines:
        m = head.match(l)
        if m and not l.startswith(" "):
            keep = any(m.group(1) == s.rstrip(".") for s in NOTES_SECTIONS)
        if keep:
            out.append(l)
    return "\n".join(out) + "\n"


def make_refs(workdir):
    """Read-only reference material shared by all workspaces: <workdir>/refs/ (files 0444, dirs 0555)."""
    refs = Path(workdir) / "refs"
    if (refs / ".done").exists():
        return refs
    if refs.exists():
        _rmtree(refs)
    refs.mkdir(parents=True)
    shutil.copy(config.PROD_HEADER, refs / "production_grouped_mm_kernel_impl.h")
    shutil.copy(config.EVAL_DIR / "cand_cfg.cpp", refs / "cand_cfg.cpp")
    shutil.copy(config.SRC_DIR / "grouped_mm_kernels.h", refs / "grouped_mm_kernels.h")
    shutil.copy(Path(__file__).parent / "knowledge_base" / "pvc_sycltla_grouped_gemm.yaml", refs / "knowledge_base.yaml")
    (refs / "notes_excerpt.txt").write_text(_excerpt_notes())
    if SUMMARY.exists():
        shutil.copy(SUMMARY, refs / "kernel_summary.md")
    for rel in TLA_FILES:
        src = config.SYCL_TLA / rel
        if src.exists():
            dst = refs / "sycl-tla" / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
    (refs / ".done").write_text("ok")
    for root, dirs, files in os.walk(refs):
        for f in files:
            os.chmod(Path(root) / f, 0o444)
    for root, dirs, files in os.walk(refs, topdown=False):
        os.chmod(root, 0o555)
    return refs


def _rmtree(p):
    for root, dirs, files in os.walk(p):
        os.chmod(root, 0o755)
    shutil.rmtree(p, ignore_errors=True)


def make_workspace(ws, base_cand, refs):
    """ws/cand = copy of base_cand; ws/refs -> shared read-only refs."""
    ws = Path(ws)
    if ws.exists():
        shutil.rmtree(ws)
    (ws / "cand").mkdir(parents=True)
    for f in (HEADER, CONFIGS):
        shutil.copy(Path(base_cand) / f, ws / "cand" / f)
    os.symlink(os.path.relpath(refs, ws), ws / "refs")
    return ws


def read_cand(cand):
    return (Path(cand) / HEADER).read_text(), (Path(cand) / CONFIGS).read_text()


def cand_sha(cand):
    import hashlib
    h = hashlib.sha256()
    for t in read_cand(cand):
        h.update(t.encode())
    return h.hexdigest()[:12]


def diff_text(a_dir, b_dir, max_lines=260):
    out = []
    for f in (HEADER, CONFIGS):
        a = (Path(a_dir) / f).read_text().splitlines(keepends=True)
        b = (Path(b_dir) / f).read_text().splitlines(keepends=True)
        out += list(difflib.unified_diff(a, b, f"base/{f}", f"cand/{f}", n=2))
    if len(out) > max_lines:
        out = out[:max_lines] + [f"... ({len(out) - max_lines} more diff lines truncated)\n"]
    return "".join(out)


_FORBIDDEN = [
    (r"\b(malloc_device|malloc_shared|malloc_host|sycl::malloc|aligned_alloc_device)\b",
     "allocations in the launch path (Params has no workspace pointer; the ABI is fixed)"),
    (r"\bqueue\.(wait|memcpy|memset|fill|copy)\s*\(|\bq\.(wait|memcpy|memset)\s*\(",
     "host synchronization / memory ops in the launch path (no host sync on offs allowed)"),
    (r"\bhost_task\b", "host tasks in the launch path"),
    (r"\bavg_rows\b|\bhost_rows\b", "tile/config choice from host-known row counts (all EP ranks must run the same kernel)"),
]
_REQUIRED = ["run_m_grouped_cfg", "run_k_grouped_cfg", "TiledMmaFor"]


def static_checks(cand):
    """Cheap pre-checks before spending a build (Xe-Forge's INVALID: ... returns). Returns a message or None."""
    try:
        hdr, cfgs = candidate.load(cand)
    except (candidate.CandidateError, ValueError, KeyError) as e:
        return f"INVALID: candidate does not load: {e}"
    for name in _REQUIRED:
        if name not in hdr:
            return (f"INVALID: the header no longer defines `{name}` which agent_eval requires "
                    f"(contract in refs/cand_cfg.cpp). Restore it.")
    # strip comments so a comment mentioning avg_rows does not trip the check
    code = re.sub(r"//[^\n]*|/\*.*?\*/", "", hdr, flags=re.S)
    for pat, why in _FORBIDDEN:
        m = re.search(pat, code)
        if m:
            return f"INVALID: `{m.group(0)}`: {why}. Remove it."
    if "Params" in code and re.search(r"struct\s+Params\s*\{", code):
        return "INVALID: Params is redefined in the header; it is a fixed ABI from grouped_mm_kernels.h."
    for k in candidate.KINDS:
        c = cfgs[k]
        if "body" not in c and c["tile"][2] % 16:
            return f"INVALID: {k}: tile K {c['tile'][2]} must be a multiple of 16 (DPAS K for 16-bit inputs)."
    return None
