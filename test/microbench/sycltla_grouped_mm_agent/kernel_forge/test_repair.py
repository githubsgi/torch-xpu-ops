"""End-to-end check of the compile-error repair loop with a scripted fake LLM (real build + eval):
    python -m kernel_forge.test_repair WORKDIR
Call 1 breaks the header (syntax error) -> build_error; call 2 must see the compiler error in its
prompt and fixes it with a harmless header edit -> the child is admitted with repairs == 1.
"""
import argparse
import sys
from pathlib import Path

from . import _paths  # noqa: F401
from ggemm_agent import llm

from . import mcts


def main():
    wd = sys.argv[1]
    a = mcts.argparse.Namespace(**vars(_args(wd)))
    f = mcts.Forge(a)
    f.ensure_root()
    calls = []

    def fake(prompt, cwd, **kw):
        h = Path(cwd) / "grouped_mm_kernel_impl.h"
        calls.append(prompt)
        if len(calls) == 1:
            h.write_text(h.read_text() + "\nthis is not c++ ;;; {\n")
            return {"ok": True, "text": "CHANGE: scripted broken edit", "cost_usd": 0.01, "error": ""}
        assert "STATUS: build_error" in prompt and "error" in prompt.lower(), prompt[:500]
        txt = h.read_text().replace("\nthis is not c++ ;;; {\n", "\n// harness repair test\n")
        h.write_text(txt)
        return {"ok": True, "text": "CHANGE: scripted fix", "cost_usd": 0.01, "error": ""}

    llm.run_claude = fake       # mcts.llm is the same module object
    root = f.tree.nodes[f.tree.root]
    from ggemm_agent import candidate
    d = f.wd / "attempts" / "a900"
    candidate.init_dir(d, src=f.tree.node_dir(root.id))
    phdr, pcfg = candidate.load(d)
    out = f.llm_expand("a900", d, root, phdr, pcfg)
    print({k: (v if k != "eval" else "...") for k, v in out.items()})
    assert out["ok"] and out["repairs"] == 1 and len(calls) == 2, out
    print("repair test OK: build error was fed back, fix admitted")


def _args(wd):
    ap = mcts.argparse.ArgumentParser()
    ns = ap.parse_args([])
    ns.workdir, ns.iterations, ns.workers, ns.knob_ratio, ns.model = wd, 1, 1, 0.0, "sonnet"
    ns.max_llm_usd, ns.llm_call_usd, ns.llm_timeout_min, ns.max_wall_min = 5, 1, 5, 0
    ns.max_repairs, ns.suite, ns.final_suite, ns.top_k, ns.c, ns.vloss = 2, "quick", "full", 1, 0.1, 0.05
    ns.alpha0, ns.alpha1, ns.anneal_steps, ns.profile, ns.seed, ns.llm_code_only = 0.5, 0.3, 1000, False, 0, False
    return ns


if __name__ == "__main__":
    main()
