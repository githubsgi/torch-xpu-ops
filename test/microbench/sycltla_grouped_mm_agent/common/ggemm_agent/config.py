"""Paths and site settings, all overridable through environment variables.

GGEMM_AGENT_RUNS     shared work dir (jobs, build cache, slot locks)
                     default ~/aimod/ggemm_agent_runs
GGEMM_SYCL_TLA_DIR   sycl-tla checkout (v0.9)      default /mnt/aimod/subratag/sycl-tla
GGEMM_NODEFILE       compute nodes, one per line    default ~/PBS_NODEFILE
GGEMM_NODES          comma list, overrides the node file (short or full names)
GGEMM_TILES_PER_NODE PVC tiles per node (FLAT)      default 12
GGEMM_BUILDS_PER_NODE concurrent icpx builds/node   default 16
GGEMM_ONEAPI_MODULES "<module use dir>:<module>"    default the dle 2026.1.2.25 LKG
GGEMM_PYTHON         python on the compute nodes    default ~/miniforge3/bin/python3
GGEMM_AOT_DEVICE     ocloc device                   default pvc
"""

import os
from pathlib import Path

HOME = Path.home()
COMMON_DIR = Path(__file__).resolve().parent.parent        # .../sycltla_grouped_mm_agent/common
AGENT_DIR = COMMON_DIR.parent                              # .../sycltla_grouped_mm_agent
REPO_DIR = AGENT_DIR.parent.parent.parent                  # torch-xpu-ops checkout
SRC_DIR = REPO_DIR / "src/ATen/native/xpu/grouped_mm/sycltla"
TUNER_DIR = REPO_DIR / "test/microbench/sycltla_grouped_mm_tune"
EVAL_DIR = COMMON_DIR / "eval"
PROD_HEADER = SRC_DIR / "grouped_mm_kernel_impl.h"

RUNS = Path(os.environ.get("GGEMM_AGENT_RUNS", str(HOME / "aimod/ggemm_agent_runs")))
SYCL_TLA = Path(os.environ.get("GGEMM_SYCL_TLA_DIR", "/mnt/aimod/subratag/sycl-tla"))
NODEFILE = Path(os.environ.get("GGEMM_NODEFILE", str(HOME / "PBS_NODEFILE")))
TILES_PER_NODE = int(os.environ.get("GGEMM_TILES_PER_NODE", "12"))
BUILDS_PER_NODE = int(os.environ.get("GGEMM_BUILDS_PER_NODE", "16"))
ONEAPI_MODULES = os.environ.get(
    "GGEMM_ONEAPI_MODULES",
    "/opt/intel/OneAPI-prerelease/deep-learning-lkg/2026.1.2.25/modulefiles:dle/2026.1.2.25")
NODE_PYTHON = os.environ.get("GGEMM_PYTHON", str(HOME / "miniforge3/bin/python3"))
AOT_DEVICE = os.environ.get("GGEMM_AOT_DEVICE", "pvc")
UNITRACE = os.environ.get("GGEMM_UNITRACE", str(HOME / "datavol/tools/pti-gpu-0.16.0-rc1/bin/unitrace"))

BUILD_TIMEOUT_S = int(os.environ.get("GGEMM_BUILD_TIMEOUT_S", "1200"))
RUN_TIMEOUT_S = int(os.environ.get("GGEMM_RUN_TIMEOUT_S", "900"))


def nodes():
    """Compute nodes to use (full host names)."""
    env = os.environ.get("GGEMM_NODES")
    if env:
        names = [n.strip() for n in env.split(",") if n.strip()]
    else:
        names = [l.strip() for l in NODEFILE.read_text().splitlines() if l.strip()]
    seen, out = set(), []
    for n in names:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def short(node):
    return node.split(".")[0]
