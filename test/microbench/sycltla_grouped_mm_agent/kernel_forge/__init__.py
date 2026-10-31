"""Kernel Forge (MCTS over kernel versions) for the SYCL-TLA grouped GEMM.

Importing the package puts ../common on sys.path so `ggemm_agent` resolves.
"""

import sys
from pathlib import Path

_COMMON = str(Path(__file__).resolve().parent.parent / "common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)
