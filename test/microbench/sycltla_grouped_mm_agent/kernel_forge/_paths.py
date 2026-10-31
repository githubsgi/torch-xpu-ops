"""Make `ggemm_agent` importable regardless of how this package was imported."""
import sys
from pathlib import Path

_COMMON = str(Path(__file__).resolve().parent.parent / "common")
if _COMMON not in sys.path:
    sys.path.insert(0, _COMMON)
