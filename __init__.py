from __future__ import annotations

import sys
from pathlib import Path

# Compatibility shim: allow imports from repository root without installation.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from trine_one_step import *  # noqa: F401,F403
