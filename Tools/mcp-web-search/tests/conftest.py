# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import sys
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]

# Make `core.*` and `custom_domains.*` importable regardless of how pytest
# is invoked (console script vs `python -m pytest`).
entry = str(TOOL_ROOT)
if entry not in sys.path:
    sys.path.insert(0, entry)
