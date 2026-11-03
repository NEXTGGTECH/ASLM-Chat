# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import sys
from pathlib import Path

# Make the Tools directory importable so tests can import the context_compression package.
TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))
