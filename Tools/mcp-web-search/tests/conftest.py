# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import sys
from pathlib import Path


# Root directory of the web search tool (the parent of this tests folder).
TOOL_ROOT = Path(__file__).resolve().parents[1]

# Make `core.*` and `custom_domains.*` importable regardless of how pytest
# is invoked (console script vs `python -m pytest`).
entry = str(TOOL_ROOT)
if entry not in sys.path:
    sys.path.insert(0, entry)

# Unit tests never download NLTK data; code paths fall back to data-free NLTK.
os.environ.setdefault("ASLM_NLTK_DOWNLOAD", "0")
