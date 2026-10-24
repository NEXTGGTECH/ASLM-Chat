# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import sys
from pathlib import Path


TOOL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = TOOL_ROOT.parent.parent
SANDBOX_SUPERVISOR = REPO_ROOT / "Tools" / "mcp-sandbox" / "supervisor"

# The deep research runner imports sandbox.temporal from the mcp-sandbox
# supervisor package; provide both roots explicitly.
for path in (REPO_ROOT, SANDBOX_SUPERVISOR):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
