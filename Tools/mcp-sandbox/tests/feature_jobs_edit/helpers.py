# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


# Package root of mcp-sandbox and its sandbox source directory; the latter is put on sys.path at import
# time so the feature tests can import the sandbox package.
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Point the sandbox at a throwaway workspace at import time unless the environment already sets one.
os.environ.setdefault("SANDBOX_HOST_WORKSPACE", str(ROOT / ".test_workspace"))


# Clear task_root children and return the task_root path.
def reset_task_root():
    from sandbox import workspace

    task_root = workspace.task_root()
    task_root.mkdir(parents=True, exist_ok=True)
    for child in list(task_root.iterdir()):
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
        except PermissionError:
            # Skip entries still locked by another process; clearing the task root is best effort.
            pass
    return task_root
