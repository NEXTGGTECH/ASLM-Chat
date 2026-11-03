# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest


# Package root of mcp-sandbox with its supervisor and sandbox source directories.
PKG_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = PKG_ROOT / "supervisor"
SRC = PKG_ROOT / "src"

# Make the supervisor and sandbox packages importable before any test module is collected.
for path in (SUPERVISOR, SRC):
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)

# Provide a default host workspace at import time, so modules that read it while being imported work.
os.environ.setdefault("SANDBOX_HOST_WORKSPACE", str(PKG_ROOT))


# Pin SANDBOX_HOST_WORKSPACE to the package root for every test.
@pytest.fixture(autouse=True)
def _sandbox_host_workspace(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SANDBOX_HOST_WORKSPACE", str(PKG_ROOT))
