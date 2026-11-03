# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# Make the browser agent package root importable so the tests can import its modules directly.
PKG_ROOT = Path(__file__).resolve().parents[1]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))


# Load mcp-server.py once per module under a private name (its file name has a hyphen, so it cannot be
# imported normally) and return the loaded bridge module for the contract tests.
@pytest.fixture(scope="module")
def bridge_module():
    bridge_path = PKG_ROOT / "mcp-server.py"
    spec = importlib.util.spec_from_file_location("browser_agent_mcp_bridge_test", bridge_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
