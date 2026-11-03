# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Package root of mcp-sandbox with its supervisor and sandbox source directories, put on sys.path at import
# time so the sandbox package can be imported.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SUPERVISOR = ROOT / "supervisor"
for path in (SUPERVISOR, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Point the sandbox at a throwaway workspace at import time unless the environment already sets one.
os.environ.setdefault("SANDBOX_HOST_WORKSPACE", str(ROOT / ".test_workspace_config"))

import sandbox.config as config_mod  # noqa: E402


# Cover the host workspace path validation of the sandbox config.
class ConfigValidationTests(unittest.TestCase):
    # Verify that a generic home directory is rejected as the workspace when DEFAULT_TASK_DIR is ".".
    def test_dot_task_dir_rejects_generic_home_workspace(self) -> None:
        with patch.object(config_mod, "DEFAULT_TASK_DIR", "."):
            self.assertFalse(config_mod._validate_workspace_path("/home/dima"))

    # Verify that a dedicated sandbox directory under home is accepted when DEFAULT_TASK_DIR is ".".
    def test_dot_task_dir_allows_dedicated_sandbox_workspace(self) -> None:
        with patch.object(config_mod, "DEFAULT_TASK_DIR", "."):
            self.assertTrue(config_mod._validate_workspace_path("/home/dima/mcp-sandbox"))

    # Verify that with a sub-directory task dir (_sandbox) a project directory under home stays valid.
    def test_subdir_task_dir_keeps_existing_workspace_validation(self) -> None:
        with patch.object(config_mod, "DEFAULT_TASK_DIR", "_sandbox"):
            self.assertTrue(config_mod._validate_workspace_path("/home/dima/project"))


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
