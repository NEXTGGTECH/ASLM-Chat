# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import unittest
import sys
from pathlib import Path

# Make the sibling helpers module importable regardless of how pytest sets up rootdir and sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import reset_task_root  # noqa: E402


# Cover a smoke check of the jobs and edit feature imports.
class FeatureJobsEditSmokeTests(unittest.TestCase):
    # Verify that the sandbox workspace, container and API modules import and expose their entry points
    # without a docker daemon.
    def test_imports_without_docker(self) -> None:
        task_root = reset_task_root()

        from sandbox import workspace
        from sandbox import container
        from sandbox.api import handle_tool

        self.assertTrue(task_root.exists())
        self.assertTrue(callable(handle_tool))
        self.assertTrue(callable(workspace.edit))
        self.assertTrue(callable(container.exec_bash))


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
