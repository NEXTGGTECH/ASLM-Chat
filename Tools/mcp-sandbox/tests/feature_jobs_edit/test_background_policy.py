# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Make the sibling helpers module importable regardless of how pytest sets up rootdir and sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import reset_task_root  # noqa: E402
from sandbox.api import handle_tool  # noqa: E402
import sandbox.exec as exec_mod  # noqa: E402


# Cover the policy that decides when a bash command runs as a background job, and how bash passes it on.
class BackgroundPolicyTests(unittest.TestCase):
    # Start every test with an empty task root.
    def setUp(self) -> None:
        reset_task_root()

    # Verify that auto mode keeps a short timeout and a simple command in the foreground.
    def test_auto_does_not_background_short_simple_command(self) -> None:
        self.assertFalse(exec_mod.should_use_background("echo ok", timeout_s=1, background="auto"))

    # Verify that auto mode backgrounds a command matching a long-running pattern (pytest) even with a
    # short timeout.
    def test_auto_backgrounds_long_running_pattern(self) -> None:
        self.assertTrue(exec_mod.should_use_background("pytest -x", timeout_s=1, background="auto"))

    # Verify that auto mode backgrounds a simple command once the timeout reaches the background
    # threshold (10 seconds).
    def test_auto_backgrounds_timeout_threshold(self) -> None:
        self.assertTrue(exec_mod.should_use_background("echo ok", timeout_s=10, background="auto"))

    # Verify that "never" and "always" override both the long-running pattern and the timeout in auto mode.
    def test_never_and_always_override_auto(self) -> None:
        self.assertFalse(exec_mod.should_use_background("pytest -x", timeout_s=60, background="never"))
        self.assertTrue(exec_mod.should_use_background("echo ok", timeout_s=1, background="always"))

    # Verify that the bash tool handler forwards the background argument unchanged to exec_bash.
    def test_api_passes_background_argument_to_exec(self) -> None:
        with patch(
            "sandbox.api.exec_bash",
            return_value={
                "exit_code": 0,
                "stdout": "ok\n",
                "stderr": "",
                "error": None,
                "elapsed_ms": 1,
                "truncated": False,
                "cwd": ".",
            },
        ) as exec_mock:
            result = handle_tool("bash", {"command": "echo ok | cat", "background": "never"})

        self.assertTrue(result["ok"])
        self.assertEqual(exec_mock.call_args.kwargs["background"], "never")

    # Verify that a plain "ls" is not intercepted by the supervisor but executed through exec_bash, so
    # the result is not marked as routed.
    def test_ls_goes_to_real_bash(self) -> None:
        with patch(
            "sandbox.api.exec_bash",
            return_value={
                "exit_code": 0,
                "stdout": "file.txt\n",
                "stderr": "",
                "error": None,
                "elapsed_ms": 1,
                "truncated": False,
                "cwd": ".",
            },
        ) as exec_mock:
            result = handle_tool("bash", {"command": "ls", "background": "always"})

        self.assertTrue(result["ok"])
        self.assertFalse(result["result"].get("routed", False))
        exec_mock.assert_called_once()


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
