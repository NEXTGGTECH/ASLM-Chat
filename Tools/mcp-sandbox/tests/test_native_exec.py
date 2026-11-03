# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import io
import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Package root of mcp-sandbox and its sandbox source directory; the latter is put on sys.path at import
# time so the sandbox package can be imported.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Point the sandbox at a throwaway workspace at import time unless the environment already sets one.
os.environ.setdefault("SANDBOX_HOST_WORKSPACE", str(ROOT / ".test_workspace"))

from sandbox import workspace  # noqa: E402
import sandbox.exec as exec_mod  # noqa: E402


# Minimal fake Popen for native exec tests.
class TextProcess:
    # Create the process with in-memory text streams, the given output and exit code (None means still
    # running).
    def __init__(self, *, stdout: str = "", stderr: str = "", returncode: int | None = 0) -> None:
        self.stdin = io.StringIO()
        self.stdout = io.StringIO(stdout)
        self.stderr = io.StringIO(stderr)
        self.returncode = returncode
        self.pid = 1234

    # Return the exit code, or None while the fake process is still running.
    def poll(self) -> int | None:
        return self.returncode

    # Mark the fake process as killed by SIGKILL.
    def kill(self) -> None:
        self.returncode = -9


# Cover native (non-docker) bash execution: working directory, process group, home directory and timeout.
class NativeExecTests(unittest.TestCase):
    # Create the task root if needed and remove everything in it.
    def setUp(self) -> None:
        self.task_root = workspace.task_root()
        self.task_root.mkdir(parents=True, exist_ok=True)
        for child in list(self.task_root.iterdir()):
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    # Verify that a native command starts in the task root as the leader of a new session (process group).
    def test_native_exec_uses_task_root_cwd_and_process_group(self) -> None:
        fake = TextProcess(stdout="ok\n")

        with patch.object(exec_mod.subprocess, "Popen", return_value=fake) as popen_mock:
            result = exec_mod._exec_bash_native("echo ok")

        self.assertEqual(result["stdout"], "ok\n")
        kwargs = popen_mock.call_args.kwargs
        self.assertEqual(kwargs["cwd"], str(self.task_root))
        self.assertTrue(kwargs["start_new_session"])

    # Verify that when a command user is configured, the process environment gets that user's HOME.
    def test_native_exec_sets_home_for_command_user(self) -> None:
        fake = TextProcess(stdout="ok\n")

        with (
            patch.object(exec_mod, "COMMAND_USER", "sandbox_user"),
            patch.object(exec_mod, "_command_user_env", return_value={"HOME": "/home/sandbox_user"}),
            patch.object(exec_mod.subprocess, "Popen", return_value=fake) as popen_mock,
        ):
            result = exec_mod._exec_bash_native("echo ok")

        self.assertEqual(result["stdout"], "ok\n")
        self.assertEqual(popen_mock.call_args.kwargs["env"]["HOME"], "/home/sandbox_user")

    # Verify that a command still running after its timeout is killed through its process group and
    # reported as timed out without an exit code.
    def test_native_timeout_kills_process_group_without_container_restart(self) -> None:
        fake = TextProcess(returncode=None)

        # Fake clock that jumps past the 1 second timeout on the second reading.
        times = iter([100.0, 101.1, 101.2])

        # Return the next scripted time, repeating the last one when the script runs out.
        def fake_time() -> float:
            return next(times, 101.2)

        with (
            patch.object(exec_mod.subprocess, "Popen", return_value=fake),
            patch.object(exec_mod, "_kill_process_group") as killpg_mock,
            patch.object(exec_mod.time, "sleep", return_value=None),
            patch.object(exec_mod.time, "time", side_effect=fake_time),
        ):
            result = exec_mod._exec_bash_native("sleep 60", timeout_s=1)

        self.assertIsNone(result["exit_code"])
        self.assertIn("timed out", result["error"])
        killpg_mock.assert_called_once_with(fake)


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
