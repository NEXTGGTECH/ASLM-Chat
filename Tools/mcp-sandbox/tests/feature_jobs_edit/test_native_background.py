# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Make the sibling helpers module importable regardless of how pytest sets up rootdir and sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import reset_task_root  # noqa: E402
import sandbox.container as container_mod  # noqa: E402
import sandbox.exec as exec_mod  # noqa: E402


# Fake subprocess.Popen object for a long-running command that never exits on its own.
class BackgroundProcess:
    # Create a process with a writable stdin, a fixed pid and no exit code yet (still running).
    def __init__(self) -> None:
        self.stdin = io.StringIO()
        self.pid = 4321
        self.returncode = None

    # Return the exit code, or None while the fake process is still running.
    def poll(self):
        return self.returncode

    # Mark the fake process as killed by SIGKILL.
    def kill(self) -> None:
        self.returncode = -9


# Cover native (non-docker) background jobs: backgrounding on timeout and finalizing the exit code.
class NativeBackgroundTests(unittest.TestCase):
    # Start every test with an empty task root and an empty job registry.
    def setUp(self) -> None:
        reset_task_root()
        container_mod.JOB_REGISTRY.reset()

    # Verify that a native command exceeding its timeout is reported as backgrounded with a bg_ job id
    # and that its process group is not killed.
    def test_native_timeout_backgrounds_without_killing_process_group(self) -> None:
        fake = BackgroundProcess()

        # Fake clock that jumps past the 1 second timeout on the second reading.
        times = iter([100.0, 101.1, 101.2])

        with patch.object(exec_mod.subprocess, "Popen", return_value=fake), \
             patch.object(exec_mod, "_kill_process_group") as killpg_mock, \
             patch.object(exec_mod.time, "time", side_effect=lambda: next(times, 101.2)):
            result = exec_mod._exec_bash_native("sleep 60", timeout_s=1, background="always")

        self.assertIsNone(result["exit_code"])
        self.assertEqual(result["error_type"], "backgrounded")
        self.assertRegex(result["job_id"], r"^bg_[0-9a-f]{8}$")
        killpg_mock.assert_not_called()

    # Verify that bringing a backgrounded native job to the foreground after its process exits reports
    # the job as done with the exit code and the new stdout.
    def test_foreground_native_job_finalizes_exit_code(self) -> None:
        fake = BackgroundProcess()

        # Fake clock that jumps past the 1 second timeout on the second reading.
        times = iter([100.0, 101.1, 101.2])

        with patch.object(exec_mod.subprocess, "Popen", return_value=fake), \
             patch.object(exec_mod.time, "time", side_effect=lambda: next(times, 101.2)):
            result = exec_mod._exec_bash_native("pytest", timeout_s=1, background="always")

        # Let the process finish after it was backgrounded: write its output and set exit code 0.
        job = container_mod.JOB_REGISTRY.get(result["job_id"])
        (job.host_job_dir / "stdout").write_text("done\n", encoding="utf-8")
        fake.returncode = 0

        status = container_mod.foreground_background_job(job.job_id)

        self.assertEqual(status["status"], "done")
        self.assertEqual(status["exit_code"], 0)
        self.assertEqual(status["new_stdout"], "done\n")


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
