# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Make the sibling helpers module importable regardless of how pytest sets up rootdir and sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import reset_task_root  # noqa: E402
import sandbox.docker_host as docker_host_mod  # noqa: E402

# Container-side spool directory of the background job id used by these tests.
TEST_JOB_DIR = "/workspace/.sandbox_jobs/bg_12345678"


# Minimal stand-in for the completed-process object returned by the patched _run_command.
class RunResult:
    # Store the fields the docker host code reads (stdout, stderr and the exit code).
    def __init__(self, stdout: str = "", stderr: str = "", returncode: int = 0) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


# Cover docker-runtime background jobs: backgrounding on timeout, reading spool output, killing and refreshing.
class DockerBackgroundTests(unittest.TestCase):
    # Start every test with an empty task root and an empty job registry.
    def setUp(self) -> None:
        reset_task_root()
        docker_host_mod.JOB_REGISTRY.reset()

    # Verify that a docker command exceeding its timeout is detached as a background job (setsid launch
    # script, partial stdout, bg_ job id, "backgrounded" error) without restarting the container.
    def test_docker_timeout_backgrounds_without_container_restart(self) -> None:
        # Arrange: a fake docker runner that records the scripts and answers the launch, status and spool reads.
        calls: list[list[str]] = []

        # Answer each shell script sent to the container by matching on what it does.
        def fake_run(args, timeout=30, **kwargs):
            calls.append(args)
            script = args[-1]
            if "setsid" in script and "job_dir/pid" in script:
                self.assertIn("export job_dir; if command -v setsid", script)
                self.assertIn("setsid bash -lc", script)
                self.assertIn("job_dir/pgid", script)
                self.assertIn("nohup bash -lc", script)
                self.assertNotIn("eval echo", script)
                self.assertIn("home_dollar='$HOME/'", script)
                return RunResult(stdout="9876\n")
            if "/status" in script and "exit_code" in script:
                return RunResult(stdout="running\n\n")
            if "/stdout" in script:
                return RunResult(stdout="partial\n")
            if "/stderr" in script:
                return RunResult(stdout="")
            return RunResult()

        # Fake clock that jumps past the 1 second timeout on the second reading.
        times = iter([100.0, 101.1, 101.2])

        # Act: run a long command with a 1 second timeout and forced background mode.
        with patch.object(docker_host_mod, "_ensure_container_running", return_value=(True, "ok")), \
             patch.object(docker_host_mod, "_run_command", side_effect=fake_run), \
             patch.object(docker_host_mod, "restart_container") as restart_mock, \
             patch.object(docker_host_mod.time, "time", side_effect=lambda: next(times, 101.2)):
            result = docker_host_mod._exec_bash_docker("pytest", timeout_s=1, background="always")

        # Assert: the job is reported as backgrounded with its partial output and the container stays up.
        self.assertEqual(result["error_type"], "backgrounded")
        self.assertEqual(result["stdout"], "partial\n")
        self.assertRegex(result["job_id"], r"^bg_[0-9a-f]{8}$")
        self.assertTrue(any("setsid bash -lc" in call[-1] for call in calls))
        restart_mock.assert_not_called()

    # Verify that bringing a docker job to the foreground reports its finished status, exit code and the
    # new stdout read from the spool.
    def test_foreground_docker_job_reads_status_and_output(self) -> None:
        job = docker_host_mod.JOB_REGISTRY.create(
            command="pytest",
            cwd=".",
            runtime="docker",
            pid=9876,
            container_job_dir=TEST_JOB_DIR,
            job_id="bg_12345678",
        )

        # Report the job as done with exit code 0 and answer the stdout and stderr spool reads.
        def fake_run(args, timeout=30, **kwargs):
            script = args[-1]
            if "/status" in script and "exit_code" in script:
                return RunResult(stdout="done\n0\n")
            if "/stdout" in script:
                return RunResult(stdout="ok\n")
            if "/stderr" in script:
                return RunResult(stdout="")
            return RunResult()

        with patch.object(docker_host_mod, "_run_command", side_effect=fake_run):
            status = docker_host_mod.foreground_background_job(job.job_id)

        self.assertEqual(status["status"], "done")
        self.assertEqual(status["exit_code"], 0)
        self.assertEqual(status["new_stdout"], "ok\n")

    # Verify that a spool holding one very long line is read through the bounded python reader instead of
    # a plain cat, and that the result is flagged as truncated.
    def test_foreground_docker_job_reads_long_single_line_with_bounded_reader(self) -> None:
        job = docker_host_mod.JOB_REGISTRY.create(
            command="python3 -c 'print(\"A\" * 120000)'",
            cwd=".",
            runtime="docker",
            pid=9876,
            container_job_dir=TEST_JOB_DIR,
            job_id="bg_12345678",
        )
        calls: list[str] = []

        # Truncation notice that the bounded reader inserts between the head and the tail of the output.
        marker = (
            "\n\n[output truncated while reading docker job spool: "
            "showed first 30000 bytes and last 30000 bytes of 120000 new bytes]\n\n"
        )

        # Answer the status read and the 120000-byte stdout spool through the bounded reader; a plain cat
        # of the stdout file fails the test.
        def fake_run(args, timeout=30, **kwargs):
            script = args[-1]
            calls.append(script)
            if "/status" in script and "exit_code" in script:
                return RunResult(stdout="done\n0\n")
            if "python3 -" in script and "/stdout" in script:
                self.assertNotIn("cat /workspace/.sandbox_jobs/bg_12345678/stdout", script)
                return RunResult(stdout=("A" * 100) + marker + ("B" * 100))
            if "python3 -" in script and "/stderr" in script:
                return RunResult(stdout="")
            if "wc -c" in script:
                return RunResult(stdout="120000\n")
            return RunResult()

        with patch.object(docker_host_mod, "_run_command", side_effect=fake_run):
            status = docker_host_mod.foreground_background_job(job.job_id)

        self.assertTrue(status["truncated"])
        self.assertIn("[output truncated while reading docker job spool:", status["new_stdout"])
        self.assertTrue(any("python3 -" in call and "/stdout" in call for call in calls))

    # Verify that killing a docker job reads the process group id from the container job directory and
    # sends TERM and then KILL to that group.
    def test_kill_docker_job_uses_container_pid_file(self) -> None:
        docker_host_mod.JOB_REGISTRY.create(
            command="pytest",
            cwd=".",
            runtime="docker",
            pid=9876,
            container_job_dir=TEST_JOB_DIR,
            job_id="bg_12345678",
        )
        calls: list[list[str]] = []

        # Record the container command and succeed.
        def fake_run(args, timeout=30, **kwargs):
            calls.append(args)
            return RunResult()

        with patch.object(docker_host_mod, "_run_command", side_effect=fake_run):
            result = docker_host_mod.kill_background_job("bg_12345678")

        self.assertEqual(result["status"], "killed")
        kill_scripts = [call[-1] for call in calls]
        self.assertTrue(any("pgid=$(cat" in script for script in kill_scripts))
        self.assertTrue(any("kill -TERM -- -\"$pgid\"" in script for script in kill_scripts))
        self.assertTrue(any("kill -KILL -- -\"$pgid\"" in script for script in kill_scripts))

    # Verify that refreshing a job still marked running, whose process group is no longer alive, marks it
    # done without an exit code.
    def test_refresh_marks_stale_running_docker_job_done(self) -> None:
        job = docker_host_mod.JOB_REGISTRY.create(
            command="pytest",
            cwd=".",
            runtime="docker",
            pid=9876,
            container_job_dir=TEST_JOB_DIR,
            job_id="bg_12345678",
        )

        # Report status running, no exit code, pid and pgid 9876 and alive=no.
        def fake_run(args, timeout=30, **kwargs):
            return RunResult(stdout="running\n\n9876\n9876\nno\n")

        with patch.object(docker_host_mod, "_run_command", side_effect=fake_run):
            refreshed = docker_host_mod._refresh_docker_job(job)

        self.assertEqual(refreshed.status, "done")
        self.assertIsNone(refreshed.exit_code)


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
