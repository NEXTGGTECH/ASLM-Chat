# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import unittest
from pathlib import Path

# Package root of mcp-sandbox; its tmp directory is put on sys.path at import time.
ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp"
if str(TMP) not in sys.path:
    sys.path.insert(0, str(TMP))

# Throwaway host workspace for these tests. The environment is configured at import time, before the
# sandbox modules are imported, so they run natively in a "container" with the job spool inside the workspace.
WORKSPACE = ROOT / ".test_workspace_jobs"
os.environ["SANDBOX_HOST_WORKSPACE"] = str(WORKSPACE)
os.environ["SANDBOX_IN_CONTAINER"] = "1"
os.environ["SANDBOX_COMMAND_USER"] = ""
os.environ["SANDBOX_JOB_ROOT"] = str(WORKSPACE / ".sandbox_jobs")

from sandbox.config import BACKGROUND_TIMEOUT_THRESHOLD  # noqa: E402
from sandbox.exec import should_use_background, _exec_bash_native_background, job_root  # noqa: E402
from sandbox.jobs import JOB_REGISTRY  # noqa: E402
from sandbox.workspace import task_root  # noqa: E402
from sandbox.supervisor import _cleanup_orphaned_job_dirs  # noqa: E402


# Return the directory that holds the background job spool directories.
def _jobs_root() -> Path:
    return job_root()


# Return True when a native bash can be started on this machine.
def _native_bash_available() -> bool:
    try:
        result = subprocess.run(["bash", "-lc", "true"], capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        # Any failure to start bash (not installed, not executable, timeout) means it is unavailable.
        return False


# Cover the should_use_background decision logic.
class TestShouldUseBackground(unittest.TestCase):
    # Verify that mode "never" never backgrounds, even with a long timeout.
    def test_never_mode_always_false(self):
        self.assertFalse(should_use_background("sleep 100", 600, "never"))

    # Verify that mode "always" always backgrounds, even for a trivial command with a short timeout.
    def test_always_mode_always_true(self):
        self.assertTrue(should_use_background("echo hi", 1, "always"))

    # Verify that in auto mode a timeout equal to the background threshold triggers backgrounding.
    def test_auto_high_timeout_triggers_background(self):
        self.assertTrue(
            should_use_background("echo hi", BACKGROUND_TIMEOUT_THRESHOLD, "auto")
        )

    # Verify that in auto mode a timeout one below the background threshold does not trigger backgrounding.
    def test_auto_low_timeout_no_background(self):
        self.assertFalse(
            should_use_background("echo hi", BACKGROUND_TIMEOUT_THRESHOLD - 1, "auto")
        )

    # Verify that in auto mode long-running commands (pip, apt-get, pytest) are backgrounded even with a
    # 1 second timeout.
    def test_auto_long_running_pattern_triggers_background(self):
        self.assertTrue(should_use_background("pip install numpy", 1, "auto"))
        self.assertTrue(should_use_background("apt-get update", 1, "auto"))
        self.assertTrue(should_use_background("pytest tests/", 1, "auto"))

    # Verify that in auto mode a plain command with a 5 second timeout stays in the foreground.
    def test_auto_plain_command_low_timeout_no_background(self):
        self.assertFalse(should_use_background("python script.py", 5, "auto"))


# Cover that job directories of synchronously completed commands are deleted.
@unittest.skipIf(not _native_bash_available(), "native bash is not available in this test environment")
class TestJobDirCleanupOnSyncCompletion(unittest.TestCase):
    # Reset the job registry and recreate an empty workspace with its task root.
    def setUp(self):
        JOB_REGISTRY.reset()
        shutil.rmtree(WORKSPACE, ignore_errors=True)
        shutil.rmtree(_jobs_root(), ignore_errors=True)
        task_root().mkdir(parents=True, exist_ok=True)

    # Reset the job registry and remove the workspace.
    def tearDown(self):
        JOB_REGISTRY.reset()
        shutil.rmtree(WORKSPACE, ignore_errors=True)

    # Verify that a command that finishes quickly exits with 0, returns its output and leaves no
    # .sandbox_jobs entry.
    def test_quick_command_no_leftover_dir(self):
        result = _exec_bash_native_background("echo hello", timeout_s=10)
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("hello", result["stdout"])
        self.assertFalse(_jobs_root().exists() and any(_jobs_root().iterdir()),
                         ".sandbox_jobs should be empty after sync completion")

    # Verify that after synchronous completion the job is marked done and its directory is deleted (or no
    # job is registered at all).
    def test_job_not_in_registry_after_dir_cleanup(self):
        result = _exec_bash_native_background("echo hi", timeout_s=10)
        self.assertEqual(result["exit_code"], 0)
        job_id = result.get("job_id")
        if job_id:
            job = JOB_REGISTRY.get(job_id)
            self.assertEqual(job.status, "done")
            if job.host_job_dir:
                self.assertFalse(job.host_job_dir.exists(),
                                 "job dir must be deleted after sync completion")
        else:
            self.assertFalse(JOB_REGISTRY.list_jobs())

    # Verify that five consecutive quick commands leave no job directories behind.
    def test_multiple_quick_commands_no_accumulation(self):
        for i in range(5):
            _exec_bash_native_background(f"echo cmd{i}", timeout_s=10)
        dirs = list(_jobs_root().iterdir()) if _jobs_root().exists() else []
        self.assertEqual(len(dirs), 0, f"Expected 0 dirs, found {len(dirs)}: {dirs}")

    # Verify that a command failing with exit code 1 also leaves no job directory behind.
    def test_failed_command_dir_also_cleaned(self):
        result = _exec_bash_native_background("exit 1", timeout_s=10)
        self.assertEqual(result["exit_code"], 1)
        dirs = list(_jobs_root().iterdir()) if _jobs_root().exists() else []
        self.assertEqual(len(dirs), 0, "Dir should be gone even after non-zero exit")


# Cover that commands which time out keep their job directory for incremental polling.
@unittest.skipIf(not _native_bash_available(), "native bash is not available in this test environment")
class TestJobDirKeptForTrulyBackgrounded(unittest.TestCase):
    # Reset the job registry and recreate an empty workspace with its task root.
    def setUp(self):
        JOB_REGISTRY.reset()
        shutil.rmtree(WORKSPACE, ignore_errors=True)
        shutil.rmtree(_jobs_root(), ignore_errors=True)
        task_root().mkdir(parents=True, exist_ok=True)

    # Reset the job registry and remove the workspace.
    def tearDown(self):
        JOB_REGISTRY.reset()
        shutil.rmtree(WORKSPACE, ignore_errors=True)

    # Verify that a command exceeding timeout_s returns error_type "backgrounded" and keeps its job
    # directory.
    def test_timed_out_job_keeps_dir(self):
        result = _exec_bash_native_background("sleep 30", timeout_s=1)
        self.assertIsNone(result["exit_code"])
        self.assertEqual(result.get("error_type"), "backgrounded")
        job_id = result.get("job_id")
        self.assertIsNotNone(job_id)
        job = JOB_REGISTRY.get(job_id)
        self.assertIsNotNone(job.host_job_dir)
        self.assertTrue(job.host_job_dir.exists(),
                        "Dir must be kept for a truly backgrounded job")

        # Cleanup the lingering process
        if job.process:
            try:
                import signal
                os.killpg(job.process.pid, signal.SIGKILL)
            except Exception:
                job.process.kill()


# Cover that purge_stale removes registry entries together with their directories on disk.
class TestPurgeStaleFilesystemCleanup(unittest.TestCase):
    # Reset the job registry and recreate an empty workspace with its task root.
    def setUp(self):
        JOB_REGISTRY.reset()
        shutil.rmtree(WORKSPACE, ignore_errors=True)
        shutil.rmtree(_jobs_root(), ignore_errors=True)
        task_root().mkdir(parents=True, exist_ok=True)

    # Reset the job registry and remove the workspace.
    def tearDown(self):
        JOB_REGISTRY.reset()
        shutil.rmtree(WORKSPACE, ignore_errors=True)

    # Verify that purge_stale removes a done job older than the TTL (backdated by two hours against a
    # one-hour TTL) and deletes its directory from disk.
    def test_purge_stale_removes_done_dirs(self):
        # Create a fake done job with a real dir
        job_dir = _jobs_root() / "bg_test001"
        job_dir.mkdir(parents=True, exist_ok=True)
        (job_dir / "stdout").write_text("out", encoding="utf-8")

        job = JOB_REGISTRY.create(
            command="echo test",
            cwd=".",
            runtime="native",
            host_job_dir=job_dir,
            job_id="bg_test001",
        )
        JOB_REGISTRY.mark_done(job.job_id, 0)

        # Force the job to appear stale by backdating started_at
        job.started_at = time.time() - 7200

        removed = JOB_REGISTRY.purge_stale(ttl_seconds=3600)
        self.assertEqual(removed, 1)
        self.assertFalse(job_dir.exists(), "purge_stale must delete the dir from disk")

    # Verify that purge_stale keeps a still-running job and its directory even with a TTL of 0.
    def test_purge_stale_keeps_running_jobs(self):
        job_dir = _jobs_root() / "bg_test002"
        job_dir.mkdir(parents=True, exist_ok=True)
        JOB_REGISTRY.create(
            command="sleep 999",
            cwd=".",
            runtime="native",
            host_job_dir=job_dir,
            job_id="bg_test002",
        )
        removed = JOB_REGISTRY.purge_stale(ttl_seconds=0)
        self.assertEqual(removed, 0)
        self.assertTrue(job_dir.exists(), "Running job dir must not be removed")


# Cover that _cleanup_orphaned_job_dirs removes leftover job directories on supervisor startup.
class TestStartupOrphanCleanup(unittest.TestCase):
    # Reset the job registry and recreate an empty workspace with its task root.
    def setUp(self):
        JOB_REGISTRY.reset()
        shutil.rmtree(WORKSPACE, ignore_errors=True)
        shutil.rmtree(_jobs_root(), ignore_errors=True)
        task_root().mkdir(parents=True, exist_ok=True)

    # Reset the job registry and remove the workspace.
    def tearDown(self):
        JOB_REGISTRY.reset()
        shutil.rmtree(WORKSPACE, ignore_errors=True)

    # Verify that orphaned job directories left over from previous sessions are all removed.
    def test_cleans_all_dirs_on_startup(self):
        for i in range(3):
            d = _jobs_root() / f"bg_orphan{i:03d}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "stdout").write_text("stale output", encoding="utf-8")

        self.assertEqual(len(list(_jobs_root().iterdir())), 3)
        _cleanup_orphaned_job_dirs()
        remaining = list(_jobs_root().iterdir()) if _jobs_root().exists() else []
        self.assertEqual(len(remaining), 0, f"All orphans must be gone, found: {remaining}")

    # Verify that the cleanup raises no error when the .sandbox_jobs directory does not exist yet.
    def test_noop_when_no_jobs_dir(self):
        jobs_root = _jobs_root()
        if jobs_root.exists():
            shutil.rmtree(jobs_root)

        # Must not raise.
        _cleanup_orphaned_job_dirs()


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main(verbosity=2)
