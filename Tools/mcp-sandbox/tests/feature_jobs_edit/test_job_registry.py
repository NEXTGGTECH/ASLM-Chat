# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the sibling helpers module importable regardless of how pytest sets up rootdir and sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import reset_task_root  # noqa: E402
from sandbox.jobs import JobRegistry  # noqa: E402


# Cover the background job registry: creation, incremental output reads and state transitions.
class JobRegistryTests(unittest.TestCase):
    # Reset the task root and prepare a job directory holding a stdout spool with "alpha" and an empty stderr.
    def setUp(self) -> None:
        self.task_root = reset_task_root()
        self.job_dir = self.task_root / "job"
        self.job_dir.mkdir()
        (self.job_dir / "stdout").write_text("alpha\n", encoding="utf-8")
        (self.job_dir / "stderr").write_text("", encoding="utf-8")

    # Verify that a created job gets a bg_ id and is listed as running with its pid.
    def test_create_and_list_job(self) -> None:
        registry = JobRegistry()

        job = registry.create(
            command="sleep 10",
            cwd=".",
            runtime="native",
            pid=123,
            host_job_dir=self.job_dir,
        )

        self.assertRegex(job.job_id, r"^bg_[0-9a-f]{8}$")
        listing = registry.list_jobs()
        self.assertEqual(listing[0]["job_id"], job.job_id)
        self.assertEqual(listing[0]["status"], "running")
        self.assertEqual(listing[0]["pid"], 123)

    # Verify that incremental reads return only the output added since the previous read, while a
    # non-incremental read returns the whole spool.
    def test_reads_output_incrementally(self) -> None:
        registry = JobRegistry()
        job = registry.create(
            command="pytest",
            cwd=".",
            runtime="native",
            pid=123,
            host_job_dir=self.job_dir,
        )

        first = registry.read_output(job.job_id, "stdout", incremental=True)
        (self.job_dir / "stdout").write_text("alpha\nbeta\n", encoding="utf-8")
        second = registry.read_output(job.job_id, "stdout", incremental=True)
        full = registry.read_output(job.job_id, "stdout", incremental=False)

        self.assertEqual(first, "alpha\n")
        self.assertEqual(second, "beta\n")
        self.assertEqual(full, "alpha\nbeta\n")

    # Verify that a job can be marked done (keeping the exit code) and then marked killed.
    def test_state_transitions(self) -> None:
        registry = JobRegistry()
        job = registry.create(
            command="pytest",
            cwd=".",
            runtime="native",
            pid=123,
            host_job_dir=self.job_dir,
        )

        registry.mark_done(job.job_id, exit_code=0)
        self.assertEqual(registry.get(job.job_id).status, "done")
        self.assertEqual(registry.get(job.job_id).exit_code, 0)

        registry.mark_killed(job.job_id)
        self.assertEqual(registry.get(job.job_id).status, "killed")


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
