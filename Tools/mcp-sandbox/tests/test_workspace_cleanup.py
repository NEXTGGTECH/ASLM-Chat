# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
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
os.environ.setdefault("SANDBOX_HOST_WORKSPACE", str(ROOT / ".test_workspace_cleanup"))

import sandbox.cleanup as cleanup_mod  # noqa: E402


# Cover the workspace cleanup: staging root entries into a tmp batch and recycling expired batches.
class WorkspaceCleanupTests(unittest.TestCase):
    # Create a temporary _sandbox task root and make the cleanup module use it.
    def setUp(self) -> None:
        self.tmpdir = tempfile.TemporaryDirectory()
        self.task_root = Path(self.tmpdir.name) / "_sandbox"
        self.task_root.mkdir(parents=True, exist_ok=True)
        self.task_root_patch = patch.object(cleanup_mod, "task_root", return_value=self.task_root)
        self.task_root_patch.start()

    # Stop the task root patch and delete the temporary directory.
    def tearDown(self) -> None:
        self.task_root_patch.stop()
        self.tmpdir.cleanup()

    # Verify that staging moves all root entries (files and directories) into one batch under tmp, writes
    # the state file and the batch metadata, and leaves nothing behind in the root.
    def test_stage_workspace_to_tmp_moves_root_entries_into_one_batch(self) -> None:
        (self.task_root / "notes.txt").write_text("hello\n", encoding="utf-8")
        (self.task_root / "repo").mkdir()
        (self.task_root / "repo" / "README.md").write_text("readme\n", encoding="utf-8")

        batch = cleanup_mod.stage_workspace_to_tmp()

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.parent, self.task_root / "tmp")
        self.assertTrue((batch / "notes.txt").is_file())
        self.assertTrue((batch / "repo" / "README.md").is_file())
        self.assertFalse((self.task_root / "notes.txt").exists())
        self.assertFalse((self.task_root / "repo").exists())
        self.assertTrue((self.task_root / cleanup_mod.STATE_FILENAME).is_file())
        self.assertTrue((batch / cleanup_mod.BATCH_METADATA).is_file())

    # Verify that staging leaves the Skills directory in the root and moves only the other entries.
    def test_stage_workspace_to_tmp_keeps_skills_root(self) -> None:
        (self.task_root / "Skills").mkdir()
        (self.task_root / "Skills" / "SKILL.md").write_text("skill\n", encoding="utf-8")
        (self.task_root / "notes.txt").write_text("hello\n", encoding="utf-8")

        batch = cleanup_mod.stage_workspace_to_tmp()

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertTrue((self.task_root / "Skills" / "SKILL.md").is_file())
        self.assertFalse((batch / "Skills").exists())
        self.assertTrue((batch / "notes.txt").is_file())

    # Verify that staging is skipped (no batch, files untouched, no tmp directory) while a background job
    # is running.
    def test_stage_workspace_to_tmp_skips_when_background_job_running(self) -> None:
        (self.task_root / "notes.txt").write_text("hello\n", encoding="utf-8")

        with patch.object(cleanup_mod, "_has_running_background_jobs", return_value=True):
            batch = cleanup_mod.stage_workspace_to_tmp()

        self.assertIsNone(batch)
        self.assertTrue((self.task_root / "notes.txt").is_file())
        self.assertFalse((self.task_root / "tmp").exists())

    # Verify that a tmp symlink pointing outside the task root is replaced by a real directory before
    # staging, so nothing is written to the symlink target.
    @unittest.skipIf(
        sys.platform == "win32",
        "Symlink creation requires elevated privileges on Windows",
    )
    def test_stage_workspace_replaces_tmp_symlink(self) -> None:
        outside = Path(self.tmpdir.name) / "outside"
        outside.mkdir()
        (self.task_root / "notes.txt").write_text("hello\n", encoding="utf-8")
        (self.task_root / "tmp").symlink_to(outside, target_is_directory=True)

        batch = cleanup_mod.stage_workspace_to_tmp()

        self.assertIsNotNone(batch)
        assert batch is not None
        self.assertEqual(batch.parent, self.task_root / "tmp")
        self.assertFalse((self.task_root / "tmp").is_symlink())
        self.assertTrue((batch / "notes.txt").is_file())
        self.assertEqual(list(outside.iterdir()), [])

    # Verify that a batch staged longer ago than the recycle time is sent to the platform trash backend
    # and reported as recycled.
    def test_recycle_due_tmp_batches_uses_trash_backend_for_expired_batches(self) -> None:
        # Arrange: a batch staged 120 seconds ago, against a recycle time of 10 seconds.
        batch = self.task_root / "tmp" / "idle-old"
        batch.mkdir(parents=True)
        (batch / "old.txt").write_text("old\n", encoding="utf-8")
        old_time = datetime.now(timezone.utc) - timedelta(seconds=120)
        cleanup_mod._write_json(
            batch / cleanup_mod.BATCH_METADATA,
            {"staged_at": old_time.isoformat()},
        )
        recycled: list[Path] = []

        # Record the batch and delete it instead of using the real platform trash.
        def fake_trash(path: Path) -> None:
            recycled.append(path)
            shutil.rmtree(path)

        with patch.object(cleanup_mod, "WORKSPACE_CLEANUP_RECYCLE_SECONDS", 10), patch.object(
            cleanup_mod,
            "_send_to_platform_trash",
            side_effect=fake_trash,
        ):
            result = cleanup_mod.recycle_due_tmp_batches()

        self.assertEqual(result, [batch])
        self.assertEqual(recycled, [batch])
        self.assertFalse(batch.exists())

    # Verify that a batch staged just now is kept when the recycle time is an hour: nothing is recycled
    # and the trash backend is not called.
    def test_recycle_due_tmp_batches_keeps_recent_batches(self) -> None:
        batch = self.task_root / "tmp" / "idle-recent"
        batch.mkdir(parents=True)
        cleanup_mod._write_json(
            batch / cleanup_mod.BATCH_METADATA,
            {"staged_at": datetime.now(timezone.utc).isoformat()},
        )

        with patch.object(cleanup_mod, "WORKSPACE_CLEANUP_RECYCLE_SECONDS", 3600), patch.object(
            cleanup_mod,
            "_send_to_platform_trash",
        ) as trash_mock:
            result = cleanup_mod.recycle_due_tmp_batches()

        self.assertEqual(result, [])
        trash_mock.assert_not_called()
        self.assertTrue(batch.exists())

    # Verify that recycling replaces a tmp symlink with a real directory without recycling anything or
    # touching the symlink target.
    @unittest.skipIf(
        sys.platform == "win32",
        "Symlink creation requires elevated privileges on Windows",
    )
    def test_recycle_due_tmp_batches_replaces_tmp_symlink(self) -> None:
        outside = Path(self.tmpdir.name) / "outside-recycle"
        outside.mkdir()
        (self.task_root / "tmp").symlink_to(outside, target_is_directory=True)

        with patch.object(cleanup_mod, "_send_to_platform_trash") as trash_mock:
            result = cleanup_mod.recycle_due_tmp_batches()

        self.assertEqual(result, [])
        trash_mock.assert_not_called()
        self.assertFalse((self.task_root / "tmp").is_symlink())
        self.assertTrue((self.task_root / "tmp").is_dir())
        self.assertEqual(list(outside.iterdir()), [])


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
