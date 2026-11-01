# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SUPERVISOR = ROOT / "supervisor"
for path in (SUPERVISOR, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("SANDBOX_WORKSPACE_CLEANUP_ENABLED", "1")
os.environ.setdefault("SANDBOX_WORKSPACE_CLEANUP_IDLE_SECONDS", "30")
os.environ.setdefault("SANDBOX_WORKSPACE_CLEANUP_RECYCLE_SECONDS", "30")
os.environ.setdefault("SANDBOX_WORKSPACE_CLEANUP_INTERVAL_SECONDS", "5")

from sandbox.api import handle_tool  # noqa: E402
import sandbox.cleanup as cleanup_mod  # noqa: E402
import sandbox.workspace as workspace_mod  # noqa: E402


@unittest.skipUnless(
    os.environ.get("RUN_SANDBOX_CLEANUP_TIMING") == "1",
    "Set RUN_SANDBOX_CLEANUP_TIMING=1 to run the real-time cleanup timing test.",
)
class WorkspaceCleanupTimingTests(unittest.TestCase):
    # Real-time idle staging and recycle with handle_tool write.
    def test_background_monitor_timing_with_real_tool_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            workspace_root = Path(tmpdir)
            task_root = workspace_root / "_sandbox"
            task_root.mkdir(parents=True, exist_ok=True)

            cleanup_mod._MONITOR_STARTED = False
            cleanup_mod._ACTIVE_CALLS = 0
            cleanup_mod._LAST_ACTIVITY_MONOTONIC = time.monotonic()

            ticks: list[float] = []
            recycled: list[str] = []
            original_run_cleanup_once = cleanup_mod.run_cleanup_once

            def recording_cleanup_once() -> None:
                ticks.append(time.monotonic())
                original_run_cleanup_once()

            def fake_trash(path: Path) -> None:
                recycled.append(str(path))
                shutil.rmtree(path)

            with patch.object(workspace_mod, "HOST_WORKSPACE", str(workspace_root)), patch.object(
                cleanup_mod,
                "task_root",
                return_value=task_root,
            ), patch.object(
                cleanup_mod,
                "run_cleanup_once",
                side_effect=recording_cleanup_once,
            ), patch.object(
                cleanup_mod,
                "_send_to_platform_trash",
                side_effect=fake_trash,
            ):
                start = time.monotonic()
                result = handle_tool(
                    "write",
                    {
                        "path": "timing-marker.txt",
                        "content": "written by real cleanup timing test\n",
                    },
                    {},
                )
                self.assertTrue(result.get("ok"), result)

                deadline = start + 80
                while time.monotonic() < deadline:
                    if len(ticks) >= 2 and recycled:
                        break
                    time.sleep(0.25)

                elapsed = time.monotonic() - start
                intervals = [
                    round(ticks[index] - ticks[index - 1], 3)
                    for index in range(1, len(ticks))
                ]
                tmp_entries = (
                    sorted(path.name for path in (task_root / "tmp").iterdir())
                    if (task_root / "tmp").exists()
                    else []
                )
                summary = {
                    "elapsed_seconds": round(elapsed, 3),
                    "cleanup_tick_offsets": [round(tick - start, 3) for tick in ticks],
                    "cleanup_tick_intervals": intervals,
                    "recycled_count": len(recycled),
                    "recycled_paths": recycled,
                    "root_entries": sorted(path.name for path in task_root.iterdir()),
                    "tmp_entries": tmp_entries,
                }

                self.assertGreaterEqual(len(ticks), 2, summary)
                self.assertTrue(intervals, summary)
                expected_interval = cleanup_mod.WORKSPACE_CLEANUP_INTERVAL_SECONDS
                self.assertTrue(
                    max(1, expected_interval - 1) <= intervals[0] <= expected_interval + 2,
                    summary,
                )
                self.assertEqual(len(recycled), 1, summary)
                self.assertFalse((task_root / "timing-marker.txt").exists(), summary)


