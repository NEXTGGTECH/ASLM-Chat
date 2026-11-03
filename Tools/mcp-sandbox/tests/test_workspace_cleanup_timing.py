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

# Package root of mcp-sandbox with its supervisor and sandbox source directories, put on sys.path at import
# time so the sandbox package can be imported.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SUPERVISOR = ROOT / "supervisor"
for path in (SUPERVISOR, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Enable the workspace cleanup with short idle, recycle and check intervals (seconds) at import time,
# before the sandbox modules read their configuration, unless the environment already sets them.
os.environ.setdefault("SANDBOX_WORKSPACE_CLEANUP_ENABLED", "1")
os.environ.setdefault("SANDBOX_WORKSPACE_CLEANUP_IDLE_SECONDS", "30")
os.environ.setdefault("SANDBOX_WORKSPACE_CLEANUP_RECYCLE_SECONDS", "30")
os.environ.setdefault("SANDBOX_WORKSPACE_CLEANUP_INTERVAL_SECONDS", "5")

from sandbox.api import handle_tool  # noqa: E402
import sandbox.cleanup as cleanup_mod  # noqa: E402
import sandbox.workspace as workspace_mod  # noqa: E402


# Cover the real-time behavior of the background cleanup monitor; it runs for about a minute, so it only runs
# when RUN_SANDBOX_CLEANUP_TIMING=1.
@unittest.skipUnless(
    os.environ.get("RUN_SANDBOX_CLEANUP_TIMING") == "1",
    "Set RUN_SANDBOX_CLEANUP_TIMING=1 to run the real-time cleanup timing test.",
)
class WorkspaceCleanupTimingTests(unittest.TestCase):
    # Verify with real time that after a file is written through handle_tool the background monitor ticks
    # at the configured interval, stages the idle workspace and recycles the batch exactly once, leaving
    # the written file out of the task root.
    def test_background_monitor_timing_with_real_tool_write(self) -> None:
        # Arrange: a temporary workspace and a fresh monitor state.
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

            # Record the time of each cleanup run and then run the real cleanup.
            def recording_cleanup_once() -> None:
                ticks.append(time.monotonic())
                original_run_cleanup_once()

            # Record the recycled batch and delete it instead of using the real platform trash.
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
                # Act: write a file through the tool, which starts the monitor, and wait up to 80 seconds
                # for at least two cleanup ticks and one recycled batch.
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

                # Collect the observed timings and the remaining directory contents for the assertion messages.
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

                # Assert: the first tick interval is close to the configured one, and the idle workspace was
                # recycled exactly once.
                self.assertGreaterEqual(len(ticks), 2, summary)
                self.assertTrue(intervals, summary)
                expected_interval = cleanup_mod.WORKSPACE_CLEANUP_INTERVAL_SECONDS
                self.assertTrue(
                    max(1, expected_interval - 1) <= intervals[0] <= expected_interval + 2,
                    summary,
                )
                self.assertEqual(len(recycled), 1, summary)
                self.assertFalse((task_root / "timing-marker.txt").exists(), summary)
