# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the sibling helpers module importable regardless of how pytest sets up rootdir and sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import reset_task_root  # noqa: E402
from sandbox import workspace  # noqa: E402
from sandbox.responses import SandboxToolError  # noqa: E402


# Cover the anchor check of the line-range edit.
class LineEditAnchorTests(unittest.TestCase):
    # Start every test with an empty task root and keep its path.
    def setUp(self) -> None:
        self.task_root = reset_task_root()

    # Verify that a line edit proceeds when the first line of the range equals the anchor.
    def test_accepts_matching_anchor(self) -> None:
        (self.task_root / "app.py").write_text("def old():\n    return 1\n", encoding="utf-8")

        workspace.edit_lines("app.py", "1:2", "def new():\n    return 2", anchor="def old():")

        self.assertEqual((self.task_root / "app.py").read_text(encoding="utf-8"), "def new():\n    return 2\n")

    # Verify that an anchor mismatch raises anchor_mismatch and suggests the nearby range (3:4) where the
    # anchor line really is.
    def test_anchor_mismatch_suggests_nearby_range(self) -> None:
        (self.task_root / "app.py").write_text("import os\n\ndef target():\n    return 1\n", encoding="utf-8")

        with self.assertRaises(SandboxToolError) as raised:
            workspace.edit_lines("app.py", "1:2", "replacement", anchor="def target():")

        self.assertEqual(raised.exception.error_type, "anchor_mismatch")
        self.assertEqual(raised.exception.result["suggestion"], "Use range 3:4.")
        self.assertEqual(raised.exception.result["actual_line_1"], "import os")

    # Verify that an anchor mismatch raises anchor_mismatch without a suggestion when the anchor text
    # does not occur in the file.
    def test_anchor_mismatch_without_suggestion(self) -> None:
        (self.task_root / "app.py").write_text("import os\nvalue = 1\n", encoding="utf-8")

        with self.assertRaises(SandboxToolError) as raised:
            workspace.edit_lines("app.py", "1:1", "replacement", anchor="def missing():")

        self.assertEqual(raised.exception.error_type, "anchor_mismatch")
        self.assertNotIn("suggestion", raised.exception.result)


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
