# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Make the sibling helpers module importable regardless of how pytest sets up rootdir and sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from helpers import reset_task_root  # noqa: E402
from sandbox.api import TOOLS, handle_tool  # noqa: E402


# Cover the edit and write tools of the sandbox API: line mode, match mode, argument validation and tool
# descriptions.
class LineEditApiTests(unittest.TestCase):
    # Start every test with an empty task root and keep its path.
    def setUp(self) -> None:
        self.task_root = reset_task_root()

    # Verify that the edit tool in lines mode replaces the given range, accepts a matching anchor and
    # reports the edited range.
    def test_handle_tool_edit_lines(self) -> None:
        handle_tool("write", {"path": "app.py", "content": "a\nb\nc\n"})

        result = handle_tool(
            "edit",
            {
                "path": "app.py",
                "mode": "lines",
                "range": "2:2",
                "content": "B",
                "anchor": "b",
            },
        )

        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual((self.task_root / "app.py").read_text(encoding="utf-8"), "a\nB\nc\n")
        self.assertEqual(result["result"]["r"], "2:2")

    # Verify that lines mode accepts the range as a [start, end] array and that the returned context
    # contains no "None" placeholder.
    def test_line_mode_array_range(self) -> None:
        handle_tool("write", {"path": "app.py", "content": "a\nb\nc\nd\n"})

        result = handle_tool(
            "edit",
            {
                "path": "app.py",
                "mode": "lines",
                "range": [2, 3],
                "content": "B\nC",
            },
        )

        self.assertTrue(result["ok"], result.get("error"))
        self.assertEqual((self.task_root / "app.py").read_text(encoding="utf-8"), "a\nB\nC\nd\n")
        self.assertNotIn("None", "\n".join(result["result"]["cx"]))

    # Verify that the legacy old_str/new_str edit without a mode still replaces the text and returns the
    # match and context fields.
    def test_legacy_match_mode_still_works(self) -> None:
        handle_tool("write", {"path": "app.txt", "content": "hello\n"})

        result = handle_tool(
            "edit",
            {
                "path": "app.txt",
                "old_str": "hello",
                "new_str": "world",
            },
        )

        self.assertTrue(result["ok"], result.get("error"))
        self.assertIn("m", result["result"])
        self.assertIn("cx", result["result"])
        self.assertEqual((self.task_root / "app.txt").read_text(encoding="utf-8"), "world\n")

    # Verify that an empty range string does not switch to line mode: the old_str/new_str edit still applies.
    def test_match_mode_ignores_empty_range(self) -> None:
        handle_tool("write", {"path": "app.txt", "content": "hello\n"})

        result = handle_tool(
            "edit",
            {
                "path": "app.txt",
                "old_str": "hello",
                "new_str": "world",
                "range": "",
            },
        )

        self.assertTrue(result["ok"], result.get("error"))
        self.assertIn("m", result["result"])
        self.assertIn("cx", result["result"])
        self.assertEqual((self.task_root / "app.txt").read_text(encoding="utf-8"), "world\n")

    # Verify that lines mode without a range is rejected with an invalid_arguments error.
    def test_line_mode_requires_range(self) -> None:
        handle_tool("write", {"path": "app.py", "content": "a\nb\nc\nd\n"})

        result = handle_tool(
            "edit",
            {
                "path": "app.py",
                "mode": "lines",
                "content": "B\nC",
            },
        )

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "invalid_arguments")

    # Verify that an edit with neither a lines range nor old_str/new_str is rejected with an
    # invalid_arguments error naming old_str.
    def test_match_mode_requires_old_and_new_strings(self) -> None:
        handle_tool("write", {"path": "app.txt", "content": "hello\n"})

        result = handle_tool("edit", {"path": "app.txt"})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "invalid_arguments")
        self.assertIn("old_str", result["error"]["message"])

    # Verify that lines mode with a range but no content is rejected with an invalid_arguments error
    # naming content.
    def test_line_mode_requires_range_and_content(self) -> None:
        handle_tool("write", {"path": "app.txt", "content": "hello\n"})

        result = handle_tool("edit", {"path": "app.txt", "mode": "lines", "range": "1:1"})

        self.assertFalse(result["ok"])
        self.assertEqual(result["error"]["type"], "invalid_arguments")
        self.assertIn("content", result["error"]["message"])

    # Verify that the edit tool description documents the range argument, 1-based lines and the
    # /workspace/_sandbox Linux paths.
    def test_edit_description_documents_lines_contract(self) -> None:
        edit_tool = next(tool for tool in TOOLS if tool["id"] == "edit")
        description = edit_tool["description"]

        self.assertIn("range", description)
        self.assertIn("1-based", description)
        self.assertIn("/workspace/_sandbox", description)
        self.assertIn("Linux paths", description)

    # Verify that the write tool description documents the /workspace/_sandbox Linux paths and that
    # Windows-style paths are rejected.
    def test_write_description_documents_workspace_relative_paths(self) -> None:
        write_tool = next(tool for tool in TOOLS if tool["id"] == "write")
        description = write_tool["description"]

        self.assertIn("/workspace/_sandbox", description)
        self.assertIn("Linux paths", description)
        self.assertIn("Windows-style paths are rejected", description)


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
