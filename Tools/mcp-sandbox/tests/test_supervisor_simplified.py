# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import shutil
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import patch

# Package root of mcp-sandbox and its sandbox source directory; the latter is put on sys.path at import
# time so the sandbox package can be imported.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Throwaway host workspace, set at import time before the sandbox modules are imported.
_FAKE_WORKSPACE = ROOT / ".test_workspace"
os.environ["SANDBOX_HOST_WORKSPACE"] = str(_FAKE_WORKSPACE)

import sandbox.workspace as workspace_mod  # noqa: E402
from sandbox.api import handle_tool  # noqa: E402
from sandbox.exec import _truncate  # noqa: E402
from sandbox.workspace import find, resolve_model_path, task_root  # noqa: E402


# Recreate an empty task root: delete it (retrying up to three times, as the removal can transiently fail
# on Windows) and create it again.
def _setup_task_root() -> None:
    tr = task_root()
    if tr.exists():
        for attempt in range(3):
            try:
                shutil.rmtree(tr)
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.1)
    tr.mkdir(parents=True, exist_ok=True)


# Cover the workspace helper functions find and resolve_model_path.
class WorkspaceHelpersTests(unittest.TestCase):
    # Start every test with an empty task root.
    def setUp(self) -> None:
        _setup_task_root()

    # Verify that find with type_filter="file" returns files and omits directories.
    def test_find_type_filter_files_only(self) -> None:
        tr = task_root()
        (tr / "afile.txt").write_text("hello", encoding="utf-8")
        (tr / "asubdir").mkdir(exist_ok=True)

        result = find(path=".", type_filter="file", max_results=50)
        matches = result.get("result", {}).get("matches", [])
        names = [entry.get("path", "") for entry in matches]

        self.assertTrue(any("afile.txt" in name for name in names))
        self.assertFalse(any("asubdir" in name for name in names))

    # Verify that outside a container an absolute host path such as /etc/passwd is rejected with ValueError.
    def test_resolve_model_path_rejects_host_absolute_paths(self) -> None:
        with patch.object(workspace_mod, "IN_CONTAINER", False):
            with self.assertRaises(ValueError):
                resolve_model_path("/etc/passwd")


# Cover which bash commands the supervisor handles itself and which fall through to real bash, plus output
# truncation.
class SupervisorContractTests(unittest.TestCase):
    # Start every test with an empty task root.
    def setUp(self) -> None:
        _setup_task_root()

    # Verify that a plain `cat file` is handled by the supervisor (routed) and returns the file content.
    def test_plain_cat_single_file_is_supervised(self) -> None:
        handle_tool("write", {"path": "notes.txt", "content": "alpha\nbeta\n"})
        result = handle_tool("bash", {"command": "cat notes.txt"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["stdout"], "alpha\nbeta\n")
        self.assertTrue(result["result"].get("routed", False))

    # Verify that `cat -n file` (cat with flags) falls through to real bash and is not routed.
    def test_cat_with_flags_falls_through_to_real_bash(self) -> None:
        with patch(
            "sandbox.api.exec_bash",
            return_value={
                "exit_code": 0,
                "stdout": "real bash\n",
                "stderr": "",
                "error": None,
                "elapsed_ms": 1,
                "truncated": False,
                "cwd": ".",
            },
        ) as exec_mock:
            result = handle_tool("bash", {"command": "cat -n notes.txt"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["stdout"], "real bash\n")
        self.assertFalse(result["result"].get("routed", False))
        exec_mock.assert_called_once()

    # Verify that grep, find and a mutating command (mkdir) each run through real bash and are not routed.
    def test_grep_find_and_mutations_use_real_bash(self) -> None:
        for command in ("grep alpha notes.txt", "find . -name '*.py'", "mkdir -p deep"):
            with patch(
                "sandbox.api.exec_bash",
                return_value={
                    "exit_code": 0,
                    "stdout": "real bash\n",
                    "stderr": "",
                    "error": None,
                    "elapsed_ms": 1,
                    "truncated": False,
                    "cwd": ".",
                },
            ) as exec_mock:
                result = handle_tool("bash", {"command": command})

            self.assertTrue(result["ok"])
            self.assertFalse(result["result"].get("routed", False), command)
            exec_mock.assert_called_once()

    # Verify that a plain cat of a large file is routed and returns a preview with a file header and
    # head and tail sections.
    def test_large_cat_returns_preview(self) -> None:
        content = "\n".join(f"line {index}" for index in range(5000))
        handle_tool("write", {"path": "big.log", "content": content})

        result = handle_tool("bash", {"command": "cat big.log"})
        stdout = result["result"]["stdout"]

        self.assertTrue(result["ok"])
        self.assertTrue(result["result"].get("routed", False))
        self.assertIn("-- big.log", stdout)
        self.assertIn("[head:", stdout)
        self.assertIn("[tail:", stdout)

    # Verify that truncating a 140000-character output keeps its beginning and end and inserts a visible
    # "[output truncated:" marker.
    def test_truncate_keeps_head_and_tail_with_visible_marker(self) -> None:
        value = "A" * 70000 + "B" * 70000
        output, truncated = _truncate(value)

        self.assertTrue(truncated)
        self.assertTrue(output.startswith("A"))
        self.assertIn("[output truncated:", output)
        self.assertTrue(output.endswith("B" * 100))


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
