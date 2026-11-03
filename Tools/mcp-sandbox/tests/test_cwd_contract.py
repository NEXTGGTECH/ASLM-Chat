# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import shutil
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Package root of mcp-sandbox and its sandbox source directory; the latter is put on sys.path at import
# time so the sandbox package can be imported.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Use the package root as host workspace at import time unless the environment already sets one.
os.environ.setdefault("SANDBOX_HOST_WORKSPACE", str(ROOT))

from sandbox import workspace  # noqa: E402
from sandbox.api import handle_tool  # noqa: E402
from sandbox.config import DEFAULT_TASK_DIR  # noqa: E402


# Build a fake exec_bash return value.
def _mock_exec(stdout: str = "", stderr: str = "", exit_code: int = 0, cwd: str = "."):
    return {
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "error": None if exit_code == 0 else f"Exit code: {exit_code}",
        "elapsed_ms": 1,
        "truncated": False,
        "cwd": cwd,
    }


# Cover how the cwd argument of the bash tool is resolved, reported and isolated between calls.
class CwdContractTests(unittest.TestCase):
    # Create the task root if needed and empty it.
    def setUp(self) -> None:
        self.task_root = workspace.task_root()
        self.task_root.mkdir(parents=True, exist_ok=True)
        for child in list(self.task_root.iterdir()):
            try:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except PermissionError:
                # Skip entries still locked by another process; clearing the task root is best effort.
                pass

    # Working directory reported by pwd

    # Verify that pwd without a cwd is passed to exec_bash and its output, the container path of the task
    # root, is returned.
    def test_pwd_at_root_reports_task_root(self) -> None:
        with patch("sandbox.api.exec_bash", return_value=_mock_exec(stdout=f"/workspace/{DEFAULT_TASK_DIR}\n")) as mock:
            result = handle_tool("bash", {"command": "pwd"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["stdout"], f"/workspace/{DEFAULT_TASK_DIR}\n")
        mock.assert_called_once()

    # Verify that pwd with cwd="repo" is passed to exec_bash and its output, the container path of the
    # subdirectory, is returned.
    def test_pwd_with_subdir_cwd_reports_subdir(self) -> None:
        handle_tool("write", {"path": "repo/file.txt", "content": "x"})
        with patch("sandbox.api.exec_bash", return_value=_mock_exec(stdout=f"/workspace/{DEFAULT_TASK_DIR}/repo\n", cwd="repo")) as mock:
            result = handle_tool("bash", {"command": "pwd", "cwd": "repo"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["stdout"], f"/workspace/{DEFAULT_TASK_DIR}/repo\n")
        mock.assert_called_once()

    # Explicit cwd argument

    # Verify that a routed cat resolves its file relative to the explicit cwd and reports that cwd.
    def test_routed_command_respects_explicit_cwd(self) -> None:
        handle_tool("write", {"path": "src/app.py", "content": "print('hi')\n"})
        result = handle_tool("bash", {"command": "cat app.py", "cwd": "src"})
        self.assertTrue(result["ok"])
        self.assertIn("hi", result["result"]["stdout"])
        self.assertEqual(result["result"]["cwd"], "src")

    # Verify that the cwd in the result equals the cwd passed in the request.
    def test_cwd_returned_matches_input_cwd(self) -> None:
        handle_tool("write", {"path": "lib/util.py", "content": "x = 1\n"})
        result = handle_tool("bash", {"command": "cat util.py", "cwd": "lib"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["cwd"], "lib")

    # Verify that cwd=None is treated as "." when calling exec_bash.
    def test_none_cwd_defaults_to_root(self) -> None:
        with patch(
            "sandbox.api.exec_bash",
            return_value=_mock_exec(stdout="ok\n"),
        ) as mock:
            handle_tool("bash", {"command": "echo ok | cat", "cwd": None})
        mock.assert_called_once()
        self.assertEqual(mock.call_args.kwargs["cwd"], ".")

    # Container-space paths

    # Verify that `cat /etc/os-release` (a container path outside the task space) falls through to real
    # bash instead of being handled by the workspace controller.
    def test_cat_container_absolute_path_falls_through(self) -> None:
        with patch(
            "sandbox.api.exec_bash",
            return_value=_mock_exec(stdout="NAME=Linux\n"),
        ) as mock:
            result = handle_tool("bash", {"command": "cat /etc/os-release"})

        # Must have fallen through — exec_bash was called
        mock.assert_called_once()

        # Must not be routed via workspace controller
        self.assertFalse(
            result.get("result", {}).get("routed", False),
            "cat /etc/os-release should not be workspace-routed",
        )

    # Verify that `ls /usr/bin` falls through to real bash and is not workspace-routed.
    def test_ls_container_absolute_path_falls_through(self) -> None:
        with patch(
            "sandbox.api.exec_bash",
            return_value=_mock_exec(stdout="python3\nbash\n"),
        ) as mock:
            result = handle_tool("bash", {"command": "ls /usr/bin"})

        mock.assert_called_once()
        self.assertFalse(
            result.get("result", {}).get("routed", False),
            "ls /usr/bin should not be workspace-routed",
        )

    # Directory changes do not persist

    # Verify that `cd / && ls` goes to real bash and does not affect the next call, which still resolves
    # files from the task root.
    def test_cd_slash_does_not_affect_next_call_cwd(self) -> None:
        handle_tool("write", {"path": "marker.txt", "content": "here"})

        with patch("sandbox.api.exec_bash", return_value=_mock_exec(stdout="bin etc\n")):
            handle_tool("bash", {"command": "cd / && ls"})

        # Next call with no cwd specified — should still find marker.txt at root
        result = handle_tool("bash", {"command": "cat marker.txt"})
        self.assertTrue(result["ok"])
        self.assertIn("here", result["result"]["stdout"])
        self.assertEqual(result["result"]["cwd"], ".")

    # Verify that consecutive calls are independent: each one takes its cwd from its own argument, not
    # from the previous call.
    def test_consecutive_routed_calls_use_explicit_cwd(self) -> None:
        handle_tool("write", {"path": "a/b/file.txt", "content": "deep"})

        with patch("sandbox.api.exec_bash", return_value=_mock_exec(stdout="b\n", cwd="a")):
            r1 = handle_tool("bash", {"command": "ls .", "cwd": "a"})
        self.assertTrue(r1["ok"])
        self.assertIn("b", r1["result"]["stdout"])
        self.assertEqual(r1["result"]["cwd"], "a")

        r2 = handle_tool("bash", {"command": "cat file.txt", "cwd": "a/b"})
        self.assertTrue(r2["ok"])
        self.assertIn("deep", r2["result"]["stdout"])
        self.assertEqual(r2["result"]["cwd"], "a/b")

    # Nested subdirectories

    # Verify that cwd="src/lib" resolves inside the task root and is reported unchanged.
    def test_cwd_subdir_resolves_correctly(self) -> None:
        handle_tool("write", {"path": "src/lib/mod.py", "content": "MOD\n"})
        result = handle_tool("bash", {"command": "cat mod.py", "cwd": "src/lib"})
        self.assertTrue(result["ok"])
        self.assertIn("MOD", result["result"]["stdout"])
        self.assertEqual(result["result"]["cwd"], "src/lib")

    # Traversal via cwd

    # Verify that cwd="../../etc" cannot be used to read outside the task root: the call either fails or
    # its output does not contain /etc/passwd content.
    def test_cwd_traversal_is_blocked(self) -> None:
        result = handle_tool("bash", {"command": "ls .", "cwd": "../../etc"})

        # Must either fail or fall through to container where it also fails
        if result.get("ok"):
            # If somehow ok, must not have escaped
            stdout = result.get("result", {}).get("stdout", "")
            self.assertNotIn("passwd", stdout, "cwd traversal escaped to /etc!")


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
