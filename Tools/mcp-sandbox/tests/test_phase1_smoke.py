# Copyright NEXTGGTECH. Elastic License 2.0.

import os
import sys
import shutil
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["SANDBOX_HOST_WORKSPACE"] = str(ROOT)

from sandbox import workspace
from sandbox.api import handle_tool


# Clear task_root except preserved fixture dirs before smoke tests.
def setup():
    task = workspace.task_root()
    task.mkdir(parents=True, exist_ok=True)
    for c in list(task.iterdir()):
        if c.name in ("clone_test", "curl_test", "downloads"):
            continue
        try:
            if c.is_dir():
                shutil.rmtree(c, ignore_errors=True)
            else:
                c.unlink(missing_ok=True)
        except Exception:
            pass


def test_large_file_auto_preview():
    # Must exceed MAX_CAT_FILE_BYTES (50KB) to trigger auto-preview
    content = "\n".join(f"def func_{i}(arg): return arg + {i}" for i in range(1, 4001))
    handle_tool("write", {"path": "big.py", "content": content})
    result = handle_tool("bash", {"command": "cat big.py"})

    stdout = result["result"]["stdout"]
    assert result["ok"], f"Expected ok=True, got error: {result.get('error')}"
    assert "-- big.py" in stdout, "Missing preview header"
    assert "[next]" in stdout, "Missing next suggestions"


def test_small_file_metadata_header():
    handle_tool("write", {"path": "small.txt", "content": "hello\nworld\n"})
    result = handle_tool("bash", {"command": "cat small.txt"})

    assert result["ok"]
    stdout = result["result"]["stdout"]
    assert stdout == "hello\nworld\n"


def test_head_metadata():
    content = "\n".join(f"line{i}" for i in range(1, 101))
    handle_tool("write", {"path": "lines.txt", "content": content})
    with patch(
        "sandbox.api.exec_bash",
        return_value={
            "exit_code": 0,
            "stdout": "line1\nline2\nline3\nline4\nline5\n",
            "stderr": "",
            "error": None,
            "elapsed_ms": 5,
            "truncated": False,
            "cwd": ".",
        },
    ):
        result = handle_tool("bash", {"command": "head -n 5 lines.txt"})

    assert result["ok"]
    stdout = result["result"]["stdout"]
    assert "line1" in stdout
    assert "line10" not in stdout
    assert not result["result"].get("routed", False)


def test_grep_with_few_results():
    handle_tool("write", {"path": "src/main.py", "content": "print('hello')\n"})
    handle_tool("write", {"path": "src/other.txt", "content": "HELLO\n"})
    with patch(
        "sandbox.api.exec_bash",
        return_value={
            "exit_code": 0,
            "stdout": "src/main.py:print('hello')\nsrc/other.txt:HELLO\n",
            "stderr": "",
            "error": None,
            "elapsed_ms": 5,
            "truncated": False,
            "cwd": ".",
        },
    ):
        result = handle_tool("bash", {"command": "grep -ri hello ."})

    assert result["ok"]
    stdout = result["result"]["stdout"]
    assert "hello" in stdout.lower()
    assert not result["result"].get("routed", False)


def test_grep_falls_through_to_real_bash_with_many_files():
    # Grep is not routed through the legacy intent controller; exec_bash handles it.
    for i in range(40):
        handle_tool("write", {
            "path": f"pkg/file_{i}.py",
            "content": "import foo\nfoo.bar()\nfoo.baz()\n",
        })
    with patch(
        "sandbox.api.exec_bash",
        return_value={
            "exit_code": 0,
            "stdout": "pkg/file_0.py:import foo\npkg/file_1.py:foo.bar()\n",
            "stderr": "",
            "error": None,
            "elapsed_ms": 5,
            "truncated": False,
            "cwd": ".",
        },
    ):
        result = handle_tool("bash", {"command": "grep -r foo pkg/"})

    assert result["ok"]
    stdout = result["result"]["stdout"]
    assert "foo" in stdout
    assert not result["result"].get("routed", False)


