# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import base64
import json
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
from sandbox import api as sandbox_api  # noqa: E402
from sandbox.api import handle_tool  # noqa: E402
from sandbox.config import DEFAULT_TASK_DIR, MAX_CAT_FILE_BYTES  # noqa: E402


# Cover the public contract of the sandbox tools: bash routing, write, edit, view_image, share_file and the
# tool schemas.
class SandboxV2ContractsTests(unittest.TestCase):
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

    # Verify that a written text file is returned by a routed `cat`.
    def test_write_and_bash_cat_text(self) -> None:
        write_result = handle_tool("write", {"path": "notes.txt", "content": "alpha\nbeta\n"})
        self.assertTrue(write_result["ok"])

        cat_result = handle_tool("bash", {"command": "cat notes.txt"})
        self.assertTrue(cat_result["ok"])
        self.assertEqual(cat_result["tool"], "bash")
        self.assertEqual(cat_result["result"]["stdout"], "alpha\nbeta\n")
        self.assertTrue(cat_result["result"].get("routed", False))

    # Verify that `head -n 5` is executed by real bash (exec_bash mock) and not routed.
    def test_head_uses_real_bash(self) -> None:
        handle_tool("write", {"path": "lines.txt", "content": "\n".join(f"line{i}" for i in range(1, 21))})
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
        self.assertTrue(result["ok"])
        self.assertIn("line1", result["result"]["stdout"])
        self.assertNotIn("line10", result["result"]["stdout"])
        self.assertFalse(result["result"].get("routed", False))

    # Verify that a routed cat accepts the container-absolute path of a task file (/workspace/<task dir>/...).
    def test_bash_cat_accepts_container_absolute_task_path(self) -> None:
        handle_tool("write", {"path": "notes.txt", "content": "alpha\nbeta\n"})

        result = handle_tool("bash", {"command": f"cat /workspace/{DEFAULT_TASK_DIR}/notes.txt"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["stdout"], "alpha\nbeta\n")
        self.assertTrue(result["result"].get("routed", False))

    # Verify that a routed cat accepts the legacy upload path /mnt/data/<file> and reads the task file
    # of the same relative path.
    def test_bash_cat_accepts_legacy_upload_absolute_path(self) -> None:
        handle_tool("write", {"path": "User/chat/notes.txt", "content": "alpha\nbeta\n"})

        result = handle_tool("bash", {"command": "cat /mnt/data/User/chat/notes.txt"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["stdout"], "alpha\nbeta\n")
        self.assertTrue(result["result"].get("routed", False))

    # Verify that `cat task/notes.txt` reads the file in a directory named "task" instead of treating the
    # prefix as an alias of the task root.
    def test_bash_cat_treats_task_as_normal_directory(self) -> None:
        handle_tool("write", {"path": "notes.txt", "content": "alpha\nbeta\n"})
        handle_tool("write", {"path": "task/notes.txt", "content": "task dir\n"})

        result = handle_tool("bash", {"command": "cat task/notes.txt"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["stdout"], "task dir\n")
        self.assertTrue(result["result"].get("routed", False))

    # Verify that a routed cat resolves its file relative to the cwd argument and reports that cwd.
    def test_supervised_cat_respects_cwd(self) -> None:
        handle_tool("write", {"path": "repo/README.md", "content": "hello from repo\n"})

        result = handle_tool("bash", {"command": "cat README.md", "cwd": "repo"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["stdout"], "hello from repo\n")
        self.assertEqual(result["result"]["cwd"], "repo")
        self.assertTrue(result["result"].get("routed", False))

    # Verify that cwd=None is passed to exec_bash as ".".
    def test_null_cwd_defaults_to_root_for_real_bash(self) -> None:
        with patch(
            "sandbox.api.exec_bash",
            return_value={
                "exit_code": 0,
                "stdout": "hello\n",
                "stderr": "",
                "error": None,
                "elapsed_ms": 5,
                "truncated": False,
                "cwd": ".",
            },
        ) as exec_mock:
            result = handle_tool("bash", {"command": "echo hello | cat", "cwd": None})

        self.assertTrue(result["ok"])
        exec_mock.assert_called_once()
        self.assertEqual(exec_mock.call_args.kwargs["cwd"], ".")

    # Verify that a routed cat of a file above the cat size limit returns an auto preview (file header and
    # head section), while `head -n 1` on the same file still goes to real bash and is not routed.
    def test_bash_cat_large_file_returns_auto_preview(self) -> None:
        # Arrange and act: write a file above the cat limit and cat it.
        oversized = "x" * (MAX_CAT_FILE_BYTES + 256)
        handle_tool("write", {"path": "big.log", "content": oversized})

        result = handle_tool("bash", {"command": "cat big.log"})
        self.assertTrue(result["ok"], f"Expected ok=True, got: {result.get('error')}")
        stdout = result["result"]["stdout"]
        self.assertIn("-- big.log", stdout)
        self.assertIn("[head:", stdout)
        self.assertTrue(result["result"].get("routed", False))

        # A head of the same file is not routed: real bash (mocked) answers.
        with patch(
            "sandbox.api.exec_bash",
            return_value={
                "exit_code": 0,
                "stdout": oversized + "\n",
                "stderr": "",
                "error": None,
                "elapsed_ms": 5,
                "truncated": False,
                "cwd": ".",
            },
        ):
            head_result = handle_tool("bash", {"command": "head -n 1 big.log"})
        self.assertTrue(head_result["ok"])
        self.assertFalse(head_result["result"].get("routed", False))

    # Verify that when the supervised read fails with FileNotFoundError, the cat falls back to real bash
    # and returns its output unrouted.
    def test_supervisor_falls_back_to_real_bash_on_path_mismatch(self) -> None:
        with patch("sandbox.api.read", side_effect=FileNotFoundError("File not found: repo/README.md")):
            with patch(
                "sandbox.api.exec_bash",
                return_value={
                    "exit_code": 0,
                    "stdout": "fallback worked\n",
                    "stderr": "",
                    "error": None,
                    "elapsed_ms": 7,
                    "truncated": False,
                    "cwd": ".",
                },
            ) as exec_mock:
                result = handle_tool("bash", {"command": "cat repo/README.md"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["stdout"], "fallback worked\n")
        self.assertFalse(result["result"].get("routed", False))
        exec_mock.assert_called_once()

    # Verify that grep, find and a mutating command (mkdir) each run through real bash and are not routed.
    def test_search_find_and_mutations_use_real_bash(self) -> None:
        cases = ("grep -ri hello .", "find . -name '*.py'", "mkdir -p deep/nested/dir")
        for command in cases:
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

    # Verify that compound commands (&& and a pipe) run through real bash and are not routed.
    def test_compound_commands_go_to_real_bash(self) -> None:
        for command in ("echo hello && cat notes.txt", "cat notes.txt | head -n 1"):
            with patch(
                "sandbox.api.exec_bash",
                return_value={
                    "exit_code": 0,
                    "stdout": "mocked real bash\n",
                    "stderr": "",
                    "error": None,
                    "elapsed_ms": 1,
                    "truncated": False,
                    "cwd": ".",
                },
            ) as exec_mock:
                result = handle_tool("bash", {"command": command})
            self.assertFalse(result.get("result", {}).get("routed", False), command)
            exec_mock.assert_called_once()

    # Verify that a routed cat of a PNG file returns image metadata ("[image file:") instead of its bytes.
    def test_read_image_reports_metadata_for_cat(self) -> None:
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8Bf5QAAAAASUVORK5CYII="
        )
        (self.task_root / "pixel.png").write_bytes(png_bytes)

        result = handle_tool("bash", {"command": "cat pixel.png"})
        self.assertTrue(result["ok"])
        self.assertIn("[image file:", result["result"]["stdout"])

    # Verify that view_image on a PNG returns its kind, mime type, size and an inline base64 preview when
    # the active model has vision capability.
    def test_view_image_returns_metadata_and_preview(self) -> None:
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8Bf5QAAAAASUVORK5CYII="
        )
        (self.task_root / "pixel.png").write_bytes(png_bytes)

        # Runtime metadata that declares the active model as vision capable.
        metadata_path = self.task_root / "model_runtime_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "active": {"engine": "test-engine", "model": "vision-model"},
                    "models": {
                        "test-engine:vision-model": {
                            "engine": "test-engine",
                            "model": "vision-model",
                            "capabilities": {"vision": True},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch.object(sandbox_api, "MODEL_RUNTIME_METADATA_PATH", metadata_path):
            result = handle_tool(
                "view_image",
                {"path": "pixel.png"},
                {"engine": "test-engine", "model_name": "vision-model"},
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "view_image")
        self.assertEqual(result["result"]["kind"], "image")
        self.assertEqual(result["result"]["mime"], "image/png")
        self.assertEqual(result["result"]["width"], 1)
        self.assertEqual(result["result"]["height"], 1)
        self.assertEqual(result["result"]["preview"]["type"], "inline_base64")
        self.assertEqual(result["result"]["preview"]["data_base64"], base64.b64encode(png_bytes).decode("utf-8"))

    # Verify that view_image withholds the image data and returns a text placeholder (vision gate not
    # allowed) when the active model has no vision capability.
    def test_view_image_withholds_preview_when_active_model_lacks_vision(self) -> None:
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8Bf5QAAAAASUVORK5CYII="
        )
        (self.task_root / "pixel.png").write_bytes(png_bytes)

        # Runtime metadata that declares the active model as text-only.
        metadata_path = self.task_root / "model_runtime_metadata.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "active": {"engine": "test-engine", "model": "text-only"},
                    "models": {
                        "test-engine:text-only": {
                            "engine": "test-engine",
                            "model": "text-only",
                            "capabilities": {"vision": False},
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

        with patch.object(sandbox_api, "MODEL_RUNTIME_METADATA_PATH", metadata_path):
            result = handle_tool(
                "view_image",
                {"path": "pixel.png"},
                {"engine": "test-engine", "model_name": "text-only"},
            )

        self.assertTrue(result["ok"])
        self.assertEqual(result["result"]["kind"], "image")
        self.assertEqual(result["result"]["preview"]["type"], "text_placeholder")
        self.assertNotIn("data_base64", result["result"]["preview"])
        self.assertFalse(result["result"]["vision_gate"]["allowed"])

    # Verify that view_image on a text file fails with a not_image error.
    def test_view_image_rejects_non_image(self) -> None:
        handle_tool("write", {"path": "notes.txt", "content": "hello\n"})

        result = handle_tool("view_image", {"path": "notes.txt"})

        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "view_image")
        self.assertEqual(result["error"]["type"], "not_image")

    # Verify that share_file on a PNG returns an image render with the mime type and an inline base64 preview.
    def test_share_file_includes_image_render_preview(self) -> None:
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8Bf5QAAAAASUVORK5CYII="
        )
        (self.task_root / "pixel.png").write_bytes(png_bytes)
        result = handle_tool("share_file", {"path": "pixel.png"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "share_file")
        self.assertEqual(result["result"]["kind"], "shared_file")
        self.assertEqual(result["result"]["render"]["type"], "image")
        self.assertEqual(result["result"]["render"]["mime_type"], "image/png")
        self.assertEqual(result["result"]["render"]["preview"]["type"], "inline_base64")

    # Verify that share_file keeps the original bytes of an animated GIF and an animated SVG in the inline
    # base64 preview, with the right mime types.
    def test_share_file_preserves_dynamic_image_bytes(self) -> None:
        # Arrange: a minimal animated GIF (two frames) and an SVG with an animate element.
        animated_gif = bytes.fromhex(
            "47494638396101000100800000000000ffffff"
            "21ff0b4e45545343415045322e300301000000"
            "21f904000a0000002c0000000001000100000202440100"
            "21f904000a0000002c0000000001000100000202440100"
            "3b"
        )
        animated_svg = (
            b'<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
            b'<circle cx="8" cy="8" r="4">'
            b'<animate attributeName="r" values="2;7;2" dur="1s" repeatCount="indefinite"/>'
            b'</circle></svg>'
        )
        cases = (
            ("animated.gif", animated_gif, "image/gif"),
            ("animated.svg", animated_svg, "image/svg+xml"),
        )

        # Act and assert for each file: share it and compare the mime type and the decoded preview bytes.
        for filename, image_bytes, expected_mime in cases:
            with self.subTest(filename=filename):
                (self.task_root / filename).write_bytes(image_bytes)
                result = handle_tool("share_file", {"path": filename})
                render = result["result"]["render"]

                self.assertEqual(render["type"], "image")
                self.assertEqual(render["mime_type"], expected_mime)
                self.assertEqual(render["preview"]["type"], "inline_base64")
                self.assertEqual(base64.b64decode(render["preview"]["data_base64"]), image_bytes)

    # Verify that share_file on a CSV file returns a table render with the format, the column names and
    # sample rows.
    def test_share_file_includes_table_render_preview_for_csv(self) -> None:
        handle_tool("write", {"path": "report.csv", "content": "name,score\nalice,10\nbob,20\n"})
        result = handle_tool("share_file", {"path": "report.csv"})
        self.assertTrue(result["ok"])
        self.assertEqual(result["tool"], "share_file")
        self.assertEqual(result["result"]["render"]["type"], "table")
        self.assertEqual(result["result"]["render"]["format"], "csv")
        self.assertEqual(result["result"]["render"]["columns"], ["name", "score"])
        self.assertEqual(result["result"]["render"]["sample_rows"][0], ["alice", "10"])

    # Verify that an old_str that occurs twice makes edit fail with an ambiguous_match error carrying the
    # match count.
    def test_edit_returns_typed_error_for_ambiguous_match(self) -> None:
        handle_tool("write", {"path": "app.txt", "content": "hello\nhello\n"})

        result = handle_tool(
            "edit",
            {
                "path": "app.txt",
                "old_str": "hello",
                "new_str": "world",
            },
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["tool"], "edit")
        self.assertEqual(result["error"]["type"], "ambiguous_match")
        self.assertEqual(result["result"]["match_count"], 2)

    # Verify that the public tool set is exactly bash, write, edit, view_image and share_file.
    def test_public_tools_are_bash_write_edit_view_image_share_file(self) -> None:
        from sandbox.api import TOOLS

        tool_ids = {tool["id"] for tool in TOOLS}
        self.assertEqual(tool_ids, {"bash", "write", "edit", "view_image", "share_file"})

    # Verify that the bash tool definition requires a description argument and documents the "3-4 words
    # maximum" guidance both in the parameter schema and in the tool description.
    def test_bash_requires_short_description(self) -> None:
        from sandbox.api import TOOLS

        bash_tool = next(tool for tool in TOOLS if tool["id"] == "bash")
        parameters = bash_tool["parameters"]
        description_schema = parameters["properties"]["description"]

        self.assertIn("description", parameters["required"])
        self.assertIn("3-4 words maximum", description_schema["description"])
        self.assertIn("3-4 words maximum", bash_tool["description"])

    # Verify that the bash tool registered with FastMCP keeps the required description argument with the
    # "3-4 words maximum" guidance and its length limits (1 to 80 characters).
    def test_fastmcp_bash_schema_preserves_description_guidance(self) -> None:
        from mcp.server.fastmcp import FastMCP
        from sandbox.tools import register_tools

        mcp = FastMCP("schema-test")
        register_tools(mcp)
        bash_tool = mcp._tool_manager.get_tool("bash")
        description_schema = bash_tool.parameters["properties"]["description"]

        self.assertIn("description", bash_tool.parameters["required"])
        self.assertIn("3-4 words maximum", description_schema["description"])
        self.assertEqual(description_schema["minLength"], 1)
        self.assertEqual(description_schema["maxLength"], 80)


# Allow running this file directly with the unittest runner.
if __name__ == "__main__":
    unittest.main()
