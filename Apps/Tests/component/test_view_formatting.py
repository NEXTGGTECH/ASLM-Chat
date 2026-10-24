# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from pathlib import Path
from django.test import SimpleTestCase
from API import mcp as tool_registry
from Apps.UI.views import _build_activity_segments, _build_chat_title, _format_runtime_error, _parse_active_tool_slugs, _strip_llm_control_tokens


pytestmark = [pytest.mark.component]


# Keep user-visible LM output clean and actionable.
class ViewFormattingTests(SimpleTestCase):
    # Test strip LLM control tokens removes service markers.
    def test_strip_llm_control_tokens_removes_service_markers(self):
        self.assertEqual(
            _strip_llm_control_tokens("<|start|>assistant<|channel|>final<|message|>Hello"),
            "Hello",
        )

    # Test format runtime error hides LM Studio model load verbosity.
    def test_format_runtime_error_hides_lms_model_load_verbosity(self):
        message = _format_runtime_error(
            "lms",
            RuntimeError(
                "Model get/load error: V Cache Quantization requires flash attention to be enabled."
            ),
        )

        self.assertIn("Flash Attention", message)
        self.assertNotIn("Model get/load error", message)

    # Test chat titles are compact and useful for attachment-only threads.
    def test_build_chat_title_handles_long_and_attachment_only_messages(self):
        self.assertEqual(_build_chat_title("Short title", False), "Short title")
        self.assertEqual(_build_chat_title("x" * 31, False), f"{'x' * 30}...")
        self.assertEqual(_build_chat_title("", True), "Attachment chat")
        self.assertEqual(_build_chat_title("", False), "New Chat")

    # Test active tool slugs support both current JSON and legacy string shapes.
    def test_parse_active_tool_slugs_supports_json_and_legacy_values(self):
        self.assertEqual(_parse_active_tool_slugs('["time_suite", "", "browser"]'), ["time_suite", "browser"])
        self.assertEqual(_parse_active_tool_slugs("time_suite"), ["time_suite"])
        self.assertEqual(_parse_active_tool_slugs(""), [])

    # Test shared files keep their UI render payload after tool result splitting.
    def test_shared_file_tool_result_keeps_ui_metadata(self):
        payload = {
            "kind": "shared_file",
            "path": "/workspace/_sandbox/wave_graph.svg",
            "host_path": str(Path("C:/tmp/oda/wave_graph.svg")),
            "filename": "wave_graph.svg",
            "mime_type": "image/svg+xml",
            "size_bytes": 123,
            "render": {
                "type": "image",
                "mime_type": "image/svg+xml",
                "preview": {"kind": "base64", "data": "abc"},
            },
        }

        model_text, extras = tool_registry.split_tool_result_payload(payload)

        self.assertEqual(model_text, "Shared file ready for download: wave_graph.svg")
        self.assertEqual(extras["structured_content"]["kind"], "shared_file")
        self.assertEqual(extras["structured_content"]["file"]["render"]["type"], "image")
        self.assertIn("/api/shared-file/download/?", extras["structured_content"]["file"]["download_url"])
        self.assertEqual(extras["tool_ui"]["kind"], "shared_file")

    def test_cancelled_tool_result_is_stable_and_tells_model_not_to_retry(self):
        model_text, extras = tool_registry.split_tool_result_payload(
            tool_registry.TOOL_EXECUTION_CANCELLED
        )

        self.assertTrue(model_text.startswith("TOOL_EXECUTION_CANCELLED:"))
        self.assertIn("Do not retry", model_text)
        self.assertNotIn("object at 0x", model_text)
        self.assertEqual(extras["tool_ui"]["status"], "cancelled")
        self.assertTrue(tool_registry.is_blocking_tool_result(tool_registry.TOOL_EXECUTION_CANCELLED))

    # Test repeated tool aliases preserve all shared files in activity segments.
    def test_build_activity_segments_keeps_repeated_share_file_aliases(self):
        class _MessageStub:
            llm_transcript = [
                {
                    "role": "tool",
                    "alias": "sandbox__share_file__0",
                    "tool_id": "share_file",
                    "tool_display_name": "Share File",
                    "arguments": {"path": "a.txt", "filename": "a.txt"},
                    "content": "Shared file ready for download: a.txt",
                    "structured_content": {
                        "kind": "shared_file",
                        "file": {"kind": "shared_file", "path": "a.txt", "filename": "a.txt"},
                    },
                    "tool_ui": {
                        "kind": "shared_file",
                        "status": "done",
                        "file": {"kind": "shared_file", "path": "a.txt", "filename": "a.txt"},
                    },
                },
                {
                    "role": "tool",
                    "alias": "sandbox__share_file__0",
                    "tool_id": "share_file",
                    "tool_display_name": "Share File",
                    "arguments": {"path": "b.txt", "filename": "b.txt"},
                    "content": "Shared file ready for download: b.txt",
                    "structured_content": {
                        "kind": "shared_file",
                        "file": {"kind": "shared_file", "path": "b.txt", "filename": "b.txt"},
                    },
                    "tool_ui": {
                        "kind": "shared_file",
                        "status": "done",
                        "file": {"kind": "shared_file", "path": "b.txt", "filename": "b.txt"},
                    },
                },
            ]

        segments = _build_activity_segments(_MessageStub())
        files = [
            (segment.get("structuredContent") or {}).get("file", {}).get("filename")
            for segment in segments
            if segment.get("type") == "tool"
        ]
        self.assertEqual(files, ["a.txt", "b.txt"])
