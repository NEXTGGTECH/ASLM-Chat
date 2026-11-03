# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import unittest

from context_compression.history_compressor import (
    _looks_like_valid_path,
    _passes_semantic_threshold,
    _raw_context_payload,
    _sanitize_semantic_items,
)


# Cover the path validation, semantic filtering and raw-context harvesting helpers of the history compressor.
class HistoryCompressorSanitizeTests(unittest.TestCase):
    # Verify that the path validator rejects code fragments and JSON-escaped code, and accepts real file paths.
    def test_looks_like_valid_path_rejects_json_escaped_code(self) -> None:
        self.assertFalse(_looks_like_valid_path(r"f:\n lines = f.readlines()"))
        self.assertFalse(_looks_like_valid_path("line.strip"))
        self.assertFalse(_looks_like_valid_path("re.match"))
        self.assertTrue(_looks_like_valid_path(r"C:\Projects\app\main.py"))
        self.assertTrue(_looks_like_valid_path("SettingsView.xaml.cs"))
        self.assertTrue(_looks_like_valid_path("archive.tar.gz"))
        self.assertFalse(_looks_like_valid_path("re.MULTILINE"))

    # Verify that assistant navigation openers ("Now let me...", "Let me...") are dropped and the real
    # summary item is kept.
    def test_sanitize_semantic_items_drops_navigation(self) -> None:
        items = [
            "assistant: Now let me do a deeper structural analysis of the repo.",
            "assistant: Let me first analyze the file for systematic issues before making changes.",
            "assistant: [Fixed import in history_compressor.py]",
        ]
        cleaned = _sanitize_semantic_items(items, limit=10, max_chars=500, allow_tool_memory=True)
        self.assertEqual(len(cleaned), 1)
        self.assertIn("Fixed import", cleaned[0])

    # Verify that, with the semantic threshold enabled, a bare title-case heading is dropped while a
    # factual sentence is kept.
    def test_sanitize_semantic_items_drops_bare_headings(self) -> None:
        items = ["Изменения:", "The compressor now filters escaped newline paths in artifacts."]
        cleaned = _sanitize_semantic_items(items, limit=10, max_chars=500, apply_semantic_threshold=True)
        self.assertEqual(len(cleaned), 1)
        self.assertIn("compressor", cleaned[0])

    # Verify that the semantic threshold rejects a short heading with a trailing colon and accepts a factual line.
    def test_passes_semantic_threshold(self) -> None:
        self.assertFalse(_passes_semantic_threshold("Изменения:"))
        self.assertTrue(_passes_semantic_threshold("YouTube URL was provided."))

    # Verify that the raw-context harvest collects a real Windows path but not the false path produced by a
    # code snippet with an escaped newline.
    def test_raw_context_payload_skips_false_windows_paths(self) -> None:
        payload = _raw_context_payload(
            [
                {
                    "role": "assistant",
                    "content": "Saved to C:\\repo\\src\\module.py\nf:\\n lines = f.readlines()",
                }
            ],
            [],
        )
        files = payload["artifacts"]["files"]
        self.assertIn(r"C:\repo\src\module.py", files)
        self.assertFalse(any("readlines" in name or "\\n" in name for name in files))


# Allow running the tests without pytest.
if __name__ == "__main__":
    unittest.main(verbosity=2)
