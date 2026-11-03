# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import unittest

from context_compression.history_compressor import build_structured_history_summary


# Overflow entries and recent user message shared by every parser case; the URL carries a stray trailing
# backtick that the summary must strip.
BASE_ENTRIES = [
    {
        "role": "user",
        "content": "Check this URL: https://www.youtube.com/watch?v=QknRUGPvsAQ`",
    },
    {
        "role": "assistant",
        "content": "The YouTube transcript loaded successfully.",
    },
]
BASE_RECENT = ["Check this URL: https://www.youtube.com/watch?v=QknRUGPvsAQ"]


# Run model output through the structured summary builder and return the payload together with its parse
# status: "fallback" when a risk flag reports unparseable or empty output, otherwise "parsed".
def _run_model_output(model_output: str) -> tuple[str, dict]:
    _summary_text, payload = build_structured_history_summary(
        overflow_entries=BASE_ENTRIES,
        recent_user_messages=BASE_RECENT,
        direct_user_directives=[],
        summarize_with_model=lambda _messages: model_output,
    )
    risk_flags = payload.get("risk_flags") if isinstance(payload.get("risk_flags"), list) else []
    fallback_markers = ("could not be parsed", "empty summary output")
    status = "fallback" if any(marker in str(flag) for flag in risk_flags for marker in fallback_markers) else "parsed"
    return status, payload


# Cover how the summary builder parses different model output shapes.
class SummaryParserCasesTests(unittest.TestCase):
    # Verify that strict, fenced and prose-wrapped JSON, contract Markdown sections and canonical labels are
    # parsed, that generic labels, malformed JSON, plain text and empty output fall back to the raw harvest,
    # and that every case keeps the cleaned URL and drops the bogus "//www.youtube.com" file.
    def test_parser_reports_parsed_or_fallback_for_model_outputs(self) -> None:
        # Arrange: a valid JSON summary reused by the three JSON cases.
        json_payload = {
            "summary_version": 1,
            "session_goal": "Check YouTube URL.",
            "current_focus": "Verify transcript loading.",
            "work_summary": "The model returned strict JSON.",
            "reflection_summary": "JSON parser path should handle this.",
            "recent_user_messages": BASE_RECENT,
            "key_facts": ["YouTube URL was provided."],
            "artifacts": {
                "files": ["//www.youtube.com"],
                "urls": ["https://www.youtube.com/watch?v=QknRUGPvsAQ`"],
                "tools_used": ["web_search__read_page"],
            },
            "open_tasks": [],
            "risk_flags": [],
            "source_memory": ["assistant: JSON parsed."],
        }

        # Cases: (name, model output, expected parse status, text expected inside work_summary).
        cases = [
            (
                "strict_json",
                json.dumps(json_payload, ensure_ascii=False),
                "parsed",
                "The model returned strict JSON.",
            ),
            (
                "fenced_json",
                "```json\n" + json.dumps(json_payload, ensure_ascii=False) + "\n```",
                "parsed",
                "The model returned strict JSON.",
            ),
            (
                "prose_wrapped_json",
                "Here is the summary:\n" + json.dumps(json_payload, ensure_ascii=False),
                "parsed",
                "The model returned strict JSON.",
            ),
            (
                "markdown_sections",
                "\n".join(
                    [
                        "## Session Goal",
                        "Check YouTube URL.",
                        "## Current Focus",
                        "Verify transcript loading.",
                        "## Work Summary",
                        "The model returned fixed Markdown.",
                        "## Reflection Summary",
                        "Markdown parser path should handle this.",
                        "## Recent User Messages",
                        "- Check this URL.",
                        "## Key Facts",
                        "- YouTube URL was provided.",
                        "## Files",
                        "- None",
                        "## URLs",
                        "- https://www.youtube.com/watch?v=QknRUGPvsAQ`",
                        "## Tools Used",
                        "- web_search__read_page",
                        "## Open Tasks",
                        "- None",
                        "## Risk Flags",
                        "- None",
                        "## Source Memory",
                        "- assistant: Markdown parsed.",
                    ]
                ),
                "parsed",
                "The model returned fixed Markdown.",
            ),
            (
                "canonical_inline_labels",
                "\n".join(
                    [
                        "**Session Goal:** Check YouTube URL.",
                        "current_focus: Verify transcript loading.",
                        "work_summary: The model returned canonical labels.",
                        "key_facts:",
                        "- YouTube URL was provided.",
                        "URLs:",
                        "- https://www.youtube.com/watch?v=QknRUGPvsAQ`",
                        "Files:",
                        "- //www.youtube.com",
                        "Source Memory:",
                        "- assistant: Canonical labels parsed.",
                    ]
                ),
                "parsed",
                "The model returned canonical labels.",
            ),
            (
                "generic_labels",
                "\n".join(
                    [
                        "Goal: Check YouTube URL.",
                        "Summary: The model used non-contract labels.",
                        "Facts:",
                        "- YouTube URL was provided.",
                    ]
                ),
                "fallback",
                "Summary: The model used non-contract labels.",
            ),
            (
                "malformed_json",
                '{"session_goal": "Check", "work_summary": ',
                "fallback",
                '{"session_goal": "Check", "work_summary":',
            ),
            (
                "plain_text",
                "I checked it and everything works, but I ignored the requested structure.",
                "fallback",
                "I checked it and everything works, but I ignored the requested structure.",
            ),
            (
                "empty_output",
                "",
                "fallback",
                "Raw compressed context was preserved without semantic extraction.",
            ),
        ]

        # Act and assert per case: the parse status and work_summary fragment match, the URL is normalized
        # (backtick removed), the web-host "file" is filtered out, and empty output reports the empty-output flag.
        for name, model_output, expected_status, expected_work_fragment in cases:
            with self.subTest(name=name):
                status, payload = _run_model_output(model_output)
                print(f"parser_case:{name} status={status}")
                self.assertEqual(status, expected_status)
                self.assertIn(expected_work_fragment, payload.get("work_summary", ""))
                self.assertEqual(
                    payload["artifacts"]["urls"],
                    ["https://www.youtube.com/watch?v=QknRUGPvsAQ"],
                )
                self.assertNotIn("//www.youtube.com", payload["artifacts"]["files"])
                if name == "empty_output":
                    self.assertEqual(
                        payload["risk_flags"],
                        ["Model returned empty summary output; raw compressed context was preserved."],
                    )


# Allow running the tests without pytest.
if __name__ == "__main__":
    unittest.main(verbosity=2)
