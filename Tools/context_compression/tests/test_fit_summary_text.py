# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json

import pytest

from context_compression.history_compressor import fit_summary_text


# Summary fitting

# Verify that a small payload within the character budget is returned unchanged, without a size-fitted risk flag.
@pytest.mark.unit
def test_fit_summary_text_returns_unchanged_when_within_budget() -> None:
    payload = {
        "summary_version": 1,
        "session_goal": "Short goal.",
        "work_summary": "Brief work.",
        "risk_flags": [],
        "artifacts": {"files": [], "urls": [], "tools_used": []},
    }
    text, fitted = fit_summary_text(payload, max_chars=50_000)
    assert "[Conversation History Summary Base]" in text
    assert fitted["session_goal"] == "Short goal."
    assert "size-fitted" not in " ".join(fitted.get("risk_flags") or [])


# Verify that an oversized payload gets a size-fitted risk flag, its key_facts list is trimmed to at most
# 8 items, and the returned text is still valid JSON after the marker line.
@pytest.mark.unit
def test_fit_summary_text_adds_risk_flag_and_truncates_lists() -> None:
    # Arrange: a payload with long narrative fields and 40-item lists, against a 1200-char budget.
    payload = {
        "summary_version": 1,
        "session_goal": "x",
        "work_summary": "y" * 5000,
        "reflection_summary": "z" * 3000,
        "key_facts": [f"fact-{index}" for index in range(40)],
        "source_memory": [f"memory-{index}" for index in range(40)],
        "open_tasks": [f"task-{index}" for index in range(40)],
        "recent_user_messages": [f"user-{index}" for index in range(40)],
        "risk_flags": [],
        "artifacts": {
            "files": [f"file-{index}.py" for index in range(40)],
            "urls": [f"https://example.com/{index}" for index in range(40)],
            "tools_used": [f"tool_{index}" for index in range(40)],
        },
    }

    # Act
    text, fitted = fit_summary_text(payload, max_chars=1200)

    # Assert: the flag is present, the list is trimmed and the text after the first line parses as JSON.
    flags = fitted.get("risk_flags") or []
    assert any("size-fitted" in str(flag) for flag in flags)
    assert len(fitted.get("key_facts") or []) <= 8
    parsed = json.loads(text.split("\n", 1)[1])
    assert isinstance(parsed, dict)
