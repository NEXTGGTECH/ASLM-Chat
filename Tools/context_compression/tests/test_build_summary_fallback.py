# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest

from context_compression.history_compressor import build_structured_history_summary


# Verify that unparseable model output makes the summary fall back to the raw harvest: the summary keeps its
# marker header, a "could not be parsed" risk flag is added, and file paths from the entries are still collected.
@pytest.mark.unit
def test_build_structured_history_summary_falls_back_on_unparseable_model_output() -> None:
    # Arrange: two entries mentioning file paths and a model callback that returns neither JSON nor sections.
    entries = [
        {"role": "user", "content": "Open C:\\repo\\src\\app.py and review it."},
        {"role": "assistant", "content": "Saved changes to Settings\\config.json."},
    ]

    # Act
    summary_text, payload = build_structured_history_summary(
        overflow_entries=entries,
        recent_user_messages=["Review the app module."],
        direct_user_directives=[],
        summarize_with_model=lambda _messages: "not valid json or markdown sections",
    )

    # Assert: the fallback warning is recorded and the raw harvest found the app.py path.
    assert summary_text.startswith("[Conversation History Summary Base]")
    risk_flags = payload.get("risk_flags") or []
    assert any("could not be parsed" in str(flag) for flag in risk_flags)
    files = payload.get("artifacts", {}).get("files", [])
    assert any("app.py" in str(path) for path in files)
