# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from context_compression.history_compressor import decide_compression, resolve_context_window_tokens


# Context window resolution

# Verify that the context window is read from the num_ctx value in the model payload defaults.
@pytest.mark.unit
def test_resolve_context_window_tokens_from_model_defaults() -> None:
    payload = {"defaults": {"num_ctx": 8192}}
    assert resolve_context_window_tokens(payload) == 8192


# Verify that without a model payload the context window comes from the runtime metadata file entry of
# the requested engine and model.
@pytest.mark.unit
def test_resolve_context_window_tokens_from_runtime_metadata(tmp_path: Path) -> None:
    # Arrange: a metadata file with one registered model that defines limits.context_window.
    meta = {
        "active": {"engine": "ollama-service", "model": "qwen"},
        "models": {
            "ollama-service:qwen": {"limits": {"context_window": 32768}},
        },
    }
    path = tmp_path / "model_runtime_metadata.json"
    path.write_text(json.dumps(meta), encoding="utf-8")

    # Act and assert: the registered window is returned.
    assert (
        resolve_context_window_tokens(
            None,
            runtime_metadata_path=path,
            active_engine="ollama-service",
            active_model="qwen",
        )
        == 32768
    )


# Compression decision

# Verify that compression is enabled once used history reaches the trigger ratio (800 of 1000 chars at 0.80),
# and that the decision carries the budget, the context window and a reason string.
@pytest.mark.unit
def test_decide_compression_enables_at_trigger_ratio() -> None:
    decision = decide_compression(
        used_history_chars=800,
        history_budget_chars=1000,
        model_info_payload={"defaults": {"num_ctx": 4096}},
        runtime_metadata_path=None,
        active_engine="",
        active_model="",
        trigger_ratio=0.80,
    )
    assert decision.enabled
    assert decision.history_budget_chars == 1000
    assert decision.context_window_tokens == 4096
    assert "used=800" in decision.reason


# Verify that compression stays disabled while used history is far below the default trigger threshold.
@pytest.mark.unit
def test_decide_compression_disabled_below_threshold() -> None:
    decision = decide_compression(
        used_history_chars=100,
        history_budget_chars=1000,
        model_info_payload=None,
        runtime_metadata_path=None,
        active_engine="",
        active_model="",
    )
    assert not decision.enabled


# Verify that debug_force_4k replaces a 128k model context window with 4096 tokens.
@pytest.mark.unit
def test_decide_compression_debug_force_4k_overrides_context_window() -> None:
    decision = decide_compression(
        used_history_chars=10_000,
        history_budget_chars=1000,
        model_info_payload={"defaults": {"num_ctx": 128_000}},
        runtime_metadata_path=None,
        active_engine="",
        active_model="",
        debug_force_4k=True,
    )
    assert decision.enabled
    assert decision.context_window_tokens == 4096
