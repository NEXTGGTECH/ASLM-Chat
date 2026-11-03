# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json

import pytest

from browser_process import BrowserProcessManager, _json_safe_context


# Verify that the worker context drops the MCP session, keeps plain values and always carries the module
# and project directories.
@pytest.mark.unit
def test_json_safe_context_strips_mcp_session() -> None:
    safe = _json_safe_context(
        {
            "mcp_session": object(),
            "engine": "ollama-service",
            "module_dir": "/tmp/project",
        }
    )
    assert "mcp_session" not in safe
    assert safe["engine"] == "ollama-service"
    assert "module_dir" in safe
    assert "project_dir" in safe


# Verify that bound methods and arbitrary runtime objects are dropped at every nesting level, and that the
# result can be serialized to JSON unchanged.
@pytest.mark.unit
def test_json_safe_context_drops_methods_and_runtime_objects_recursively() -> None:
    # Runtime object with a bound method, standing in for a lock, event or session.
    class RuntimeObject:
        # Provide a bound method, which is callable and must be dropped.
        def callback(self) -> None:
            return None

    # Build a context with runtime-only values at the top level and inside a nested dict and list.
    runtime = RuntimeObject()
    safe = _json_safe_context(
        {
            "callback": runtime.callback,
            "cancel_event": runtime,
            "nested": {
                "label": "browser",
                "callback": runtime.callback,
                "items": ["kept", runtime.callback, runtime],
            },
        }
    )

    # Only the plain values survive, and the result round-trips through JSON.
    assert "callback" not in safe
    assert "cancel_event" not in safe
    assert safe["nested"] == {"label": "browser", "items": ["kept"]}
    assert json.loads(json.dumps(safe)) == safe


# Verify that only worker exit and broken-pipe errors are retryable, while successful responses and
# ordinary tool errors are not.
@pytest.mark.unit
def test_worker_response_is_retryable_only_for_transient_errors() -> None:
    manager = BrowserProcessManager()
    assert manager._worker_response_is_retryable({"ok": False, "error": "browser worker exited"})
    assert manager._worker_response_is_retryable({"ok": False, "error": "broken pipe while writing"})
    assert not manager._worker_response_is_retryable({"ok": True})
    assert not manager._worker_response_is_retryable({"ok": False, "error": "element ref not found"})


# Verify that the stale-refs notice is prepended both to a plain string result and to the model_context
# of a structured result.
@pytest.mark.unit
def test_restored_refs_message_prepends_note_for_string_and_dict() -> None:
    manager = BrowserProcessManager()
    text = manager._restored_refs_message("snapshot body")
    assert "previous browser refs are no longer valid" in text
    assert "snapshot body" in text

    structured = manager._restored_refs_message(
        {"model_context": "ctx", "ui": {"status": "done"}}
    )
    assert isinstance(structured, dict)
    assert "previous browser refs are no longer valid" in structured["model_context"]
