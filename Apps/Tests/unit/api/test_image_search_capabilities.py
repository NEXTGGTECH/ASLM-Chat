# Copyright NEXTGGTECH. Elastic License 2.0.

import json
from unittest.mock import Mock

import pytest

from API import google_genai, lms, mcp, ollama, openai


pytestmark = [pytest.mark.unit]


# Point the tools directory at a temporary folder and return a writer for model_runtime_metadata.json.
@pytest.fixture
def runtime_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp, "TOOLS_DIR", tmp_path)
    path = tmp_path / "model_runtime_metadata.json"

    # Write the metadata file with one active model of the given engine and its vision capability, and return the
    # file path.
    def write(engine="openai", vision=True):
        path.write_text(json.dumps({
            "active": {"engine": engine, "model": "model"},
            "models": {f"{engine}:model": {"capabilities": {"vision": vision}}},
        }), encoding="utf-8")
        return path

    return write


# Exception that stops an adapter before it issues a model request, after the tools were resolved.
class ToolsPrepared(Exception):
    pass


# Verify that every adapter offers the image_search tool exactly when the runtime metadata JSON says the model
# supports vision, without asking the provider or any separate cache.
@pytest.mark.parametrize("adapter,engine", [
    (openai, "openai"), (lms, "lms"),
    (google_genai, "google-genai"), (ollama, "ollama-service"),
])
@pytest.mark.parametrize("vision", [True, False])
def test_all_adapters_offer_image_search_from_runtime_json(
    monkeypatch, runtime_metadata, adapter, engine, vision,
):
    # Arrange: runtime metadata for the engine and a web_search server that exposes image_search.
    runtime_metadata(engine, vision)

    # Runtime tools must not depend on another provider request or separate cache.
    provider_lookup = Mock(side_effect=AssertionError("unexpected provider lookup"))
    monkeypatch.setattr(adapter, "get_model_settings", provider_lookup)
    if adapter is google_genai:
        monkeypatch.setattr(adapter, "_get_cached_model_capabilities", provider_lookup)
    monkeypatch.setattr(mcp, "get_server", lambda *args, **kwargs: {
        "id": "web_search",
        "tools": [{
            "id": "image_search", "alias": "web_search__image_search",
            "name": "Images", "description": "Search images", "parameters": {},
        }],
    })
    build_tools = mcp.build_ollama_tools
    captured = {}

    # Build the real tools, record the tool list and alias lookup, then stop the adapter.
    def capture(*args, **kwargs):
        captured["tools"], captured["lookup"] = build_tools(*args, **kwargs)
        raise ToolsPrepared

    # Act: run the adapter up to the point where its tools are built.
    monkeypatch.setattr(mcp, "build_ollama_tools", capture)
    with pytest.raises(ToolsPrepared):
        if adapter is google_genai:
            adapter._build_google_tools(["web_search"], "model", {})
        else:
            next(adapter._run_tool_loop(Mock(), "model", [], {}, ["web_search"], {}))

    # Assert: image_search is present only for vision models and no provider lookup happened.
    assert ("web_search__image_search" in captured["lookup"]) is vision
    provider_lookup.assert_not_called()


# Verify that rewriting the metadata file changes the vision answer immediately, without a restart.
def test_metadata_changes_are_visible_without_restart(runtime_metadata):
    runtime_metadata(vision=True)
    assert mcp.model_supports_vision("openai", "model") is True
    runtime_metadata(vision=False)
    assert mcp.model_supports_vision("openai", "model") is False


# Verify that an explicitly named model without its own record does not inherit the vision flag of the active
# model, while omitting the engine and model falls back to the active one.
def test_explicit_model_does_not_inherit_another_active_models_vision(runtime_metadata):
    runtime_metadata(vision=True)
    assert mcp.model_supports_vision("openai", "missing") is False
    assert mcp.model_supports_vision("lms", "model") is False
    assert mcp.model_supports_vision(None, None) is True


# Verify that malformed or incomplete metadata files (including a non-boolean vision value) report no vision
# support, so image_search stays hidden.
@pytest.mark.parametrize("payload", [
    "invalid json", "null", "[]", "{}",
    '{"models": []}',
    '{"models": {"openai:model": null}}',
    '{"models": {"openai:model": {"capabilities": []}}}',
    '{"models": {"openai:model": {"capabilities": {"vision": "false"}}}}',
])
def test_invalid_metadata_keeps_image_search_hidden(runtime_metadata, payload):
    path = runtime_metadata()
    path.write_text(payload, encoding="utf-8")
    assert mcp.model_supports_vision("openai", "model") is False


# Verify that a missing metadata file reports no vision support, so image_search stays hidden.
def test_missing_metadata_keeps_image_search_hidden(runtime_metadata):
    path = runtime_metadata()
    path.unlink()
    assert mcp.model_supports_vision("openai", "model") is False
