# Copyright NEXTGGTECH. Elastic License 2.0.

import json
from unittest.mock import Mock

import pytest

from API import google_genai, lms, mcp, ollama, openai


pytestmark = [pytest.mark.unit]


@pytest.fixture
def runtime_metadata(tmp_path, monkeypatch):
    monkeypatch.setattr(mcp, "TOOLS_DIR", tmp_path)
    path = tmp_path / "model_runtime_metadata.json"

    def write(engine="openai", vision=True):
        path.write_text(json.dumps({
            "active": {"engine": engine, "model": "model"},
            "models": {f"{engine}:model": {"capabilities": {"vision": vision}}},
        }), encoding="utf-8")
        return path

    return write


class ToolsPrepared(Exception):
    """Stop before issuing a model request, after resolving the tools."""


@pytest.mark.parametrize("adapter,engine", [
    (openai, "openai"), (lms, "lms"),
    (google_genai, "google-genai"), (ollama, "ollama-service"),
])
@pytest.mark.parametrize("vision", [True, False])
def test_all_adapters_offer_image_search_from_runtime_json(
    monkeypatch, runtime_metadata, adapter, engine, vision,
):
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

    def capture(*args, **kwargs):
        captured["tools"], captured["lookup"] = build_tools(*args, **kwargs)
        raise ToolsPrepared

    monkeypatch.setattr(mcp, "build_ollama_tools", capture)
    with pytest.raises(ToolsPrepared):
        if adapter is google_genai:
            adapter._build_google_tools(["web_search"], "model", {})
        else:
            next(adapter._run_tool_loop(Mock(), "model", [], {}, ["web_search"], {}))

    assert ("web_search__image_search" in captured["lookup"]) is vision
    provider_lookup.assert_not_called()


def test_metadata_changes_are_visible_without_restart(runtime_metadata):
    runtime_metadata(vision=True)
    assert mcp.model_supports_vision("openai", "model") is True
    runtime_metadata(vision=False)
    assert mcp.model_supports_vision("openai", "model") is False


def test_explicit_model_does_not_inherit_another_active_models_vision(runtime_metadata):
    runtime_metadata(vision=True)
    assert mcp.model_supports_vision("openai", "missing") is False
    assert mcp.model_supports_vision("lms", "model") is False
    assert mcp.model_supports_vision(None, None) is True


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


def test_missing_metadata_keeps_image_search_hidden(runtime_metadata):
    path = runtime_metadata()
    path.unlink()
    assert mcp.model_supports_vision("openai", "model") is False
