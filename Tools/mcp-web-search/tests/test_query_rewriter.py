# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.config.settings import load_search_config
from core.query import rewriter


@pytest.fixture
def runtime_files(tmp_path, monkeypatch):
    metadata = tmp_path / "model_runtime_metadata.json"
    settings = tmp_path / "settings.json"
    settings.write_text("{}", encoding="utf-8")
    config = tmp_path / "search_config.json"
    config.write_text(json.dumps({"query_rewriter": {
        "enabled": True,
        "mode": "always",
        "endpoint": "http://legacy.invalid/chat/completions",
        "model": "legacy-model",
    }}), encoding="utf-8")
    monkeypatch.setattr(rewriter, "_RUNTIME_METADATA_FILE", metadata)
    monkeypatch.setattr(rewriter, "_ENGINE_SETTINGS_FILE", settings)
    monkeypatch.setattr(rewriter, "load_search_config", lambda: load_search_config(path=config))
    monkeypatch.setattr(rewriter, "_cache", {})
    monkeypatch.setattr(rewriter, "_cache_lock", asyncio.Lock())
    monkeypatch.setattr(rewriter, "_warned_no_target", False)
    return metadata, settings


# Old overrides never change the engine, model, endpoint or credentials used for a rewrite.
@pytest.mark.parametrize(("engine", "settings", "url", "api_key"), [
    ("ollama-service", {"ollama-service_port": 21111}, "http://127.0.0.1:21111/api/chat", ""),
    ("ollama-service", {}, "http://127.0.0.1:20003/api/chat", ""),
    ("ollama-service", {"ollama-service_port": "invalid"}, "http://127.0.0.1:20003/api/chat", ""),
    ("lms", {"lms_url": "127.0.0.1:1235"}, "http://127.0.0.1:1235/v1/chat/completions", ""),
    ("openai", {"openai_url": "https://provider.example/v1", "openai_api_key": "provider-key"},
     "https://provider.example/v1/chat/completions", "provider-key"),
    ("google-genai", {"google_genai_api_key": "google-key"},
     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "google-key"),
])
def test_rewrite_uses_active_engine_and_model(runtime_files, monkeypatch, engine, settings, url, api_key):
    metadata, engine_settings = runtime_files
    metadata.write_text(json.dumps({"active": {"engine": engine, "model": "current-model"}}), encoding="utf-8")
    engine_settings.write_text(json.dumps(settings), encoding="utf-8")
    requests = []

    def respond(request):
        requests.append(request)
        content = json.dumps({"queries": ["clean search terms"]})
        message = {"content": content}
        payload = {"message": message} if engine == "ollama-service" else {"choices": [{"message": message}]}
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient
    monkeypatch.setattr(rewriter.httpx, "AsyncClient", lambda **kwargs: client(
        transport=httpx.MockTransport(respond), **kwargs,
    ))

    assert asyncio.run(rewriter.rewrite_query("find useful search terms")) == ("clean search terms", True)
    assert len(requests) == 1
    assert str(requests[0].url) == url
    assert json.loads(requests[0].content)["model"] == "current-model"
    assert requests[0].headers.get("Authorization", "") == (f"Bearer {api_key}" if api_key else "")


# Both entry points reuse results only while the active model and endpoint stay the same.
@pytest.mark.parametrize("expand", [False, True])
def test_cache_follows_active_model_and_engine(runtime_files, monkeypatch, expand):
    metadata, settings = runtime_files
    calls = []

    async def call_model(target, timeout, thought):
        calls.append(target)
        return [f"query {len(calls)}", "extra query"]

    monkeypatch.setattr(rewriter, "_call_model", call_model)

    async def run():
        rewrite = rewriter.expanded_queries if expand else rewriter.rewrite_query
        for index, (engine, model, port) in enumerate([
            ("ollama-service", "first-model", 20003),
            ("ollama-service", "second-model", 20003),
            ("lms", "second-model", 20003),
            ("ollama-service", "second-model", 21111),
        ], start=1):
            metadata.write_text(json.dumps({"active": {"engine": engine, "model": model}}), encoding="utf-8")
            settings.write_text(json.dumps({"ollama-service_port": port}), encoding="utf-8")
            expected = [f"query {index}", "extra query"] if expand else (f"query {index}", True)
            assert await rewrite("same thought") == expected
            assert await rewrite("same thought") == expected
            assert len(calls) == index
            assert calls[-1].model == model
        metadata.unlink()
        assert await rewrite("same thought") == (None if expand else ("same thought", False))
        assert len(calls) == 4

    asyncio.run(run())


# A missing runtime model cannot fall back to legacy endpoint/model settings.
def test_missing_active_model_passes_query_through(runtime_files, monkeypatch):
    async def call_model(*args):
        pytest.fail("No model call should be made without an active model")

    monkeypatch.setattr(rewriter, "_call_model", call_model)
    assert asyncio.run(rewriter.rewrite_query("original query")) == ("original query", False)
    assert asyncio.run(rewriter.expanded_queries("original query")) is None
