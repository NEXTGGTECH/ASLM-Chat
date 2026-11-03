# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from core.config.settings import load_search_config
from core.query import rewriter


# Provide temp runtime metadata and engine settings files for the rewriter, with a legacy search config
# that enables the rewriter, and reset its cache and warning state. Return the metadata and settings paths.
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
    monkeypatch.setattr(rewriter, "_load_engine_settings", lambda: json.loads(settings.read_text(encoding="utf-8")))
    monkeypatch.setattr(rewriter, "load_search_config", lambda: load_search_config(path=config))
    monkeypatch.setattr(rewriter, "_cache", {})
    monkeypatch.setattr(rewriter, "_cache_lock", asyncio.Lock())
    monkeypatch.setattr(rewriter, "_warned_no_target", False)
    return metadata, settings


# Verify that a rewrite always uses the active engine and model from the runtime metadata, with the
# endpoint and credentials derived from the engine settings. The legacy endpoint and model in the search
# config never change them.
@pytest.mark.parametrize(("engine", "settings", "url", "api_key"), [
    ("ollama-service", {"ollama-service_port": 21111}, "http://127.0.0.1:21111/api/chat", ""),
    ("ollama-service", {}, "http://127.0.0.1:20003/api/chat", ""),
    ("ollama-service", {"ollama-service_port": "invalid"}, "http://127.0.0.1:20003/api/chat", ""),
    ("lms", {"lms_url": "127.0.0.1:1235"}, "http://127.0.0.1:1235/v1/chat/completions", ""),
    ("lms", {"lms_url": "http://127.0.0.1:1235/v1"}, "http://127.0.0.1:1235/v1/chat/completions", ""),
    ("openai", {"openai_url": "https://provider.example/v1", "openai_api_key": "provider-key"},
     "https://provider.example/v1/chat/completions", "provider-key"),
    ("google-genai", {"google_genai_api_key": "google-key"},
     "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "google-key"),
])
def test_rewrite_uses_active_engine_and_model(runtime_files, monkeypatch, engine, settings, url, api_key):
    # Arrange: the active engine and its settings.
    metadata, engine_settings = runtime_files
    metadata.write_text(json.dumps({"active": {"engine": engine, "model": "current-model"}}), encoding="utf-8")
    engine_settings.write_text(json.dumps(settings), encoding="utf-8")
    requests = []

    # Record the request and answer in the response shape of the engine (Ollama or OpenAI-style).
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

    # Act and assert: one request went to the expected URL with the active model and the right credentials.
    assert asyncio.run(rewriter.rewrite_query("find useful search terms")) == ("clean search terms", True)
    assert len(requests) == 1
    assert str(requests[0].url) == url
    assert json.loads(requests[0].content)["model"] == "current-model"
    assert requests[0].headers.get("Authorization", "") == (f"Bearer {api_key}" if api_key else "")


# Verify that both entry points (rewrite_query and expanded_queries) reuse cached results only while the
# active model and endpoint stay the same, and that a missing active model gives the pass-through result
# without a new model call.
@pytest.mark.parametrize("expand", [False, True])
def test_cache_follows_active_model_and_engine(runtime_files, monkeypatch, expand):
    metadata, settings = runtime_files
    calls = []

    # Record the model target and return a numbered query plus an extra one.
    async def call_model(target, timeout, thought):
        calls.append(target)
        return [f"query {len(calls)}", "extra query"]

    monkeypatch.setattr(rewriter, "_call_model", call_model)

    # Walk through four active model/endpoint combinations: each new one must call the model once, the
    # repeat call must hit the cache.
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

        # Without an active model nothing is rewritten and the model is not called again.
        metadata.unlink()
        assert await rewrite("same thought") == (None if expand else ("same thought", False))
        assert len(calls) == 4

    asyncio.run(run())


# Verify that a missing active runtime model cannot fall back to the legacy endpoint and model settings:
# the query passes through unchanged and no model call is made.
def test_missing_active_model_passes_query_through(runtime_files, monkeypatch):
    # Fail the test if the model is called.
    async def call_model(*args):
        pytest.fail("No model call should be made without an active model")

    monkeypatch.setattr(rewriter, "_call_model", call_model)
    assert asyncio.run(rewriter.rewrite_query("original query")) == ("original query", False)
    assert asyncio.run(rewriter.expanded_queries("original query")) is None


# Verify that the engine settings are read from the environment (ASLM_OPENAI_URL) even when the settings
# file does not exist.
def test_shared_settings_include_environment_without_settings_file(monkeypatch, tmp_path):
    settings = rewriter._project_settings()
    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "missing.json")
    monkeypatch.setenv("ASLM_OPENAI_URL", "https://environment.example/v1")
    settings._invalidate_settings_cache()
    try:
        assert rewriter._load_engine_settings()["openai_url"] == "https://environment.example/v1"
    finally:
        settings._invalidate_settings_cache()


# Load mcp-server.py as a throwaway module with search I/O event logging disabled, and return it.
@pytest.fixture
def search_server(monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "mcp-server.py"
    spec = importlib.util.spec_from_file_location("rewriter_pipeline_test", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    monkeypatch.setattr(server, "write_search_io_event", lambda *_: None)
    return server


# Verify that the rewritten query variants reach the executed search plan: medium effort keeps up to
# three variants, high effort keeps only the first one. The model is called once per search.
@pytest.mark.parametrize("variants,effort,expected", [
    (["clean query"], "medium", ["clean query"]),
    (["first query", "second query", "third query"], "medium", ["first query", "second query", "third query"]),
    (["first query", "second query"], "high", ["first query"]),
])
def test_rewritten_queries_reach_executed_plan(
    runtime_files, monkeypatch, search_server, variants, effort, expected,
):
    metadata, settings = runtime_files
    metadata.write_text(json.dumps({"active": {"engine": "lms", "model": "model"}}), encoding="utf-8")
    calls, plans = [], []

    # Record the thought and return the parametrized variants.
    async def call_model(target, timeout, thought):
        calls.append(thought)
        return variants

    # Capture the executed plan instead of running a real search.
    async def run_plan(plan):
        plans.append(plan)
        return {"sources": [], "model_context": "Test evidence"}

    monkeypatch.setattr(rewriter, "_call_model", call_model)
    monkeypatch.setattr(search_server, "run_web_search_plan", run_plan)
    asyncio.run(search_server._call_web_search({"query": "find information about this topic", "effort": effort}))
    assert len(calls) == 1
    assert len(plans) == 1
    assert [query["compiled_query"] for query in plans[0]["queries"]] == expected


# Verify that a failing rewriter model call is made only once for a search and the original arguments
# are returned unchanged.
def test_rewriter_failure_is_not_retried_for_one_search(runtime_files, monkeypatch, search_server):
    metadata, _ = runtime_files
    metadata.write_text(json.dumps({"active": {"engine": "lms", "model": "model"}}), encoding="utf-8")
    calls = []

    # Record the call and fail with a timeout.
    async def call_model(*args):
        calls.append(args)
        raise TimeoutError("slow model")

    monkeypatch.setattr(rewriter, "_call_model", call_model)
    args = {"query": "original query"}
    assert asyncio.run(search_server._rewrite_search_args(args)) == args
    assert len(calls) == 1


# Verify that an invalid rewrite result (an empty query) leaves the valid original search arguments
# unchanged.
def test_invalid_rewrite_preserves_valid_original_plan(monkeypatch, search_server):
    # Return an invalid expansion with only an empty query.
    async def expand(query):
        return [""]

    monkeypatch.setattr(search_server, "expanded_queries", expand)
    args = {"query": "valid query"}
    assert asyncio.run(search_server._rewrite_search_args(args)) == args
