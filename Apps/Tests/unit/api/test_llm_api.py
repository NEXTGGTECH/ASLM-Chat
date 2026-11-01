# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from API import llm_api


pytestmark = [pytest.mark.unit]


# Build one scripted title stream from visible content parts.
def _title_chunks(*parts: str):
    return iter([{"message": {"role": "assistant", "content": part}} for part in parts])


# Engine module resolution.


# Engine aliases resolve to their adapter modules.
def test_engine_aliases_resolve_to_adapter_modules():
    assert llm_api._get_engine_module("ollama").__name__ == "API.ollama"
    assert llm_api._get_engine_module("gemini").__name__ == "API.google_genai"
    assert llm_api._get_engine_module("lm-studio").__name__ == "API.lms"


# Unknown engines fail with a clear error.
def test_unknown_engine_raises_value_error():
    with pytest.raises(ValueError, match="Unsupported LLM engine"):
        llm_api._get_engine_module("not-an-engine")


# Call dispatching.


# generate forwards model, messages, and kwargs to the adapter.
def test_generate_forwards_to_adapter_module():
    adapter = SimpleNamespace()
    captured = {}

    def fake_generate(model_name, messages, **kwargs):
        captured["model_name"] = model_name
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return iter([{"message": {"role": "assistant", "content": "ok"}}])

    adapter.generate = fake_generate

    with patch.object(llm_api, "_get_engine_module", return_value=adapter):
        chunks = list(
            llm_api.generate("ollama", "llama3", [{"role": "user", "content": "hi"}], stream=True)
        )

    assert chunks[0]["message"]["content"] == "ok"
    assert captured["model_name"] == "llama3"
    assert captured["kwargs"] == {"stream": True}


# Adapters without generate fail with an explicit error.
def test_generate_raises_for_missing_adapter_function():
    adapter = SimpleNamespace()

    with patch.object(llm_api, "_get_engine_module", return_value=adapter):
        with pytest.raises(NotImplementedError, match="generate"):
            llm_api.generate("ollama", "m", [])


# prepare_runtime matches the adapter's signature arity.
def test_prepare_runtime_matches_adapter_signature():
    calls = []
    zero_arg = SimpleNamespace()
    zero_arg.prepare_runtime = lambda: calls.append("zero")
    one_arg = SimpleNamespace()
    one_arg.prepare_runtime = lambda engine: calls.append(engine)

    with patch.object(llm_api, "_get_engine_module", return_value=zero_arg):
        llm_api.prepare_runtime("ollama")
    with patch.object(llm_api, "_get_engine_module", return_value=one_arg):
        llm_api.prepare_runtime("openai")

    assert calls == ["zero", "openai"]


# Adapters without prepare_runtime are simply skipped.
def test_prepare_runtime_is_noop_without_adapter_support():
    with patch.object(llm_api, "_get_engine_module", return_value=SimpleNamespace()):
        llm_api.prepare_runtime("lms")


# cleanup_runtime calls the adapter cleanup when present.
def test_cleanup_runtime_calls_adapter_cleanup():
    adapter = SimpleNamespace()
    adapter.cleanup_runtime = lambda: None

    with patch.object(llm_api, "_get_engine_module", return_value=adapter):
        llm_api.cleanup_runtime("ollama")


# reload_model fails explicitly when the adapter lacks support.
def test_reload_model_raises_without_support():
    with patch.object(llm_api, "_get_engine_module", return_value=SimpleNamespace()):
        with pytest.raises(NotImplementedError, match="reload_model"):
            llm_api.reload_model("openai", "m")


# Aborting without an engine signals every loaded adapter.
def test_abort_generation_signals_all_engines(monkeypatch):
    aborted = []

    def fake_import(name):
        adapter = SimpleNamespace()
        adapter.abort_generation = lambda: aborted.append(name)
        return adapter

    monkeypatch.setattr(
        llm_api, "ENGINE_MODULES", {name: name for name in ("API.ollama", "API.openai")}
    )
    monkeypatch.setattr("importlib.import_module", fake_import)
    monkeypatch.setattr(llm_api.tool_registry, "abort_active_tools", lambda generation_id=None: None)

    llm_api.abort_generation()

    assert sorted(aborted) == ["API.ollama", "API.openai"]


# Chat title generation.


# Empty user messages never reach the model.
def test_generate_chat_title_returns_empty_for_empty_message():
    assert llm_api.generate_chat_title("ollama", "m", "   ") == ""


# Plain models use the short output limit and the capped context.
def test_generate_chat_title_plain_model_uses_output_limit(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return _title_chunks("My Great Title")

    monkeypatch.setattr(llm_api, "generate", fake_generate)
    monkeypatch.setattr(llm_api.settings, "is_ollama_engine", lambda engine: True)

    title = llm_api.generate_chat_title("ollama", "llama3", "Please write a haiku")

    assert title == "My Great Title"
    assert len(calls) == 1
    assert calls[0]["options"] == {"num_ctx": 4096, "num_predict": 32}


# Reasoning blocks are stripped from the title text.
def test_generate_chat_title_strips_reasoning_blocks(monkeypatch):
    monkeypatch.setattr(
        llm_api, "generate", lambda **kwargs: _title_chunks("<think>secret plan</think>Clean Title")
    )
    monkeypatch.setattr(llm_api.settings, "is_ollama_engine", lambda engine: True)

    assert llm_api.generate_chat_title("ollama", "m", "hello") == "Clean Title"


# Error output is discarded and the next strategy wins.
def test_generate_chat_title_drops_error_output_and_falls_back(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return _title_chunks("[error during generation: quota]")
        return _title_chunks("Recovered Title")

    monkeypatch.setattr(llm_api, "generate", fake_generate)
    monkeypatch.setattr(llm_api.settings, "is_ollama_engine", lambda engine: True)

    # Without thinking support the strategy retries once before giving up.
    assert llm_api.generate_chat_title("ollama", "m", "hello") == "Recovered Title"
    assert len(calls) == 2


# Toggle models disable thinking on the first attempt.
def test_generate_chat_title_disables_thinking_first_for_toggle_models(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return _title_chunks("Toggled Title")

    monkeypatch.setattr(llm_api, "generate", fake_generate)
    monkeypatch.setattr(llm_api.settings, "is_ollama_engine", lambda engine: True)

    model_info = {"supports_thinking": True, "supports_think_toggle": True}
    title = llm_api.generate_chat_title("ollama", "m", "hello", model_info=model_info)

    assert title == "Toggled Title"
    assert len(calls) == 1
    assert calls[0]["think"] is False


# A failed first attempt retries with the lowest advertised level.
def test_generate_chat_title_retries_with_think_level(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("empty stream")
        return _title_chunks("Level Title")

    monkeypatch.setattr(llm_api, "generate", fake_generate)
    monkeypatch.setattr(llm_api.settings, "is_ollama_engine", lambda engine: True)

    # Toggle models try think=False first, then the lowest advertised level.
    model_info = {
        "supports_think_toggle": True,
        "supports_think_level": True,
        "think_level_options": ["medium", "low"],
    }
    title = llm_api.generate_chat_title("ollama", "m", "hello", model_info=model_info)

    assert title == "Level Title"
    assert len(calls) == 2
    assert calls[0]["think"] is False
    assert calls[1]["think_level"] == "low"


# OpenAI models use the advertised token parameter name.
def test_generate_chat_title_uses_openai_token_parameter(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return _title_chunks("OpenAI Title")

    monkeypatch.setattr(llm_api, "generate", fake_generate)
    monkeypatch.setattr(llm_api.settings, "is_ollama_engine", lambda engine: False)

    model_info = {"supported_parameters": ["max_completion_tokens"]}
    title = llm_api.generate_chat_title("openai", "gpt", "hello", model_info=model_info)

    assert title == "OpenAI Title"
    assert calls[0]["options"] == {"max_completion_tokens": 32}


# The Ollama context window is capped for title generation.
def test_generate_chat_title_caps_context_for_ollama(monkeypatch):
    calls = []

    def fake_generate(**kwargs):
        calls.append(kwargs)
        return _title_chunks("Capped")

    monkeypatch.setattr(llm_api, "generate", fake_generate)
    monkeypatch.setattr(llm_api.settings, "is_ollama_engine", lambda engine: True)

    model_info = {"context_length": 200000}
    llm_api.generate_chat_title("ollama", "m", "hello", model_info=model_info)

    assert calls[0]["options"]["num_ctx"] == 4096
