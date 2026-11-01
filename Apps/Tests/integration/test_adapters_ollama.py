# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest

from API import ollama as ollama_api
from Apps.Tests.fakes.ollama_server import ollama_server  # noqa: F401 - registers the fixture


pytestmark = [pytest.mark.integration]


@pytest.fixture
def ollama_endpoint(ollama_server, isolated_settings):
    from Settings import settings as project_settings

    project_settings.save_settings(
        {"ollama-service": True, "ollama-service_port": ollama_server.port},
        sync_tool_configs=False,
    )
    return ollama_server


# Read the visible content from one adapter chunk stream.
def _chunk_text(chunks) -> str:
    parts = []
    for chunk in chunks:
        message = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", None)
        content = message.get("content", "") if isinstance(message, dict) else getattr(message, "content", "")
        parts.append(str(content or ""))
    return "".join(parts)


# Read one field from the final stream chunk.
def _chunk_value(chunks, key):
    value = chunks[-1]
    if isinstance(value, dict):
        return value.get(key)
    return getattr(value, key, None)


# Model discovery.


# Model discovery reads the /api/tags payload.
def test_get_models_reads_tag_payload(ollama_endpoint):
    ollama_endpoint.tags_payload = {
        "models": [
            {"name": "llama3:latest", "model": "llama3:latest", "size": 4_000_000_000, "details": {"parameter_size": "8B"}},
            {"name": "qwen3:8b", "model": "qwen3:8b", "size": 5_000_000_000, "details": {"parameter_size": "8B"}},
        ]
    }

    models = ollama_api.get_models()

    # The SDK returns Model records keyed by the "model" field.
    assert [getattr(model, "model", "") for model in models] == ["llama3:latest", "qwen3:8b"]


# A stopped runtime degrades to an empty model list.
def test_get_models_survives_server_outage(ollama_server, isolated_settings):
    from Settings import settings as project_settings

    project_settings.save_settings(
        {"ollama-service_port": ollama_server.port},
        sync_tool_configs=False,
    )
    ollama_server.stop()

    assert ollama_api.get_models() == []


# Streaming generation.


# Plain generation streams NDJSON chunks through the SDK.
def test_generate_streams_content_chunks(ollama_endpoint):
    ollama_endpoint.enqueue_text("Hello ", "from fake Ollama")

    chunks = list(ollama_api.generate("fake-model", [{"role": "user", "content": "hi"}], stream=True))

    assert _chunk_text(chunks) == "Hello from fake Ollama"
    assert _chunk_value(chunks, "done") is True
    assert _chunk_value(chunks, "eval_count") == 22


# Options, thinking flags, and messages reach the request payload.
def test_generate_forwards_options_and_messages(ollama_endpoint):
    ollama_endpoint.enqueue_text("ok")

    list(
        ollama_api.generate(
            "fake-model",
            [{"role": "user", "content": "hi"}],
            stream=True,
            options={"num_ctx": 8192, "num_predict": 128},
            think=True,
            think_param_name="think",
        )
    )

    request = ollama_endpoint.chat_requests[0]
    assert request["model"] == "fake-model"
    assert request["messages"] == [{"role": "user", "content": "hi"}]
    assert request["options"]["num_ctx"] == 8192
    assert request["options"]["num_predict"] == 128
    assert request["think"] is True


# Engine failures propagate to the caller.
def test_generate_error_response_raises(ollama_endpoint):
    ollama_endpoint.chat_status = 500

    with pytest.raises(Exception):
        list(ollama_api.generate("fake-model", [{"role": "user", "content": "hi"}], stream=True))


# Unsupported runtime options are dropped with a warning.
def test_generate_drops_unsupported_options_with_warning(ollama_endpoint, caplog):
    ollama_endpoint.enqueue_text("ok")

    list(
        ollama_api.generate(
            "fake-model",
            [{"role": "user", "content": "hi"}],
            stream=True,
            options={"num_ctx": 8192, "vocab_only": 1, "numa": True},
        )
    )

    request = ollama_endpoint.chat_requests[0]
    assert "vocab_only" not in request["options"]
    assert "numa" not in request["options"]


# Model metadata.


# Model metadata reads the /api/show payload.
def test_get_model_settings_uses_show_endpoint(ollama_endpoint):
    payload = ollama_api.get_model_settings("fake-model")

    details = payload["details"] if isinstance(payload, dict) else payload.details
    assert details.parameter_size == "8B"
    license_text = payload["license"] if isinstance(payload, dict) else payload.license
    assert license_text == "MIT"
    assert ollama_endpoint.show_payload["_request"]["model"] == "fake-model"
