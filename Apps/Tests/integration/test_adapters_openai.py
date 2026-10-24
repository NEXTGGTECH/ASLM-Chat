# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os

import pytest

from API import openai as openai_api
from Apps.Tests.fakes.openai_server import openai_server  # noqa: F401 - registers the fixture


pytestmark = [pytest.mark.integration]


@pytest.fixture
def openai_endpoint(openai_server, isolated_settings, monkeypatch):
    from Settings import settings as project_settings

    # The OpenAI SDK inherits proxy environment variables; a system proxy on
    # the host would otherwise swallow the loopback test traffic.
    for key in list(os.environ):
        if key.lower() in {"http_proxy", "https_proxy", "all_proxy", "http-s_proxy"}:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("NO_PROXY", "*")

    project_settings.save_settings(
        {
            "openai": True,
            "openai_url": f"127.0.0.1:{openai_server.port}/v1",
            "openai_api_key": "test-key-123",
        },
        sync_tool_configs=False,
    )
    return openai_server


# Read one message field from an adapter chunk stream.
def _chunk_field(chunks, name):
    parts = []
    for chunk in chunks:
        message = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", None)
        value = message.get(name, "") if isinstance(message, dict) else getattr(message, name, "")
        parts.append(str(value or ""))
    return "".join(parts)


# Model discovery.


# Model discovery reads the /v1/models payload with bearer auth.
def test_get_models_reads_list_payload(openai_endpoint):
    openai_endpoint.models_payload = {
        "object": "list",
        "data": [
            {"id": "gpt-fake", "object": "model", "owned_by": "testing"},
            {"id": "gpt-mini", "object": "model", "owned_by": "testing"},
        ],
    }

    models = openai_api.get_models()

    assert [model["id"] for model in models] == ["gpt-fake", "gpt-mini"]
    assert openai_endpoint.auth_headers[0] == "Bearer test-key-123"


# An unreachable endpoint degrades to an empty model list.
def test_get_models_survives_outage(openai_server, isolated_settings):
    from Settings import settings as project_settings

    project_settings.save_settings(
        {"openai_url": f"127.0.0.1:{openai_server.port}/v1"},
        sync_tool_configs=False,
    )
    openai_server.stop()

    assert openai_api.get_models() == []


# Streaming generation.


# Plain generation streams SSE deltas through the SDK.
def test_generate_streams_content(openai_endpoint):
    openai_endpoint.enqueue_text("Hello ", "from fake OpenAI")

    chunks = list(
        openai_api.generate("gpt-fake", [{"role": "user", "content": "hi"}], stream=True)
    )

    visible = _chunk_field(chunks, "content")
    assert "Hello " in visible
    assert "from fake OpenAI" in visible


# Options and reasoning levels reach the request payload.
def test_generate_forwards_options_and_reasoning_level(openai_endpoint):
    openai_endpoint.enqueue_text("ok")

    list(
        openai_api.generate(
            "gpt-fake",
            [{"role": "user", "content": "hi"}],
            stream=True,
            options={"max_tokens": 123, "temperature": 0.4},
            think_level="high",
            think_level_param_name="reasoning_effort",
        )
    )

    request = openai_endpoint.chat_requests[0]
    assert request["model"] == "gpt-fake"
    assert request["max_tokens"] == 123
    assert request["temperature"] == 0.4
    assert request["reasoning_effort"] == "high"


# The configured API key travels as a bearer token.
def test_generate_sends_bearer_auth(openai_endpoint):
    openai_endpoint.enqueue_text("ok")

    list(openai_api.generate("gpt-fake", [{"role": "user", "content": "hi"}], stream=True))

    assert openai_endpoint.auth_headers[0] == "Bearer test-key-123"


# Engine failures propagate to the caller.
def test_generate_error_response_raises(openai_endpoint):
    openai_endpoint.chat_status = 500

    with pytest.raises(Exception):
        list(openai_api.generate("gpt-fake", [{"role": "user", "content": "hi"}], stream=True))


# Reasoning channels.


# Reasoning deltas map onto the normalized chunk shape.
def test_generate_maps_reasoning_deltas(openai_endpoint):
    openai_endpoint.enqueue_delta({"reasoning_content": "thinking hard"})
    openai_endpoint.enqueue_delta({"content": "Answer"})

    chunks = list(
        openai_api.generate("gpt-fake", [{"role": "user", "content": "hi"}], stream=True)
    )

    assert _chunk_field(chunks, "thinking") == "thinking hard"
    assert _chunk_field(chunks, "content") == "Answer"
