# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from django.urls import reverse

from Apps.Data.models import Chat
from Apps.Tests.fakes.ollama_server import ollama_server  # noqa: F401 - registers the fixture


pytestmark = [pytest.mark.e2e]


# Provide the shared fixture stack: the fake Ollama server wired into real (isolated) settings.
# Yield the fake server.
@pytest.fixture
def chat_backend(ollama_server, isolated_settings, db):
    from Settings import settings as project_settings

    project_settings.save_settings(
        {"ollama-service": True, "ollama-service_port": ollama_server.port},
        sync_tool_configs=False,
    )

    # Only the process-manager boundary is faked; generation flows through
    # the real llm_api dispatch into API.ollama and the fake HTTP server.
    with (
        patch("Apps.API.views.llm_api.prepare_runtime"),
        patch("API.ollama.prepare_runtime"),
        patch("Apps.UI.views.llm_api.generate_chat_title", return_value=""),
    ):
        yield ollama_server


# Post one chat request as JSON to the chat API through the Django test client.
def _post_chat(payload: dict):
    from django.test import Client

    client = Client()
    return client.post(
        reverse("chat_api"),
        data=json.dumps(payload),
        content_type="application/json",
    )


# Full chat cycle

# Verify that a new chat streams the engine answer and persists the user and assistant messages.
def test_chat_api_full_flow_persists_streamed_answer(chat_backend):
    # Arrange: the fake engine answers with two text chunks.
    chat_backend.enqueue_text("Hello ", "from fake Ollama")

    # Act: send the first message of a new chat.
    response = _post_chat({"message": "Hi there", "model": "fake-model", "engine": "ollama-service"})

    # Assert: the streamed body carries both chunks.
    assert response.status_code == 200
    body = b"".join(response.streaming_content).decode("utf-8")
    assert "Hello " in body
    assert "from fake Ollama" in body

    # Assert: the chat holds the user message and the full assistant answer.
    chat_id = response.headers.get("X-Chat-ID")
    assert chat_id
    chat = Chat.objects.get(id=chat_id)
    messages = list(chat.messages.order_by("created_at", "id"))
    assert [message.role for message in messages] == ["user", "assistant"]
    assert messages[0].content == "Hi there"
    assert "Hello " in messages[1].content and "from fake Ollama" in messages[1].content

    # The request reached the fake engine with the model and preset options.
    request = chat_backend.chat_requests[-1]
    assert request["model"] == "fake-model"
    assert request["messages"][-1]["content"] == "Hi there"


# Verify that a second message sent with the chat id appends to the existing chat and persists the
# second assistant answer.
def test_chat_api_second_turn_appends_to_existing_chat(chat_backend):
    chat_backend.enqueue_text("First answer")
    first = _post_chat({"message": "One", "model": "fake-model", "engine": "ollama-service"})
    chat_id = first.headers.get("X-Chat-ID")

    chat_backend.enqueue_text("Second answer")
    second = _post_chat({"chat_id": chat_id, "message": "Two", "model": "fake-model", "engine": "ollama-service"})

    assert second.status_code == 200
    assert second.headers.get("X-Chat-ID") == chat_id

    # The stream must be consumed for the final assistant message to persist.
    body = b"".join(second.streaming_content).decode("utf-8")
    assert "Second answer" in body

    chat = Chat.objects.get(id=chat_id)
    contents = list(chat.messages.order_by("created_at", "id").values_list("role", "content"))
    assert len(contents) == 4
    assert contents[3][0] == "assistant"
    assert "Second answer" in contents[3][1]


# Request validation

# Verify that a disabled engine is rejected with a 400 before any chat or message is created.
def test_chat_api_rejects_disabled_engine(chat_backend, isolated_settings):
    from Settings import settings as project_settings

    project_settings.save_settings({"ollama-service": False}, sync_tool_configs=False)

    response = _post_chat({"message": "Hi", "model": "fake-model", "engine": "ollama-service"})

    assert response.status_code == 400
    assert "not enabled" in response.json()["error"]
    assert Chat.objects.count() == 0


# Verify that a request without a model is rejected with a 400 before any chat is created.
def test_chat_api_rejects_missing_model(chat_backend):
    response = _post_chat({"message": "Hi", "engine": "ollama-service"})

    assert response.status_code == 400
    assert Chat.objects.count() == 0


# Failure handling

# Verify that an unreachable engine surfaces as an error inside the 200 stream instead of a crashed
# request.
def test_chat_api_streams_error_when_engine_unreachable(chat_backend):
    chat_backend.stop()

    response = _post_chat({"message": "Hi", "model": "fake-model", "engine": "ollama-service"})

    assert response.status_code == 200
    body = b"".join(response.streaming_content).decode("utf-8")
    assert "rror" in body or "error" in body.lower()
