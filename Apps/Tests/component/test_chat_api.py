# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import pytest
from unittest.mock import patch
from django.test import Client, TestCase
from django.urls import reverse
from API import mcp as tool_registry
from Tools.system_prompts import get_system_prompt
from Apps.Data.models import Chat, Message, MessageAttachment
from Apps.API.views import _build_chat_history, _stream_chat_response
from Apps.Tests.support import ToolRegistryTestMixin


pytestmark = [pytest.mark.component]


# Chat API

# Cover the chat API (validation, streaming, persistence, tools and context compression) without calling a
# real model backend.
class ChatApiTests(ToolRegistryTestMixin, TestCase):
    # Replace chat title generation with a mock that returns an empty title.
    def setUp(self):
        super().setUp()
        self.client = Client()
        self._chat_title_patch = patch("Apps.UI.views.llm_api.generate_chat_title", return_value="")
        self.mock_generate_chat_title = self._chat_title_patch.start()
        self.addCleanup(self._chat_title_patch.stop)

    # Verify that the chat API answers a malformed JSON body with 400 and an "Invalid JSON format" error
    # before touching runtime services.
    def test_chat_api_rejects_invalid_json_body(self):
        response = self.client.post(
            reverse("chat_api"),
            data="{",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Invalid JSON format")

    # Verify that the chat API answers a request without a model with 400 and a missing-model error.
    def test_chat_api_rejects_missing_model(self):
        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Missing model parameter")

    # Verify that a first message creates a chat with the user message and the assistant reply, streams
    # the reply text, returns the chat id header and prepares the runtime of the resolved engine.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_creates_new_chat_and_streams_response(
        self,
        _mock_engine,
        mock_generate,
        mock_prepare_runtime,
    ):
        mock_generate.return_value = [{"message": {"content": "Hi there"}}]

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.has_header("X-Chat-ID"))
        self.assertEqual(b"".join(response.streaming_content), b"Hi there")
        self.assertEqual(Chat.objects.count(), 1)
        self.assertEqual(Chat.objects.first().messages.count(), 2)
        mock_prepare_runtime.assert_any_call("ollama-service")

    # Verify that a generated chat title is stored on the new chat and returned URL-encoded in the
    # X-Chat-Title header.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.UI.views._build_model_info_payload", return_value={})
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_stores_generated_title(
        self,
        _mock_engine,
        _mock_model_info,
        mock_generate,
        _mock_prepare_runtime,
    ):
        self.mock_generate_chat_title.return_value = "Название чата"
        mock_generate.return_value = [{"message": {"content": "Ответ"}}]

        response = self.client.post(
            reverse("chat_api"),
            data=json.dumps({"message": "Первое сообщение", "model": "test-model"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response["X-Chat-Title"],
            "%D0%9D%D0%B0%D0%B7%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%87%D0%B0%D1%82%D0%B0",
        )
        self.assertEqual(Chat.objects.get().title, "Название чата")
        self.mock_generate_chat_title.assert_called_once()

    # Verify that a chat which starts with an instant "/search" turn and continues with a thinking turn
    # switches the system prompt, tool context and request options from the instant contract to the
    # thinking contract, and that no instant rules leak into the thinking history.
    @patch(
        "Apps.API.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["tools", "thinking"],
            "template": "{{ if .Tools }}{{ end }}{{ if .ToolCalls }}{{ end }}",
            "think_param_name": "think",
            "think_level_param_name": "think_level",
        },
    )
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_same_chat_switches_from_instant_to_thinking_contract(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
        _mock_model_settings,
    ):
        # Arrange: a web search tool server that can be selected for the chat.
        self.write_server(
            "web_search",
            '''
            MCP_SERVER = {"id": "web_search", "name": "Web Search"}
            TOOLS = [
                {"id": "web_search", "name": "Web Search", "parameters": {"type": "object", "properties": {}}},
                {"id": "read_page", "name": "Read Page", "parameters": {"type": "object", "properties": {}}},
            ]
            def call_tool(tool_id, arguments, context=None):
                return {"ok": True}
            ''',
        )
        mock_generate.return_value = [{"message": {"content": "Instant answer"}}]

        # Act: send the first turn as an instant search with thinking off and capture the generate arguments.
        instant_response = self.client.post(
            reverse("chat_api"),
            data=json.dumps({
                "message": "/search First turn",
                "model": "llama3",
                "tool_server_ids": ["web_search"],
                "options": {"think": False},
            }),
            content_type="application/json",
        )
        self.assertEqual(instant_response.status_code, 200)
        b"".join(instant_response.streaming_content)
        instant_kwargs = dict(mock_generate.call_args.kwargs)
        chat_id = instant_response["X-Chat-ID"]

        # Act: continue the same chat with a normal turn that asks for medium thinking.
        mock_generate.reset_mock()
        mock_generate.return_value = [{"message": {"content": "Thinking answer"}}]
        thinking_response = self.client.post(
            reverse("chat_api"),
            data=json.dumps({
                "chat_id": chat_id,
                "message": "Second turn",
                "model": "llama3",
                "tool_server_ids": ["web_search"],
                "options": {"think_level": "medium"},
            }),
            content_type="application/json",
        )
        self.assertEqual(thinking_response.status_code, 200)
        b"".join(thinking_response.streaming_content)
        thinking_kwargs = dict(mock_generate.call_args.kwargs)

        # Assert: the instant turn used the instant prompt, a forced three-query batch and no round limit.
        self.assertIn("Instant/no-thinking web lookup rules", instant_kwargs["messages"][0]["content"])
        self.assertNotIn("Web-search research planning rules", instant_kwargs["messages"][0]["content"])
        self.assertIn("exactly three complementary query strings", instant_kwargs["messages"][0]["content"])
        self.assertTrue(instant_kwargs["tool_context"]["instant_mode"])
        self.assertEqual(instant_kwargs["tool_context"]["instant_search_batch_size"], 3)
        self.assertEqual(instant_kwargs["tool_context"]["forced_tool_name"], "web_search")
        self.assertNotIn("max_tool_rounds", instant_kwargs["tool_context"])

        # Assert: the thinking turn used the research prompt, dropped every instant setting and applied the
        # requested think level.
        self.assertIn("Web research:", thinking_kwargs["messages"][0]["content"])
        self.assertNotIn("Instant/no-thinking web lookup rules", thinking_kwargs["messages"][0]["content"])
        self.assertNotIn("instant_mode", thinking_kwargs["tool_context"])
        self.assertNotIn("instant_search_batch_size", thinking_kwargs["tool_context"])
        self.assertNotIn("max_tool_rounds", thinking_kwargs["tool_context"])
        self.assertNotIn("think", thinking_kwargs)
        self.assertEqual(thinking_kwargs["think_level"], "medium")

        # Assert: the history holds one system message built from the thinking prompt, keeps the instant
        # answer and carries no instant rules.
        thinking_system_messages = [
            message
            for message in thinking_kwargs["messages"]
            if message.get("role") == "system"
        ]
        self.assertEqual(len(thinking_system_messages), 1)
        self.assertTrue(
            thinking_system_messages[0]["content"].startswith(
                get_system_prompt(instant_mode=False)
            )
        )
        self.assertIn(
            {"role": "assistant", "content": "Instant answer"},
            thinking_kwargs["messages"],
        )
        self.assertNotIn(
            "Instant/no-thinking web lookup rules",
            json.dumps(thinking_kwargs["messages"], ensure_ascii=False),
        )

    # Verify that a request with only an attachment and no message text creates a chat titled
    # "Attachment chat" whose user message has empty content.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="lms")
    def test_chat_api_creates_attachment_only_thread(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = [{"message": {"content": "Done"}}]

        # Act: post a request that has an attachment but no message.
        response = self.client.post(
            reverse("chat_api"),
            data=json.dumps(
                {
                    "model": "qwen",
                    "attachments": [
                        {
                            "name": "note.txt",
                            "mime_type": "text/plain",
                            "data": "SGVsbG8=",
                        },
                    ],
                }
            ),
            content_type="application/json",
        )

        # Assert: the chat is created with the placeholder title and an empty user message.
        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        chat = Chat.objects.get()
        self.assertEqual(chat.title, "Attachment chat")
        self.assertEqual(chat.messages.filter(role="user").get().content, "")

    # Verify that a tool server selected in the request is passed to the Ollama generate call with the
    # engine in the tool context and is stored as the active tool of the chat.
    @patch(
        "Apps.API.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["tools"],
            "template": "{{ if .Tools }}{{ end }}{{ if .ToolCalls }}{{ end }}",
        },
    )
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_passes_selected_tool_server_to_ollama(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
        _mock_model_settings,
    ):
        # Arrange: a time tool server that supports the model.
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite"}
            TOOLS = [
                {"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}},
                {"id": "timezone_name", "name": "Timezone Name", "parameters": {"type": "object", "properties": {}}},
            ]
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )
        mock_generate.return_value = [{"message": {"content": "Done"}}]

        # Act: send a message that selects the tool server.
        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3.1","tool_server_id":"time_suite"}',
            content_type="application/json",
        )

        # Assert: the tool server reaches generate and is persisted on the chat.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(b''.join(response.streaming_content), b'Done')
        self.assertEqual(mock_generate.call_args.kwargs["tool_server_ids"], ["time_suite"])
        self.assertEqual(mock_generate.call_args.kwargs["tool_context"]["engine"], "ollama-service")
        self.assertEqual(Chat.objects.first().active_tool_slug, '["time_suite"]')

    # Verify that selecting a tool server that does not exist answers 400 with an unknown-or-unsupported
    # tool server error.
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_rejects_unknown_tool_server(self, _mock_engine):
        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3","tool_server_id":"missing"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown or unsupported tool server", response.json()["error"])

    # Verify that the streamed body contains the thinking block, the tool call marker with the server and
    # tool ids, and the final answer text.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_stream_includes_server_and_tool_markers(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = iter([
            {"message": {"thinking": "Searching..."}},
            {"tool_event": {"server_id": "time_suite", "server_name": "Time Suite", "tool_id": "time_now", "tool_name": "Current Time", "alias": "time_suite__time_now", "arguments": {"label": "now"}}},
            {"message": {"content": "Done"}},
        ])

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        body = b''.join(response.streaming_content).decode('utf-8')
        self.assertIn('<think>\nSearching...', body)
        self.assertIn('\"server_id\": \"time_suite\"', body)
        self.assertIn('\"tool_id\": \"time_now\"', body)
        self.assertIn('Done', body)

    # Verify that a response consisting only of reasoning is kept: the stream shows the thinking block and
    # the assistant message is stored with empty content and the reasoning in its transcript.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_persists_reasoning_only_response(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = iter([
            {"message": {"thinking": "Planning the answer."}},
        ])

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        body = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn("<think>\nPlanning the answer.", body)
        assistant_message = Message.objects.filter(role="assistant").latest("created_at")
        self.assertEqual(assistant_message.content, "")
        self.assertEqual(assistant_message.llm_transcript[0]["role"], "assistant")
        self.assertEqual(assistant_message.llm_transcript[0]["thinking"], "Planning the answer.")

    # Verify that streamed reasoning is already saved to the assistant message transcript after the first
    # streamed chunk, before the full response has completed.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_buffers_reasoning_while_streaming(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = iter([
            {"message": {"thinking": "Live reasoning buffer."}},
            {"message": {"content": "Done"}},
        ])

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        stream = iter(response.streaming_content)
        first_chunk = next(stream).decode("utf-8")
        self.assertEqual(first_chunk, "<think>\n")
        assistant_message = Message.objects.filter(role="assistant").latest("created_at")
        self.assertEqual(assistant_message.llm_transcript[0]["thinking"], "Live reasoning buffer.")
        b"".join(stream)

    # Verify that streaming auto-compression can trigger while reasoning is streamed, without waiting for
    # the next send: the body carries a reasoning-triggered compression marker, the compression receives
    # the reasoning draft, and a separate compression message is stored.
    @patch("Apps.API.views._build_manual_compression_event")
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    def test_stream_chat_response_auto_compresses_at_reasoning_safe_point(
        self,
        mock_generate,
        _mock_prepare_runtime,
        mock_build_compression_event,
    ):
        # Arrange: a chat with an empty assistant message, a mocked compression event and a reasoning stream.
        chat = Chat.objects.create(title="Chat")
        assistant_message = Message.objects.create(chat=chat, role="assistant", content="", llm_transcript=[])
        compression_event = {
            "role": "tool",
            "alias": "context_compression_summary",
            "name": "context_compression_summary",
            "tool_name": "context_compression_summary",
            "tool_id": "context_compression_summary",
            "content": "Compressed history",
            "arguments": {},
        }
        mock_build_compression_event.return_value = compression_event
        mock_generate.return_value = iter([
            {"message": {"thinking": "Live reasoning buffer."}},
            {"message": {"content": "Done"}},
        ])

        # Act: stream the response of the chat.
        body = "".join(_stream_chat_response(
            "ollama-service",
            {
                "engine": "ollama-service",
                "model_name": "llama3",
                "messages": [],
                "stream": True,
            },
            "generation-1",
            chat=chat,
            assistant_message_record=assistant_message,
            session_id=str(chat.id),
            model_info_payload={},
            system_prompt="System",
        ))

        # Assert: the reasoning-triggered compression is announced, built from the draft and persisted.
        self.assertIn("<context_compression>", body)
        self.assertIn('"auto_trigger": "reasoning"', body)
        self.assertIn('"restart_generation": true', body)
        kwargs = mock_build_compression_event.call_args.kwargs
        self.assertEqual(kwargs["draft_text"], "Live reasoning buffer.")
        self.assertEqual(kwargs["exclude_message_ids"], {assistant_message.id})
        self.assertFalse(kwargs["summarize_with_model_enabled"])
        compression_message = (
            Message.objects
            .filter(role="assistant", llm_transcript__0__alias="context_compression_summary")
            .latest("created_at")
        )
        self.assertNotEqual(compression_message.id, assistant_message.id)
        self.assertEqual(compression_message.llm_transcript[0]["arguments"]["auto_trigger"], "reasoning")
        self.assertTrue(compression_message.llm_transcript[0]["arguments"]["restart_generation"])

    # Verify that streaming auto-compression also triggers when a tool call starts: the body carries a
    # tool-call-triggered compression marker, the compression draft mentions the tool, and a compression
    # message is stored.
    @patch("Apps.API.views._build_manual_compression_event")
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    def test_stream_chat_response_auto_compresses_at_tool_call_safe_point(
        self,
        mock_generate,
        _mock_prepare_runtime,
        mock_build_compression_event,
    ):
        # Arrange: a chat with an empty assistant message, a mocked compression event and a stream that
        # starts with a tool call.
        chat = Chat.objects.create(title="Chat")
        assistant_message = Message.objects.create(chat=chat, role="assistant", content="", llm_transcript=[])
        compression_event = {
            "role": "tool",
            "alias": "context_compression_summary",
            "name": "context_compression_summary",
            "tool_name": "context_compression_summary",
            "tool_id": "context_compression_summary",
            "content": "Compressed history",
            "arguments": {},
        }
        mock_build_compression_event.return_value = compression_event
        mock_generate.return_value = iter([
            {
                "tool_event": {
                    "server_id": "time_suite",
                    "server_name": "Time Suite",
                    "tool_id": "time_now",
                    "tool_name": "Current Time",
                    "alias": "time_suite__time_now",
                    "arguments": {"label": "now"},
                },
            },
            {"message": {"content": "Done"}},
        ])

        # Act: stream the response of the chat.
        body = "".join(_stream_chat_response(
            "ollama-service",
            {
                "engine": "ollama-service",
                "model_name": "llama3",
                "messages": [],
                "stream": True,
            },
            "generation-1",
            chat=chat,
            assistant_message_record=assistant_message,
            session_id=str(chat.id),
            model_info_payload={},
            system_prompt="System",
        ))

        # Assert: the tool-call-triggered compression is announced, built from the draft and persisted.
        self.assertIn("<context_compression>", body)
        self.assertIn('"auto_trigger": "tool_call"', body)
        kwargs = mock_build_compression_event.call_args.kwargs
        self.assertIn("time_now", kwargs["draft_text"])
        compression_message = (
            Message.objects
            .filter(role="assistant", llm_transcript__0__alias="context_compression_summary")
            .latest("created_at")
        )
        self.assertEqual(compression_message.llm_transcript[0]["arguments"]["auto_trigger"], "tool_call")

    # Verify that the server-side history builder counts the current prompt towards the 80% compression
    # threshold: with a very long current message it returns a compression event and a history that starts
    # with the system message, then the compressed summary, and ends with the current message.
    @patch("Apps.API.views.build_structured_history_summary")
    def test_build_chat_history_compresses_when_current_prompt_crosses_threshold(
        self,
        mock_build_summary,
    ):
        # Arrange: a mocked summary and a chat with older messages plus a very long current user message.
        mock_build_summary.return_value = (
            "Compressed prior history",
            {"summary_version": 1, "work_summary": "Compressed prior history"},
        )
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Older user context")
        Message.objects.create(chat=chat, role="assistant", content="Older assistant context")
        current_user = Message.objects.create(chat=chat, role="user", content="x" * 9800)

        # Act: build the history for a small 4096-token context window.
        llm_messages, compression_event = _build_chat_history(
            chat,
            current_user,
            current_user.content,
            "system",
            "lms",
            "qwen3",
            {"context_length": 4096, "defaults": {"num_ctx": 4096}},
        )

        # Assert: the compression event reports usage at or above 80% and the history is rebuilt around the
        # summary.
        self.assertIsNotNone(compression_event)
        self.assertEqual(compression_event["arguments"]["context_window_tokens"], 4096)
        self.assertGreaterEqual(
            compression_event["arguments"]["used_history_chars"],
            int(compression_event["arguments"]["history_budget_chars"] * 0.8),
        )
        self.assertEqual(llm_messages[0]["role"], "system")
        self.assertIn("Compressed prior history", llm_messages[1]["content"])
        self.assertEqual(llm_messages[-1]["content"], current_user.content)

    # Verify that image and file attachments are both stored and that the outbound LM Studio user message
    # carries the image as an image and the file as inline text named "[Attached file: ...]".
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="lms")
    def test_chat_api_persists_generic_attachments_and_builds_lms_messages(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = [{"message": {"content": "Done"}}]

        # Act: post a message with one image and one text file attachment.
        response = self.client.post(
            reverse("chat_api"),
            data=json.dumps(
                {
                    "message": "Hello",
                    "model": "qwen",
                    "attachments": [
                        {
                            "kind": "image",
                            "name": "photo.png",
                            "mime_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        },
                        {
                            "kind": "file",
                            "name": "note.txt",
                            "mime_type": "text/plain",
                            "data": "SGVsbG8gZnJvbSBmaWxl",
                        },
                    ],
                }
            ),
            content_type="application/json",
        )

        # Assert: both attachments are stored.
        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        self.assertEqual(MessageAttachment.objects.count(), 2)

        # Assert: the message sent to the model has one image and the file content inline.
        outbound_messages = mock_generate.call_args.kwargs["messages"]
        current_user_message = outbound_messages[-1]
        self.assertEqual(current_user_message["role"], "user")
        self.assertEqual(len(current_user_message["images"]), 1)
        self.assertIn("[Attached file: note.txt]", current_user_message["content"])
        self.assertIn("Hello from file", current_user_message["content"])

    # Verify that selecting a tool server for an LM Studio model without tool calling support answers
    # 400 with a "does not support tool calling" error.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings", return_value={"supports_tool_calling": False, "supports_files": True})
    @patch("Apps.API.views.llm_api.get_model_settings", return_value={"supports_tool_calling": False, "supports_files": True})
    @patch("Apps.API.views._resolve_request_engine", return_value="lms")
    def test_chat_api_rejects_tool_server_when_lms_model_lacks_tool_support(
        self,
        _mock_engine,
        _mock_model_settings,
        _mock_preset_model_settings,
    ):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite"}
            TOOLS = [{"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}}]
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"qwen","tool_server_id":"time_suite"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("does not support tool calling", response.json()["error"])

    # Verify that selecting a tool server for an Ollama model whose capabilities omit tools answers 400
    # with a "does not support tool calling" error, even though its template mentions tools.
    @patch(
        "Apps.API.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["completion"],
            "template": "{{ if .Tools }}tools{{ end }}{{ if .ToolCalls }}calls{{ end }}",
        },
    )
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_rejects_tool_server_when_ollama_capabilities_omit_tools(
        self,
        _mock_engine,
        _mock_model_settings,
    ):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite"}
            TOOLS = [{"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}}]
            def supports(engine=None, model_name=None):
                return engine == "ollama-service"
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"model-without-tools","tool_server_id":"time_suite"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("does not support tool calling", response.json()["error"])

    # Verify that selecting a tool server for an OpenAI model without tool calling support answers 400
    # with a "does not support tool calling" error.
    @patch("Apps.API.views.llm_api.get_model_settings", return_value={"supports_tool_calling": False, "supports_files": True})
    @patch("Apps.API.views._resolve_request_engine", return_value="openai")
    def test_chat_api_rejects_tool_server_when_openai_model_lacks_tool_support(
        self,
        _mock_engine,
        _mock_model_settings,
    ):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite"}
            TOOLS = [{"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}}]
            def supports(engine=None, model_name=None):
                return engine == "openai"
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"gpt-test","tool_server_id":"time_suite"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("does not support tool calling", response.json()["error"])

    # Verify that the assistant message stores the visible answer text in its content and the full machine
    # transcript (reasoning step, tool result and final answer) with tool server details in llm_transcript.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_saves_visible_content_and_machine_transcript(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        # Arrange: a stream with reasoning, a tool call, its result, the final answer and the matching
        # transcript messages.
        mock_generate.return_value = iter([
            {"message": {"thinking": "Plan first."}},
            {"transcript_message": {"role": "assistant", "content": "", "thinking": "Plan first.", "tool_calls": [{"function": {"name": "time_suite__time_now", "arguments": {"label": "now"}}}]}},
            {"tool_event": {"server_id": "time_suite", "server_name": "Time Suite", "tool_id": "time_now", "tool_name": "Current Time", "alias": "time_suite__time_now", "arguments": {"label": "now"}}},
            {"tool_result": {"role": "tool", "name": "time_suite__time_now", "tool_name": "time_suite__time_now", "content": '{"ok": true}', "server_id": "time_suite", "server_name": "Time Suite", "tool_id": "time_now", "tool_display_name": "Current Time", "arguments": {"label": "now"}}},
            {"message": {"content": "Done"}},
            {"transcript_message": {"role": "assistant", "content": "Done"}},
        ])

        # Act: send a message and consume the whole stream.
        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        # Assert: the stored message has the visible answer and a three-entry transcript with the tool result.
        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        assistant_message = Message.objects.filter(role="assistant").latest("created_at")
        self.assertEqual(assistant_message.content, "Done")
        self.assertEqual(len(assistant_message.llm_transcript), 3)
        self.assertEqual(assistant_message.llm_transcript[1]["role"], "tool")
        self.assertEqual(assistant_message.llm_transcript[1]["server_name"], "Time Suite")

    # Verify that a follow-up message sends the stored machine transcript of the earlier assistant
    # message (reasoning, tool result, answer) to the model instead of only its visible text.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_uses_stored_transcript_for_follow_up_messages(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        # Arrange: a chat whose assistant message has a stored transcript with a tool call and result.
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Hello")
        Message.objects.create(
            chat=chat,
            role="assistant",
            content="Visible answer",
            llm_transcript=[
                {"role": "assistant", "content": "", "thinking": "Plan", "tool_calls": [{"function": {"name": "time_suite__time_now", "arguments": {"label": "now"}}}]},
                {"role": "tool", "name": "time_suite__time_now", "tool_name": "time_suite__time_now", "content": '{"ok": true}'},
                {"role": "assistant", "content": "Visible answer"},
            ],
        )
        mock_generate.return_value = [{"message": {"content": "Next"}}]

        # Act: send a follow-up message to the existing chat.
        response = self.client.post(
            reverse("chat_api"),
            data=f'{{"chat_id":"{chat.id}","message":"Follow up","model":"llama3"}}',
            content_type="application/json",
        )

        # Assert: the history replays user, assistant with reasoning, tool and assistant, then the new message.
        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        history_messages = mock_generate.call_args.kwargs["messages"]
        non_system = [item for item in history_messages if item.get("role") != "system"]
        self.assertEqual([item["role"] for item in non_system[:4]], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(non_system[1]["thinking"], "Plan")
        self.assertEqual(non_system[2]["name"], "time_suite__time_now")
        self.assertEqual(history_messages[-1]["content"], "Follow up")

    # Verify that for an old assistant message without a stored transcript the history sent to the model
    # contains only the visible answer, with the legacy think and tool call markup removed.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_strips_legacy_ui_markup_when_transcript_is_missing(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        # Arrange: a chat whose assistant message content still holds UI markup and has no transcript.
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Hello")
        Message.objects.create(
            chat=chat,
            role="assistant",
            content='<think>\nPlan\n</think>\n<tool_call>{"alias":"time_suite__time_now"}</tool_call>Visible answer',
        )
        mock_generate.return_value = [{"message": {"content": "Next"}}]

        # Act: send a follow-up message to the existing chat.
        response = self.client.post(
            reverse("chat_api"),
            data=f'{{"chat_id":"{chat.id}","message":"Follow up","model":"llama3"}}',
            content_type="application/json",
        )

        # Assert: the earlier assistant message reaches the model as plain visible text.
        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        history_messages = mock_generate.call_args.kwargs["messages"]
        assistant_messages = [item for item in history_messages if item.get("role") == "assistant"]
        self.assertEqual(assistant_messages[0], {"role": "assistant", "content": "Visible answer"})

    # Verify that service control tokens in the model output are stripped from both the streamed text and
    # the stored assistant message.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="lms")
    def test_chat_api_strips_service_control_tokens_from_visible_output(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = [
            {"message": {"content": "<|start|>assistant<|channel|>final<|message|>Hello"}},
        ]

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hi","model":"qwen3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(b"".join(response.streaming_content).decode("utf-8"), "Hello")
        assistant_message = Message.objects.filter(role="assistant").latest("created_at")
        self.assertEqual(assistant_message.content, "Hello")


# Generate API

# Cover the stateless generate API, which streams model output without persisting chat rows.
class GenerateApiTests(ToolRegistryTestMixin, TestCase):
    # Create a fresh test client for every test.
    def setUp(self):
        super().setUp()
        self.client = Client()

    # Verify that the generate API streams the model reply and writes no chat or message rows, returning
    # session and generation id headers but no chat id header.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_streams_without_db_writes(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Stateless reply"}}]

        response = self.client.post(
            reverse("generate_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.has_header("X-Chat-ID"))
        self.assertTrue(response.has_header("X-Session-ID"))
        self.assertTrue(response.has_header("X-Generation-ID"))
        self.assertEqual(b"".join(response.streaming_content), b"Stateless reply")
        self.assertEqual(Chat.objects.count(), 0)
        self.assertEqual(Message.objects.count(), 0)

    # Verify that the generate API passes the supplied earlier messages and the new message to generate
    # (the new message last) and echoes the given session id in the X-Session-ID header.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_passes_messages_to_generate(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Follow-up"}}]

        # Act: post earlier messages, a new message and a session id.
        response = self.client.post(
            reverse("generate_api"),
            data=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Earlier question"},
                        {"role": "assistant", "content": "Earlier answer"},
                    ],
                    "message": "Next question",
                    "model": "llama3",
                    "session_id": "module-session-1",
                }
            ),
            content_type="application/json",
        )

        # Assert: the earlier messages are kept, the new message comes last and the session id is echoed.
        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        kwargs = mock_generate.call_args.kwargs
        messages = kwargs["messages"]
        self.assertEqual(messages[-1], {"role": "user", "content": "Next question"})
        self.assertIn({"role": "assistant", "content": "Earlier answer"}, messages)
        self.assertEqual(response["X-Session-ID"], "module-session-1")

    # Verify that the generate API answers a request without a model with 400 and a missing-model error.
    def test_generate_api_rejects_missing_model(self):
        response = self.client.post(
            reverse("generate_api"),
            data='{"message":"Hello"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Missing model parameter")

    # Verify that an inline attachment without message text reaches generate as the last user message
    # mentioning the attachment name.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_supports_inline_attachments(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Seen"}}]

        # Act: post a request with an attachment but no message text.
        response = self.client.post(
            reverse("generate_api"),
            data=json.dumps(
                {
                    "model": "llama3",
                    "attachments": [
                        {
                            "name": "note.txt",
                            "mime_type": "text/plain",
                            "data": "SGVsbG8=",
                        },
                    ],
                }
            ),
            content_type="application/json",
        )

        # Assert: the last message sent to the model is a user message that mentions the attachment.
        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        messages = mock_generate.call_args.kwargs["messages"]
        user_entry = messages[-1]
        self.assertEqual(user_entry["role"], "user")
        self.assertIn("note.txt", str(user_entry.get("content") or ""))

    # Verify that tool servers selected in a generate request reach generate, with the session id used as
    # the chat id of the tool context.
    @patch(
        "Apps.API.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["tools"],
            "template": "{{ if .Tools }}{{ end }}{{ if .ToolCalls }}{{ end }}",
        },
    )
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_passes_tool_servers_to_generate(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
        _mock_model_settings,
    ):
        # Arrange: a time tool server that supports the model.
        self.write_server(
            "time_suite",
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite"}
            TOOLS = [
                {"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}},
            ]
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )
        mock_generate.return_value = [{"message": {"content": "Tool reply"}}]

        # Act: send a message that selects the tool server for a named session.
        response = self.client.post(
            reverse("generate_api"),
            data=json.dumps(
                {
                    "message": "What time is it?",
                    "model": "llama3",
                    "tool_server_ids": ["time_suite"],
                    "session_id": "tool-session",
                }
            ),
            content_type="application/json",
        )

        # Assert: generate receives the tool server and a tool context keyed by the session id.
        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        kwargs = mock_generate.call_args.kwargs
        self.assertEqual(kwargs["tool_server_ids"], ["time_suite"])
        self.assertEqual(kwargs["tool_context"]["chat_id"], "tool-session")

    # Verify that the generate API switches the system prompt, the tool context, the web search and read
    # page tool schemas and the per-response tool quotas between instant mode (thinking off) and thinking
    # mode (think level set).
    @patch(
        "Apps.API.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["tools", "thinking"],
            "template": "{{ if .Tools }}{{ end }}{{ if .ToolCalls }}{{ end }}",
            "think_param_name": "think",
            "think_level_param_name": "think_level",
        },
    )
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_switches_prompt_schema_rounds_and_quotas_with_reasoning_mode(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
        _mock_model_settings,
    ):
        # Arrange: a web search server with advanced search and read page tools.
        self.write_server(
            "web_search",
            '''
            MCP_SERVER = {"id": "web_search", "name": "Web Search"}
            TOOLS = [
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Advanced search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "web": {"type": "string"},
                            "shopping": {"type": "string"},
                            "academic": {"type": "string"},
                            "onion": {"type": "string"},
                            "effort": {"type": "string"},
                            "call_description": {"type": "string"},
                        },
                    },
                },
                {
                    "id": "read_page",
                    "name": "Read Page",
                    "description": "Read exact pages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "maxItems": 10, "items": {"type": "string"}},
                                ]
                            }
                        },
                    },
                },
            ]
            def call_tool(tool_id, arguments, context=None):
                return {"ok": True}
            ''',
        )
        mock_generate.return_value = [{"message": {"content": "Done"}}]

        # Send one generate request with the given options and return the generate arguments together with
        # the tools and lookup built for that request's tool context.
        def run_mode(options):
            # Send the request and read the whole stream.
            response = self.client.post(
                reverse("generate_api"),
                data=json.dumps({
                    "message": "Check current information",
                    "model": "llama3",
                    "tool_server_ids": ["web_search"],
                    "include_skills_baseline": False,
                    "options": options,
                }),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 200)
            b"".join(response.streaming_content)

            # Build the tools exactly as the engine adapter would for this tool context.
            kwargs = dict(mock_generate.call_args.kwargs)
            tool_context = dict(kwargs["tool_context"])
            tools, lookup = tool_registry.build_ollama_tools(
                kwargs["tool_server_ids"],
                engine=kwargs["engine"],
                model_name=kwargs["model_name"],
                instant_mode=bool(tool_context.get("instant_mode")),
                instant_search_batch_size=tool_context.get("instant_search_batch_size"),
            )

            # Reset the mock so the next request can be inspected on its own.
            mock_generate.reset_mock()
            mock_generate.return_value = [{"message": {"content": "Done"}}]
            return kwargs, tools, lookup

        # Act: run one request in instant mode and one in thinking mode.
        instant_kwargs, instant_tools, instant_lookup = run_mode({"think": False})
        thinking_kwargs, thinking_tools, thinking_lookup = run_mode({"think_level": "medium"})

        # Assert: each mode gets its own system prompt.
        instant_system = instant_kwargs["messages"][0]["content"]
        thinking_system = thinking_kwargs["messages"][0]["content"]
        self.assertTrue(instant_system.startswith(get_system_prompt(instant_mode=True)))
        self.assertTrue(thinking_system.startswith(get_system_prompt(instant_mode=False)))
        self.assertIn("Instant/no-thinking web lookup rules", instant_system)
        self.assertNotIn("Web-search research planning rules", instant_system)
        self.assertIn("Web research:", thinking_system)
        self.assertNotIn("Instant/no-thinking web lookup rules", thinking_system)

        # Assert: the tool context and think options differ per mode and carry no tool round limit.
        self.assertIs(instant_kwargs["think"], False)
        self.assertTrue(instant_kwargs["tool_context"]["instant_mode"])
        self.assertNotIn("instant_search_batch_size", instant_kwargs["tool_context"])
        self.assertNotIn("max_tool_rounds", instant_kwargs["tool_context"])
        self.assertEqual(thinking_kwargs["think_level"], "medium")
        self.assertNotIn("instant_mode", thinking_kwargs["tool_context"])
        self.assertNotIn("instant_search_batch_size", thinking_kwargs["tool_context"])
        self.assertNotIn("max_tool_rounds", thinking_kwargs["tool_context"])

        # Assert: instant mode shrinks the web search schema and limits read page batches to three URLs, while
        # thinking mode keeps the full search schema.
        instant_by_name = {tool["function"]["name"]: tool for tool in instant_tools}
        thinking_by_name = {tool["function"]["name"]: tool for tool in thinking_tools}
        instant_search_schema = instant_by_name["web_search__web_search"]["function"]["parameters"]
        thinking_search_schema = thinking_by_name["web_search__web_search"]["function"]["parameters"]
        self.assertEqual(set(instant_search_schema["properties"]), {"query", "description", "operators"})
        self.assertEqual(instant_search_schema["properties"]["query"]["oneOf"][1]["maxItems"], 3)
        self.assertIn("effort", thinking_search_schema["properties"])
        self.assertIn("shopping", thinking_search_schema["properties"])

        instant_read_schema = instant_by_name["web_search__read_page"]["function"]["parameters"]
        instant_read_array = instant_read_schema["properties"]["url"]["oneOf"][1]
        self.assertEqual(instant_read_array["maxItems"], 3)

        # Assert: instant mode allows two web searches per response and rejects the third, while thinking
        # mode still accepts a second call.
        instant_event = tool_registry.build_tool_event(
            instant_lookup,
            {"name": "web_search__web_search", "arguments": {"query": "x", "description": "Checking"}},
        )
        thinking_event = tool_registry.build_tool_event(
            thinking_lookup,
            {"name": "web_search__web_search", "arguments": {"web": "x", "effort": "medium"}},
        )
        instant_counters: dict[str, int] = {}
        thinking_counters: dict[str, int] = {}
        self.assertIsNone(tool_registry.consume_tool_quota(instant_event, instant_counters))
        self.assertIsNone(tool_registry.consume_tool_quota(instant_event, instant_counters))
        self.assertIsNotNone(tool_registry.consume_tool_quota(instant_event, instant_counters))
        self.assertIsNone(tool_registry.consume_tool_quota(thinking_event, thinking_counters))
        self.assertIsNone(tool_registry.consume_tool_quota(thinking_event, thinking_counters))

    # Verify that an earlier message supplied with its llm_transcript is replayed in the history sent to
    # the model, including the tool result entry.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_replays_llm_transcript_in_history(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Done"}}]

        # Act: post an earlier assistant message that carries a transcript with a tool call and result.
        response = self.client.post(
            reverse("generate_api"),
            data=json.dumps(
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "Calling tool",
                            "llm_transcript": [
                                {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [{"id": "call-1", "function": {"name": "time_now", "arguments": "{}"}}],
                                },
                                {
                                    "role": "tool",
                                    "tool_call_id": "call-1",
                                    "content": "12:00",
                                },
                            ],
                        }
                    ],
                    "message": "Continue",
                    "model": "llama3",
                }
            ),
            content_type="application/json",
        )

        # Assert: the tool result of the transcript is part of the history sent to the model.
        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        messages = mock_generate.call_args.kwargs["messages"]
        self.assertTrue(any(entry.get("role") == "tool" and entry.get("content") == "12:00" for entry in messages))


# Message ids and regeneration

# Cover message ids in the response headers, regeneration without duplicating the user message and the
# chat update time after generation.
class MessageIdAndRegenerateTests(ToolRegistryTestMixin, TestCase):
    # Create a fresh test client for every test.
    def setUp(self):
        super().setUp()
        self.client = Client()

    # Verify that the chat API returns X-User-Message-ID and X-Assistant-Message-ID headers, so the
    # frontend can stamp fresh rows without a page reload, and that both ids point at stored messages.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_returns_message_id_headers(self, _mock_engine, mock_generate, _mock_runtime):
        mock_generate.return_value = [{"message": {"content": "Hi"}}]

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )
        b"".join(response.streaming_content)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.has_header("X-User-Message-ID"), "X-User-Message-ID header missing")
        self.assertTrue(response.has_header("X-Assistant-Message-ID"), "X-Assistant-Message-ID header missing")

        user_id = int(response["X-User-Message-ID"])
        assistant_id = int(response["X-Assistant-Message-ID"])
        self.assertTrue(Message.objects.filter(id=user_id, role="user").exists())
        self.assertTrue(Message.objects.filter(id=assistant_id, role="assistant").exists())

    # Verify that after a normal send and a regenerate the chat holds exactly one user message and one
    # assistant message, not two copies of the same prompt.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_regenerate_does_not_duplicate_user_message(self, _mock_engine, mock_generate, _mock_runtime):
        mock_generate.return_value = [{"message": {"content": "Answer"}}]

        # Act: send the first message.
        r1 = self.client.post(
            reverse("chat_api"),
            data='{"message":"hello","model":"llama3"}',
            content_type="application/json",
        )
        b"".join(r1.streaming_content)
        chat_id = r1["X-Chat-ID"]

        self.assertEqual(Message.objects.filter(chat__id=chat_id, role="user").count(), 1)

        # Act: regenerate the answer of the same chat.
        mock_generate.return_value = [{"message": {"content": "New answer"}}]
        r2 = self.client.post(
            reverse("regenerate_chat_api", args=[chat_id]),
            data='{"model":"llama3"}',
            content_type="application/json",
        )
        b"".join(r2.streaming_content)

        # Assert: regeneration replaced the assistant answer and did not add another user message.
        self.assertEqual(r2.status_code, 200)
        user_count = Message.objects.filter(chat__id=chat_id, role="user").count()
        self.assertEqual(user_count, 1, f"Expected 1 user message after regenerate, got {user_count}")
        assistant_count = Message.objects.filter(chat__id=chat_id, role="assistant").count()
        self.assertEqual(assistant_count, 1, f"Expected 1 assistant message after regenerate, got {assistant_count}")

    # Verify that chat.updated_at is bumped when messages are added, so the sidebar sort order stays
    # correct.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_updated_at_is_bumped_after_generation(self, _mock_engine, mock_generate, _mock_runtime):
        import time

        mock_generate.return_value = [{"message": {"content": "Hi"}}]

        chat = Chat.objects.create(title="Test")
        ts_before = chat.updated_at

        time.sleep(0.05)

        response = self.client.post(
            reverse("chat_api"),
            data=f'{{"message":"Hi","model":"llama3","chat_id":"{chat.id}"}}',
            content_type="application/json",
        )
        b"".join(response.streaming_content)

        chat.refresh_from_db()
        self.assertGreater(chat.updated_at, ts_before, "chat.updated_at was not bumped after generation")
