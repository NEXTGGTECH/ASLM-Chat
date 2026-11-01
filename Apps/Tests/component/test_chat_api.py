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


# Exercise chat API basics without calling a real model backend.
class ChatApiTests(ToolRegistryTestMixin, TestCase):
    # Set up the test fixture.
    def setUp(self):
        super().setUp()
        self.client = Client()
        self._chat_title_patch = patch("Apps.UI.views.llm_api.generate_chat_title", return_value="")
        self.mock_generate_chat_title = self._chat_title_patch.start()
        self.addCleanup(self._chat_title_patch.stop)

    # Test chat API rejects invalid JSON before touching runtime services.
    def test_chat_api_rejects_invalid_json_body(self):
        response = self.client.post(
            reverse("chat_api"),
            data="{",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Invalid JSON format")

    # Test chat API requires a model name.
    def test_chat_api_rejects_missing_model(self):
        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Missing model parameter")

    # Test chat API creates new chat and streams response.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api creates new chat and streams response.
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

    # Test a generated title is stored and returned with a new chat.
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

        self.assertIn("Instant/no-thinking web lookup rules", instant_kwargs["messages"][0]["content"])
        self.assertNotIn("Web-search research planning rules", instant_kwargs["messages"][0]["content"])
        self.assertIn("exactly three complementary query strings", instant_kwargs["messages"][0]["content"])
        self.assertTrue(instant_kwargs["tool_context"]["instant_mode"])
        self.assertEqual(instant_kwargs["tool_context"]["instant_search_batch_size"], 3)
        self.assertEqual(instant_kwargs["tool_context"]["forced_tool_name"], "web_search")
        self.assertNotIn("max_tool_rounds", instant_kwargs["tool_context"])

        self.assertIn("Web research:", thinking_kwargs["messages"][0]["content"])
        self.assertNotIn("Instant/no-thinking web lookup rules", thinking_kwargs["messages"][0]["content"])
        self.assertNotIn("instant_mode", thinking_kwargs["tool_context"])
        self.assertNotIn("instant_search_batch_size", thinking_kwargs["tool_context"])
        self.assertNotIn("max_tool_rounds", thinking_kwargs["tool_context"])
        self.assertNotIn("think", thinking_kwargs)
        self.assertEqual(thinking_kwargs["think_level"], "medium")

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

    # Test chat API supports an attachment-only prompt.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="lms")
    # Verify chat api creates attachment only thread.
    def test_chat_api_creates_attachment_only_thread(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = [{"message": {"content": "Done"}}]

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

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        chat = Chat.objects.get()
        self.assertEqual(chat.title, "Attachment chat")
        self.assertEqual(chat.messages.filter(role="user").get().content, "")

    # Test chat API passes selected tool server to Ollama.
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
    # Verify chat api passes selected tool server to ollama.
    def test_chat_api_passes_selected_tool_server_to_ollama(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
        _mock_model_settings,
    ):
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

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3.1","tool_server_id":"time_suite"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(b''.join(response.streaming_content), b'Done')
        self.assertEqual(mock_generate.call_args.kwargs["tool_server_ids"], ["time_suite"])
        self.assertEqual(mock_generate.call_args.kwargs["tool_context"]["engine"], "ollama-service")
        self.assertEqual(Chat.objects.first().active_tool_slug, '["time_suite"]')

    # Test chat API rejects unknown tool server.
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api rejects unknown tool server.
    def test_chat_api_rejects_unknown_tool_server(self, _mock_engine):
        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3","tool_server_id":"missing"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown or unsupported tool server", response.json()["error"])

    # Test chat API stream includes server and tool markers.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api stream includes server and tool markers.
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

    # Test chat API keeps reasoning-only output instead of deleting it on abort.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api persists reasoning only response.
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

    # Test streamed reasoning is buffered before the full response completes.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api buffers reasoning while streaming.
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

    # Test streaming compression can run during reasoning without waiting for the next send.
    @patch("Apps.API.views._build_manual_compression_event")
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    # Verify stream chat response auto compresses at reasoning safe point.
    def test_stream_chat_response_auto_compresses_at_reasoning_safe_point(
        self,
        mock_generate,
        _mock_prepare_runtime,
        mock_build_compression_event,
    ):
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

    # Test streaming compression also checks the threshold when a tool call starts.
    @patch("Apps.API.views._build_manual_compression_event")
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    # Verify stream chat response auto compresses at tool call safe point.
    def test_stream_chat_response_auto_compresses_at_tool_call_safe_point(
        self,
        mock_generate,
        _mock_prepare_runtime,
        mock_build_compression_event,
    ):
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

    # Test the server-side history builder uses the current prompt for the 80% trigger.
    @patch("Apps.API.views.build_structured_history_summary")
    # Verify build chat history compresses when current prompt crosses threshold.
    def test_build_chat_history_compresses_when_current_prompt_crosses_threshold(
        self,
        mock_build_summary,
    ):
        mock_build_summary.return_value = (
            "Compressed prior history",
            {"summary_version": 1, "work_summary": "Compressed prior history"},
        )
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Older user context")
        Message.objects.create(chat=chat, role="assistant", content="Older assistant context")
        current_user = Message.objects.create(chat=chat, role="user", content="x" * 9800)

        llm_messages, compression_event = _build_chat_history(
            chat,
            current_user,
            current_user.content,
            "system",
            "lms",
            "qwen3",
            {"context_length": 4096, "defaults": {"num_ctx": 4096}},
        )

        self.assertIsNotNone(compression_event)
        self.assertEqual(compression_event["arguments"]["context_window_tokens"], 4096)
        self.assertGreaterEqual(
            compression_event["arguments"]["used_history_chars"],
            int(compression_event["arguments"]["history_budget_chars"] * 0.8),
        )
        self.assertEqual(llm_messages[0]["role"], "system")
        self.assertIn("Compressed prior history", llm_messages[1]["content"])
        self.assertEqual(llm_messages[-1]["content"], current_user.content)

    # Test chat API persists generic attachments and builds LM Studio messages.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="lms")
    # Verify chat api persists generic attachments and builds lms messages.
    def test_chat_api_persists_generic_attachments_and_builds_lms_messages(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = [{"message": {"content": "Done"}}]

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

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        self.assertEqual(MessageAttachment.objects.count(), 2)

        outbound_messages = mock_generate.call_args.kwargs["messages"]
        current_user_message = outbound_messages[-1]
        self.assertEqual(current_user_message["role"], "user")
        self.assertEqual(len(current_user_message["images"]), 1)
        self.assertIn("[Attached file: note.txt]", current_user_message["content"])
        self.assertIn("Hello from file", current_user_message["content"])

    # Test chat API rejects tool server when LM Studio model lacks tool support.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings", return_value={"supports_tool_calling": False, "supports_files": True})
    @patch("Apps.API.views.llm_api.get_model_settings", return_value={"supports_tool_calling": False, "supports_files": True})
    @patch("Apps.API.views._resolve_request_engine", return_value="lms")
    # Verify chat api rejects tool server when lms model lacks tool support.
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

    # Test chat API rejects tool server when Ollama capabilities omit tools.
    @patch(
        "Apps.API.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["completion"],
            "template": "{{ if .Tools }}tools{{ end }}{{ if .ToolCalls }}calls{{ end }}",
        },
    )
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api rejects tool server when ollama capabilities omit tools.
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

    # Test chat API rejects tool server when OpenAI model lacks tool support.
    @patch("Apps.API.views.llm_api.get_model_settings", return_value={"supports_tool_calling": False, "supports_files": True})
    @patch("Apps.API.views._resolve_request_engine", return_value="openai")
    # Verify chat api rejects tool server when openai model lacks tool support.
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


    # Test chat API saves visible content and machine transcript.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api saves visible content and machine transcript.
    def test_chat_api_saves_visible_content_and_machine_transcript(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = iter([
            {"message": {"thinking": "Plan first."}},
            {"transcript_message": {"role": "assistant", "content": "", "thinking": "Plan first.", "tool_calls": [{"function": {"name": "time_suite__time_now", "arguments": {"label": "now"}}}]}},
            {"tool_event": {"server_id": "time_suite", "server_name": "Time Suite", "tool_id": "time_now", "tool_name": "Current Time", "alias": "time_suite__time_now", "arguments": {"label": "now"}}},
            {"tool_result": {"role": "tool", "name": "time_suite__time_now", "tool_name": "time_suite__time_now", "content": '{"ok": true}', "server_id": "time_suite", "server_name": "Time Suite", "tool_id": "time_now", "tool_display_name": "Current Time", "arguments": {"label": "now"}}},
            {"message": {"content": "Done"}},
            {"transcript_message": {"role": "assistant", "content": "Done"}},
        ])

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        assistant_message = Message.objects.filter(role="assistant").latest("created_at")
        self.assertEqual(assistant_message.content, "Done")
        self.assertEqual(len(assistant_message.llm_transcript), 3)
        self.assertEqual(assistant_message.llm_transcript[1]["role"], "tool")
        self.assertEqual(assistant_message.llm_transcript[1]["server_name"], "Time Suite")

    # Test chat API uses stored transcript for follow up messages.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api uses stored transcript for follow up messages.
    def test_chat_api_uses_stored_transcript_for_follow_up_messages(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
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

        response = self.client.post(
            reverse("chat_api"),
            data=f'{{"chat_id":"{chat.id}","message":"Follow up","model":"llama3"}}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        history_messages = mock_generate.call_args.kwargs["messages"]
        non_system = [item for item in history_messages if item.get("role") != "system"]
        self.assertEqual([item["role"] for item in non_system[:4]], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(non_system[1]["thinking"], "Plan")
        self.assertEqual(non_system[2]["name"], "time_suite__time_now")
        self.assertEqual(history_messages[-1]["content"], "Follow up")

    # Test chat API strips legacy UI markup when transcript is missing.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api strips legacy ui markup when transcript is missing.
    def test_chat_api_strips_legacy_ui_markup_when_transcript_is_missing(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Hello")
        Message.objects.create(
            chat=chat,
            role="assistant",
            content='<think>\nPlan\n</think>\n<tool_call>{"alias":"time_suite__time_now"}</tool_call>Visible answer',
        )
        mock_generate.return_value = [{"message": {"content": "Next"}}]

        response = self.client.post(
            reverse("chat_api"),
            data=f'{{"chat_id":"{chat.id}","message":"Follow up","model":"llama3"}}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        history_messages = mock_generate.call_args.kwargs["messages"]
        assistant_messages = [item for item in history_messages if item.get("role") == "assistant"]
        self.assertEqual(assistant_messages[0], {"role": "assistant", "content": "Visible answer"})

    # Test chat API strips service control tokens from visible output.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="lms")
    # Verify chat api strips service control tokens from visible output.
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

# Exercise stateless generate API without persisting chat rows.
class GenerateApiTests(ToolRegistryTestMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.client = Client()

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

    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_passes_messages_to_generate(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Follow-up"}}]

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

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        kwargs = mock_generate.call_args.kwargs
        messages = kwargs["messages"]
        self.assertEqual(messages[-1], {"role": "user", "content": "Next question"})
        self.assertIn({"role": "assistant", "content": "Earlier answer"}, messages)
        self.assertEqual(response["X-Session-ID"], "module-session-1")

    def test_generate_api_rejects_missing_model(self):
        response = self.client.post(
            reverse("generate_api"),
            data='{"message":"Hello"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Missing model parameter")

    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_supports_inline_attachments(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Seen"}}]

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

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        messages = mock_generate.call_args.kwargs["messages"]
        user_entry = messages[-1]
        self.assertEqual(user_entry["role"], "user")
        self.assertIn("note.txt", str(user_entry.get("content") or ""))

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

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        kwargs = mock_generate.call_args.kwargs
        self.assertEqual(kwargs["tool_server_ids"], ["time_suite"])
        self.assertEqual(kwargs["tool_context"]["chat_id"], "tool-session")

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

        def run_mode(options):
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
            kwargs = dict(mock_generate.call_args.kwargs)
            tool_context = dict(kwargs["tool_context"])
            tools, lookup = tool_registry.build_ollama_tools(
                kwargs["tool_server_ids"],
                engine=kwargs["engine"],
                model_name=kwargs["model_name"],
                instant_mode=bool(tool_context.get("instant_mode")),
                instant_search_batch_size=tool_context.get("instant_search_batch_size"),
            )
            mock_generate.reset_mock()
            mock_generate.return_value = [{"message": {"content": "Done"}}]
            return kwargs, tools, lookup

        instant_kwargs, instant_tools, instant_lookup = run_mode({"think": False})
        thinking_kwargs, thinking_tools, thinking_lookup = run_mode({"think_level": "medium"})

        instant_system = instant_kwargs["messages"][0]["content"]
        thinking_system = thinking_kwargs["messages"][0]["content"]
        self.assertTrue(instant_system.startswith(get_system_prompt(instant_mode=True)))
        self.assertTrue(thinking_system.startswith(get_system_prompt(instant_mode=False)))
        self.assertIn("Instant/no-thinking web lookup rules", instant_system)
        self.assertNotIn("Web-search research planning rules", instant_system)
        self.assertIn("Web research:", thinking_system)
        self.assertNotIn("Instant/no-thinking web lookup rules", thinking_system)

        self.assertIs(instant_kwargs["think"], False)
        self.assertTrue(instant_kwargs["tool_context"]["instant_mode"])
        self.assertNotIn("instant_search_batch_size", instant_kwargs["tool_context"])
        self.assertNotIn("max_tool_rounds", instant_kwargs["tool_context"])
        self.assertEqual(thinking_kwargs["think_level"], "medium")
        self.assertNotIn("instant_mode", thinking_kwargs["tool_context"])
        self.assertNotIn("instant_search_batch_size", thinking_kwargs["tool_context"])
        self.assertNotIn("max_tool_rounds", thinking_kwargs["tool_context"])

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

    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_replays_llm_transcript_in_history(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Done"}}]

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

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        messages = mock_generate.call_args.kwargs["messages"]
        self.assertTrue(any(entry.get("role") == "tool" and entry.get("content") == "12:00" for entry in messages))

# Verify the three critical fixes: message IDs in headers, no user duplication
# on regenerate, and chat.updated_at bumped on every mutation.
class MessageIdAndRegenerateTests(ToolRegistryTestMixin, TestCase):
    # Prepare shared fixtures for each test case.
    def setUp(self):
        super().setUp()
        self.client = Client()

    # chat_api must return X-User-Message-ID and X-Assistant-Message-ID headers
    # so the frontend can stamp fresh rows without waiting for a page reload.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api returns message id headers.
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

    # After a normal send + regenerate the chat must contain exactly one user
    # message — not two copies of the same prompt.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify regenerate does not duplicate user message.
    def test_regenerate_does_not_duplicate_user_message(self, _mock_engine, mock_generate, _mock_runtime):
        mock_generate.return_value = [{"message": {"content": "Answer"}}]

        r1 = self.client.post(
            reverse("chat_api"),
            data='{"message":"hello","model":"llama3"}',
            content_type="application/json",
        )
        b"".join(r1.streaming_content)
        chat_id = r1["X-Chat-ID"]

        self.assertEqual(Message.objects.filter(chat__id=chat_id, role="user").count(), 1)

        mock_generate.return_value = [{"message": {"content": "New answer"}}]
        r2 = self.client.post(
            reverse("regenerate_chat_api", args=[chat_id]),
            data='{"model":"llama3"}',
            content_type="application/json",
        )
        b"".join(r2.streaming_content)

        self.assertEqual(r2.status_code, 200)
        user_count = Message.objects.filter(chat__id=chat_id, role="user").count()
        self.assertEqual(user_count, 1, f"Expected 1 user message after regenerate, got {user_count}")
        assistant_count = Message.objects.filter(chat__id=chat_id, role="assistant").count()
        self.assertEqual(assistant_count, 1, f"Expected 1 assistant message after regenerate, got {assistant_count}")

    # chat.updated_at must be bumped when messages are added so the sidebar
    # sort order stays correct.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat updated at is bumped after generation.
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
