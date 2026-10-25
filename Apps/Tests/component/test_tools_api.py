# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import io
import threading
import zipfile
import pytest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from API import google_genai as google_genai_api
from API import llm_api
from API import lms as lms_api
from API import mcp as tool_registry
from API import ollama as ollama_api
from API import openai as openai_api
from Settings import settings as project_settings
from Apps.Data.models import Chat, ChatBranch, Message, MessageAttachment, MessageAttachmentKind, MessageImage
from Apps.API.views import _serialize_tool_call_marker
from Apps.Tests.support import ToolRegistryTestMixin


pytestmark = [pytest.mark.component]


# MCP reload endpoint tests.
class McpReloadApiTests(SimpleTestCase):
    @patch("Apps.UI.views._list_tool_servers_cached")
    @patch("Apps.UI.views._clear_tool_server_cache")
    @patch.object(tool_registry, "reset_cache")
    def test_reload_restarts_sessions_and_returns_fresh_servers(
        self,
        reset_cache_mock,
        clear_cache_mock,
        list_servers_mock,
    ):
        servers = [{"id": "custom", "name": "Custom", "user_mcp": True, "tools": []}]
        list_servers_mock.return_value = servers

        response = self.client.post(
            reverse("mcp_reload_api"),
            data=json.dumps({"model": "test-model"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tool_servers"], servers)
        reset_cache_mock.assert_called_once_with()
        clear_cache_mock.assert_called_once_with()
        list_servers_mock.assert_called_once_with(project_settings.get_llm_engine(), "test-model")

    def test_reload_rejects_non_post_requests(self):
        response = self.client.get(reverse("mcp_reload_api"))
        self.assertEqual(response.status_code, 405)

class ToolCancellationTests(SimpleTestCase):
    def tearDown(self):
        tool_registry.abort_active_tools()
        with tool_registry._ACTIVE_TOOL_EXECUTIONS_LOCK:
            tool_registry._ACTIVE_TOOL_EXECUTIONS.clear()

    def test_abort_active_tools_only_cancels_matching_generation(self):
        first = tool_registry.ActiveToolExecution("generation-a")
        second = tool_registry.ActiveToolExecution("generation-b")
        first_callback = Mock()
        second_callback = Mock()
        first.set_cancel_callback(first_callback)
        second.set_cancel_callback(second_callback)
        tool_registry._register_active_tool(first)
        tool_registry._register_active_tool(second)

        cancelled_count = tool_registry.abort_active_tools("generation-a")

        self.assertEqual(cancelled_count, 1)
        self.assertTrue(first.cancelled.is_set())
        self.assertFalse(second.cancelled.is_set())
        first_callback.assert_called_once_with()
        second_callback.assert_not_called()

    def test_abort_active_research_only_cancels_matching_session(self):
        first = tool_registry.ActiveToolExecution(
            "generation-a",
            research_session_id="deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        second = tool_registry.ActiveToolExecution(
            "generation-a",
            research_session_id="deep-research:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        unrelated = tool_registry.ActiveToolExecution("generation-a")
        first_callback = Mock()
        second_callback = Mock()
        unrelated_callback = Mock()
        first.set_cancel_callback(first_callback)
        second.set_cancel_callback(second_callback)
        unrelated.set_cancel_callback(unrelated_callback)
        tool_registry._register_active_tool(first)
        tool_registry._register_active_tool(second)
        tool_registry._register_active_tool(unrelated)

        cancelled_count = tool_registry.abort_active_research(
            "deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )

        self.assertEqual(cancelled_count, 1)
        self.assertTrue(first.cancelled.is_set())
        self.assertFalse(second.cancelled.is_set())
        self.assertFalse(unrelated.cancelled.is_set())
        first_callback.assert_called_once_with()
        second_callback.assert_not_called()
        unrelated_callback.assert_not_called()

    def test_streamed_in_process_tool_receives_cancel_event_and_stops(self):
        started = threading.Event()
        stopped = threading.Event()

        def blocking_handler(_arguments, context):
            started.set()
            context["cancel_event"].wait(timeout=2)
            stopped.set()
            return "finished"

        tool = {
            "id": "wait",
            "alias": "local__wait",
            "name": "Wait",
            "description": "",
        }
        server = {
            "id": "local",
            "name": "Local",
            "external": False,
            "server_file": Path("Tools/local/mcp-server.py"),
            "tool_handlers": {"wait": blocking_handler},
            "server_callable": None,
        }
        lookup = {"local__wait": {"server": server, "tool": tool}}
        outcome = {}

        def consume() -> None:
            stream = tool_registry.stream_ollama_tool(
                lookup,
                "local__wait",
                {},
                context={"generation_id": "generation-a"},
            )
            try:
                while True:
                    next(stream)
            except StopIteration as exc:
                outcome["result"] = exc.value

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(started.wait(timeout=1))

        self.assertEqual(tool_registry.abort_active_tools("generation-a"), 1)
        consumer.join(timeout=1)

        self.assertFalse(consumer.is_alive())
        self.assertTrue(stopped.wait(timeout=1))
        self.assertTrue(tool_registry.is_tool_execution_cancelled(outcome.get("result")))

    def test_research_stop_maps_interrupted_worker_error_to_cancelled_result(self):
        session_id = "deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        started = threading.Event()
        outcome = {}

        def interrupted_call(*_args, **kwargs):
            cancellation = kwargs["cancellation"]
            started.set()
            cancellation.cancelled.wait(timeout=2)
            raise RuntimeError("worker process was terminated")

        def consume() -> None:
            with patch.object(tool_registry, "call_ollama_tool", side_effect=interrupted_call):
                stream = tool_registry.stream_ollama_tool(
                    {},
                    "deep_research__deep_research",
                    {"session_id": session_id},
                    context={"generation_id": "generation-a"},
                )
                try:
                    while True:
                        next(stream)
                except StopIteration as exc:
                    outcome["result"] = exc.value

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(started.wait(timeout=1))

        self.assertEqual(tool_registry.abort_active_research(session_id), 1)
        consumer.join(timeout=1)

        self.assertFalse(consumer.is_alive())
        self.assertTrue(tool_registry.is_tool_execution_cancelled(outcome.get("result")))

    def test_research_worker_monitors_durable_stop_while_provider_is_blocked(self):
        session_id = "deep-research:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        provider_started = threading.Event()
        stop_requested = threading.Event()
        cancellation_checkpointed = threading.Event()
        outcome = {}

        def blocked_provider(*_args, **kwargs):
            cancellation = kwargs["cancellation"]
            provider_started.set()
            cancellation.cancelled.wait(timeout=2)
            raise RuntimeError("blocked provider interrupted")

        def consume() -> None:
            def checkpoint_cancelled(*_args, **_kwargs):
                cancellation_checkpointed.set()
                return {}

            with (
                patch.object(tool_registry, "call_ollama_tool", side_effect=blocked_provider),
                patch(
                    "Tools.deep_research.control.latest_cancel_requested",
                    side_effect=lambda _session_id: stop_requested.is_set(),
                ),
                patch(
                    "Tools.deep_research.control.update_state",
                    side_effect=checkpoint_cancelled,
                ),
            ):
                stream = tool_registry.stream_ollama_tool(
                    {},
                    "deep_research__deep_research",
                    {"session_id": session_id},
                    context={"generation_id": "generation-a"},
                )
                try:
                    while True:
                        next(stream)
                except StopIteration as exc:
                    outcome["result"] = exc.value

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(provider_started.wait(timeout=1))

        stop_requested.set()
        consumer.join(timeout=2)

        self.assertFalse(consumer.is_alive())
        self.assertTrue(cancellation_checkpointed.wait(timeout=1))
        self.assertTrue(tool_registry.is_tool_execution_cancelled(outcome.get("result")))

    def test_ollama_cancellation_forces_the_next_round_to_run_without_tools(self):
        ollama_api._abort_event.clear()
        self.addCleanup(ollama_api._abort_event.clear)
        round_tools = []
        round_prompts = []
        round_results = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "deep_research__deep_research",
                        "arguments": {"topic": "Test"},
                    }
                }],
            },
            {"role": "assistant", "content": "Stopped as requested."},
        ]

        def fake_stream_round(_client, _model, conversation, _kwargs, *, tools=None):
            round_tools.append(tools)
            round_prompts.append(str(conversation[-1].get("content") or ""))
            result = round_results[len(round_tools) - 1]

            def stream():
                if False:
                    yield {}
                return result

            return stream()

        def cancelled_tool_stream(*_args, **_kwargs):
            if False:
                yield {}
            return tool_registry.TOOL_EXECUTION_CANCELLED

        event = {
            "alias": "deep_research__deep_research__0",
            "server_id": "deep-research",
            "server_name": "Deep Research",
            "tool_id": "deep_research",
            "tool_name": "Deep Research",
            "arguments": {"topic": "Test"},
        }
        with (
            patch.object(ollama_api, "_stream_round", side_effect=fake_stream_round),
            patch.object(
                tool_registry,
                "build_ollama_tools",
                return_value=([{"type": "function"}], {}),
            ),
            patch.object(tool_registry, "prepare_tool_calls", side_effect=lambda _lookup, calls: calls),
            patch.object(
                tool_registry,
                "limit_tool_calls_from_context",
                side_effect=lambda _context, calls, _lookup: calls,
            ),
            patch.object(ollama_api, "_build_tool_event", return_value=event),
            patch.object(tool_registry, "stream_ollama_tool", side_effect=cancelled_tool_stream),
        ):
            list(
                ollama_api._run_tool_loop(
                    SimpleNamespace(),
                    "gemma-test",
                    [{"role": "user", "content": "Research this"}],
                    {},
                    ["deep-research"],
                    {},
                )
            )

        self.assertIsNotNone(round_tools[0])
        self.assertIsNone(round_tools[1])
        self.assertIn("Do not call any tool again", round_prompts[1])

    def test_main_chat_abort_does_not_start_a_cancellation_acknowledgement_round(self):
        ollama_api._abort_event.set()
        self.addCleanup(ollama_api._abort_event.clear)
        round_count = 0

        def fake_stream_round(_client, _model, _conversation, _kwargs, *, tools=None):
            nonlocal round_count
            round_count += 1
            if round_count > 1:
                raise AssertionError("main abort must not start another provider request")

            def stream():
                if False:
                    yield {}
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "deep_research__deep_research",
                            "arguments": {"topic": "Test"},
                        }
                    }],
                }

            return stream()

        def cancelled_tool_stream(*_args, **_kwargs):
            if False:
                yield {}
            return tool_registry.TOOL_EXECUTION_CANCELLED

        event = {
            "alias": "deep_research__deep_research__0",
            "server_id": "deep-research",
            "server_name": "Deep Research",
            "tool_id": "deep_research",
            "tool_name": "Deep Research",
            "arguments": {"topic": "Test"},
        }
        with (
            patch.object(ollama_api, "_stream_round", side_effect=fake_stream_round),
            patch.object(
                tool_registry,
                "build_ollama_tools",
                return_value=([{"type": "function"}], {}),
            ),
            patch.object(tool_registry, "prepare_tool_calls", side_effect=lambda _lookup, calls: calls),
            patch.object(
                tool_registry,
                "limit_tool_calls_from_context",
                side_effect=lambda _context, calls, _lookup: calls,
            ),
            patch.object(ollama_api, "_build_tool_event", return_value=event),
            patch.object(tool_registry, "stream_ollama_tool", side_effect=cancelled_tool_stream),
        ):
            list(
                ollama_api._run_tool_loop(
                    SimpleNamespace(),
                    "gemma-test",
                    [{"role": "user", "content": "Research this"}],
                    {},
                    ["deep-research"],
                    {},
                )
            )

        self.assertEqual(round_count, 1)

    def test_cancelled_persistent_worker_is_not_restarted(self):
        cancellation = tool_registry.ActiveToolExecution("generation-a")
        session = tool_registry.ExternalWorkerSession(Path("Tools/example/mcp-server.py"), Path("python"))
        session.process = SimpleNamespace(stdin=Mock(), stdout=Mock())

        def cancelled_read(_timeout):
            cancellation.cancelled.set()
            return ""

        with (
            patch.object(session, "_start") as start_mock,
            patch.object(session, "_read_response_line", side_effect=cancelled_read),
            patch.object(session, "close") as close_mock,
        ):
            with self.assertRaises(tool_registry.ToolExecutionCancelled):
                session.request("call", {}, timeout_s=1, cancellation=cancellation)

        start_mock.assert_called_once_with()
        close_mock.assert_called_once_with()

    def test_cancelled_during_persistent_worker_start_is_closed_before_request(self):
        cancellation = tool_registry.ActiveToolExecution("generation-a")
        session = tool_registry.ExternalWorkerSession(Path("Tools/example/mcp-server.py"), Path("python"))

        def cancel_during_start():
            cancellation.cancelled.set()

        with (
            patch.object(session, "_start", side_effect=cancel_during_start) as start_mock,
            patch.object(session, "close") as close_mock,
        ):
            with self.assertRaises(tool_registry.ToolExecutionCancelled):
                session.request("call", {}, timeout_s=1, cancellation=cancellation)

        start_mock.assert_called_once_with()
        close_mock.assert_called_once_with()

    @patch("API.llm_api.tool_registry.abort_active_tools")
    @patch("API.llm_api._get_engine_module")
    def test_llm_abort_forwards_generation_id_to_tools(self, get_module, abort_tools):
        adapter = SimpleNamespace(abort_generation=Mock())
        get_module.return_value = adapter

        llm_api.abort_generation("ollama-service", generation_id="generation-a")

        abort_tools.assert_called_once_with("generation-a")
        adapter.abort_generation.assert_called_once_with()

# Cover per-response tool quota guardrails.
class ToolQuotaTests(SimpleTestCase):
    # High-effort web search is expensive, so keep it bounded per response.
    def test_high_effort_web_search_limits_to_three_calls(self):
        tool_event = {"tool_id": "web_search", "tool_name": "Web search"}
        counters: dict[str, int] = {}
        arguments = {"query": "cheap coding model", "effort": "high"}

        self.assertIsNone(tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments))
        self.assertIsNone(tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments))
        self.assertIsNone(tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments))

        error = tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments)
        self.assertIsNotNone(error)
        self.assertIn("high mode is unavailable", str(error))
        self.assertIn("use medium or low", str(error))

    # Lower-effort searches keep the existing broader budget.
    def test_normal_web_search_keeps_default_quota(self):
        tool_event = {"tool_id": "web_search", "tool_name": "Web search"}
        counters: dict[str, int] = {}
        arguments = {"query": "cheap coding model", "effort": "medium"}

        for _index in range(4):
            self.assertIsNone(tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments))

    def test_instant_search_and_read_page_each_allow_two_calls(self):
        counters: dict[str, int] = {}
        for tool_id in ("web_search", "read_page"):
            event = {"tool_id": tool_id, "tool_name": tool_id, "instant_mode": True}
            self.assertIsNone(tool_registry.consume_tool_quota(event, counters, arguments={}))
            self.assertIsNone(tool_registry.consume_tool_quota(event, counters, arguments={}))
            self.assertIn(
                "at most 2 times",
                str(tool_registry.consume_tool_quota(event, counters, arguments={})),
            )

class InstantToolSchemaTests(SimpleTestCase):
    def test_all_engine_adapters_forward_forced_instant_batch_size(self):
        context = {"instant_mode": True, "instant_search_batch_size": 3}
        adapter_cases = (
            (
                ollama_api,
                lambda: list(ollama_api._run_tool_loop(
                    None, "model", [{"role": "user", "content": "Find"}], {}, ["web_search"], context
                )),
                "_stream_round",
            ),
            (
                openai_api,
                lambda: list(openai_api._run_tool_loop(
                    None, "model", [{"role": "user", "content": "Find"}], {}, ["web_search"], context
                )),
                "_stream_openai_round",
            ),
            (
                lms_api,
                lambda: list(lms_api._run_tool_loop(
                    None, "model", [{"role": "user", "content": "Find"}], {}, ["web_search"], context
                )),
                "_stream_openai_round",
            ),
        )

        for adapter, invoke, stream_name in adapter_cases:
            with self.subTest(adapter=adapter.__name__):
                with (
                    patch.object(tool_registry, "build_ollama_tools", return_value=([], {})) as builder,
                    patch.object(adapter, stream_name, return_value=iter(())),
                ):
                    invoke()
                self.assertEqual(builder.call_args.kwargs["instant_search_batch_size"], 3)

        with patch.object(tool_registry, "build_ollama_tools", return_value=([], {})) as builder:
            google_genai_api._build_google_tools(["web_search"], "model", context)
        self.assertEqual(builder.call_args.kwargs["instant_search_batch_size"], 3)

    def test_instant_search_schema_is_minimal_and_thinking_schema_is_unchanged(self):
        server = {
            "id": "web_search",
            "name": "Web Search",
            "description": "Search tools",
            "tools": [
                {
                    "id": "web_search",
                    "alias": "web_search__web_search",
                    "name": "Web Search",
                    "description": "Full search",
                    "parameters": {
                        "type": "object",
                        "properties": {"web": {}, "shopping": {}, "effort": {}},
                    },
                    "prepares_arguments": True,
                },
                {
                    "id": "read_page",
                    "alias": "web_search__read_page",
                    "name": "Read Page",
                    "description": "Read pages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}},
                                ]
                            }
                        },
                    },
                    "prepares_arguments": False,
                },
            ],
        }
        with patch.object(tool_registry, "get_server", return_value=server):
            instant_tools, instant_lookup = tool_registry.build_ollama_tools(
                ["web_search"], instant_mode=True
            )
            forced_instant_tools, forced_instant_lookup = tool_registry.build_ollama_tools(
                ["web_search"],
                instant_mode=True,
                instant_search_batch_size=3,
            )
            thinking_tools, _thinking_lookup = tool_registry.build_ollama_tools(["web_search"])

        instant_schema = instant_tools[0]["function"]["parameters"]
        self.assertEqual(set(instant_schema["properties"]), {"query", "description", "operators"})
        self.assertEqual(instant_schema["properties"]["query"]["oneOf"][1]["maxItems"], 3)
        self.assertTrue(instant_lookup["web_search__web_search"]["tool"]["instant_mode"])
        forced_query_schema = forced_instant_tools[0]["function"]["parameters"]["properties"]["query"]
        self.assertEqual(forced_query_schema["type"], "array")
        self.assertEqual(forced_query_schema["minItems"], 3)
        self.assertEqual(forced_query_schema["maxItems"], 3)
        self.assertEqual(
            forced_instant_lookup["web_search__web_search"]["tool"]["instant_search_batch_size"],
            3,
        )
        self.assertIn("effort", thinking_tools[0]["function"]["parameters"]["properties"])

    def test_forced_instant_search_preflight_requires_exactly_three_queries(self):
        preparer = Mock(return_value={
            "ok": True,
            "arguments": {
                "query": ["alpha", "beta", "gamma"],
                "description": "Checking sources",
            },
            "tool_ui": {"kind": "web_search", "status": "pending", "instant_mode": True},
        })
        tool = {
            "id": "web_search",
            "alias": "web_search__web_search",
            "instant_mode": True,
            "instant_search_batch_size": 3,
            "prepares_arguments": True,
        }
        server = {
            "id": "web_search",
            "external": False,
            "tool_preparers": {"web_search": preparer},
        }
        lookup = {tool["alias"]: {"server": server, "tool": tool}}

        rejected = tool_registry.prepare_tool_call(lookup, {
            "name": tool["alias"],
            "arguments": {"query": "alpha", "description": "Checking sources"},
        })
        accepted = tool_registry.prepare_tool_call(lookup, {
            "name": tool["alias"],
            "arguments": {
                "query": ["alpha", "beta", "gamma"],
                "description": "Checking sources",
            },
        })

        self.assertEqual(
            rejected["preflight_error_result"]["ui"]["error"]["code"],
            "INVALID_INSTANT_SEARCH_BATCH",
        )
        self.assertEqual(rejected["tool_ui"]["query_count"], 0)
        self.assertNotIn("preflight_error_result", accepted)
        preparer.assert_called_once()

        preparer.reset_mock()
        preparer.return_value = {
            "ok": True,
            "arguments": {
                "query": ["alpha", "beta", "gamma"],
                "description": "Checking alpha · Checking beta · Checking gamma",
            },
            "tool_ui": {"kind": "web_search", "status": "pending", "instant_mode": True},
        }
        collapsed = tool_registry.prepare_tool_calls(lookup, [
            {"name": tool["alias"], "arguments": {"query": "alpha", "description": "Checking alpha"}},
            {"name": tool["alias"], "arguments": {"query": "beta", "description": "Checking beta"}},
            {"name": tool["alias"], "arguments": {"query": "gamma", "description": "Checking gamma"}},
        ])

        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["arguments"]["query"], ["alpha", "beta", "gamma"])
        self.assertEqual(collapsed[0]["tool_ui"]["collapsed_parallel_calls"], 3)
        preparer.assert_called_once()

    def test_in_process_tool_preparer_is_discovered_and_executed(self):
        preparer = Mock(return_value={
            "ok": True,
            "arguments": {"query": "alpha", "description": "Checking alpha"},
            "tool_ui": {"kind": "web_search", "status": "pending", "instant_mode": True},
        })
        module = SimpleNamespace(
            MCP_SERVER={"id": "web_search", "name": "Web Search"},
            TOOLS=[{
                "id": "web_search",
                "name": "Web Search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            }],
            TOOL_PREPARERS={"web_search": preparer},
            call_tool=lambda *_args, **_kwargs: {},
        )
        server = tool_registry._extract_server_definition(
            module,
            "mcp-web-search",
            Path("Tools/mcp-web-search/mcp-server.py"),
        )
        tool = dict(server["tools"][0])
        tool["instant_mode"] = True
        lookup = {tool["alias"]: {"server": server, "tool": tool}}

        prepared = tool_registry.prepare_tool_calls(lookup, [{
            "id": "call-1",
            "name": tool["alias"],
            "arguments": {"query": "alpha", "description": "Checking alpha"},
        }])

        self.assertTrue(server["tools"][0]["prepares_arguments"])
        preparer.assert_called_once()
        self.assertTrue(preparer.call_args.args[0]["_instant_mode"])
        self.assertTrue(prepared[0]["tool_ui"]["instant_mode"])

    def test_ollama_instant_search_read_and_retry_still_reaches_final_answer(self):
        server = {"id": "web_search", "name": "Web Search"}
        search_tool = {
            "id": "web_search",
            "alias": "web_search__web_search",
            "name": "Web Search",
            "instant_mode": True,
        }
        read_tool = {
            "id": "read_page",
            "alias": "web_search__read_page",
            "name": "Read Page",
            "instant_mode": True,
        }
        lookup = {
            search_tool["alias"]: {"server": server, "tool": search_tool},
            read_tool["alias"]: {"server": server, "tool": read_tool},
        }
        round_results = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": search_tool["alias"],
                        "arguments": {"query": "first", "description": "Finding sources"},
                    }
                }],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": read_tool["alias"],
                        "arguments": {"url": "https://example.com/source"},
                    }
                }],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": search_tool["alias"],
                        "arguments": {"query": "retry", "description": "Retrying search"},
                    }
                }],
            },
            {"role": "assistant", "content": "Final answer"},
        ]

        def fake_stream_round(_client, _model, _conversation, _kwargs, *, tools=None):
            result = round_results.pop(0)

            def stream():
                if False:
                    yield {}
                return result

            return stream()

        def fake_tool_stream(*_args, **_kwargs):
            if False:
                yield {}
            return {
                "model_context": "Tool result",
                "sources": [],
                "ui": {"kind": "web_search", "status": "done", "instant_mode": True},
            }

        with (
            patch.object(ollama_api, "_stream_round", side_effect=fake_stream_round),
            patch.object(tool_registry, "build_ollama_tools", return_value=([{"type": "function"}], lookup)),
            patch.object(tool_registry, "stream_ollama_tool", side_effect=fake_tool_stream),
        ):
            chunks = list(
                ollama_api._run_tool_loop(
                    SimpleNamespace(),
                    "test-model",
                    [{"role": "user", "content": "Find this"}],
                    {},
                    ["web_search"],
                    {"instant_mode": True},
                )
            )

        self.assertFalse(round_results)
        self.assertTrue(any(
            chunk.get("transcript_message", {}).get("content") == "Final answer"
            for chunk in chunks
            if isinstance(chunk, dict)
        ))
        self.assertFalse(any(
            "tool loop exceeded" in str(chunk).lower()
            for chunk in chunks
        ))

class ToolCooldownTests(SimpleTestCase):
    def setUp(self):
        tool_registry._TOOL_COOLDOWNS.clear()

    def tearDown(self):
        tool_registry._TOOL_COOLDOWNS.clear()

    # Identical tool arguments in separate chats must not share a cooldown.
    def test_web_search_cooldown_is_scoped_by_chat_id(self):
        tool_event = {"tool_id": "web_search", "tool_name": "Web search"}
        arguments = {"query": "same query", "effort": "medium"}
        first_chat = {"chat_id": "chat-a"}
        second_chat = {"chat_id": "chat-b"}

        tool_registry.remember_tool_cooldown(tool_event, arguments, context=first_chat)

        self.assertIsNotNone(
            tool_registry.consume_tool_cooldown(tool_event, arguments, context=first_chat)
        )
        self.assertIsNone(
            tool_registry.consume_tool_cooldown(tool_event, arguments, context=second_chat)
        )

    # The scope applies to page reads as well as searches.
    def test_read_page_cooldown_is_scoped_by_chat_id(self):
        tool_event = {"tool_id": "read_page", "tool_name": "Read page"}
        arguments = {"url": "https://example.com/page"}

        tool_registry.remember_tool_cooldown(
            tool_event, arguments, context={"chat_id": "chat-a"}
        )

        self.assertIsNone(
            tool_registry.consume_tool_cooldown(
                tool_event, arguments, context={"chat_id": "chat-b"}
            )
        )

class ToolPreflightTests(SimpleTestCase):
    def _lookup(self):
        tool = {
            "id": "web_search",
            "alias": "web_search__web_search",
            "name": "Web Search",
            "prepares_arguments": True,
        }
        server = {
            "id": "web_search",
            "name": "Web Search",
            "external": True,
            "server_file": Path("Tools/mcp-web-search/mcp-server.py"),
        }
        return {"web_search__web_search": {"server": server, "tool": tool}}

    @patch.object(tool_registry, "_run_worker")
    def test_preflight_replaces_raw_arguments_before_tool_event(self, worker_mock):
        worker_mock.return_value = {
            "ok": True,
            "arguments": {
                "call_description": "Verify canonical sources",
                "web": "canonical",
                "effort": "medium",
            },
            "tool_ui": {
                "kind": "web_search",
                "status": "pending",
                "description": "Verify canonical sources",
                "search_request": {
                    "schema_mode": "advanced",
                    "description": "Verify canonical sources",
                    "effort": "medium",
                    "queries": [{"vertical": "web", "compiled_query": "canonical", "operators": {}}],
                },
            },
        }
        raw = {
            "call_description": "Verify raw sources",
            "web": "raw model value",
        }
        call = tool_registry.prepare_tool_call(
            self._lookup(),
            {"name": "web_search__web_search", "arguments": raw},
        )
        event = tool_registry.build_tool_event(self._lookup(), call)

        self.assertEqual(call["raw_arguments"], raw)
        self.assertNotIn("raw model value", json.dumps(event))
        self.assertEqual(event["arguments"]["web"], "canonical")
        self.assertEqual(event["tool_ui"]["description"], "Verify canonical sources")
        self.assertEqual(event["tool_ui"]["search_request"]["queries"][0]["compiled_query"], "canonical")
        marker = _serialize_tool_call_marker(event)
        self.assertIn('"tool_ui"', marker)
        self.assertNotIn("raw model value", marker)

        transcript = tool_registry.canonicalize_transcript_tool_calls(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "web_search__web_search",
                            "arguments": json.dumps(raw),
                        },
                    }
                ],
            },
            [call],
        )
        self.assertNotIn("raw model value", json.dumps(transcript))
        transcript_arguments = transcript["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(transcript_arguments), event["arguments"])

    @patch.object(tool_registry, "_run_worker")
    def test_rejected_preflight_keeps_structured_error_for_adapter_short_circuit(self, worker_mock):
        error_result = {
            "model_context": "INVALID_SEARCH_PLAN: $: must include a vertical query",
            "sources": [],
            "ui": {"kind": "web_search", "status": "rejected"},
        }
        worker_mock.return_value = {
            "ok": False,
            "arguments": {},
            "tool_ui": error_result["ui"],
            "error_result": error_result,
        }
        raw_arguments = {"query": "legacy"}
        call = tool_registry.prepare_tool_call(
            self._lookup(),
            {"name": "web_search__web_search", "arguments": raw_arguments},
        )

        self.assertEqual(call["arguments"], {})
        self.assertEqual(call["preflight_error_result"], error_result)
        self.assertEqual(call["tool_ui"]["status"], "rejected")
        self.assertEqual(call["tool_ui"]["rejected_arguments"], raw_arguments)
        self.assertTrue(tool_registry.is_blocking_tool_result(call["preflight_error_result"]))

    @patch.object(tool_registry, "_run_worker")
    def test_parallel_advanced_search_calls_are_preflighted_as_one_batch(self, worker_mock):
        def prepare_batch(_server_file, _operation, payload, persistent=False):
            self.assertTrue(persistent)
            arguments = payload["arguments"]
            self.assertEqual(arguments["web"], ["first query", "second query"])
            return {
                "ok": True,
                "arguments": arguments,
                "tool_ui": {
                    "kind": "web_search",
                    "status": "pending",
                    "query_count": 2,
                },
            }

        worker_mock.side_effect = prepare_batch
        calls = [
            {
                "id": "call-1",
                "name": "web_search__web_search",
                "arguments": {
                    "call_description": "Check first source",
                    "web": "first query",
                    "effort": "medium",
                },
            },
            {
                "id": "call-2",
                "name": "web_search__web_search",
                "arguments": {
                    "call_description": "Check second source",
                    "web": "second query",
                    "effort": "medium",
                },
            },
        ]

        prepared = tool_registry.prepare_tool_calls(self._lookup(), calls)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["id"], "call-1")
        self.assertEqual(prepared[0]["parallel_batch_size"], 2)
        self.assertEqual([call["id"] for call in prepared[0]["absorbed_tool_calls"]], ["call-2"])
        worker_mock.assert_called_once()

    @patch.object(tool_registry, "_run_worker")
    def test_parallel_legacy_search_batch_over_server_limit_is_one_atomic_rejection(self, worker_mock):
        def reject_oversized_batch(_server_file, _operation, payload, persistent=False):
            self.assertTrue(persistent)
            query_count = len(payload["arguments"]["query"])
            self.assertEqual(query_count, 4)
            error_result = {
                "error": {
                    "code": "INVALID_SEARCH_PLAN",
                    "issues": [{"path": "$", "message": "batch permits at most 2 queries total"}],
                },
                "sources": [],
                "model_context": "INVALID_SEARCH_PLAN: $: batch permits at most 2 queries total",
                "ui": {"kind": "web_search", "status": "rejected", "query_count": 0},
            }
            return {
                "ok": False,
                "arguments": {},
                "tool_ui": error_result["ui"],
                "error_result": error_result,
            }

        worker_mock.side_effect = reject_oversized_batch
        calls = [
            {
                "id": f"call-{index}",
                "name": "web_search__web_search",
                "arguments": {
                    "call_description": f"Check source {index}",
                    "query": f"query {index}",
                    "effort": "medium",
                },
            }
            for index in range(1, 5)
        ]

        prepared = tool_registry.prepare_tool_calls(self._lookup(), calls)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["tool_ui"]["status"], "rejected")
        self.assertIn("4 parallel web_search calls", prepared[0]["preflight_error_result"]["model_context"])
        self.assertIn("batch permits at most 2 queries total", prepared[0]["preflight_error_result"]["model_context"])
        worker_mock.assert_called_once()

        transcript = tool_registry.canonicalize_transcript_tool_calls(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call["id"],
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                    for call in calls
                ],
                "google_parts": [
                    {
                        "function_call": {
                            "name": call["name"],
                            "args": call["arguments"],
                        },
                        "thought_signature": f"signature-{call['id']}",
                    }
                    for call in calls
                ],
            },
            prepared,
        )
        self.assertEqual([call["id"] for call in transcript["tool_calls"]], ["call-1"])
        self.assertEqual(len(transcript["google_parts"]), 1)

# Cover local tool-server listing and chat persistence endpoints.
class ToolApiTests(ToolRegistryTestMixin, TestCase):
    def setUp(self):
        super().setUp()
        self._engine_enabled_patch = patch(
            "Apps.API.views.settings.is_engine_enabled",
            return_value=True,
        )
        self._engine_enabled_patch.start()

    def tearDown(self):
        self._engine_enabled_patch.stop()
        super().tearDown()

    # Test tools API returns discovered servers.
    def test_tools_api_returns_discovered_servers(self):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite", "description": "Time helpers"}
            TOOLS = [{"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}}]
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )

        response = self.client.get(reverse("tools_api"), {"engine": "ollama-service", "model": "llama3"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["tool_servers"],
            [{
                "id": "time_suite",
                "name": "Time Suite",
                "description": "Time helpers",
                "tool_count": 1,
                "tools": [{"id": "time_now", "name": "Current Time", "description": ""}],
            }],
        )

    # Test load chat API returns active tool server id.
    def test_load_chat_api_returns_active_tool_server_id(self):
        chat = Chat.objects.create(title="Chat", active_tool_slug="time_suite")
        Message.objects.create(
            chat=chat,
            role="assistant",
            content="Visible answer",
            llm_transcript=[
                {"role": "assistant", "content": "", "thinking": "Plan"},
                {"role": "tool", "name": "time_suite__time_now", "tool_name": "time_suite__time_now", "content": '{"ok": true}', "server_id": "time_suite", "server_name": "Time Suite", "tool_id": "time_now", "tool_display_name": "Current Time", "arguments": {"label": "now"}},
                {"role": "assistant", "content": "Visible answer"},
            ],
        )

        response = self.client.get(reverse("load_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["active_tool_server_id"], "time_suite")
        self.assertEqual(payload["messages"][0]["content"], "Visible answer")
        self.assertEqual(payload["messages"][0]["activity_segments"][0]["type"], "thought")
        self.assertEqual(payload["messages"][0]["activity_segments"][1]["type"], "tool")

    # Test load chat API returns all active tool server ids.
    def test_load_chat_api_returns_multiple_active_tool_server_ids(self):
        chat = Chat.objects.create(title="Chat", active_tool_slug='["time_suite", "browser"]')
        Message.objects.create(chat=chat, role="user", content="Hello")

        response = self.client.get(reverse("load_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["active_tool_server_ids"], ["time_suite", "browser"])
        self.assertEqual(payload["active_tool_server_id"], "time_suite")

    # Test load chat API returns attachment metadata without inline data.
    def test_load_chat_api_returns_attachment_metadata_without_inline_data(self):
        chat = Chat.objects.create(title="Chat")
        message = Message.objects.create(chat=chat, role="user", content="See file")
        attachment = MessageAttachment.objects.create(
            message=message,
            kind="file",
            name="note.txt",
            mime_type="text/plain",
            data="SGVsbG8=",
            size_bytes=5,
        )

        response = self.client.get(reverse("load_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        serialized = payload["messages"][0]["attachments"][0]
        self.assertEqual(serialized["id"], attachment.id)
        self.assertEqual(serialized["content_url"], f"/api/attachment/attachment/{attachment.id}/content/")
        self.assertNotIn("data_url", serialized)
        self.assertNotIn("extracted_text", serialized)

    # Test attachment content API streams stored bytes on demand.
    def test_attachment_content_api_streams_stored_bytes(self):
        chat = Chat.objects.create(title="Chat")
        message = Message.objects.create(chat=chat, role="user", content="See file")
        attachment = MessageAttachment.objects.create(
            message=message,
            kind="file",
            name="note.txt",
            mime_type="text/plain",
            data="SGVsbG8=",
            size_bytes=5,
        )

        response = self.client.get(reverse("attachment_content_api", args=["attachment", attachment.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"Hello")
        self.assertEqual(response["Content-Type"], "text/plain")

    # Test attachment content API streams legacy image records.
    def test_attachment_content_api_streams_legacy_image_bytes(self):
        chat = Chat.objects.create(title="Chat")
        message = Message.objects.create(chat=chat, role="user", content="See image")
        image = MessageImage.objects.create(
            message=message,
            mime_type="image/png",
            data="SGVsbG8=",
            order=2,
        )

        response = self.client.get(reverse("attachment_content_api", args=["image", image.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"Hello")
        self.assertEqual(response["Content-Type"], "image/png")
        self.assertIn('filename="image-3"', response["Content-Disposition"])

    # Test attachment content API rejects unknown record types.
    def test_attachment_content_api_rejects_unknown_record_type(self):
        response = self.client.get(reverse("attachment_content_api", args=["unknown", 1]))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "Unknown attachment type")

    # Test delete last assistant API returns the user message to regenerate.
    def test_delete_last_assistant_api_returns_user_message_for_regeneration(self):
        chat = Chat.objects.create(title="Chat")
        user_message = Message.objects.create(chat=chat, role="user", content="See this")
        MessageAttachment.objects.create(
            message=user_message,
            kind=MessageAttachmentKind.IMAGE,
            name="photo.png",
            mime_type="image/png",
            data="iVBORw0KGgo=",
            size_bytes=8,
        )
        assistant_message = Message.objects.create(chat=chat, role="assistant", content="Answer")

        response = self.client.delete(reverse("delete_last_assistant_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["user_message"]["content"], "See this")
        self.assertEqual(payload["user_message"]["attachments"][0]["name"], "photo.png")
        self.assertEqual(payload["user_message"]["images"], ["data:image/png;base64,iVBORw0KGgo="])
        self.assertFalse(Message.objects.filter(id=assistant_message.id).exists())

    # Test delete last assistant API rejects chats ending with a user message.
    def test_delete_last_assistant_api_rejects_when_last_message_is_user(self):
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Still pending")

        response = self.client.delete(reverse("delete_last_assistant_api", args=[chat.id]))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Last message is not from assistant")

    # Test delete message API removes only the selected message.
    def test_delete_message_api_removes_selected_message(self):
        chat = Chat.objects.create(title="Chat")
        first = Message.objects.create(chat=chat, role="user", content="First")
        second = Message.objects.create(chat=chat, role="assistant", content="Second")

        response = self.client.delete(reverse("delete_message_api", args=[first.id]))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Message.objects.filter(id=first.id).exists())
        self.assertTrue(Message.objects.filter(id=second.id).exists())

    # A branch is a lossless copy through the selected turn with links on both sides.
    def test_branch_message_api_copies_history_and_exposes_bidirectional_markers(self):
        chat = Chat.objects.create(title="Original", active_tool_slug='["sandbox"]')
        first = Message.objects.create(chat=chat, role="user", content="First")
        MessageAttachment.objects.create(
            message=first,
            kind=MessageAttachmentKind.IMAGE,
            name="image.png",
            mime_type="image/png",
            data="aW1hZ2U=",
            size_bytes=5,
        )
        target = Message.objects.create(
            chat=chat,
            role="assistant",
            content="Answer",
            llm_transcript=[{"role": "assistant", "thinking": "Full thought", "content": "Answer"}],
        )
        Message.objects.create(chat=chat, role="user", content="Must not be copied")

        response = self.client.post(reverse("branch_message_api", args=[target.id]), data="{}", content_type="application/json")

        self.assertEqual(response.status_code, 201)
        child = Chat.objects.get(id=response.json()["chat_id"])
        self.assertEqual(child.active_tool_slug, '["sandbox"]')
        copied = list(child.messages.order_by("created_at", "id"))
        self.assertEqual([item.content for item in copied], ["First", "Answer"])
        self.assertEqual(copied[1].llm_transcript[0]["thinking"], "Full thought")
        self.assertEqual(copied[0].attachments.get().data, "aW1hZ2U=")
        branch = ChatBranch.objects.get(child_chat=child)
        self.assertEqual(branch.source_message_id, target.id)
        self.assertEqual(branch.child_message_id, copied[1].id)

        source_payload = self.client.get(reverse("load_chat_api", args=[chat.id])).json()
        child_payload = self.client.get(reverse("load_chat_api", args=[child.id])).json()
        source_target = next(item for item in source_payload["messages"] if item["id"] == target.id)
        child_target = next(item for item in child_payload["messages"] if item["id"] == copied[1].id)
        self.assertEqual(source_target["branch_links"][0]["direction"], "to_branch")
        self.assertEqual(source_target["branch_links"][0]["chat_id"], str(child.id))
        self.assertEqual(child_target["branch_links"][0]["direction"], "to_origin")
        self.assertEqual(child_target["branch_links"][0]["chat_id"], str(chat.id))

    # Editing a user message invalidates every later turn while keeping attachments.
    def test_edit_message_api_replaces_user_text_and_truncates_continuation(self):
        chat = Chat.objects.create(title="Chat")
        target = Message.objects.create(chat=chat, role="user", content="Old text")
        attachment = MessageAttachment.objects.create(
            message=target,
            kind=MessageAttachmentKind.FILE,
            name="note.txt",
            mime_type="text/plain",
            data="bm90ZQ==",
            size_bytes=4,
        )
        stale = Message.objects.create(chat=chat, role="assistant", content="Stale answer")

        response = self.client.patch(
            reverse("edit_message_api", args=[target.id]),
            data=json.dumps({"content": "New\ntext"}),
            content_type="application/json",
        )

        target.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(target.content, "New\ntext")
        self.assertFalse(Message.objects.filter(id=stale.id).exists())
        self.assertTrue(MessageAttachment.objects.filter(id=attachment.id).exists())

    # The archive contains readable linked Markdown, full transcript JSON and binary assets.
    def test_export_chat_api_returns_lossless_zip(self):
        chat = Chat.objects.create(title="Export test")
        message = Message.objects.create(
            chat=chat,
            role="assistant",
            content="Supported claim [cabc-1]",
            llm_transcript=[
                {"role": "assistant", "thinking": "Do not shorten", "content": "Supported claim [cabc-1]"},
                {
                    "role": "tool",
                    "content": "result",
                    "structured_content": {"sources": [{"id": "cabc-1", "url": "https://example.com/source"}]},
                },
            ],
        )
        MessageAttachment.objects.create(
            message=message,
            kind=MessageAttachmentKind.IMAGE,
            name="chart.png",
            mime_type="image/png",
            data="cG5nLWJ5dGVz",
            size_bytes=9,
        )

        response = self.client.get(reverse("export_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(set(archive.namelist()), {"chat.md", "chat.json", "attachments/chart.png"})
            markdown = archive.read("chat.md").decode("utf-8")
            payload = json.loads(archive.read("chat.json"))
            self.assertIn("[CABC-1](https://example.com/source)", markdown)
            self.assertIn('"thinking": "Do not shorten"', markdown)
            self.assertEqual(payload["messages"][0]["llm_transcript"][0]["thinking"], "Do not shorten")
            self.assertEqual(archive.read("attachments/chart.png"), b"png-bytes")

    # Test rename chat API trims and persists the title.
    def test_rename_chat_api_updates_title(self):
        chat = Chat.objects.create(title="Old")

        response = self.client.patch(
            reverse("rename_chat_api", args=[chat.id]),
            data='{"title":"  New title  "}',
            content_type="application/json",
        )

        chat.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "New title")
        self.assertEqual(chat.title, "New title")

    # Test delete chat API removes the whole thread.
    def test_delete_chat_api_removes_thread_and_messages(self):
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Hello")

        response = self.client.delete(reverse("delete_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Chat.objects.filter(id=chat.id).exists())
        self.assertEqual(Message.objects.count(), 0)
