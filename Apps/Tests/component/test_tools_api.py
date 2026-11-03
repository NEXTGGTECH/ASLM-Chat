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


# MCP reload

# Cover the MCP reload endpoint.
class McpReloadApiTests(SimpleTestCase):
    # Verify that a POST resets the tool registry cache and the server list cache, then returns the
    # freshly listed servers for the active engine and the requested model.
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

    # Verify that the reload endpoint answers a GET request with 405 Method Not Allowed.
    def test_reload_rejects_non_post_requests(self):
        response = self.client.get(reverse("mcp_reload_api"))
        self.assertEqual(response.status_code, 405)


# Tool cancellation

# Cover cancellation of running tools by generation or research session, including worker handling
# and the follow-up behavior of the Ollama tool loop.
class ToolCancellationTests(SimpleTestCase):
    # Abort every tool execution still registered and clear the registry between tests.
    def tearDown(self):
        tool_registry.abort_active_tools()
        with tool_registry._ACTIVE_TOOL_EXECUTIONS_LOCK:
            tool_registry._ACTIVE_TOOL_EXECUTIONS.clear()

    # Verify that aborting one generation id cancels only the executions of that generation and calls
    # only their cancel callbacks.
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

    # Verify that aborting one research session id cancels only the execution bound to that session;
    # another session and an execution without a research session of the same generation keep running.
    def test_abort_active_research_only_cancels_matching_session(self):
        # Arrange: two executions with different research sessions and one without a session, each
        # registered with its own cancel callback.
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

        # Act: abort the first research session.
        cancelled_count = tool_registry.abort_active_research(
            "deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )

        # Assert: only the first execution and its callback are cancelled.
        self.assertEqual(cancelled_count, 1)
        self.assertTrue(first.cancelled.is_set())
        self.assertFalse(second.cancelled.is_set())
        self.assertFalse(unrelated.cancelled.is_set())
        first_callback.assert_called_once_with()
        second_callback.assert_not_called()
        unrelated_callback.assert_not_called()

    # Verify that a streamed in-process tool receives a cancel event in its context, that aborting its
    # generation sets the event so the handler stops, and that the stream ends with the cancelled result.
    def test_streamed_in_process_tool_receives_cancel_event_and_stops(self):
        started = threading.Event()
        stopped = threading.Event()

        # Block until the cancel event from the context is set, then report that the handler stopped.
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

        # Drain the tool stream on a worker thread and store its final return value in outcome.
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

        # Act: start the stream, wait until the handler runs, then abort the generation.
        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(started.wait(timeout=1))

        self.assertEqual(tool_registry.abort_active_tools("generation-a"), 1)
        consumer.join(timeout=1)

        # Assert: the consumer finished, the handler stopped and the stream result is the cancelled marker.
        self.assertFalse(consumer.is_alive())
        self.assertTrue(stopped.wait(timeout=1))
        self.assertTrue(tool_registry.is_tool_execution_cancelled(outcome.get("result")))

    # Verify that when a research stop interrupts the worker and the call fails with a worker error,
    # the stream returns the cancelled result instead of raising.
    def test_research_stop_maps_interrupted_worker_error_to_cancelled_result(self):
        session_id = "deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        started = threading.Event()
        outcome = {}

        # Wait for the cancellation, then fail like a worker process that was terminated.
        def interrupted_call(*_args, **kwargs):
            cancellation = kwargs["cancellation"]
            started.set()
            cancellation.cancelled.wait(timeout=2)
            raise RuntimeError("worker process was terminated")

        # Drain the research tool stream on a worker thread and store its final return value in outcome.
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

    # Verify that while the provider call is blocked, a durable stop request recorded for the research
    # session is noticed by the worker, which checkpoints the cancelled state and returns the cancelled result.
    def test_research_worker_monitors_durable_stop_while_provider_is_blocked(self):
        session_id = "deep-research:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        provider_started = threading.Event()
        stop_requested = threading.Event()
        cancellation_checkpointed = threading.Event()
        outcome = {}

        # Block like a stuck provider until the execution is cancelled, then fail as if interrupted.
        def blocked_provider(*_args, **kwargs):
            cancellation = kwargs["cancellation"]
            provider_started.set()
            cancellation.cancelled.wait(timeout=2)
            raise RuntimeError("blocked provider interrupted")

        # Drain the research tool stream on a worker thread with the stop lookup and state update patched.
        def consume() -> None:
            # Record that the cancelled state was checkpointed and return an empty state.
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

        # Act: wait until the provider is blocked, then raise the durable stop request.
        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(provider_started.wait(timeout=1))

        stop_requested.set()
        consumer.join(timeout=2)

        # Assert: the worker finished, checkpointed the cancellation and returned the cancelled result.
        self.assertFalse(consumer.is_alive())
        self.assertTrue(cancellation_checkpointed.wait(timeout=1))
        self.assertTrue(tool_registry.is_tool_execution_cancelled(outcome.get("result")))

    # Verify that when a tool call is cancelled but the chat itself is not aborted, the Ollama tool loop
    # runs one more round with tools disabled and a prompt telling the model not to call any tool again.
    def test_ollama_cancellation_forces_the_next_round_to_run_without_tools(self):
        # Arrange: a first round that requests a tool call and a second round that answers.
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

        # Record the tools and the last prompt of each provider round and replay the prepared round results.
        def fake_stream_round(_client, _model, conversation, _kwargs, *, tools=None):
            round_tools.append(tools)
            round_prompts.append(str(conversation[-1].get("content") or ""))
            result = round_results[len(round_tools) - 1]

            # Yield nothing and return the prepared round result as the stream value.
            def stream():
                if False:
                    yield {}
                return result

            return stream()

        # Yield nothing and end the tool stream with the cancelled marker.
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

        # Act: run the tool loop with every provider and tool dependency replaced by fakes.
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

        # Assert: the first round had tools, the second had none and carried the no-more-tools prompt.
        self.assertIsNotNone(round_tools[0])
        self.assertIsNone(round_tools[1])
        self.assertIn("Do not call any tool again", round_prompts[1])

    # Verify that when the main chat generation is aborted while a tool call is cancelled, the Ollama
    # tool loop does not start another provider round to acknowledge the cancellation.
    def test_main_chat_abort_does_not_start_a_cancellation_acknowledgement_round(self):
        # Arrange: the main abort flag is already set and the first round requests a tool call.
        ollama_api._abort_event.set()
        self.addCleanup(ollama_api._abort_event.clear)
        round_count = 0

        # Count provider rounds, fail on a second one and stream a single tool-call response.
        def fake_stream_round(_client, _model, _conversation, _kwargs, *, tools=None):
            nonlocal round_count
            round_count += 1
            if round_count > 1:
                raise AssertionError("main abort must not start another provider request")

            # Yield nothing and return the assistant message with the tool call as the stream value.
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

        # Yield nothing and end the tool stream with the cancelled marker.
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

        # Act: run the tool loop with every provider and tool dependency replaced by fakes.
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

        # Assert: only the first provider round ran.
        self.assertEqual(round_count, 1)

    # Verify that a persistent worker whose response read ends because the execution was cancelled is
    # closed and not restarted, and that the request raises ToolExecutionCancelled.
    def test_cancelled_persistent_worker_is_not_restarted(self):
        cancellation = tool_registry.ActiveToolExecution("generation-a")
        session = tool_registry.ExternalWorkerSession(Path("Tools/example/mcp-server.py"), Path("python"))
        session.process = SimpleNamespace(stdin=Mock(), stdout=Mock())

        # Mark the execution as cancelled and return an empty response line, as a read cut short would.
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

    # Verify that when the execution is cancelled while the persistent worker starts, the session is
    # closed and the request raises ToolExecutionCancelled.
    def test_cancelled_during_persistent_worker_start_is_closed_before_request(self):
        cancellation = tool_registry.ActiveToolExecution("generation-a")
        session = tool_registry.ExternalWorkerSession(Path("Tools/example/mcp-server.py"), Path("python"))

        # Mark the execution as cancelled while the worker is starting.
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

    # Verify that aborting generation through the LLM API aborts the active tools of the same generation
    # id and then aborts the engine adapter.
    @patch("API.llm_api.tool_registry.abort_active_tools")
    @patch("API.llm_api._get_engine_module")
    def test_llm_abort_forwards_generation_id_to_tools(self, get_module, abort_tools):
        adapter = SimpleNamespace(abort_generation=Mock())
        get_module.return_value = adapter

        llm_api.abort_generation("ollama-service", generation_id="generation-a")

        abort_tools.assert_called_once_with("generation-a")
        adapter.abort_generation.assert_called_once_with()


# Tool quotas

# Cover per-response tool quota guardrails.
class ToolQuotaTests(SimpleTestCase):
    # Verify that high-effort web search is limited to three calls per response, as it is expensive.
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

    # Verify that medium-effort web search keeps the default quota and still allows a fourth call.
    def test_normal_web_search_keeps_default_quota(self):
        tool_event = {"tool_id": "web_search", "tool_name": "Web search"}
        counters: dict[str, int] = {}
        arguments = {"query": "cheap coding model", "effort": "medium"}

        for _index in range(4):
            self.assertIsNone(tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments))

    # Verify that in instant mode web_search and read_page each allow two calls per response and
    # reject the third with an "at most 2 times" message.
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


# Instant tool schemas

# Cover the instant-mode tool schema, its forced batch size and the preflight of instant tool calls.
class InstantToolSchemaTests(SimpleTestCase):
    # Verify that the Ollama, OpenAI, LM Studio and Google adapters all pass the forced instant search
    # batch size from the context on to the tool builder.
    def test_all_engine_adapters_forward_forced_instant_batch_size(self):
        # Arrange: one tool loop invocation and the provider stream function to replace for each adapter.
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

        # Act and assert: run each adapter's tool loop with an empty provider stream and check the batch
        # size passed to the tool builder.
        for adapter, invoke, stream_name in adapter_cases:
            with self.subTest(adapter=adapter.__name__):
                with (
                    patch.object(tool_registry, "build_ollama_tools", return_value=([], {})) as builder,
                    patch.object(adapter, stream_name, return_value=iter(())),
                ):
                    invoke()
                self.assertEqual(builder.call_args.kwargs["instant_search_batch_size"], 3)

        # The Google adapter builds its tools directly instead of running a tool loop.
        with patch.object(tool_registry, "build_ollama_tools", return_value=([], {})) as builder:
            google_genai_api._build_google_tools(["web_search"], "model", context)
        self.assertEqual(builder.call_args.kwargs["instant_search_batch_size"], 3)

    # Verify that instant mode reduces the web search schema to query, description and operators (with a
    # forced batch size turning query into an array of exactly that size) and that the normal thinking
    # schema keeps its effort parameter.
    def test_instant_search_schema_is_minimal_and_thinking_schema_is_unchanged(self):
        # Arrange: a web search server with a full search tool and a read page tool.
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

        # Act: build the tools in instant mode, in instant mode with a forced batch size and in normal mode.
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

        # Assert: the instant schema is minimal, the forced one is a fixed-size array, the normal one is unchanged.
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

    # Verify that with a forced batch size of three the preflight rejects a call with a single query using
    # INVALID_INSTANT_SEARCH_BATCH, accepts a call with three queries, and collapses three parallel
    # single-query calls into one call with all three queries.
    def test_forced_instant_search_preflight_requires_exactly_three_queries(self):
        # Arrange: an in-process web search tool with a mocked argument preparer and a forced batch size of 3.
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

        # Act: prepare one call with a single query and one call with three queries.
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

        # Assert: the single query is rejected without calling the preparer, the three queries pass through it.
        self.assertEqual(
            rejected["preflight_error_result"]["ui"]["error"]["code"],
            "INVALID_INSTANT_SEARCH_BATCH",
        )
        self.assertEqual(rejected["tool_ui"]["query_count"], 0)
        self.assertNotIn("preflight_error_result", accepted)
        preparer.assert_called_once()

        # Parallel single-query calls are merged into one call with a combined description.
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

    # Verify that a TOOL_PREPARERS entry of an in-process server module is discovered (the tool is marked
    # as preparing arguments) and is called with the instant-mode flag, and that its tool UI reaches the
    # prepared call.
    def test_in_process_tool_preparer_is_discovered_and_executed(self):
        # Arrange: a server module with one tool and a mocked preparer.
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

        # Act: prepare one call to the tool.
        prepared = tool_registry.prepare_tool_calls(lookup, [{
            "id": "call-1",
            "name": tool["alias"],
            "arguments": {"query": "alpha", "description": "Checking alpha"},
        }])

        # Assert: the preparer was discovered, called with the instant-mode flag and its UI was kept.
        self.assertTrue(server["tools"][0]["prepares_arguments"])
        preparer.assert_called_once()
        self.assertTrue(preparer.call_args.args[0]["_instant_mode"])
        self.assertTrue(prepared[0]["tool_ui"]["instant_mode"])

    # Verify that in instant mode the Ollama tool loop survives a search, a page read and a retry search
    # and still reaches the final answer without hitting the tool loop limit.
    def test_ollama_instant_search_read_and_retry_still_reaches_final_answer(self):
        # Arrange: a search tool, a read page tool and four provider rounds (search, read, search, answer).
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

        # Replay the prepared provider rounds one by one.
        def fake_stream_round(_client, _model, _conversation, _kwargs, *, tools=None):
            result = round_results.pop(0)

            # Yield nothing and return the next prepared round result as the stream value.
            def stream():
                if False:
                    yield {}
                return result

            return stream()

        # Yield nothing and return a finished web search tool result.
        def fake_tool_stream(*_args, **_kwargs):
            if False:
                yield {}
            return {
                "model_context": "Tool result",
                "sources": [],
                "ui": {"kind": "web_search", "status": "done", "instant_mode": True},
            }

        # Act: run the instant-mode tool loop with the provider and tool streams replaced by fakes.
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

        # Assert: all rounds were consumed, the final answer was streamed and no loop-limit error appeared.
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


# Tool cooldowns

# Cover the per-chat cooldown of repeated tool calls.
class ToolCooldownTests(SimpleTestCase):
    # Start every test with no remembered cooldowns.
    def setUp(self):
        tool_registry._TOOL_COOLDOWNS.clear()

    # Drop the cooldowns remembered by the test.
    def tearDown(self):
        tool_registry._TOOL_COOLDOWNS.clear()

    # Verify that a web search cooldown applies to the chat that made the call but not to another chat
    # that sends identical arguments.
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

    # Verify that the per-chat scope also applies to page reads: a cooldown remembered for one chat does
    # not apply to another chat.
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


# Tool preflight

# Cover the preflight of tool calls (argument preparation, rejection and batching) before the tool runs.
class ToolPreflightTests(SimpleTestCase):
    # Build a lookup with one external web search tool that prepares its arguments.
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

    # Verify that the arguments prepared by the worker replace the raw model arguments in the tool event,
    # the streamed tool call marker and the canonicalized transcript, while the call still keeps the raw
    # arguments.
    @patch.object(tool_registry, "_run_worker")
    def test_preflight_replaces_raw_arguments_before_tool_event(self, worker_mock):
        # Arrange: a worker that answers with canonical arguments and a UI search request.
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

        # Act: prepare the call and build its tool event.
        call = tool_registry.prepare_tool_call(
            self._lookup(),
            {"name": "web_search__web_search", "arguments": raw},
        )
        event = tool_registry.build_tool_event(self._lookup(), call)

        # Assert: the event carries only the canonical arguments and UI, the raw value is only kept on the call.
        self.assertEqual(call["raw_arguments"], raw)
        self.assertNotIn("raw model value", json.dumps(event))
        self.assertEqual(event["arguments"]["web"], "canonical")
        self.assertEqual(event["tool_ui"]["description"], "Verify canonical sources")
        self.assertEqual(event["tool_ui"]["search_request"]["queries"][0]["compiled_query"], "canonical")

        # The streamed marker exposes the UI but not the raw value.
        marker = _serialize_tool_call_marker(event)
        self.assertIn('"tool_ui"', marker)
        self.assertNotIn("raw model value", marker)

        # The stored transcript is rewritten to the canonical arguments.
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

    # Verify that a preflight rejected by the worker clears the arguments and keeps the structured error
    # result and the rejected raw arguments on the call, so an adapter can short-circuit the tool run.
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

    # Verify that two parallel advanced search calls are prepared by one worker request: the first call
    # carries both queries as the batch and absorbs the second call.
    @patch.object(tool_registry, "_run_worker")
    def test_parallel_advanced_search_calls_are_preflighted_as_one_batch(self, worker_mock):
        # Check that both queries arrive in one persistent worker request and report them as prepared.
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

        # Act: prepare both parallel calls together.
        prepared = tool_registry.prepare_tool_calls(self._lookup(), calls)

        # Assert: one batched call remains and the worker was asked once.
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["id"], "call-1")
        self.assertEqual(prepared[0]["parallel_batch_size"], 2)
        self.assertEqual([call["id"] for call in prepared[0]["absorbed_tool_calls"]], ["call-2"])
        worker_mock.assert_called_once()

    # Verify that four parallel legacy search calls exceeding the server batch limit are rejected together
    # as one atomic rejection and that the transcript is collapsed to the single remaining call.
    @patch.object(tool_registry, "_run_worker")
    def test_parallel_legacy_search_batch_over_server_limit_is_one_atomic_rejection(self, worker_mock):
        # Check that all four queries arrive in one persistent worker request and reject the batch, because
        # only two queries are permitted in total.
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

        # Act: prepare the four parallel calls together.
        prepared = tool_registry.prepare_tool_calls(self._lookup(), calls)

        # Assert: a single rejected call remains whose message names the four calls and the limit.
        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["tool_ui"]["status"], "rejected")
        self.assertIn("4 parallel web_search calls", prepared[0]["preflight_error_result"]["model_context"])
        self.assertIn("batch permits at most 2 queries total", prepared[0]["preflight_error_result"]["model_context"])
        worker_mock.assert_called_once()

        # The transcript with all four calls is reduced to the one rejected call, also in the Google parts.
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


# Tool and chat APIs

# Cover local tool-server listing and chat persistence endpoints.
class ToolApiTests(ToolRegistryTestMixin, TestCase):
    # Treat every engine as enabled, so the API views accept requests for any engine.
    def setUp(self):
        super().setUp()
        self._engine_enabled_patch = patch(
            "Apps.API.views.settings.is_engine_enabled",
            return_value=True,
        )
        self._engine_enabled_patch.start()

    # Remove the engine-enabled patch.
    def tearDown(self):
        self._engine_enabled_patch.stop()
        super().tearDown()

    # Verify that the tools API returns the discovered server with its tool count and tool summaries
    # for the requested engine and model.
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

    # Verify that loading a chat returns its active tool server id and builds the activity segments
    # (a thought and a tool call) of an assistant message from its stored transcript.
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

    # Verify that a chat storing several active tool servers returns all of their ids, with the first
    # one also reported as the single active tool server id.
    def test_load_chat_api_returns_multiple_active_tool_server_ids(self):
        chat = Chat.objects.create(title="Chat", active_tool_slug='["time_suite", "browser"]')
        Message.objects.create(chat=chat, role="user", content="Hello")

        response = self.client.get(reverse("load_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["active_tool_server_ids"], ["time_suite", "browser"])
        self.assertEqual(payload["active_tool_server_id"], "time_suite")

    # Verify that loading a chat returns attachment metadata without the inline attachment data.
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

    # Verify that the attachment content endpoint streams the decoded stored bytes with the attachment
    # MIME type.
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

    # Verify that the attachment content endpoint also streams legacy image records, with the image
    # MIME type and a generated file name based on the image order.
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

    # Verify that the attachment content endpoint answers an unknown record type with 404 and an error.
    def test_attachment_content_api_rejects_unknown_record_type(self):
        response = self.client.get(reverse("attachment_content_api", args=["unknown", 1]))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "Unknown attachment type")

    # Verify that deleting the last assistant message removes it and returns the preceding user message
    # with its attachments and image data URLs, so the answer can be regenerated.
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

    # Verify that deleting the last assistant message of a chat whose last message is from the user
    # answers 400 with an error.
    def test_delete_last_assistant_api_rejects_when_last_message_is_user(self):
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Still pending")

        response = self.client.delete(reverse("delete_last_assistant_api", args=[chat.id]))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Last message is not from assistant")

    # Verify that deleting a message removes only that message and keeps the other ones.
    def test_delete_message_api_removes_selected_message(self):
        chat = Chat.objects.create(title="Chat")
        first = Message.objects.create(chat=chat, role="user", content="First")
        second = Message.objects.create(chat=chat, role="assistant", content="Second")

        response = self.client.delete(reverse("delete_message_api", args=[first.id]))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Message.objects.filter(id=first.id).exists())
        self.assertTrue(Message.objects.filter(id=second.id).exists())

    # Verify that branching at a message creates a new chat holding a lossless copy of the history up to
    # that turn (tool selection, attachments and full transcript, without later messages) and that both
    # chats expose branch links to each other when loaded.
    def test_branch_message_api_copies_history_and_exposes_bidirectional_markers(self):
        # Arrange: a chat with a user message with an attachment, the assistant turn to branch at and a
        # later message.
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

        # Act: branch at the assistant message.
        response = self.client.post(reverse("branch_message_api", args=[target.id]), data="{}", content_type="application/json")

        # Assert: the child chat copies the history up to the target and records the branch.
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

        # Loading both chats shows a link to the branch on the source and a link back to the origin on the child.
        source_payload = self.client.get(reverse("load_chat_api", args=[chat.id])).json()
        child_payload = self.client.get(reverse("load_chat_api", args=[child.id])).json()
        source_target = next(item for item in source_payload["messages"] if item["id"] == target.id)
        child_target = next(item for item in child_payload["messages"] if item["id"] == copied[1].id)
        self.assertEqual(source_target["branch_links"][0]["direction"], "to_branch")
        self.assertEqual(source_target["branch_links"][0]["chat_id"], str(child.id))
        self.assertEqual(child_target["branch_links"][0]["direction"], "to_origin")
        self.assertEqual(child_target["branch_links"][0]["chat_id"], str(chat.id))

    # Verify that editing a user message replaces its text, deletes every later message and keeps the
    # attachments of the edited message.
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

    # Verify that exporting a chat returns a ZIP archive with readable Markdown that links the cited
    # sources, the chat JSON with the full untruncated transcript, and the binary attachments.
    def test_export_chat_api_returns_lossless_zip(self):
        # Arrange: an assistant message with a cited source, a full transcript and an image attachment.
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

        # Act: export the chat.
        response = self.client.get(reverse("export_chat_api", args=[chat.id]))

        # Assert: the archive holds Markdown, JSON and the attachment with their expected content.
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

    # Verify that renaming a chat trims the new title and persists it.
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

    # Verify that deleting a chat removes the chat and its messages.
    def test_delete_chat_api_removes_thread_and_messages(self):
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Hello")

        response = self.client.delete(reverse("delete_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Chat.objects.filter(id=chat.id).exists())
        self.assertEqual(Message.objects.count(), 0)
