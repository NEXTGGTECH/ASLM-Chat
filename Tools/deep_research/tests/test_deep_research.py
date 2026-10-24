# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock


from Tools.deep_research import runner as deep_research
from Tools.deep_research import service
from API import mcp as tool_registry
from Services import venv_manager
from Services.tool_worker import WorkerEventEmitter
from sandbox import temporal as temporal_sandbox_module


def _metadata_file(payload: dict) -> tempfile.NamedTemporaryFile:
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".json", encoding="utf-8", delete=False)
    json.dump(payload, handle)
    handle.close()
    return handle


class RuntimeModelTests(unittest.TestCase):
    def test_caller_selection_wins_over_stale_active_metadata(self):
        metadata = _metadata_file(
            {
                "active": {"engine": "openai", "model": "stale-model"},
                "models": {
                    "ollama-service:current-model": {
                        "engine": "ollama-service",
                        "model": "current-model",
                        "capabilities": {"tools": True, "thinking": True},
                    }
                },
            }
        )
        self.addCleanup(Path(metadata.name).unlink, missing_ok=True)

        runtime = deep_research.resolve_runtime_model(
            {"engine": "ollama-service", "model_name": "current-model"},
            metadata_path=Path(metadata.name),
        )

        self.assertEqual(runtime["engine"], "ollama-service")
        self.assertEqual(runtime["model"], "current-model")
        self.assertTrue(runtime["capabilities"]["thinking"])
        self.assertEqual(runtime["metadata_source"], "caller_context")

    def test_rejects_a_model_explicitly_without_tool_support(self):
        metadata = _metadata_file(
            {
                "active": {"engine": "openai", "model": "plain"},
                "models": {
                    "openai:plain": {
                        "engine": "openai",
                        "model": "plain",
                        "capabilities": {"tools": False},
                    }
                },
            }
        )
        self.addCleanup(Path(metadata.name).unlink, missing_ok=True)

        with self.assertRaisesRegex(RuntimeError, "does not support tool calling"):
            deep_research.resolve_runtime_model({}, metadata_path=Path(metadata.name))

    def test_runtime_resolution_failure_marks_prepared_session_failed(self):
        session_id = "deep-research:20000000000000000000000000000001"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with (
                mock.patch.object(deep_research.research_control, "STATE_ROOT", root / "state"),
                mock.patch.object(deep_research.research_control, "COMMAND_ROOT", root / "commands"),
            ):
                deep_research.research_control.create_session(session_id, topic="Test")

                with self.assertRaises(RuntimeError):
                    deep_research.run_deep_research(
                        {"topic": "Test", "session_id": session_id},
                        {},
                        metadata_path=root / "missing-runtime.json",
                    )

                state = deep_research.research_control.read_state(session_id)
                self.assertEqual(state["status"], "failed")
                self.assertFalse(state["can_stop"])


class OrchestrationTests(unittest.TestCase):
    def test_state_machine_reflects_selects_searches_and_synthesizes_without_model_tools(self):
        metadata = _metadata_file(
            {
                "active": {"engine": "openai", "model": "research-model"},
                "models": {
                    "openai:research-model": {
                        "engine": "openai",
                        "model": "research-model",
                        "capabilities": {"tools": True, "thinking": True},
                    }
                },
            }
        )
        self.addCleanup(Path(metadata.name).unlink, missing_ok=True)
        source = {
            "id": "c0042-1",
            "url": "https://example.com/source",
            "title": "Primary source",
        }
        search_sources = [
            source,
            *[
                {
                    "id": f"c0042-{index}",
                    "url": f"https://example.com/source-{index}",
                    "title": f"Search source {index}",
                    "domain": "example.com",
                }
                for index in range(2, 19)
            ],
        ]
        model_outputs = [
            '{"summary":"Verify the claim","steps":[{"id":"s1","title":"Find primary evidence"}],'
            '"candidates":[{"text":"claim primary evidence","vertical":"web","purpose":"primary source"}]}',
            '{"assessment":"Primary evidence is missing.","gaps":["primary source"],'
            '"updates":[{"id":"s1","status":"active"}],"complete":false,'
            '"candidates":[{"text":"claim primary evidence","vertical":"web","purpose":"primary source"}]}',
            '{"queries":[{"text":"claim primary evidence","vertical":"web","purpose":"best direct query"}]}',
            '{"assessment":"The primary evidence is verified.","gaps":[],'
            '"updates":[{"id":"s1","status":"done"}],"complete":true,"candidates":[]}',
            '{"assessment":"Final audit confirms the primary evidence.","gaps":[],'
            '"updates":[{"id":"s1","status":"done"}],"complete":true,"candidates":[]}',
            "Verified report [c0042-1]",
        ]
        generation_chunks = [
            iter([{"message": {"role": "assistant", "content": output}}])
            for output in model_outputs
        ]

        event_log_dir = tempfile.TemporaryDirectory()
        self.addCleanup(event_log_dir.cleanup)
        streamed_events: list[dict] = []
        control_dir = tempfile.TemporaryDirectory()
        self.addCleanup(control_dir.cleanup)

        def fake_tool(_lookup, alias, _arguments, context=None):
            del context
            if alias == "web_search__web_search":
                return {
                    "_tool_result_structured": {
                        "model_context": "Primary evidence [c0042-1]",
                        "sources": search_sources,
                    }
                }
            return {
                "_tool_result_structured": {
                    "model_context": "Full primary source [c0042-1]",
                    "sources": [source],
                }
            }

        from Tools.deep_research.orchestrator import control as research_control

        with (
            mock.patch.object(deep_research.llm_api, "generate", side_effect=generation_chunks) as generate,
            mock.patch.object(tool_registry, "build_ollama_tools", return_value=([], {"web_search__web_search": {}, "web_search__read_page": {}})),
            mock.patch.object(tool_registry, "call_ollama_tool", side_effect=fake_tool) as call_tool,
            mock.patch.object(tool_registry, "clear_tool_runtime_scope"),
            mock.patch.object(research_control, "STATE_ROOT", Path(control_dir.name) / "state"),
            mock.patch.object(research_control, "COMMAND_ROOT", Path(control_dir.name) / "commands"),
        ):
            result = deep_research.run_deep_research(
                {
                    "topic": "Investigate the claim",
                    "max_rounds": 2,
                    "session_id": "deep-research:11111111111111111111111111111111",
                },
                {
                    "engine": "openai",
                    "model_name": "research-model",
                    "chat_id": "outer",
                    "event_callback": streamed_events.append,
                    "auto_approve": True,
                },
                metadata_path=Path(metadata.name),
                logs_dir=Path(event_log_dir.name),
            )

        self.assertEqual(generate.call_count, 6)
        for generation_call in generate.call_args_list:
            self.assertEqual(generation_call.args[:2], ("openai", "research-model"))
            self.assertNotIn("tool_server_ids", generation_call.kwargs)
            self.assertTrue(generation_call.kwargs["think"])
            self.assertEqual(generation_call.kwargs["think_level"], "high")
        self.assertEqual(
            [call.args[1] for call in call_tool.call_args_list],
            ["web_search__web_search", "web_search__read_page"],
        )

        self.assertEqual(
            result["report"],
            "Verified report [Primary source](https://example.com/source)",
        )
        self.assertEqual(
            result["model_context"],
            "Verified report [Primary source](https://example.com/source)",
        )
        self.assertEqual(result["sources"][0]["id"], "c11111111-1")
        self.assertEqual(result["sources"][0]["citation_aliases"], ["c0042-1"])
        self.assertEqual(result["sources"][0]["url"], source["url"])
        self.assertEqual(result["ui"]["result_count"], 18)
        self.assertEqual(result["ui"]["status"], "completed")
        self.assertEqual(result["ui"]["queries_used"], 1)
        event_log_path = Path(result["event_log"]["path"])
        self.assertTrue(event_log_path.is_file())
        persisted_events = [
            json.loads(line)
            for line in event_log_path.read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(
            [event["sequence"] for event in persisted_events],
            list(range(1, len(persisted_events) + 1)),
        )
        self.assertEqual(streamed_events, persisted_events)
        event_types = {event["type"] for event in persisted_events}
        self.assertIn("planning_started", event_types)
        self.assertIn("approval_required", event_types)
        self.assertIn("reflection_completed", event_types)
        self.assertIn("queries_selected", event_types)
        self.assertIn("search_completed", event_types)
        self.assertIn("reading_started", event_types)
        self.assertIn("reading_completed", event_types)
        self.assertIn("synthesis_started", event_types)
        self.assertIn("session_completed", event_types)
        self.assertNotIn("model_output_delta", event_types)
        search_completed = next(
            event for event in persisted_events if event["type"] == "search_completed"
        )
        self.assertEqual(search_completed["data"]["source_count"], 18)
        self.assertEqual(len(search_completed["data"]["sources"]), 18)
        reading_started = next(
            event for event in persisted_events if event["type"] == "reading_started"
        )
        self.assertEqual(reading_started["data"]["tool_id"], "read_page")
        expected_read_urls = [source["url"], search_sources[1]["url"]]
        self.assertEqual(reading_started["data"]["urls"], expected_read_urls)
        reading_completed = next(
            event for event in persisted_events if event["type"] == "reading_completed"
        )
        self.assertEqual(reading_completed["data"]["tool_id"], "read_page")
        self.assertEqual(reading_completed["data"]["urls"], expected_read_urls)

    def test_parent_model_context_resolves_grouped_citations_to_links(self):
        result = deep_research.replace_citation_handles_with_markdown_links(
            "Claim [c0008-2, c0008-7]; unknown [c9999-1].",
            [
                {"id": "c0008-2", "url": "https://example.com/primary"},
                {"id": "c0008-7", "url": "https://example.com/second"},
                {"id": "c9999-1", "url": "javascript:alert(1)"},
            ],
        )

        self.assertEqual(
            result,
            (
                "Claim [c0008-2](https://example.com/primary) "
                "[c0008-7](https://example.com/second); unknown [c9999-1]."
            ),
        )

    def test_parent_model_context_links_citation_before_parenthesized_paragraph(self):
        result = deep_research.replace_citation_handles_with_markdown_links(
            "Claim [c0002-1]\n\n(Additional context.)",
            [{"id": "c0002-1", "url": "https://example.com/source"}],
        )

        self.assertEqual(
            result,
            "Claim [c0002-1](https://example.com/source)\n\n(Additional context.)",
        )

    def test_event_logger_advances_iteration_after_tool_results(self):
        logs = tempfile.TemporaryDirectory()
        self.addCleanup(logs.cleanup)
        logger = deep_research.ResearchEventLogger(
            "deep-research:test",
            logs_dir=Path(logs.name),
        )
        self.addCleanup(logger.close)

        logger.observe_chunk(
            "research",
            {"message": {"thinking": "First reasoning"}},
        )
        logger.observe_chunk(
            "research",
            {"tool_result": {"role": "tool", "content": "Evidence"}},
        )
        logger.observe_chunk(
            "research",
            {"message": {"content": "Second iteration"}},
        )

        records = [
            json.loads(line)
            for line in logger.path.read_text(encoding="utf-8").splitlines()
        ]
        deltas = [event for event in records if event["type"] == "model_output_delta"]
        self.assertEqual([event["iteration"] for event in deltas], [1, 2])
        self.assertIn("iteration_started", [event["type"] for event in records])

    def test_worker_event_socket_is_independent_from_stdout_protocol(self):
        received: list[dict] = []
        arrived = threading.Event()

        def receive(event: dict) -> None:
            received.append(event)
            arrived.set()

        channel = tool_registry.ToolEventSocketChannel(receive)
        emitter = WorkerEventEmitter(channel.config)
        try:
            emitter.emit({"type": "model_output_delta", "data": {"content": "live"}})
            self.assertTrue(arrived.wait(2))
        finally:
            emitter.close()
            channel.close()

        self.assertEqual(received[0]["data"]["content"], "live")

    def test_stream_tool_yields_activity_before_returning_result(self):
        def fake_call(
            _lookup,
            _alias,
            _arguments,
            context=None,
            event_callback=None,
            cancellation=None,
        ):
            del context, cancellation
            event_callback({"type": "first"})
            event_callback({"type": "second"})
            return "finished"

        lookup = {
            "suite__work": {
                "server": {"id": "suite"},
                "tool": {"id": "work"},
            }
        }
        with mock.patch.object(tool_registry, "call_ollama_tool", side_effect=fake_call):
            stream = tool_registry.stream_ollama_tool(lookup, "suite__work", {})
            progress = next(stream)
            with self.assertRaises(StopIteration) as stopped:
                next(stream)

        event_batch = progress["tool_progress"]["event"]
        self.assertEqual(event_batch["type"], "event_batch")
        self.assertEqual(event_batch["count"], 2)
        self.assertEqual(
            [event["type"] for event in event_batch["events"]],
            ["first", "second"],
        )
        self.assertEqual(stopped.exception.value, "finished")

    def test_stream_tool_flushes_full_batches_without_waiting_for_completion(self):
        release_tool = threading.Event()

        def fake_call(
            _lookup,
            _alias,
            _arguments,
            context=None,
            event_callback=None,
            cancellation=None,
        ):
            del context, cancellation
            event_callback({"sequence": 1, "type": "first"})
            event_callback({"sequence": 2, "type": "second"})
            event_callback({"sequence": 3, "type": "third"})
            release_tool.wait(2)
            return "finished"

        lookup = {
            "suite__work": {
                "server": {"id": "suite"},
                "tool": {"id": "work"},
            }
        }
        with (
            mock.patch.object(tool_registry, "call_ollama_tool", side_effect=fake_call),
            mock.patch.object(tool_registry, "TOOL_EVENT_BATCH_MAX_EVENTS", 2),
        ):
            stream = tool_registry.stream_ollama_tool(lookup, "suite__work", {})
            first_progress = next(stream)
            release_tool.set()
            second_progress = next(stream)
            with self.assertRaises(StopIteration) as stopped:
                next(stream)

        first_batch = first_progress["tool_progress"]["event"]
        second_batch = second_progress["tool_progress"]["event"]
        self.assertEqual(first_batch["count"], 2)
        self.assertEqual(first_batch["first_sequence"], 1)
        self.assertEqual(first_batch["last_sequence"], 2)
        self.assertEqual(second_batch["count"], 1)
        self.assertEqual(second_batch["first_sequence"], 3)
        self.assertEqual(stopped.exception.value, "finished")


class ContractTests(unittest.TestCase):
    def test_application_service_canonicalizes_long_running_call(self):
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(
            service.control, "STATE_ROOT", Path(directory) / "state"
        ), mock.patch.object(
            service.control, "COMMAND_ROOT", Path(directory) / "commands"
        ):
            prepared = service.prepare_research(
                "  A topic  ", instructions=" concise ", max_rounds=999
            )
        self.assertEqual(prepared["arguments"]["topic"], "A topic")
        self.assertEqual(prepared["arguments"]["max_rounds"], 25)
        self.assertEqual(prepared["arguments"]["timeout_s"], 3600)
        self.assertEqual(prepared["tool_ui"]["status"], "planning")
        self.assertRegex(prepared["arguments"]["session_id"], r"^deep-research:[a-f0-9]{32}$")
        self.assertEqual(prepared["tool_ui"]["query_budget"], 50)

    def test_registry_filters_every_non_research_tool(self):
        server = {
            "id": "sandbox",
            "name": "Sandbox",
            "description": "",
            "tools": [
                {"id": "bash", "alias": "sandbox__bash", "name": "Bash", "description": "", "parameters": {}},
                {"id": "write", "alias": "sandbox__write", "name": "Write", "description": "", "parameters": {}},
            ],
        }
        with mock.patch.object(tool_registry, "get_server", return_value=server):
            tools, lookup = tool_registry.build_ollama_tools(
                ["sandbox"], allowed_tool_aliases=["sandbox__bash"]
            )
        self.assertEqual([item["function"]["name"] for item in tools], ["sandbox__bash"])
        self.assertEqual(list(lookup), ["sandbox__bash"])

    def test_per_session_tool_limits_are_bounded(self):
        self.assertEqual(tool_registry.tool_round_limit_from_context({"max_tool_rounds": 7}, 100), 7)
        self.assertEqual(tool_registry.tool_round_limit_from_context({"max_tool_rounds": 500}, 100), 100)
        calls = [{"name": str(index)} for index in range(4)]
        self.assertEqual(
            tool_registry.limit_tool_calls_from_context({"max_parallel_tool_calls": 2}, calls),
            calls[:2],
        )

    def test_search_limit_does_not_limit_consecutive_bash_calls(self):
        server = lambda server_id: {"server": {"id": server_id}, "tool": {}}  # noqa: E731
        lookup = {
            "web_search__web_search": server("web_search"),
            "web_search__read_page": server("web_search"),
            "sandbox__bash": server("sandbox"),
        }
        calls = [
            {"name": "web_search__web_search"},
            {"name": "web_search__read_page"},
            {"name": "web_search__read_page"},
            {"name": "sandbox__bash", "arguments": {"command": "one"}},
            {"name": "sandbox__bash", "arguments": {"command": "two"}},
            {"name": "sandbox__bash", "arguments": {"command": "three"}},
        ]
        limited = tool_registry.limit_tool_calls_from_context(
            {"max_parallel_tool_calls_by_server": {"web_search": 2}},
            calls,
            lookup,
        )
        self.assertEqual(
            [call["name"] for call in limited],
            [
                "web_search__web_search",
                "web_search__read_page",
                "sandbox__bash",
                "sandbox__bash",
                "sandbox__bash",
            ],
        )

    def test_research_runtime_scope_can_be_fully_cleared(self):
        context = {"chat_id": "deep-research:clean-room"}
        event = {"tool_id": "web_search", "alias": "web_search__web_search"}
        arguments = {"queries": [{"query": "fresh state"}]}
        tool_registry.remember_tool_cooldown(event, arguments, context=context)
        self.assertIsNotNone(
            tool_registry.consume_tool_cooldown(event, arguments, context=context)
        )

        tool_registry.clear_tool_runtime_scope(context)

        self.assertIsNone(
            tool_registry.consume_tool_cooldown(event, arguments, context=context)
        )

    def test_research_compaction_keeps_latest_tool_pair(self):
        prefix = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "goal"},
            {"role": "assistant", "content": "plan"},
            {"role": "user", "content": "execute"},
        ]
        conversation = prefix + [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "old"}]},
            {"role": "tool", "tool_call_id": "old", "content": "old evidence [c0001-1]"},
            {"role": "assistant", "content": "old reflection"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "new"}]},
            {"role": "tool", "tool_call_id": "new", "content": "new evidence [c0002-1]"},
        ]
        compactor = mock.Mock(return_value="## Verified Claims\n- old [c0001-1]")
        compacted = tool_registry.maybe_compact_tool_conversation(
            {
                "conversation_compactor": compactor,
                "compression_trigger_chars": 1,
                "compression_prefix_messages": 4,
            },
            conversation,
        )
        self.assertEqual(compacted[:4], prefix)
        self.assertIn("[deep research memory]", compacted[4]["content"])
        self.assertEqual(compacted[-2:], conversation[-2:])
        self.assertNotIn("old evidence", json.dumps(compacted, ensure_ascii=False))

    def test_google_compaction_does_not_split_function_call_from_response(self):
        conversation = [
            {"role": "user", "parts": [{"text": "goal"}]},
            {"role": "model", "parts": [{"text": "plan"}]},
            {"role": "user", "parts": [{"text": "execute"}]},
            {"role": "model", "parts": [{"function_call": {"name": "old"}}]},
            {"role": "user", "parts": [{"function_response": {"name": "old"}}]},
            {"role": "model", "parts": [{"text": "reflection"}]},
            {"role": "model", "parts": [{"function_call": {"name": "new"}}]},
            {"role": "user", "parts": [{"function_response": {"name": "new"}}]},
        ]
        compacted = tool_registry.maybe_compact_tool_conversation(
            {
                "conversation_compactor": mock.Mock(return_value="memory"),
                "compression_trigger_chars": 1,
                "compression_prefix_messages": 4,
            },
            conversation,
            provider_format="google",
        )
        self.assertEqual(compacted[:3], conversation[:3])
        self.assertIn("[deep research memory]", compacted[3]["parts"][0]["text"])
        self.assertEqual(compacted[-2:], conversation[-2:])

    def test_temporal_sandbox_workspace_is_removed_after_scope(self):
        with mock.patch.object(temporal_sandbox_module, "_remove_container") as remove_container:
            with temporal_sandbox_module.temporal_sandbox() as spec:
                workspace = Path(spec["host_workspace"])
                self.assertTrue(workspace.is_dir())
                self.assertTrue(spec["container_name"].startswith("aslm-deep-research-"))
            self.assertFalse(workspace.exists())
            remove_container.assert_called_once_with(spec["container_name"])

    def test_temporal_run_command_has_no_restart_policy_and_uses_custom_mount(self):
        command = temporal_sandbox_module._build_run_command(
            "image:test",
            include_storage_limit=False,
            container_name="aslm-deep-research-0123456789ab",
            task_host_path="C:/Temp/aslm-deep-research-test",
            restart_policy=None,
            auto_remove=True,
        )
        self.assertNotIn("--restart", command)
        self.assertIn("--rm", command)
        self.assertIn("aslm-deep-research-0123456789ab", command)
        self.assertIn("C:/Temp/aslm-deep-research-test:/workspace/_sandbox", command)

    def test_temporal_bash_reuses_the_session_container(self):
        with mock.patch.object(temporal_sandbox_module, "_remove_container"):
            with temporal_sandbox_module.temporal_sandbox() as spec:
                completed = subprocess.CompletedProcess(
                    args=["docker", "exec"],
                    returncode=0,
                    stdout="persisted\n",
                    stderr="",
                )
                with mock.patch.object(
                    temporal_sandbox_module,
                    "_ensure_container",
                    return_value=(True, "running"),
                ) as ensure_container, mock.patch.object(
                    temporal_sandbox_module.subprocess,
                    "run",
                    return_value=completed,
                ) as run:
                    result = temporal_sandbox_module.run_temporal_bash(
                        {"command": "cat state.txt", "cwd": "."},
                        {"temporal_sandbox": spec},
                    )

                self.assertTrue(result["ok"])
                self.assertEqual(result["result"]["stdout"], "persisted\n")
                ensure_container.assert_called_once()
                self.assertIn(spec["container_name"], run.call_args.args[0])

    def test_streaming_process_emits_stdout_and_stderr_chunks(self):
        events: list[dict] = []
        return_code, stdout, stderr, timed_out = temporal_sandbox_module._run_process_streaming(
            [
                sys.executable,
                "-u",
                "-c",
                "import sys; print('out', flush=True); print('err', file=sys.stderr, flush=True)",
            ],
            stdin_value=None,
            timeout_s=5,
            context={"event_callback": events.append},
        )

        self.assertEqual(return_code, 0)
        self.assertFalse(timed_out)
        self.assertEqual(stdout.strip(), "out")
        self.assertEqual(stderr.strip(), "err")
        output_events = [event for event in events if event["type"] == "bash_output"]
        self.assertEqual(
            "".join(event["data"]["content"] for event in output_events if event["data"]["stream"] == "stdout").strip(),
            "out",
        )
        self.assertEqual(
            "".join(event["data"]["content"] for event in output_events if event["data"]["stream"] == "stderr").strip(),
            "err",
        )

    def test_server_venv_can_own_multiple_tools(self):
        config = _metadata_file(
            {
                "fileVersion": 1,
                "venvs": [
                    {"id": "server", "path": "Data/venvs/server", "tools": ["one", "two"]}
                ],
            }
        )
        self.addCleanup(Path(config.name).unlink, missing_ok=True)
        with mock.patch.object(venv_manager, "REQUIREMENTS_FILE", Path(config.name)):
            self.assertEqual(venv_manager.get_tool_venv_id("two"), "server")


if __name__ == "__main__":
    unittest.main()
