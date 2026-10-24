# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import io
import sys
import textwrap
import pytest
from pathlib import Path
from unittest.mock import patch
from API import mcp as tool_registry
from Apps.Tests.support import ToolRegistryTestCase


pytestmark = [pytest.mark.component]


# Tool registry tests.
# Verify MCP-style local server discovery and execution.
class LocalServerRegistryTests(ToolRegistryTestCase):
    """Verify discovery and execution of local MCP-style server modules."""

    # Discover valid local server modules.
    def test_list_servers_discovers_valid_server_modules(self):
        self.write_server(
            "time_suite",
            """
            MCP_SERVER = {
                "id": "time_suite",
                "name": "Time Suite",
                "description": "Time helpers",
            }

            TOOLS = [
                {
                    "id": "time_now",
                    "name": "Current Time",
                    "description": "Return the current time.",
                    "parameters": {"type": "object", "properties": {}},
                },
                {
                    "id": "timezone_name",
                    "name": "Timezone Name",
                    "description": "Return the active timezone name.",
                    "parameters": {"type": "object", "properties": {}},
                },
            ]

            def call_tool(tool_id, arguments, context=None):
                return {"tool_id": tool_id}
            """,
        )

        payload = tool_registry.list_servers()
        self.assertEqual(len(payload), 1)
        self.assertEqual(payload[0]["id"], "time_suite")
        self.assertEqual(payload[0]["tool_count"], 2)
        self.assertEqual(payload[0]["tools"][0]["id"], "time_now")

    # Hide servers that do not support the requested engine.
    def test_supports_filter_hides_servers_for_unsupported_engines(self):
        self.write_server(
            "ollama_only",
            """
            MCP_SERVER = {"id": "ollama_only", "name": "Ollama Only"}
            TOOLS = [{"id": "echo", "name": "Echo", "parameters": {"type": "object", "properties": {}}}]

            def supports(engine=None, model_name=None):
                return engine == "ollama-service"

            def call_tool(tool_id, arguments, context=None):
                return "ok"
            """,
        )

        self.assertEqual(tool_registry.list_servers(engine="openai"), [])
        self.assertEqual(tool_registry.list_servers(engine="ollama-service")[0]["id"], "ollama_only")

    # Build one OpenAI-style tool entry per local server tool.
    def test_build_ollama_tools_registers_multiple_tools(self):
        self.write_server(
            "multi",
            """
            MCP_SERVER = {"id": "multi", "name": "Multi"}
            TOOLS = [
                {"id": "alpha", "name": "Alpha", "parameters": {"type": "object", "properties": {}}},
                {"id": "beta", "name": "Beta", "parameters": {"type": "object", "properties": {}}},
            ]

            def call_tool(tool_id, arguments, context=None):
                return {"tool_id": tool_id}
            """,
        )

        tools, lookup = tool_registry.build_ollama_tools("multi", engine="ollama-service", model_name="llama3")
        aliases = [tool["function"]["name"] for tool in tools]

        self.assertEqual(len(tools), 2)
        self.assertIn("multi__alpha", aliases)
        self.assertIn("multi__beta", aliases)
        self.assertIn("multi__alpha", lookup)
        self.assertEqual(lookup["multi__alpha"]["tool"]["id"], "alpha")

    # Pass context through tool execution and serialize the result.
    def test_call_ollama_tool_serializes_results_and_passes_context(self):
        self.write_server(
            "context_suite",
            """
            MCP_SERVER = {"id": "context_suite", "name": "Context Suite"}
            TOOLS = [{
                "id": "context_echo",
                "name": "Context Echo",
                "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
            }]

            def call_tool(tool_id, arguments, context=None):
                return {
                    "tool_id": tool_id,
                    "value": arguments.get("value"),
                    "chat_id": context.get("chat_id"),
                    "server_name": context.get("server_name"),
                }
            """,
        )

        tools, lookup = tool_registry.build_ollama_tools("context_suite", engine="ollama-service", model_name="llama3")
        self.assertEqual(len(tools), 1)

        payload = tool_registry.call_ollama_tool(
            lookup,
            "context_suite__context_echo",
            {"value": "hello"},
            context={"chat_id": "chat-1"},
        )

        self.assertIn('"tool_id": "context_echo"', payload)
        self.assertIn('"value": "hello"', payload)
        self.assertIn('"chat_id": "chat-1"', payload)
        self.assertIn('"server_name": "Context Suite"', payload)

    # Skip invalid local server modules without breaking discovery.
    def test_invalid_server_modules_are_skipped(self):
        self.write_server(
            "invalid",
            """
            MCP_SERVER = {"id": "invalid", "name": "Invalid"}
            TOOLS = [{"id": "echo", "name": "Echo", "parameters": {"type": "object", "properties": {}}}]
            """,
        )

        self.assertEqual(tool_registry.list_servers(), [])

    # Execute servers that expose dedicated tool handlers.
    def test_tool_handlers_are_supported_without_generic_dispatcher(self):
        self.write_server(
            "handler_suite",
            """
            MCP_SERVER = {"id": "handler_suite", "name": "Handler Suite"}
            TOOLS = [{"id": "echo", "name": "Echo", "parameters": {"type": "object", "properties": {}}}]

            def echo(arguments, context=None):
                return {
                    "value": arguments.get("value"),
                    "alias": context.get("tool_alias"),
                }

            TOOL_HANDLERS = {"echo": echo}
            """,
        )

        _tools, lookup = tool_registry.build_ollama_tools("handler_suite")
        payload = tool_registry.call_ollama_tool(
            lookup,
            "handler_suite__echo",
            {"value": "hello"},
        )

        self.assertIn('"value": "hello"', payload)
        self.assertIn('"alias": "handler_suite__echo"', payload)

    # Execute async tool handlers through the sync registry API.
    def test_async_tool_handlers_are_supported(self):
        self.write_server(
            "async_suite",
            """
            MCP_SERVER = {"id": "async_suite", "name": "Async Suite"}
            TOOLS = [{"id": "echo", "name": "Echo", "parameters": {"type": "object", "properties": {}}}]

            async def echo(arguments, context=None):
                return {"value": arguments.get("value")}

            TOOL_HANDLERS = {"echo": echo}
            """,
        )

        _tools, lookup = tool_registry.build_ollama_tools("async_suite")
        payload = tool_registry.call_ollama_tool(
            lookup,
            "async_suite__echo",
            {"value": "async"},
        )

        self.assertIn('"value": "async"', payload)

    # Test persistent workers ignore heartbeat lines and wait for the final envelope.
    def test_external_worker_session_skips_heartbeat_lines(self):
        session = tool_registry.ExternalWorkerSession(Path("dummy-server.py"), Path(sys.executable))
        session.process = type(
            "FakeProcess",
            (),
            {"stdout": io.StringIO('{"event": "heartbeat"}\n{"ok": true, "result": "done"}\n')},
        )()

        self.assertEqual(session._read_response_line(timeout_s=1), '{"ok": true, "result": "done"}\n')

    # Test a silent persistent worker is killed instead of blocking forever.
    def test_external_worker_session_times_out_silent_worker(self):
        server_file = self.tools_dir / "slow_suite" / "mcp-server.py"
        server_file.parent.mkdir(parents=True, exist_ok=True)
        server_file.write_text(
            textwrap.dedent(
                """
                import time

                MCP_SERVER = {"id": "slow_suite", "name": "Slow Suite"}
                TOOLS = [{"id": "slow", "name": "Slow", "parameters": {"type": "object", "properties": {}}}]

                def call_tool(tool_id, arguments, context=None):
                    time.sleep(10)
                    return "late"
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )
        session = tool_registry.ExternalWorkerSession(server_file, Path(sys.executable))
        try:
            with (
                patch.object(tool_registry, "WORKER_RESPONSE_IDLE_TIMEOUT_SECONDS", 0.2),
                self.assertRaisesRegex(RuntimeError, "Tool worker stopped"),
            ):
                session.request("call", {"tool_id": "slow", "arguments": {}}, timeout_s=0.3)
        finally:
            session.close()

    # Return a readable error for unknown tool aliases.
    def test_call_ollama_tool_returns_error_for_unknown_alias(self):
        payload = tool_registry.call_ollama_tool({}, "missing__tool", {})

        self.assertEqual(payload, "Unknown tool: missing__tool")
