# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import threading
import pytest
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import patch
from django.test import SimpleTestCase
from Services import user_mcp_client
from Settings.mcp_json import UserMcpServerEntry


pytestmark = [pytest.mark.unit]


# Cover the persistent connection handling of the user MCP client: reuse, reconnect and cancellation.
class UserMcpPersistentSessionTests(SimpleTestCase):
    # Close any leftover persistent sessions before each test.
    def setUp(self):
        super().setUp()
        user_mcp_client.shutdown_all()

    # Close the persistent sessions opened by the test.
    def tearDown(self):
        user_mcp_client.shutdown_all()
        super().tearDown()

    # Build a stdio user MCP server entry for the fake server.
    @staticmethod
    def _entry() -> UserMcpServerEntry:
        return UserMcpServerEntry(
            config_key="test",
            server_id="test",
            display_name="Test MCP",
            transport="stdio",
            command="fake-mcp",
            args=[],
            env=None,
            cwd=None,
            url=None,
            headers=None,
        )

    # Verify that listing tools and repeated tool calls share one connection, which stays open until
    # shutdown_all() closes it.
    def test_reuses_connection_until_shutdown(self):
        # Arrange: a fake connection that counts enters, exits and tool calls.
        counters = {"entered": 0, "exited": 0, "calls": 0}

        # Fake MCP session exposing one echo tool.
        class FakeSession:
            # Return the single echo tool definition.
            async def list_tools(self):
                return SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name="echo",
                            description="Echo input",
                            inputSchema={"type": "object", "properties": {}},
                        )
                    ]
                )

            # Count the call and echo the tool name and argument back as text.
            async def call_tool(self, name, arguments):
                counters["calls"] += 1
                return SimpleNamespace(
                    isError=False,
                    structuredContent=None,
                    content=[SimpleNamespace(text=f"{name}:{arguments['value']}")],
                )

        # Open a fake connection, counting how often it is entered and exited.
        @asynccontextmanager
        async def fake_connect(_entry):
            counters["entered"] += 1
            try:
                yield FakeSession()
            finally:
                counters["exited"] += 1

        # Act: list tools and call the tool twice through the patched connector.
        with patch.object(user_mcp_client, "_connect_session", fake_connect):
            definitions, error = user_mcp_client.fetch_tool_definitions(self._entry())
            first = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {"value": "one"})
            second = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {"value": "two"})

            # Assert: one shared connection served all three requests and is still open.
            self.assertIsNone(error)
            self.assertEqual(definitions[0]["mcp_tool_name"], "echo")
            self.assertEqual(first, "echo:one")
            self.assertEqual(second, "echo:two")
            self.assertEqual(counters, {"entered": 1, "exited": 0, "calls": 2})

            # Shutting down closes the shared connection.
            user_mcp_client.shutdown_all()
            self.assertEqual(counters["exited"], 1)

    # Verify that a failed transport is discarded so the next invocation reconnects and succeeds.
    def test_reconnects_after_transport_failure(self):
        counters = {"entered": 0}

        # Fake MCP session whose first connection fails and later connections answer.
        class FakeSession:
            # Fail on the first connection, answer "reconnected" afterwards.
            async def call_tool(self, _name, _arguments):
                if counters["entered"] == 1:
                    raise ConnectionError("transport stopped")
                return SimpleNamespace(
                    isError=False,
                    structuredContent=None,
                    content=[SimpleNamespace(text="reconnected")],
                )

        # Open a fake connection and count how many connections were made.
        @asynccontextmanager
        async def fake_connect(_entry):
            counters["entered"] += 1
            yield FakeSession()

        with patch.object(user_mcp_client, "_connect_session", fake_connect):
            failed = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {})
            recovered = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {})

        self.assertIn("transport stopped", failed)
        self.assertEqual(recovered, "reconnected")
        self.assertEqual(counters["entered"], 2)

    # Verify that cancelling an entry aborts a tool call that is blocked inside the server: the caller
    # returns a message about the closed connection and the connection is exited.
    def test_cancel_entry_stops_active_call(self):
        # Events signalling that the call started and the connection closed, and the caller's result.
        started = threading.Event()
        exited = threading.Event()
        result = {}

        # Fake MCP session whose tool call blocks forever.
        class FakeSession:
            # Signal the start and then wait until the call is cancelled.
            async def call_tool(self, _name, _arguments):
                started.set()
                await asyncio.Event().wait()

        # Open a fake connection and signal when it is closed.
        @asynccontextmanager
        async def fake_connect(_entry):
            try:
                yield FakeSession()
            finally:
                exited.set()

        # Call the tool on a worker thread and store its result.
        def invoke() -> None:
            result["value"] = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {})

        # Act: start the blocking call, wait until it is running, then cancel the entry.
        with patch.object(user_mcp_client, "_connect_session", fake_connect):
            caller = threading.Thread(target=invoke, daemon=True)
            caller.start()
            self.assertTrue(started.wait(timeout=1))

            user_mcp_client.cancel_entry(self._entry())
            caller.join(timeout=1)

        # Assert: the call returned, the connection closed and the result mentions the closure.
        self.assertFalse(caller.is_alive())
        self.assertTrue(exited.is_set())
        self.assertIn("closed", result.get("value", "").lower())
