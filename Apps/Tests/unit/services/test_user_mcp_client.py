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


# Persistent user MCP connection tests.
class UserMcpPersistentSessionTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        user_mcp_client.shutdown_all()

    def tearDown(self):
        user_mcp_client.shutdown_all()
        super().tearDown()

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

    # list_tools and repeated calls must share one initialized transport.
    def test_reuses_connection_until_shutdown(self):
        counters = {"entered": 0, "exited": 0, "calls": 0}

        class FakeSession:
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

            async def call_tool(self, name, arguments):
                counters["calls"] += 1
                return SimpleNamespace(
                    isError=False,
                    structuredContent=None,
                    content=[SimpleNamespace(text=f"{name}:{arguments['value']}")],
                )

        @asynccontextmanager
        async def fake_connect(_entry):
            counters["entered"] += 1
            try:
                yield FakeSession()
            finally:
                counters["exited"] += 1

        with patch.object(user_mcp_client, "_connect_session", fake_connect):
            definitions, error = user_mcp_client.fetch_tool_definitions(self._entry())
            first = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {"value": "one"})
            second = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {"value": "two"})

            self.assertIsNone(error)
            self.assertEqual(definitions[0]["mcp_tool_name"], "echo")
            self.assertEqual(first, "echo:one")
            self.assertEqual(second, "echo:two")
            self.assertEqual(counters, {"entered": 1, "exited": 0, "calls": 2})

            user_mcp_client.shutdown_all()
            self.assertEqual(counters["exited"], 1)

    # A failed transport is discarded so the next invocation reconnects.
    def test_reconnects_after_transport_failure(self):
        counters = {"entered": 0}

        class FakeSession:
            async def call_tool(self, _name, _arguments):
                if counters["entered"] == 1:
                    raise ConnectionError("transport stopped")
                return SimpleNamespace(
                    isError=False,
                    structuredContent=None,
                    content=[SimpleNamespace(text="reconnected")],
                )

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

    def test_cancel_entry_stops_active_call(self):
        started = threading.Event()
        exited = threading.Event()
        result = {}

        class FakeSession:
            async def call_tool(self, _name, _arguments):
                started.set()
                await asyncio.Event().wait()

        @asynccontextmanager
        async def fake_connect(_entry):
            try:
                yield FakeSession()
            finally:
                exited.set()

        def invoke() -> None:
            result["value"] = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {})

        with patch.object(user_mcp_client, "_connect_session", fake_connect):
            caller = threading.Thread(target=invoke, daemon=True)
            caller.start()
            self.assertTrue(started.wait(timeout=1))

            user_mcp_client.cancel_entry(self._entry())
            caller.join(timeout=1)

        self.assertFalse(caller.is_alive())
        self.assertTrue(exited.is_set())
        self.assertIn("closed", result.get("value", "").lower())
