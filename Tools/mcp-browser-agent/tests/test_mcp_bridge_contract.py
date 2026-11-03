# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import os
from unittest.mock import AsyncMock, patch

import pytest


# Verify that the server is identified as browser_agent, exposes exactly the eight browser tools, and has a
# handler for each of them.
@pytest.mark.unit
def test_mcp_server_metadata_and_tool_surface(bridge_module) -> None:
    assert bridge_module.MCP_SERVER["id"] == "browser_agent"
    tool_ids = {tool["id"] for tool in bridge_module.TOOLS}
    assert tool_ids == {
        "browser_navigate",
        "browser_snapshot",
        "browser_click",
        "browser_key",
        "browser_scroll",
        "browser_text",
        "browser_wait_for_user",
        "browser_screenshot",
    }
    assert set(bridge_module.TOOL_HANDLERS) == tool_ids


# Verify that supports() accepts the four tool-calling engines and rejects an unknown engine.
@pytest.mark.unit
@pytest.mark.parametrize(
    "engine",
    ["ollama-service", "lms", "openai", "google-genai", "unknown-engine"],
)
def test_supports_known_engines(bridge_module, engine: str) -> None:
    expected = engine in ("ollama-service", "lms", "openai", "google-genai")
    assert bridge_module.supports(engine=engine) is expected


# Verify that a plain string is returned unchanged and a list of text blocks is joined with blank lines.
@pytest.mark.unit
def test_flatten_content_joins_text_chunks(bridge_module) -> None:
    # Minimal stand-in for an MCP text content block.
    class Block:
        # Store the text that _flatten_content reads from the text attribute.
        def __init__(self, text: str) -> None:
            self.text = text

    assert bridge_module._flatten_content("plain") == "plain"
    assert bridge_module._flatten_content([Block("a"), Block("b")]) == "a\n\nb"


# Verify that navigation keepalive settings use the 3 second interval and a message naming the URL with
# https:// added to a bare host.
@pytest.mark.unit
def test_browser_keepalive_settings_normalizes_navigate_url(bridge_module) -> None:
    interval, message = bridge_module._browser_keepalive_settings(
        "browser_navigate",
        {"url": "example.com"},
    )
    assert interval == 3.0
    assert "https://example.com" in message


# Verify that call_tool passes the tool id, arguments and context to _execute_browser_tool and returns its
# result (the executor itself is replaced by a mock).
@pytest.mark.unit
def test_call_tool_delegates_to_execute_browser_tool(bridge_module, monkeypatch) -> None:
    monkeypatch.setenv("ASLM_BROWSER_AGENT_INLINE", "1")
    with patch.object(
        bridge_module,
        "_execute_browser_tool",
        new_callable=AsyncMock,
        return_value="ok",
    ) as execute:
        result = asyncio.run(
            bridge_module.call_tool("browser_snapshot", {"full": False}, {})
        )
    execute.assert_awaited_once_with("browser_snapshot", {"full": False}, {})
    assert result == "ok"


# Verify that a tool other than browser_wait_for_user is refused with a "BLOCKED:" message while the
# browser is waiting for the user.
@pytest.mark.unit
def test_execute_browser_tool_local_blocked_while_waiting(bridge_module, monkeypatch) -> None:
    monkeypatch.setenv("ASLM_BROWSER_AGENT_INLINE", "1")
    import browser as browser_module

    # Mark the browser as waiting for the user and try to click.
    monkeypatch.setattr(browser_module, "_waiting_for_user", True)
    result = asyncio.run(
        bridge_module._execute_browser_tool_local(
            "browser_click",
            {"ref": "e0"},
            {},
        )
    )

    # The refusal may come back as plain text or wrapped with portal UI metadata.
    if isinstance(result, dict):
        text = str(result.get("model_context") or "")
    else:
        text = str(result)
    assert text.startswith("BLOCKED:")
