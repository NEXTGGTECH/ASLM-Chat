# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import atexit
import hashlib
import json
import logging
import os
import threading
from concurrent.futures import TimeoutError as FutureTimeoutError
from contextlib import asynccontextmanager, suppress
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamablehttp_client

from Settings.mcp_json import UserMcpServerEntry, _slugify
from Settings.proxy_policy import build_proxy_environment_overlay

logger = logging.getLogger(__name__)

# Timeouts in seconds: listing tools (also the session handshake), one tool call, and the extra
# slack the manager adds when it waits for the connection task so the inner timeout fires first.
LIST_TOOLS_TIMEOUT = 15.0
CALL_TOOL_TIMEOUT = 120.0
MANAGER_TIMEOUT_BUFFER = 5.0


# Build a stable, secret-free identity for one configured MCP connection.
def _entry_cache_key(entry: UserMcpServerEntry) -> str:
    payload = {
        "server_id": entry.server_id,
        "transport": entry.transport,
        "command": entry.command,
        "args": list(entry.args),
        "env": dict(entry.env or {}),
        "cwd": entry.cwd,
        "url": entry.url,
        "headers": dict(entry.headers or {}),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


# Normalize MCP tool input schemas into JSON Schema objects.
def _normalize_parameters_schema(schema: Any) -> dict[str, Any]:
    if schema is None:
        return {"type": "object", "properties": {}}
    if hasattr(schema, "model_dump"):
        dumped = schema.model_dump(mode="python", exclude_none=True)
    elif isinstance(schema, dict):
        dumped = dict(schema)
    else:
        dumped = {"type": "object", "properties": {}}

    if not isinstance(dumped, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(dumped)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    if not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}
    return normalized


# Convert MCP list_tools results into ASLM tool definition payloads.
def _tool_definitions_from_mcp_tools(
    server_id: str,
    mcp_tools: list[Any],
) -> tuple[list[dict[str, Any]], str | None]:
    definitions: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()
    error: str | None = None

    # Build one definition per tool, naming unnamed tools "tool_N".
    for index, tool in enumerate(mcp_tools, start=1):
        raw_name = str(getattr(tool, "name", "") or "").strip()
        if not raw_name:
            raw_name = f"tool_{index}"

        # Assign a stable slug and avoid collisions within one server.
        base_slug = _slugify(raw_name)
        slug = base_slug
        suffix = 2
        while slug in seen_slugs:
            slug = f"{base_slug}_{suffix}"
            suffix += 1
        seen_slugs.add(slug)

        description = str(getattr(tool, "description", "") or "").strip()
        input_schema = getattr(tool, "inputSchema", None)

        definitions.append(
            {
                "id": slug,
                "alias": f"{server_id}__{slug}",
                "name": raw_name,
                "description": description,
                "parameters": _normalize_parameters_schema(input_schema),
                "mcp_tool_name": raw_name,
            }
        )

    if not definitions:
        error = "Server returned no tools"
    return definitions, error


# Format one MCP call_tool result as plain text for the chat layer.
def _format_call_tool_result(result: Any) -> str:
    # An error result is reduced to the text of its content blocks.
    if getattr(result, "isError", False):
        parts: list[str] = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(str(text))
        message = "\n".join(parts).strip() or "Tool reported an error."
        return f"MCP tool error: {message}"

    # A successful result is the structured content (as JSON) followed by the text blocks.
    chunks: list[str] = []
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict) and structured:
        chunks.append(json.dumps(structured, ensure_ascii=False, indent=2))

    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            chunks.append(str(text))

    if not chunks:
        return "Tool returned no textual content."
    return "\n\n".join(chunks).strip()


# Open one short-lived MCP session for stdio or streamable HTTP transport.
@asynccontextmanager
async def _connect_session(entry: UserMcpServerEntry):
    # Streamable HTTP transport: connect to the server URL with the configured headers.
    if entry.transport == "http":
        assert entry.url
        headers = dict(entry.headers) if entry.headers else None
        async with streamablehttp_client(
            entry.url,
            headers=headers,
            timeout=30.0,
            sse_read_timeout=300.0,
        ) as streams:
            read_stream, write_stream, _ = streams
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=LIST_TOOLS_TIMEOUT)
                yield session
        return

    # Stdio transport: launch the configured command with the proxy environment overlay, sending
    # the child's stderr to the null device so it cannot pollute this process's output.
    assert entry.command
    devnull = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
    try:
        params = StdioServerParameters(
            command=entry.command,
            args=list(entry.args),
            env=build_proxy_environment_overlay(entry.env),
            cwd=entry.cwd,
        )
        async with stdio_client(params, errlog=devnull) as streams:
            read_stream, write_stream = streams
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=LIST_TOOLS_TIMEOUT)
                yield session
    finally:
        devnull.close()


# Own one MCP transport and serialize all requests through its lifetime task.
# Requests are queued as (operation, payload, timeout, future) tuples and answered one at a time
# by _serve, which holds the session open. After the first failed request, or when the task ends,
# the connection stops accepting requests and fails everything still queued.
class _PersistentMcpConnection:
    # Remember the server entry and start the task that serves the request queue.
    def __init__(self, entry: UserMcpServerEntry) -> None:
        self.entry = entry
        self._accepting_requests = True
        self._queue: asyncio.Queue[tuple[str, dict[str, Any], float, asyncio.Future[Any]]] = asyncio.Queue()
        self._task = asyncio.create_task(
            self._serve(),
            name=f"user-mcp-{entry.server_id}",
        )

    # Return whether the connection still accepts requests and its task is running.
    @property
    def is_alive(self) -> bool:
        return self._accepting_requests and not self._task.done()

    # Queue one request ("list_tools" or "call_tool") and wait for the result of the serving task.
    async def request(self, operation: str, payload: dict[str, Any], timeout: float) -> Any:
        if not self.is_alive:
            raise RuntimeError(f"MCP connection for {self.entry.server_id} is not running")

        result_future = asyncio.get_running_loop().create_future()
        await self._queue.put((operation, payload, timeout, result_future))
        return await result_future

    # Stop accepting requests and cancel the serving task, which closes the transport.
    async def close(self) -> None:
        self._accepting_requests = False
        if not self._task.done():
            self._task.cancel()
        with suppress(asyncio.CancelledError):
            await self._task

    # Keep one session open and answer queued requests one by one until the first failure or a
    # cancellation, then fail the in-flight and still queued requests.
    async def _serve(self) -> None:
        active_future: asyncio.Future[Any] | None = None
        failure: Exception | None = None
        try:
            async with _connect_session(self.entry) as session:
                while True:
                    # Take the next request and run it with its own timeout.
                    operation, payload, timeout, active_future = await self._queue.get()
                    try:
                        if operation == "list_tools":
                            result = await asyncio.wait_for(session.list_tools(), timeout=timeout)
                        elif operation == "call_tool":
                            result = await asyncio.wait_for(
                                session.call_tool(
                                    str(payload.get("name") or ""),
                                    payload.get("arguments") or {},
                                ),
                                timeout=timeout,
                            )
                        else:
                            raise ValueError(f"Unsupported MCP operation: {operation}")
                    except Exception as exc:
                        # A failed request poisons the connection: report it to the caller and
                        # leave the loop so the transport is closed and recreated on next use.
                        self._accepting_requests = False
                        failure = exc
                        if not active_future.done():
                            active_future.set_exception(exc)
                        active_future = None
                        break

                    # Hand the result to the waiting caller (unless it gave up in the meantime).
                    if not active_future.done():
                        active_future.set_result(result)
                    active_future = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._accepting_requests = False
            failure = exc
            logger.warning("Persistent user MCP connection failed for %s: %s", self.entry.server_id, exc)
        finally:
            # Whatever ended the task, fail the in-flight request and every queued one with the
            # original failure (or a generic "closed" error) so no caller waits forever.
            self._accepting_requests = False
            closed_error = failure or RuntimeError(
                f"MCP connection for {self.entry.server_id} was closed"
            )
            if active_future is not None and not active_future.done():
                active_future.set_exception(closed_error)
            while not self._queue.empty():
                _, _, _, queued_future = self._queue.get_nowait()
                if not queued_future.done():
                    queued_future.set_exception(closed_error)


# Host persistent MCP connections on one process-wide asyncio event loop.
# The loop runs on a daemon thread started on first use; synchronous callers submit coroutines
# to it and wait for the result. Connections are cached by a hash of the server configuration.
class _PersistentMcpSessionManager:
    # Create the manager without starting the loop thread.
    def __init__(self) -> None:
        # The background loop, its thread, an event set once the loop exists, and the lock that
        # guards starting and stopping the thread.
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._state_lock = threading.Lock()

        # Live connections by configuration hash (see _entry_cache_key).
        self._connections: dict[str, _PersistentMcpConnection] = {}

    # Run the event loop of the manager thread; on exit cancel the remaining tasks and close the loop.
    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()

    # Start the loop thread when it is not running and wait until its loop exists.
    def _ensure_started(self) -> None:
        # Fast path without the lock, then a second check under the lock so only one thread starts it.
        if self._thread is not None and self._thread.is_alive() and self._loop is not None:
            return

        with self._state_lock:
            if self._thread is not None and self._thread.is_alive() and self._loop is not None:
                return
            self._ready.clear()
            self._thread = threading.Thread(
                target=self._thread_main,
                name="aslm-user-mcp-sessions",
                daemon=True,
            )
            self._thread.start()
            self._ready.wait()

    # Send one request over the cached connection of the entry, replacing a dead connection
    # (or creating the first one) on the manager loop.
    async def _request_async(
        self,
        entry: UserMcpServerEntry,
        operation: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> Any:
        cache_key = _entry_cache_key(entry)
        connection = self._connections.get(cache_key)
        if connection is None or not connection.is_alive:
            if connection is not None:
                await connection.close()
            connection = _PersistentMcpConnection(entry)
            self._connections[cache_key] = connection
        return await connection.request(operation, payload, timeout)

    # Run one request on the manager loop and block the calling thread for the result.
    # Raise TimeoutError when the loop does not answer within the request timeout plus the
    # list-tools handshake time and a small buffer.
    def request(
        self,
        entry: UserMcpServerEntry,
        operation: str,
        payload: dict[str, Any],
        timeout: float,
    ) -> Any:
        self._ensure_started()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(
            self._request_async(entry, operation, payload, timeout),
            self._loop,
        )
        try:
            return future.result(timeout=LIST_TOOLS_TIMEOUT + timeout + MANAGER_TIMEOUT_BUFFER)
        except FutureTimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"MCP {operation} timed out for {entry.server_id}") from exc

    # Close every cached connection and forget them all.
    async def _shutdown_all_async(self) -> None:
        connections = list(self._connections.values())
        self._connections.clear()
        if connections:
            await asyncio.gather(
                *(connection.close() for connection in connections),
                return_exceptions=True,
            )

    # Close and forget the cached connection of one server entry, if any.
    async def _cancel_entry_async(self, entry: UserMcpServerEntry) -> None:
        connection = self._connections.pop(_entry_cache_key(entry), None)
        if connection is not None:
            await connection.close()

    # Close the active transport for one configured MCP server.
    # Do nothing when the loop thread is not running; log a warning when closing takes too long.
    def cancel_entry(self, entry: UserMcpServerEntry) -> None:
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None or not thread.is_alive():
            return
        future = asyncio.run_coroutine_threadsafe(self._cancel_entry_async(entry), loop)
        try:
            future.result(timeout=LIST_TOOLS_TIMEOUT + MANAGER_TIMEOUT_BUFFER)
        except FutureTimeoutError:
            future.cancel()
            logger.warning("Timed out while cancelling user MCP server %s", entry.server_id)

    # Close every cached connection but keep the loop thread alive for reuse.
    def shutdown_all(self) -> None:
        loop = self._loop
        thread = self._thread
        if loop is None or thread is None or not thread.is_alive():
            self._connections.clear()
            return
        future = asyncio.run_coroutine_threadsafe(self._shutdown_all_async(), loop)
        try:
            future.result(timeout=LIST_TOOLS_TIMEOUT + MANAGER_TIMEOUT_BUFFER)
        except FutureTimeoutError:
            future.cancel()
            logger.warning("Timed out while shutting down persistent user MCP sessions")

    # Close every connection, then stop the loop and join its thread. A later request starts a new thread.
    def close(self) -> None:
        self.shutdown_all()
        with self._state_lock:
            loop = self._loop
            thread = self._thread
            self._loop = None
            self._thread = None
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)


# Process-wide manager that all module-level functions below share.
_session_manager = _PersistentMcpSessionManager()


# Close every cached MCP transport while keeping the manager available for reuse.
def shutdown_all() -> None:
    _session_manager.shutdown_all()


# Stop one in-flight user MCP call and discard its cached connection.
def cancel_entry(entry: UserMcpServerEntry) -> None:
    _session_manager.cancel_entry(entry)


# Close all MCP transports and stop the loop thread when the interpreter exits, so no MCP server
# child process is left behind.
atexit.register(_session_manager.close)


# List tools from one user MCP server over a persistent connection.
def fetch_tool_definitions(entry: UserMcpServerEntry) -> tuple[list[dict[str, Any]], str | None]:
    try:
        listed = _session_manager.request(entry, "list_tools", {}, LIST_TOOLS_TIMEOUT)
        tools = list(getattr(listed, "tools", None) or [])
        return _tool_definitions_from_mcp_tools(entry.server_id, tools)
    except Exception as exc:  # pragma: no cover - runtime / network
        logger.warning("User MCP list_tools failed for %s: %s", entry.server_id, exc)
        return [], f"{type(exc).__name__}: {exc}"


# Invoke one MCP tool through the cached per-server connection.
def call_user_mcp_tool(entry: UserMcpServerEntry, mcp_tool_name: str, arguments: dict[str, Any]) -> str:
    try:
        # Call the tool and format its result as text.
        result = _session_manager.request(
            entry,
            "call_tool",
            {"name": mcp_tool_name, "arguments": arguments or {}},
            CALL_TOOL_TIMEOUT,
        )
        formatted = _format_call_tool_result(result)

        # GitHub search tools that hit the API rate limit are answered from the public HTML search instead.
        if str(entry.server_id or "").strip().lower() == "github":
            from Services.github_html_fallback import (
                GITHUB_SEARCH_TYPE_BY_TOOL,
                is_github_rate_limit_result,
                search_github_html,
            )

            if (
                mcp_tool_name in GITHUB_SEARCH_TYPE_BY_TOOL
                and is_github_rate_limit_result(formatted)
            ):
                try:
                    fallback = search_github_html(mcp_tool_name, arguments or {})
                except Exception as fallback_exc:  # noqa: BLE001
                    logger.warning(
                        "GitHub HTML fallback failed for %s: %s",
                        mcp_tool_name,
                        fallback_exc,
                    )
                    fallback = f"GITHUB_HTML_FALLBACK_FAILED: {fallback_exc}"
                if fallback:
                    return fallback
        return formatted
    except Exception as exc:  # pragma: no cover
        logger.exception("User MCP call_tool failed for %s.%s", entry.server_id, mcp_tool_name)
        return f"User MCP tool execution failed: {exc}"
