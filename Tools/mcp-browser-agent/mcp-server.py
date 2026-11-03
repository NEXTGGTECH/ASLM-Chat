# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool

# This server's folder and the project root; the folder is put on sys.path so the sibling modules below
# can be imported even though this file is loaded under a different module name.
SERVER_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVER_ROOT.parent.parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from browser_portal import (
    apply_browser_portal_events,
    publish_browser_portal_frame,
    reset_browser_portal_state,
    with_browser_portal_ui,
)
from browser_screenshot import capture_browser_screenshot
from browser_text import handle_browser_text

log = logging.getLogger("browser-agent")

# Identity of this tool server as shown to ASLM.
MCP_SERVER = {
    "id": "browser_agent",
    "name": "Browser Agent",
    "description": "Browser automation with snapshots, clicks, typing, and screenshots.",
}

# Tool definitions exposed to the model: id, display name, description and JSON-schema parameters.
TOOLS = [
    {
        "id": "browser_navigate",
        "name": "Browser Navigate",
        "description": (
            "Open a URL in the browser and return a compact controls-only snapshot of the loaded page. "
            "The snapshot lists text inputs, buttons/controls, and links with fresh ref IDs, roles, labels, and regions. "
            "It does not include page text by default; call browser_snapshot(full=true) when you need full page content. "
            "Always call this first before using any other browser tool — ref IDs are only valid after a navigate or snapshot call. "
            "Handles redirects automatically and reports the final URL."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "URL to navigate to"},
            },
            "required": ["url"],
        },
    },
    {
        "id": "browser_snapshot",
        "name": "Browser Snapshot",
        "description": (
            "Refresh the browser state and get fresh ref IDs after a page change. "
            "By default this returns a compact controls-only view: text inputs, buttons/controls, links, and a small JSON parsed state. "
            "It does not include page text by default. "
            "Set full=true when you need full page text, navigation/footer content, or the raw accessibility tree. "
            "This tool only observes. Use browser_scroll to reveal content outside the viewport. "
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full": {
                    "type": "boolean",
                    "description": "Include full page text and raw accessibility tree (default: false)",
                    "default": False,
                },
            },
        },
    },
    {
        "id": "browser_click",
        "name": "Browser Click",
        "description": (
            "Click one interactive element by its ref ID. "
            "ref: single element ref ID from the snapshot (e.g. 'e5'). "
            "Returns a compact updated snapshot after the action. "
            "If the click triggers navigation to a new URL, returns a full snapshot of the new page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ref": {
                    "type": "string",
                    "description": "Single element ref ID (for example 'e5')",
                },
            },
            "required": ["ref"],
        },
    },
    {
        "id": "browser_key",
        "name": "Browser Key",
        "description": (
            "Press one keyboard key on the current page or focused element. "
            "Use for Enter, Escape, Tab, ArrowDown, ArrowUp, Space, PageDown, PageUp, and shortcuts such as Control+A. "
            "Do not use this for text entry; use browser_text to write, edit, replace, or delete text. "
            "Returns a compact updated snapshot after the key press."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Keyboard key or shortcut to press.",
                },
            },
            "required": ["key"],
        },
    },
    {
        "id": "browser_scroll",
        "name": "Browser Scroll",
        "description": (
            "Scroll the page viewport up or down, then return a compact snapshot. "
            "Use before browser_snapshot when needed to reveal content outside the viewport."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "direction": {
                    "type": "string",
                    "enum": ["up", "down"],
                    "description": "Scroll direction.",
                },
                "amount": {
                    "type": "integer",
                    "description": "Pixels to scroll (default: 700).",
                    "default": 700,
                },
            },
            "required": ["direction"],
        },
    },
    {
        "id": "browser_text",
        "name": "Browser Text",
        "description": (
            "Read, set, replace, or delete text in the active page editor/input. "
            "Use for text entry, multi-line text, code editors, and rich text editors. "
            "If text is provided and action is omitted, the tool sets the field text. "
            "ref is optional; when omitted the tool uses the focused editor or the first visible editor-like control. "
            "Supports textarea/input/contenteditable plus common CodeMirror and Ace editors. "
            "For action='set', provide text. "
            "For action='replace', provide old_text/new_text or range plus text. "
            "For action='delete', provide old_text, range, or all=true. "
            "range is 1-based line syntax like '3:5'; insertion before line 3 can be expressed as '3:2'. "
            "The tool verifies the final editor value and returns before/after lengths plus a compact snapshot."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["read", "set", "replace", "delete"],
                    "description": "Text operation to perform. Omit when providing text; that is treated as set.",
                },
                "ref": {
                    "type": "string",
                    "description": "Optional element ref ID from the current snapshot.",
                },
                "text": {
                    "type": "string",
                    "description": "Replacement text for set, range replace, or insertion.",
                },
                "old_text": {
                    "type": "string",
                    "description": "Exact text to replace or delete.",
                },
                "new_text": {
                    "type": "string",
                    "description": "Replacement text when using old_text matching.",
                },
                "replace_all": {
                    "type": "boolean",
                    "description": "Replace/delete all exact matches instead of requiring a single match.",
                    "default": False,
                },
                "range": {
                    "type": "string",
                    "description": "1-based line range such as '12:18'. Use '12:11' to insert before line 12.",
                },
                "all": {
                    "type": "boolean",
                    "description": "For action='delete', clear the whole editor.",
                    "default": False,
                },
                "press_enter": {
                    "type": "boolean",
                    "description": "Press Enter after writing text.",
                    "default": False,
                },
            },
        },
    },
    {
        "id": "browser_wait_for_user",
        "name": "Browser Wait For User",
        "description": (
            "Pause automation and display a message asking the user to perform a manual action in the browser. "
            "Use when the page requires human interaction that cannot be automated: CAPTCHA solving, login with credentials, age verification gate, cookie consent popup, or 2FA prompt. "
            "message should clearly describe what the user needs to do. "
            "The browser remains open and the session stays active during the wait. "
            "After timeout_seconds the tool resumes and returns a fresh snapshot of whatever state the page is in."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "What the user needs to do"},
                "timeout_seconds": {
                    "type": "integer",
                    "description": "Seconds to wait (default: 45)",
                    "default": 45,
                },
            },
            "required": ["message"],
        },
    },
    {
        "id": "browser_screenshot",
        "name": "Browser Screenshot",
        "description": (
            "Take a PNG screenshot of the current browser page and save it to the currently active sandbox workspace. "
            "full_page=False (default) captures only the visible viewport. "
            "full_page=True captures the entire scrollable page height. "
            "For vision-capable models, returns an inline image preview that can be inspected visually. "
            "For non-vision models, returns image metadata, file paths, and a text placeholder. "
            "Use this for visual verification when the accessibility/text snapshot is not enough."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "full_page": {
                    "type": "boolean",
                    "description": "Capture full scrollable page (default: false)",
                    "default": False,
                },
            },
        },
    },
]


# Tool server API

# Return whether this tool server can be offered for the given engine; only the listed engines support
# tool-calling. The model name does not affect the answer.
def supports(engine: str | None = None, model_name: str | None = None) -> bool:
    return engine in ("ollama-service", "lms", "openai", "google-genai")


# Result and keepalive helpers

# Convert MCP-style content payloads into plain text for local tool execution. Strings pass through, lists
# of content items are joined with blank lines (using each item's text attribute when it has one), and
# any other object is converted through its text attribute or str().
def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            text = getattr(item, "text", None)
            if text is not None:
                chunks.append(str(text))
            elif isinstance(item, str):
                chunks.append(item)
            else:
                chunks.append(str(item))
        return "\n\n".join(chunk for chunk in chunks if chunk)

    text = getattr(content, "text", None)
    if text is not None:
        return str(text)

    return str(content)


# Run a coroutine while optionally sending keepalive logs to an MCP session. Without a session the
# coroutine is just awaited; with one, a debug log message is sent every interval seconds until the
# coroutine finishes or a send fails (the work itself continues in that case).
async def _run_with_keepalive(coro, session=None, interval: float = 3.0, message: str = "working..."):
    if session is None:
        return await coro

    # The result of the coroutine and the event that tells the keepalive loop to stop.
    result = None
    done = asyncio.Event()

    # Periodically send debug log messages until the main coroutine finishes.
    async def _keepalive_loop() -> None:
        while not done.is_set():
            # Sleep for one interval, or wake up early when the work is done.
            try:
                await asyncio.wait_for(asyncio.shield(done.wait()), timeout=interval)
            except asyncio.TimeoutError:
                # A timeout is the normal case: the interval elapsed without the work finishing.
                pass

            if done.is_set():
                continue

            try:
                send_log_message = getattr(session, "send_log_message", None)
                if send_log_message is not None:
                    await send_log_message(level="debug", data=message, logger="browser-agent")
            except BaseException:
                # A failing session stops the keepalives; the main coroutine is not affected.
                break

    # Run the tool coroutine and signal completion to the keepalive loop.
    async def _run() -> None:
        nonlocal result
        result = await coro
        done.set()

    await asyncio.gather(_run(), _keepalive_loop())
    return result


# Pick keepalive interval and status message for a given browser tool name: navigation names its target
# URL (https:// is assumed when missing), waiting for the user polls less often, others use the default.
def _browser_keepalive_settings(name: str, arguments: dict[str, Any] | None) -> tuple[float, str]:
    args = arguments or {}
    if name == "browser_navigate":
        target_url = str(args.get("url", "")).strip()
        if target_url and not target_url.startswith("http"):
            target_url = "https://" + target_url
        return 3.0, f"navigating to {target_url or 'page'}..."
    if name == "browser_wait_for_user":
        return 5.0, "waiting for user..."
    return 3.0, "working..."


# Tool execution

# Execute one browser action in-process and return plain text or structured output. The action runs on the
# dedicated browser event loop with keepalive logs to the MCP session. A browser that died is relaunched
# and the action retried once; other failures come back as "Error: ..." text.
async def _execute_browser_tool_local(
    name: str,
    arguments: dict[str, Any] | None,
    context: dict[str, Any] | None = None,
) -> Any:
    # Imported lazily so that loading this server does not start importing Camoufox and Playwright.
    from browser import (
        _click_by_role_and_name,
        run_in_browser_loop,
        _take_compact_snapshot,
        _take_snapshot,
        _wait_for_spa_content,
        is_browser_closed_error,
        last_known_url,
        log,
        state,
    )
    import browser as browser_module

    # Normalize the call: the context always names the project folders, and the MCP session (if any) is
    # used for keepalive messages while the action runs.
    args = arguments or {}
    safe_context = {
        "module_dir": str(PROJECT_ROOT),
        "project_dir": str(PROJECT_ROOT),
        **(context or {}),
    }
    session = safe_context.get("mcp_session")
    keepalive_interval, keepalive_message = _browser_keepalive_settings(name, args)

    # Run the requested browser tool against the shared Playwright page.
    async def _run_action() -> Any:
        state.tool_context = safe_context

        # While the user is completing a manual step, only another wait call may use the browser.
        if browser_module._waiting_for_user and name != "browser_wait_for_user":
            return (
                "BLOCKED: browser is waiting for user action.\n"
                "The user has not finished yet (login / CAPTCHA / confirmation).\n"
                "Call browser_wait_for_user again to wait longer or re-check progress."
            )

        # Make sure a browser is running. Every tool except navigate needs a page: after a relaunch the
        # last known URL is reopened (click and text then get a fresh snapshot, since their refs are
        # stale); without any page the model is told to navigate first.
        launched_browser = await state.ensure_open()

        if name not in {"browser_navigate"}:
            current_url = state.page.url if state.page else "about:blank"
            if not current_url or current_url in {"about:blank", ""}:
                restore_url = last_known_url()
                if launched_browser and restore_url:
                    log.info("Restoring browser page after relaunch: %s", restore_url)
                    await state.page.goto(restore_url, wait_until="domcontentloaded", timeout=30000)
                    await _wait_for_spa_content(state.page, timeout_ms=6000)
                    if name in {"browser_click", "browser_text"}:
                        restored_snapshot = _flatten_content(
                            await _take_snapshot(
                                action_context=f"Restored browser after relaunch: {restore_url}",
                                run_dismiss=True,
                            )
                        )
                        return (
                            "Browser was relaunched after the previous browser process disappeared. "
                            "The old element refs are no longer valid. Use the fresh refs below and retry the action.\n\n"
                            f"{restored_snapshot}"
                        )
                    current_url = state.page.url if state.page else "about:blank"

            if not current_url or current_url in {"about:blank", ""}:
                return f"No page is open yet. Call browser_navigate(url) first, then retry {name}."

        # Dispatch by tool name; each branch returns text for the model (or a structured screenshot).
        try:
            # Open the URL (https:// is assumed when missing) and report redirects in the snapshot.
            if name == "browser_navigate":
                target_url = str(args.get("url", "")).strip()
                if not target_url:
                    return "Error: url is required."
                if not target_url.startswith("http"):
                    target_url = "https://" + target_url

                log.info("Navigating to: %s", target_url)
                await state.page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
                await _wait_for_spa_content(state.page, timeout_ms=6000)

                new_url = state.page.url
                action_context = f"Navigated to {new_url}"
                if new_url != target_url:
                    action_context = f"Navigation: {target_url} -> {new_url} (redirected)"

                return _flatten_content(await _take_snapshot(action_context=action_context, run_dismiss=True))

            # Refresh the snapshot, optionally with full page text and the accessibility tree.
            if name == "browser_snapshot":
                full = bool(args.get("full", False))
                return _flatten_content(
                    await _take_snapshot(
                        full=full,
                    )
                )

            # Click an element by ref; a click that changes the URL returns a detailed snapshot of the new
            # page, otherwise a compact one.
            if name == "browser_click":
                ref = args.get("ref")

                if ref is None:
                    return "Error: ref is required."

                ref_value = str(ref)
                previous_url = state.page.url

                log.info("Clicking element: %s", ref_value)
                await _click_by_role_and_name(ref_value)
                await state.page.wait_for_timeout(1500)

                new_url = state.page.url
                if new_url != previous_url:
                    return _flatten_content(
                        await _take_snapshot(action_context=f"Clicked [{ref_value}] -> navigated to {new_url}")
                    )

                return _flatten_content(await _take_compact_snapshot(action_context=f"Clicked [{ref_value}]"))

            # Press one key or shortcut on the page.
            if name == "browser_key":
                key_name = str(args.get("key", "")).strip()
                if not key_name:
                    return "Error: key is required."
                log.info("Pressing key: %s", key_name)
                await state.page.keyboard.press(key_name)
                await state.page.wait_for_timeout(800)
                return _flatten_content(await _take_compact_snapshot(action_context=f"Pressed {key_name}"))

            # Scroll the viewport with the mouse wheel.
            if name == "browser_scroll":
                direction = str(args.get("direction", "")).strip().lower()
                if direction not in {"up", "down"}:
                    return "Error: direction must be 'up' or 'down'."
                amount = int(args.get("amount", 700))
                delta = amount if direction == "down" else -amount
                await state.page.mouse.wheel(0, delta)
                await state.page.wait_for_timeout(600)
                return _flatten_content(
                    await _take_compact_snapshot(action_context=f"Scrolled {direction} {amount}px")
                )

            # Read or edit text in an editor through the browser_text handler.
            if name == "browser_text":
                return await handle_browser_text(
                    args,
                    page=state.page,
                    keyboard=state.page.keyboard,
                    take_snapshot=_take_compact_snapshot,
                    flatten_content=_flatten_content,
                )

            # Pause for a manual step (CAPTCHA, login, ...): announce it on stderr, mark the browser as
            # waiting, and for up to timeout_seconds mirror the page to the portal and apply the user's
            # portal events, ending early on a finish event.
            if name == "browser_wait_for_user":
                message = str(args.get("message", "Please interact with the browser."))
                timeout_seconds = int(args.get("timeout_seconds", 45))

                log.info("Waiting %ss for user: %s", timeout_seconds, message)
                print(f"\n{'=' * 60}", file=sys.stderr)
                print("[browser-agent] USER ACTION REQUIRED:", file=sys.stderr)
                print(f"  {message}", file=sys.stderr)
                print(f"  Waiting {timeout_seconds} seconds...", file=sys.stderr)
                print(f"{'=' * 60}\n", file=sys.stderr)
                sys.stderr.flush()

                # Start a new portal session and publish the first waiting frame.
                browser_module._waiting_for_user = True
                wait_started = asyncio.get_running_loop().time()
                portal_session_id = reset_browser_portal_state(
                    safe_context,
                    message=message,
                    timeout_seconds=timeout_seconds,
                )
                await publish_browser_portal_frame(
                    state.page,
                    safe_context,
                    status="waiting",
                    message=message,
                    timeout_seconds=timeout_seconds,
                    session_id=portal_session_id,
                )

                # Poll: apply portal events, publish a fresh frame, and probe that the page is still alive.
                wait_exit_reason = "timeout"
                try:
                    while asyncio.get_running_loop().time() - wait_started < timeout_seconds:
                        should_finish = await apply_browser_portal_events(
                            state.page,
                            safe_context,
                            session_id=portal_session_id,
                        )
                        await publish_browser_portal_frame(
                            state.page,
                            safe_context,
                            status="waiting",
                            message=message,
                            timeout_seconds=timeout_seconds,
                            session_id=portal_session_id,
                        )
                        if should_finish:
                            wait_exit_reason = "finish_event"
                            break
                        try:
                            await state.page.evaluate("() => document.readyState")
                        except Exception as exc:
                            if is_browser_closed_error(exc):
                                wait_exit_reason = "browser_closed"
                                raise

                            # Other probe errors are transient; keep serving the portal.
                            pass
                        await asyncio.sleep(0.1)
                    else:
                        # The loop ended without a break: the wait timed out.
                        wait_exit_reason = "timeout"
                except Exception:
                    raise
                finally:
                    # Always clear the waiting flag and publish a final "done" frame, even on errors.
                    browser_module._waiting_for_user = False
                    await publish_browser_portal_frame(
                        state.page,
                        safe_context,
                        status="done",
                        message=message,
                        timeout_seconds=timeout_seconds,
                        session_id=portal_session_id,
                    )

                return _flatten_content(
                    await _take_snapshot(
                        action_context=f"Resumed after {timeout_seconds}s user wait",
                        run_dismiss=True,
                    )
                )

            # Save a PNG screenshot and return its structured description.
            if name == "browser_screenshot":
                full_page = bool(args.get("full_page", False))
                return await capture_browser_screenshot(full_page, safe_context)

            return f"Unknown tool: {name}"
        except Exception as exc:
            # A timeout or a dead browser closes the session and propagates so the caller can retry on a
            # fresh one; any other failure is reported to the model as text.
            if isinstance(exc, asyncio.TimeoutError) or is_browser_closed_error(exc):
                await state.close()
                raise
            return f"Error: {type(exc).__name__}: {exc}"

    # Execute once and attach portal UI metadata when the result is plain text.
    async def _perform_once() -> Any:
        result = await _run_action()
        return await with_browser_portal_ui(
            result,
            tool_name=name,
            arguments=args,
            page=state.page,
        )

    # Retry once after browser death by closing state and relaunching.
    async def _perform() -> Any:
        for attempt in range(2):
            try:
                return await _perform_once()
            except Exception as exc:
                if attempt == 0 and (isinstance(exc, asyncio.TimeoutError) or is_browser_closed_error(exc)):
                    log.warning("Browser action failed because the browser session died; retrying after relaunch: %s", exc)
                    await state.close()
                    continue
                return f"Error: {type(exc).__name__}: {exc}"

    # Run the whole action on the browser loop, sending keepalives to the MCP session meanwhile.
    return await run_in_browser_loop(
        _perform(),
        session=session,
        interval=keepalive_interval,
        message=keepalive_message,
    )


# Execute one browser action through the worker subprocess or inline when configured. Inside the worker
# itself (ASLM_BROWSER_AGENT_WORKER=1) or with ASLM_BROWSER_AGENT_INLINE=1 the action runs in this process;
# otherwise it is forwarded to the shared browser process manager.
async def _execute_browser_tool(name: str, arguments: dict[str, Any] | None, context: dict[str, Any] | None = None) -> Any:
    if os.getenv("ASLM_BROWSER_AGENT_WORKER") == "1" or os.getenv("ASLM_BROWSER_AGENT_INLINE") == "1":
        return await _execute_browser_tool_local(name, arguments or {}, context or {})

    from browser_process import browser_process_manager

    # Forward the call to the worker subprocess with the same context and keepalive settings.
    safe_context = {
        "module_dir": str(PROJECT_ROOT),
        "project_dir": str(PROJECT_ROOT),
        **(context or {}),
    }
    session = safe_context.get("mcp_session")
    keepalive_interval, keepalive_message = _browser_keepalive_settings(name, arguments or {})
    return await browser_process_manager.call(
        name,
        arguments or {},
        safe_context,
        session=session,
        interval=keepalive_interval,
        message=keepalive_message,
    )


# Tool registration

# Build an ASLM-compatible per-tool wrapper that delegates to _execute_browser_tool.
def _make_tool_handler(tool_id: str):
    # Delegate one ASLM tool id to the shared browser execution pipeline.
    async def _handler(arguments: dict[str, Any] | None, context: dict[str, Any] | None = None) -> Any:
        return await _execute_browser_tool(tool_id, arguments or {}, context or {})

    return _handler


# Per-tool handlers keyed by tool id, in the form ASLM expects.
TOOL_HANDLERS = {
    tool["id"]: _make_tool_handler(tool["id"])
    for tool in TOOLS
}


# Generic ASLM-compatible dispatcher for Browser Agent tools.
async def call_tool(tool_id: str, arguments: dict[str, Any] | None, context: dict[str, Any] | None = None) -> Any:
    return await _execute_browser_tool(tool_id, arguments or {}, context or {})


# Attach Browser Agent tools to the provided MCP server instance.
def register_tools(server) -> None:
    # Describe the Browser Agent tool surface for MCP list_tools.
    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name=tool["id"],
                description=tool["description"],
                inputSchema=tool["parameters"],
            )
            for tool in TOOLS
        ]

    # Execute the requested Browser Agent tool and return MCP CallToolResult. The MCP session of the
    # current request is passed along for keepalive messages; structured results are returned as JSON text.
    @server.call_tool()
    async def server_call_tool(name: str, arguments: dict[str, Any] | None) -> CallToolResult:
        log.info("server_call_tool: %s", name)
        session = None
        try:
            session = server.request_context.session
        except Exception:
            # Outside a request context there is no session; run without keepalive messages.
            session = None

        result = await _execute_browser_tool(
            name,
            arguments or {},
            {
                "mcp_session": session,
                "module_dir": str(PROJECT_ROOT),
                "project_dir": str(PROJECT_ROOT),
            },
        )
        if isinstance(result, str):
            text = result
        else:
            text = json.dumps(result, ensure_ascii=False, indent=2)
        return CallToolResult(content=[TextContent(type="text", text=text)])
