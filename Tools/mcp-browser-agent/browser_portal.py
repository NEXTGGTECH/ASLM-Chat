# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import base64
import json
import time
import uuid
from pathlib import Path
from typing import Any

from config import DOWNLOADS_DIR

# Cache of the lazily imported browser module (see _get_browser_mod).
_browser_mod = None


# Lazy-import browser module to avoid circular imports at module load time.
def _get_browser_mod():
    global _browser_mod
    if _browser_mod is None:
        import browser as _b
        _browser_mod = _b
    return _browser_mod


# Counter for logging published portal frames; it is reset whenever a portal session starts.
_PORTAL_FRAME_PUBLISH_LOG_COUNTER = 0


# Browser portal debug logging is intentionally disabled.
def append_browser_portal_debug_event(
    context: dict[str, Any] | None,
    event: str,
    **fields: Any,
) -> None:
    return None


# Frames and status

# Map a tool result string to a portal UI status (done, failed, waiting): "Error:" and "Tool execution
# failed:" prefixes mean failed, a "Blocked:" prefix means waiting, anything else is done.
def _status_from_result(result: Any) -> str:
    text = str(result or "").strip().lower()
    if text.startswith("error:") or text.startswith("tool execution failed:"):
        return "failed"
    if text.startswith("blocked:"):
        return "waiting"
    return "done"


# Capture a small JPEG viewport frame for the chat browser portal. Return the frame dict (inline base64
# preview, size, page URL and viewport size) or None when there is no page or the screenshot fails.
async def capture_browser_portal_frame(page: Any) -> dict[str, Any] | None:
    if page is None:
        return None

    try:
        data = await page.screenshot(type="jpeg", quality=58, full_page=False)
    except Exception:
        # The page may be closing or mid-navigation; the portal simply shows no new frame.
        return None

    viewport = page.viewport_size or {}
    frame: dict[str, Any] = {
        "mime": "image/jpeg",
        "preview": {
            "type": "inline_base64",
            "mime_type": "image/jpeg",
            "data_base64": base64.b64encode(data).decode("utf-8"),
        },
        "size_bytes": len(data),
        "url": getattr(page, "url", "") or "",
    }
    width = viewport.get("width") if isinstance(viewport, dict) else None
    height = viewport.get("height") if isinstance(viewport, dict) else None
    if width:
        frame["width"] = width
    if height:
        frame["height"] = height
    return frame


# Portal state files

# Resolve the on-disk root directory for portal state and events: below the module's Data/runtime folder
# when the context names one, otherwise below the configured downloads folder.
def browser_portal_root(context: dict[str, Any] | None = None) -> Path:
    safe_context = context or {}
    module_dir = str(safe_context.get("module_dir") or safe_context.get("project_dir") or "").strip()
    selected = safe_context.get("selected_tool_server_ids")
    sandbox_enabled = bool(safe_context.get("sandbox_enabled"))
    if isinstance(selected, list):
        sandbox_enabled = sandbox_enabled or any(str(item) == "sandbox" for item in selected)
    if module_dir:
        return Path(module_dir) / "Data" / "runtime" / "browser_portal"
    return DOWNLOADS_DIR / "browser_portal"


# Return the directory where portal UI events are queued.
def browser_portal_events_dir(context: dict[str, Any] | None = None) -> Path:
    return browser_portal_root(context) / "events"


# Return the path to the portal state.json file.
def browser_portal_state_path(context: dict[str, Any] | None = None) -> Path:
    return browser_portal_root(context) / "state.json"


# Reset portal session files and write initial waiting state; return new session_id. Counters and the
# cached a11y bundle are cleared, leftover queued events are deleted and the deadline is set from the
# timeout (at least one second).
def reset_browser_portal_state(context: dict[str, Any] | None, *, message: str, timeout_seconds: int) -> str:
    global _PORTAL_FRAME_PUBLISH_LOG_COUNTER
    _PORTAL_FRAME_PUBLISH_LOG_COUNTER = 0
    try:
        _get_browser_mod().reset_portal_a11y_state()
    except Exception:
        # Clearing the a11y cache is optional; the portal state is reset even if the browser module fails.
        pass

    # Create a fresh session with its folders and timing.
    root = browser_portal_root(context)
    events_dir = browser_portal_events_dir(context)
    session_id = uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)
    now = time.time()
    timeout_seconds = max(1, int(timeout_seconds))
    deadline_at = now + timeout_seconds

    # Drop events queued by a previous session.
    for path in events_dir.glob("event_*.json"):
        try:
            path.unlink()
        except OSError:
            # A file that cannot be removed (locked or already gone) is not fatal for a reset.
            pass

    # Publish the initial waiting state.
    _write_portal_state(
        context,
        {
            "ok": True,
            "status": "waiting",
            "session_id": session_id,
            "message": message,
            "timeout_seconds": timeout_seconds,
            "started_at": now,
            "deadline_at": deadline_at,
            "version": int(time.time() * 1000),
            "updated_at": now,
        },
    )
    return session_id


# Atomically write portal state.json with optional session_id guard. When a session_id is given and the
# stored state belongs to a different session, nothing is written and False is returned; otherwise the
# payload is stamped with that session id. The write goes through a temporary file that replaces the
# state file; on failure it is retried, then written in place as a last resort (OSError is re-raised if
# that fails too).
def _write_portal_state(
    context: dict[str, Any] | None,
    payload: dict[str, Any],
    *,
    session_id: str | None = None,
) -> bool:
    path = browser_portal_state_path(context)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Refuse to overwrite the state of a newer session.
    if session_id:
        current = read_browser_portal_state(context)
        current_session_id = str(current.get("session_id") or "")
        if current_session_id and current_session_id != session_id:
            append_browser_portal_debug_event(
                context,
                "portal_state_write_skipped_session_mismatch",
                requested_session_id=session_id,
                current_session_id=current_session_id,
                payload_status=payload.get("status"),
            )
            return False
        payload = {**payload, "session_id": session_id}
    text = json.dumps(payload, ensure_ascii=False)
    encoding = "utf-8"

    # Windows: os.replace may raise PermissionError if state.json is briefly
    # locked (AV, concurrent readers). Retry, then fall back to in-place write.
    last_exc: OSError | None = None
    for attempt in range(12):
        temp_path = path.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            temp_path.write_text(text, encoding=encoding)
            temp_path.replace(path)
            return True
        except PermissionError as exc:
            last_exc = exc
            try:
                if temp_path.is_file():
                    temp_path.unlink()
            except OSError:
                # Cleanup of the temporary file is best-effort; the next attempt uses a new name.
                pass
            time.sleep(0.012 * (attempt + 1))
        except OSError as exc:
            last_exc = exc
            try:
                if temp_path.is_file():
                    temp_path.unlink()
            except OSError:
                # Cleanup of the temporary file is best-effort; the next attempt uses a new name.
                pass
            time.sleep(0.012 * (attempt + 1))

    # All atomic attempts failed: write the state file in place.
    try:
        path.write_text(text, encoding=encoding)
        append_browser_portal_debug_event(
            context,
            "portal_state_write_inplace_fallback",
            path=str(path),
            error=str(last_exc) if last_exc else "",
        )
        return True
    except OSError as exc:
        append_browser_portal_debug_event(
            context,
            "portal_state_write_failed",
            path=str(path),
            error_type=type(exc).__name__,
            error=str(exc),
        )
        raise


# Read and parse the current portal state.json payload.
def read_browser_portal_state(context: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        payload = json.loads(browser_portal_state_path(context).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ok": False, "error": "No browser portal frame is available yet."}
    return payload if isinstance(payload, dict) else {"ok": False, "error": "Invalid browser portal state."}


# Portal event queue

# Write one UI event JSON file into the portal events directory. Files are named by timestamp so that
# they are read back in the order they were queued.
def enqueue_browser_portal_event(payload: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, Any]:
    events_dir = browser_portal_events_dir(context)
    events_dir.mkdir(parents=True, exist_ok=True)
    event_id = uuid.uuid4().hex
    event = {
        "id": event_id,
        "created_at": time.time(),
        **(payload if isinstance(payload, dict) else {}),
    }
    path = events_dir / f"event_{int(time.time() * 1000)}_{event_id}.json"
    path.write_text(json.dumps(event, ensure_ascii=False), encoding="utf-8")
    append_browser_portal_debug_event(
        context,
        "portal_event_enqueued",
        event_id=event_id,
        path=str(path),
        payload=event,
    )
    return {"ok": True, "event_id": event_id}


# Read and remove queued portal events, optionally filtered by session_id. Events of another session are
# left in the queue; unreadable or non-object files are deleted.
def _pop_browser_portal_events(
    context: dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    events = []
    events_dir = browser_portal_events_dir(context)
    events_dir.mkdir(parents=True, exist_ok=True)

    # Oldest first: the file names start with a millisecond timestamp.
    for path in sorted(events_dir.glob("event_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # An unreadable or half-written file is treated as junk and removed below.
            payload = None
        if isinstance(payload, dict):
            event_session_id = str(payload.get("session_id") or "")
            if session_id and event_session_id and event_session_id != session_id:
                append_browser_portal_debug_event(
                    context,
                    "portal_event_skipped_session_mismatch",
                    path=str(path),
                    requested_session_id=session_id,
                    event_session_id=event_session_id,
                    payload=payload,
                )
                continue
            try:
                path.unlink()
            except OSError:
                # The event is still delivered; a leftover file would only be read again later.
                pass
            events.append(payload)
            append_browser_portal_debug_event(
                context,
                "portal_event_popped",
                path=str(path),
                session_id=session_id,
                payload=payload,
            )
        else:
            # Remove the junk file.
            try:
                path.unlink()
            except OSError:
                # A file that cannot be removed now is retried on the next pop.
                pass
    return events


# Publishing and applying

# Capture frame and a11y bundle and merge them into portal state.json. The start time and deadline of the
# running session are kept from the existing state (or created from timeout_seconds), so they survive
# repeated publishes. Return the written payload.
async def publish_browser_portal_frame(
    page: Any,
    context: dict[str, Any] | None = None,
    *,
    status: str = "waiting",
    message: str = "",
    timeout_seconds: int | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    # Capture the viewport frame and the (throttled) a11y bundle.
    frame = await capture_browser_portal_frame(page)
    a11y: dict[str, Any] | None = None
    try:
        a11y = await _get_browser_mod().capture_portal_a11y_bundle(page)
    except Exception:
        # The a11y bundle is an optional extra; publish the frame without it.
        pass

    # Keep the timing of the current session; start new timing only when a timeout was requested.
    now = time.time()
    existing_state = read_browser_portal_state(context)
    existing_session_id = str(existing_state.get("session_id") or "")
    same_session = not session_id or not existing_session_id or existing_session_id == session_id
    started_at = existing_state.get("started_at") if same_session else None
    deadline_at = existing_state.get("deadline_at") if same_session else None
    if started_at is None and timeout_seconds is not None:
        started_at = now
    if deadline_at is None and timeout_seconds is not None:
        try:
            deadline_at = float(started_at if started_at is not None else now) + max(1, int(timeout_seconds))
        except (TypeError, ValueError):
            deadline_at = now + max(1, int(timeout_seconds))

    # Assemble the state payload; optional parts are included only when known.
    payload: dict[str, Any] = {
        "ok": True,
        "status": status,
        "frame": frame,
        "url": getattr(page, "url", "") if page is not None else "",
        "version": int(time.time() * 1000),
        "updated_at": now,
    }
    if a11y is not None:
        payload["a11y"] = a11y
    if message:
        payload["message"] = message
    if timeout_seconds is not None:
        payload["timeout_seconds"] = timeout_seconds
    if started_at is not None:
        payload["started_at"] = started_at
    if deadline_at is not None:
        payload["deadline_at"] = deadline_at
    _write_portal_state(context, payload, session_id=session_id)
    return payload


# Apply queued portal UI events (click, scroll, key, type, click_ref) to the page. Return True when a
# "finish" event was received, i.e. the user ended the portal session. A failing event is logged and
# skipped so the remaining events are still applied.
async def apply_browser_portal_events(
    page: Any,
    context: dict[str, Any] | None = None,
    *,
    session_id: str | None = None,
) -> bool:
    should_finish = False
    for event in _pop_browser_portal_events(context, session_id=session_id):
        event_type = str(event.get("type") or "").strip().lower()
        append_browser_portal_debug_event(
            context,
            "portal_event_apply_start",
            session_id=session_id,
            event_type=event_type,
            payload=event,
            page_url=getattr(page, "url", "") if page is not None else "",
        )

        # A finish event only sets the flag; without a page no other event can be applied.
        if event_type == "finish":
            should_finish = True
            append_browser_portal_debug_event(
                context,
                "portal_event_apply_finish_requested",
                session_id=session_id,
                payload=event,
            )
            continue
        if page is None:
            append_browser_portal_debug_event(
                context,
                "portal_event_apply_skipped_no_page",
                session_id=session_id,
                event_type=event_type,
                payload=event,
            )
            continue

        # Replay the event on the page by type. Click coordinates are given in the portal view's own size
        # and are scaled to the browser viewport.
        try:
            if event_type == "click":
                viewport = page.viewport_size or {"width": 1280, "height": 800}
                browser_w = int(viewport.get("width") or 1280)
                browser_h = int(viewport.get("height") or 800)
                view_w = max(1, int(event.get("viewport_width") or browser_w))
                view_h = max(1, int(event.get("viewport_height") or browser_h))
                x = float(event.get("x") or 0) * browser_w / view_w
                y = float(event.get("y") or 0) * browser_h / view_h
                await page.mouse.click(x, y)
                await page.wait_for_timeout(120)
                append_browser_portal_debug_event(
                    context,
                    "portal_event_apply_click_done",
                    session_id=session_id,
                    x=x,
                    y=y,
                    browser_width=browser_w,
                    browser_height=browser_h,
                    view_width=view_w,
                    view_height=view_h,
                )
            elif event_type == "scroll":
                # Scroll by the wheel deltas.
                await page.mouse.wheel(float(event.get("delta_x") or 0), float(event.get("delta_y") or 0))
                await page.wait_for_timeout(80)
                append_browser_portal_debug_event(
                    context,
                    "portal_event_apply_scroll_done",
                    session_id=session_id,
                    delta_x=event.get("delta_x"),
                    delta_y=event.get("delta_y"),
                )
            elif event_type == "key":
                # Press one key (an empty key only waits).
                key = str(event.get("key") or "")
                if key:
                    await page.keyboard.press(key)
                await page.wait_for_timeout(80)
                append_browser_portal_debug_event(
                    context,
                    "portal_event_apply_key_done",
                    session_id=session_id,
                    key=key,
                )
            elif event_type == "type":
                # Insert text into the focused element in one step.
                text = str(event.get("text") or "")
                if text:
                    await page.keyboard.insert_text(text)
                await page.wait_for_timeout(80)
                append_browser_portal_debug_event(
                    context,
                    "portal_event_apply_type_done",
                    session_id=session_id,
                    text_length=len(text),
                    text_preview=text[:200],
                )
            elif event_type == "click_ref":
                # Click an element of the last snapshot by its ref.
                ref = str(event.get("ref") or "").strip()
                if ref:
                    await _get_browser_mod()._click_by_role_and_name(ref)
                    await page.wait_for_timeout(120)
                    append_browser_portal_debug_event(
                        context,
                        "portal_event_apply_click_ref_done",
                        session_id=session_id,
                        ref=ref,
                    )
                else:
                    append_browser_portal_debug_event(
                        context,
                        "portal_event_apply_click_ref_missing",
                        session_id=session_id,
                    )
            else:
                # Unknown event types are only logged.
                append_browser_portal_debug_event(
                    context,
                    "portal_event_apply_unknown_type",
                    session_id=session_id,
                    event_type=event_type,
                    payload=event,
                )
        except Exception as exc:
            # A failing event must not prevent the remaining queued events from being applied.
            append_browser_portal_debug_event(
                context,
                "portal_event_apply_error",
                session_id=session_id,
                event_type=event_type,
                error_type=type(exc).__name__,
                error=str(exc),
                payload=event,
            )
            continue
    return should_finish


# Wrap a text tool result with UI metadata without changing model-facing text. Non-string results are
# returned unchanged; a string becomes a dict whose model_context is the original text, plus the portal
# kind, tool name, arguments, status and a fresh viewport frame (also repeated under "ui").
async def with_browser_portal_ui(
    result: Any,
    *,
    tool_name: str,
    arguments: dict[str, Any],
    page: Any,
) -> Any:
    if not isinstance(result, str):
        return result

    # Attach a fresh frame and the status derived from the text; the model sees only model_context.
    frame = await capture_browser_portal_frame(page)
    return {
        "model_context": result,
        "kind": "browser_portal",
        "tool": tool_name,
        "arguments": dict(arguments or {}),
        "status": _status_from_result(result),
        "frame": frame,
        "ui": {
            "kind": "browser_portal",
            "status": _status_from_result(result),
            "tool": tool_name,
            "arguments": dict(arguments or {}),
            "frame": frame,
        },
    }
