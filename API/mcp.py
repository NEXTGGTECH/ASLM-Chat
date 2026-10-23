# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import atexit
import contextlib
import copy
import importlib.util
import inspect
import json
import logging
import os
import re
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from types import ModuleType
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

from Services import user_mcp_client
from Settings import mcp_json
from Settings import skills as skills_config
from Settings import settings as runtime_settings
from Settings.proxy_policy import apply_loopback_proxy_bypass

logger = logging.getLogger(__name__)

TOOLS_DIR = Path(__file__).resolve().parent.parent / "Tools"
MODULE_ROOT = Path(__file__).resolve().parent.parent
SERVER_FILENAME = "mcp-server.py"
WORKER_FILE = Path(__file__).resolve().parent.parent / "Services" / "tool_worker.py"
SERVER_DISPATCHER_NAMES = ("call_tool", "run_tool", "execute_tool", "execute")
SERVER_METADATA_NAMES = ("MCP_SERVER", "SERVER")
TOOL_HANDLER_NAMES = ("TOOL_HANDLERS", "TOOL_EXECUTORS")
TOOL_PREPARER_NAMES = ("TOOL_PREPARERS", "TOOL_ARGUMENT_PREPARERS")
WORKER_RESPONSE_IDLE_TIMEOUT_SECONDS = 20.0
WORKER_TIMEOUT_BUFFER_SECONDS = 45.0
WORKER_DEFAULT_TIMEOUT_SECONDS = 180.0
WORKER_MAX_TIMEOUT_SECONDS = 7200.0

_SERVER_CACHE_SIGNATURE: tuple[tuple[str, int], ...] | None = None
_SERVER_CACHE: dict[str, dict[str, Any]] = {}
_SCOPED_REGISTRY_CACHE: dict[str, tuple[tuple[tuple[str, int], ...], dict[str, dict[str, Any]]]] = {}
_WORKER_SESSION_LOCK = threading.Lock()
_WORKER_SESSIONS: dict[str, "ExternalWorkerSession"] = {}
_ASYNC_CALLABLE_RUNNER: "AsyncCallableRunner | None" = None
_ASYNC_CALLABLE_RUNNER_LOCK = threading.Lock()
_SEARCH_IO_LOG_LOCK = threading.Lock()
_SEARCH_IO_LOG_PATH = TOOLS_DIR / "mcp-web-search" / "logs" / "model_search_io.json"
_TOOL_COOLDOWN_SECONDS = 30.0
_TOOL_COOLDOWN_LOCK = threading.Lock()
_TOOL_COOLDOWNS: dict[str, float] = {}
_ACTIVE_TOOL_EXECUTIONS_LOCK = threading.Lock()
_ACTIVE_TOOL_EXECUTIONS: set["ActiveToolExecution"] = set()
TOOL_CALL_QUOTAS = {
    "web_search": 15,
    "read_page": 10,
}
HIGH_EFFORT_WEB_SEARCH_QUOTA = 3
INSTANT_TOOL_CALL_QUOTAS = {
    "web_search": 2,
    "read_page": 2,
}
INSTANT_SEARCH_BATCH_LIMIT = 3

INSTANT_WEB_SEARCH_DESCRIPTION = (
    "Fast low-effort web lookup. Each call accepts one query or a batch of up to three. "
    "The description is the only search text shown to the user."
)
INSTANT_WEB_SEARCH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "query": {
            "oneOf": [
                {"type": "string", "minLength": 1, "maxLength": 220},
                {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": INSTANT_SEARCH_BATCH_LIMIT,
                    "items": {"type": "string", "minLength": 1, "maxLength": 220},
                },
            ],
            "description": "One web query or up to three independent queries.",
        },
        "description": {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
            "description": "Short user-facing action label in the user's language.",
        },
        "operators": {
            "type": "object",
            "description": (
                "Optional shared constraints: exact_phrases, or_terms, or_groups, exclude_terms, "
                "site_include, site_exclude, file_types, title_terms, url_terms, after, before."
            ),
        },
    },
    "required": ["query", "description"],
}


def _instant_web_search_contract(required_batch_size: int | None = None) -> tuple[str, dict[str, Any]]:
    """Return the instant search description/schema for this generation request."""

    if required_batch_size != INSTANT_SEARCH_BATCH_LIMIT:
        return INSTANT_WEB_SEARCH_DESCRIPTION, copy.deepcopy(INSTANT_WEB_SEARCH_SCHEMA)

    schema = copy.deepcopy(INSTANT_WEB_SEARCH_SCHEMA)
    schema["properties"]["query"] = {
        "type": "array",
        "minItems": INSTANT_SEARCH_BATCH_LIMIT,
        "maxItems": INSTANT_SEARCH_BATCH_LIMIT,
        "items": {"type": "string", "minLength": 1, "maxLength": 220},
        "description": (
            "Exactly three complementary web queries. They run concurrently in one forced "
            "instant search call."
        ),
    }
    description = (
        "Forced instant web lookup. Every call must contain exactly three complementary query "
        "strings. Effort is fixed to low. The description is the only search text shown to the user."
    )
    return description, schema
TOOL_BLOCK_FINAL_PROMPT = (
    "Tool loop guard: repeated tool calls were blocked because they duplicate recent work, "
    "hit a quota, or are in cooldown. Do not call tools again. Use the evidence already "
    "present in the conversation and answer the user now. If the collected evidence is "
    "insufficient for a claim, say that clearly instead of searching again."
)
TOOL_CANCEL_FINAL_PROMPT = (
    "The user stopped the active tool call. Do not call any tool again in this response "
    "and do not restart the cancelled work under a new session. Briefly acknowledge the stop "
    "or answer only from evidence already present in the conversation."
)
NON_MODEL_TOOL_SERVER_IDS = {"deep_research"}


class ToolExecutionCancelled(Exception):
    """Raised inside a tool runner after its owning generation is stopped."""


TOOL_EXECUTION_CANCELLED = object()


class ActiveToolExecution:
    """Track and cancel one in-flight tool invocation."""

    def __init__(self, generation_id: str = "", *, research_session_id: str = "") -> None:
        self.generation_id = str(generation_id or "")
        self.research_session_id = str(research_session_id or "").strip()
        self.cancelled = threading.Event()
        self._lock = threading.Lock()
        self._cancel_callback = None

    def set_cancel_callback(self, callback) -> None:
        call_immediately = False
        with self._lock:
            if self.cancelled.is_set():
                call_immediately = True
            else:
                self._cancel_callback = callback
        if call_immediately and callable(callback):
            callback()

    def clear_cancel_callback(self) -> None:
        with self._lock:
            self._cancel_callback = None

    def cancel(self) -> None:
        with self._lock:
            if self.cancelled.is_set():
                return
            self.cancelled.set()
            callback = self._cancel_callback
        if callable(callback):
            try:
                callback()
            except Exception as exc:  # noqa: BLE001 - cancellation remains best effort
                logger.warning("Failed to stop active tool resource: %s", exc)


def _register_active_tool(execution: ActiveToolExecution) -> None:
    with _ACTIVE_TOOL_EXECUTIONS_LOCK:
        _ACTIVE_TOOL_EXECUTIONS.add(execution)


def _unregister_active_tool(execution: ActiveToolExecution) -> None:
    with _ACTIVE_TOOL_EXECUTIONS_LOCK:
        _ACTIVE_TOOL_EXECUTIONS.discard(execution)


def abort_active_tools(generation_id: str | None = None) -> int:
    """Cancel tool invocations owned by one generation, or all when omitted."""

    requested_id = str(generation_id or "")
    with _ACTIVE_TOOL_EXECUTIONS_LOCK:
        executions = [
            execution
            for execution in _ACTIVE_TOOL_EXECUTIONS
            if not requested_id or execution.generation_id == requested_id
        ]
    for execution in executions:
        execution.cancel()
    return len(executions)


def abort_active_research(session_id: str) -> int:
    """Cancel only the active tool invocation for one Deep Research session."""

    requested_id = str(session_id or "").strip()
    if not requested_id:
        return 0
    with _ACTIVE_TOOL_EXECUTIONS_LOCK:
        executions = [
            execution
            for execution in _ACTIVE_TOOL_EXECUTIONS
            if execution.research_session_id == requested_id
        ]
    for execution in executions:
        execution.cancel()
    return len(executions)


def is_tool_execution_cancelled(result: Any) -> bool:
    """Return whether a streamed tool call ended because generation was stopped."""

    return result is TOOL_EXECUTION_CANCELLED


# Run async tool handlers on one persistent event loop.
class AsyncCallableRunner:

    # Initialize the shared runner state.
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    # Host the dedicated asyncio loop on a background thread.
    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    # Start the background loop thread when it is not already running.
    def ensure_started(self) -> None:
        if self._thread is not None and self._thread.is_alive() and self._loop is not None:
            return

        with self._lock:
            if self._thread is not None and self._thread.is_alive() and self._loop is not None:
                return

            self._ready.clear()
            self._thread = threading.Thread(
                target=self._thread_main,
                name="aslm-async-tool-runner",
                daemon=True,
            )
            self._thread.start()
            self._ready.wait()

    # Run one coroutine on the shared loop and wait for its result.
    def run(
        self,
        coro: Any,
        *,
        timeout: float | None = None,
        cancellation: ActiveToolExecution | None = None,
    ) -> Any:
        self.ensure_started()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if cancellation is not None:
            cancellation.set_cancel_callback(future.cancel)
        try:
            return future.result(timeout=timeout)
        finally:
            if cancellation is not None:
                cancellation.clear_cancel_callback()

    # Stop the background loop and join its thread.
    def close(self) -> None:
        loop = self._loop
        thread = self._thread
        self._loop = None
        self._thread = None
        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)


# Return the process-wide async callable runner singleton.
def _get_async_callable_runner() -> AsyncCallableRunner:
    global _ASYNC_CALLABLE_RUNNER
    with _ASYNC_CALLABLE_RUNNER_LOCK:
        if _ASYNC_CALLABLE_RUNNER is None:
            _ASYNC_CALLABLE_RUNNER = AsyncCallableRunner()
        return _ASYNC_CALLABLE_RUNNER


# Build a clean environment for one isolated Python venv.
def _venv_subprocess_env(python_path: Path) -> dict[str, str]:
    """Return subprocess environment aligned with the selected venv."""

    from Services import venv_manager

    env = venv_manager.get_subprocess_environment(python_path)
    apply_loopback_proxy_bypass(env)
    venv_path = python_path.parent.parent
    env["VIRTUAL_ENV"] = str(venv_path)
    env["PATH"] = str(python_path.parent) + os.pathsep + env.get("PATH", "")
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env.pop("PYTHONHOME", None)
    # Let the sandbox tool auto-launch Docker on demand when the model calls it.
    # Only the sandbox worker reads this; other tools ignore it.
    env["SANDBOX_AUTO_START_DOCKER"] = "1"
    # Hand the ASLM-assigned warm-browser daemon port to the tool venv (only web-search
    # reads it). Lets ASLM place the daemon in its chosen port range; the tool defaults to
    # the same value if the var is absent.
    try:
        env["ASLM_BROWSER_DAEMON_PORT"] = str(runtime_settings.get("browser-daemon-port", 20010))
    except Exception:  # noqa: BLE001 — never let settings access block a tool launch
        pass
    return env


# Keep one isolated tool worker process alive for stateful servers.
class ExternalWorkerSession:

    # Bind one worker process to a server entrypoint and venv Python.
    def __init__(self, server_file: Path, python_path: Path) -> None:
        self.server_file = server_file
        self.python_path = python_path
        self.process: subprocess.Popen[str] | None = None
        self.lock = threading.Lock()

    # Spawn the worker process when it is missing or has exited.
    def _start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return

        self.close()
        self.process = subprocess.Popen(
            [str(self.python_path), str(WORKER_FILE), "serve", str(self.server_file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(self.server_file.parent),
            env=_venv_subprocess_env(self.python_path),
            bufsize=1,
            **_tool_process_group_kwargs(),
        )

    # Send one JSON request to the worker and return its result envelope.
    def request(
        self,
        operation: str,
        payload: dict[str, Any] | None = None,
        *,
        timeout_s: float | None = None,
        cancellation: ActiveToolExecution | None = None,
    ) -> Any:
        """Send one request to the worker process and return its result."""

        with self.lock:
            request_payload = {
                "operation": operation,
                "payload": payload or {},
            }

            raw_response = ""
            last_error: Exception | None = None
            for attempt in range(2):
                if cancellation is not None and cancellation.cancelled.is_set():
                    raise ToolExecutionCancelled()
                self._start()
                # Cancellation may arrive while a new persistent worker is
                # spawning.  Do not let that worker survive long enough to
                # receive the request or be mistaken for a retry candidate.
                if cancellation is not None and cancellation.cancelled.is_set():
                    self.close()
                    raise ToolExecutionCancelled()
                assert self.process is not None
                assert self.process.stdin is not None
                assert self.process.stdout is not None

                try:
                    self.process.stdin.write(json.dumps(request_payload, ensure_ascii=False) + "\n")
                    self.process.stdin.flush()
                    raw_response = self._read_response_line(timeout_s or _worker_timeout_seconds(operation, payload))
                except TimeoutError as exc:
                    last_error = exc
                    raw_response = ""
                    self.close()
                    break
                except (BrokenPipeError, OSError) as exc:
                    last_error = exc
                    raw_response = ""

                if raw_response:
                    break

                self.close()
                if cancellation is not None and cancellation.cancelled.is_set():
                    raise ToolExecutionCancelled()
                if attempt == 0:
                    logger.warning("Tool worker stopped before response; restarting once for %s", self.server_file)
                    continue

            if not raw_response:
                self.close()
                if last_error is not None:
                    raise RuntimeError(f"Tool worker stopped for {self.server_file}: {last_error}")
                raise RuntimeError(f"Tool worker stopped for {self.server_file}.")

            try:
                envelope = json.loads(raw_response)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Tool worker returned invalid JSON: {raw_response[:500]}") from exc

            if not isinstance(envelope, dict):
                raise RuntimeError(f"Tool worker returned invalid envelope: {raw_response[:500]}")

            if not envelope.get("ok"):
                raise RuntimeError(str(envelope.get("error") or "Unknown tool worker error."))

            return envelope.get("result")

    # Read worker protocol lines until a final JSON envelope arrives.
    def _read_response_line(self, timeout_s: float) -> str:
        """Read worker protocol lines until a final envelope arrives."""

        process = self.process
        if process is None or process.stdout is None:
            return ""

        deadline = time.monotonic() + max(1.0, float(timeout_s))
        while True:
            lines: Queue[str] = Queue()

            # Read one stdout line in a background thread.
            def reader() -> None:
                try:
                    lines.put(process.stdout.readline())
                except Exception:
                    lines.put("")

            thread = threading.Thread(target=reader, name="aslm-tool-worker-readline", daemon=True)
            thread.start()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self.close()
                raise TimeoutError(f"Tool worker timed out after {timeout_s:.0f}s for {self.server_file}.")
            wait_s = min(WORKER_RESPONSE_IDLE_TIMEOUT_SECONDS, remaining)
            thread.join(timeout=wait_s)
            if thread.is_alive():
                self.close()
                raise TimeoutError(
                    f"Tool worker produced no heartbeat or result within {wait_s:.0f}s for {self.server_file}."
                )

            raw_line = lines.get()
            if not raw_line:
                return ""
            try:
                envelope = json.loads(raw_line)
            except json.JSONDecodeError:
                return raw_line
            if isinstance(envelope, dict) and envelope.get("event") == "heartbeat":
                continue
            return raw_line

    # Stop the worker process if it is running.
    def close(self) -> None:

        process = self.process
        self.process = None
        if process is None:
            return

        try:
            if process.poll() is None:
                _terminate_tool_process(process)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                try:
                    if stream is not None:
                        stream.close()
                except Exception:
                    pass


def _tool_process_group_kwargs() -> dict[str, Any]:
    """Start tool workers in a group that can be terminated with descendants."""

    if os.name == "nt":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_tool_process(process: subprocess.Popen[Any]) -> None:
    """Terminate a tool worker and any child processes it launched."""

    if process.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=3,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass

    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        if os.name != "nt":
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        else:
            process.kill()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=1)


class ToolEventSocketChannel:
    """Receive structured worker events without multiplexing them onto stdout."""

    def __init__(self, callback) -> None:
        self.callback = callback
        self.token = secrets.token_urlsafe(24)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(2)
        self.server.settimeout(0.2)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self._serve,
            name="aslm-tool-event-channel",
            daemon=True,
        )
        self.thread.start()

    @property
    def config(self) -> dict[str, Any]:
        return {
            "host": "127.0.0.1",
            "port": int(self.server.getsockname()[1]),
            "token": self.token,
        }

    def _serve(self) -> None:
        while not self.stop_event.is_set():
            try:
                connection, _address = self.server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with connection:
                stream = connection.makefile("r", encoding="utf-8", errors="replace")
                with stream:
                    for raw_line in stream:
                        if self.stop_event.is_set():
                            return
                        try:
                            envelope = json.loads(raw_line)
                        except json.JSONDecodeError:
                            continue
                        if not isinstance(envelope, dict) or envelope.get("token") != self.token:
                            continue
                        event = envelope.get("event")
                        if not isinstance(event, dict):
                            continue
                        try:
                            self.callback(event)
                        except Exception as exc:  # noqa: BLE001 - observers cannot break tools
                            logger.warning("Tool event callback failed: %s", exc)

    def close(self) -> None:
        self.stop_event.set()
        with contextlib.suppress(OSError):
            self.server.close()
        self.thread.join(timeout=1)


# Print one shared runtime event.
def _print_runtime_event(message: str) -> None:
    """Emit one console-visible runtime event."""

    print(f"[ASLM-Chat] {message}", flush=True)


# Return a wall-clock timeout for one worker request.
def _worker_timeout_seconds(operation: str, payload: dict[str, Any] | None = None) -> float:
    """Return a wall-clock timeout for one worker request."""

    timeout_s = WORKER_DEFAULT_TIMEOUT_SECONDS
    payload = payload if isinstance(payload, dict) else {}
    if operation == "call":
        arguments = payload.get("arguments")
        if isinstance(arguments, dict):
            try:
                timeout_s = float(arguments.get("timeout_s") or timeout_s)
            except (TypeError, ValueError):
                timeout_s = WORKER_DEFAULT_TIMEOUT_SECONDS
    return min(WORKER_MAX_TIMEOUT_SECONDS, max(1.0, timeout_s + WORKER_TIMEOUT_BUFFER_SECONDS))


# Check whether debug logging is enabled.
def _is_debug_logging_enabled() -> bool:
    """Return whether debug-or-higher MCP events should be printed."""

    return runtime_settings.is_console_debug_enabled()


# Check whether trace logging is enabled.
def _is_trace_logging_enabled() -> bool:
    """Return whether trace-level MCP events should be printed."""

    return runtime_settings.is_console_trace_enabled()


# Render one compact debug preview.
def _preview_jsonish(value: Any, limit: int = 240) -> str:
    """Return a compact one-line preview for arguments and results."""

    try:
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        rendered = str(value)

    rendered = re.sub(r"\s+", " ", str(rendered or "")).strip()
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[:max(0, limit - 3)].rstrip()}..."


# Summarize one tool result.
def _summarize_tool_result(result: Any) -> str:
    """Return a short textual summary of a tool result payload."""

    if isinstance(result, dict) and "_image_b64" in result:
        return f"image:{result.get('_mime_type', 'image')} path={result.get('_path', '') or '(inline)'}"
    if result is None:
        return "empty"
    if isinstance(result, str):
        return f"text chars={len(result)} preview={_preview_jsonish(result, limit=140)}"
    if isinstance(result, (dict, list, tuple)):
        try:
            size_hint = len(result)
        except TypeError:
            size_hint = "?"
        return f"{type(result).__name__} size={size_hint} preview={_preview_jsonish(result, limit=140)}"
    return f"{type(result).__name__} value={_preview_jsonish(result, limit=140)}"


# Summarize one tool context.
def _summarize_tool_context(context: dict[str, Any]) -> str:
    """Return a compact summary of the runtime context passed into a tool."""

    parts: list[str] = []
    for key in ("engine", "model_name", "chat_id", "tool_round_index", "tool_call_index"):
        value = context.get(key)
        if value not in {None, ""}:
            parts.append(f"{key}={value}")

    if _is_trace_logging_enabled():
        for key in ("server_file", "tools_dir"):
            value = context.get(key)
            if value not in {None, ""}:
                parts.append(f"{key}={value}")

    return ", ".join(parts) if parts else "none"


# Enforce per-response tool quotas and cooldowns.
# Return the canonical tool id used for per-response quotas.
def _quota_tool_id(tool_event: dict[str, Any] | None) -> str:
    """Return the canonical tool id used for per-response quotas."""

    if not isinstance(tool_event, dict):
        return ""
    tool_id = _slugify(str(tool_event.get("tool_id") or ""))
    alias = _slugify(str(tool_event.get("alias") or ""))
    for quota_tool_id in TOOL_CALL_QUOTAS:
        if tool_id == quota_tool_id or alias == quota_tool_id or alias.endswith(f"__{quota_tool_id}"):
            return quota_tool_id
    return ""


# Return the normalized web-search effort value from tool arguments.
def _search_effort(arguments: dict[str, Any] | None) -> str:
    """Return the normalized web-search effort value from tool arguments."""

    if not isinstance(arguments, dict):
        return ""

    value = arguments.get("effort")
    if value is None and isinstance(arguments.get("query"), dict):
        value = arguments["query"].get("effort")
    return str(value or "").strip().lower()


# Return the per-response quota for one tool call.
def _tool_quota_limit(
    quota_tool_id: str,
    arguments: dict[str, Any] | None = None,
    tool_event: dict[str, Any] | None = None,
) -> int:
    """Return the per-response quota for one tool call."""

    if bool((tool_event or {}).get("instant_mode")):
        return INSTANT_TOOL_CALL_QUOTAS.get(quota_tool_id, TOOL_CALL_QUOTAS[quota_tool_id])
    if quota_tool_id == "web_search" and _search_effort(arguments) == "high":
        return HIGH_EFFORT_WEB_SEARCH_QUOTA
    return TOOL_CALL_QUOTAS[quota_tool_id]


# Append complete model/search tool IO as a readable JSON array.
def _write_search_io_event(event: dict[str, Any]) -> None:
    """Append complete model/search tool IO as a readable JSON array."""

    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **dict(event or {}),
    }
    record = _without_duplicate_preview(record)
    try:
        _SEARCH_IO_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _SEARCH_IO_LOG_LOCK:
            entries: list[Any] = []
            if _SEARCH_IO_LOG_PATH.exists():
                try:
                    loaded = json.loads(_SEARCH_IO_LOG_PATH.read_text(encoding="utf-8") or "[]")
                    if isinstance(loaded, list):
                        entries = loaded
                except Exception:
                    entries = []
            entries.append(record)
            _SEARCH_IO_LOG_PATH.write_text(
                json.dumps(entries, ensure_ascii=False, default=str, indent=2),
                encoding="utf-8",
            )
    except Exception:
        logger.debug("Failed to write search IO JSON event.", exc_info=True)


# Remove preview fields from diagnostics when they duplicate snippet text.
def _without_duplicate_preview(value: Any) -> Any:
    """Remove preview fields from diagnostics when they exactly duplicate snippet."""

    if isinstance(value, list):
        return [_without_duplicate_preview(item) for item in value]
    if not isinstance(value, dict):
        return value

    cleaned = {
        key: _without_duplicate_preview(item)
        for key, item in value.items()
    }
    snippet = cleaned.get("snippet")
    preview = cleaned.get("preview")
    if isinstance(snippet, str) and isinstance(preview, str) and snippet == preview:
        cleaned.pop("preview", None)
    return cleaned


# Log exactly what search/read-page tools receive from and return to the model.
def log_search_tool_io(
    phase: str,
    tool_event: dict[str, Any] | None,
    arguments: Any = None,
    context: dict[str, Any] | None = None,
    result: Any = None,
    error: Any = None,
    elapsed_seconds: float | None = None,
) -> None:
    """Log exactly what search/read-page tools receive from and return to the model."""

    quota_tool_id = _quota_tool_id(tool_event)
    if not quota_tool_id:
        return
    _write_search_io_event(
        {
            "layer": "api_tool_bridge",
            "phase": phase,
            "tool_id": quota_tool_id,
            "tool_event": tool_event or {},
            "context": context or {},
            "arguments": arguments,
            "result": result,
            "error": str(error) if error is not None else None,
            "elapsed_seconds": elapsed_seconds,
        }
    )


# Increment one quota counter and return an error when the call is over limit.
def consume_tool_quota(
    tool_event: dict[str, Any] | None,
    counters: dict[str, int],
    arguments: dict[str, Any] | None = None,
) -> str | None:
    """Increment one quota counter and return an error message if the call is over limit."""

    quota_tool_id = _quota_tool_id(tool_event)
    if not quota_tool_id:
        return None

    limit = _tool_quota_limit(quota_tool_id, arguments, tool_event)
    current = int(counters.get(quota_tool_id, 0) or 0)
    if current >= limit:
        display_name = str((tool_event or {}).get("tool_name") or quota_tool_id).strip() or quota_tool_id
        if quota_tool_id == "web_search" and _search_effort(arguments) == "high":
            return (
                f"Tool quota exceeded: {display_name} high mode is unavailable "
                "for the rest of this assistant response; use medium or low."
            )
        return (
            f"Tool quota exceeded: {display_name} can be used at most {limit} times "
            "in one assistant response. Stop calling this tool and answer with the evidence already collected."
        )

    counters[quota_tool_id] = current + 1
    return None


# Return whether a tool result is a guardrail message, not fresh evidence.
def is_blocking_tool_result(result: Any) -> bool:
    """Return whether a tool result is a guardrail/block message, not fresh evidence."""

    if is_tool_execution_cancelled(result):
        return True
    if isinstance(result, dict):
        text = str(
            result.get("model_context")
            or result.get("content")
            or result.get("error")
            or ""
        ).strip()
    else:
        text = str(result or "").strip()
    return text.startswith(
        (
            "INVALID_SEARCH_PLAN:",
            "PARALLEL_SEARCH_BATCH_REJECTED:",
            "Duplicate tool call blocked:",
            "Duplicate web_search blocked:",
            "Duplicate read_page blocked:",
            "Tool quota exceeded:",
            "TOOL_EXECUTION_CANCELLED:",
        )
    )


# Return the instruction used when the tool loop must stop retrying tools.
def forced_final_prompt_after_tool_blocks() -> str:
    """Return the instruction used when the tool loop must stop retrying tools."""

    return TOOL_BLOCK_FINAL_PROMPT


def forced_final_prompt_after_tool_cancellation() -> str:
    """Return the instruction for the immediate tool-free round after a user stop."""

    return TOOL_CANCEL_FINAL_PROMPT


# How many times the tool loop re-prompts a model that ignored user-required tools.
MAX_REQUIRED_TOOL_NUDGES = 2


# Read an optional per-session tool-loop limit without exceeding the adapter cap.
def tool_round_limit_from_context(
    tool_context: dict[str, Any] | None,
    adapter_limit: int,
) -> int:
    """Return a bounded number of model/tool rounds for this generation."""

    raw_limit = tool_context.get("max_tool_rounds") if isinstance(tool_context, dict) else None
    try:
        requested_limit = int(raw_limit)
    except (TypeError, ValueError):
        requested_limit = adapter_limit
    return min(max(1, requested_limit), max(1, int(adapter_limit)))


# Bound one assistant turn's parallel calls before its transcript is canonicalized.
def limit_tool_calls_from_context(
    tool_context: dict[str, Any] | None,
    tool_calls: list[dict[str, Any]],
    tool_lookup: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Apply optional global and per-server parallel call limits."""

    raw_limit = tool_context.get("max_parallel_tool_calls") if isinstance(tool_context, dict) else None
    try:
        global_limit = max(1, int(raw_limit)) if raw_limit not in (None, "") else None
    except (TypeError, ValueError):
        global_limit = None

    raw_server_limits = (
        tool_context.get("max_parallel_tool_calls_by_server")
        if isinstance(tool_context, dict)
        else None
    )
    server_limits: dict[str, int] = {}
    if isinstance(raw_server_limits, dict):
        for raw_server_id, raw_server_limit in raw_server_limits.items():
            try:
                server_limits[str(raw_server_id).strip()] = max(1, int(raw_server_limit))
            except (TypeError, ValueError):
                continue

    limited = tool_calls[:global_limit] if global_limit is not None else list(tool_calls)
    if not server_limits:
        return limited

    server_counts: dict[str, int] = {}
    output: list[dict[str, Any]] = []
    for tool_call in limited:
        alias = str(tool_call.get("name") or "")
        server_id = tool_server_id_for_alias(tool_lookup, alias)
        server_limit = server_limits.get(server_id)
        if server_limit is not None and server_counts.get(server_id, 0) >= server_limit:
            continue
        output.append(tool_call)
        if server_limit is not None:
            server_counts[server_id] = server_counts.get(server_id, 0) + 1
    return output


def _conversation_char_count(conversation: list[dict[str, Any]]) -> int:
    try:
        return len(json.dumps(conversation, ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        return sum(len(str(message)) for message in conversation)


def _message_contains_tool_call(message: dict[str, Any], provider_format: str) -> bool:
    if message.get("tool_calls"):
        return True
    if provider_format != "google":
        return False
    parts = message.get("parts") if isinstance(message.get("parts"), list) else []
    return any(isinstance(part, dict) and isinstance(part.get("function_call"), dict) for part in parts)


def maybe_compact_tool_conversation(
    tool_context: dict[str, Any] | None,
    conversation: list[dict[str, Any]],
    *,
    provider_format: str = "standard",
) -> list[dict[str, Any]]:
    """Compact completed tool history while retaining the newest provider-native tool pair."""

    if not isinstance(tool_context, dict):
        return conversation
    compactor = tool_context.get("conversation_compactor")
    if not callable(compactor):
        return conversation
    try:
        trigger_chars = max(1, int(tool_context.get("compression_trigger_chars") or 0))
    except (TypeError, ValueError):
        return conversation
    if trigger_chars <= 0 or _conversation_char_count(conversation) < trigger_chars:
        return conversation

    try:
        prefix_count = max(0, int(tool_context.get("compression_prefix_messages") or 0))
    except (TypeError, ValueError):
        prefix_count = 0
    prefix_count = min(prefix_count, len(conversation))
    while (
        prefix_count > 0
        and isinstance(conversation[prefix_count - 1], dict)
        and _message_contains_tool_call(conversation[prefix_count - 1], provider_format)
    ):
        prefix_count -= 1

    keep_from = len(conversation)
    for index in range(len(conversation) - 1, prefix_count - 1, -1):
        message = conversation[index]
        if isinstance(message, dict) and _message_contains_tool_call(message, provider_format):
            keep_from = index
            break
    if keep_from <= prefix_count:
        return conversation

    compacted_span = [dict(message) for message in conversation[prefix_count:keep_from]]
    if not compacted_span:
        return conversation
    try:
        summary = str(compactor(compacted_span, provider_format=provider_format) or "").strip()
    except Exception as exc:  # noqa: BLE001 - compression must not break the active tool loop
        logger.warning("Tool conversation compression failed: %s", exc)
        return conversation
    if not summary:
        return conversation

    memory_text = "[deep research memory]\n" + summary
    if provider_format == "google":
        memory_message = {"role": "user", "parts": [{"text": memory_text}]}
    else:
        memory_message = {"role": "user", "content": memory_text}
    return [
        *conversation[:prefix_count],
        memory_message,
        *conversation[keep_from:],
    ]


# Resolve the owning tool-server id for one public tool alias.
def tool_server_id_for_alias(tool_lookup: dict[str, dict[str, Any]] | None, alias: str) -> str:
    """Return the owning tool-server id for one public tool alias."""

    entry = (tool_lookup or {}).get(str(alias or "")) or {}
    server = entry.get("server") if isinstance(entry, dict) else None
    if not isinstance(server, dict):
        return ""
    return str(server.get("id") or "")


def available_tools_prompt(tool_lookup: dict[str, dict[str, Any]]) -> str:
    """Describe only tools actually exposed for this generation, including alias filters."""
    if not tool_lookup:
        return ""
    lines = [
        "Available tools for this request (use their supplied parameter schemas):",
        "For tasks beyond basic conversation, you MUST call the relevant tools before answering. "
        "Inspect this list before claiming a capability is unavailable. "
        "Do not call unrelated tools just to satisfy the requirement.",
    ]
    for alias, entry in tool_lookup.items():
        tool = entry.get("tool") or {}
        description = str(tool.get("description") or tool.get("name") or "").strip()
        summary = description.splitlines()[0][:200] if description else ""
        lines.append(f"- {alias}: {summary}")
    seen_servers: set[str] = set()
    for entry in tool_lookup.values():
        server = entry.get("server") or {}
        server_id = str(server.get("server_file") or server.get("id") or "")
        if server_id in seen_servers or not server.get("has_model_context"):
            continue
        seen_servers.add(server_id)
        try:
            if server.get("external"):
                # A fresh process reads current sandbox.env values, independent of discovery caches.
                context = _run_worker(server["server_file"], "model_context", {})
            else:
                provider = getattr(server.get("module"), "get_model_context", None)
                context = provider() if callable(provider) else ""
            if isinstance(context, str) and context.strip():
                lines.append(context.strip()[:8000])
        except Exception:
            logger.warning("Could not read model context for tool server %s", server.get("id"))
            lines.append("Tool environment details unavailable; inspect the environment before heavy work.")
    return "\n".join(lines)


def with_available_tools_prompt(
    conversation: list[dict[str, Any]], tool_lookup: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Add live capability context without mutating the caller's history."""
    prompt = available_tools_prompt(tool_lookup)
    if not prompt:
        return conversation
    messages = [dict(message) for message in conversation]
    # Merge into the existing system message for backends with one system slot.
    if messages and messages[0].get("role") == "system" and isinstance(messages[0].get("content"), str):
        messages[0]["content"] = messages[0]["content"] + "\n\n" + prompt
    else:
        messages.insert(0, {"role": "system", "content": prompt})
    return messages


# Read user-required tool servers from the generation context.
def required_tool_server_ids_from_context(
    tool_context: dict[str, Any] | None,
    tool_lookup: dict[str, dict[str, Any]] | None,
) -> list[str]:
    """Return required tool-server ids that actually expose tools in this loop."""

    raw_ids = tool_context.get("required_tool_server_ids") if isinstance(tool_context, dict) else None
    if isinstance(raw_ids, str):
        raw_ids = [raw_ids]
    if not isinstance(raw_ids, list):
        return []

    available_server_ids = {
        tool_server_id_for_alias(tool_lookup, alias)
        for alias in (tool_lookup or {})
    }
    required: list[str] = []
    for raw_id in raw_ids:
        server_id = str(raw_id or "").strip()
        if server_id and server_id not in required and server_id in available_server_ids:
            required.append(server_id)
    return required


# Build the corrective prompt for a model that answered without the required tools.
def required_tools_reminder_prompt(
    unmet_server_ids: list[str],
    tool_lookup: dict[str, dict[str, Any]] | None,
) -> str:
    """Return the supervisor prompt sent when required tools were not called."""

    aliases_by_server: dict[str, list[str]] = {}
    for alias in tool_lookup or {}:
        server_id = tool_server_id_for_alias(tool_lookup, alias)
        if server_id in unmet_server_ids:
            aliases_by_server.setdefault(server_id, []).append(alias)

    listing = "; ".join(
        f"'{server_id}' (tools: {', '.join(sorted(aliases_by_server.get(server_id, []))) or server_id})"
        for server_id in unmet_server_ids
    )
    return (
        "[command supervisor] The user explicitly requested these tool servers for this message, "
        f"but you have not called any of their tools yet: {listing}. Do not answer from memory. "
        "Call the most relevant of these tools now, then continue your answer using the results."
    )


# Return a stable representation for duplicate tool-call detection.
def _canonical_tool_arguments(value: Any) -> Any:
    """Return a stable representation for duplicate tool-call detection."""

    if isinstance(value, str):
        text = re.sub(r"\s+", " ", value).strip()
        if text and text[0] in "[{":
            try:
                return _canonical_tool_arguments(json.loads(text))
            except Exception:
                return text
        return text
    if isinstance(value, dict):
        return {
            str(key): _canonical_tool_arguments(value[key])
            for key in sorted(value.keys(), key=str)
        }
    if isinstance(value, list):
        return [_canonical_tool_arguments(item) for item in value]
    return value


# Return an error message when the same quota-controlled tool call repeats.
def consume_duplicate_tool_call(
    tool_event: dict[str, Any] | None,
    arguments: dict[str, Any] | None,
    seen_signatures: set[str],
) -> str | None:
    """Return an error message when the same quota-controlled tool call repeats."""

    quota_tool_id = _quota_tool_id(tool_event)
    if not quota_tool_id:
        return None

    canonical_args = _canonical_tool_arguments(arguments or {})
    try:
        args_key = json.dumps(canonical_args, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        args_key = str(canonical_args)
    signature = f"{quota_tool_id}:{args_key}"

    if signature in seen_signatures:
        display_name = str((tool_event or {}).get("tool_name") or quota_tool_id).strip() or quota_tool_id
        return (
            f"Duplicate tool call blocked: {display_name} was already called with the same arguments "
            "in this assistant response. Do not retry identical searches or page reads; use the existing result."
        )

    seen_signatures.add(signature)
    return None


# Extract normalized read_page URL arguments from one tool payload.
def _read_page_urls(arguments: dict[str, Any] | None) -> list[str]:
    if not isinstance(arguments, dict):
        return []
    raw_url = arguments.get("url", "")
    if isinstance(raw_url, str):
        text = raw_url.strip()
        if text and text[0] == "[":
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    raw_url = parsed
                else:
                    raw_url = text
            except Exception:
                raw_url = text
        else:
            raw_url = text
    if isinstance(raw_url, list):
        return [str(item).strip() for item in raw_url if str(item or "").strip()]
    return [str(raw_url).strip()] if str(raw_url or "").strip() else []


# Normalize one read_page URL for cooldown comparison.
def _normalize_read_page_url(url: str) -> str:
    text = re.sub(r"\s+", " ", str(url or "").strip())
    if not text:
        return ""
    try:
        parsed = urlsplit(text)
    except Exception:
        return text.lower()
    if not parsed.scheme or not parsed.netloc:
        return text.rstrip("/").lower()
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or ""
    if path != "/":
        path = path.rstrip("/")
    return urlunsplit((scheme, netloc, path, parsed.query, ""))


# Normalize one cooldown key payload for stable comparison.
def _normalize_cooldown_value(value: Any) -> Any:
    canonical = _canonical_tool_arguments(value)
    if isinstance(canonical, str):
        return re.sub(r"\s+", " ", canonical).strip().lower()
    if isinstance(canonical, dict):
        return {
            str(key): _normalize_cooldown_value(canonical[key])
            for key in sorted(canonical.keys(), key=str)
        }
    if isinstance(canonical, list):
        return [_normalize_cooldown_value(item) for item in canonical]
    return canonical


# Resolve the conversation boundary for cooldown bookkeeping.
def _tool_cooldown_scope(context: dict[str, Any] | None) -> str:
    chat_id = str((context or {}).get("chat_id") or "").strip().lower()
    return f"chat:{chat_id}" if chat_id else "chat:<unknown>"


# Build cooldown keys for one search or read_page tool invocation.
def _tool_cooldown_keys(
    tool_event: dict[str, Any] | None,
    arguments: dict[str, Any] | None,
    *,
    context: dict[str, Any] | None = None,
) -> list[tuple[str, str]]:
    tool_id = _quota_tool_id(tool_event)
    scope = _tool_cooldown_scope(context)
    if tool_id == "read_page":
        urls = [_normalize_read_page_url(url) for url in _read_page_urls(arguments)]
        return [
            (f"{scope}:{tool_id}:url:{url}", url)
            for url in dict.fromkeys(url for url in urls if url)
        ]
    if tool_id == "web_search":
        canonical = _normalize_cooldown_value(arguments or {})
        try:
            key = json.dumps(canonical, ensure_ascii=False, sort_keys=True, default=str)
        except TypeError:
            key = str(canonical)
        label = _preview_jsonish(canonical, limit=220)
        return [(f"{scope}:{tool_id}:args:{key}", label)] if key and key != "{}" else []
    return []


# Format the duplicate-tool cooldown message shown to the model.
def _tool_cooldown_message(tool_id: str, entries: list[tuple[str, int]]) -> str:
    display = "web_search" if tool_id == "web_search" else "read_page"
    thing = "query" if tool_id == "web_search" else "URL"
    lines = [
        f"Duplicate {display} blocked: this {thing} was already used in the last 30 seconds.",
        f"Do not call {display} for it again right now; use the content already returned by the previous tool result.",
        "Repeated calls waste tool calls and add no new evidence.",
        "",
        "Cooldown entries:",
    ]
    for label, remaining in entries:
        lines.append(f"- {label} ({remaining}s remaining)")
    return "\n".join(lines)


# Block repeated search/read-page calls within a short cooldown window.
def consume_tool_cooldown(
    tool_event: dict[str, Any] | None,
    arguments: dict[str, Any] | None,
    *,
    context: dict[str, Any] | None = None,
) -> str | None:
    """Block repeated search/read-page calls within a short cooldown window."""

    tool_id = _quota_tool_id(tool_event)
    entries = _tool_cooldown_keys(tool_event, arguments, context=context)
    if not tool_id or not entries:
        return None

    now = time.monotonic()
    with _TOOL_COOLDOWN_LOCK:
        expired = [key for key, expires_at in _TOOL_COOLDOWNS.items() if expires_at <= now]
        for key in expired:
            _TOOL_COOLDOWNS.pop(key, None)

        duplicate_entries: list[tuple[str, int]] = []
        for key, label in entries:
            expires_at = _TOOL_COOLDOWNS.get(key)
            if expires_at and expires_at > now:
                duplicate_entries.append((label, max(1, int(round(expires_at - now)))))

    if len(duplicate_entries) == len(entries):
        return _tool_cooldown_message(tool_id, duplicate_entries)
    return None


# Mark search/read-page calls as recently used.
def remember_tool_cooldown(
    tool_event: dict[str, Any] | None,
    arguments: dict[str, Any] | None,
    *,
    context: dict[str, Any] | None = None,
) -> None:
    """Mark search/read-page calls as recently used."""

    entries = _tool_cooldown_keys(tool_event, arguments, context=context)
    if not entries:
        return
    expires_at = time.monotonic() + _TOOL_COOLDOWN_SECONDS
    with _TOOL_COOLDOWN_LOCK:
        for key, _label in entries:
            _TOOL_COOLDOWNS[key] = expires_at


def clear_tool_runtime_scope(context: dict[str, Any] | None) -> None:
    """Remove in-memory tool bookkeeping owned by one isolated conversation."""

    scope_prefix = f"{_tool_cooldown_scope(context)}:"
    with _TOOL_COOLDOWN_LOCK:
        scoped_keys = [key for key in _TOOL_COOLDOWNS if key.startswith(scope_prefix)]
        for key in scoped_keys:
            _TOOL_COOLDOWNS.pop(key, None)


# Delegate read_page cooldown checks to the shared tool cooldown helper.
def consume_read_page_cooldown(
    tool_event: dict[str, Any] | None,
    arguments: dict[str, Any] | None,
    *,
    context: dict[str, Any] | None = None,
) -> str | None:
    return consume_tool_cooldown(tool_event, arguments, context=context)


# Delegate read_page cooldown bookkeeping to the shared tool cooldown helper.
def remember_read_page_cooldown(
    tool_event: dict[str, Any] | None,
    arguments: dict[str, Any] | None,
    *,
    context: dict[str, Any] | None = None,
) -> None:
    remember_tool_cooldown(tool_event, arguments, context=context)


# Manage the server discovery cache.
# Clear the cached registry.
def reset_cache() -> None:
    """Clear cached discovery so local edits are picked up immediately."""

    global _SERVER_CACHE_SIGNATURE, _SERVER_CACHE

    _SERVER_CACHE_SIGNATURE = None
    _SERVER_CACHE = {}
    _SCOPED_REGISTRY_CACHE.clear()
    user_mcp_client.shutdown_all()
    close_external_workers()


# Normalize one optional module directory path.
def _normalize_module_dir(module_dir: str | Path | None) -> Path:
    if module_dir is None:
        return MODULE_ROOT
    return Path(module_dir).resolve()


# Build a stable cache key for one tools/module directory pair.
def _registry_scope_key(tools_dir: Path, module_dir: Path) -> str:
    return f"{tools_dir.resolve()}|{module_dir.resolve()}"


# Resolve the tools directory for one registry scope.
def _resolve_tools_dir(tools_dir: str | Path | None = None) -> Path:
    if tools_dir is None:
        return TOOLS_DIR
    return Path(tools_dir).resolve()


# Yield one server's source files.
def _iter_server_source_files(server_dir: Path):
    """Yield relevant source files for one local MCP server."""

    for path in sorted(server_dir.rglob("*"), key=lambda item: str(item).casefold()):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in {".py", ".json"}:
            continue
        yield path


# Build the server cache signature.
def _server_signature_for(tools_dir: Path, module_dir: Path) -> tuple[tuple[str, int], ...]:
    """Build a stable signature for every local MCP server source file."""

    if not tools_dir.exists():
        return ()

    entries: list[tuple[str, int]] = []
    for child in sorted(tools_dir.iterdir(), key=lambda item: item.name.casefold()):
        server_file = child / SERVER_FILENAME
        if not child.is_dir() or not server_file.is_file():
            continue

        for source_file in _iter_server_source_files(child):
            try:
                stat = source_file.stat()
            except OSError:
                continue

            entries.append((str(source_file), stat.st_mtime_ns))

    mcp_sig = mcp_json.mcp_json_signature_for(module_dir)
    if mcp_sig is not None:
        entries.append(mcp_sig)

    return tuple(entries)


# Build the server cache signature.
def _server_signature() -> tuple[tuple[str, int], ...]:
    return _server_signature_for(TOOLS_DIR, MODULE_ROOT)


# Yield server entrypoint files.
def _iter_server_files_for(tools_dir: Path):
    """Yield top-level MCP server entrypoints from one Tools directory."""

    if not tools_dir.exists():
        return

    for child in sorted(tools_dir.iterdir(), key=lambda item: item.name.casefold()):
        server_file = child / SERVER_FILENAME
        if child.is_dir() and server_file.is_file():
            yield server_file


# Yield server entrypoint files.
def _iter_server_files():
    yield from _iter_server_files_for(TOOLS_DIR)


# Load and normalize server definitions.
# Normalize one server id.
def _slugify(value: str) -> str:
    """Normalize folder and public identifiers into a stable slug."""

    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized or "tool"

# Purge modules from one server root.
def _purge_modules_under(server_root: Path) -> None:
    """Drop previously imported modules loaded from one tool directory."""

    resolved_root = server_root.resolve()
    stale_module_names: list[str] = []

    for module_name, module in list(sys.modules.items()):
        module_file = getattr(module, "__file__", None)
        if not module_file:
            continue

        try:
            module_path = Path(module_file).resolve()
        except OSError:
            continue

        if module_path.is_relative_to(resolved_root):
            stale_module_names.append(module_name)

    for module_name in stale_module_names:
        sys.modules.pop(module_name, None)


# Load one server module.
def _load_module(server_file: Path) -> ModuleType:
    """Load one ``mcp-server.py`` file into an isolated Python module."""

    _purge_modules_under(server_file.parent)

    module_name = f"aslm_chat_mcp_server_{_slugify(server_file.parent.name)}"
    spec = importlib.util.spec_from_file_location(module_name, server_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MCP server module from {server_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Return the configured worker Python for one tool server when available.
def _get_worker_python(server_file: Path) -> Path | None:
    """Return the isolated Python executable assigned to one tool server."""

    try:
        from Services import venv_manager

        return venv_manager.get_tool_python(server_file.parent.name)
    except Exception as exc:
        logger.warning("Could not resolve tool venv for %s: %s", server_file, exc)
        return None


# Close all persistent external workers.
def close_external_workers() -> None:
    """Stop persistent external tool workers owned by this process."""

    with _WORKER_SESSION_LOCK:
        sessions = list(_WORKER_SESSIONS.values())
        _WORKER_SESSIONS.clear()

    for session in sessions:
        session.close()

    global _ASYNC_CALLABLE_RUNNER
    with _ASYNC_CALLABLE_RUNNER_LOCK:
        runner = _ASYNC_CALLABLE_RUNNER
        _ASYNC_CALLABLE_RUNNER = None
    if runner is not None:
        runner.close()


atexit.register(close_external_workers)


# Return one persistent external worker for a server file.
def _get_worker_session(server_file: Path) -> ExternalWorkerSession:
    """Return a long-lived worker session for one external tool server."""

    python_path = _get_worker_python(server_file)
    if python_path is None:
        raise RuntimeError(f"No isolated Python environment is available for {server_file.parent.name}.")

    session_key = str(server_file.resolve())
    with _WORKER_SESSION_LOCK:
        session = _WORKER_SESSIONS.get(session_key)
        if session is None or session.python_path != python_path:
            if session is not None:
                session.close()
            session = ExternalWorkerSession(server_file, python_path)
            _WORKER_SESSIONS[session_key] = session
        return session


# Execute one isolated tool worker operation.
def _run_worker(
    server_file: Path,
    operation: str,
    payload: dict[str, Any] | None = None,
    *,
    persistent: bool = False,
    cancellation: ActiveToolExecution | None = None,
) -> Any:
    """Run a tool worker operation and return its result payload."""

    if persistent:
        session = _get_worker_session(server_file)
        if cancellation is not None:
            cancellation.set_cancel_callback(session.close)
        try:
            return session.request(
                operation,
                payload,
                timeout_s=_worker_timeout_seconds(operation, payload),
                cancellation=cancellation,
            )
        finally:
            if cancellation is not None:
                cancellation.clear_cancel_callback()

    python_path = _get_worker_python(server_file)
    if python_path is None:
        raise RuntimeError(f"No isolated Python environment is available for {server_file.parent.name}.")

    request_payload = json.dumps(payload or {}, ensure_ascii=False)
    process = subprocess.Popen(
        [str(python_path), str(WORKER_FILE), operation, str(server_file)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(server_file.parent),
        env=_venv_subprocess_env(python_path),
        **_tool_process_group_kwargs(),
    )
    if cancellation is not None:
        cancellation.set_cancel_callback(lambda: _terminate_tool_process(process))
    try:
        stdout, stderr = process.communicate(
            input=request_payload,
            timeout=_worker_timeout_seconds(operation, payload),
        )
    except subprocess.TimeoutExpired:
        _terminate_tool_process(process)
        raise
    finally:
        if cancellation is not None:
            cancellation.clear_cancel_callback()

    stdout = (stdout or "").strip()
    stderr = (stderr or "").strip()
    if not stdout:
        raise RuntimeError(stderr or f"Tool worker returned no output for {server_file}.")

    envelope = None
    for line in reversed(stdout.splitlines()):
        try:
            envelope = json.loads(line)
            break
        except json.JSONDecodeError:
            continue
    if not isinstance(envelope, dict):
        raise RuntimeError(f"Tool worker returned invalid JSON: {stdout[:500]}")

    if not envelope.get("ok"):
        error = str(envelope.get("error") or stderr or "Unknown tool worker error.")
        raise RuntimeError(error)

    return envelope.get("result")


# Load one server definition through its isolated worker.
def _load_external_server(server_file: Path) -> dict[str, Any]:
    """Load one server definition without importing it into the Django process."""

    description = _run_worker(server_file, "describe", {})
    if not isinstance(description, dict):
        raise ValueError("Tool worker describe response must be a dictionary.")

    server_id = _slugify(str(description.get("id") or server_file.parent.name))
    tools = description.get("tools")
    if not isinstance(tools, list) or not tools:
        raise ValueError("Tool worker did not return any tools.")

    return {
        "id": server_id,
        "name": str(description.get("name") or server_file.parent.name).strip() or server_file.parent.name,
        "description": str(description.get("description") or "").strip(),
        "requires_docker": bool(description.get("requires_docker")),
        "fresh_process_per_call": bool(description.get("fresh_process_per_call")),
        "has_model_context": bool(description.get("has_model_context")),
        "tools": tools,
        "module": None,
        "supports": None,
        "server_callable": None,
        "tool_handlers": {},
        "server_file": server_file,
        "external": True,
    }


# Normalize one tool schema.
def _normalize_schema(schema: Any) -> dict[str, Any]:
    """Return a JSON-schema-like mapping suitable for tool payloads."""

    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(schema)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    if not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}

    return normalized

# Resolve the server dispatcher.
def _resolve_server_callable(module: ModuleType):
    """Return the generic server dispatcher when one is exported."""

    for attr_name in SERVER_DISPATCHER_NAMES:
        candidate = getattr(module, attr_name, None)
        if callable(candidate):
            return candidate

    return None

# Collect explicit tool handlers.
def _normalize_tool_handlers(module: ModuleType) -> dict[str, Any]:
    """Return explicit per-tool handlers exported by a server module."""

    raw_handlers: Any = None
    for attr_name in TOOL_HANDLER_NAMES:
        raw_handlers = getattr(module, attr_name, None)
        if raw_handlers is not None:
            break

    if not isinstance(raw_handlers, dict):
        return {}

    normalized_handlers: dict[str, Any] = {}
    for raw_key, raw_value in raw_handlers.items():
        if callable(raw_value):
            normalized_handlers[_slugify(str(raw_key or ""))] = raw_value

    return normalized_handlers


def _normalize_tool_preparers(module: ModuleType) -> dict[str, Any]:
    """Collect optional argument-only preflight handlers from an in-process server."""

    raw_preparers: Any = None
    for attr_name in TOOL_PREPARER_NAMES:
        raw_preparers = getattr(module, attr_name, None)
        if raw_preparers is not None:
            break
    if not isinstance(raw_preparers, dict):
        return {}
    return {
        _slugify(str(raw_key or "")): raw_value
        for raw_key, raw_value in raw_preparers.items()
        if callable(raw_value)
    }

# Validate the tool list.
def _normalize_server_tools(raw_tools: Any, server_id: str) -> list[dict[str, Any]]:
    """Validate and normalize tool definitions exposed by one server."""

    if not isinstance(raw_tools, list) or not raw_tools:
        raise ValueError("MCP server module must expose a non-empty TOOLS list")

    normalized_tools: list[dict[str, Any]] = []
    seen_tool_ids: set[str] = set()

    for index, raw_tool in enumerate(raw_tools, start=1):
        if not isinstance(raw_tool, dict):
            raise ValueError("Each MCP server tool definition must be a dictionary")

        tool_id = _slugify(str(raw_tool.get("id") or f"tool_{index}"))
        if tool_id in seen_tool_ids:
            raise ValueError(f"Duplicate tool id '{tool_id}' in server '{server_id}'")

        seen_tool_ids.add(tool_id)

        name = str(raw_tool.get("name") or tool_id).strip() or tool_id
        description = str(raw_tool.get("description") or "").strip()
        parameters = _normalize_schema(raw_tool.get("parameters"))

        normalized_tools.append(
            {
                "id": tool_id,
                "alias": f"{server_id}__{tool_id}",
                "name": name,
                "description": description,
                "parameters": parameters,
                "prepares_arguments": bool(raw_tool.get("prepares_arguments")),
            }
        )

    return normalized_tools


# Append servers from MCP/mcp.json to the registry payload.
def _merge_user_mcp_servers(
    discovered: dict[str, dict[str, Any]],
    *,
    module_dir: Path | None = None,
) -> None:
    """Append servers from ``MCP/mcp.json`` to the registry payload."""

    try:
        resolved_module_dir = _normalize_module_dir(module_dir)
        if resolved_module_dir == MODULE_ROOT:
            mcp_json.ensure_default_mcp_json()
        reserved = set(discovered.keys())
        entries = (
            mcp_json.iter_user_mcp_entries(reserved)
            if resolved_module_dir == MODULE_ROOT
            else mcp_json.iter_user_mcp_entries_for(resolved_module_dir, reserved)
        )
        for entry in entries:
            tools, err = user_mcp_client.fetch_tool_definitions(entry)
            description = f"User MCP server ({entry.transport})"
            if err:
                description = f"{description}. {err}"
            placeholder = resolved_module_dir / "MCP" / f".user_mcp_{entry.server_id}"
            discovered[entry.server_id] = {
                "id": entry.server_id,
                "name": entry.display_name,
                "description": description[:500],
                "tools": tools,
                "user_mcp": True,
                "user_mcp_entry": entry,
                "module": None,
                "supports": None,
                "server_callable": None,
                "tool_handlers": {},
                "server_file": placeholder,
                "external": False,
                "source_tools_dir": str(_resolve_tools_dir(resolved_module_dir / "Tools")),
                "source_module_dir": str(resolved_module_dir),
            }
    except Exception as exc:
        logger.warning("Failed to merge user MCP servers: %s", exc)


# Build one server definition.
def _extract_server_definition(module: ModuleType, folder_name: str, server_file: Path) -> dict[str, Any]:
    """Validate a local MCP module and normalize its public metadata."""

    raw_server: Any = {}
    # Support both current and legacy metadata export names.
    for attr_name in SERVER_METADATA_NAMES:
        raw_server = getattr(module, attr_name, None)
        if raw_server is not None:
            break

    if raw_server is None:
        raw_server = {}
    if not isinstance(raw_server, dict):
        raise ValueError("MCP server module must expose an MCP_SERVER dictionary")

    raw_tools = getattr(module, "TOOLS", None)
    if raw_tools is None:
        legacy_tool = getattr(module, "TOOL", None)
        if isinstance(legacy_tool, dict):
            raw_tools = [legacy_tool]

    server_id = _slugify(str(raw_server.get("id") or folder_name))
    server_name = str(raw_server.get("name") or folder_name).strip() or folder_name
    description = str(raw_server.get("description") or "").strip()
    supports_fn = getattr(module, "supports", None)
    tool_handlers = _normalize_tool_handlers(module)
    tool_preparers = _normalize_tool_preparers(module)
    server_callable = _resolve_server_callable(module)
    tools = _normalize_server_tools(raw_tools, server_id)
    for tool_definition in tools:
        if tool_definition["id"] in tool_preparers:
            tool_definition["prepares_arguments"] = True

    # A server is valid only if it exposes either dedicated handlers per tool
    # or one generic dispatcher that can receive tool invocations.
    if not tool_handlers and server_callable is None:
        raise ValueError(
            "MCP server module must expose TOOL_HANDLERS or a generic call_tool/tool dispatcher"
        )

    return {
        "id": server_id,
        "name": server_name,
        "description": description,
        "requires_docker": bool(raw_server.get("requires_docker")),
        "has_model_context": callable(getattr(module, "get_model_context", None)),
        "tools": tools,
        "module": module,
        "supports": supports_fn if callable(supports_fn) else None,
        "server_callable": server_callable,
        "tool_handlers": tool_handlers,
        "tool_preparers": tool_preparers,
        "server_file": server_file,
        "external": False,
        "source_tools_dir": str(server_file.parent.parent),
        "source_module_dir": str(server_file.parent.parent.parent),
    }


# Refresh the server registry.
def _ensure_registry_loaded_for(
    tools_dir: Path | None = None,
    module_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Discover and cache valid local MCP-style server modules for one scope."""

    resolved_tools_dir = _resolve_tools_dir(tools_dir)
    resolved_module_dir = _normalize_module_dir(module_dir)
    scope_key = _registry_scope_key(resolved_tools_dir, resolved_module_dir)
    signature = _server_signature_for(resolved_tools_dir, resolved_module_dir)

    cached = _SCOPED_REGISTRY_CACHE.get(scope_key)
    if cached is not None and cached[0] == signature:
        return cached[1]
    if cached is not None and cached[0] != signature:
        close_external_workers()

    if resolved_tools_dir == TOOLS_DIR and resolved_module_dir == MODULE_ROOT:
        global _SERVER_CACHE_SIGNATURE, _SERVER_CACHE
        if signature == _SERVER_CACHE_SIGNATURE:
            _SCOPED_REGISTRY_CACHE[scope_key] = (signature, _SERVER_CACHE)
            return _SERVER_CACHE

    user_mcp_client.shutdown_all()

    discovered: dict[str, dict[str, Any]] = {}
    for server_file in _iter_server_files_for(resolved_tools_dir):
        folder_name = server_file.parent.name

        try:
            if _get_worker_python(server_file) is not None:
                server_definition = _load_external_server(server_file)
            else:
                module = _load_module(server_file)
                server_definition = _extract_server_definition(module, folder_name, server_file)
            server_definition["source_tools_dir"] = str(resolved_tools_dir)
            server_definition["source_module_dir"] = str(resolved_module_dir)
            discovered[server_definition["id"]] = server_definition
        except Exception as exc:
            logger.warning("Skipping invalid MCP server module %s: %s", server_file, exc)

    _merge_user_mcp_servers(discovered, module_dir=resolved_module_dir)

    _SCOPED_REGISTRY_CACHE[scope_key] = (signature, discovered)
    if resolved_tools_dir == TOOLS_DIR and resolved_module_dir == MODULE_ROOT:
        _SERVER_CACHE_SIGNATURE = signature
        _SERVER_CACHE = discovered
    return discovered


# Refresh the server registry.
def _ensure_registry_loaded() -> dict[str, dict[str, Any]]:
    """Discover and cache valid local MCP-style server modules."""

    return _ensure_registry_loaded_for(TOOLS_DIR, MODULE_ROOT)

# Check whether one server is supported.
def _server_is_supported(
    server_definition: dict[str, Any],
    engine: str | None,
    model_name: str | None,
) -> bool:
    """Return whether a server supports the current engine and model."""

    if server_definition.get("user_mcp"):
        return True

    if server_definition.get("external"):
        try:
            return bool(_run_worker(
                Path(server_definition["server_file"]),
                "supports",
                {"engine": engine, "model_name": model_name},
            ))
        except Exception as exc:
            logger.warning("Server %s support check failed: %s", server_definition["id"], exc)
            return False

    supports_fn = server_definition.get("supports")
    if not callable(supports_fn):
        return True

    try:
        return bool(supports_fn(engine=engine, model_name=model_name))
    except TypeError:
        # Older handlers may still accept only positional arguments.
        try:
            return bool(supports_fn(engine, model_name))
        except Exception as exc:
            logger.warning("Server %s support check failed: %s", server_definition["id"], exc)
            return False
    except Exception as exc:
        logger.warning("Server %s support check failed: %s", server_definition["id"], exc)
        return False

# List matching servers.
def list_servers(
    engine: str | None = None,
    model_name: str | None = None,
    *,
    tools_dir: str | Path | None = None,
    module_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Return discovered servers that support the current engine and model."""

    registry = _ensure_registry_loaded_for(tools_dir, module_dir)
    return [
        _serialize_server(server_definition)
        for server_definition in registry.values()
        if server_definition.get("id") not in NON_MODEL_TOOL_SERVER_IDS
        and _server_is_supported(server_definition, engine, model_name)
    ]

# Get one matching server.
def get_server(
    server_id: str | None,
    engine: str | None = None,
    model_name: str | None = None,
    *,
    tools_dir: str | Path | None = None,
    module_dir: str | Path | None = None,
    tool_source_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Return one discovered server when it is available in the current context."""

    normalized_id = _slugify(str(server_id or ""))
    if not normalized_id or normalized_id in NON_MODEL_TOOL_SERVER_IDS:
        return None

    source = (tool_source_map or {}).get(normalized_id) or {}
    scoped_tools_dir = source.get("tools_dir") or tools_dir
    scoped_module_dir = source.get("module_dir") or module_dir

    registry = _ensure_registry_loaded_for(scoped_tools_dir, scoped_module_dir)
    server_definition = registry.get(normalized_id)
    if not server_definition:
        if scoped_tools_dir is not None or scoped_module_dir is not None:
            registry = _ensure_registry_loaded()
            server_definition = registry.get(normalized_id)
        if not server_definition:
            return None
    if not _server_is_supported(server_definition, engine, model_name):
        return None

    return server_definition


# Serialize registry data.
# Serialize one tool.
def _serialize_tool(tool_definition: dict[str, Any]) -> dict[str, Any]:
    """Return the frontend-facing representation of one tool."""

    return {
        "id": tool_definition["id"],
        "name": tool_definition["name"],
        "description": tool_definition["description"],
    }

# Serialize one server.
def _serialize_server(server_definition: dict[str, Any]) -> dict[str, Any]:
    """Return the frontend-facing representation of one server."""

    tools = [_serialize_tool(tool_definition) for tool_definition in server_definition["tools"]]
    payload: dict[str, Any] = {
        "id": server_definition["id"],
        "name": server_definition["name"],
        "description": server_definition["description"],
        "tool_count": len(tools),
        "tools": tools,
    }
    if server_definition.get("requires_docker"):
        payload["requires_docker"] = True
    if server_definition.get("user_mcp"):
        payload["user_mcp"] = True
    return payload

# Build Ollama-compatible tool definitions.
def build_ollama_tools(
    server_ids: str | list[str] | None,
    engine: str | None = None,
    model_name: str | None = None,
    *,
    tool_source_map: dict[str, dict[str, Any]] | None = None,
    allowed_tool_aliases: list[str] | set[str] | tuple[str, ...] | None = None,
    instant_mode: bool = False,
    instant_search_batch_size: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return Ollama-compatible tool payloads for one or more selected servers."""

    if isinstance(server_ids, str):
        server_ids = [server_ids] if server_ids else []
    elif not server_ids:
        server_ids = []

    tools: list[dict[str, Any]] = []
    tool_lookup: dict[str, dict[str, Any]] = {}
    allowed_aliases = {
        str(alias).strip()
        for alias in (allowed_tool_aliases or [])
        if str(alias).strip()
    }

    for server_id in server_ids:
        server_definition = get_server(
            server_id,
            engine=engine,
            model_name=model_name,
            tool_source_map=tool_source_map,
        )
        if not server_definition:
            continue

        for tool_definition in server_definition["tools"]:
            alias = tool_definition["alias"]
            if allowed_aliases and alias not in allowed_aliases:
                continue
            # Tool aliases are global in the conversation, so skip duplicates
            # when multiple selected servers resolve to the same public alias.
            if alias in tool_lookup:
                continue
            public_tool_definition = tool_definition
            if instant_mode and server_definition.get("id") == "web_search":
                public_tool_definition = copy.deepcopy(tool_definition)
                public_tool_definition["instant_mode"] = True
                if public_tool_definition.get("id") == "web_search":
                    required_batch_size = (
                        INSTANT_SEARCH_BATCH_LIMIT
                        if instant_search_batch_size == INSTANT_SEARCH_BATCH_LIMIT
                        else None
                    )
                    description, parameters = _instant_web_search_contract(required_batch_size)
                    public_tool_definition["description"] = description
                    public_tool_definition["parameters"] = parameters
                    if required_batch_size is not None:
                        public_tool_definition["instant_search_batch_size"] = required_batch_size
                elif public_tool_definition.get("id") == "read_page":
                    parameters = copy.deepcopy(public_tool_definition.get("parameters") or {})
                    url_schema = (parameters.get("properties") or {}).get("url")
                    if isinstance(url_schema, dict):
                        for option in url_schema.get("oneOf") or []:
                            if isinstance(option, dict) and option.get("type") == "array":
                                option["maxItems"] = INSTANT_SEARCH_BATCH_LIMIT
                    public_tool_definition["parameters"] = parameters
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": alias,
                        "description": public_tool_definition["description"] or public_tool_definition["name"],
                        "parameters": public_tool_definition["parameters"],
                    },
                }
            )
            tool_lookup[alias] = {"server": server_definition, "tool": public_tool_definition}

    if tools and _is_debug_logging_enabled():
        selected_servers = []
        for server_id in server_ids:
            server_definition = get_server(
                server_id,
                engine=engine,
                model_name=model_name,
                tool_source_map=tool_source_map,
            )
            if server_definition:
                selected_servers.append(server_definition["id"])

        _print_runtime_event(
            "Tool registry prepared: "
            f"engine={engine or 'unknown'}, "
            f"model={model_name or '(auto)'}, "
            f"servers={selected_servers or list(server_ids)}, "
            f"tools={len(tools)}"
        )
        if _is_trace_logging_enabled():
            aliases = ", ".join(sorted(tool_lookup.keys(), key=str.casefold))
            _print_runtime_event(f"Tool aliases: {aliases}")

    return tools, tool_lookup


# Execute tool callables.
# Run one async callable.
async def _run_async_callable(callable_fn, *args: Any) -> Any:
    """Execute an async callable with the provided arguments."""

    return await callable_fn(*args)

# Run one sync callable.
def _run_sync_callable(callable_fn, *args: Any) -> Any:
    """Execute a synchronous callable with the provided arguments."""

    return callable_fn(*args)

# Execute one callable safely.
def _execute_callable(
    callable_fn,
    *args: Any,
    cancellation: ActiveToolExecution | None = None,
) -> Any:
    """Execute sync and async callables behind one shared helper."""

    if inspect.iscoroutinefunction(callable_fn):
        return _get_async_callable_runner().run(
            _run_async_callable(callable_fn, *args),
            cancellation=cancellation,
        )

    return _run_sync_callable(callable_fn, *args)

# Dispatch one generic server handler.
def _dispatch_server_callable(
    callable_fn,
    tool_id: str,
    arguments: dict[str, Any],
    context: dict[str, Any],
    cancellation: ActiveToolExecution | None = None,
) -> Any:
    """Call a generic server dispatcher with a tolerant signature strategy."""

    parameter_names = list(inspect.signature(callable_fn).parameters)

    # Accept older dispatchers that still use positional signatures instead of
    # the newer ``(tool_id, arguments, context)`` convention.
    if len(parameter_names) <= 1:
        return _execute_callable(callable_fn, arguments, cancellation=cancellation)

    if len(parameter_names) == 2:
        first_name = parameter_names[0].lower()
        if first_name in {"tool_id", "tool", "name", "tool_name"}:
            return _execute_callable(callable_fn, tool_id, arguments, cancellation=cancellation)

        return _execute_callable(callable_fn, arguments, context, cancellation=cancellation)

    return _execute_callable(callable_fn, tool_id, arguments, context, cancellation=cancellation)


# Serialize tool results.
# Serialize one tool result.
def _serialize_tool_result(result: Any) -> str:
    """Convert a tool result into text suitable for a model tool message."""

    if result is None:
        return "Tool returned no content."

    if isinstance(result, str):
        return result

    shared_file = _extract_shared_file_payload(result)
    if shared_file is not None:
        filename = str(shared_file.get("filename") or shared_file.get("name") or shared_file.get("path") or "file")
        return str(shared_file.get("model_context") or f"Shared file ready for download: {filename}")

    if isinstance(result, dict) and isinstance(result.get("model_context"), str):
        return result["model_context"]

    if isinstance(result, (dict, list, tuple, int, float, bool)):
        return json.dumps(result, ensure_ascii=False, indent=2)

    return str(result)


# Normalize tool result payloads.
# Parse JSON tool payloads returned as plain strings.
def _coerce_tool_result_object(result: Any) -> Any:
    """Parse JSON tool payloads returned as plain strings."""

    if not isinstance(result, str):
        return result
    text = result.strip()
    if not text.startswith("{"):
        return result
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return result
    return parsed if isinstance(parsed, dict) else result


# Return a normalized shared-file payload from direct or sandbox-wrapped results.
def _extract_shared_file_payload(result: Any) -> dict[str, Any] | None:
    """Return a normalized shared-file payload from direct or sandbox-wrapped results."""

    result = _coerce_tool_result_object(result)
    if not isinstance(result, dict):
        return None

    candidate: Any = result
    if result.get("tool") == "share_file" and isinstance(result.get("result"), dict):
        candidate = result["result"]
    elif result.get("kind") != "shared_file" and isinstance(result.get("file"), dict):
        candidate = result["file"]

    if not isinstance(candidate, dict) or candidate.get("kind") != "shared_file":
        return None

    path = str(candidate.get("path") or candidate.get("file") or "").strip()
    if not path:
        return None

    filename = str(
        candidate.get("filename")
        or candidate.get("name")
        or Path(path).name
        or "download"
    )
    payload = dict(candidate)
    payload["kind"] = "shared_file"
    payload["path"] = path
    payload["filename"] = filename
    if not payload.get("download_url") and not payload.get("downloadUrl"):
        host_path = str(payload.get("host_path") or "").strip()
        if host_path:
            payload["download_url"] = "/api/shared-file/download/?" + urlencode(
                {"path": host_path, "name": filename}
            )
    return payload


# Return frontend metadata for rich structured tool results.
def _extract_structured_tool_result(result: Any) -> dict[str, Any] | None:
    """Return frontend metadata for rich structured tool results."""

    result = _coerce_tool_result_object(result)
    if not isinstance(result, dict):
        return None
    shared_file = _extract_shared_file_payload(result)
    if shared_file is not None:
        return {
            "kind": "shared_file",
            "file": shared_file,
            "ui": {
                "kind": "shared_file",
                "status": "done",
                "file": shared_file,
            },
        }
    if not isinstance(result.get("model_context"), str):
        return None
    if not isinstance(result.get("ui"), dict) and not isinstance(result.get("sources"), list):
        return None
    return result


# Split one tool result into model-visible text and UI-only metadata.
def split_tool_result_payload(content: Any) -> tuple[str, dict[str, Any]]:
    """Split one tool result into model-visible text and UI-only metadata."""

    if is_tool_execution_cancelled(content):
        model_context = (
            "TOOL_EXECUTION_CANCELLED: The user stopped this tool call. "
            "Do not retry it unless the user explicitly asks you to start again."
        )
        structured = {
            "model_context": model_context,
            "sources": [],
            "ui": {
                "kind": "tool_execution",
                "status": "cancelled",
                "cancelled": True,
            },
        }
        return model_context, {
            "structured_content": structured,
            "tool_ui": structured["ui"],
        }

    if isinstance(content, dict) and "_image_b64" in content:
        return f"[Image: {content.get('_path', 'image')}]", _image_tool_result_extras(content)

    structured = _extract_structured_tool_result(content)
    if isinstance(structured, dict):
        extras: dict[str, Any] = {"structured_content": structured}
        ui_payload = structured.get("ui")
        if isinstance(ui_payload, dict):
            extras["tool_ui"] = ui_payload
        return _serialize_tool_result(content), extras

    if isinstance(content, dict) and "_tool_result_content" in content:
        extras: dict[str, Any] = {}
        wrapped_structured = content.get("_tool_result_structured")
        if isinstance(wrapped_structured, dict):
            extras["structured_content"] = wrapped_structured
            ui_payload = wrapped_structured.get("ui")
            if isinstance(ui_payload, dict):
                extras["tool_ui"] = ui_payload
        return str(content.get("_tool_result_content") or ""), extras

    return _serialize_tool_result(content), {}


# Build UI-only metadata for an inline image tool result.
def _image_tool_result_extras(content: dict[str, Any]) -> dict[str, Any]:
    """Build UI-only metadata for an inline image tool result."""

    data_base64 = content.get("_image_b64")
    if not data_base64:
        return {}

    mime_type = str(content.get("_mime_type") or "image/png").strip().lower()
    if not mime_type or mime_type == "image":
        mime_type = "image/png"
    elif "/" not in mime_type:
        mime_type = f"image/{mime_type}"
    payload: dict[str, Any] = {
        "path": content.get("_path", ""),
        "kind": "image",
        "mime": mime_type,
        "preview": {
            "type": "inline_base64",
            "mime_type": mime_type,
            "data_base64": data_base64,
        },
    }
    for source_key, target_key in (
        ("_size_bytes", "size_bytes"),
        ("_width", "width"),
        ("_height", "height"),
    ):
        value = content.get(source_key)
        if value is not None:
            payload[target_key] = value

    return {
        "structured_content": payload,
        "tool_ui": {
            "image": payload,
        },
    }


# Extract one inline image payload.
def _extract_inline_image_payload(result: Any) -> dict[str, Any] | None:
    """Extract an Ollama image payload from the sandbox v2 read envelope."""

    if not isinstance(result, dict) or not result.get("ok"):
        return None

    payload = result.get("result")
    if not isinstance(payload, dict):
        return None

    preview = payload.get("preview")
    if not isinstance(preview, dict):
        return None

    data_base64 = preview.get("data_base64")
    if not data_base64:
        return None

    return {
        "_image_b64": data_base64,
        "_mime_type": preview.get("mime_type", payload.get("mime", "image/png")),
        "_path": payload.get("path", ""),
        "_size_bytes": payload.get("size_bytes"),
        "_width": payload.get("width"),
        "_height": payload.get("height"),
    }

def prepare_tool_call(
    tool_lookup: dict[str, dict[str, Any]], tool_call: dict[str, Any]
) -> dict[str, Any]:
    """Run an optional argument-only preflight before a tool event is published."""

    prepared_call = dict(tool_call or {})
    raw_arguments = prepared_call.get("arguments") if isinstance(prepared_call.get("arguments"), dict) else {}
    prepared_call["raw_arguments"] = raw_arguments
    lookup_entry = tool_lookup.get(str(prepared_call.get("name") or ""))
    if not isinstance(lookup_entry, dict):
        prepared_call["arguments"] = raw_arguments
        return prepared_call
    tool_definition = lookup_entry.get("tool") if isinstance(lookup_entry.get("tool"), dict) else {}
    server_definition = lookup_entry.get("server") if isinstance(lookup_entry.get("server"), dict) else {}
    instant_mode = bool(tool_definition.get("instant_mode"))
    required_search_batch_size = int(tool_definition.get("instant_search_batch_size", 0) or 0)
    if instant_mode and tool_definition.get("id") == "web_search" and required_search_batch_size:
        raw_queries = raw_arguments.get("query")
        query_count = len(raw_queries) if isinstance(raw_queries, list) else 0
        if query_count != required_search_batch_size:
            description = str(raw_arguments.get("description") or "").strip()
            message = (
                "INVALID_INSTANT_SEARCH_BATCH: this slash-command search requires exactly "
                f"{required_search_batch_size} query strings in one web_search call."
            )
            prepared_call["arguments"] = {}
            prepared_call["tool_ui"] = {
                "kind": "web_search",
                "status": "rejected",
                "instant_mode": True,
                "description": description,
                "query_count": query_count,
                "error": {"code": "INVALID_INSTANT_SEARCH_BATCH"},
            }
            prepared_call["preflight_error_result"] = {
                "sources": [],
                "model_context": message,
                "ui": dict(prepared_call["tool_ui"]),
            }
            return prepared_call
    if instant_mode and tool_definition.get("id") == "read_page":
        raw_urls = raw_arguments.get("url")
        urls = raw_urls if isinstance(raw_urls, list) else [raw_urls]
        urls = [str(url or "").strip() for url in urls if str(url or "").strip()]
        prepared_call["tool_ui"] = {
            "kind": "read_page",
            "status": "pending",
            "instant_mode": True,
        }
        if not urls or len(urls) > INSTANT_SEARCH_BATCH_LIMIT:
            message = (
                "INVALID_READ_PAGE_BATCH: instant mode requires one to three exact URLs "
                "in a single read_page call."
            )
            prepared_call["arguments"] = {}
            prepared_call["preflight_error_result"] = {
                "sources": [],
                "model_context": message,
                "ui": {
                    "kind": "read_page",
                    "status": "rejected",
                    "instant_mode": True,
                },
            }
            return prepared_call
        prepared_call["arguments"] = {"url": urls[0] if len(urls) == 1 else urls}
        return prepared_call
    if not tool_definition.get("prepares_arguments"):
        prepared_call["arguments"] = raw_arguments
        return prepared_call

    try:
        worker_arguments = dict(raw_arguments)
        if instant_mode and tool_definition.get("id") == "web_search":
            worker_arguments["_instant_mode"] = True
        if server_definition.get("external"):
            result = _run_worker(
                Path(server_definition["server_file"]),
                "prepare",
                {"tool_id": tool_definition.get("id"), "arguments": worker_arguments},
                persistent=not bool(server_definition.get("fresh_process_per_call")),
            )
        else:
            preparers = server_definition.get("tool_preparers")
            preparer = preparers.get(tool_definition.get("id")) if isinstance(preparers, dict) else None
            if not callable(preparer):
                raise RuntimeError("tool argument preparer is not registered")
            result = _execute_callable(preparer, worker_arguments)
        if not isinstance(result, dict):
            raise RuntimeError("tool preparer returned a non-object response")
    except Exception as exc:  # noqa: BLE001
        message = f"Tool preparation failed: {type(exc).__name__}: {exc}"
        result = {
            "ok": False,
            "arguments": {},
            "tool_ui": {"kind": str(tool_definition.get("id") or "tool"), "status": "error"},
            "error_result": {
                "sources": [],
                "model_context": message,
                "ui": {"kind": str(tool_definition.get("id") or "tool"), "status": "error"},
            },
        }

    prepared_call["arguments"] = result.get("arguments") if isinstance(result.get("arguments"), dict) else {}
    parallel_batch_size = int(prepared_call.get("parallel_batch_size", 0) or 0)
    if isinstance(result.get("tool_ui"), dict):
        prepared_call["tool_ui"] = copy.deepcopy(result["tool_ui"])
        if instant_mode:
            prepared_call["tool_ui"]["instant_mode"] = True
        if parallel_batch_size > 1:
            prepared_call["tool_ui"]["collapsed_parallel_calls"] = parallel_batch_size
    if not bool(result.get("ok", True)):
        if isinstance(prepared_call.get("tool_ui"), dict):
            prepared_call["tool_ui"]["rejected_arguments"] = copy.deepcopy(raw_arguments)
        error_result = copy.deepcopy(result.get("error_result") or "Tool preparation failed.")
        if parallel_batch_size > 1 and isinstance(error_result, dict):
            existing_context = str(error_result.get("model_context") or "").strip()
            prefix = (
                f"PARALLEL_SEARCH_BATCH_REJECTED: {parallel_batch_size} parallel web_search calls "
                "were collapsed into one atomic batch before execution. "
            )
            error_result["model_context"] = prefix + existing_context
            if isinstance(error_result.get("ui"), dict):
                error_result["ui"]["collapsed_parallel_calls"] = parallel_batch_size
        if instant_mode and isinstance(error_result, dict) and isinstance(error_result.get("ui"), dict):
            error_result["ui"]["instant_mode"] = True
        prepared_call["preflight_error_result"] = error_result
    return prepared_call


def _tool_definition_id(
    tool_lookup: dict[str, dict[str, Any]], tool_call: dict[str, Any]
) -> str:
    lookup_entry = tool_lookup.get(str((tool_call or {}).get("name") or ""))
    if not isinstance(lookup_entry, dict):
        return ""
    tool_definition = lookup_entry.get("tool")
    if not isinstance(tool_definition, dict):
        return ""
    return str(tool_definition.get("id") or "").strip()


def _normalized_batch_effort(arguments: dict[str, Any]) -> str:
    return str(arguments.get("effort") or "medium").strip().lower()


def _normalized_batch_bool(arguments: dict[str, Any], key: str) -> bool:
    value = arguments.get(key, False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


_ADVANCED_SEARCH_VERTICAL_KEYS = ("web", "academic", "shopping", "onion")


def _web_search_call_mode(arguments: dict[str, Any]) -> str:
    if any(key in arguments for key in _ADVANCED_SEARCH_VERTICAL_KEYS):
        return "advanced"
    if "query" in arguments:
        return "legacy"
    return ""


def _rejected_parallel_search_call(
    tool_call: dict[str, Any], *, message: str, call_count: int
) -> dict[str, Any]:
    rejected = dict(tool_call)
    rejected["arguments"] = {}
    rejected["tool_ui"] = {
        "kind": "web_search",
        "status": "rejected",
        "result_count": 0,
        "query_count": call_count,
        "collapsed_parallel_calls": call_count,
        "error": {"code": "INCOMPATIBLE_PARALLEL_SEARCH_BATCH"},
    }
    rejected["preflight_error_result"] = {
        "error": {"code": "INCOMPATIBLE_PARALLEL_SEARCH_BATCH"},
        "sources": [],
        "model_context": f"INCOMPATIBLE_PARALLEL_SEARCH_BATCH: {message}",
        "ui": dict(rejected["tool_ui"]),
    }
    return rejected


def _merge_parallel_web_search_calls(tool_calls: list[dict[str, Any]]) -> dict[str, Any]:
    first = dict(tool_calls[0])
    first["absorbed_tool_calls"] = [dict(call) for call in tool_calls[1:]]
    first["parallel_batch_size"] = len(tool_calls)

    raw_arguments = [
        dict(call.get("arguments") or {})
        if isinstance(call.get("arguments"), dict)
        else {}
        for call in tool_calls
    ]
    modes = {_web_search_call_mode(arguments) for arguments in raw_arguments}
    if len(modes) != 1 or not modes or "" in modes:
        first["raw_arguments"] = raw_arguments
        return _rejected_parallel_search_call(
            first,
            message="parallel web_search calls must all use the same advanced or legacy argument shape",
            call_count=len(tool_calls),
        )

    efforts = {_normalized_batch_effort(arguments) for arguments in raw_arguments}
    if len(efforts) != 1:
        first["raw_arguments"] = raw_arguments
        return _rejected_parallel_search_call(
            first,
            message="parallel web_search calls must use the same effort before they can be batched",
            call_count=len(tool_calls),
        )

    mode = next(iter(modes))
    merged_arguments = copy.deepcopy(raw_arguments[0])
    instant_shape = mode == "legacy" and all(
        "description" in arguments and "query" in arguments
        for arguments in raw_arguments
    )
    if instant_shape:
        operator_values = [
            _canonical_tool_arguments(arguments.get("operators") or {})
            for arguments in raw_arguments
        ]
        if any(value != operator_values[0] for value in operator_values[1:]):
            first["raw_arguments"] = raw_arguments
            return _rejected_parallel_search_call(
                first,
                message="parallel instant searches must use the same operators",
                call_count=len(tool_calls),
            )
        descriptions: list[str] = []
        for arguments in raw_arguments:
            description = str(arguments.get("description") or "").strip()
            if description and description not in descriptions:
                descriptions.append(description)
        merged_arguments["description"] = " · ".join(descriptions)[:80]
    if mode == "advanced":
        for vertical in _ADVANCED_SEARCH_VERTICAL_KEYS:
            merged_values: list[Any] = []
            for arguments in raw_arguments:
                if vertical not in arguments:
                    continue
                value = arguments.get(vertical)
                if isinstance(value, list):
                    merged_values.extend(copy.deepcopy(value))
                else:
                    merged_values.append(copy.deepcopy(value))
            if not merged_values:
                merged_arguments.pop(vertical, None)
            elif len(merged_values) == 1:
                merged_arguments[vertical] = merged_values[0]
            else:
                merged_arguments[vertical] = merged_values
    else:
        verticals = {arguments.get("vertical", "web") for arguments in raw_arguments
                     if isinstance(arguments.get("vertical", "web"), str)}
        if len(verticals) != 1 or any(
            not isinstance(arguments.get("vertical", "web"), str) for arguments in raw_arguments
        ):
            return _rejected_parallel_search_call(
                first,
                message="parallel searches must use the same vertical; call different verticals separately",
                call_count=len(tool_calls),
            )
        routing_keys = ("shopping", "academic", "onion")
        for key in routing_keys:
            values = {_normalized_batch_bool(arguments, key) for arguments in raw_arguments}
            if len(values) != 1:
                first["raw_arguments"] = raw_arguments
                return _rejected_parallel_search_call(
                    first,
                    message=f"parallel legacy web_search calls must use the same {key} routing value",
                    call_count=len(tool_calls),
                )
        merged_queries: list[Any] = []
        for arguments in raw_arguments:
            value = arguments.get("query")
            if isinstance(value, list):
                merged_queries.extend(copy.deepcopy(value))
            else:
                merged_queries.append(copy.deepcopy(value))
        merged_arguments["query"] = merged_queries

    first["arguments"] = merged_arguments
    return first


def _coalesce_parallel_web_search_calls(
    tool_lookup: dict[str, dict[str, Any]], tool_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for tool_call in tool_calls:
        if _tool_definition_id(tool_lookup, tool_call) != "web_search":
            continue
        alias = str(tool_call.get("name") or "")
        groups.setdefault(alias, []).append(tool_call)

    emitted_aliases: set[str] = set()
    output: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        alias = str(tool_call.get("name") or "")
        group = groups.get(alias, [])
        if len(group) < 2:
            output.append(tool_call)
            continue
        if alias in emitted_aliases:
            continue
        emitted_aliases.add(alias)
        output.append(_merge_parallel_web_search_calls(group))
    return output


def _tool_definition_is_instant(
    tool_lookup: dict[str, dict[str, Any]], tool_call: dict[str, Any]
) -> bool:
    lookup_entry = tool_lookup.get(str((tool_call or {}).get("name") or "")) or {}
    tool_definition = lookup_entry.get("tool") if isinstance(lookup_entry, dict) else {}
    return bool(isinstance(tool_definition, dict) and tool_definition.get("instant_mode"))


def _coalesce_parallel_instant_read_page_calls(
    tool_lookup: dict[str, dict[str, Any]], tool_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for tool_call in tool_calls:
        if (
            _tool_definition_id(tool_lookup, tool_call) == "read_page"
            and _tool_definition_is_instant(tool_lookup, tool_call)
        ):
            groups.setdefault(str(tool_call.get("name") or ""), []).append(tool_call)

    emitted_aliases: set[str] = set()
    output: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        alias = str(tool_call.get("name") or "")
        group = groups.get(alias, [])
        if len(group) < 2:
            output.append(tool_call)
            continue
        if alias in emitted_aliases:
            continue
        emitted_aliases.add(alias)
        merged = dict(group[0])
        merged["absorbed_tool_calls"] = [dict(call) for call in group[1:]]
        merged["parallel_batch_size"] = len(group)
        urls: list[Any] = []
        for call in group:
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            value = arguments.get("url")
            urls.extend(value if isinstance(value, list) else [value])
        merged["arguments"] = {"url": urls}
        output.append(merged)
    return output


def prepare_tool_calls(
    tool_lookup: dict[str, dict[str, Any]], tool_calls: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    coalesced = _coalesce_parallel_web_search_calls(tool_lookup, list(tool_calls or []))
    coalesced = _coalesce_parallel_instant_read_page_calls(tool_lookup, coalesced)
    return [
        tool_call
        if tool_call.get("preflight_error_result") is not None
        else prepare_tool_call(tool_lookup, tool_call)
        for tool_call in coalesced
    ]


def build_tool_event(
    tool_lookup: dict[str, dict[str, Any]], tool_call: dict[str, Any]
) -> dict[str, Any]:
    """Build the shared canonical tool event used by every model adapter."""

    alias = str(tool_call.get("name") or "")
    lookup_entry = tool_lookup.get(alias, {})
    server_definition = lookup_entry.get("server", {})
    tool_definition = lookup_entry.get("tool", {})
    event = {
        "alias": alias,
        "server_id": server_definition.get("id") or "",
        "server_name": server_definition.get("name") or server_definition.get("id") or "",
        "tool_id": tool_definition.get("id") or alias,
        "tool_name": tool_definition.get("name") or alias,
        "arguments": tool_call.get("arguments") or {},
    }
    if tool_definition.get("instant_mode"):
        event["instant_mode"] = True
    if isinstance(tool_call.get("tool_ui"), dict):
        event["tool_ui"] = tool_call["tool_ui"]
    elif tool_definition.get("instant_mode"):
        event["tool_ui"] = {
            "kind": str(tool_definition.get("id") or "tool"),
            "status": "pending",
            "instant_mode": True,
        }
    return event


def canonicalize_transcript_tool_calls(
    transcript_message: dict[str, Any], tool_calls: list[dict[str, Any]]
) -> dict[str, Any]:
    """Replace provider tool arguments before the transcript is persisted or published."""

    message = copy.deepcopy(transcript_message or {})
    raw_calls = message.get("tool_calls")
    if not isinstance(raw_calls, list):
        return message

    remaining = list(tool_calls or [])
    canonical_calls: list[dict[str, Any]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, dict):
            continue
        function_payload = raw_call.get("function")
        if not isinstance(function_payload, dict):
            continue
        name = str(function_payload.get("name") or "")
        raw_id = str(raw_call.get("id") or "")
        match_index = next(
            (
                index
                for index, prepared in enumerate(remaining)
                if (
                    (
                        bool(raw_id)
                        and bool(str(prepared.get("id") or ""))
                        and str(prepared.get("id") or "") == raw_id
                    )
                    or (
                        (not raw_id or not str(prepared.get("id") or ""))
                        and str(prepared.get("name") or "") == name
                    )
                )
            ),
            None,
        )
        if match_index is None:
            continue
        prepared = remaining.pop(match_index)
        canonical = prepared.get("arguments")
        canonical = canonical if isinstance(canonical, dict) else {}
        if isinstance(function_payload.get("arguments"), str):
            function_payload["arguments"] = json.dumps(
                canonical,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        else:
            function_payload["arguments"] = canonical
        canonical_calls.append(raw_call)
    message["tool_calls"] = canonical_calls

    google_parts = message.get("google_parts")
    if isinstance(google_parts, list):
        remaining_names = [str(call.get("name") or "") for call in tool_calls or []]
        filtered_parts: list[Any] = []
        for raw_part in google_parts:
            if not isinstance(raw_part, dict) or not isinstance(raw_part.get("function_call"), dict):
                filtered_parts.append(raw_part)
                continue
            function_name = str(raw_part["function_call"].get("name") or "")
            try:
                name_index = remaining_names.index(function_name)
            except ValueError:
                continue
            remaining_names.pop(name_index)
            # Keep the provider-native signed function-call part byte-for-byte apart
            # from the surrounding deepcopy. Gemini may reject altered signed parts.
            filtered_parts.append(raw_part)
        message["google_parts"] = filtered_parts
    return message


# Execute one local tool.
def call_ollama_tool(
    tool_lookup: dict[str, dict[str, Any]],
    alias: str,
    arguments: dict[str, Any] | None,
    context: dict[str, Any] | None = None,
    event_callback=None,
    cancellation: ActiveToolExecution | None = None,
) -> str | dict:
    """Execute a local tool and serialize its result for Ollama.

    Returns a str for normal results, or a dict with ``_image_b64`` and
    ``_mime_type`` keys when the tool returns an image payload.
    """

    lookup_entry = tool_lookup.get(alias)
    if not lookup_entry:
        return f"Unknown tool: {alias}"

    server_definition = lookup_entry["server"]
    tool_definition = lookup_entry["tool"]
    call_arguments = arguments if isinstance(arguments, dict) else {}
    call_context = dict(context or {})

    # Fill the standard context expected by local tool handlers.
    call_context.setdefault("server_id", server_definition["id"])
    call_context.setdefault("server_name", server_definition["name"])
    call_context.setdefault("tool_id", tool_definition["id"])
    call_context.setdefault("tool_name", tool_definition["name"])
    call_context.setdefault("tool_alias", tool_definition["alias"])
    if not server_definition.get("user_mcp"):
        call_context.setdefault("server_file", str(server_definition["server_file"]))
    call_context.setdefault(
        "tools_dir",
        str(server_definition.get("source_tools_dir") or TOOLS_DIR),
    )
    call_context.setdefault(
        "module_dir",
        str(server_definition.get("source_module_dir") or MODULE_ROOT),
    )
    started_at = time.perf_counter()

    if _is_debug_logging_enabled():
        _print_runtime_event(
            "Tool starting: "
            f"server={server_definition['id']}, "
            f"tool={tool_definition['id']}, "
            f"alias={tool_definition['alias']}, "
            f"context={_summarize_tool_context(call_context)}, "
            f"args={_preview_jsonish(call_arguments, limit=180)}"
        )

    try:
        if server_definition["id"] == "sandbox":
            skills_config.sync_skills_to_sandbox()

        if server_definition.get("user_mcp"):
            entry = server_definition["user_mcp_entry"]
            mcp_tool_name = str(tool_definition.get("mcp_tool_name") or tool_definition["id"] or "").strip()
            if not mcp_tool_name:
                return "Tool execution failed: missing MCP tool name"
            if cancellation is not None:
                cancellation.set_cancel_callback(lambda: user_mcp_client.cancel_entry(entry))
            try:
                result = user_mcp_client.call_user_mcp_tool(entry, mcp_tool_name, call_arguments)
            finally:
                if cancellation is not None:
                    cancellation.clear_cancel_callback()
            if _is_debug_logging_enabled():
                _print_runtime_event(
                    "Tool completed: "
                    f"server={server_definition['id']}, "
                    f"tool={tool_definition['id']}, "
                    f"status=ok, "
                    f"took={time.perf_counter() - started_at:.2f}s, "
                    f"result={_summarize_tool_result(result)}"
                )
            return result

        if server_definition.get("external"):
            worker_context = json.loads(json.dumps(call_context, ensure_ascii=False, default=str))
            event_channel = ToolEventSocketChannel(event_callback) if callable(event_callback) else None
            if event_channel is not None:
                worker_context["_event_channel"] = event_channel.config
            worker_payload = {
                "tool_id": tool_definition["id"],
                "arguments": call_arguments,
                "context": worker_context,
            }
            server_file = Path(server_definition["server_file"])
            try:
                try:
                    result = _run_worker(
                        server_file,
                        "call",
                        worker_payload,
                        persistent=not bool(server_definition.get("fresh_process_per_call")),
                        cancellation=cancellation,
                    )
                except RuntimeError as worker_exc:
                    if (
                        "Tool worker stopped" not in str(worker_exc)
                        or (cancellation is not None and cancellation.cancelled.is_set())
                    ):
                        raise
                    logger.warning(
                        "Persistent tool worker stopped for %s; retrying call in one-shot mode.",
                        server_file,
                    )
                    result = _run_worker(
                        server_file,
                        "call",
                        worker_payload,
                        persistent=False,
                        cancellation=cancellation,
                    )
            finally:
                if event_channel is not None:
                    event_channel.close()
        else:
            if cancellation is not None:
                call_context["cancel_event"] = cancellation.cancelled
            handler = server_definition["tool_handlers"].get(tool_definition["id"])
            if handler is None:
                handler = server_definition["tool_handlers"].get(tool_definition["alias"])

            # Prefer a dedicated handler when present, then fall back to the
            # generic dispatcher exported by the server module.
            if handler is not None:
                signature = inspect.signature(handler)
                if len(signature.parameters) <= 1:
                    result = _execute_callable(handler, call_arguments, cancellation=cancellation)
                else:
                    result = _execute_callable(
                        handler,
                        call_arguments,
                        call_context,
                        cancellation=cancellation,
                    )
            elif server_definition["server_callable"] is not None:
                result = _dispatch_server_callable(
                    server_definition["server_callable"],
                    tool_definition["id"],
                    call_arguments,
                    call_context,
                    cancellation=cancellation,
                )
            else:
                return f"Tool execution failed: no handler registered for {tool_definition['id']}"

        if tool_definition.get("instant_mode") and isinstance(result, dict):
            result = copy.deepcopy(result)
            result_ui = result.get("ui") if isinstance(result.get("ui"), dict) else {}
            result_ui["instant_mode"] = True
            result["ui"] = result_ui

        # Image payloads stay structured so multimodal adapters can feed them
        # back to the model without flattening them into plain text.
        image_payload = _extract_inline_image_payload(result)
        if image_payload is not None:
            if _is_debug_logging_enabled():
                _print_runtime_event(
                    "Tool completed: "
                    f"server={server_definition['id']}, "
                    f"tool={tool_definition['id']}, "
                    f"status=ok, "
                    f"took={time.perf_counter() - started_at:.2f}s, "
                    f"result={_summarize_tool_result(image_payload)}"
                )
            return image_payload

        if _is_debug_logging_enabled():
            _print_runtime_event(
                "Tool completed: "
                f"server={server_definition['id']}, "
                f"tool={tool_definition['id']}, "
                f"status=ok, "
                f"took={time.perf_counter() - started_at:.2f}s, "
                f"result={_summarize_tool_result(result)}"
            )
        structured_result = _extract_structured_tool_result(_coerce_tool_result_object(result))
        if structured_result is not None:
            return {
                "_tool_result_content": _serialize_tool_result(result),
                "_tool_result_structured": structured_result,
            }
        return _serialize_tool_result(result)
    except Exception as exc:
        logger.exception("Tool execution failed for %s.%s", server_definition["id"], tool_definition["id"])
        _print_runtime_event(
            "Tool failed: "
            f"server={server_definition['id']}, "
            f"tool={tool_definition['id']}, "
            f"status=error, "
            f"took={time.perf_counter() - started_at:.2f}s, "
            f"error={exc}"
        )
        return f"Tool execution failed: {exc}"


TOOL_EVENT_BATCH_INTERVAL_SECONDS = 3.0
TOOL_EVENT_BATCH_MAX_EVENTS = 64
TOOL_EVENT_BATCH_MAX_BYTES = 64 * 1024


def stream_ollama_tool(
    tool_lookup: dict[str, dict[str, Any]],
    alias: str,
    arguments: dict[str, Any] | None,
    context: dict[str, Any] | None = None,
):
    """Yield realtime activity events and return the final tool result.

    Tool execution runs on a background thread so adapters can immediately
    forward side-channel events while the normal worker response is pending.
    """

    updates: Queue[dict[str, Any]] = Queue()
    finished = threading.Event()
    outcome: dict[str, Any] = {}
    lookup_entry = tool_lookup.get(alias) or {}
    server_definition = lookup_entry.get("server") or {}
    tool_definition = lookup_entry.get("tool") or {}
    call_context = dict(context or {})
    call_arguments = arguments if isinstance(arguments, dict) else {}
    research_session_id = str(call_arguments.get("session_id") or "").strip()
    if not research_session_id.startswith("deep-research:"):
        research_session_id = ""
    execution = ActiveToolExecution(
        str(call_context.get("generation_id") or ""),
        research_session_id=research_session_id,
    )
    _register_active_tool(execution)

    def execute() -> None:
        try:
            outcome["result"] = call_ollama_tool(
                tool_lookup,
                alias,
                arguments,
                context=call_context,
                event_callback=updates.put,
                cancellation=execution,
            )
            if execution.cancelled.is_set():
                outcome["result"] = TOOL_EXECUTION_CANCELLED
        except BaseException as exc:  # noqa: BLE001 - re-raised on generator thread
            if execution.cancelled.is_set():
                outcome["result"] = TOOL_EXECUTION_CANCELLED
            else:
                outcome["error"] = exc
        finally:
            _unregister_active_tool(execution)
            finished.set()

    thread = threading.Thread(
        target=execute,
        name=f"aslm-tool-stream-{alias}",
        daemon=True,
    )
    thread.start()

    if research_session_id:
        def watch_research_stop() -> None:
            """Bridge the durable UI command to a blocking provider/tool call."""

            try:
                from Tools.deep_research import control as deep_research_control
            except Exception as exc:  # pragma: no cover - project import guard
                logger.warning("Could not start Deep Research stop monitor: %s", exc)
                return
            while not finished.wait(0.25):
                try:
                    if deep_research_control.latest_cancel_requested(research_session_id):
                        execution.cancel()
                        cancelled_at = time.time()
                        deep_research_control.update_state(
                            research_session_id,
                            status="cancelled",
                            phase="cancelled",
                            latest_action="Research stopped by user",
                            can_approve=False,
                            can_edit=False,
                            can_stop=False,
                            cancelled_at=cancelled_at,
                            completed_at=cancelled_at,
                        )
                        return
                except Exception as exc:  # noqa: BLE001 - transient state reads are retried
                    logger.debug("Deep Research stop monitor could not read state: %s", exc)

        threading.Thread(
            target=watch_research_stop,
            name=f"aslm-research-stop-{research_session_id.rsplit(':', 1)[-1][:8]}",
            daemon=True,
        ).start()

    batch: list[dict[str, Any]] = []
    batch_bytes = 0
    batch_started_at: float | None = None

    def progress_batch(events: list[dict[str, Any]]) -> dict[str, Any]:
        first_sequence = events[0].get("sequence") if events else None
        last_sequence = events[-1].get("sequence") if events else None
        return {
            "tool_progress": {
                "alias": alias,
                "server_id": server_definition.get("id"),
                "tool_id": tool_definition.get("id"),
                "event": {
                    "type": "event_batch",
                    "count": len(events),
                    "first_sequence": first_sequence,
                    "last_sequence": last_sequence,
                    "events": events,
                },
            }
        }

    while not finished.is_set() or not updates.empty() or batch:
        now = time.monotonic()
        timeout = 0.05
        if batch_started_at is not None:
            timeout = min(
                timeout,
                max(0.0, TOOL_EVENT_BATCH_INTERVAL_SECONDS - (now - batch_started_at)),
            )
        try:
            event = updates.get(timeout=timeout)
        except Empty:
            event = None

        if event is not None:
            event_bytes = len(
                json.dumps(event, ensure_ascii=False, default=str).encode("utf-8")
            )
            if batch and batch_bytes + event_bytes > TOOL_EVENT_BATCH_MAX_BYTES:
                yield progress_batch(batch)
                batch = []
                batch_bytes = 0
                batch_started_at = None
            if not batch:
                batch_started_at = time.monotonic()
            batch.append(event)
            batch_bytes += event_bytes

        now = time.monotonic()
        should_flush = bool(batch) and (
            len(batch) >= TOOL_EVENT_BATCH_MAX_EVENTS
            or batch_bytes >= TOOL_EVENT_BATCH_MAX_BYTES
            or (
                batch_started_at is not None
                and now - batch_started_at >= TOOL_EVENT_BATCH_INTERVAL_SECONDS
            )
            or (finished.is_set() and updates.empty())
        )
        if should_flush:
            yield progress_batch(batch)
            batch = []
            batch_bytes = 0
            batch_started_at = None

    thread.join(timeout=1)
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("result")
