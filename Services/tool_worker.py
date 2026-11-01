# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import contextlib
import importlib.util
import inspect
import json
import re
import socket
import sys
import threading
from pathlib import Path
from types import ModuleType
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from Settings.proxy_policy import apply_loopback_proxy_bypass

SERVER_DISPATCHER_NAMES = ("call_tool", "run_tool", "execute_tool", "execute")
SERVER_METADATA_NAMES = ("MCP_SERVER", "SERVER")
TOOL_HANDLER_NAMES = ("TOOL_HANDLERS", "TOOL_EXECUTORS")
TOOL_PREPARER_NAMES = ("TOOL_PREPARERS", "TOOL_ARGUMENT_PREPARERS")
WORKER_HEARTBEAT_SECONDS = 5.0


class WorkerEventEmitter:
    """Send structured realtime tool events over a dedicated local socket."""

    def __init__(self, config: dict[str, Any] | None) -> None:
        config = config if isinstance(config, dict) else {}
        self.host = str(config.get("host") or "127.0.0.1")
        self.port = int(config.get("port") or 0)
        self.token = str(config.get("token") or "")
        self.connection: socket.socket | None = None
        self.lock = threading.Lock()
        self.disabled = not (self.host == "127.0.0.1" and self.port > 0 and self.token)

    def emit(self, event: dict[str, Any]) -> None:
        if self.disabled or not isinstance(event, dict):
            return
        envelope = json.dumps(
            {"token": self.token, "event": _to_jsonable(event)},
            ensure_ascii=False,
        ).encode("utf-8") + b"\n"
        with self.lock:
            try:
                if self.connection is None:
                    self.connection = socket.create_connection((self.host, self.port), timeout=3)
                self.connection.sendall(envelope)
            except (OSError, ValueError):
                self.close()
                self.disabled = True

    def close(self) -> None:
        connection = self.connection
        self.connection = None
        if connection is not None:
            with contextlib.suppress(OSError):
                connection.close()


# Run async tool handlers on one persistent background event loop.
class AsyncCallableRunner:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._lock = threading.Lock()

    # Own the asyncio loop inside a dedicated daemon thread.
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
                name="aslm-tool-worker-async-runner",
                daemon=True,
            )
            self._thread.start()
            self._ready.wait()

    # Execute one coroutine on the background loop and block for the result.
    def run(self, coro: Any) -> Any:
        self.ensure_started()
        assert self._loop is not None
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

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


_ASYNC_CALLABLE_RUNNER = AsyncCallableRunner()


# Module loading and metadata

# Normalize public identifiers into stable lower-case slugs.
def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized or "tool"


# Load one tool server module from disk and register its folder on sys.path.
def _load_module(server_file: Path) -> ModuleType:
    server_root = server_file.parent
    if str(server_root) not in sys.path:
        sys.path.insert(0, str(server_root))

    module_name = f"aslm_chat_worker_{_slugify(server_root.name)}"
    spec = importlib.util.spec_from_file_location(module_name, server_file)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MCP server module from {server_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Read MCP_SERVER (or SERVER) metadata from a loaded module.
def _server_metadata(module: ModuleType, folder_name: str) -> dict[str, Any]:
    raw_server: Any = {}
    for attr_name in SERVER_METADATA_NAMES:
        raw_server = getattr(module, attr_name, None)
        if raw_server is not None:
            break

    if raw_server is None:
        raw_server = {}
    if not isinstance(raw_server, dict):
        raise ValueError("MCP server module must expose an MCP_SERVER dictionary")

    metadata: dict[str, Any] = {
        "id": _slugify(str(raw_server.get("id") or folder_name)),
        "name": str(raw_server.get("name") or folder_name).strip() or folder_name,
        "description": str(raw_server.get("description") or "").strip(),
    }
    if raw_server.get("requires_docker"):
        metadata["requires_docker"] = True
    if raw_server.get("fresh_process_per_call"):
        metadata["fresh_process_per_call"] = True
    if callable(getattr(module, "get_model_context", None)):
        metadata["has_model_context"] = True
    return metadata


# Normalize one JSON-schema-like tool parameters object.
def _normalize_schema(schema: Any) -> dict[str, Any]:
    if not isinstance(schema, dict):
        return {"type": "object", "properties": {}}

    normalized = dict(schema)
    normalized.setdefault("type", "object")
    normalized.setdefault("properties", {})
    if not isinstance(normalized.get("properties"), dict):
        normalized["properties"] = {}
    return normalized


# Build normalized tool definitions from a module TOOLS export.
def _server_tools(module: ModuleType, server_id: str) -> list[dict[str, Any]]:
    raw_tools = getattr(module, "TOOLS", None)
    if raw_tools is None:
        legacy_tool = getattr(module, "TOOL", None)
        raw_tools = [legacy_tool] if isinstance(legacy_tool, dict) else None

    if not isinstance(raw_tools, list) or not raw_tools:
        raise ValueError("MCP server module must expose a non-empty TOOLS list")

    tools: list[dict[str, Any]] = []
    preparers = _tool_preparers(module)
    seen_ids: set[str] = set()
    for index, raw_tool in enumerate(raw_tools, start=1):
        if not isinstance(raw_tool, dict):
            continue

        tool_id = _slugify(str(raw_tool.get("id") or f"tool_{index}"))
        if tool_id in seen_ids:
            continue

        seen_ids.add(tool_id)
        tool_payload = {
                "id": tool_id,
                "alias": f"{server_id}__{tool_id}",
                "name": str(raw_tool.get("name") or tool_id).strip() or tool_id,
                "description": str(raw_tool.get("description") or "").strip(),
                "parameters": _normalize_schema(raw_tool.get("parameters")),
            }
        if tool_id in preparers:
            tool_payload["prepares_arguments"] = True
        tools.append(tool_payload)

    return tools


# Collect explicit per-tool handler callables from the module.
def _tool_handlers(module: ModuleType) -> dict[str, Any]:
    raw_handlers: Any = None
    for attr_name in TOOL_HANDLER_NAMES:
        raw_handlers = getattr(module, attr_name, None)
        if raw_handlers is not None:
            break

    if not isinstance(raw_handlers, dict):
        return {}

    return {
        _slugify(str(raw_key or "")): raw_value
        for raw_key, raw_value in raw_handlers.items()
        if callable(raw_value)
    }


def _tool_preparers(module: ModuleType) -> dict[str, Any]:
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


# Return the generic tool dispatcher callable when the module exports one.
def _server_callable(module: ModuleType):
    for attr_name in SERVER_DISPATCHER_NAMES:
        candidate = getattr(module, attr_name, None)
        if callable(candidate):
            return candidate
    return None


# Tool execution

# Invoke sync or async callables, routing coroutines through the shared loop.
def _execute_callable(callable_fn, *args: Any) -> Any:
    if inspect.iscoroutinefunction(callable_fn):
        return _ASYNC_CALLABLE_RUNNER.run(callable_fn(*args))
    return callable_fn(*args)


# Call a generic dispatcher using a tolerant signature-matching strategy.
def _dispatch_server_callable(callable_fn, tool_id: str, arguments: dict[str, Any], context: dict[str, Any]) -> Any:
    parameter_names = list(inspect.signature(callable_fn).parameters)
    if len(parameter_names) <= 1:
        return _execute_callable(callable_fn, arguments)
    if len(parameter_names) == 2:
        first_name = parameter_names[0].lower()
        if first_name in {"tool_id", "tool", "name", "tool_name"}:
            return _execute_callable(callable_fn, tool_id, arguments)
        return _execute_callable(callable_fn, arguments, context)
    return _execute_callable(callable_fn, tool_id, arguments, context)


# Convert arbitrary return values into JSON-serializable data.
def _to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _to_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_to_jsonable(child) for child in value]

    text = getattr(value, "text", None)
    if text is not None:
        return str(text)

    return str(value)


# Public worker operations

# Return public metadata and tool list for one tool server file.
def describe(server_file: Path) -> dict[str, Any]:
    module = _load_module(server_file)
    metadata = _server_metadata(module, server_file.parent.name)
    metadata["tools"] = _server_tools(module, metadata["id"])
    return metadata


def model_context(server_file: Path) -> str:
    """Read explicitly exported, model-safe environment details without executing a tool."""
    module = _load_module(server_file)
    provider = getattr(module, "get_model_context", None)
    result = _execute_callable(provider) if callable(provider) else ""
    return result if isinstance(result, str) else ""


# Return whether the server supports the requested engine and model pair.
def supports(server_file: Path, payload: dict[str, Any]) -> bool:
    module = _load_module(server_file)
    supports_fn = getattr(module, "supports", None)
    if not callable(supports_fn):
        return True

    engine = payload.get("engine")
    model_name = payload.get("model_name")
    try:
        return bool(supports_fn(engine=engine, model_name=model_name))
    except TypeError:
        return bool(supports_fn(engine, model_name))


# Execute one tool call against a server module.
def call(server_file: Path, payload: dict[str, Any]) -> Any:
    module = _load_module(server_file)
    tool_id = _slugify(str(payload.get("tool_id") or ""))
    arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
    context = dict(payload.get("context")) if isinstance(payload.get("context"), dict) else {}
    event_emitter = WorkerEventEmitter(context.pop("_event_channel", None))
    if not event_emitter.disabled:
        context["event_callback"] = event_emitter.emit

    try:
        handlers = _tool_handlers(module)
        handler = handlers.get(tool_id)
        if handler is not None:
            signature = inspect.signature(handler)
            if len(signature.parameters) <= 1:
                return _execute_callable(handler, arguments)
            return _execute_callable(handler, arguments, context)

        dispatcher = _server_callable(module)
        if dispatcher is None:
            raise ValueError(f"No handler registered for tool '{tool_id}'.")

        return _dispatch_server_callable(dispatcher, tool_id, arguments, context)
    finally:
        event_emitter.close()


def prepare(server_file: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Prepare one tool call without performing its external work."""

    module = _load_module(server_file)
    tool_id = _slugify(str(payload.get("tool_id") or ""))
    arguments = payload.get("arguments") if isinstance(payload.get("arguments"), dict) else {}
    preparer = _tool_preparers(module).get(tool_id)
    if preparer is None:
        return {"ok": True, "arguments": arguments}
    prepared = _execute_callable(preparer, arguments)
    if not isinstance(prepared, dict):
        raise ValueError(f"Tool preparer for '{tool_id}' must return an object.")
    return prepared


# Worker protocol helpers

# Print one JSON worker envelope to stdout.
def _print_response(ok: bool, payload: Any, *, output=sys.stdout) -> int:
    key = "result" if ok else "error"
    print(json.dumps({"ok": ok, key: _to_jsonable(payload)}, ensure_ascii=False), file=output)
    output.flush()
    return 0 if ok else 1


# Emit one protocol-safe heartbeat line for the parent process.
def _worker_heartbeat(output=sys.stdout) -> None:
    print(json.dumps({"event": "heartbeat"}, ensure_ascii=False), file=output, flush=True)


# Execute one worker request without printing the envelope.
def _execute_request(operation: str, server_file: Path, payload: dict[str, Any]) -> tuple[bool, Any]:
    try:
        if operation == "describe":
            return True, describe(server_file)
        if operation == "model_context":
            return True, model_context(server_file)
        if operation == "supports":
            return True, supports(server_file, payload)
        if operation == "call":
            return True, call(server_file, payload)
        if operation == "prepare":
            return True, prepare(server_file, payload)
        return False, f"Unknown worker operation: {operation}"
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


# Run a persistent newline-delimited JSON worker on stdin until EOF.
def serve(server_file: Path) -> int:
    output = sys.stdout
    try:
        for raw_line in sys.stdin:
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            try:
                request = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                ok, payload = False, f"JSONDecodeError: {exc}"
            else:
                if not isinstance(request, dict):
                    ok, payload = False, "Worker request must be a JSON object."
                else:
                    operation = str(request.get("operation") or "").strip().lower()
                    request_payload = request.get("payload")
                    if not isinstance(request_payload, dict):
                        request_payload = {}
                    done = threading.Event()
                    outcome: dict[str, Any] = {}

                    # Run tool work on a side thread so heartbeats can continue on the main thread.
                    def run_request() -> None:
                        with contextlib.redirect_stdout(sys.stderr):
                            ok, payload = _execute_request(operation, server_file, request_payload)
                        outcome["ok"] = ok
                        outcome["payload"] = payload
                        done.set()

                    worker = threading.Thread(
                        target=run_request,
                        name="aslm-tool-worker-request",
                        daemon=True,
                    )
                    worker.start()
                    while not done.wait(WORKER_HEARTBEAT_SECONDS):
                        _worker_heartbeat(output)
                    ok = bool(outcome.get("ok"))
                    payload = outcome.get("payload")

            key = "result" if ok else "error"
            print(json.dumps({"ok": ok, key: _to_jsonable(payload)}, ensure_ascii=False), file=output, flush=True)
    finally:
        _ASYNC_CALLABLE_RUNNER.close()

    return 0


# CLI entry point for one-shot and persistent worker modes.
def main() -> int:
    apply_loopback_proxy_bypass()

    if len(sys.argv) < 3:
        return _print_response(False, "Usage: tool_worker.py <describe|supports|model_context|prepare|call> <server_file>")

    operation = sys.argv[1].strip().lower()
    server_file = Path(sys.argv[2]).resolve()
    if operation == "serve":
        return serve(server_file)

    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        payload = {}
    if not isinstance(payload, dict):
        payload = {}

    try:
        output = sys.stdout
        with contextlib.redirect_stdout(sys.stderr):
            ok, response_payload = _execute_request(operation, server_file, payload)
        return _print_response(ok, response_payload, output=output)
    except Exception as exc:
        return _print_response(False, f"{type(exc).__name__}: {exc}")


if __name__ == "__main__":
    raise SystemExit(main())
