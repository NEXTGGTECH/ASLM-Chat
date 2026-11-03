# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import importlib
import logging
import threading
import time
from typing import Any

import ollama

from API import mcp as tool_registry
from Settings import settings

logger = logging.getLogger(__name__)

# Cancellation flag of the adapter: set by abort_generation(), polled between streamed chunks and
# tool rounds, and cleared when a new generation starts.
_abort_event = threading.Event()


# Request cancellation of the active Ollama generation; the streaming loops notice it on the next chunk.
def abort_generation() -> None:
    _abort_event.set()


# Option keys that the Ollama chat API expects at the top level of the request instead of inside
# the nested options dict.
OLLAMA_TOP_LEVEL_CHAT_KEYS = {"format", "keep_alive", "logprobs", "top_logprobs"}

# ASLM-Chat request keys that only drive the adapter and must never be forwarded to client.chat.
OLLAMA_INTERNAL_CHAT_KEYS = {
    "system",
    "prompt",
    "tool_id",
    "tool_server_id",
    "tool_server_ids",
    "tool_context",
    "think",
    "think_level",
    "think_param_name",
    "think_level_param_name",
    "load_config",
    "sync_operation_defaults",
    "engine",
}

# Option keys the bundled Ollama runtime no longer accepts; they are dropped from requests.
OLLAMA_UNSUPPORTED_OPTION_KEYS = {
    "embedding_only",
    "f16_kv",
    "logits_all",
    "low_vram",
    "mirostat",
    "mirostat_eta",
    "mirostat_tau",
    "numa",
    "penalize_newline",
    "tfs_z",
    "use_mlock",
    "vocab_only",
}

# Default cap on tool-calling rounds per generation; the tool context may override it.
MAX_TOOL_ROUNDS = 100

# Readiness retry for SDK calls while the managed runtime is still warming up: attempt count and
# the base delay in seconds, which is multiplied by the attempt number.
RUNTIME_REQUEST_ATTEMPTS = 4
RUNTIME_RETRY_DELAY_SECONDS = 0.5

# Sets of dropped option names that were already reported, so each set is logged only once; the lock
# guards the registry across request threads.
_unsupported_options_lock = threading.Lock()
_reported_unsupported_option_sets: set[tuple[str, ...]] = set()


# Runtime lifecycle

# Import the managed Ollama service module lazily; return None when it is unavailable. The module name
# contains a hyphen, so it can only be loaded through importlib.
def _get_ollama_service_module():
    try:
        return importlib.import_module("Services.ollama-service")
    except ImportError as exc:
        logger.warning("[Ollama API] Could not import Services.ollama-service: %s", exc)
        return None


# Start the managed Ollama runtime (through the Ollama service module) so it is running before requests.
def prepare_runtime(engine: str | None = None) -> None:
    ollama_service = _get_ollama_service_module()
    if ollama_service is not None:
        ollama_service.start_ollama(engine=engine)


# Stop the managed Ollama runtime when the engine is deselected.
def cleanup_runtime() -> None:
    ollama_service = _get_ollama_service_module()
    if ollama_service is not None:
        ollama_service.stop_ollama()


# Logging helpers

# Print one adapter runtime event into the shared console with the ASLM-Chat prefix.
def _print_runtime_event(message: str) -> None:
    print(f"[ASLM-Chat] {message}", flush=True)


# Return whether debug-or-higher Ollama adapter events should be printed.
def _is_debug_logging_enabled() -> bool:
    return settings.is_console_debug_enabled()


# Return whether trace-level adapter events should be printed.
def _is_trace_logging_enabled() -> bool:
    return settings.is_console_trace_enabled()


# Return a compact one-line preview of a value for adapter diagnostics: non-strings are rendered as
# JSON when possible, whitespace is collapsed and the text is cut to limit characters with an ellipsis.
def _preview_jsonish(value: Any, limit: int = 240) -> str:
    if isinstance(value, str):
        rendered = value
    else:
        try:
            import json

            rendered = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            # Values that JSON cannot encode fall back to their plain string form.
            rendered = str(value)

    rendered = " ".join(str(rendered or "").split())
    if len(rendered) <= limit:
        return rendered
    return f"{rendered[:max(0, limit - 3)].rstrip()}..."


# Return whether an option name can be forwarded to the current Ollama chat API, which is true for
# any non-empty name that is not listed in OLLAMA_UNSUPPORTED_OPTION_KEYS.
def is_supported_runtime_option_key(option_name: Any) -> bool:
    normalized = str(option_name or "").strip()
    return bool(normalized) and normalized not in OLLAMA_UNSUPPORTED_OPTION_KEYS


# Return a readable comma-separated list of the tool aliases requested by the model, or "none".
def _summarize_tool_names(tool_calls: list[dict[str, Any]]) -> str:
    names = [str(tool_call.get("name", "") or "").strip() for tool_call in tool_calls]
    names = [name for name in names if name]
    return ", ".join(names) if names else "none"


# Client and models

# Create an Ollama client for the managed local service on the configured port.
def get_client() -> ollama.Client:
    port = settings.get("ollama-service_port", 30002)
    host = f"http://127.0.0.1:{port}"

    # This client only ever talks to the managed loopback service. Do not let
    # HTTPX reinterpret a Windows system proxy as a route to local Ollama.
    return ollama.Client(host=host, trust_env=False)


# Run one Ollama SDK call and retry it a few times with a growing delay, giving the managed runtime
# time to answer while it warms up. The last error is raised once the attempts are exhausted.
def _call_with_runtime_retry(operation: Any, description: str) -> Any:
    last_error: Exception | None = None
    for attempt in range(1, RUNTIME_REQUEST_ATTEMPTS + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt >= RUNTIME_REQUEST_ATTEMPTS:
                break

            logger.debug(
                "[Ollama API] %s failed while runtime is warming up (attempt %s/%s): %s",
                description,
                attempt,
                RUNTIME_REQUEST_ATTEMPTS,
                exc,
            )
            time.sleep(RUNTIME_RETRY_DELAY_SECONDS * attempt)

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Ollama operation failed: {description}")


# Return the locally available Ollama models, or an empty list when listing fails. The SDK answer may
# be a dict or an object with a models attribute.
def get_models() -> list[dict[str, Any]]:
    client = get_client()
    try:
        response = _call_with_runtime_retry(client.list, "list models")
    except Exception as exc:
        logger.error("[Ollama API] Error listing models: %s", exc)
        return []

    if isinstance(response, dict):
        return response.get("models", [])

    return getattr(response, "models", []) or []


# Pull a model from Ollama and return the SDK result; with stream=True in kwargs it is a progress
# iterator. Failures are logged and re-raised.
def download_model(model_name: str, **kwargs: Any) -> Any:
    client = get_client()
    stream = kwargs.get("stream", False)

    try:
        return client.pull(model_name, stream=stream)
    except Exception as exc:
        logger.error("[Ollama API] Error downloading model %s: %s", model_name, exc)
        raise


# Return the metadata and Modelfile-style settings of an Ollama model. Failures are logged and re-raised.
def get_model_settings(model_name: str) -> Any:
    client = get_client()
    try:
        return _call_with_runtime_retry(lambda: client.show(model_name), f"show model {model_name}")
    except Exception as exc:
        logger.error("[Ollama API] Error fetching settings for %s: %s", model_name, exc)
        raise


# Payload normalization

# Return the first of the given field names found on a dict or an attribute object, or the default
# when none is present. SDK results can be either shape.
def _get_field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(value, dict) and name in value:
            return value.get(name)
        if hasattr(value, name):
            return getattr(value, name)

    return default


# Convert an Ollama tool-call payload into a {"name", "arguments"} dict, or None when it has no name.
# Missing arguments become an empty dict and non-dict arguments are wrapped as {"value": ...}.
def _normalize_tool_call(raw_call: Any) -> dict[str, Any] | None:
    raw_function = _get_field(raw_call, "function", default=raw_call)
    name = str(_get_field(raw_function, "name", default="") or "").strip()
    if not name:
        return None

    arguments = _get_field(raw_function, "arguments", default={})
    if arguments is None:
        arguments = {}
    if isinstance(arguments, str):
        arguments = {"value": arguments}
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}

    return {"name": name, "arguments": arguments}


# Merge streamed tool calls into the already collected ones, skipping entries that are already present.
def _merge_tool_calls(existing: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = list(existing)
    for item in incoming:
        if item not in merged:
            merged.append(item)

    return merged


# Convert an Ollama message object (or None) into a JSON-compatible assistant message dict with
# optional thinking text and function-style tool calls.
def _normalize_message(raw_message: Any) -> dict[str, Any]:
    if raw_message is None:
        return {"role": "assistant", "content": ""}

    role = str(_get_field(raw_message, "role", default="assistant") or "assistant")
    content = str(_get_field(raw_message, "content", default="") or "")
    thinking = str(_get_field(raw_message, "thinking", default="") or "")
    tool_calls = _get_field(raw_message, "tool_calls", "toolCalls", default=[]) or []

    normalized_message = {"role": role, "content": content}
    if thinking:
        normalized_message["thinking"] = thinking

    normalized_calls = []

    # Normalize incoming tool calls into the same function-call shape used by
    # the other adapters so the shared tool loop can stay provider-agnostic.
    for raw_call in tool_calls:
        tool_call = _normalize_tool_call(raw_call)
        if tool_call:
            normalized_calls.append(
                {
                    "function": {
                        "name": tool_call["name"],
                        "arguments": tool_call["arguments"],
                    }
                }
            )

    if normalized_calls:
        normalized_message["tool_calls"] = normalized_calls

    return normalized_message


# Request preparation

# Drop the options that the bundled Ollama runtime no longer accepts. Return the cleaned options and
# the names of the dropped keys; options that are not a dict are returned untouched.
def _sanitize_request_options(options: Any) -> tuple[Any, list[str]]:
    if not isinstance(options, dict):
        return options, []

    clean_options: dict[str, Any] = {}
    dropped_options: list[str] = []
    for key, value in options.items():
        normalized_key = str(key or "").strip()
        if not is_supported_runtime_option_key(normalized_key):
            dropped_options.append(normalized_key)
            continue
        clean_options[key] = value

    return clean_options, dropped_options


# Print one concise console summary of the removed unsupported options; each distinct set of names
# is reported only once per process.
def _report_dropped_options(dropped_options: list[str]) -> None:
    unique_dropped = tuple(sorted({str(key).strip() for key in dropped_options if str(key).strip()}))
    if not unique_dropped:
        return

    with _unsupported_options_lock:
        if unique_dropped in _reported_unsupported_option_sets:
            return
        _reported_unsupported_option_sets.add(unique_dropped)

    _print_runtime_event(
        "Ollama request sanitized: dropped unsupported options "
        f"[{', '.join(unique_dropped)}]."
    )


# Build the keyword arguments for client.chat from the request kwargs: remove adapter-only keys,
# sanitize the nested options, hoist the options that belong at the top request level and fold
# think/think_level into the single think value. Return the call kwargs and the dropped option names.
def _prepare_chat_kwargs_with_metadata(kwargs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    # Strip the keys that only drive the adapter.
    call_kwargs = {key: value for key, value in kwargs.items() if key not in OLLAMA_INTERNAL_CHAT_KEYS}
    options = call_kwargs.get("options")
    dropped_options: list[str] = []

    if isinstance(options, dict):
        # Older runtimes reject some bundled option keys, so sanitize first and
        # then hoist the subset that belongs at the top request level.
        options, dropped_options = _sanitize_request_options(options)
        call_kwargs["options"] = options
        for key in OLLAMA_TOP_LEVEL_CHAT_KEYS:
            if key in options:
                call_kwargs[key] = options.pop(key)

        if not options:
            call_kwargs.pop("options", None)

    # Resolve the thinking mode.
    think = kwargs.get("think")
    think_level = kwargs.get("think_level")
    if think is not None:
        # Ollama accepts either a boolean-like think flag or a coarse level,
        # so map the pair into the most expressive single value available.
        if bool(think) and think_level in {"low", "medium", "high"}:
            call_kwargs["think"] = think_level
        else:
            call_kwargs["think"] = think
    elif think_level in {"low", "medium", "high"}:
        call_kwargs["think"] = think_level

    return call_kwargs, dropped_options


# Compatibility wrapper around _prepare_chat_kwargs_with_metadata that returns only the call kwargs
# and discards the list of dropped options.
def _prepare_chat_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    call_kwargs, _ = _prepare_chat_kwargs_with_metadata(kwargs)
    return call_kwargs


# Tool-aware streaming

# Build a tool message that can be fed back into Ollama. When content is an image dict (it has
# _image_b64), the message uses Ollama's images field so the model receives the image visually.
# With a tool_event, the alias, server and display details are attached for the UI.
def _build_tool_message(
    tool_name: str,
    content: str | dict[str, Any],
    tool_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if isinstance(content, dict) and "_image_b64" in content:
        # Image result: the text part only names the image, the pixels travel in the images field.
        image_path = content.get("_path", "image")
        _model_content, tool_extras = tool_registry.split_tool_result_payload(content)
        payload = {
            "role": "tool",
            "name": tool_name,
            "tool_name": tool_name,
            "content": f"[Image: {image_path}]",
            "images": [content["_image_b64"]],
        }
        payload.update(tool_extras)
    else:
        # Text result: the model-facing content is split from the extras kept for the UI.
        model_content, tool_extras = tool_registry.split_tool_result_payload(content)
        payload = {
            "role": "tool",
            "name": tool_name,
            "tool_name": tool_name,
            "content": model_content,
        }
        payload.update(tool_extras)

    # Attach the alias, server and display details of the invocation, with the tool name as fallback.
    if tool_event:
        payload.update(
            {
                "alias": tool_event.get("alias") or tool_name,
                "server_id": tool_event.get("server_id") or "",
                "server_name": tool_event.get("server_name") or "",
                "tool_id": tool_event.get("tool_id") or tool_name,
                "tool_display_name": tool_event.get("tool_name") or tool_name,
                "arguments": tool_event.get("arguments") or {},
            }
        )
        if "tool_ui" not in payload and isinstance(tool_event.get("tool_ui"), dict):
            payload["tool_ui"] = dict(tool_event["tool_ui"])

    return payload


# Serialize one tool invocation so the UI can render it during streaming.
def _build_tool_event(tool_lookup: dict[str, dict[str, Any]], tool_call: dict[str, Any]) -> dict[str, Any]:
    return tool_registry.build_tool_event(tool_lookup, tool_call)


# Stream one Ollama round as a generator: yield {"message": ...} chunks for the UI while the model
# answers and return the assembled assistant message (content, thinking and tool calls) as the
# generator result. When no chunk had visible text, the final message is yielded once at the end.
def _stream_round(
    client: ollama.Client,
    model_name: str,
    conversation: list[dict[str, Any]],
    base_kwargs: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
):
    assistant_message: dict[str, Any] = {"role": "assistant", "content": ""}
    thinking_buffer = ""
    tool_calls: list[dict[str, Any]] = []
    yielded_visible_chunk = False

    stream_kwargs = dict(base_kwargs)
    if tools:
        stream_kwargs["tools"] = tools

    # Keep a full assistant message in parallel with streamed chunks so the
    # caller receives both incremental UI updates and one final transcript.
    for raw_chunk in client.chat(model=model_name, messages=conversation, stream=True, **stream_kwargs):
        if _abort_event.is_set():
            break
        normalized_chunk = _normalize_message(_get_field(raw_chunk, "message", default=raw_chunk))
        thinking_part = normalized_chunk.get("thinking", "") or ""
        content_part = normalized_chunk.get("content", "") or ""
        incoming_tool_calls = [
            _normalize_tool_call(raw_call)
            for raw_call in normalized_chunk.get("tool_calls", [])
        ]
        incoming_tool_calls = [tool_call for tool_call in incoming_tool_calls if tool_call]

        # Accumulate the chunk into the final message and merge any tool calls it carries.
        if thinking_part:
            thinking_buffer += thinking_part
        if content_part:
            assistant_message["content"] += content_part
        if incoming_tool_calls:
            tool_calls = _merge_tool_calls(tool_calls, incoming_tool_calls)

        # Forward only chunks that carry visible text or thinking to the UI.
        if thinking_part or content_part:
            yielded_visible_chunk = True
            chunk_payload = {"role": "assistant", "content": content_part}
            if thinking_part:
                chunk_payload["thinking"] = thinking_part
            yield {"message": chunk_payload}

    # Finish the assembled assistant message with the collected thinking and tool calls.
    if thinking_buffer:
        assistant_message["thinking"] = thinking_buffer
    if tool_calls:
        assistant_message["tool_calls"] = [
            {"function": {"name": call["name"], "arguments": call["arguments"]}}
            for call in tool_calls
        ]

    if not yielded_visible_chunk and (assistant_message.get("thinking") or assistant_message.get("content")):
        yield {"message": assistant_message}

    return assistant_message


# Re-yield every chunk of a _stream_round generator and return its final assistant message, or an
# empty assistant message when the generator returned nothing.
def _yield_stream_round(round_stream):
    while True:
        try:
            yield next(round_stream)
        except StopIteration as stop:
            return stop.value or {"role": "assistant", "content": ""}


# Resolve local tools through Ollama tool calling with streaming output. The generator yields UI chunks
# ("message"), the canonical assistant messages ("transcript_message"), "tool_events" and
# "tool_result" items. Each round streams one model answer, runs the tool calls it requested (with
# cooldown, duplicate and quota guards) and feeds the results back, until the model answers without
# tools, the round limit is reached or the loop is cancelled. Repeated blocked or cancelled results
# force one final tool-free answer.
def _run_tool_loop(
    client: ollama.Client,
    model_name: str,
    messages: list[dict[str, Any]],
    call_kwargs: dict[str, Any],
    tool_server_ids: list[str],
    tool_context: dict[str, Any],
):
    # Build the tool definitions for the selected servers; without tools this is a plain streamed round.
    tools, tool_lookup = tool_registry.build_ollama_tools(
        tool_server_ids,
        engine="ollama-service",
        model_name=model_name,
        tool_source_map=tool_context.get("tool_source_map") if isinstance(tool_context, dict) else None,
        allowed_tool_aliases=tool_context.get("allowed_tool_aliases") if isinstance(tool_context, dict) else None,
        instant_mode=bool(tool_context.get("instant_mode")) if isinstance(tool_context, dict) else False,
        instant_search_batch_size=tool_context.get("instant_search_batch_size") if isinstance(tool_context, dict) else None,
    )
    base_kwargs = {key: value for key, value in call_kwargs.items() if key != "stream"}

    if not tools:
        yield from _yield_stream_round(_stream_round(client, model_name, messages, base_kwargs))
        return

    # Report the prepared loop in the debug console.
    if _is_debug_logging_enabled():
        _print_runtime_event(
            "Tool loop prepared: "
            f"engine=ollama-service, "
            f"model={model_name}, "
            f"servers={tool_server_ids}, "
            f"tools={len(tools)}"
        )
        if _is_trace_logging_enabled():
            _print_runtime_event(f"Tool loop base options: {_preview_jsonish(base_kwargs, limit=320)}")

    # Loop state: a copy of the conversation extended with the available-tools prompt, per-generation
    # tool guards (quota counters, seen call signatures, blocked-result streak) and the tracking of
    # servers whose tools the request requires, with the number of reminders still allowed.
    conversation = [dict(message) for message in messages]
    conversation = tool_registry.with_available_tools_prompt(conversation, tool_lookup)
    tool_quota_counters: dict[str, int] = {}
    seen_tool_signatures: set[str] = set()
    consecutive_blocked_tool_results = 0
    required_tool_server_ids = tool_registry.required_tool_server_ids_from_context(tool_context, tool_lookup)
    satisfied_tool_server_ids: set[str] = set()
    required_nudges_left = tool_registry.MAX_REQUIRED_TOOL_NUDGES if required_tool_server_ids else 0

    # Run model rounds until the model stops calling tools or the round limit is reached.
    max_tool_rounds = tool_registry.tool_round_limit_from_context(tool_context, MAX_TOOL_ROUNDS)
    for round_index in range(max_tool_rounds):
        round_started_at = time.perf_counter()
        assistant_message = yield from _yield_stream_round(
            _stream_round(client, model_name, conversation, base_kwargs, tools=tools)
        )
        conversation.append(assistant_message)

        # Tool calls may arrive in parts, so normalize the final payload again.
        tool_calls = [
            _normalize_tool_call(raw_call)
            for raw_call in assistant_message.get("tool_calls", [])
        ]
        tool_calls = [tool_call for tool_call in tool_calls if tool_call]
        tool_calls = tool_registry.prepare_tool_calls(tool_lookup, tool_calls)
        tool_calls = tool_registry.limit_tool_calls_from_context(tool_context, tool_calls, tool_lookup)

        # Store the assistant message with its canonical tool calls in the conversation and the transcript.
        canonical_assistant_message = tool_registry.canonicalize_transcript_tool_calls(
            assistant_message, tool_calls
        )
        conversation[-1] = canonical_assistant_message
        yield {"transcript_message": canonical_assistant_message}

        # No tool calls: the model answered. Remind it of required tools that were not used yet,
        # otherwise finish the loop.
        if not tool_calls:
            unmet_tool_server_ids = [
                server_id for server_id in required_tool_server_ids
                if server_id not in satisfied_tool_server_ids
            ]
            if unmet_tool_server_ids and required_nudges_left > 0:
                required_nudges_left -= 1
                conversation.append(
                    {
                        "role": "user",
                        "content": tool_registry.required_tools_reminder_prompt(
                            unmet_tool_server_ids, tool_lookup
                        ),
                    }
                )
                continue
            if _is_debug_logging_enabled():
                _print_runtime_event(
                    "Tool loop completed: "
                    f"model={model_name}, "
                    f"rounds={round_index + 1}, "
                    f"took={time.perf_counter() - round_started_at:.2f}s, "
                    "tool_calls=0"
                )
            return

        # Remember which required tool servers this round touches.
        for tool_call in tool_calls:
            satisfied_server_id = tool_registry.tool_server_id_for_alias(tool_lookup, tool_call["name"])
            if satisfied_server_id:
                satisfied_tool_server_ids.add(satisfied_server_id)

        # Report the requested tool calls in the debug console.
        if _is_debug_logging_enabled():
            _print_runtime_event(
                "Tool round requested: "
                f"model={model_name}, "
                f"round={round_index + 1}, "
                f"count={len(tool_calls)}, "
                f"aliases={_summarize_tool_names(tool_calls)}"
            )
            if _is_trace_logging_enabled():
                _print_runtime_event(
                    f"Tool round payload: {_preview_jsonish(tool_calls, limit=420)}"
                )

        # Announce the round's tool events to the UI; each alias gets its call index so that several
        # calls of the same tool stay distinguishable.
        tool_events = [_build_tool_event(tool_lookup, tool_call) for tool_call in tool_calls]
        for _i, _ev in enumerate(tool_events):
            _ev["alias"] = f"{_ev['alias']}__{_i}"
        yield {"tool_events": tool_events}

        # Execute the round's tool calls one by one.
        cancelled_tool_result_seen = False
        for tool_call_index, (tool_call, tool_event) in enumerate(zip(tool_calls, tool_events), start=1):
            if _is_debug_logging_enabled():
                _print_runtime_event(
                    "Tool dispatched: "
                    f"round={round_index + 1}, "
                    f"index={tool_call_index}, "
                    f"alias={tool_event.get('alias')}, "
                    f"server={tool_event.get('server_id')}, "
                    f"tool={tool_event.get('tool_id')}"
                )

            # Build the per-call context and log the request.
            call_context = dict(tool_context or {})
            call_context.update(
                {
                    "engine": "ollama-service",
                    "model_name": model_name,
                    "tool_alias": tool_call["name"],
                    "tool_arguments": tool_call.get("arguments") or {},
                    "raw_tool_arguments": tool_call.get("raw_arguments") or {},
                    "tool_call_index": tool_call_index,
                    "tool_round_index": round_index + 1,
                }
            )
            tool_registry.log_search_tool_io(
                "request",
                tool_event,
                arguments=tool_call.get("arguments") or {},
                context=call_context,
            )

            # Resolve the result: a preflight error, a cooldown, a duplicate-call or a quota error
            # short-circuits the call, otherwise the tool runs and its cooldown is recorded.
            preflight_error = tool_call.get("preflight_error_result")
            if preflight_error is not None:
                tool_result = preflight_error
            else:
                tool_cooldown_error = tool_registry.consume_tool_cooldown(
                    tool_event,
                    tool_call.get("arguments") or {},
                    context=call_context,
                )
                if tool_cooldown_error is not None:
                    tool_result = tool_cooldown_error
                else:
                    duplicate_error = tool_registry.consume_duplicate_tool_call(
                        tool_event,
                        tool_call.get("arguments") or {},
                        seen_tool_signatures,
                    )
                    if duplicate_error is not None:
                        tool_result = duplicate_error
                    else:
                        quota_error = tool_registry.consume_tool_quota(
                            tool_event,
                            tool_quota_counters,
                            arguments=tool_call.get("arguments") or {},
                        )
                        if quota_error is not None:
                            tool_result = quota_error
                        else:
                            tool_result = yield from tool_registry.stream_ollama_tool(
                                tool_lookup,
                                tool_call["name"],
                                tool_call.get("arguments") or {},
                                context=call_context,
                            )
                            tool_registry.remember_tool_cooldown(
                                tool_event,
                                tool_call.get("arguments") or {},
                                context=call_context,
                            )

            # Track cancelled and blocked results; they decide whether a final answer is forced.
            cancelled_tool_result_seen = (
                cancelled_tool_result_seen
                or tool_registry.is_tool_execution_cancelled(tool_result)
            )
            if tool_registry.is_blocking_tool_result(tool_result):
                consecutive_blocked_tool_results += 1
            else:
                consecutive_blocked_tool_results = 0
            tool_message = _build_tool_message(tool_call["name"], tool_result, tool_event)

            # Feed only model-supported fields back into Ollama; UI metadata
            # stays on the streamed tool event below.
            conversation_tool_message = {
                key: value
                for key, value in tool_message.items()
                if key not in {"structured_content", "tool_ui", "images"}
            }
            conversation.append(conversation_tool_message)
            tool_registry.log_search_tool_io(
                "response",
                tool_event,
                arguments=tool_call.get("arguments") or {},
                context=call_context,
                result=conversation_tool_message,
            )

            # Strip the model-facing images field; UI-only preview metadata stays
            # on tool_ui/structured_content for rendering the inspected image.
            ui_tool_message = {k: v for k, v in tool_message.items() if k != "images"}
            if "images" in tool_message:
                ui_tool_message["content"] = tool_message.get("content", "[image]")
            if _is_debug_logging_enabled():
                result_content = ui_tool_message.get("content", "")
                result_type = "image" if "images" in tool_message else "text"
                _print_runtime_event(
                    "Tool result forwarded: "
                    f"round={round_index + 1}, "
                    f"index={tool_call_index}, "
                    f"alias={tool_event.get('alias')}, "
                    f"type={result_type}, "
                    f"chars={len(str(result_content or ''))}"
                )
            yield {"tool_result": ui_tool_message}

        # Stop on user cancellation; after a cancelled tool or two blocked results in a row, ask
        # the model for one final answer without tools instead of looping further.
        if cancelled_tool_result_seen and _abort_event.is_set():
            return
        if cancelled_tool_result_seen or consecutive_blocked_tool_results >= 2:
            conversation.append(
                {
                    "role": "user",
                    "content": (
                        tool_registry.forced_final_prompt_after_tool_cancellation()
                        if cancelled_tool_result_seen
                        else tool_registry.forced_final_prompt_after_tool_blocks()
                    ),
                }
            )
            assistant_message = yield from _yield_stream_round(
                _stream_round(client, model_name, conversation, base_kwargs)
            )
            yield {"transcript_message": assistant_message}
            return

        # Compact the conversation when it grows too large before the next round.
        conversation = tool_registry.maybe_compact_tool_conversation(
            tool_context, conversation, provider_format="standard"
        )

    # The round limit was exhausted without a final answer.
    yield {"message": {"content": "[Error during generation: tool loop exceeded the safety limit.]"}}


# Generation

# Yield the chunks of an iterator until a cancellation request arrives.
def _iter_with_abort(iterator):
    for chunk in iterator:
        if _abort_event.is_set():
            break
        yield chunk


# Generate a chat response through Ollama and return a chunk iterator. Legacy tool_id/tool_server_id
# inputs are folded into one tool server id list; with tools selected the local tool loop streams the
# answer, otherwise the client stream is returned directly with cancellation support. Errors are
# logged and re-raised.
def generate(model_name: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
    client = get_client()

    # Normalize legacy single-id tool inputs into one shared list format.
    raw_ids = kwargs.pop("tool_server_ids", None) or kwargs.pop("tool_server_id", None) or kwargs.pop("tool_id", None)
    if isinstance(raw_ids, str):
        tool_server_ids = [raw_ids] if raw_ids.strip() else []
    elif isinstance(raw_ids, list):
        tool_server_ids = [str(s) for s in raw_ids if str(s).strip()]
    else:
        tool_server_ids = []
    tool_context = dict(kwargs.pop("tool_context", {}) or {})

    try:
        # Reset cancellation left over from a previous generation and build the chat call kwargs.
        _abort_event.clear()
        call_kwargs, dropped_options = _prepare_chat_kwargs_with_metadata(kwargs)
        _report_dropped_options(dropped_options)

        # Tool-enabled flows stay inside the local tool loop; plain chats can
        # stream directly from the Ollama client iterator.
        if tool_server_ids:
            return _run_tool_loop(client, model_name, messages, call_kwargs, tool_server_ids, tool_context)

        return _iter_with_abort(client.chat(model=model_name, messages=messages, **call_kwargs))
    except Exception as exc:
        logger.error("[Ollama API] Error generating response from %s: %s", model_name, exc)
        raise
