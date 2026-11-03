# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import base64
import hashlib
import json
import logging
import re
import threading
import time
import unicodedata
from urllib.parse import urlsplit, urlunsplit
from typing import Any

from API import mcp as tool_registry
from Settings import settings

logger = logging.getLogger(__name__)

# Cancellation flag of the adapter, set by abort_generation().
_abort_event = threading.Event()

# Runtime cache of capability facts learned per (endpoint scope, model); the lock guards it across threads.
_model_capability_cache_lock = threading.Lock()
_model_capability_cache: dict[tuple[str, str], dict[str, Any]] = {}

# Runtime cache of model availability per (endpoint scope, API key hash, model); entries expire after
# MODEL_AVAILABILITY_TTL_SECONDS and the lock guards the cache across threads.
_model_availability_cache_lock = threading.Lock()
_model_availability_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

# Default cap on tool-calling rounds per generation; the tool context may override it.
MAX_TOOL_ROUNDS = 100

# How long, in seconds, a cached availability result stays valid.
MODEL_AVAILABILITY_TTL_SECONDS = 15 * 60

# Opening and closing tags that models use to wrap inline reasoning inside the answer text.
REASONING_TAG_PAIRS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
)

# Chat-template control tokens (for example <|start|> or <|im_end|>) that some models leak into their
# text; _sanitize_generated_text removes them before the text reaches the chat UI.
CONTROL_TOKEN_PATTERNS = (
    re.compile(
        r"<\|start\|>\s*(?:assistant|user|system)?\s*(?:<\|channel\|>\s*(?:final|analysis|commentary))?\s*(?:<\|message\|>)?",
        flags=re.IGNORECASE,
    ),
    re.compile(r"<\|start\|>", flags=re.IGNORECASE),
    re.compile(r"<\|channel\|>\s*(?:final|analysis|commentary)", flags=re.IGNORECASE),
    re.compile(r"<\|message\|>", flags=re.IGNORECASE),
    re.compile(r"<\|return\|>", flags=re.IGNORECASE),
    re.compile(r"<\|startoftext\|>", flags=re.IGNORECASE),
    re.compile(r"<\|im_(?:start|end)\|>", flags=re.IGNORECASE),
    re.compile(r"<\|(?:assistant|user|system|endoftext)\|>", flags=re.IGNORECASE),
)

# Capability names (normalized keys and tokens) that Gemini model metadata uses to advertise tool calling,
# vision, audio input, video input and reasoning support.
TOOL_CAPABILITY_NAMES = {
    "tool",
    "tools",
    "tool_calling",
    "tool_use",
    "function_calling",
    "functions",
    "supports_tools",
    "supports_tool_calling",
}
VISION_CAPABILITY_NAMES = {
    "vision",
    "multimodal",
    "image",
    "images",
    "video",
    "audio",
    "image_input",
    "multimodal_input",
    "supports_vision",
}
AUDIO_INPUT_CAPABILITY_NAMES = {"audio", "audio_input", "input_audio", "supports_audio_input"}
VIDEO_INPUT_CAPABILITY_NAMES = {"video", "video_input", "input_video", "supports_video_input"}
REASONING_CAPABILITY_NAMES = {
    "think",
    "thinking",
    "thought",
    "thoughts",
    "reasoning",
    "supports_reasoning",
    "reasoning_enabled",
}

# Thinking levels offered to the UI for models that support selectable reasoning effort.
THINK_LEVEL_OPTIONS = ["minimal", "low", "medium", "high"]

# Generic request option names mapped to the Google GenAI config field they correspond to.
GOOGLE_OPTION_ALIASES = {
    "max_tokens": "max_output_tokens",
    "max_completion_tokens": "max_output_tokens",
    "n": "candidate_count",
    "stop": "stop_sequences",
}

# Request options that are passed through to the generate-content config.
GOOGLE_CONFIG_KEYS = {
    "audio_timestamp",
    "cached_content",
    "candidate_count",
    "frequency_penalty",
    "labels",
    "logprobs",
    "max_output_tokens",
    "media_resolution",
    "presence_penalty",
    "response_json_schema",
    "response_logprobs",
    "response_mime_type",
    "response_modalities",
    "response_schema",
    "routing_config",
    "safety_settings",
    "seed",
    "service_tier",
    "speech_config",
    "stop_sequences",
    "temperature",
    "top_k",
    "top_p",
}

# Request options that belong to the nested thinking config instead of the top-level config.
THINKING_CONFIG_KEYS = {"include_thoughts", "thinking_budget", "thinking_level"}


# Runtime state

# Return a shallow copy of a cached payload in which lists and dicts are copied one level deep, so
# callers cannot mutate the cache entry; anything that is not a dict yields an empty dict.
def _clone_cache_payload(value: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}

    cloned: dict[str, Any] = {}
    for key, child in value.items():
        if isinstance(child, list):
            cloned[key] = list(child)
        elif isinstance(child, dict):
            cloned[key] = dict(child)
        else:
            cloned[key] = child
    return cloned


# Signal the active Google GenAI generation to stop.
def abort_generation() -> None:
    _abort_event.set()


# Endpoint and SDK

# Import the Google GenAI SDK lazily so the app can boot without it; raise ImportError with an
# actionable message when the google-genai package is missing.
def _get_sdk():
    try:
        from google import genai
    except ImportError as exc:
        raise ImportError("The 'google-genai' package is required for Google GenAI support.") from exc

    return genai


# Split a configured Gemini endpoint into (base_url, api_version): a trailing version segment such as
# /v1beta is cut off and returned separately, so the SDK does not prefix the version twice.
def _normalize_google_base_url(base_url: str) -> tuple[str, str | None]:
    raw_base_url = str(base_url or "").strip()
    if not raw_base_url:
        return "", None

    parsed = urlsplit(raw_base_url)
    trimmed_path = parsed.path.rstrip("/")
    match = re.search(r"/(v1(?:alpha|beta(?:\d+)?)?)$", trimmed_path, flags=re.IGNORECASE)
    if not match:
        return raw_base_url, None

    api_version = match.group(1)
    normalized_path = trimmed_path[: -len(match.group(0))].rstrip("/")
    normalized_base = urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            f"{normalized_path}/" if normalized_path else "/",
            parsed.query,
            parsed.fragment,
        )
    )
    return normalized_base, api_version


# Return the normalized endpoint scope (base URL plus API version) used to key the Gemini runtime caches.
def _get_runtime_scope() -> str:
    configured_base_url = settings.get_engine_url("google-genai")
    normalized_base_url, api_version = _normalize_google_base_url(configured_base_url)
    endpoint = str(normalized_base_url or configured_base_url or "").strip().rstrip("/")
    version = str(api_version or "").strip().lower()
    return f"{endpoint}|{version}"


# Return a stable non-reversible SHA-256 hash of the active Gemini API key, so cache keys can be
# scoped to the key without holding the key itself.
def _get_api_key_hash() -> str:
    api_key = str(settings.get_google_genai_api_key() or "")
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


# Runtime caches

# Return the capability-cache key for one Gemini model: the endpoint scope plus the normalized model id.
def _get_model_capability_cache_key(model_name: str) -> tuple[str, str]:
    return (_get_runtime_scope(), _normalize_model_identifier(model_name))


# Return the availability-cache key for one Gemini model: endpoint scope, API key hash and normalized
# model id.
def _get_model_availability_cache_key(model_name: str) -> tuple[str, str, str]:
    return (_get_runtime_scope(), _get_api_key_hash(), _normalize_model_identifier(model_name))


# Return cached endpoint-scoped capabilities for one Gemini model.
def _get_cached_model_capabilities(model_name: str) -> dict[str, Any]:
    with _model_capability_cache_lock:
        return _clone_cache_payload(_model_capability_cache.get(_get_model_capability_cache_key(model_name)))


# Merge new endpoint-scoped capability facts into the runtime cache and return a copy of the merged
# entry. Updates whose value is None are skipped so unknown facts never overwrite known ones.
def _update_cached_model_capabilities(model_name: str, **updates: Any) -> dict[str, Any]:
    cleaned_updates = {
        str(key): (list(value) if isinstance(value, list) else value)
        for key, value in updates.items()
        if value is not None
    }
    cache_key = _get_model_capability_cache_key(model_name)
    with _model_capability_cache_lock:
        entry = _clone_cache_payload(_model_capability_cache.get(cache_key))
        entry.update(cleaned_updates)
        _model_capability_cache[cache_key] = entry
        return _clone_cache_payload(entry)


# Return key-scoped availability info when the cache entry is still fresh; an expired entry is evicted
# and None is returned.
def _get_cached_model_availability(model_name: str) -> dict[str, Any] | None:
    cache_key = _get_model_availability_cache_key(model_name)
    now = time.monotonic()
    with _model_availability_cache_lock:
        entry = _model_availability_cache.get(cache_key)
        if not isinstance(entry, dict):
            return None
        expires_at = float(entry.get("expires_at", 0) or 0)
        if expires_at <= now:
            _model_availability_cache.pop(cache_key, None)
            return None
        return _clone_cache_payload(entry)


# Cache whether one Gemini model is usable for the current key and endpoint, with the reason, until
# MODEL_AVAILABILITY_TTL_SECONDS has passed. Return a copy of the stored entry.
def _set_cached_model_availability(model_name: str, available: bool, reason: str) -> dict[str, Any]:
    cache_key = _get_model_availability_cache_key(model_name)
    entry = {
        "available": bool(available),
        "reason": str(reason or ""),
        "expires_at": time.monotonic() + MODEL_AVAILABILITY_TTL_SECONDS,
    }
    with _model_availability_cache_lock:
        _model_availability_cache[cache_key] = entry
    return _clone_cache_payload(entry)


# Clear Gemini runtime caches. Intended for tests and debug flows.
def _reset_runtime_caches() -> None:
    with _model_capability_cache_lock:
        _model_capability_cache.clear()
    with _model_availability_cache_lock:
        _model_availability_cache.clear()


# Client lifecycle

# Close a client instance, ignoring clients whose SDK version has no close method or fails to close.
def _close_client(client: Any) -> None:
    try:
        client.close()
    except Exception:
        # Closing is best-effort cleanup; an SDK without close() or a failed close is harmless.
        pass


# Create a fresh Google GenAI client for the configured Gemini API endpoint. The configured ASLM
# endpoint stays authoritative, and the API key is passed only when one is set.
def _get_client():
    genai = _get_sdk()
    api_key = settings.get_google_genai_api_key()
    configured_base_url = settings.get_engine_url("google-genai")
    base_url, api_version = _normalize_google_base_url(configured_base_url)

    client_kwargs: dict[str, Any] = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    if base_url:
        # Keep the configured ASLM endpoint authoritative so gateways can
        # expose either a root host or a versioned path without double-prefixes.
        http_options: dict[str, Any] = {"base_url": base_url}
        if api_version:
            http_options["api_version"] = api_version
        client_kwargs["http_options"] = http_options

    return genai.Client(**client_kwargs)


# Value conversion

# Return a lower-case identifier for a key or token in which every run of characters other than
# letters and digits becomes one underscore.
def _normalize_key_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


# Return a predictable boolean from a JSON-like value: None gives the default, and strings such as
# "true", "yes", "enabled" or "off" are recognized; any other value uses plain truthiness.
def _bool_from_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on", "enabled", "supported"}:
            return True
        if normalized in {"false", "0", "no", "off", "disabled", "unsupported"}:
            return False
    return bool(value)


# Convert a scalar into a positive integer, or None when it is not numeric, not positive or a bool.
def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if numeric_value > 0 else None


# Convert a string or a collection into a list of stripped, non-empty strings; duplicates are removed
# case-insensitively and the first spelling is kept.
def _coerce_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else []
    if isinstance(value, (list, tuple, set)):
        items = [str(item).strip() for item in value if str(item).strip()]
        seen: set[str] = set()
        normalized_items: list[str] = []
        for item in items:
            token = item.lower()
            if token in seen:
                continue
            seen.add(token)
            normalized_items.append(item)
        return normalized_items
    return []


# Return a binary payload as a base64 string for JSON-safe storage; strings are assumed to be base64
# already and are returned stripped, and anything else gives None.
def _encode_binary_payload(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytearray):
        value = bytes(value)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


# Decode a stored base64 payload back into raw bytes; bytes pass through unchanged. Missing padding is
# repaired on a second attempt, and undecodable input gives None.
def _decode_binary_payload(value: Any) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, bytes):
        return value

    payload = str(value or "").strip()
    if not payload:
        return None

    try:
        return base64.b64decode(payload, validate=True)
    except Exception:
        # Strict decoding failed; retry leniently after restoring the stripped padding.
        try:
            padding = "=" * (-len(payload) % 4)
            return base64.b64decode(f"{payload}{padding}")
        except Exception:
            return None


# Remove the chat-template control tokens that must never reach the chat UI.
def _sanitize_generated_text(text: str) -> str:
    cleaned = str(text or "")
    for pattern in CONTROL_TOKEN_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


# Convert SDK objects into plain Python containers (dicts, lists and scalars) that can be serialized
# to JSON. Bytes become base64 strings, and objects are flattened through their first working
# serializer (to_dict, model_dump or dict), then through their public attributes, then str().
def _to_plain_data(value: Any) -> Any:
    # Scalars pass through and bytes are base64-encoded.
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return _encode_binary_payload(value)

    # Containers are converted element by element.
    if isinstance(value, dict):
        return {str(key): _to_plain_data(child) for key, child in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(child) for child in value]

    # SDK objects: use the first serializer method that works.
    for attr_name in ("to_dict", "model_dump", "dict"):
        serializer = getattr(value, attr_name, None)
        if not callable(serializer):
            continue
        try:
            if attr_name == "model_dump":
                serialized = serializer(mode="python", exclude_none=False)
            else:
                serialized = serializer()
        except TypeError:
            try:
                serialized = serializer()
            except Exception:
                # The serializer also fails without arguments; try the next one.
                continue
        except Exception:
            # This serializer is unusable for the object; try the next one.
            continue
        return _to_plain_data(serialized)

    # Last resorts: public instance attributes, then the string form.
    if hasattr(value, "__dict__"):
        return {
            str(key): _to_plain_data(child)
            for key, child in vars(value).items()
            if not str(key).startswith("_")
        }

    return str(value)


# Model identifiers

# Return a lower-case model identifier without any "models/" prefix, so ids from different API
# responses can be compared.
def _normalize_model_identifier(value: Any) -> str:
    normalized = str(value or "").strip().strip("/").lower()
    if "/models/" in normalized:
        normalized = normalized.rsplit("/models/", 1)[-1]
    if normalized.startswith("models/"):
        normalized = normalized.split("/", 1)[1]
    return normalized


# Return whether two Gemini model identifiers likely refer to the same model: they are equal after
# normalization or share the same last path segment.
def _model_identifiers_match(expected: Any, actual: Any) -> bool:
    normalized_expected = _normalize_model_identifier(expected)
    normalized_actual = _normalize_model_identifier(actual)
    if not normalized_expected or not normalized_actual:
        return False
    if normalized_expected == normalized_actual:
        return True

    expected_tail = normalized_expected.rsplit("/", 1)[-1]
    actual_tail = normalized_actual.rsplit("/", 1)[-1]
    return bool(expected_tail) and expected_tail == actual_tail


# Return the UI-facing model identifier for one Gemini model payload, taken from the first of the
# name, id or model fields that is set.
def _extract_model_name(payload: dict[str, Any]) -> str:
    for key in ("name", "id", "model"):
        value = payload.get(key)
        if value:
            normalized = _normalize_model_identifier(value)
            if normalized:
                return normalized
            return str(value)
    return ""


# Model payload lookup

# Return the plain payloads of the available base and tuned Gemini models, without duplicates. The
# list call is tried with different query configs because SDK versions accept different ones; the
# last error is raised only when no attempt produced a model.
def _iter_model_payloads(client: Any) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    last_error: Exception | None = None
    query_attempts = (
        {"query_base": True},
        {"query_base": False},
        None,
    )

    # Query base models only, then tuned models only, then fall back to the SDK default listing.
    for query_config in query_attempts:
        try:
            response = client.models.list(config=query_config) if query_config is not None else client.models.list()
            entries = list(response)
        except Exception as exc:
            last_error = exc
            continue

        # Keep each distinct model once, in the order it was returned.
        for entry in entries:
            payload = _to_plain_data(entry)
            if not isinstance(payload, dict):
                continue
            model_name = _extract_model_name(payload)
            if not model_name or model_name in seen_names:
                continue
            seen_names.add(model_name)
            payloads.append(payload)

    if not payloads and last_error is not None:
        raise last_error

    return payloads


# Return the plain metadata payload of one Gemini model. The direct lookup is tried with several
# spellings of the name, then the full model list is searched; when everything fails the last error
# is raised, or a minimal payload with just the name is returned if no call raised.
def _get_model_payload(client: Any, model_name: str) -> dict[str, Any]:
    requested_name = _normalize_model_identifier(model_name)
    last_error: Exception | None = None
    attempted_names: list[str] = []

    # Direct lookup by the given, normalized and models/-prefixed name.
    for candidate in (model_name, requested_name, f"models/{requested_name}"):
        candidate_name = str(candidate or "").strip()
        if not candidate_name or candidate_name in attempted_names:
            continue
        attempted_names.append(candidate_name)
        try:
            payload = _to_plain_data(client.models.get(model=candidate_name))
        except Exception as exc:
            last_error = exc
            continue
        if isinstance(payload, dict) and payload:
            return payload

    # Fall back to searching the model list for a matching identifier.
    for payload in _iter_model_payloads(client):
        if _model_identifiers_match(requested_name, _extract_model_name(payload)):
            return payload

    if last_error is not None:
        raise last_error

    return {"name": requested_name or model_name}


# Capability detection

# Split a free-form value into normalized capability tokens: the whole normalized name plus each
# underscore-separated part.
def _tokenize_string(value: Any) -> set[str]:
    normalized = _normalize_key_name(value)
    if not normalized:
        return set()
    return {normalized, *[part for part in normalized.split("_") if part]}


# Return the normalized tokens found in the keys and string values of a nested model payload.
def _collect_capability_tokens(value: Any) -> set[str]:
    if value is None:
        return set()

    if isinstance(value, dict):
        tokens: set[str] = set()
        for key, child in value.items():
            tokens.update(_tokenize_string(key))
            tokens.update(_collect_capability_tokens(child))
        return tokens

    if isinstance(value, (list, tuple, set)):
        tokens: set[str] = set()
        for child in value:
            tokens.update(_collect_capability_tokens(child))
        return tokens

    if isinstance(value, str):
        return _tokenize_string(value)

    return set()


# Return the first boolean-like value stored under one of the feature names anywhere in a nested
# payload, or None when the payload does not mention the feature at all.
def _extract_feature_flag(value: Any, feature_names: set[str]) -> bool | None:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = _normalize_key_name(key)
            if normalized_key in feature_names and isinstance(child, (bool, str, int, float)):
                return _bool_from_value(child)
            nested = _extract_feature_flag(child, feature_names)
            if nested is not None:
                return nested
        return None

    if isinstance(value, (list, tuple, set)):
        for child in value:
            nested = _extract_feature_flag(child, feature_names)
            if nested is not None:
                return nested

    return None


# Return the normalized names of the actions (such as generateContent) that a Gemini model payload
# lists as supported.
def _extract_supported_actions(raw_model: dict[str, Any]) -> set[str]:
    raw_actions = raw_model.get("supported_actions", raw_model.get("supportedGenerationMethods", []))
    if not isinstance(raw_actions, list):
        return set()
    return {_normalize_key_name(action) for action in raw_actions if _normalize_key_name(action)}


# Structured Google errors

# Return the structured Google SDK error payload (the details attribute) when it is a dict, otherwise
# an empty dict.
def _get_google_error_payload(exc: Exception) -> dict[str, Any]:
    details = getattr(exc, "details", None)
    return details if isinstance(details, dict) else {}


# Return the structured per-error detail entries of a Google SDK error; the details list may sit under
# an "error" key or at the top level of the payload.
def _get_google_error_entries(exc: Exception) -> list[dict[str, Any]]:
    payload = _get_google_error_payload(exc)
    error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else payload
    detail_entries = error_payload.get("details") if isinstance(error_payload, dict) else None
    if not isinstance(detail_entries, list):
        return []
    return [entry for entry in detail_entries if isinstance(entry, dict)]


# Return the numeric error code of a Google SDK error, read from the exception or else from its
# structured payload, or None when there is none.
def _get_google_error_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    if code is None:
        payload = _get_google_error_payload(exc)
        error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        if isinstance(error_payload, dict):
            code = error_payload.get("code")
    try:
        return int(code)
    except (TypeError, ValueError):
        return None


# Return the upper-case status marker (such as RESOURCE_EXHAUSTED) of a Google SDK error, read from the
# exception or else from its structured payload, or an empty string.
def _get_google_error_status(exc: Exception) -> str:
    status = getattr(exc, "status", None)
    if not status:
        payload = _get_google_error_payload(exc)
        error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        if isinstance(error_payload, dict):
            status = error_payload.get("status")
    return str(status or "").strip().upper()


# Return the message of a Google SDK error, read from the exception or else from its structured
# payload, falling back to the exception text.
def _get_google_error_message(exc: Exception) -> str:
    message = getattr(exc, "message", None)
    if not message:
        payload = _get_google_error_payload(exc)
        error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        if isinstance(error_payload, dict):
            message = error_payload.get("message")
    return str(message or str(exc) or "").strip()


# Return the quota violations of all detail entries of a structured Google SDK error as one flat list.
def _iter_quota_violations(exc: Exception) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    for entry in _get_google_error_entries(exc):
        raw_violations = entry.get("violations")
        if not isinstance(raw_violations, list):
            continue
        for violation in raw_violations:
            if isinstance(violation, dict):
                violations.append(violation)
    return violations


# Return whether the structured error likely refers to the provided model: an empty model name always
# matches, otherwise the model must appear in the message or in a quota violation dimension.
def _error_mentions_model(exc: Exception, model_name: str) -> bool:
    normalized_model = _normalize_model_identifier(model_name)
    if not normalized_model:
        return True

    message = _get_google_error_message(exc).lower()
    if normalized_model in message or f"models/{normalized_model}" in message:
        return True

    for violation in _iter_quota_violations(exc):
        dimensions = violation.get("quotaDimensions", {})
        if not isinstance(dimensions, dict):
            continue
        if _model_identifiers_match(normalized_model, dimensions.get("model")):
            return True

    return False


# Return whether the error is a Google quota or rate-limit failure (HTTP 429 or RESOURCE_EXHAUSTED).
def _is_resource_exhausted_error(exc: Exception) -> bool:
    return _get_google_error_code(exc) == 429 or _get_google_error_status(exc) == "RESOURCE_EXHAUSTED"


# Return whether the error indicates zero entitlement for the current key and model: a quota failure
# for this model whose message states a limit of 0.
def _is_zero_quota_error(exc: Exception, model_name: str) -> bool:
    if not _is_resource_exhausted_error(exc) or not _error_mentions_model(exc, model_name):
        return False

    message = _get_google_error_message(exc).lower()
    if "limit: 0" in message or "limit 0" in message:
        return True

    return False


# Return whether the backend says the model cannot serve generateContent: a not-found error for this
# model that mentions the method, or any message that pairs generateContent with an unsupported marker.
def _is_generate_content_unsupported_error(exc: Exception, model_name: str) -> bool:
    code = _get_google_error_code(exc)
    status = _get_google_error_status(exc)
    message = _normalize_key_name(_get_google_error_message(exc))
    if code == 404 or status == "NOT_FOUND":
        return (
            _error_mentions_model(exc, model_name)
            and (
                "generatecontent" in message
                or "supported_methods" in message
                or "supported_for_generatecontent" in message
                or "not_found_for_api_version" in message
            )
        )

    return "generatecontent" in message and any(
        marker in message for marker in ("not_supported", "unsupported", "not_found")
    )


# Return whether a backend error message explicitly says that tools or function calling are unsupported.
def _is_tool_unsupported_error(exc: Exception) -> bool:
    message = _normalize_key_name(_get_google_error_message(exc))
    if not message:
        return False

    has_tool_marker = any(token in message for token in ("tool", "function", "function_call"))
    has_negative_marker = any(
        token in message
        for token in (
            "not_supported",
            "unsupported",
            "not_available",
            "does_not_support",
            "only_supported",
            "tool_use_not_supported",
            "function_calling_not_supported",
        )
    )
    return has_tool_marker and has_negative_marker


# Return whether the backend explicitly rejected Gemini thinking_level with an invalid-argument error.
def _is_thinking_level_unsupported_error(exc: Exception) -> bool:
    code = _get_google_error_code(exc)
    status = _get_google_error_status(exc)
    if code != 400 and status != "INVALID_ARGUMENT":
        return False

    message = _normalize_key_name(_get_google_error_message(exc))
    return (
        "thinking_level" in message and any(marker in message for marker in ("not_supported", "unsupported"))
    ) or "thinking_level_is_not_supported" in message


# Update the runtime caches from a structured Google SDK error and return a short code naming what was
# learned (for example "zero_quota"), or None when the error says nothing about the model.
def _learn_from_google_error(model_name: str, exc: Exception) -> str | None:
    if _is_generate_content_unsupported_error(exc, model_name):
        _update_cached_model_capabilities(model_name, supports_generate_content=False)
        _set_cached_model_availability(model_name, False, "unsupported_generate_content")
        return "unsupported_generate_content"

    if _is_zero_quota_error(exc, model_name):
        _set_cached_model_availability(model_name, False, "zero_quota")
        return "zero_quota"

    if _is_resource_exhausted_error(exc):
        # Keep the model visible when the key merely hit a temporary limit.
        _set_cached_model_availability(model_name, True, "resource_exhausted")
        return "resource_exhausted"

    if _is_tool_unsupported_error(exc):
        _update_cached_model_capabilities(model_name, supports_tool_calling=False)
        return "tool_unsupported"

    if _is_thinking_level_unsupported_error(exc):
        _update_cached_model_capabilities(
            model_name,
            supports_thinking=True,
            supports_think_level=False,
            supports_think_toggle=True,
            think_level_options=[],
        )
        return "thinking_level_unsupported"

    return None


# Capability snapshot

# Return the lower-case reasoning level options from a metadata value, looking inside option containers
# such as enum or options; on/off/true/false entries are toggles, not levels, and are skipped.
def _coerce_reasoning_level_options(value: Any) -> list[str]:
    if isinstance(value, dict):
        for key in ("enum", "options", "allowed_options", "allowedOptions", "values", "items"):
            options = _coerce_reasoning_level_options(value.get(key))
            if options:
                return options
        return []

    normalized_values = []
    for item in _coerce_string_list(value):
        normalized_item = str(item).strip().lower()
        if normalized_item in {"on", "off", "true", "false"}:
            continue
        normalized_values.append(normalized_item)
    return normalized_values


# Search nested Gemini metadata for explicit reasoning-level options published under a thinking_level,
# think_level or reasoning_effort key.
def _extract_reasoning_level_options(value: Any) -> list[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            normalized_key = _normalize_key_name(key)
            if normalized_key in {"thinking_level", "think_level", "reasoning_effort"}:
                options = _coerce_reasoning_level_options(child)
                if options:
                    return options
            options = _extract_reasoning_level_options(child)
            if options:
                return options
        return []

    if isinstance(value, (list, tuple, set)):
        for child in value:
            options = _extract_reasoning_level_options(child)
            if options:
                return options

    return []


# Return the endpoint-scoped Gemini capabilities of one model as a dict of supports_* flags plus the
# selectable thinking levels. Facts learned at runtime from backend errors (the capability cache) win
# over what the published model metadata suggests, and the resolved values are cached again.
def _build_model_capability_snapshot(
    model_name: str,
    raw_model: dict[str, Any],
) -> dict[str, Any]:
    # Gather the cached facts, the metadata tokens and whether this is a multimodal Gemini model.
    cached_capabilities = _get_cached_model_capabilities(model_name)
    capability_tokens = _collect_capability_tokens(raw_model)
    supported_actions = _extract_supported_actions(raw_model)
    normalized_model_name = str(model_name or raw_model.get("name") or "").strip().lower()
    known_gemini_multimodal = (
        "gemini-" in normalized_model_name
        and not any(marker in normalized_model_name for marker in ("embedding", "imagen", "veo"))
    )

    # Explicit feature flags published in the model metadata; None means the metadata does not say.
    explicit_tool_support = _extract_feature_flag(raw_model, TOOL_CAPABILITY_NAMES)
    explicit_vision_support = _extract_feature_flag(raw_model, VISION_CAPABILITY_NAMES)
    explicit_audio_input_support = _extract_feature_flag(raw_model, AUDIO_INPUT_CAPABILITY_NAMES)
    explicit_video_input_support = _extract_feature_flag(raw_model, VIDEO_INPUT_CAPABILITY_NAMES)
    explicit_reasoning_support = _extract_feature_flag(raw_model, REASONING_CAPABILITY_NAMES)
    explicit_reasoning_level_options = _extract_reasoning_level_options(raw_model)

    # Content generation: the cached fact, else the supported action names.
    supports_generate_content = cached_capabilities.get("supports_generate_content")
    if not isinstance(supports_generate_content, bool):
        # The public model listing does not always expose one clean boolean, so
        # fall back to supported action names when the cache is still empty.
        supports_generate_content = any("generatecontent" in action for action in supported_actions)

    # Tool calling: the cached fact, else the explicit flag, else a tool token or generateContent support.
    supports_tool_calling = cached_capabilities.get("supports_tool_calling")
    if not isinstance(supports_tool_calling, bool):
        supports_tool_calling = (
            bool(explicit_tool_support)
            if explicit_tool_support is not None
            else bool(capability_tokens & TOOL_CAPABILITY_NAMES) or supports_generate_content
        )

    # Vision: the cached fact, else the explicit flag, else vision/multimodal tokens or a known
    # multimodal Gemini model that can generate content.
    supports_vision = cached_capabilities.get("supports_vision")
    if not isinstance(supports_vision, bool):
        supports_vision = (
            bool(explicit_vision_support)
            if explicit_vision_support is not None
            else (
                supports_generate_content
                and (
                    bool(capability_tokens & VISION_CAPABILITY_NAMES)
                    or "multimodal" in capability_tokens
                    or known_gemini_multimodal
                )
            )
        )

    # Audio and video input: the cached fact, else the explicit flag, else matching tokens or a known
    # multimodal Gemini model that can generate content.
    supports_audio_input = cached_capabilities.get("supports_audio_input")
    if not isinstance(supports_audio_input, bool):
        supports_audio_input = (
            bool(explicit_audio_input_support)
            if explicit_audio_input_support is not None
            else bool(capability_tokens & AUDIO_INPUT_CAPABILITY_NAMES) or (
                supports_generate_content and known_gemini_multimodal
            )
        )

    supports_video_input = cached_capabilities.get("supports_video_input")
    if not isinstance(supports_video_input, bool):
        supports_video_input = (
            bool(explicit_video_input_support)
            if explicit_video_input_support is not None
            else bool(capability_tokens & VIDEO_INPUT_CAPABILITY_NAMES) or (
                supports_generate_content and known_gemini_multimodal
            )
        )

    # Thinking: published metadata decides when it mentions reasoning at all, otherwise the cached
    # fact is used, and the default is off.
    published_thinking_metadata = (
        "thinking" in raw_model
        or explicit_reasoning_support is not None
        or bool(capability_tokens & REASONING_CAPABILITY_NAMES)
    )
    published_thinking_support = (
        _bool_from_value(raw_model.get("thinking"), default=False)
        or bool(explicit_reasoning_support)
        or bool(capability_tokens & REASONING_CAPABILITY_NAMES)
    )
    cached_thinking_support = cached_capabilities.get("supports_thinking")
    supports_thinking = (
        published_thinking_support
        if published_thinking_metadata
        else cached_thinking_support
    )
    if not isinstance(supports_thinking, bool):
        supports_thinking = False

    # Reuse level options learned from real model requests when public metadata
    # does not publish them.
    think_level_options = list(explicit_reasoning_level_options)
    cached_level_options = cached_capabilities.get("think_level_options")
    if not think_level_options and isinstance(cached_level_options, list):
        think_level_options = [str(item) for item in cached_level_options if str(item).strip()]

    # Level selection: supported when level options exist, otherwise the cached fact, default off. The
    # default levels fill in when selection is supported without published options, and the options
    # are cleared when it is not supported.
    cached_think_level_support = cached_capabilities.get("supports_think_level")
    supports_think_level = (
        True
        if think_level_options
        else cached_think_level_support
    )
    if not isinstance(supports_think_level, bool):
        supports_think_level = False

    if supports_think_level and not think_level_options:
        think_level_options = list(THINK_LEVEL_OPTIONS)
    if not supports_think_level:
        think_level_options = []

    # A plain on/off thinking toggle is offered when thinking is supported without selectable levels.
    supports_think_toggle = cached_capabilities.get("supports_think_toggle")
    if not isinstance(supports_think_toggle, bool):
        supports_think_toggle = bool(supports_thinking) and not bool(supports_think_level)

    snapshot = {
        "supports_generate_content": bool(supports_generate_content),
        "supports_tool_calling": bool(supports_tool_calling),
        "supports_vision": bool(supports_vision),
        "supports_audio_input": bool(supports_audio_input),
        "supports_video_input": bool(supports_video_input),
        "supports_thinking": bool(supports_thinking),
        "supports_think_toggle": bool(supports_think_toggle),
        "supports_think_level": bool(supports_think_level),
        "think_level_options": think_level_options,
    }

    # Cache the snapshot, except for thinking facts that were only defaulted because neither the
    # metadata nor an earlier error said anything, so later metadata can still decide them.
    cached_snapshot = dict(snapshot)
    if not published_thinking_metadata:
        cached_snapshot.pop("supports_thinking", None)
        cached_snapshot.pop("supports_think_toggle", None)
    if (
        not explicit_reasoning_level_options
        and not cached_level_options
        and not isinstance(cached_think_level_support, bool)
    ):
        cached_snapshot.pop("supports_think_level", None)
        cached_snapshot.pop("think_level_options", None)
    _update_cached_model_capabilities(model_name, **cached_snapshot)
    return snapshot


# Tool calls and attachments

# Return a tool-call arguments payload as a dict: JSON object strings are parsed, and any other
# value is wrapped as {"value": ...}.
def _normalize_tool_call_arguments(raw_arguments: Any) -> dict[str, Any]:
    if raw_arguments is None:
        return {}
    if isinstance(raw_arguments, dict):
        return raw_arguments
    if isinstance(raw_arguments, str):
        stripped_value = raw_arguments.strip()
        if not stripped_value:
            return {}
        try:
            parsed_value = json.loads(stripped_value)
        except (TypeError, json.JSONDecodeError):
            return {"value": stripped_value}
        return parsed_value if isinstance(parsed_value, dict) else {"value": parsed_value}
    return {"value": raw_arguments}


# Decode a base64-encoded image attachment, accepting a data: URL; an empty value gives empty bytes.
# Invalid base64 raises, unlike _decode_binary_payload.
def _decode_image_bytes(image_base64: str) -> bytes:
    payload = str(image_base64 or "").strip()
    if not payload:
        return b""
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


# Convert tool result content into a Gemini function response dict: dicts and JSON object strings
# are used as they are, and any other value is wrapped as {"result": ...}.
def _normalize_tool_response_payload(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content

    if isinstance(content, str):
        stripped = content.strip()
        if stripped:
            try:
                parsed = json.loads(stripped)
            except (TypeError, json.JSONDecodeError):
                return {"result": stripped}
            if isinstance(parsed, dict):
                return parsed
            return {"result": parsed}
        return {"result": ""}

    if isinstance(content, (list, tuple, int, float, bool)):
        return {"result": content}

    return {"result": str(content or "")}


# Normalize OpenAI-style or Gemini-style content parts into Gemini request parts. Parts that already
# carry Gemini fields (text, thought, function call or response, inline or file data) are kept with
# those fields; otherwise text, image and file parts in the OpenAI shape are converted.
def _normalize_google_request_parts(raw_parts: Any) -> list[dict[str, Any]]:
    # Accept a single part or a list of parts.
    if isinstance(raw_parts, dict):
        raw_parts = [raw_parts]
    if not isinstance(raw_parts, list):
        return []

    normalized_parts: list[dict[str, Any]] = []
    for raw_part in raw_parts:
        if raw_part is None:
            continue

        # A bare value becomes a text part.
        if not isinstance(raw_part, dict):
            text_value = str(raw_part or "")
            if text_value:
                normalized_parts.append({"text": text_value})
            continue

        # Preserve already-Gemini-shaped parts first so transcript replay does
        # not lose fields that the generic OpenAI shape does not model.
        direct_part: dict[str, Any] = {}
        if raw_part.get("text") is not None:
            direct_part["text"] = str(raw_part.get("text") or "")
        if raw_part.get("thought") is not None:
            direct_part["thought"] = _bool_from_value(raw_part.get("thought"), default=False)

        thought_signature = _decode_binary_payload(
            raw_part.get("thought_signature", raw_part.get("thoughtSignature"))
        )
        if thought_signature:
            direct_part["thought_signature"] = thought_signature

        # Function calls and function responses (a response may carry nested parts of its own).
        function_call = raw_part.get("function_call")
        if isinstance(function_call, dict):
            function_name = str(function_call.get("name", "") or "").strip()
            if function_name:
                direct_part["function_call"] = {
                    "name": function_name,
                    "args": _normalize_tool_call_arguments(function_call.get("args")),
                }

        function_response = raw_part.get("function_response")
        if isinstance(function_response, dict):
            function_name = str(function_response.get("name", "") or "").strip()
            response_payload = function_response.get("response")
            response_parts = _normalize_google_request_parts(function_response.get("parts"))
            if function_name:
                direct_part["function_response"] = {
                    "name": function_name,
                    "response": _normalize_tool_response_payload(response_payload),
                }
                if response_parts:
                    direct_part["function_response"]["parts"] = response_parts

        # Inline binary data (base64 text is decoded to bytes) and file references by URI.
        inline_data = raw_part.get("inline_data")
        if isinstance(inline_data, dict):
            raw_data = inline_data.get("data")
            inline_bytes = _decode_binary_payload(raw_data)
            if inline_bytes is None and raw_data is not None:
                inline_bytes = _decode_image_bytes(str(raw_data or ""))
            if inline_bytes:
                direct_part["inline_data"] = {
                    "data": inline_bytes,
                    "mime_type": str(inline_data.get("mime_type") or inline_data.get("mimeType") or "application/octet-stream"),
                }

        file_data = raw_part.get("file_data")
        if isinstance(file_data, dict):
            file_uri = str(file_data.get("file_uri") or file_data.get("fileUri") or "").strip()
            if file_uri:
                direct_part["file_data"] = {"file_uri": file_uri}
                mime_type = str(file_data.get("mime_type") or file_data.get("mimeType") or "").strip()
                if mime_type:
                    direct_part["file_data"]["mime_type"] = mime_type

        # A part that carried any Gemini field is used as is.
        if direct_part:
            normalized_parts.append(direct_part)
            continue

        part_type = _normalize_key_name(raw_part.get("type"))

        # Fall back to OpenAI-style content-part normalization when the payload
        # came from a provider-agnostic transcript entry.
        if part_type in {"text", "input_text"}:
            text_value = str(raw_part.get("text", "") or "")
            if text_value:
                normalized_parts.append({"text": text_value})
            continue

        # Image parts become inline data; a data: URL supplies its own MIME type, otherwise JPEG is assumed.
        if part_type in {"image_url", "input_image"}:
            image_payload = raw_part.get("image_url", raw_part.get("input_image"))
            if isinstance(image_payload, dict):
                image_payload = (
                    image_payload.get("url")
                    or image_payload.get("data")
                    or image_payload.get("base64")
                )
            image_bytes = _decode_image_bytes(str(image_payload or ""))
            if image_bytes:
                mime_type = "image/jpeg"
                payload_str = str(image_payload or "")
                if payload_str.startswith("data:"):
                    mime_type = payload_str.split(";", 1)[0].split(":", 1)[-1] or mime_type
                normalized_parts.append(
                    {
                        "inline_data": {
                            "data": image_bytes,
                            "mime_type": mime_type,
                        }
                    }
                )
            continue

        # File parts that reference a URI become file data parts.
        if part_type in {"file", "input_file"}:
            file_payload = raw_part.get("file_data") or raw_part.get("file")
            if isinstance(file_payload, dict):
                file_uri = str(file_payload.get("file_uri") or file_payload.get("fileUri") or file_payload.get("uri") or "").strip()
                if file_uri:
                    part: dict[str, Any] = {"file_data": {"file_uri": file_uri}}
                    mime_type = str(file_payload.get("mime_type") or file_payload.get("mimeType") or "").strip()
                    if mime_type:
                        part["file_data"]["mime_type"] = mime_type
                    normalized_parts.append(part)

    return normalized_parts


# Serialize one assistant response part for transcript replay, keeping text (unless include_text is
# false), the thought flag and signature, and function calls. Return None when nothing is left.
def _build_google_history_part(raw_part: Any, *, include_text: bool = True) -> dict[str, Any] | None:
    part = _to_plain_data(raw_part)
    if not isinstance(part, dict):
        return None

    # Copy the text (when requested) and the thought flag, then the thought signature, which Gemini needs
    # back to continue a reasoning turn.
    history_part: dict[str, Any] = {}
    if include_text and part.get("text") is not None:
        history_part["text"] = _sanitize_generated_text(str(part.get("text", "") or ""))
    if part.get("thought") is not None:
        history_part["thought"] = _bool_from_value(part.get("thought"), default=False)

    thought_signature = _encode_binary_payload(
        part.get("thought_signature", part.get("thoughtSignature"))
    )
    if thought_signature:
        history_part["thought_signature"] = thought_signature

    # Keep function calls with normalized arguments.
    function_call = part.get("function_call")
    if isinstance(function_call, dict):
        function_name = str(function_call.get("name", "") or "").strip()
        if function_name:
            history_part["function_call"] = {
                "name": function_name,
                "args": _normalize_tool_call_arguments(function_call.get("args")),
            }

    return history_part if history_part else None


# Return whether preserved Gemini history already includes function calls.
def _history_parts_have_function_call(parts: list[dict[str, Any]]) -> bool:
    return any(isinstance(part, dict) and isinstance(part.get("function_call"), dict) for part in parts)


# Return whether preserved Gemini parts contain a function call without its thought signature, which
# makes the parts unusable for a lossless replay.
def _google_parts_have_unsigned_function_call(raw_parts: Any) -> bool:
    if isinstance(raw_parts, dict):
        raw_parts = [raw_parts]
    if not isinstance(raw_parts, list):
        return False

    for raw_part in raw_parts:
        if not isinstance(raw_part, dict):
            continue
        if not isinstance(raw_part.get("function_call"), dict):
            continue
        signature = raw_part.get("thought_signature", raw_part.get("thoughtSignature"))
        if not _encode_binary_payload(signature):
            return True
    return False


# Convert ASLM chat messages into Gemini-compatible contents. Return the system instruction (all system
# messages joined) and the list of user/model contents with text, images, media, tool calls and tool
# responses as Gemini parts.
def _build_google_contents(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    # Collected system messages, the resulting contents, decoded images by digest (so a repeated image
    # is decoded once) and a flag that drops tool results whose function call could not be replayed.
    system_fragments: list[str] = []
    contents: list[dict[str, Any]] = []
    image_bytes_cache: dict[str, bytes] = {}
    skip_orphan_tool_responses = False

    for message_index, message in enumerate(messages, start=1):
        # Read the message fields; list and dict content is handled through the structured parts below.
        role = str(message.get("role", "user")).lower()
        raw_content = message.get("content", "")
        content = str(raw_content or "") if not isinstance(raw_content, (list, dict)) else ""
        images = [str(item) for item in (message.get("images") or []) if str(item).strip()]
        image_mime_types = message.get("image_mime_types") or []
        media = message.get("media") if isinstance(message.get("media"), list) else []

        # System messages are collected into the system instruction instead of the contents.
        if role == "system":
            if content:
                system_fragments.append(content)
            continue

        parts: list[dict[str, Any]] = []

        # Tool results become a function response of the user turn, unless the call they answer
        # was not replayed.
        if role == "tool":
            if skip_orphan_tool_responses:
                continue
            tool_name = (
                str(
                    message.get("name")
                    or message.get("tool_name")
                    or message.get("alias")
                    or f"tool_{message_index}"
                ).strip()
                or f"tool_{message_index}"
            )
            parts.append(
                {
                    "function_response": {
                        "name": tool_name,
                        "response": _normalize_tool_response_payload(content),
                    }
                }
            )
            contents.append({"role": "user", "parts": parts})
            continue

        skip_orphan_tool_responses = False

        # Prefer preserved Gemini-native parts when they exist so follow-up
        # rounds can replay tool calls and thought signatures losslessly.
        google_parts = message.get("google_parts") if isinstance(message.get("google_parts"), list) else None
        has_unsigned_google_function_call = _google_parts_have_unsigned_function_call(google_parts)
        structured_parts = _normalize_google_request_parts(
            None if has_unsigned_google_function_call else (google_parts if google_parts is not None else raw_content)
        )
        used_structured_parts = bool(structured_parts)
        if used_structured_parts:
            parts.extend(structured_parts)
        elif content:
            parts.append({"text": content})
        if role == "assistant" and has_unsigned_google_function_call:
            skip_orphan_tool_responses = True

        # Attach images as inline data (JPEG unless a MIME type was given), decoding each distinct image once.
        for image_index, image_base64 in enumerate(images):
            mime_type = "image/jpeg"
            if image_index < len(image_mime_types):
                mime_type = str(image_mime_types[image_index] or mime_type)
            image_digest = hashlib.sha256(image_base64.encode("utf-8")).hexdigest()
            image_bytes = image_bytes_cache.get(image_digest)
            if image_bytes is None:
                image_bytes = _decode_image_bytes(image_base64)
                if image_bytes:
                    image_bytes_cache[image_digest] = image_bytes
            if image_bytes:
                parts.append(
                    {
                        "inline_data": {
                            "data": image_bytes,
                            "mime_type": mime_type,
                        }
                    }
                )

        # Attach other media items (audio, video, files) as inline data with their own MIME type.
        for media_item in media:
            if not isinstance(media_item, dict):
                continue
            media_data = str(media_item.get("data") or "").strip()
            if not media_data:
                continue
            media_bytes = _decode_image_bytes(media_data)
            if media_bytes:
                parts.append(
                    {
                        "inline_data": {
                            "data": media_bytes,
                            "mime_type": str(media_item.get("mime_type") or "application/octet-stream"),
                        }
                    }
                )

        if role == "assistant" and not used_structured_parts and not has_unsigned_google_function_call:
            # Rebuild assistant tool calls from the generic transcript shape
            # when no Gemini-native parts were stored with the message.
            for raw_tool_call in message.get("tool_calls") or []:
                function_payload = raw_tool_call.get("function", {}) if isinstance(raw_tool_call, dict) else {}
                function_name = str(function_payload.get("name", "") or "").strip()
                if not function_name:
                    continue
                parts.append(
                    {
                        "function_call": {
                            "name": function_name,
                            "args": _normalize_tool_call_arguments(function_payload.get("arguments")),
                        }
                    }
                )

        # Messages without any usable part are dropped; assistant messages map to the model role.
        if not parts:
            continue

        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": parts,
            }
        )

    return "\n\n".join(fragment for fragment in system_fragments if fragment).strip(), contents


# Return the Gemini tool declarations for one or more tool servers together with the alias lookup
# used to dispatch calls; the declarations list is empty when the servers expose no usable tool.
def _build_google_tools(
    tool_server_ids: list[str],
    model_name: str,
    tool_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    # Build the OpenAI-style definitions through the shared tool registry.
    openai_tools, tool_lookup = tool_registry.build_ollama_tools(
        tool_server_ids,
        engine="google-genai",
        model_name=model_name,
        tool_source_map=tool_context.get("tool_source_map") if isinstance(tool_context, dict) else None,
        allowed_tool_aliases=tool_context.get("allowed_tool_aliases") if isinstance(tool_context, dict) else None,
        instant_mode=bool(tool_context.get("instant_mode")) if isinstance(tool_context, dict) else False,
        instant_search_batch_size=tool_context.get("instant_search_batch_size") if isinstance(tool_context, dict) else None,
    )
    if not openai_tools:
        return [], tool_lookup

    # Convert each definition into a Gemini function declaration.
    function_declarations: list[dict[str, Any]] = []
    for tool_definition in openai_tools:
        function_payload = tool_definition.get("function", {}) if isinstance(tool_definition, dict) else {}
        function_name = str(function_payload.get("name", "") or "").strip()
        if not function_name:
            continue
        function_declarations.append(
            {
                "name": function_name,
                "description": str(function_payload.get("description") or function_name),
                "parameters_json_schema": function_payload.get("parameters") or {"type": "object", "properties": {}},
            }
        )

    if not function_declarations:
        return [], tool_lookup

    return [{"function_declarations": function_declarations}], tool_lookup


# Request configuration

# Convert a stop payload (string, list or other value) into the Gemini list form, or None when empty.
def _coerce_stop_sequences(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else None
    if isinstance(value, list):
        sequences = [str(item).strip() for item in value if str(item).strip()]
        return sequences or None
    return [str(value)]


# Split generic generation options into a Gemini GenerateContentConfig payload: supported options are
# copied under their Gemini names, thinking options are collected into thinking_config, the ASLM
# think and think_level values are mapped onto it, and tools are attached with automatic function
# calling disabled.
def _build_google_request_config(
    options: dict[str, Any],
    *,
    system_instruction: str = "",
    think: Any = None,
    think_level: Any = None,
    think_param_name: str = "include_thoughts",
    think_level_param_name: str = "thinking_level",
    tools: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if system_instruction:
        config["system_instruction"] = system_instruction

    thinking_config: dict[str, Any] = {}
    preconfigured_thinking = options.get("thinking_config")
    if isinstance(preconfigured_thinking, dict):
        thinking_config.update(preconfigured_thinking)

    # Keep only the Gemini-supported top-level keys and collect thinking fields
    # separately so custom aliases can be normalized in one place.
    for raw_key, raw_value in (options or {}).items():
        normalized_key = GOOGLE_OPTION_ALIASES.get(str(raw_key), str(raw_key))
        if normalized_key == "thinking_config":
            continue
        if normalized_key in THINKING_CONFIG_KEYS:
            thinking_config[normalized_key] = raw_value
            continue
        if normalized_key not in GOOGLE_CONFIG_KEYS:
            continue
        if normalized_key == "stop_sequences":
            stop_sequences = _coerce_stop_sequences(raw_value)
            if stop_sequences:
                config["stop_sequences"] = stop_sequences
            continue
        config[normalized_key] = raw_value

    # Map the ASLM think and think_level values onto the thinking config.
    normalized_think_param_name = _normalize_key_name(think_param_name)
    normalized_level_param_name = _normalize_key_name(think_level_param_name)

    if think is not None:
        # Gemini models vary between boolean thought toggles and budget-style
        # reasoning controls, so map the shared ASLM fields carefully.
        if normalized_think_param_name == "thinking_budget":
            thinking_config["thinking_budget"] = int(think)
        elif normalized_think_param_name == "include_thoughts":
            thinking_config["include_thoughts"] = _bool_from_value(think, default=True)
        else:
            thinking_config[str(think_param_name or "include_thoughts")] = think

    if think_level is not None:
        level_value = str(think_level).strip()
        if normalized_level_param_name in {"thinking_level", "think_level"} and level_value:
            thinking_config["thinking_level"] = level_value.upper()
        elif think_level_param_name:
            thinking_config[str(think_level_param_name)] = think_level

    # Whenever any thinking option is set, thought summaries are requested unless turned off explicitly.
    if thinking_config:
        thinking_config.setdefault("include_thoughts", True)
        config["thinking_config"] = thinking_config

    # Attach the tool declarations; function calling stays in AUTO mode unless the caller set its own.
    if tools:
        # Disable automatic client-side execution because ASLM resolves tools
        # locally and needs the raw function-call payloads back.
        config["tools"] = tools
        config["automatic_function_calling"] = {"disable": True}
        tool_config = dict(config.get("tool_config", {}) or {})
        if not isinstance(tool_config.get("function_calling_config"), dict):
            tool_config["function_calling_config"] = {"mode": "AUTO"}
        config["tool_config"] = tool_config

    return config


# Return whether a Gemini request config currently sends thinking_level.
def _config_uses_thinking_level(config: dict[str, Any]) -> bool:
    thinking_config = config.get("thinking_config")
    return isinstance(thinking_config, dict) and thinking_config.get("thinking_level") is not None


# Return a cloned Gemini request config without thinking_level.
def _strip_thinking_level_from_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized_config = dict(config or {})
    thinking_config = normalized_config.get("thinking_config")
    if not isinstance(thinking_config, dict):
        return normalized_config

    normalized_thinking_config = dict(thinking_config)
    normalized_thinking_config.pop("thinking_level", None)
    normalized_thinking_config.pop("thinkingLevel", None)
    if normalized_thinking_config:
        normalized_config["thinking_config"] = normalized_thinking_config
    else:
        normalized_config.pop("thinking_config", None)
    return normalized_config


# Apply learned runtime capability overrides to a Gemini request config: thinking_level is removed
# for models that were learned not to support it. Return a copy and leave the input untouched.
def _apply_learned_request_preferences(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    cached_capabilities = _get_cached_model_capabilities(model_name)
    if cached_capabilities.get("supports_think_level") is False:
        return _strip_thinking_level_from_config(config)
    return dict(config or {})


# Return a request config clone with function tools disabled.
def _strip_tools_from_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized_config = dict(config or {})
    normalized_config.pop("tools", None)
    normalized_config.pop("automatic_function_calling", None)
    normalized_config.pop("tool_config", None)
    return normalized_config


# Tool events and messages

# Serialize one tool invocation so the UI can render it during streaming.
def _build_tool_event(tool_lookup: dict[str, dict[str, Any]], tool_call: dict[str, Any]) -> dict[str, Any]:
    return tool_registry.build_tool_event(tool_lookup, tool_call)


# Build a tool message payload for the shared ASLM transcript: the model-facing content, the extras
# kept for the UI and, when a tool event is given, the alias, server and display details.
def _build_tool_message(
    tool_name: str,
    tool_call_id: str,
    content: str | dict[str, Any],
    tool_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    # Split the result into the model-facing content and the extras kept for the UI.
    model_content, tool_extras = tool_registry.split_tool_result_payload(content)
    payload: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
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


# Reasoning text parsing

# Streaming parser that splits inline reasoning (text wrapped in tags such as <think>) from visible
# content. It is stateful across chunks: it remembers whether it is inside a reasoning block and holds
# back a tail that may be the beginning of a tag cut by a chunk boundary. Opening tags are accepted
# only at the start of the text or after a clear separator, so tags quoted in prose stay visible.
class _ReasoningTextParser:
    # Create a parser for the given (opening, closing) tag pairs and start outside a reasoning block.
    def __init__(self, tag_pairs: tuple[tuple[str, str], ...] = REASONING_TAG_PAIRS) -> None:
        # Opening and closing tags, in the same order as the pairs.
        self._start_tags = [start for start, _end in tag_pairs]
        self._end_tags = [end for _start, end in tag_pairs]

        # Parsing state: inside a reasoning block, the held-back possible tag prefix and the last
        # visible character of earlier chunks (None before any text).
        self._in_reasoning = False
        self._pending = ""
        self._visible_boundary_char: str | None = None

    # Return (index, tag) for the earliest of the given tags found in the source, or None. Outside a
    # reasoning block an opening tag only counts when it sits at a start boundary.
    def _find_next_tag(self, source: str, tags: list[str]) -> tuple[int, str] | None:
        next_match: tuple[int, str] | None = None
        for tag in tags:
            search_from = 0
            index = source.find(tag, search_from)
            while (
                index != -1
                and not self._in_reasoning
                and not self._is_reasoning_start_boundary(source, index)
            ):
                search_from = index + len(tag)
                index = source.find(tag, search_from)
            if index == -1:
                continue
            if next_match is None or index < next_match[0]:
                next_match = (index, tag)
        return next_match

    # Return whether a tag at the given index may open a reasoning block: it must start the text or
    # follow a newline, a ">" or punctuation, ignoring horizontal whitespace before it.
    def _is_reasoning_start_boundary(self, source: str, tag_index: int) -> bool:
        cursor = tag_index - 1
        while cursor >= 0 and source[cursor] in " \t\f\v":
            cursor -= 1
        previous = source[cursor] if cursor >= 0 else self._visible_boundary_char
        if previous is None:
            return True
        return previous in "\r\n>" or unicodedata.category(previous).startswith("P")

    # Remember the last non-horizontal-whitespace visible character across chunks.
    def _remember_visible_text(self, value: str) -> None:
        for character in reversed(str(value or "")):
            if character not in " \t\f\v":
                self._visible_boundary_char = character
                return

    # Split off the tail of the source that could be the beginning of one of the tags, because a tag may
    # be cut by a chunk boundary. Return the text that is safe to emit and the held-back tail.
    @staticmethod
    def _split_possible_tag_prefix(source: str, tags: list[str]) -> tuple[str, str]:
        reserve = 0
        for tag in tags:
            max_check = min(len(tag) - 1, len(source))
            for size in range(max_check, 0, -1):
                if source.endswith(tag[:size]):
                    reserve = max(reserve, size)
                    break
        if reserve:
            return source[:-reserve], source[-reserve:]
        return source, ""

    # Parse one chunk and return the (thinking, visible_content) fragments it completes. Content that
    # the SDK already flags as reasoning is returned entirely as thinking; otherwise reasoning tags
    # switch between the two, and a possible partial tag at the end is kept for the next call.
    def feed(self, content: str, reasoning_type: str | None = None) -> tuple[str, str]:
        thinking_parts: list[str] = []
        content_parts: list[str] = []

        # SDK-flagged reasoning needs no tag parsing.
        normalized_reasoning_type = _normalize_key_name(reasoning_type)
        if normalized_reasoning_type in {"reasoning", "thought", "thinking"}:
            return str(content or ""), ""

        # Prepend the held-back tail and consume the text tag by tag, toggling the reasoning state
        # at every tag. Text without a further tag is emitted except for a possible partial tag.
        source = f"{self._pending}{content or ''}"
        self._pending = ""

        while source:
            tags = self._end_tags if self._in_reasoning else self._start_tags
            next_tag = self._find_next_tag(source, tags)
            if next_tag is None:
                visible, pending = self._split_possible_tag_prefix(source, tags)
                if self._in_reasoning:
                    thinking_parts.append(visible)
                else:
                    content_parts.append(visible)
                    self._remember_visible_text(visible)
                self._pending = pending
                break

            index, tag = next_tag
            chunk = source[:index]
            if chunk:
                if self._in_reasoning:
                    thinking_parts.append(chunk)
                else:
                    content_parts.append(chunk)
                    self._remember_visible_text(chunk)
            source = source[index + len(tag):]
            self._in_reasoning = not self._in_reasoning

        return "".join(thinking_parts), "".join(content_parts)

    # Flush the held-back partial fragment at the end of the stream and return it as (thinking, visible),
    # depending on whether the parser is still inside a reasoning block.
    def flush(self) -> tuple[str, str]:
        pending = self._pending
        self._pending = ""
        if not pending:
            return "", ""
        if self._in_reasoning:
            return pending, ""
        self._remember_visible_text(pending)
        return "", pending


# Response streaming

# Return the parts of the first candidate of an SDK response object (or streamed chunk).
def _get_response_parts(response: Any) -> list[Any]:
    parts = getattr(response, "parts", None)
    if isinstance(parts, list):
        return parts

    payload = _to_plain_data(response)
    if not isinstance(payload, dict):
        return []

    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list) or not candidates:
        return []

    first_candidate = candidates[0] if isinstance(candidates[0], dict) else {}
    content = first_candidate.get("content", {}) if isinstance(first_candidate, dict) else {}
    parts = content.get("parts", []) if isinstance(content, dict) else []
    return parts if isinstance(parts, list) else []


# Parse response parts into (reasoning text, visible text, tool calls). Text goes through the
# reasoning parser, so SDK-flagged thoughts and tagged reasoning both end up as reasoning text.
def _parse_google_response_parts(
    parser: _ReasoningTextParser,
    parts: list[Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    thinking_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for raw_part in parts:
        part = _to_plain_data(raw_part)
        if not isinstance(part, dict):
            continue

        # Function-call parts become tool calls and carry no text.
        function_call = part.get("function_call")
        if isinstance(function_call, dict):
            function_name = str(function_call.get("name", "") or "").strip()
            if function_name:
                tool_calls.append(
                    {
                        "id": str(function_call.get("id", "") or ""),
                        "name": function_name,
                        "arguments": _normalize_tool_call_arguments(function_call.get("args")),
                    }
                )
            continue

        # Text parts are cleaned of control tokens and split into reasoning and visible text; parts
        # flagged as thoughts by the SDK count as reasoning.
        text_fragment = _sanitize_generated_text(str(part.get("text", "") or ""))
        if not text_fragment:
            continue

        reasoning_type = "reasoning" if _bool_from_value(part.get("thought"), default=False) else None
        thinking_part, content_part = parser.feed(text_fragment, reasoning_type=reasoning_type)
        if thinking_part:
            thinking_parts.append(thinking_part)
        if content_part:
            content_parts.append(content_part)

    return "".join(thinking_parts), "".join(content_parts), tool_calls


# Run one Gemini round as a generator: yield {"message": ...} chunks while the model answers (or one
# chunk for a non-streaming call when stream is false) and return the assembled assistant message,
# with its OpenAI-style tool calls and the Gemini-native google_parts used for lossless replay. When
# the endpoint rejects thinking_level before any output, the request is retried once without it.
def _stream_google_round(
    client: Any,
    model_name: str,
    contents: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    stream: bool = True,
):
    # Accumulated answer: visible text, reasoning text, deduplicated tool calls with their signatures,
    # the Gemini parts for replay and whether any chunk was already streamed to the caller.
    assistant_content = ""
    assistant_thinking = ""
    tool_calls: list[dict[str, Any]] = []
    seen_tool_call_signatures: set[str] = set()
    assistant_history_parts: list[dict[str, Any]] = []
    parser = _ReasoningTextParser()
    yielded_chunk = False

    # Apply learned request fixes first.
    effective_config = _apply_learned_request_preferences(model_name, config)
    allow_thinking_level_retry = _config_uses_thinking_level(effective_config)

    while True:
        try:
            # Retry in-place when the endpoint rejects ``thinking_level`` so a
            # learned compatibility fix can salvage the same request.
            response = (
                client.models.generate_content_stream(model=model_name, contents=contents, config=effective_config)
                if stream
                else client.models.generate_content(model=model_name, contents=contents, config=effective_config)
            )

            iterable = response if stream else [response]

            for raw_chunk in iterable:
                if _abort_event.is_set():
                    break

                # Keep thought signatures and function calls of the chunk for replay; text is recorded below.
                chunk_parts = _get_response_parts(raw_chunk)
                for raw_part in chunk_parts:
                    history_part = _build_google_history_part(raw_part, include_text=False)
                    if history_part is not None:
                        assistant_history_parts.append(history_part)

                # Parse visible text, reasoning fragments, and function calls
                # from the same chunk so transcript replay stays ordered.
                thinking_part, content_part, chunk_tool_calls = _parse_google_response_parts(
                    parser,
                    chunk_parts,
                )

                # Record new text in the answer and the replay history, then stream it to the caller.
                # Thinking text fills a preceding thought part that has no text yet.
                if thinking_part or content_part:
                    yielded_chunk = True
                    assistant_content += content_part
                    assistant_thinking += thinking_part
                    if thinking_part:
                        if (
                            assistant_history_parts
                            and assistant_history_parts[-1].get("thought") is True
                            and "text" not in assistant_history_parts[-1]
                        ):
                            assistant_history_parts[-1]["text"] = thinking_part
                        else:
                            assistant_history_parts.append({"text": thinking_part, "thought": True})
                    if content_part:
                        assistant_history_parts.append({"text": content_part})
                    payload = {"role": "assistant", "content": content_part}
                    if thinking_part:
                        payload["thinking"] = thinking_part
                    yield {"message": payload}

                for tool_call_index, tool_call in enumerate(chunk_tool_calls, start=1):
                    # Gemini can repeat the same tool call across chunks, so
                    # deduplicate by a stable serialized signature.
                    tool_call_id = str(tool_call.get("id") or f"call_{len(tool_calls) + tool_call_index}_{tool_call['name']}")
                    signature = json.dumps(
                        {
                            "id": tool_call_id,
                            "name": tool_call["name"],
                            "arguments": tool_call.get("arguments") or {},
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                    )
                    if signature in seen_tool_call_signatures:
                        continue
                    seen_tool_call_signatures.add(signature)
                    tool_calls.append(
                        {
                            "id": tool_call_id,
                            "name": tool_call["name"],
                            "arguments": tool_call.get("arguments") or {},
                        }
                    )
            break
        except Exception as exc:
            # Learn from the error, retry once without thinking_level if the endpoint rejected it before
            # any output was streamed, and re-raise every other failure.
            _learn_from_google_error(model_name, exc)
            if (
                allow_thinking_level_retry
                and not yielded_chunk
                and _is_thinking_level_unsupported_error(exc)
            ):
                effective_config = _strip_thinking_level_from_config(effective_config)
                allow_thinking_level_retry = False
                continue
            raise

    # Flush any buffered tag fragment once the SDK stream is finished.
    tail_thinking, tail_content = parser.flush()
    if tail_thinking or tail_content:
        yielded_chunk = True
        assistant_content += tail_content
        assistant_thinking += tail_thinking
        if tail_thinking:
            if (
                assistant_history_parts
                and assistant_history_parts[-1].get("thought") is True
                and "text" not in assistant_history_parts[-1]
            ):
                assistant_history_parts[-1]["text"] = tail_thinking
            else:
                assistant_history_parts.append({"text": tail_thinking, "thought": True})
        if tail_content:
            assistant_history_parts.append({"text": tail_content})
        payload = {"role": "assistant", "content": tail_content}
        if tail_thinking:
            payload["thinking"] = tail_thinking
        yield {"message": payload}

    # Assemble the final message: tool calls in the OpenAI shape for the shared tool loop, and the
    # Gemini-native parts for replay, to which function calls are added if the stream reported none.
    assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant_content}
    if assistant_thinking:
        assistant_message["thinking"] = assistant_thinking
    if tool_calls:
        assistant_message["tool_calls"] = [
            {
                "id": tool_call["id"],
                "type": "function",
                "function": {
                    "name": tool_call["name"],
                    "arguments": json.dumps(tool_call.get("arguments") or {}, ensure_ascii=False),
                },
            }
            for tool_call in tool_calls
        ]
        if not _history_parts_have_function_call(assistant_history_parts):
            assistant_history_parts.extend(
                {
                    "function_call": {
                        "name": tool_call["name"],
                        "args": tool_call.get("arguments") or {},
                    }
                }
                for tool_call in tool_calls
            )
    if assistant_history_parts:
        assistant_message["google_parts"] = assistant_history_parts

    # Emit the final message once when text was assembled without having been streamed.
    if not yielded_chunk and (assistant_content or assistant_thinking):
        yield {"message": assistant_message}

    return assistant_message


# Re-yield every chunk of a round stream and return its final assistant message, or an empty
# assistant message when the stream returned nothing.
def _yield_stream_round(round_stream):
    while True:
        try:
            yield next(round_stream)
        except StopIteration as stop:
            return stop.value or {"role": "assistant", "content": ""}


# Convert an assembled assistant message back into Gemini conversation content: preserved google_parts
# are replayed as they are, otherwise text and tool calls are rebuilt from the generic transcript
# shape. Return None when the message has nothing to send.
def _assistant_message_to_content(assistant_message: dict[str, Any]) -> dict[str, Any] | None:
    parts = _normalize_google_request_parts(assistant_message.get("google_parts"))
    if parts:
        return {"role": "model", "parts": parts}

    # Fall back to the generic transcript shape when Gemini-native parts were
    # not preserved on the assembled assistant message.
    parts = []
    visible_content = str(assistant_message.get("content", "") or "")
    if visible_content:
        parts.append({"text": visible_content})

    for raw_tool_call in assistant_message.get("tool_calls") or []:
        if not isinstance(raw_tool_call, dict):
            continue
        function_payload = raw_tool_call.get("function", {})
        if not isinstance(function_payload, dict):
            function_payload = {}
        function_name = str(function_payload.get("name", "") or "").strip()
        if not function_name:
            continue
        parts.append(
            {
                "function_call": {
                    "name": function_name,
                    "args": _normalize_tool_call_arguments(function_payload.get("arguments")),
                }
            }
        )

    if not parts:
        return None

    return {"role": "model", "parts": parts}


# Tool loop

# Resolve local tools through Gemini function calling while keeping ASLM transcript markers. The
# generator yields UI chunks ("message"), "transcript_message", "tool_events" and "tool_result" items.
# Each round streams one model answer, runs the tool calls it requested (with cooldown, duplicate and
# quota guards) and sends the results back as function responses, until the model answers without
# tools, the round limit is reached or the user cancels. Repeated blocked or cancelled results force
# one final answer without tools.
def _run_tool_loop(
    client: Any,
    model_name: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tool_server_ids: list[str],
    tool_context: dict[str, Any],
    *,
    think: Any = None,
    think_level: Any = None,
    think_param_name: str = "include_thoughts",
    think_level_param_name: str = "thinking_level",
    stream: bool = True,
):
    # Build the tool declarations, the Gemini conversation and the request config with the
    # available-tools prompt added to the system instruction.
    tools, tool_lookup = _build_google_tools(tool_server_ids, model_name, tool_context)
    system_instruction, conversation = _build_google_contents(messages)
    capability_prompt = tool_registry.available_tools_prompt(tool_lookup)
    system_instruction = "\n\n".join(part for part in (system_instruction, capability_prompt) if part)
    request_config = _build_google_request_config(
        options,
        system_instruction=system_instruction,
        think=think,
        think_level=think_level,
        think_param_name=think_param_name,
        think_level_param_name=think_level_param_name,
        tools=tools,
    )
    request_config = _apply_learned_request_preferences(model_name, request_config)

    # Stream one direct round when no local tools are configured.
    if not tools:
        yield from _yield_stream_round(
            _stream_google_round(client, model_name, conversation, request_config, stream=stream)
        )
        return

    # Track quotas, duplicate signatures, and repeated guardrail results.
    tool_quota_counters: dict[str, int] = {}
    seen_tool_signatures: set[str] = set()
    consecutive_blocked_tool_results = 0
    required_tool_server_ids = tool_registry.required_tool_server_ids_from_context(tool_context, tool_lookup)
    satisfied_tool_server_ids: set[str] = set()
    required_nudges_left = tool_registry.MAX_REQUIRED_TOOL_NUDGES if required_tool_server_ids else 0

    # Run model rounds until the model stops calling tools or the round limit is reached.
    max_tool_rounds = tool_registry.tool_round_limit_from_context(tool_context, MAX_TOOL_ROUNDS)
    for round_index in range(max_tool_rounds):
        # Re-apply learned preferences each round because earlier failures may
        # have updated the runtime capability cache.
        request_config = _apply_learned_request_preferences(model_name, request_config)
        assistant_message = yield from _yield_stream_round(
            _stream_google_round(client, model_name, conversation, request_config, stream=stream)
        )

        # No tool calls: the model answered. Remind it of required tools that were not used yet,
        # otherwise finish the loop.
        raw_tool_calls = assistant_message.get("tool_calls") or []
        if not raw_tool_calls:
            assistant_content = _assistant_message_to_content(assistant_message)
            if assistant_content is not None:
                conversation.append(assistant_content)
            yield {"transcript_message": assistant_message}
            unmet_tool_server_ids = [
                server_id for server_id in required_tool_server_ids
                if server_id not in satisfied_tool_server_ids
            ]
            if unmet_tool_server_ids and required_nudges_left > 0:
                required_nudges_left -= 1
                conversation.append(
                    {
                        "role": "user",
                        "parts": [
                            {
                                "text": tool_registry.required_tools_reminder_prompt(
                                    unmet_tool_server_ids, tool_lookup
                                )
                            }
                        ],
                    }
                )
                continue
            return

        tool_calls: list[dict[str, Any]] = []

        # Convert provider payloads into the adapter's shared dispatch format
        # before handing them to the local MCP tool bridge.
        for tool_call_index, raw_tool_call in enumerate(raw_tool_calls, start=1):
            function_payload = raw_tool_call.get("function", {}) if isinstance(raw_tool_call, dict) else {}
            function_name = str(function_payload.get("name", "") or "").strip()
            if not function_name:
                continue
            tool_calls.append(
                {
                    "id": str(raw_tool_call.get("id") or f"call_{round_index}_{tool_call_index}_{function_name}"),
                    "name": function_name,
                    "arguments": _normalize_tool_call_arguments(function_payload.get("arguments")),
                }
            )

        # Resolve aliases, apply the per-round call limit and remember which required tool servers
        # this round touches.
        tool_calls = tool_registry.prepare_tool_calls(tool_lookup, tool_calls)
        tool_calls = tool_registry.limit_tool_calls_from_context(tool_context, tool_calls, tool_lookup)

        for tool_call in tool_calls:
            satisfied_server_id = tool_registry.tool_server_id_for_alias(tool_lookup, tool_call["name"])
            if satisfied_server_id:
                satisfied_tool_server_ids.add(satisfied_server_id)

        # Store the assistant message with its canonical tool calls in the conversation and the transcript.
        canonical_assistant_message = tool_registry.canonicalize_transcript_tool_calls(
            assistant_message, tool_calls
        )
        assistant_content = _assistant_message_to_content(canonical_assistant_message)
        if assistant_content is not None:
            conversation.append(assistant_content)
        yield {"transcript_message": canonical_assistant_message}

        # Announce the round's tool events to the UI; each alias gets its call index so that several
        # calls of the same tool stay distinguishable.
        tool_response_parts: list[dict[str, Any]] = []
        tool_events = [_build_tool_event(tool_lookup, tool_call) for tool_call in tool_calls]
        for _i, _ev in enumerate(tool_events):
            _ev["alias"] = f"{_ev['alias']}__{_i}"
        yield {"tool_events": tool_events}

        # Execute the round's tool calls one by one, starting with the per-call context and request log.
        cancelled_tool_result_seen = False
        for tool_call_index, (tool_call, tool_event) in enumerate(zip(tool_calls, tool_events), start=1):
            call_context = dict(tool_context or {})
            call_context.update(
                {
                    "engine": "google-genai",
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

            tool_message = _build_tool_message(
                tool_call["name"],
                tool_call["id"],
                tool_result,
                tool_event,
            )

            # Gemini expects tool results as ``function_response`` parts in the
            # next user turn rather than as OpenAI-style tool-role messages.
            tool_response_parts.append(
                {
                    "function_response": {
                        "name": tool_call["name"],
                        "response": _normalize_tool_response_payload(tool_message["content"]),
                    }
                }
            )
            tool_registry.log_search_tool_io(
                "response",
                tool_event,
                arguments=tool_call.get("arguments") or {},
                context=call_context,
                result=tool_response_parts[-1],
            )
            yield {"tool_result": dict(tool_message)}

        # Send all results of the round back to the model as one user turn.
        if tool_response_parts:
            conversation.append({"role": "user", "parts": tool_response_parts})

        # Force one final answer when repeated guardrail tool results stall progress.
        if cancelled_tool_result_seen and _abort_event.is_set():
            return
        if cancelled_tool_result_seen or consecutive_blocked_tool_results >= 2:
            conversation.append(
                {
                    "role": "user",
                    "parts": [{
                        "text": (
                            tool_registry.forced_final_prompt_after_tool_cancellation()
                            if cancelled_tool_result_seen
                            else tool_registry.forced_final_prompt_after_tool_blocks()
                        )
                    }],
                }
            )
            final_config = _strip_tools_from_config(request_config)
            assistant_message = yield from _yield_stream_round(
                _stream_google_round(client, model_name, conversation, final_config, stream=stream)
            )
            yield {"transcript_message": assistant_message}
            return

        # Compact the conversation when it grows too large before the next round.
        conversation = tool_registry.maybe_compact_tool_conversation(
            tool_context, conversation, provider_format="google"
        )

    # The round limit was exhausted without a final answer.
    yield {"message": {"content": "[Error during generation: tool loop exceeded the safety limit.]"}}


# Return whether the conversation already contains tool messages or assistant tool calls.
def _conversation_uses_tools(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if str(message.get("role", "")).lower() == "tool":
            return True
        if message.get("tool_calls"):
            return True
    return False


# Public adapter API

# Return the models exposed by the configured Gemini API endpoint, keeping only those that can
# generate content and have not been learned to be unavailable for the current key.
def get_models() -> list[Any]:
    client = _get_client()
    try:
        models: list[dict[str, Any]] = []
        for payload in _iter_model_payloads(client):
            model_name = _extract_model_name(payload)
            if not model_name:
                continue

            # Skip models that cannot generate content or were learned to be unavailable for this key.
            capability_snapshot = _build_model_capability_snapshot(
                model_name,
                payload,
            )
            if not capability_snapshot.get("supports_generate_content"):
                continue
            cached_availability = _get_cached_model_availability(model_name)
            if isinstance(cached_availability, dict) and not bool(cached_availability.get("available", True)):
                continue
            models.append(
                {
                    "id": model_name,
                    "name": model_name,
                    "model": model_name,
                }
            )
        return models
    finally:
        _close_client(client)


# Raise because Gemini API models are remote and cannot be downloaded locally.
def download_model(model_name: str, **kwargs: Any) -> Any:
    raise NotImplementedError("Google GenAI models are remote and cannot be downloaded locally.")


# Return the settings payload of one Gemini model for the UI: context length, default parameter values,
# supported parameters, thinking and input capabilities, and the raw model metadata.
def get_model_settings(model_name: str) -> dict[str, Any]:
    # Fetch the model metadata and resolve its capabilities.
    client = _get_client()
    try:
        raw_model = _get_model_payload(client, model_name)
        capability_snapshot = _build_model_capability_snapshot(
            model_name,
            raw_model,
        )
    finally:
        _close_client(client)

    defaults: dict[str, Any] = {}

    # Export stable defaults only for values that the public metadata exposes.
    for field_name in ("temperature", "top_p", "top_k"):
        if raw_model.get(field_name) is not None:
            defaults[field_name] = raw_model[field_name]

    # Fill in fallback defaults: one candidate, an output limit capped at 8192 tokens and the
    # thinking control that the model supports.
    output_token_limit = _coerce_positive_int(raw_model.get("output_token_limit"))

    if capability_snapshot["supports_generate_content"]:
        defaults.setdefault("candidate_count", 1)
        if output_token_limit is not None:
            defaults.setdefault("max_output_tokens", min(output_token_limit, 8192))
        else:
            defaults.setdefault("max_output_tokens", 8192)
    if capability_snapshot["supports_think_level"]:
        defaults.setdefault("thinking_level", "medium")
    elif capability_snapshot["supports_think_toggle"]:
        defaults.setdefault("include_thoughts", True)

    supported_parameters = set()

    # Supported parameters are derived from the resolved capability snapshot
    # rather than blindly exposing every known Gemini config field.
    if capability_snapshot["supports_generate_content"]:
        supported_parameters.update(
            {
                "candidate_count",
                "frequency_penalty",
                "logprobs",
                "max_output_tokens",
                "presence_penalty",
                "response_json_schema",
                "response_logprobs",
                "response_mime_type",
                "response_schema",
                "seed",
                "stop_sequences",
                "temperature",
                "top_k",
                "top_p",
            }
        )
    if capability_snapshot["supports_thinking"]:
        supported_parameters.update({"include_thoughts", "thinking_budget"})
    if capability_snapshot["supports_think_level"]:
        supported_parameters.add("thinking_level")

    # Capability names shown in the UI; file attachments are always available.
    capabilities: list[str] = []
    if capability_snapshot["supports_vision"]:
        capabilities.append("vision")
    if capability_snapshot["supports_audio_input"]:
        capabilities.append("audio_input")
    if capability_snapshot["supports_video_input"]:
        capabilities.append("video_input")
    if capability_snapshot["supports_tool_calling"]:
        capabilities.append("tools")
    if capability_snapshot["supports_thinking"]:
        capabilities.append("thinking")
    capabilities.append("files")

    # Assemble the settings payload.
    runtime_limits: dict[str, Any] = {}
    if output_token_limit is not None:
        runtime_limits["output_token_limit"] = output_token_limit

    return {
        "model": _extract_model_name(raw_model) or model_name,
        "context_length": _coerce_positive_int(raw_model.get("input_token_limit")) or 8192,
        "defaults": defaults,
        "supported_parameters": sorted(supported_parameters, key=str.casefold),
        "supports_thinking": capability_snapshot["supports_thinking"],
        "supports_think_toggle": capability_snapshot["supports_think_toggle"],
        "supports_think_level": capability_snapshot["supports_think_level"],
        "think_param_name": "include_thoughts",
        "think_level_param_name": "thinking_level",
        "think_level_options": list(capability_snapshot["think_level_options"]),
        "supports_vision": capability_snapshot["supports_vision"],
        "supports_audio_input": capability_snapshot["supports_audio_input"],
        "supports_video_input": capability_snapshot["supports_video_input"],
        "supports_tool_calling": capability_snapshot["supports_tool_calling"],
        # ASLM-Chat file attachments are serialized into universal text context.
        "supports_files": True,
        "capabilities": capabilities,
        "runtime_limits": runtime_limits,
        "custom_fields": [],
        "raw": raw_model,
    }


# Generate a streamed or non-streamed response through Google GenAI as a generator of chunks. Tool
# servers selected in the kwargs run through the tool loop; otherwise one direct round is streamed,
# and the final assistant message is added to the transcript. The client is always closed afterwards.
def generate(model_name: str, messages: list[dict[str, Any]], **kwargs: Any):
    # Normalize tool server identifiers from every supported call shape.
    raw_ids = kwargs.pop("tool_server_ids", None) or kwargs.pop("tool_server_id", None) or kwargs.pop("tool_id", None)
    if isinstance(raw_ids, str):
        tool_server_ids = [raw_ids] if raw_ids.strip() else []
    elif isinstance(raw_ids, list):
        tool_server_ids = [str(server_id) for server_id in raw_ids if str(server_id).strip()]
    else:
        tool_server_ids = []

    tool_context = dict(kwargs.pop("tool_context", {}) or {})
    stream = bool(kwargs.get("stream", False))

    client = _get_client()
    try:
        _abort_event.clear()

        # Route tool-enabled sessions through the shared tool loop.
        if tool_server_ids:
            yield from _run_tool_loop(
                client,
                model_name,
                messages,
                dict(kwargs.get("options", {}) or {}),
                tool_server_ids,
                tool_context,
                think=kwargs.get("think"),
                think_level=kwargs.get("think_level"),
                think_param_name=str(kwargs.get("think_param_name", "include_thoughts") or "include_thoughts"),
                think_level_param_name=str(kwargs.get("think_level_param_name", "thinking_level") or "thinking_level"),
                stream=stream,
            )
            return

        # Build the shared Gemini request payload once for direct rounds.
        system_instruction, conversation = _build_google_contents(messages)
        request_config = _build_google_request_config(
            dict(kwargs.get("options", {}) or {}),
            system_instruction=system_instruction,
            think=kwargs.get("think"),
            think_level=kwargs.get("think_level"),
            think_param_name=str(kwargs.get("think_param_name", "include_thoughts") or "include_thoughts"),
            think_level_param_name=str(kwargs.get("think_level_param_name", "thinking_level") or "thinking_level"),
        )
        request_config = _apply_learned_request_preferences(model_name, request_config)

        # Preserve existing tool transcripts when the conversation already contains them.
        if _conversation_uses_tools(messages):
            assistant_message = yield from _yield_stream_round(
                _stream_google_round(
                    client,
                    model_name,
                    conversation,
                    request_config,
                    stream=stream,
                )
            )
            if isinstance(assistant_message, dict):
                yield {"transcript_message": assistant_message}
            return

        # Run a standard assistant round when no tool context is involved.
        assistant_message = yield from _yield_stream_round(
            _stream_google_round(
                client,
                model_name,
                conversation,
                request_config,
                stream=stream,
            )
        )
        if isinstance(assistant_message, dict):
            yield {"transcript_message": assistant_message}
    finally:
        _close_client(client)
