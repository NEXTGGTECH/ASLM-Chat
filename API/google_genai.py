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

_abort_event = threading.Event()
_model_capability_cache_lock = threading.Lock()
_model_capability_cache: dict[tuple[str, str], dict[str, Any]] = {}
_model_availability_cache_lock = threading.Lock()
_model_availability_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

MAX_TOOL_ROUNDS = 100
MODEL_AVAILABILITY_TTL_SECONDS = 15 * 60
REASONING_TAG_PAIRS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
)
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
THINK_LEVEL_OPTIONS = ["minimal", "low", "medium", "high"]
GOOGLE_OPTION_ALIASES = {
    "max_tokens": "max_output_tokens",
    "max_completion_tokens": "max_output_tokens",
    "n": "candidate_count",
    "stop": "stop_sequences",
}
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
THINKING_CONFIG_KEYS = {"include_thoughts", "thinking_budget", "thinking_level"}


# Return a safe shallow copy for one cached payload.
def _clone_cache_payload(value: dict[str, Any] | None) -> dict[str, Any]:
    """Return a safe shallow copy for one cached payload."""

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
    """Signal the active Google GenAI generation to stop."""

    _abort_event.set()


# Import the Google GenAI SDK lazily so the app can boot without it.
def _get_sdk():
    """Import the Google GenAI SDK lazily so the app can boot without it."""

    try:
        from google import genai
    except ImportError as exc:
        raise ImportError("The 'google-genai' package is required for Google GenAI support.") from exc

    return genai


# Split a configured Gemini endpoint into (base_url, api_version).
def _normalize_google_base_url(base_url: str) -> tuple[str, str | None]:
    """Split a configured Gemini endpoint into ``(base_url, api_version)``."""

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


# Return the normalized endpoint scope used for Gemini runtime caches.
def _get_runtime_scope() -> str:
    """Return the normalized endpoint scope used for Gemini runtime caches."""

    configured_base_url = settings.get_engine_url("google-genai")
    normalized_base_url, api_version = _normalize_google_base_url(configured_base_url)
    endpoint = str(normalized_base_url or configured_base_url or "").strip().rstrip("/")
    version = str(api_version or "").strip().lower()
    return f"{endpoint}|{version}"


# Return a stable non-reversible hash for the active Gemini API key.
def _get_api_key_hash() -> str:
    """Return a stable non-reversible hash for the active Gemini API key."""

    api_key = str(settings.get_google_genai_api_key() or "")
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


# Return the stable capability-cache key for one Gemini model.
def _get_model_capability_cache_key(model_name: str) -> tuple[str, str]:
    """Return the stable capability-cache key for one Gemini model."""

    return (_get_runtime_scope(), _normalize_model_identifier(model_name))


# Return the stable key-scoped availability-cache key for one Gemini model.
def _get_model_availability_cache_key(model_name: str) -> tuple[str, str, str]:
    """Return the stable key-scoped availability-cache key for one Gemini model."""

    return (_get_runtime_scope(), _get_api_key_hash(), _normalize_model_identifier(model_name))


# Return cached endpoint-scoped capabilities for one Gemini model.
def _get_cached_model_capabilities(model_name: str) -> dict[str, Any]:
    """Return cached endpoint-scoped capabilities for one Gemini model."""

    with _model_capability_cache_lock:
        return _clone_cache_payload(_model_capability_cache.get(_get_model_capability_cache_key(model_name)))


# Merge new endpoint-scoped capability facts into the runtime cache.
def _update_cached_model_capabilities(model_name: str, **updates: Any) -> dict[str, Any]:
    """Merge new endpoint-scoped capability facts into the runtime cache."""

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


# Return key-scoped availability info when the cache entry is still fresh.
def _get_cached_model_availability(model_name: str) -> dict[str, Any] | None:
    """Return key-scoped availability info when the cache entry is still fresh."""

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


# Cache whether one Gemini model is usable for the current key and endpoint.
def _set_cached_model_availability(model_name: str, available: bool, reason: str) -> dict[str, Any]:
    """Cache whether one Gemini model is usable for the current key and endpoint."""

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
    """Clear Gemini runtime caches. Intended for tests and debug flows."""

    with _model_capability_cache_lock:
        _model_capability_cache.clear()
    with _model_availability_cache_lock:
        _model_availability_cache.clear()


# Safely close a client instance when the SDK exposes close.
def _close_client(client: Any) -> None:
    """Safely close a client instance when the SDK exposes ``close``."""

    try:
        client.close()
    except Exception:
        pass


# Create a fresh Google GenAI client for the configured Gemini API endpoint.
def _get_client():
    """Create a fresh Google GenAI client for the configured Gemini API endpoint."""

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


# Return a normalized lower-case identifier for one key or token.
def _normalize_key_name(value: Any) -> str:
    """Return a normalized lower-case identifier for one key or token."""

    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


# Return a predictable boolean from JSON-like payloads.
def _bool_from_value(value: Any, default: bool = False) -> bool:
    """Return a predictable boolean from JSON-like payloads."""

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


# Convert a scalar into a positive integer when possible.
def _coerce_positive_int(value: Any) -> int | None:
    """Convert a scalar into a positive integer when possible."""

    if isinstance(value, bool):
        return None
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if numeric_value > 0 else None


# Convert a scalar or collection into a normalized list of strings.
def _coerce_string_list(value: Any) -> list[str]:
    """Convert a scalar or collection into a normalized list of strings."""

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


# Return one binary payload as a base64 string for JSON-safe storage.
def _encode_binary_payload(value: Any) -> str | None:
    """Return one binary payload as a base64 string for JSON-safe storage."""

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


# Decode one stored base64 payload back into raw bytes.
def _decode_binary_payload(value: Any) -> bytes | None:
    """Decode one stored base64 payload back into raw bytes."""

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
        try:
            padding = "=" * (-len(payload) % 4)
            return base64.b64decode(f"{payload}{padding}")
        except Exception:
            return None


# Drop service control tokens that should never reach the chat UI.
def _sanitize_generated_text(text: str) -> str:
    """Drop service control tokens that should never reach the chat UI."""

    cleaned = str(text or "")
    for pattern in CONTROL_TOKEN_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


# Convert SDK objects into plain Python containers.
def _to_plain_data(value: Any) -> Any:
    """Convert SDK objects into plain Python containers."""

    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (bytes, bytearray)):
        return _encode_binary_payload(value)

    if isinstance(value, dict):
        return {str(key): _to_plain_data(child) for key, child in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(child) for child in value]

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
                continue
        except Exception:
            continue
        return _to_plain_data(serialized)

    if hasattr(value, "__dict__"):
        return {
            str(key): _to_plain_data(child)
            for key, child in vars(value).items()
            if not str(key).startswith("_")
        }

    return str(value)


# Return a normalized model identifier for fuzzy Gemini comparisons.
def _normalize_model_identifier(value: Any) -> str:
    """Return a normalized model identifier for fuzzy Gemini comparisons."""

    normalized = str(value or "").strip().strip("/").lower()
    if "/models/" in normalized:
        normalized = normalized.rsplit("/models/", 1)[-1]
    if normalized.startswith("models/"):
        normalized = normalized.split("/", 1)[1]
    return normalized


# Return whether two Gemini model identifiers likely refer to the same model.
def _model_identifiers_match(expected: Any, actual: Any) -> bool:
    """Return whether two Gemini model identifiers likely refer to the same model."""

    normalized_expected = _normalize_model_identifier(expected)
    normalized_actual = _normalize_model_identifier(actual)
    if not normalized_expected or not normalized_actual:
        return False
    if normalized_expected == normalized_actual:
        return True

    expected_tail = normalized_expected.rsplit("/", 1)[-1]
    actual_tail = normalized_actual.rsplit("/", 1)[-1]
    return bool(expected_tail) and expected_tail == actual_tail


# Return the UI-facing model identifier for one Gemini model payload.
def _extract_model_name(payload: dict[str, Any]) -> str:
    """Return the UI-facing model identifier for one Gemini model payload."""

    for key in ("name", "id", "model"):
        value = payload.get(key)
        if value:
            normalized = _normalize_model_identifier(value)
            if normalized:
                return normalized
            return str(value)
    return ""


# Return base and tuned Gemini model payloads when available.
def _iter_model_payloads(client: Any) -> list[dict[str, Any]]:
    """Return base and tuned Gemini model payloads when available."""

    payloads: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    last_error: Exception | None = None
    query_attempts = (
        {"query_base": True},
        {"query_base": False},
        None,
    )

    for query_config in query_attempts:
        try:
            response = client.models.list(config=query_config) if query_config is not None else client.models.list()
            entries = list(response)
        except Exception as exc:
            last_error = exc
            continue

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


# Return capability metadata for one Gemini model.
def _get_model_payload(client: Any, model_name: str) -> dict[str, Any]:
    """Return capability metadata for one Gemini model."""

    requested_name = _normalize_model_identifier(model_name)
    last_error: Exception | None = None
    attempted_names: list[str] = []

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

    for payload in _iter_model_payloads(client):
        if _model_identifiers_match(requested_name, _extract_model_name(payload)):
            return payload

    if last_error is not None:
        raise last_error

    return {"name": requested_name or model_name}


# Split one free-form value into normalized capability tokens.
def _tokenize_string(value: Any) -> set[str]:
    """Split one free-form value into normalized capability tokens."""

    normalized = _normalize_key_name(value)
    if not normalized:
        return set()
    return {normalized, *[part for part in normalized.split("_") if part]}


# Return normalized tokens extracted from a nested model payload.
def _collect_capability_tokens(value: Any) -> set[str]:
    """Return normalized tokens extracted from a nested model payload."""

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


# Return one nested boolean-ish feature flag when it exists.
def _extract_feature_flag(value: Any, feature_names: set[str]) -> bool | None:
    """Return one nested boolean-ish feature flag when it exists."""

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


# Return normalized Gemini supported action names.
def _extract_supported_actions(raw_model: dict[str, Any]) -> set[str]:
    """Return normalized Gemini supported action names."""

    raw_actions = raw_model.get("supported_actions", raw_model.get("supportedGenerationMethods", []))
    if not isinstance(raw_actions, list):
        return set()
    return {_normalize_key_name(action) for action in raw_actions if _normalize_key_name(action)}


# Return the structured Google SDK error payload when present.
def _get_google_error_payload(exc: Exception) -> dict[str, Any]:
    """Return the structured Google SDK error payload when present."""

    details = getattr(exc, "details", None)
    return details if isinstance(details, dict) else {}


# Return structured per-error detail entries from one Google SDK error.
def _get_google_error_entries(exc: Exception) -> list[dict[str, Any]]:
    """Return structured per-error detail entries from one Google SDK error."""

    payload = _get_google_error_payload(exc)
    error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else payload
    detail_entries = error_payload.get("details") if isinstance(error_payload, dict) else None
    if not isinstance(detail_entries, list):
        return []
    return [entry for entry in detail_entries if isinstance(entry, dict)]


# Return the structured Google SDK error code when available.
def _get_google_error_code(exc: Exception) -> int | None:
    """Return the structured Google SDK error code when available."""

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


# Return the structured Google SDK status marker when available.
def _get_google_error_status(exc: Exception) -> str:
    """Return the structured Google SDK status marker when available."""

    status = getattr(exc, "status", None)
    if not status:
        payload = _get_google_error_payload(exc)
        error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        if isinstance(error_payload, dict):
            status = error_payload.get("status")
    return str(status or "").strip().upper()


# Return the structured Google SDK message when available.
def _get_google_error_message(exc: Exception) -> str:
    """Return the structured Google SDK message when available."""

    message = getattr(exc, "message", None)
    if not message:
        payload = _get_google_error_payload(exc)
        error_payload = payload.get("error") if isinstance(payload.get("error"), dict) else payload
        if isinstance(error_payload, dict):
            message = error_payload.get("message")
    return str(message or str(exc) or "").strip()


# Return flattened quota violations from one structured Google SDK error.
def _iter_quota_violations(exc: Exception) -> list[dict[str, Any]]:
    """Return flattened quota violations from one structured Google SDK error."""

    violations: list[dict[str, Any]] = []
    for entry in _get_google_error_entries(exc):
        raw_violations = entry.get("violations")
        if not isinstance(raw_violations, list):
            continue
        for violation in raw_violations:
            if isinstance(violation, dict):
                violations.append(violation)
    return violations


# Return whether the structured error likely refers to the provided model.
def _error_mentions_model(exc: Exception, model_name: str) -> bool:
    """Return whether the structured error likely refers to the provided model."""

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


# Return whether the error represents one Google quota/rate-limit failure.
def _is_resource_exhausted_error(exc: Exception) -> bool:
    """Return whether the error represents one Google quota/rate-limit failure."""

    return _get_google_error_code(exc) == 429 or _get_google_error_status(exc) == "RESOURCE_EXHAUSTED"


# Return whether the error indicates zero entitlement for the current key/model.
def _is_zero_quota_error(exc: Exception, model_name: str) -> bool:
    """Return whether the error indicates zero entitlement for the current key/model."""

    if not _is_resource_exhausted_error(exc) or not _error_mentions_model(exc, model_name):
        return False

    message = _get_google_error_message(exc).lower()
    if "limit: 0" in message or "limit 0" in message:
        return True

    return False


# Return whether the backend says the model cannot serve generateContent.
def _is_generate_content_unsupported_error(exc: Exception, model_name: str) -> bool:
    """Return whether the backend says the model cannot serve ``generateContent``."""

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


# Return whether one backend error explicitly says tools are unsupported.
def _is_tool_unsupported_error(exc: Exception) -> bool:
    """Return whether one backend error explicitly says tools are unsupported."""

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


# Return whether the backend explicitly rejected Gemini thinking_level.
def _is_thinking_level_unsupported_error(exc: Exception) -> bool:
    """Return whether the backend explicitly rejected Gemini ``thinking_level``."""

    code = _get_google_error_code(exc)
    status = _get_google_error_status(exc)
    if code != 400 and status != "INVALID_ARGUMENT":
        return False

    message = _normalize_key_name(_get_google_error_message(exc))
    return (
        "thinking_level" in message and any(marker in message for marker in ("not_supported", "unsupported"))
    ) or "thinking_level_is_not_supported" in message


# Update runtime caches from one structured Google SDK error.
def _learn_from_google_error(model_name: str, exc: Exception) -> str | None:
    """Update runtime caches from one structured Google SDK error."""

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


# Return normalized reasoning level options from one nested metadata value.
def _coerce_reasoning_level_options(value: Any) -> list[str]:
    """Return normalized reasoning level options from one nested metadata value."""

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


# Search nested Gemini metadata for explicit reasoning-level options.
def _extract_reasoning_level_options(value: Any) -> list[str]:
    """Search nested Gemini metadata for explicit reasoning-level options."""

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


# Return endpoint-scoped Gemini capabilities for one model.
def _build_model_capability_snapshot(
    model_name: str,
    raw_model: dict[str, Any],
) -> dict[str, Any]:
    """Return endpoint-scoped Gemini capabilities for one model."""

    cached_capabilities = _get_cached_model_capabilities(model_name)
    capability_tokens = _collect_capability_tokens(raw_model)
    supported_actions = _extract_supported_actions(raw_model)
    normalized_model_name = str(model_name or raw_model.get("name") or "").strip().lower()
    known_gemini_multimodal = (
        "gemini-" in normalized_model_name
        and not any(marker in normalized_model_name for marker in ("embedding", "imagen", "veo"))
    )

    explicit_tool_support = _extract_feature_flag(raw_model, TOOL_CAPABILITY_NAMES)
    explicit_vision_support = _extract_feature_flag(raw_model, VISION_CAPABILITY_NAMES)
    explicit_audio_input_support = _extract_feature_flag(raw_model, AUDIO_INPUT_CAPABILITY_NAMES)
    explicit_video_input_support = _extract_feature_flag(raw_model, VIDEO_INPUT_CAPABILITY_NAMES)
    explicit_reasoning_support = _extract_feature_flag(raw_model, REASONING_CAPABILITY_NAMES)
    explicit_reasoning_level_options = _extract_reasoning_level_options(raw_model)

    supports_generate_content = cached_capabilities.get("supports_generate_content")
    if not isinstance(supports_generate_content, bool):
        # The public model listing does not always expose one clean boolean, so
        # fall back to supported action names when the cache is still empty.
        supports_generate_content = any("generatecontent" in action for action in supported_actions)

    supports_tool_calling = cached_capabilities.get("supports_tool_calling")
    if not isinstance(supports_tool_calling, bool):
        supports_tool_calling = (
            bool(explicit_tool_support)
            if explicit_tool_support is not None
            else bool(capability_tokens & TOOL_CAPABILITY_NAMES) or supports_generate_content
        )

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


# Return one tool-call arguments payload as a dictionary.
def _normalize_tool_call_arguments(raw_arguments: Any) -> dict[str, Any]:
    """Return one tool-call arguments payload as a dictionary."""

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


# Decode one base64-encoded image attachment.
def _decode_image_bytes(image_base64: str) -> bytes:
    """Decode one base64-encoded image attachment."""

    payload = str(image_base64 or "").strip()
    if not payload:
        return b""
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    return base64.b64decode(payload)


# Convert tool result content into Gemini function response JSON.
def _normalize_tool_response_payload(content: Any) -> dict[str, Any]:
    """Convert tool result content into Gemini function response JSON."""

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


# Normalize OpenAI-style or Gemini-style content parts into Gemini request parts.
def _normalize_google_request_parts(raw_parts: Any) -> list[dict[str, Any]]:
    """Normalize OpenAI-style or Gemini-style content parts into Gemini request parts."""

    if isinstance(raw_parts, dict):
        raw_parts = [raw_parts]
    if not isinstance(raw_parts, list):
        return []

    normalized_parts: list[dict[str, Any]] = []
    for raw_part in raw_parts:
        if raw_part is None:
            continue

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


# Serialize one assistant response part for transcript replay.
def _build_google_history_part(raw_part: Any, *, include_text: bool = True) -> dict[str, Any] | None:
    """Serialize one assistant response part for transcript replay."""

    part = _to_plain_data(raw_part)
    if not isinstance(part, dict):
        return None

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
    """Return whether preserved Gemini history already includes function calls."""

    return any(isinstance(part, dict) and isinstance(part.get("function_call"), dict) for part in parts)


# Return whether preserved Gemini parts contain a function call without its signature.
def _google_parts_have_unsigned_function_call(raw_parts: Any) -> bool:
    """Return whether preserved Gemini parts contain a function call without its signature."""

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


# Convert ASLM chat messages into Gemini-compatible contents.
def _build_google_contents(messages: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    """Convert ASLM chat messages into Gemini-compatible contents."""

    system_fragments: list[str] = []
    contents: list[dict[str, Any]] = []
    image_bytes_cache: dict[str, bytes] = {}
    skip_orphan_tool_responses = False

    for message_index, message in enumerate(messages, start=1):
        role = str(message.get("role", "user")).lower()
        raw_content = message.get("content", "")
        content = str(raw_content or "") if not isinstance(raw_content, (list, dict)) else ""
        images = [str(item) for item in (message.get("images") or []) if str(item).strip()]
        image_mime_types = message.get("image_mime_types") or []
        media = message.get("media") if isinstance(message.get("media"), list) else []

        if role == "system":
            if content:
                system_fragments.append(content)
            continue

        parts: list[dict[str, Any]] = []

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

        if not parts:
            continue

        contents.append(
            {
                "role": "model" if role == "assistant" else "user",
                "parts": parts,
            }
        )

    return "\n\n".join(fragment for fragment in system_fragments if fragment).strip(), contents


# Return Gemini-compatible tool declarations for one or more servers.
def _build_google_tools(
    tool_server_ids: list[str],
    model_name: str,
    tool_context: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Return Gemini-compatible tool declarations for one or more servers."""

    try:
        model_vision = bool(_get_cached_model_capabilities(model_name).get("supports_vision"))
    except Exception:  # noqa: BLE001 — capability lookup must never break the loop
        model_vision = False
    openai_tools, tool_lookup = tool_registry.build_ollama_tools(
        tool_server_ids,
        engine="google-genai",
        model_vision=model_vision,
        model_name=model_name,
        tool_source_map=tool_context.get("tool_source_map") if isinstance(tool_context, dict) else None,
        allowed_tool_aliases=tool_context.get("allowed_tool_aliases") if isinstance(tool_context, dict) else None,
        instant_mode=bool(tool_context.get("instant_mode")) if isinstance(tool_context, dict) else False,
        instant_search_batch_size=tool_context.get("instant_search_batch_size") if isinstance(tool_context, dict) else None,
    )
    if not openai_tools:
        return [], tool_lookup

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


# Convert a stop payload into the Gemini list form.
def _coerce_stop_sequences(value: Any) -> list[str] | None:
    """Convert a stop payload into the Gemini list form."""

    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return [stripped] if stripped else None
    if isinstance(value, list):
        sequences = [str(item).strip() for item in value if str(item).strip()]
        return sequences or None
    return [str(value)]


# Split generic generation options into a Gemini GenerateContentConfig payload.
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
    """Split generic generation options into a Gemini ``GenerateContentConfig`` payload."""

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

    if thinking_config:
        thinking_config.setdefault("include_thoughts", True)
        config["thinking_config"] = thinking_config

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


# Return whether one Gemini request config currently sends thinking_level.
def _config_uses_thinking_level(config: dict[str, Any]) -> bool:
    """Return whether one Gemini request config currently sends ``thinking_level``."""

    thinking_config = config.get("thinking_config")
    return isinstance(thinking_config, dict) and thinking_config.get("thinking_level") is not None


# Return a cloned Gemini request config without thinking_level.
def _strip_thinking_level_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a cloned Gemini request config without ``thinking_level``."""

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


# Apply learned runtime capability overrides to one Gemini request config.
def _apply_learned_request_preferences(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Apply learned runtime capability overrides to one Gemini request config."""

    cached_capabilities = _get_cached_model_capabilities(model_name)
    if cached_capabilities.get("supports_think_level") is False:
        return _strip_thinking_level_from_config(config)
    return dict(config or {})


# Return a request config clone with function tools disabled.
def _strip_tools_from_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a request config clone with function tools disabled."""

    normalized_config = dict(config or {})
    normalized_config.pop("tools", None)
    normalized_config.pop("automatic_function_calling", None)
    normalized_config.pop("tool_config", None)
    return normalized_config


# Serialize one tool invocation so the UI can render it during streaming.
def _build_tool_event(tool_lookup: dict[str, dict[str, Any]], tool_call: dict[str, Any]) -> dict[str, Any]:
    """Serialize one tool invocation so the UI can render it during streaming."""
    return tool_registry.build_tool_event(tool_lookup, tool_call)


# Build a tool message payload for the shared ASLM transcript.
def _build_tool_message(
    tool_name: str,
    tool_call_id: str,
    content: str | dict[str, Any],
    tool_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a tool message payload for the shared ASLM transcript."""

    model_content, tool_extras = tool_registry.split_tool_result_payload(content)
    payload: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": model_content,
    }
    payload.update(tool_extras)

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


# Split streamed reasoning text from visible content.
class _ReasoningTextParser:

    # Initialize the parser state.
    def __init__(self, tag_pairs: tuple[tuple[str, str], ...] = REASONING_TAG_PAIRS) -> None:
        self._start_tags = [start for start, _end in tag_pairs]
        self._end_tags = [end for _start, end in tag_pairs]
        self._in_reasoning = False
        self._pending = ""
        self._visible_boundary_char: str | None = None

    # Find next tag.
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

    # Accept opening tags only at the start of text or after a clear separator.
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

    # Split possible tag prefix.
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

    # Return parsed (thinking, visible_content) fragments.
    def feed(self, content: str, reasoning_type: str | None = None) -> tuple[str, str]:
        """Return parsed ``(thinking, visible_content)`` fragments."""

        thinking_parts: list[str] = []
        content_parts: list[str] = []

        normalized_reasoning_type = _normalize_key_name(reasoning_type)
        if normalized_reasoning_type in {"reasoning", "thought", "thinking"}:
            return str(content or ""), ""

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

    # Flush any pending partial fragment at the end of the stream.
    def flush(self) -> tuple[str, str]:
        """Flush any pending partial fragment at the end of the stream."""

        pending = self._pending
        self._pending = ""
        if not pending:
            return "", ""
        if self._in_reasoning:
            return pending, ""
        self._remember_visible_text(pending)
        return "", pending


# Return the first-candidate response parts from one SDK response object.
def _get_response_parts(response: Any) -> list[Any]:
    """Return the first-candidate response parts from one SDK response object."""

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


# Parse visible text, reasoning text, and tool calls from one SDK response.
def _parse_google_response_parts(
    parser: _ReasoningTextParser,
    parts: list[Any],
) -> tuple[str, str, list[dict[str, Any]]]:
    """Parse visible text, reasoning text, and tool calls from one SDK response."""

    thinking_parts: list[str] = []
    content_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []

    for raw_part in parts:
        part = _to_plain_data(raw_part)
        if not isinstance(part, dict):
            continue

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


# Stream one Gemini round and return the assembled assistant message.
def _stream_google_round(
    client: Any,
    model_name: str,
    contents: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    stream: bool = True,
):
    """Stream one Gemini round and return the assembled assistant message."""

    assistant_content = ""
    assistant_thinking = ""
    tool_calls: list[dict[str, Any]] = []
    seen_tool_call_signatures: set[str] = set()
    assistant_history_parts: list[dict[str, Any]] = []
    parser = _ReasoningTextParser()
    yielded_chunk = False

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

    if not yielded_chunk and (assistant_content or assistant_thinking):
        yield {"message": assistant_message}

    return assistant_message


# Yield every chunk from a round stream and return the final assistant message.
def _yield_stream_round(round_stream):
    """Yield every chunk from a round stream and return the final assistant message."""

    while True:
        try:
            yield next(round_stream)
        except StopIteration as stop:
            return stop.value or {"role": "assistant", "content": ""}


# Convert one assembled assistant message back into Gemini conversation content.
def _assistant_message_to_content(assistant_message: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one assembled assistant message back into Gemini conversation content."""

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


# Resolve local tools through Gemini function-calling while keeping ASLM transcript markers.
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
    """Resolve local tools through Gemini function-calling while keeping ASLM transcript markers."""

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

    max_tool_rounds = tool_registry.tool_round_limit_from_context(tool_context, MAX_TOOL_ROUNDS)
    for round_index in range(max_tool_rounds):
        # Re-apply learned preferences each round because earlier failures may
        # have updated the runtime capability cache.
        request_config = _apply_learned_request_preferences(model_name, request_config)
        assistant_message = yield from _yield_stream_round(
            _stream_google_round(client, model_name, conversation, request_config, stream=stream)
        )

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

        tool_calls = tool_registry.prepare_tool_calls(tool_lookup, tool_calls)
        tool_calls = tool_registry.limit_tool_calls_from_context(tool_context, tool_calls, tool_lookup)

        for tool_call in tool_calls:
            satisfied_server_id = tool_registry.tool_server_id_for_alias(tool_lookup, tool_call["name"])
            if satisfied_server_id:
                satisfied_tool_server_ids.add(satisfied_server_id)

        canonical_assistant_message = tool_registry.canonicalize_transcript_tool_calls(
            assistant_message, tool_calls
        )
        assistant_content = _assistant_message_to_content(canonical_assistant_message)
        if assistant_content is not None:
            conversation.append(assistant_content)
        yield {"transcript_message": canonical_assistant_message}

        tool_response_parts: list[dict[str, Any]] = []
        tool_events = [_build_tool_event(tool_lookup, tool_call) for tool_call in tool_calls]
        for _i, _ev in enumerate(tool_events):
            _ev["alias"] = f"{_ev['alias']}__{_i}"
        yield {"tool_events": tool_events}

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

        conversation = tool_registry.maybe_compact_tool_conversation(
            tool_context, conversation, provider_format="google"
        )

    yield {"message": {"content": "[Error during generation: tool loop exceeded the safety limit.]"}}


# Return whether the current conversation already contains tool state.
def _conversation_uses_tools(messages: list[dict[str, Any]]) -> bool:
    """Return whether the current conversation already contains tool state."""

    for message in messages:
        if str(message.get("role", "")).lower() == "tool":
            return True
        if message.get("tool_calls"):
            return True
    return False


# Public adapter API.

# Return models exposed by the configured Gemini API endpoint.
def get_models() -> list[Any]:
    """Return models exposed by the configured Gemini API endpoint."""

    client = _get_client()
    try:
        models: list[dict[str, Any]] = []
        for payload in _iter_model_payloads(client):
            model_name = _extract_model_name(payload)
            if not model_name:
                continue
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
    """Raise because Gemini API models are remote and cannot be downloaded locally."""

    raise NotImplementedError("Google GenAI models are remote and cannot be downloaded locally.")


# Return capability metadata for one model from Google GenAI.
def get_model_settings(model_name: str) -> dict[str, Any]:
    """Return capability metadata for one model from Google GenAI."""

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


# Generate a streamed or non-streamed response through Google GenAI.
def generate(model_name: str, messages: list[dict[str, Any]], **kwargs: Any):
    """Generate a streamed or non-streamed response through Google GenAI."""

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
