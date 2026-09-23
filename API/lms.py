# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import threading
import unicodedata
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request

from API import mcp as tool_registry
from Settings import settings
from Settings.proxy_policy import urlopen_with_loopback_bypass

logger = logging.getLogger(__name__)

_abort_event = threading.Event()
_model_state_lock = threading.Lock()
_synced_operation_defaults_signatures: dict[str, str] = {}


# Stop the active LM Studio generation.
def abort_generation() -> None:
    """Signal the active LM Studio generation to stop."""

    _abort_event.set()


INTERNAL_CHAT_KEYS = {"system", "prompt", "tool_id", "tool_server_id", "tool_server_ids", "tool_context", "think", "think_level", "engine"}
MAX_TOOL_ROUNDS = 100
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
REASONING_LEVEL_VALUES = {"minimal", "low", "medium", "high", "xhigh"}
OPENAI_DIRECT_OPTION_ALIASES = {
    "num_predict": "max_tokens",
}
OPENAI_DIRECT_OPTION_KEYS = {
    "frequency_penalty",
    "logprobs",
    "logit_bias",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "n",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "stream_options",
    "temperature",
    "top_logprobs",
    "tool_choice",
    "tools",
    "top_p",
    "user",
    "verbosity",
}


# Build LM Studio SDK clients.
# Normalize the configured API host.
def _extract_api_host(raw_address: str) -> str:
    """Normalize the configured LM Studio address into a client host value."""

    parsed = urlparse(raw_address)
    if parsed.scheme and parsed.netloc:
        return parsed.netloc
    if parsed.scheme and parsed.path:
        return parsed.path
    return raw_address.strip().rstrip("/")


# Import the LM Studio SDK.
def _get_sdk():
    """Import the LM Studio SDK lazily so the app can boot without it."""

    try:
        import lmstudio as lms
    except ImportError as exc:
        raise ImportError("The 'lmstudio' package is required for LM Studio support.") from exc

    return lms


# Create a fresh SDK client.
def _get_client():
    """Create a fresh LM Studio client for the configured server."""

    lms = _get_sdk()
    api_host = _extract_api_host(settings.get_engine_url("lms"))
    return lms, lms.Client(api_host)


# Close one SDK client safely.
def _close_client(client: Any) -> None:
    """Safely close a client instance when the SDK exposes ``close``."""

    try:
        client.close()
    except Exception:
        pass


# Get one model handle.
def _get_model_handle(client: Any, model_name: str):
    """Return an SDK handle for one already-loaded model."""

    return client.llm.model(model_name)


# Extract one model name.
def _coerce_model_name(entry: Any) -> str:
    """Extract a stable model identifier from LM Studio SDK objects."""

    for attr_name in ("model_key", "model", "identifier", "id", "display_name"):
        value = getattr(entry, attr_name, None)
        if value:
            return str(value)

    nested_info = getattr(entry, "info", None)
    if nested_info is not None and nested_info is not entry:
        nested_name = _coerce_model_name(nested_info)
        if nested_name:
            return nested_name

    get_info = getattr(entry, "get_info", None)
    if not callable(get_info):
        return ""

    try:
        info = get_info()
    except Exception:
        return ""

    if hasattr(info, "to_dict"):
        info = info.to_dict()
    if not isinstance(info, dict):
        return ""

    for key in ("model_key", "modelKey", "display_name", "displayName", "identifier", "id"):
        value = info.get(key)
        if value:
            return str(value)

    return ""


# Collect unique model names.
def _collect_unique_model_names(entries: list[Any]) -> list[str]:
    """Return unique model names while preserving the original order."""

    unique_names: list[str] = []
    seen_names: set[str] = set()

    for entry in entries:
        model_name = _coerce_model_name(entry)
        if not model_name or model_name in seen_names:
            continue
        seen_names.add(model_name)
        unique_names.append(model_name)

    return unique_names


# List models through one client method.
def _list_models_with_client(method_name: str) -> list[Any]:
    """Call one LM Studio listing method and always close the client."""

    _lms, client = _get_client()
    try:
        return list(getattr(client, method_name)())
    finally:
        _close_client(client)


# Normalize LM Studio metadata.
# Coerce one boolean-like value.
def _bool_from_value(value: Any, default: bool = False) -> bool:
    """Return a predictable boolean from SDK and dict payloads."""

    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return bool(value)


# Build one config signature.
def _config_signature(config: dict[str, Any]) -> str:
    """Return a stable hashable signature for one LM Studio config."""

    try:
        return json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(config or {})


# Normalize one model identifier.
def _normalize_model_identifier(value: Any) -> str:
    """Return a normalized model identifier for fuzzy LM Studio comparisons."""

    return str(value or "").strip().strip("/").lower()


# Compare two model identifiers.
def _model_identifiers_match(expected: Any, actual: Any) -> bool:
    """Return whether two LM Studio model identifiers likely refer to the same model."""

    normalized_expected = _normalize_model_identifier(expected)
    normalized_actual = _normalize_model_identifier(actual)
    if not normalized_expected or not normalized_actual:
        return False
    if normalized_expected == normalized_actual:
        return True

    expected_tail = normalized_expected.rsplit("/", 1)[-1]
    actual_tail = normalized_actual.rsplit("/", 1)[-1]
    return bool(expected_tail) and expected_tail == actual_tail


# List loaded model names.
def _list_loaded_model_names(client: Any) -> list[str]:
    """Return loaded model names visible to the current LM Studio client."""

    try:
        loaded_models = list(client.list_loaded_models())
    except Exception:
        return []

    return _collect_unique_model_names(loaded_models)


# Ensure one model is loaded.
def _assert_model_is_loaded(client: Any, model_name: str) -> None:
    """Raise when the requested model is not already loaded in LM Studio."""

    loaded_model_names = _list_loaded_model_names(client)
    if any(_model_identifiers_match(model_name, loaded_name) for loaded_name in loaded_model_names):
        return

    if loaded_model_names:
        raise RuntimeError(
            f"LM Studio model is not loaded: {model_name}. "
            f"Loaded models: {', '.join(loaded_model_names)}"
        )

    raise RuntimeError(
        f"LM Studio model is not loaded: {model_name}. "
        "Load the model in LM Studio first."
    )


# Load one model through LM Studio just-in-time loading.
def _ensure_model_loaded(client: Any, model_name: str) -> None:
    """Get-or-load the requested model so generation can run on downloaded models.

    LM Studio serves any downloaded model through JIT loading, so request a
    handle (which loads the model when needed) instead of failing when it is
    merely downloaded but not yet loaded.
    """

    try:
        client.llm.model(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"LM Studio could not load model: {model_name}. "
            "Confirm it is downloaded and the LM Studio server is running. "
            f"({exc})"
        ) from exc


# Serialize one model info payload.
def _serialize_model_info(info: Any) -> dict[str, Any]:
    """Convert SDK model info into a JSON-compatible dictionary."""

    if hasattr(info, "to_dict"):
        payload = info.to_dict()
        if isinstance(payload, dict):
            return payload

    nested_info = getattr(info, "info", None)
    if nested_info is not None and nested_info is not info:
        payload = _serialize_model_info(nested_info)
        if payload:
            return payload

    get_info = getattr(info, "get_info", None)
    if callable(get_info):
        try:
            payload = _serialize_model_info(get_info())
            if payload:
                return payload
        except Exception:
            pass

    if isinstance(info, dict):
        return info

    payload: dict[str, Any] = {}
    field_map = {
        "modelKey": ("model_key", "model"),
        "displayName": ("display_name", "name"),
        "identifier": ("identifier", "id"),
        "vision": ("vision",),
        "trainedForToolUse": ("trained_for_tool_use",),
        "maxContextLength": ("max_context_length",),
        "contextLength": ("context_length",),
        "paramsString": ("params_string",),
        "architecture": ("architecture",),
        "format": ("format",),
        "path": ("path",),
        "sizeBytes": ("size_bytes",),
    }
    for key, attr_names in field_map.items():
        for attr_name in attr_names:
            value = getattr(info, attr_name, None)
            if value is None:
                continue
            payload[key] = str(value) if key == "path" else value
            break

    return payload


# Read capabilities independently of the loaded-model lifecycle.
def _get_catalog_model_info(model_name: str) -> dict[str, Any]:
    """Read public capabilities for loaded and unloaded models without loading them."""
    base_url = settings.get_engine_url("lms").strip().rstrip("/")
    if not base_url.startswith(("http://", "https://")):
        base_url = f"http://{base_url}"
    base_url = re.sub(r"/(?:api/)?v[01]$", "", base_url)
    request = Request(f"{base_url}/api/v1/models", headers={"Accept": "application/json"})
    try:
        with urlopen_with_loopback_bypass(request, timeout=5) as response:
            catalog = json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError):
        logger.debug("[LM Studio API] Native model catalog is unavailable", exc_info=True)
        return {}

    models = catalog.get("models", []) if isinstance(catalog, dict) else []
    candidates = []
    for model in models if isinstance(models, list) else []:
        if not isinstance(model, dict) or model.get("type") != "llm":
            continue
        names = [model.get("key")]
        names.extend(model.get("variants") or [])
        names.extend(instance.get("id") for instance in model.get("loaded_instances", []) if isinstance(instance, dict))
        if any(_normalize_model_identifier(model_name) == _normalize_model_identifier(name) for name in names if name):
            return model
        if _model_identifiers_match(model_name, model.get("key")):
            candidates.append(model)
    return candidates[0] if len(candidates) == 1 else {}


# Fetch one raw model info payload.
def _get_raw_model_info(client: Any, model_name: str) -> dict[str, Any]:
    """Return the raw LM Studio model-info payload when the SDK exposes it."""

    get_raw_info = getattr(client.llm, "_get_api_model_info", None)
    if callable(get_raw_info):
        try:
            payload = get_raw_info(model_name)
            if isinstance(payload, dict):
                return payload
        except Exception:
            pass

    _info, payload = _get_model_info(client, model_name)
    return payload if isinstance(payload, dict) else {}


# Remove empty nested config values.
def _clean_nested_config(value: Any) -> Any:
    """Drop empty nested config values while preserving scalars."""

    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, child in value.items():
            normalized_child = _clean_nested_config(child)
            if normalized_child in ({}, [], "", None):
                continue
            normalized[str(key)] = normalized_child
        return normalized

    if isinstance(value, list):
        normalized_list = [_clean_nested_config(child) for child in value]
        return [child for child in normalized_list if child not in ({}, [], "", None)]

    return value


# Normalize one model config payload.
def _normalize_model_config(config: Any) -> dict[str, Any]:
    """Return a normalized dictionary for raw LM Studio config sections."""

    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict):
        return {}
    normalized = _clean_nested_config(config)
    return normalized if isinstance(normalized, dict) else {}


# Merge nested dictionaries.
def _merge_nested_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge two dictionaries without mutating either input."""

    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


# Resolve the LM Studio home directory.
def _get_lmstudio_home() -> str:
    """Return the local LM Studio home directory."""

    user_home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    return os.path.join(user_home, ".lmstudio")


# Read one JSON file.
def _read_json_file(path: str) -> Any:
    """Return decoded JSON from one file path when it exists."""

    if not path or not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# Normalize LM Studio KV config fields.
# Unwrap one KV field value.
def _unwrap_kv_field_value(value: Any) -> Any:
    """Convert LM Studio checkbox-like KV values into plain Python values."""

    if isinstance(value, dict) and "checked" in value:
        if not _bool_from_value(value.get("checked"), default=False):
            return None
        return value.get("value")
    return value


# Set one nested client value.
def _set_nested_client_value(target: dict[str, Any], path: str, value: Any) -> None:
    """Assign one normalized value into a nested client-config dictionary."""

    parts = [part for part in str(path or "").split(".") if part]
    if not parts:
        return

    cursor = target
    for part in parts[:-1]:
        existing = cursor.get(part)
        if not isinstance(existing, dict):
            existing = {}
            cursor[part] = existing
        cursor = existing
    cursor[parts[-1]] = value


# Apply one KV mapping.
def _apply_kv_mapping(target: dict[str, Any], mapping: Any, parts: list[str], value: Any) -> bool:
    """Apply one LM Studio KV field through the SDK mapping when possible."""

    if not parts or not isinstance(mapping, dict):
        return False

    node = mapping.get(parts[0])
    if node is None:
        return False

    if len(parts) == 1:
        update_client_config = getattr(node, "update_client_config", None)
        if callable(update_client_config):
            update_client_config(target, value)
            return True
        return False

    if isinstance(node, dict):
        return _apply_kv_mapping(target, node, parts[1:], value)

    return False


# Build one fallback KV path.
def _kv_field_fallback_path(prefix: str, remainder: str) -> str:
    """Return a best-effort client path for one unmapped LM Studio KV field."""

    if prefix == "llm.load" and remainder.startswith("llama."):
        remainder = remainder[len("llama.") :]
    if prefix == "llm.prediction" and remainder.startswith("llama."):
        remainder = remainder[len("llama.") :]
    if prefix == "llm.load" and remainder.startswith("acceleration."):
        remainder = remainder.replace("acceleration.offloadRatio", "gpu.ratio", 1)

    if remainder.startswith("ext.virtualModel.customField."):
        return remainder

    if "." not in remainder:
        return remainder

    return remainder.split(".")[-1]


# Convert KV fields into client config.
def _kv_fields_to_client_config(section_key: str, fields: list[dict[str, Any]]) -> dict[str, Any]:
    """Convert LM Studio KV field lists into the adapter's client-config shape."""

    if not isinstance(fields, list) or not fields:
        return {}

    try:
        from lmstudio._kv_config import SUPPORTED_SERVER_KEYS
    except Exception:
        SUPPORTED_SERVER_KEYS = {}

    mapping = SUPPORTED_SERVER_KEYS.get(section_key, {}) if isinstance(SUPPORTED_SERVER_KEYS, dict) else {}
    normalized: dict[str, Any] = {}

    for field in fields:
        if not isinstance(field, dict):
            continue

        raw_key = str(field.get("key") or "").strip()
        if not raw_key:
            continue

        raw_value = field.get("value")
        if raw_value is None:
            continue

        if raw_key == section_key:
            continue

        matched_prefix = ""
        if raw_key.startswith(f"{section_key}."):
            matched_prefix = section_key
        elif raw_key.startswith("llm.load.") and section_key == "llm.load":
            matched_prefix = "llm.load"
        elif raw_key.startswith("llm.prediction.") and section_key == "llm.prediction":
            matched_prefix = "llm.prediction"

        if matched_prefix:
            remainder = raw_key[len(matched_prefix) + 1 :]
            # Prefer SDK-provided mappings when available, but keep a fallback
            # path so custom or newer fields still reach the runtime config.
            if _apply_kv_mapping(normalized, mapping, remainder.split("."), raw_value):
                continue
            value = _clean_nested_config(_unwrap_kv_field_value(raw_value))
            if value in ({}, [], "", None):
                continue
            fallback_path = _kv_field_fallback_path(matched_prefix, remainder)
            _set_nested_client_value(normalized, fallback_path, value)
            continue

        value = _clean_nested_config(_unwrap_kv_field_value(raw_value))
        if value in ({}, [], "", None):
            continue
        if raw_key.startswith("ext.virtualModel.customField."):
            normalized[raw_key] = value
            continue

        _set_nested_client_value(normalized, raw_key, value)

    return _normalize_model_config(normalized)


# Read and sync persisted LM Studio defaults.
# Read one model index record.
def _get_model_index_record(model_name: str) -> dict[str, Any]:
    """Return the local LM Studio model-index record for one model identifier."""

    cache_path = os.path.join(_get_lmstudio_home(), ".internal", "model-index-cache.json")
    payload = _read_json_file(cache_path)
    records = payload.get("models", []) if isinstance(payload, dict) else []
    if not isinstance(records, list):
        return {}

    exact_match = None
    for record in records:
        if not isinstance(record, dict):
            continue
        if str(record.get("indexedModelIdentifier") or "") == model_name:
            exact_match = record
            if str(record.get("defaultIdentifier") or "") == model_name:
                return record
    return exact_match or {}


# Read disk operation defaults.
def _get_disk_model_operation_defaults(model_name: str) -> dict[str, Any]:
    """Return saved LM Studio operation defaults for one model."""

    owner, separator, name = model_name.partition("/")
    if not separator or not owner or not name:
        return {}

    config_path = os.path.join(
        _get_lmstudio_home(),
        ".internal",
        "user-concrete-model-default-config",
        owner,
        f"{name}.json",
    )
    payload = _read_json_file(config_path)
    if not isinstance(payload, dict):
        return {}

    operation_fields = payload.get("operation", {}).get("fields", []) if isinstance(payload.get("operation"), dict) else []
    return _kv_fields_to_client_config(
        "llm.prediction",
        operation_fields if isinstance(operation_fields, list) else [],
    )


# Build the default-config path.
def _get_model_default_config_path(model_name: str) -> str:
    """Return the LM Studio user default-config path for one model."""

    owner, separator, name = model_name.partition("/")
    if not separator or not owner or not name:
        return ""

    return os.path.join(
        _get_lmstudio_home(),
        ".internal",
        "user-concrete-model-default-config",
        owner,
        f"{name}.json",
    )


# Sync operation defaults to disk.
def _sync_operation_defaults_to_disk(model_name: str, operation_config: dict[str, Any] | None) -> None:
    """Persist custom LM Studio operation fields so the server applies them during generation."""

    if not isinstance(operation_config, dict):
        return

    custom_fields = []
    for key in sorted(operation_config.keys(), key=str.casefold):
        normalized_key = str(key or "").strip()
        if not normalized_key.startswith("ext.virtualModel.customField."):
            continue
        custom_fields.append({"key": normalized_key, "value": operation_config[key]})

    # Skip disk writes when the generated custom-field payload is unchanged.
    signature = _config_signature({"fields": custom_fields})
    with _model_state_lock:
        if _synced_operation_defaults_signatures.get(model_name) == signature:
            return

    config_path = _get_model_default_config_path(model_name)
    if not config_path:
        return

    payload = _read_json_file(config_path)
    if not isinstance(payload, dict):
        payload = {"preset": "", "operation": {"fields": []}, "load": {"fields": []}}

    operation_payload = payload.get("operation")
    if not isinstance(operation_payload, dict):
        operation_payload = {"fields": []}
        payload["operation"] = operation_payload

    existing_fields = operation_payload.get("fields", [])
    if not isinstance(existing_fields, list):
        existing_fields = []

    preserved_fields = []
    # Only rewrite the extension-backed custom fields and preserve everything
    # else exactly as LM Studio already stored it.
    for field in existing_fields:
        if not isinstance(field, dict):
            continue
        field_key = str(field.get("key") or "").strip()
        if field_key.startswith("ext.virtualModel.customField."):
            continue
        preserved_fields.append(field)

    next_fields = preserved_fields + custom_fields
    if next_fields == existing_fields:
        with _model_state_lock:
            _synced_operation_defaults_signatures[model_name] = signature
        return

    operation_payload["fields"] = next_fields
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with _model_state_lock:
        _synced_operation_defaults_signatures[model_name] = signature


# Derive runtime capabilities and request options.
# Extract reasoning settings.
def _extract_reasoning_settings(operation_defaults: dict[str, Any]) -> tuple[Any | None, Any | None]:
    """Return canonical reasoning toggle and level defaults from operation config."""

    think_value = operation_defaults.get("think")
    think_level_value = (
        operation_defaults.get("reasoning_effort")
        or operation_defaults.get("think_level")
        or operation_defaults.get("thinking_level")
    )

    reasoning_config = operation_defaults.get("reasoning")
    if isinstance(reasoning_config, dict):
        if think_value is None and "enabled" in reasoning_config:
            think_value = reasoning_config.get("enabled")
        if think_level_value is None and "effort" in reasoning_config:
            think_level_value = reasoning_config.get("effort")

    return think_value, think_level_value


# Collect reasoning field hints.
def _collect_reasoning_field_hints(field_definition: dict[str, Any]) -> str:
    """Build one searchable string for reasoning custom-field detection."""

    parts = [
        str(field_definition.get("key") or ""),
        str(field_definition.get("displayName") or ""),
        str(field_definition.get("description") or ""),
    ]
    for effect in field_definition.get("effects") or []:
        if not isinstance(effect, dict):
            continue
        parts.append(str(effect.get("variable") or ""))
        parts.append(str(effect.get("content") or ""))
    return " ".join(parts).strip().lower()


# Detect custom reasoning fields.
def _detect_reasoning_custom_fields(raw_info: dict[str, Any]) -> tuple[str | None, Any | None, str | None, Any | None]:
    """Infer reasoning toggle/level custom fields from LM Studio model metadata."""

    toggle_param_name = None
    toggle_default = None
    level_param_name = None
    level_default = None

    for field_definition in raw_info.get("customFields") or []:
        if not isinstance(field_definition, dict):
            continue

        field_key = str(field_definition.get("key") or "").strip()
        if not field_key:
            continue

        hint_blob = _collect_reasoning_field_hints(field_definition)
        if "reason" not in hint_blob and "think" not in hint_blob:
            continue

        # Use both field type and hint text because LM Studio custom fields
        # are flexible and rarely provide one canonical reasoning schema.
        field_type = str(field_definition.get("type") or "").strip().lower()
        default_value = field_definition.get("defaultValue")

        if field_type == "boolean" and toggle_param_name is None:
            toggle_param_name = field_key
            toggle_default = default_value
            continue

        if field_type in {"string", "select"} and level_param_name is None:
            normalized_default = str(default_value or "").strip().lower()
            if normalized_default in REASONING_LEVEL_VALUES:
                level_param_name = field_key
                level_default = normalized_default

    return toggle_param_name, toggle_default, level_param_name, level_default


# Extract reasoning level options.
def _extract_reasoning_level_options(raw_info: dict[str, Any]) -> list[str]:
    """Return supported reasoning level values from LM Studio custom fields."""

    for field_definition in raw_info.get("customFields") or []:
        if not isinstance(field_definition, dict):
            continue

        hint_blob = _collect_reasoning_field_hints(field_definition)
        if "reason" not in hint_blob and "think" not in hint_blob:
            continue

        field_type = str(field_definition.get("type") or "").strip().lower()
        if field_type not in {"string", "select"}:
            continue

        options = field_definition.get("options") or []
        if not isinstance(options, list):
            continue

        normalized_options: list[str] = []
        for option in options:
            if isinstance(option, dict):
                value = str(option.get("value") or "").strip().lower()
            else:
                value = str(option or "").strip().lower()
            if not value or value in normalized_options:
                continue
            normalized_options.append(value)
        if normalized_options:
            return normalized_options

    return []


# Detect local GPU devices.
def _get_local_gpu_devices() -> list[dict[str, Any]]:
    """Return local NVIDIA GPU devices with both numeric ids and labels."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    if result.returncode != 0:
        return []

    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2 or not parts[0]:
            continue
        try:
            device_id = int(parts[0])
        except ValueError:
            continue
        devices.append({"id": device_id, "name": parts[1] or f"GPU {device_id}"})

    return devices


# Remove control tokens from generated text.
def _sanitize_generated_text(text: str) -> str:
    """Drop service control tokens that should never reach the chat UI."""

    cleaned = str(text or "")
    for pattern in CONTROL_TOKEN_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


# Read one model info payload.
def _get_model_info(client: Any, model_name: str) -> tuple[Any | None, dict[str, Any]]:
    """Return model info from the SDK or loaded-model listing."""

    try:
        info = client.llm.get_model_info(model_name)
        return info, _serialize_model_info(info)
    except Exception:
        pass

    try:
        loaded_models = list(client.list_loaded_models())
    except Exception:
        loaded_models = []

    for entry in loaded_models:
        if _model_identifiers_match(model_name, _coerce_model_name(entry)):
            return entry, _serialize_model_info(entry)

    return None, {"model": model_name}


# Build reasoning parsing config.
def _build_reasoning_parsing_config(enabled: bool) -> dict[str, Any]:
    """Return the default reasoning parsing config for LM Studio."""

    return {
        "enabled": enabled,
        "startString": "<think>",
        "endString": "</think>",
    }


# Prepare native prediction options.
def _prepare_native_prediction_options(
    options: dict[str, Any] | None,
    think: Any = None,
    think_level: Any = None,
    think_param_name: str = "think",
    think_level_param_name: str = "reasoning_effort",
) -> dict[str, Any]:
    """Normalize LM Studio SDK prediction options."""

    prepared = dict(options or {})
    prepared.pop("raw", None)
    prepared["reasoningParsing"] = _build_reasoning_parsing_config(True)
    normalized_think_param = str(think_param_name or "").strip()
    normalized_level_param = str(think_level_param_name or "").strip()
    if think is not None and normalized_think_param and not normalized_think_param.startswith("ext."):
        prepared.setdefault(normalized_think_param, think)
    if think_level is not None and normalized_level_param and not normalized_level_param.startswith("ext."):
        prepared.setdefault(normalized_level_param, think_level)
    return prepared


# Prepare OpenAI-compatible prediction options.
def _prepare_openai_prediction_options(
    options: dict[str, Any] | None,
    think: Any = None,
    think_level: Any = None,
    think_param_name: str = "think",
    think_level_param_name: str = "reasoning_effort",
) -> dict[str, Any]:
    """Normalize LM Studio OpenAI-compatible options for tool-calling flows."""

    direct_options: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}

    for raw_key, raw_value in (options or {}).items():
        if raw_key == "raw":
            continue
        normalized_key = OPENAI_DIRECT_OPTION_ALIASES.get(raw_key, raw_key)
        if normalized_key in OPENAI_DIRECT_OPTION_KEYS:
            direct_options[normalized_key] = raw_value
        else:
            extra_body[raw_key] = raw_value

    if not _bool_from_value(direct_options.get("logprobs")):
        direct_options.pop("top_logprobs", None)

    if think_level and "reasoning_effort" not in direct_options and "reasoning_effort" not in extra_body:
        direct_options["reasoning_effort"] = think_level
    if think is not None and think_param_name:
        normalized_think_key = OPENAI_DIRECT_OPTION_ALIASES.get(think_param_name, think_param_name)
        target_options = direct_options if normalized_think_key in OPENAI_DIRECT_OPTION_KEYS else extra_body
        target_options.setdefault(normalized_think_key, think)

    extra_body["reasoningParsing"] = _build_reasoning_parsing_config(True)

    if extra_body:
        merged_extra_body = dict(direct_options.get("extra_body", {}) or {})
        merged_extra_body.update(extra_body)
        direct_options["extra_body"] = merged_extra_body

    return direct_options


# Create the OpenAI-compatible LM Studio client.
def _get_openai_client():
    """Create an OpenAI client pointing at the LM Studio server."""

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The 'openai' package is required for LM Studio tool support.") from exc

    raw_url = settings.get_engine_url("lms")
    if not raw_url.startswith("http"):
        raw_url = f"http://{raw_url}"
    if not raw_url.rstrip("/").endswith("/v1"):
        raw_url = raw_url.rstrip("/") + "/v1"

    return OpenAI(base_url=raw_url, api_key="lm-studio")


# Build OpenAI-compatible messages.
def _build_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ASLM chat messages into OpenAI-compatible payloads."""

    payload: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = message.get("content", "") or ""
        images = message.get("images") or []
        image_mime_types = message.get("image_mime_types") or []
        media = message.get("media") if isinstance(message.get("media"), list) else []

        if role == "tool":
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id", message.get("name", "call")),
                    "content": content,
                }
            )
            continue

        if images or media:
            content_parts: list[dict[str, Any]] = []
            if content:
                content_parts.append({"type": "text", "text": content})
            for index, image_base64 in enumerate(images):
                image_mime_type = "image/jpeg"
                if index < len(image_mime_types):
                    image_mime_type = str(image_mime_types[index] or image_mime_type)
                content_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"},
                    }
                )
            for media_item in media:
                if not isinstance(media_item, dict):
                    continue
                media_data = str(media_item.get("data") or "").strip()
                mime_type = str(media_item.get("mime_type") or "application/octet-stream").lower()
                media_kind = str(media_item.get("kind") or "").lower()
                if not media_data:
                    continue
                if media_kind == "audio" and mime_type in {"audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav"}:
                    content_parts.append(
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": media_data,
                                "format": "mp3" if mime_type in {"audio/mpeg", "audio/mp3"} else "wav",
                            },
                        }
                    )
                    continue
                content_parts.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": str(media_item.get("name") or f"uploaded-{media_kind or 'media'}"),
                            "file_data": f"data:{mime_type};base64,{media_data}",
                        },
                    }
                )
            entry: dict[str, Any] = {"role": role, "content": content_parts}
        else:
            entry = {"role": role, "content": content}

        if role == "assistant" and message.get("tool_calls"):
            entry["tool_calls"] = message["tool_calls"]

        if role == "assistant" and message.get("thinking"):
            # LM Studio accepts a top-level 'thinking' field on assistant messages
            # in its OpenAI-compatible endpoint. Without this the model loses its
            # own reasoning context on every subsequent turn.
            entry["thinking"] = message["thinking"]

        payload.append(entry)

    return payload


# Build one tool event.
def _build_tool_event(tool_lookup: dict[str, dict[str, Any]], tool_call: dict[str, Any]) -> dict[str, Any]:
    """Serialize one tool invocation so the UI can render it during streaming."""
    return tool_registry.build_tool_event(tool_lookup, tool_call)


# Build one tool message.
def _build_tool_message(
    tool_name: str,
    tool_call_id: str,
    content: str | dict[str, Any],
    tool_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a tool message payload for the OpenAI-compatible conversation."""

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


# Parse streamed reasoning fragments.
class _ReasoningTextParser:

    # Initialize parser state.
    def __init__(self, tag_pairs: tuple[tuple[str, str], ...] = REASONING_TAG_PAIRS) -> None:
        self._tag_pairs = tag_pairs
        self._start_tags = [start for start, _end in tag_pairs]
        self._end_tags = [end for _start, end in tag_pairs]
        self._in_reasoning = False
        self._pending = ""
        self._visible_boundary_char: str | None = None

    # Find the next matching tag.
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

    # Preserve a possible partial tag suffix.
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

    # Parse one streamed fragment.
    def feed(self, content: str, reasoning_type: str | None = None) -> tuple[str, str]:
        """Return parsed ``(thinking, visible_content)`` fragments."""

        thinking_parts: list[str] = []
        content_parts: list[str] = []

        if reasoning_type == "reasoningStartTag":
            self._in_reasoning = True
            return "", ""
        if reasoning_type == "reasoningEndTag":
            self._in_reasoning = False
            return "", ""
        if reasoning_type == "reasoning":
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

    # Flush any buffered fragment.
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


# Stream native and OpenAI-compatible LM Studio conversations.
# Upload one image handle.
def _prepare_image_handle(
    client: Any,
    image_base64: str,
    name: str,
    cache: dict[str, Any] | None = None,
) -> Any:
    """Upload one image to LM Studio and return its file handle."""

    digest = hashlib.sha256(str(image_base64 or "").encode("utf-8")).hexdigest()
    if cache is not None and digest in cache:
        return cache[digest]

    image_bytes = base64.b64decode(image_base64)
    handle = client.prepare_image(io.BytesIO(image_bytes), name=name)
    if cache is not None:
        cache[digest] = handle
    return handle


# Build native chat history.
def _build_native_chat_history(client: Any, messages: list[dict[str, Any]]):
    """Convert generic chat messages into an LM Studio chat history."""

    lms = _get_sdk()
    chat = lms.Chat()
    image_handle_cache: dict[str, Any] = {}

    for message_index, message in enumerate(messages, start=1):
        role = str(message.get("role", "user")).lower()
        content = str(message.get("content", "") or "")
        images = [str(item) for item in (message.get("images") or []) if str(item).strip()]

        if role == "system":
            chat.add_system_prompt(content)
            continue

        if role == "assistant":
            chat.add_assistant_response(content)
            continue

        if role == "tool":
            chat.add_tool_result(
                {
                    "toolCallId": str(message.get("tool_call_id", message.get("name", f"tool_{message_index}")) or f"tool_{message_index}"),
                    "content": content,
                }
            )
            continue

        # LM Studio expects uploaded image handles instead of inline base64, so
        # prepare those files before adding the user message.
        prepared_images = []
        for image_index, image_base64 in enumerate(images, start=1):
            prepared_images.append(
                _prepare_image_handle(
                    client,
                    image_base64,
                    name=f"image-{message_index}-{image_index}.jpg",
                    cache=image_handle_cache,
                )
            )

        if content:
            chat.add_user_message(content, images=prepared_images)
        else:
            chat.add_user_message("", images=prepared_images)

    return chat


# Stream one native LM Studio response.
def _stream_native_response(model: Any, chat: Any, options: dict[str, Any], stream: bool):
    """Yield parsed LM Studio SDK output."""

    parser = _ReasoningTextParser()
    visible_content = ""
    thinking_content = ""
    yielded_chunk = False

    if stream:
        # Native streaming emits small fragments plus reasoning annotations, so
        # parse them incrementally and keep an assembled final message.
        for fragment in model.respond_stream(chat, config=options or None):
            if _abort_event.is_set():
                break

            fragment_content = _sanitize_generated_text(str(getattr(fragment, "content", "") or ""))
            reasoning_type = str(getattr(fragment, "reasoning_type", "none") or "none")
            thinking_part, content_part = parser.feed(fragment_content, reasoning_type=reasoning_type)
            if not thinking_part and not content_part:
                continue

            yielded_chunk = True
            visible_content += content_part
            thinking_content += thinking_part
            payload = {"role": "assistant", "content": content_part}
            if thinking_part:
                payload["thinking"] = thinking_part
            yield {"message": payload}
    else:
        # Reuse the same parser in non-streaming mode so both paths emit the
        # same final assistant shape.
        result = model.respond(chat, config=options or None)
        result_content = _sanitize_generated_text(str(getattr(result, "content", "") or ""))
        thinking_part, content_part = parser.feed(result_content, reasoning_type="none")
        visible_content += content_part
        thinking_content += thinking_part
        yielded_chunk = True
        payload = {"role": "assistant", "content": content_part}
        if thinking_part:
            payload["thinking"] = thinking_part
        yield {"message": payload}

    tail_thinking, tail_content = parser.flush()
    if tail_thinking or tail_content:
        visible_content += tail_content
        thinking_content += tail_thinking
        yielded_chunk = True
        payload = {"role": "assistant", "content": tail_content}
        if tail_thinking:
            payload["thinking"] = tail_thinking
        yield {"message": payload}

    assistant_message: dict[str, Any] = {"role": "assistant", "content": visible_content}
    if thinking_content:
        assistant_message["thinking"] = thinking_content

    if not yielded_chunk and (visible_content or thinking_content):
        yield {"message": assistant_message}


# Stream one OpenAI-compatible LM Studio round.
def _stream_openai_round(
    client,
    model_name: str,
    conversation: list[dict[str, Any]],
    options: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    forced_tool_name: str = "",
    require_tool_call: bool = False,
):
    """Stream one LM Studio round via OpenAI API and return the assembled message."""

    request_kwargs: dict[str, Any] = dict(options)
    if tools:
        request_kwargs["tools"] = tools
        available_tool_names = {
            str((tool.get("function") or {}).get("name") or "").strip()
            for tool in tools
            if isinstance(tool, dict)
        }
        if forced_tool_name and forced_tool_name in available_tool_names:
            request_kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": forced_tool_name},
            }
        elif require_tool_call:
            request_kwargs["tool_choice"] = "required"
        else:
            request_kwargs["tool_choice"] = "auto"

    assistant_content = ""
    assistant_thinking = ""
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    parser = _ReasoningTextParser()

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=conversation,
            stream=True,
            **request_kwargs,
        )
    except Exception as exc:
        # Not every OpenAI-compatible backend accepts forced tool_choice payloads;
        # degrade to auto so the request still runs and the loop supervisor can nudge.
        if request_kwargs.get("tool_choice") in (None, "auto"):
            raise
        logger.warning("[LM Studio API] Forced tool_choice rejected, retrying with auto: %s", exc)
        request_kwargs["tool_choice"] = "auto"
        response = client.chat.completions.create(
            model=model_name,
            messages=conversation,
            stream=True,
            **request_kwargs,
        )

    # LM Studio's OpenAI-compatible endpoint streams deltas, so tool calls must
    # be reconstructed chunk-by-chunk just like provider OpenAI responses.
    for chunk in response:
        if _abort_event.is_set():
            break

        for choice in getattr(chunk, "choices", []) or []:
            delta = getattr(choice, "delta", None)
            if not delta:
                continue

            content_part = _sanitize_generated_text(str(getattr(delta, "content", "") or ""))
            if content_part:
                thinking_part, visible_part = parser.feed(content_part, reasoning_type="none")
                assistant_content += visible_part
                assistant_thinking += thinking_part
                if thinking_part or visible_part:
                    payload = {"content": visible_part}
                    if thinking_part:
                        payload["thinking"] = thinking_part
                    yield {"message": payload}

            # Tool-call ids, names, and JSON arguments may arrive separately.
            for tc in getattr(delta, "tool_calls", []) or []:
                idx = getattr(tc, "index", 0)
                if idx not in tool_calls_by_index:
                    tool_calls_by_index[idx] = {
                        "id": getattr(tc, "id", "") or "",
                        "type": "function",
                        "function": {"name": "", "arguments": ""},
                    }
                entry = tool_calls_by_index[idx]
                if getattr(tc, "id", None):
                    entry["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn:
                    if getattr(fn, "name", None):
                        entry["function"]["name"] += fn.name
                    if getattr(fn, "arguments", None):
                        entry["function"]["arguments"] += fn.arguments

    tail_thinking, tail_content = parser.flush()
    if tail_thinking or tail_content:
        assistant_content += tail_content
        assistant_thinking += tail_thinking
        payload = {"content": tail_content}
        if tail_thinking:
            payload["thinking"] = tail_thinking
        yield {"message": payload}

    assembled_tool_calls = [tool_calls_by_index[idx] for idx in sorted(tool_calls_by_index)]
    assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant_content}
    if assistant_thinking:
        assistant_message["thinking"] = assistant_thinking
    if assembled_tool_calls:
        assistant_message["tool_calls"] = assembled_tool_calls

    return assistant_message


# Drain one streamed round.
def _yield_stream_round(round_stream):
    """Yield every chunk from a round stream and return the final assistant message."""

    while True:
        try:
            yield next(round_stream)
        except StopIteration as stop:
            return stop.value or {"role": "assistant", "content": ""}


# Run the tool loop.
def _run_tool_loop(
    client,
    model_name: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tool_server_ids: list[str],
    tool_context: dict[str, Any],
    *,
    conversation: list[dict[str, Any]] | None = None,
):
    """Resolve local tools through LM Studio's OpenAI-compatible tool-calling."""

    tools, tool_lookup = tool_registry.build_ollama_tools(
        tool_server_ids,
        engine="lms",
        model_name=model_name,
        tool_source_map=tool_context.get("tool_source_map") if isinstance(tool_context, dict) else None,
        allowed_tool_aliases=tool_context.get("allowed_tool_aliases") if isinstance(tool_context, dict) else None,
        instant_mode=bool(tool_context.get("instant_mode")) if isinstance(tool_context, dict) else False,
        instant_search_batch_size=tool_context.get("instant_search_batch_size") if isinstance(tool_context, dict) else None,
    )

    if not tools:
        request_messages = conversation if conversation is not None else _build_openai_messages(messages)
        yield from _yield_stream_round(
            _stream_openai_round(client, model_name, request_messages, options)
        )
        return

    conversation = conversation if conversation is not None else _build_openai_messages(messages)
    tool_quota_counters: dict[str, int] = {}
    seen_tool_signatures: set[str] = set()
    consecutive_blocked_tool_results = 0
    forced_tool_name = str(tool_context.get("forced_tool_name") or "").strip() if isinstance(tool_context, dict) else ""
    required_tool_server_ids = tool_registry.required_tool_server_ids_from_context(tool_context, tool_lookup)
    satisfied_tool_server_ids: set[str] = set()
    required_nudges_left = tool_registry.MAX_REQUIRED_TOOL_NUDGES if required_tool_server_ids else 0
    # Round 0 enforces user-required tools up front; later rounds re-enforce only
    # right after a supervisor nudge so the model can still finish normally.
    enforce_required_tools_this_round = True

    max_tool_rounds = tool_registry.tool_round_limit_from_context(tool_context, MAX_TOOL_ROUNDS)
    for round_index in range(max_tool_rounds):
        round_forced_tool_name = ""
        round_requires_tool_call = False
        if enforce_required_tools_this_round:
            forced_server_id = tool_registry.tool_server_id_for_alias(tool_lookup, forced_tool_name)
            if forced_tool_name and forced_server_id not in satisfied_tool_server_ids:
                round_forced_tool_name = forced_tool_name
            elif any(server_id not in satisfied_tool_server_ids for server_id in required_tool_server_ids):
                round_requires_tool_call = True
        enforce_required_tools_this_round = False

        # Each round either completes the answer or produces another batch of
        # tool calls that must be executed locally.
        assistant_message = yield from _yield_stream_round(
            _stream_openai_round(
                client,
                model_name,
                conversation,
                options,
                tools=tools,
                forced_tool_name=round_forced_tool_name,
                require_tool_call=round_requires_tool_call,
            )
        )
        conversation.append(assistant_message)

        raw_tool_calls = assistant_message.get("tool_calls") or []
        if not raw_tool_calls:
            unmet_tool_server_ids = [
                server_id for server_id in required_tool_server_ids
                if server_id not in satisfied_tool_server_ids
            ]
            if unmet_tool_server_ids and required_nudges_left > 0:
                # Supervisor: the model answered while ignoring user-required tools,
                # so feed a corrective prompt and run another enforced round.
                required_nudges_left -= 1
                enforce_required_tools_this_round = True
                yield {"transcript_message": assistant_message}
                conversation.append(
                    {
                        "role": "user",
                        "content": tool_registry.required_tools_reminder_prompt(
                            unmet_tool_server_ids,
                            tool_lookup,
                        ),
                    }
                )
                continue
            yield {"transcript_message": assistant_message}
            return

        tool_calls = []
        # Convert OpenAI-compatible tool payloads into the internal dispatch
        # shape used by the shared MCP tool bridge.
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                arguments = {}
            tool_calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "id": tc.get("id", f"call_{round_index}_{name}"),
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
        conversation[-1] = canonical_assistant_message
        yield {"transcript_message": canonical_assistant_message}

        tool_events = [_build_tool_event(tool_lookup, tool_call) for tool_call in tool_calls]
        for _i, _ev in enumerate(tool_events):
            _ev["alias"] = f"{_ev['alias']}__{_i}"
        yield {"tool_events": tool_events}

        cancelled_tool_result_seen = False
        for tool_call_index, (tool_call, tool_event) in enumerate(zip(tool_calls, tool_events), start=1):
            call_context = dict(tool_context or {})
            call_context.update(
                {
                    "engine": "lms",
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
            # The next model round consumes the resolved tool result as a
            # standard tool-role message in the conversation history.
            conversation_tool_message = {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": tool_message["content"],
            }
            conversation.append(conversation_tool_message)
            tool_registry.log_search_tool_io(
                "response",
                tool_event,
                arguments=tool_call.get("arguments") or {},
                context=call_context,
                result=conversation_tool_message,
            )
            yield {"tool_result": dict(tool_message)}

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
                _stream_openai_round(client, model_name, conversation, options)
            )
            yield {"transcript_message": assistant_message}
            return

        conversation = tool_registry.maybe_compact_tool_conversation(
            tool_context, conversation, provider_format="standard"
        )

    yield {"message": {"content": "[Error during generation: tool loop exceeded the safety limit.]"}}


# Expose the public adapter API.
# Detect tool usage in the conversation.
def _conversation_uses_tools(messages: list[dict[str, Any]]) -> bool:
    """Return whether the current conversation already contains tool state."""

    for message in messages:
        if str(message.get("role", "")).lower() == "tool":
            return True
        if message.get("tool_calls"):
            return True
    return False


# List available models.
def get_models() -> list[Any]:
    """Return models available in the configured LM Studio server.

    LM Studio can serve any downloaded model through just-in-time loading, so
    list every downloaded LLM rather than only the currently loaded ones (a
    running server with no model loaded would otherwise report an empty list).
    Fall back to the loaded list when the downloaded query is unavailable.
    """

    try:
        _lms, client = _get_client()
    except Exception as exc:
        logger.error("[LM Studio API] Error connecting to LM Studio: %s", exc)
        return []

    try:
        downloaded_models: list[Any] = []
        try:
            downloaded_models = list(client.llm.list_downloaded())
        except Exception as exc:
            logger.warning("[LM Studio API] Error listing downloaded models: %s", exc)

        loaded_models: list[Any] = []
        try:
            loaded_models = list(client.list_loaded_models())
        except Exception as exc:
            logger.warning("[LM Studio API] Error listing loaded models: %s", exc)
    finally:
        _close_client(client)

    # Prefer the downloaded list (canonical model keys); only fall back to the
    # loaded list when the downloaded query yielded nothing.
    return _collect_unique_model_names(downloaded_models or loaded_models)


# Release runtime resources when needed.
def cleanup_runtime() -> None:
    """LM Studio model lifecycle is managed outside this adapter."""

    return None


# Reject local download requests.
def download_model(model_name: str, **kwargs: Any) -> Any:
    """Raise because LM Studio downloads are managed outside this adapter."""

    raise NotImplementedError("LM Studio model downloads are managed by LM Studio.")


# Generate one model response.
def generate(model_name: str, messages: list[dict[str, Any]], **kwargs: Any):
    """Generate a streamed or non-streamed response through LM Studio."""

    # Normalize legacy single-id inputs into the shared multi-server format.
    raw_ids = kwargs.pop("tool_server_ids", None) or kwargs.pop("tool_server_id", None) or kwargs.pop("tool_id", None)
    if isinstance(raw_ids, str):
        tool_server_ids = [raw_ids] if raw_ids.strip() else []
    elif isinstance(raw_ids, list):
        tool_server_ids = [str(s) for s in raw_ids if str(s).strip()]
    else:
        tool_server_ids = []

    tool_context = dict(kwargs.pop("tool_context", {}) or {})
    stream = bool(kwargs.get("stream", False))
    raw_options = dict(kwargs.get("options", {}) or {})
    sync_operation_defaults = dict(kwargs.get("sync_operation_defaults", {}) or {})
    think = kwargs.get("think")
    think_level = kwargs.get("think_level")
    think_param_name = str(kwargs.get("think_param_name", "think") or "think")
    think_level_param_name = str(kwargs.get("think_level_param_name", "reasoning_effort") or "reasoning_effort")

    _abort_event.clear()
    _sync_operation_defaults_to_disk(model_name, sync_operation_defaults)

    # Load the model up front (JIT) so both the native and OpenAI-compatible
    # paths can serve any downloaded model, not just pre-loaded ones.
    _lms, preload_client = _get_client()
    try:
        _ensure_model_loaded(preload_client, model_name)
    finally:
        _close_client(preload_client)

    if tool_server_ids or _conversation_uses_tools(messages):
        # Tool-aware flows use the OpenAI-compatible endpoint because it
        # exposes tool calling, while plain chats can stay on the native SDK.
        openai_client = _get_openai_client()
        options = _prepare_openai_prediction_options(
            raw_options,
            think=think,
            think_level=think_level,
            think_param_name=think_param_name,
            think_level_param_name=think_level_param_name,
        )
        for key in INTERNAL_CHAT_KEYS:
            options.pop(key, None)
        try:
            if tool_server_ids:
                yield from _run_tool_loop(openai_client, model_name, messages, options, tool_server_ids, tool_context)
                return
            conversation = _build_openai_messages(messages)
            yield from _yield_stream_round(_stream_openai_round(openai_client, model_name, conversation, options))
            return
        finally:
            _close_client(openai_client)

    _lms, client = _get_client()
    try:
        # The model was already JIT-loaded above; _get_model_handle re-uses it.
        model = _get_model_handle(client, model_name)
        chat = _build_native_chat_history(client, messages)
        options = _prepare_native_prediction_options(
            raw_options,
            think=think,
            think_level=think_level,
            think_param_name=think_param_name,
            think_level_param_name=think_level_param_name,
        )
        yield from _stream_native_response(model, chat, options, stream=stream)
    except Exception as exc:
        logger.error("[LM Studio API] Error generating response from %s: %s", model_name, exc)
        raise
    finally:
        _close_client(client)


# Read one model's settings.
def get_model_settings(model_name: str) -> dict[str, Any]:
    """Combine the downloaded-model catalog with optional loaded-instance settings."""

    catalog_info = _get_catalog_model_info(model_name)
    info = None
    raw_info: dict[str, Any] = {}
    fallback_info: dict[str, Any] = {}
    client = None
    try:
        if not catalog_info or catalog_info.get("loaded_instances"):
            _lms, client = _get_client()
            raw_info = _get_raw_model_info(client, model_name)
            info, fallback_info = _get_model_info(client, model_name)
    except Exception:
        if not catalog_info:
            raise
        logger.debug("[LM Studio API] Using catalog without loaded-instance settings", exc_info=True)
    finally:
        if client is not None:
            _close_client(client)

    model_index_record = _get_model_index_record(model_name)
    model_virtual = model_index_record.get("virtual", {}) if isinstance(model_index_record, dict) else {}
    indexed_custom_fields = model_virtual.get("customFieldDefinitions", []) if isinstance(model_virtual, dict) else []
    disk_operation_defaults = _get_disk_model_operation_defaults(model_name)

    # Merge runtime info, indexed metadata, and user defaults so the settings
    # panel reflects what LM Studio will actually use for this model.
    raw_info = raw_info or fallback_info or {}
    if indexed_custom_fields and not isinstance(raw_info.get("customFields"), list):
        raw_info["customFields"] = indexed_custom_fields
    if model_virtual.get("metadataOverridesReasoning") is True and "reasoning" not in raw_info:
        raw_info["reasoning"] = True

    metadata_overrides = _normalize_model_config(raw_info.get("metadataOverrides") or {})
    catalog_capabilities = catalog_info.get("capabilities", {})
    if not isinstance(catalog_capabilities, dict):
        catalog_capabilities = {}
    catalog_reasoning = catalog_capabilities.get("reasoning")
    config_payload = _normalize_model_config(raw_info.get("config") or {})
    operation_defaults = _merge_nested_dicts(_normalize_model_config(config_payload.get("operation") or {}), disk_operation_defaults)

    custom_fields = raw_info.get("customFields") if isinstance(raw_info.get("customFields"), list) else []
    raw_info["customFields"] = custom_fields
    custom_toggle_name, custom_toggle_default, custom_level_name, custom_level_default = _detect_reasoning_custom_fields(
        {"customFields": custom_fields}
    )
    think_level_options = _extract_reasoning_level_options({"customFields": custom_fields})

    if custom_toggle_name and custom_toggle_name not in operation_defaults and custom_toggle_default is not None:
        operation_defaults[custom_toggle_name] = custom_toggle_default
    if custom_level_name and custom_level_name not in operation_defaults and custom_level_default is not None:
        operation_defaults[custom_level_name] = custom_level_default

    think_value, think_level_value = _extract_reasoning_settings(operation_defaults)
    reasoning_config = operation_defaults.get("reasoning")
    if isinstance(reasoning_config, dict) and set(reasoning_config.keys()).issubset({"enabled", "effort"}):
        operation_defaults.pop("reasoning", None)

    # Normalize canonical ASLM keys even when LM Studio stores provider-specific
    # reasoning fields under custom names.
    if think_value is not None and "think" not in operation_defaults:
        operation_defaults["think"] = think_value
    if think_level_value is not None and "reasoning_effort" not in operation_defaults:
        operation_defaults["reasoning_effort"] = think_level_value

    context_length = (
        raw_info.get("contextLength")
        or raw_info.get("maxContextLength")
        or metadata_overrides.get("contextLength")
        or metadata_overrides.get("maxContextLength")
        or getattr(info, "context_length", None)
        or getattr(info, "max_context_length", None)
        or catalog_info.get("max_context_length")
        or 8192
    )
    supports_vision = _bool_from_value(
        catalog_capabilities.get(
            "vision",
            raw_info.get("vision", metadata_overrides.get("vision", getattr(info, "vision", False))),
        ),
        default=False,
    )
    supports_tool_calling = _bool_from_value(
        catalog_capabilities.get(
            "trained_for_tool_use",
            raw_info.get(
                "trainedForToolUse",
                metadata_overrides.get("trainedForToolUse", getattr(info, "trained_for_tool_use", False)),
            ),
        ),
        default=False,
    )
    supports_thinking = _bool_from_value(
        raw_info.get("reasoning", metadata_overrides.get("reasoning")),
        default=False,
    )
    supports_thinking = (
        supports_thinking
        or bool(catalog_reasoning)
        or bool(model_virtual.get("metadataOverridesReasoning"))
        or custom_toggle_name is not None
        or think_value is not None
        or think_level_value is not None
    )
    supports_think_level = think_level_value is not None or custom_level_name is not None
    supports_think_toggle = custom_toggle_name is not None or isinstance(think_value, bool)

    think_param_name = custom_toggle_name or "think"
    think_level_param_name = custom_level_name or "reasoning_effort"
    if not think_level_options and supports_think_level:
        if custom_level_name or think_level_param_name == "reasoning_effort":
            think_level_options = ["low", "medium", "high"]
        elif think_level_param_name in {"think_level", "thinking_level"}:
            think_level_options = ["low", "medium", "high"]

    gpu_devices = _get_local_gpu_devices()
    gpu_count = len(gpu_devices)

    capabilities: list[str] = []
    if supports_vision:
        capabilities.append("vision")
    if supports_tool_calling:
        capabilities.append("tools")
    if supports_thinking:
        capabilities.append("thinking")
    capabilities.append("files")

    return {
        "model": model_name,
        "context_length": int(context_length),
        "capabilities_source": "lms-catalog" if catalog_capabilities else "lms-sdk",
        "metadata_fallback": not catalog_capabilities and not any(
            key in raw_info or key in metadata_overrides for key in ("vision", "trainedForToolUse")
        ),
        "defaults": operation_defaults,
        "supports_thinking": supports_thinking,
        "supports_think_toggle": supports_think_toggle,
        "supports_think_level": supports_think_level,
        "think_param_name": think_param_name,
        "think_level_param_name": think_level_param_name,
        "think_level_options": think_level_options,
        "supports_vision": supports_vision,
        "supports_tool_calling": supports_tool_calling,
        "supports_files": True,
        "capabilities": capabilities,
        "custom_fields": custom_fields,
        "runtime_limits": {
            "cpu_threads": max(int(os.cpu_count() or 1), 1),
            "gpu_count": gpu_count,
            "gpu_devices": gpu_devices,
            "main_gpu_max": max(gpu_count - 1, 0),
        },
        "raw": raw_info,
    }
