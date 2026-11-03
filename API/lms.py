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

# Process-wide state: the flag that asks the running generation to stop, the lock guarding model state
# updates, and the signature of the last operation defaults synced per model (to skip identical re-syncs).
_abort_event = threading.Event()
_model_state_lock = threading.Lock()
_synced_operation_defaults_signatures: dict[str, str] = {}


# Signal the active LM Studio generation to stop; the generation loops poll this flag.
def abort_generation() -> None:
    _abort_event.set()


# Chat config keys consumed by the application itself; they are removed from the OpenAI-compatible request options.
INTERNAL_CHAT_KEYS = {"system", "prompt", "tool_id", "tool_server_id", "tool_server_ids", "tool_context", "think", "think_level", "engine"}

# Upper bound of model-and-tool rounds in one generation unless the tool context sets its own limit.
MAX_TOOL_ROUNDS = 100

# Opening and closing tags that wrap reasoning text in model output.
REASONING_TAG_PAIRS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
)

# Patterns for chat-template control tokens (harmony and ChatML style) that must not leak into visible text.
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

# Accepted reasoning effort levels.
REASONING_LEVEL_VALUES = {"minimal", "low", "medium", "high", "xhigh"}

# Chat option names renamed to their OpenAI request name, and the option keys sent as top-level OpenAI
# request fields (everything else goes into extra_body).
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


# SDK clients

# Normalize the configured LM Studio address into the host value the SDK client expects.
# A URL yields its host and port, a bare scheme-prefixed address its path, anything else the trimmed text.
def _extract_api_host(raw_address: str) -> str:
    parsed = urlparse(raw_address)
    if parsed.scheme and parsed.netloc:
        return parsed.netloc
    if parsed.scheme and parsed.path:
        return parsed.path
    return raw_address.strip().rstrip("/")


# Import the LM Studio SDK lazily so the app can boot without it; raise ImportError with a clear message
# when the package is missing.
def _get_sdk():
    try:
        import lmstudio as lms
    except ImportError as exc:
        raise ImportError("The 'lmstudio' package is required for LM Studio support.") from exc

    return lms


# Create a fresh LM Studio client for the configured server and return it together with the SDK module.
# The caller owns the client and must close it.
def _get_client():
    lms = _get_sdk()
    api_host = _extract_api_host(settings.get_engine_url("lms"))
    return lms, lms.Client(api_host)


# Close a client instance, ignoring any error raised by the SDK.
def _close_client(client: Any) -> None:
    try:
        client.close()
    except Exception:
        # Closing is best-effort cleanup; a failing close must not mask the result or error of the caller.
        pass


# Return an SDK handle for the named model (the SDK loads the model on demand).
def _get_model_handle(client: Any, model_name: str):
    return client.llm.model(model_name)


# Model names and loading

# Extract a stable model identifier from an LM Studio SDK object (or dict): direct attributes first, then
# the nested info object, then get_info(). Return an empty string when nothing usable is found.
def _coerce_model_name(entry: Any) -> str:
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
        # The SDK object could not describe itself; treat the name as unknown.
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


# Return the unique model names of the entries in their original order, skipping entries without a name.
def _collect_unique_model_names(entries: list[Any]) -> list[str]:
    unique_names: list[str] = []
    seen_names: set[str] = set()

    for entry in entries:
        model_name = _coerce_model_name(entry)
        if not model_name or model_name in seen_names:
            continue
        seen_names.add(model_name)
        unique_names.append(model_name)

    return unique_names


# Call one listing method of a fresh LM Studio client (for example list_downloaded_models) and always
# close the client afterwards.
def _list_models_with_client(method_name: str) -> list[Any]:
    _lms, client = _get_client()
    try:
        return list(getattr(client, method_name)())
    finally:
        _close_client(client)


# Coerce a boolean-like value from SDK and dict payloads: None gives the default, strings such as
# "true", "1", "yes", "on" (and their negatives) are parsed, anything else uses Python truthiness.
def _bool_from_value(value: Any, default: bool = False) -> bool:
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


# Return a stable string signature of a config (sorted-key JSON, or str() for values JSON cannot encode),
# used to detect whether a config changed.
def _config_signature(config: dict[str, Any]) -> str:
    try:
        return json.dumps(config or {}, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(config or {})


# Return a model identifier trimmed, without surrounding slashes and lowercased, for fuzzy comparisons.
def _normalize_model_identifier(value: Any) -> str:
    return str(value or "").strip().strip("/").lower()


# Return whether two model identifiers likely refer to the same model: equal after normalization, or equal
# in their last path segment (so "publisher/model" matches "model").
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


# Return the names of the models currently loaded in LM Studio, or an empty list when they cannot be listed.
def _list_loaded_model_names(client: Any) -> list[str]:
    try:
        loaded_models = list(client.list_loaded_models())
    except Exception:
        # An unreachable server or an SDK without this call is reported as "nothing loaded".
        return []

    return _collect_unique_model_names(loaded_models)


# Raise RuntimeError when the requested model is not already loaded in LM Studio; the message lists the
# models that are loaded.
def _assert_model_is_loaded(client: Any, model_name: str) -> None:
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


# Get or load the requested model so generation can run on downloaded models. LM Studio serves any
# downloaded model through just-in-time loading, so request a handle (which loads the model when needed)
# instead of failing when it is merely downloaded but not yet loaded. Raise RuntimeError when loading fails.
def _ensure_model_loaded(client: Any, model_name: str) -> None:
    try:
        client.llm.model(model_name)
    except Exception as exc:
        raise RuntimeError(
            f"LM Studio could not load model: {model_name}. "
            "Confirm it is downloaded and the LM Studio server is running. "
            f"({exc})"
        ) from exc


# Model info

# Convert SDK model info into a JSON-compatible dictionary. Try, in order: to_dict(), the nested info
# object, get_info(), the value itself when it is already a dict, and finally a field-by-field copy of the
# known attributes (camelCase keys, as the LM Studio payloads use).
def _serialize_model_info(info: Any) -> dict[str, Any]:
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
            # get_info() may need a live connection; fall through to the attribute copy below.
            pass

    if isinstance(info, dict):
        return info

    # Copy the first present attribute of each known field; the path is stringified for JSON.
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


# Read the public capabilities of a loaded or unloaded model from the native LM Studio catalog endpoint
# (/api/v1/models) without loading the model. Return the catalog entry whose key, variant or loaded instance
# id matches the name, else the only entry that matches fuzzily, else an empty dict (also when the catalog
# is unavailable).
def _get_catalog_model_info(model_name: str) -> dict[str, Any]:
    # Resolve the server root: add a scheme when missing and strip a trailing /v0, /v1 or /api/v1 suffix.
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

    # Search the llm entries: an exact identifier match wins immediately, fuzzy key matches are collected
    # and only used when they are unambiguous.
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


# Return the raw LM Studio model-info payload when the SDK exposes it, otherwise the payload from
# _get_model_info.
def _get_raw_model_info(client: Any, model_name: str) -> dict[str, Any]:
    get_raw_info = getattr(client.llm, "_get_api_model_info", None)
    if callable(get_raw_info):
        try:
            payload = get_raw_info(model_name)
            if isinstance(payload, dict):
                return payload
        except Exception:
            # The private SDK call may be missing or fail for this model; use the public fallback below.
            pass

    _info, payload = _get_model_info(client, model_name)
    return payload if isinstance(payload, dict) else {}


# Config normalization

# Recursively drop empty values (empty dicts, lists, strings and None) from nested config data,
# keeping scalar values such as 0 and False as they are.
def _clean_nested_config(value: Any) -> Any:
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


# Return a cleaned dictionary for a raw LM Studio config section (objects with to_dict() are converted
# first); anything that is not a dict yields an empty dict.
def _normalize_model_config(config: Any) -> dict[str, Any]:
    if hasattr(config, "to_dict"):
        config = config.to_dict()
    if not isinstance(config, dict):
        return {}
    normalized = _clean_nested_config(config)
    return normalized if isinstance(normalized, dict) else {}


# Recursively merge two dictionaries without mutating either input; values from override win.
def _merge_nested_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


# LM Studio files on disk

# Return the local LM Studio home directory (.lmstudio under the user profile).
def _get_lmstudio_home() -> str:
    user_home = os.environ.get("USERPROFILE") or os.path.expanduser("~")
    return os.path.join(user_home, ".lmstudio")


# Return the decoded JSON of a file, or None when the path is empty, the file is missing or unreadable,
# or it does not contain valid JSON.
def _read_json_file(path: str) -> Any:
    if not path or not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return None


# Convert a checkbox-like KV value ({"checked": ..., "value": ...}) into a plain value: the inner value when
# checked, None when unchecked. Other values are returned unchanged.
def _unwrap_kv_field_value(value: Any) -> Any:
    if isinstance(value, dict) and "checked" in value:
        if not _bool_from_value(value.get("checked"), default=False):
            return None
        return value.get("value")
    return value


# Assign a value into a nested client-config dictionary at a dotted path, creating (or replacing non-dict)
# intermediate levels. An empty path is ignored.
def _set_nested_client_value(target: dict[str, Any], path: str, value: Any) -> None:
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


# Apply one KV field through the SDK's key mapping by walking its dotted path parts. Return True when a
# mapping node with update_client_config handled the value, False when the SDK has no mapping for it.
def _apply_kv_mapping(target: dict[str, Any], mapping: Any, parts: list[str], value: Any) -> bool:
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


# Return a best-effort client config path for a KV field the SDK has no mapping for: the llama. prefix is
# dropped, the llm.load offload ratio becomes gpu.ratio, custom virtual-model fields keep their full key and
# any other dotted key is reduced to its last segment.
def _kv_field_fallback_path(prefix: str, remainder: str) -> str:
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


# Convert a list of LM Studio KV fields of one section (for example llm.prediction) into the nested
# client-config shape used by this adapter. Fields the SDK can map are applied through its mapping, the
# rest through a fallback path; empty or unchecked values are skipped.
def _kv_fields_to_client_config(section_key: str, fields: list[dict[str, Any]]) -> dict[str, Any]:
    if not isinstance(fields, list) or not fields:
        return {}

    # Load the SDK's private key mapping; without it every field uses the fallback path.
    try:
        from lmstudio._kv_config import SUPPORTED_SERVER_KEYS
    except Exception:
        SUPPORTED_SERVER_KEYS = {}

    mapping = SUPPORTED_SERVER_KEYS.get(section_key, {}) if isinstance(SUPPORTED_SERVER_KEYS, dict) else {}
    normalized: dict[str, Any] = {}

    # Walk the fields, skipping malformed ones, null values and the bare section key itself.
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

        # Fields of this section are resolved relative to it; other keys are stored by their full path.
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


# Return the record of the local LM Studio model index for a model identifier. A record that is both
# indexed and the default identifier of that model wins; otherwise the last indexed match is used, or an
# empty dict.
def _get_model_index_record(model_name: str) -> dict[str, Any]:
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


# Return the operation (prediction) defaults saved by LM Studio for a model, read from its per-model
# default-config file. Only identifiers of the form owner/name have such a file; others yield an empty dict.
def _get_disk_model_operation_defaults(model_name: str) -> dict[str, Any]:
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


# Return the path of the LM Studio user default-config file for a model, or an empty string when the
# identifier is not of the form owner/name.
def _get_model_default_config_path(model_name: str) -> str:
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


# Persist the custom (ext.virtualModel.customField.*) operation fields of a config into the model's LM Studio
# default-config file so the server applies them during generation. All other fields already stored by
# LM Studio are preserved, and the write is skipped when nothing changed since the last sync.
def _sync_operation_defaults_to_disk(model_name: str, operation_config: dict[str, Any] | None) -> None:
    if not isinstance(operation_config, dict):
        return

    # Collect the custom fields to write, in a stable key order.
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

    # Load the existing file, or start from an empty LM Studio default-config document.
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

    # Only rewrite the extension-backed custom fields and preserve everything
    # else exactly as LM Studio already stored it.
    preserved_fields = []
    for field in existing_fields:
        if not isinstance(field, dict):
            continue
        field_key = str(field.get("key") or "").strip()
        if field_key.startswith("ext.virtualModel.customField."):
            continue
        preserved_fields.append(field)

    # Write the merged field list, or only remember the signature when the file already matches.
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


# Reasoning detection

# Return the reasoning toggle and level defaults of an operation config as (think, level). Top-level think,
# reasoning_effort, think_level and thinking_level keys win; a nested "reasoning" section (enabled, effort)
# only fills what is still missing.
def _extract_reasoning_settings(operation_defaults: dict[str, Any]) -> tuple[Any | None, Any | None]:
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


# Join the key, display name, description and effect texts of a custom field definition into one lowercase
# string that reasoning field detection can search.
def _collect_reasoning_field_hints(field_definition: dict[str, Any]) -> str:
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


# Infer the reasoning toggle and level custom fields from LM Studio model metadata. Return
# (toggle field key, toggle default, level field key, level default); entries stay None when no matching
# field exists. A toggle is the first boolean field and a level the first string or select field whose
# default is a known reasoning level, among fields whose texts mention "reason" or "think".
def _detect_reasoning_custom_fields(raw_info: dict[str, Any]) -> tuple[str | None, Any | None, str | None, Any | None]:
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


# Return the supported reasoning level values (lowercase, unique, in declared order) of the first reasoning
# related string or select custom field that lists options, or an empty list when there is none.
def _extract_reasoning_level_options(raw_info: dict[str, Any]) -> list[str]:
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

        # Options are plain values or {"value": ...} dictionaries; keep each distinct non-empty value once.
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


# Local hardware and text cleanup

# Return the local NVIDIA GPU devices as {"id": index, "name": label} dictionaries, read from nvidia-smi.
# An empty list means nvidia-smi is missing, failed or listed no valid device.
def _get_local_gpu_devices() -> list[dict[str, Any]]:
    # Ask nvidia-smi for "index,name" rows with a short timeout so a missing or hung driver cannot stall.
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

    # Parse each row into a numeric id and a label (falling back to "GPU <id>"); unparsable rows are skipped.
    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2 or not parts[0]:
            continue
        try:
            device_id = int(parts[0])
        except ValueError:
            # The first column is not a GPU index (for example an error or header line).
            continue
        devices.append({"id": device_id, "name": parts[1] or f"GPU {device_id}"})

    return devices


# Remove chat-template control tokens that should never reach the chat UI from generated text.
def _sanitize_generated_text(text: str) -> str:
    cleaned = str(text or "")
    for pattern in CONTROL_TOKEN_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


# Prediction options

# Return model info as (SDK object, serialized dict), taken from the SDK or else from the loaded-model
# listing. When the model cannot be found the result is (None, {"model": model_name}).
def _get_model_info(client: Any, model_name: str) -> tuple[Any | None, dict[str, Any]]:
    try:
        info = client.llm.get_model_info(model_name)
        return info, _serialize_model_info(info)
    except Exception:
        # The direct lookup fails for models that are not loaded; try the loaded-model listing next.
        pass

    try:
        loaded_models = list(client.list_loaded_models())
    except Exception:
        # An unreachable server means no loaded models can be matched.
        loaded_models = []

    for entry in loaded_models:
        if _model_identifiers_match(model_name, _coerce_model_name(entry)):
            return entry, _serialize_model_info(entry)

    return None, {"model": model_name}


# Return the LM Studio reasoning parsing config that splits reasoning out of the reply at <think> tags.
def _build_reasoning_parsing_config(enabled: bool) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "startString": "<think>",
        "endString": "</think>",
    }


# Normalize prediction options for the LM Studio SDK: drop the "raw" entry, always enable reasoning
# parsing, and add the think toggle and level under their parameter names unless the caller already set
# them. Parameter names starting with "ext." are custom fields synced to disk instead, so they are skipped.
def _prepare_native_prediction_options(
    options: dict[str, Any] | None,
    think: Any = None,
    think_level: Any = None,
    think_param_name: str = "think",
    think_level_param_name: str = "reasoning_effort",
) -> dict[str, Any]:
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


# Normalize prediction options for the OpenAI-compatible endpoint used in tool-calling flows. Known OpenAI
# request fields (after aliasing, e.g. num_predict to max_tokens) stay top-level; everything else, the
# think parameter when it is not an OpenAI field, and the reasoning parsing config go into extra_body.
def _prepare_openai_prediction_options(
    options: dict[str, Any] | None,
    think: Any = None,
    think_level: Any = None,
    think_param_name: str = "think",
    think_level_param_name: str = "reasoning_effort",
) -> dict[str, Any]:
    direct_options: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}

    # Split the options into top-level OpenAI fields and extra_body entries.
    for raw_key, raw_value in (options or {}).items():
        if raw_key == "raw":
            continue
        normalized_key = OPENAI_DIRECT_OPTION_ALIASES.get(raw_key, raw_key)
        if normalized_key in OPENAI_DIRECT_OPTION_KEYS:
            direct_options[normalized_key] = raw_value
        else:
            # Unknown options are forwarded to LM Studio through extra_body under their original name.
            extra_body[raw_key] = raw_value

    # top_logprobs is only valid together with logprobs.
    if not _bool_from_value(direct_options.get("logprobs")):
        direct_options.pop("top_logprobs", None)

    # Add the think level and toggle unless the caller already set them.
    if think_level and "reasoning_effort" not in direct_options and "reasoning_effort" not in extra_body:
        direct_options["reasoning_effort"] = think_level
    if think is not None and think_param_name:
        normalized_think_key = OPENAI_DIRECT_OPTION_ALIASES.get(think_param_name, think_param_name)
        target_options = direct_options if normalized_think_key in OPENAI_DIRECT_OPTION_KEYS else extra_body
        target_options.setdefault(normalized_think_key, think)

    extra_body["reasoningParsing"] = _build_reasoning_parsing_config(True)

    # Merge into an extra_body the caller may already have passed.
    if extra_body:
        merged_extra_body = dict(direct_options.get("extra_body", {}) or {})
        merged_extra_body.update(extra_body)
        direct_options["extra_body"] = merged_extra_body

    return direct_options


# OpenAI-compatible client and messages

# Create an OpenAI client pointing at the OpenAI-compatible /v1 endpoint of the LM Studio server.
# Raise ImportError when the openai package is missing.
def _get_openai_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The 'openai' package is required for LM Studio tool support.") from exc

    # Normalize the configured URL: add a scheme and the /v1 suffix when missing. The API key is a placeholder.
    raw_url = settings.get_engine_url("lms")
    if not raw_url.startswith("http"):
        raw_url = f"http://{raw_url}"
    if not raw_url.rstrip("/").endswith("/v1"):
        raw_url = raw_url.rstrip("/") + "/v1"

    return OpenAI(base_url=raw_url, api_key="lm-studio")


# Convert ASLM chat messages into OpenAI-compatible message payloads. Tool results become tool messages;
# messages with images or media become multi-part content (text, image_url, input_audio or file parts);
# assistant tool calls and thinking text are carried over.
def _build_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = message.get("content", "") or ""
        images = message.get("images") or []
        image_mime_types = message.get("image_mime_types") or []
        media = message.get("media") if isinstance(message.get("media"), list) else []

        # Tool results are passed through as tool messages.
        if role == "tool":
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id", message.get("name", "call")),
                    "content": content,
                }
            )
            continue

        # Build multi-part content for attachments: text first, then images, then audio and other files.
        if images or media:
            content_parts: list[dict[str, Any]] = []
            if content:
                content_parts.append({"type": "text", "text": content})
            for index, image_base64 in enumerate(images):
                # Images arrive as bare base64; the MIME type list is optional, so default to JPEG.
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

                # Only mp3 and wav audio goes out as input_audio; every other media item is a file part.
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
            # Plain text message.
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


# Tool events and messages

# Serialize one tool invocation so the UI can render it during streaming.
def _build_tool_event(tool_lookup: dict[str, dict[str, Any]], tool_call: dict[str, Any]) -> dict[str, Any]:
    return tool_registry.build_tool_event(tool_lookup, tool_call)


# Build the tool message of the OpenAI-compatible conversation for one tool result. The model sees only the
# result content; extra result data and the tool event details (alias, server, display name, arguments, UI
# payload) are attached for the application and the chat history.
def _build_tool_message(
    tool_name: str,
    tool_call_id: str,
    content: str | dict[str, Any],
    tool_event: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_content, tool_extras = tool_registry.split_tool_result_payload(content)
    payload: dict[str, Any] = {
        "role": "tool",
        "tool_call_id": tool_call_id,
        "name": tool_name,
        "content": model_content,
    }
    payload.update(tool_extras)

    # Attach the tool event details, falling back to the tool name where the event has no alias or id.
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


# Incremental parser that splits streamed model output into reasoning text and visible text at reasoning
# tags (<think>, <thinking>, <reasoning>). Fragments may cut a tag in half, so a possible partial tag at
# the end of a fragment is held back until the next fragment (or flush) decides what it was.
class _ReasoningTextParser:
    # Store the tag pairs and reset the parsing state.
    def __init__(self, tag_pairs: tuple[tuple[str, str], ...] = REASONING_TAG_PAIRS) -> None:
        # Opening and closing tags that delimit reasoning text.
        self._tag_pairs = tag_pairs
        self._start_tags = [start for start, _end in tag_pairs]
        self._end_tags = [end for _start, end in tag_pairs]

        # Parsing state: whether the parser is inside a reasoning block, the held-back possible partial tag,
        # and the last visible non-blank character of earlier fragments (used for the tag boundary check).
        self._in_reasoning = False
        self._pending = ""
        self._visible_boundary_char: str | None = None

    # Return the position and text of the earliest of the given tags in the source, or None. While outside
    # reasoning, opening tags only count when they sit at a valid boundary.
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

    # Return whether an opening tag at tag_index may start reasoning: it must be at the start of the text or
    # follow (ignoring blanks) a newline, a ">" or punctuation, so a tag quoted mid-sentence stays visible.
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

    # Split the source into (text that is safe to emit, trailing text that could be the start of one of the
    # tags). The trailing part is held back until the next fragment shows whether the tag is completed.
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

    # Parse one streamed fragment and return (thinking text, visible text) found in it. When the SDK already
    # annotates the fragment (reasoning_type), that annotation is trusted: start and end tags only switch the
    # mode and "reasoning" fragments are returned as thinking. Plain fragments are scanned for tags.
    def feed(self, content: str, reasoning_type: str | None = None) -> tuple[str, str]:
        thinking_parts: list[str] = []
        content_parts: list[str] = []

        # Honor the reasoning annotations provided by the native SDK stream.
        if reasoning_type == "reasoningStartTag":
            self._in_reasoning = True
            return "", ""
        if reasoning_type == "reasoningEndTag":
            self._in_reasoning = False
            return "", ""
        if reasoning_type == "reasoning":
            return str(content or ""), ""

        # Scan the fragment (with any held-back text in front) for the next tag that toggles reasoning mode.
        source = f"{self._pending}{content or ''}"
        self._pending = ""

        while source:
            tags = self._end_tags if self._in_reasoning else self._start_tags
            next_tag = self._find_next_tag(source, tags)
            if next_tag is None:
                # No tag left: emit the text and hold back a possible partial tag at its end.
                visible, pending = self._split_possible_tag_prefix(source, tags)
                if self._in_reasoning:
                    thinking_parts.append(visible)
                else:
                    content_parts.append(visible)
                    self._remember_visible_text(visible)
                self._pending = pending
                break

            # Emit the text before the tag, drop the tag itself and flip between reasoning and visible mode.
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

    # Release the held-back partial fragment at the end of the stream as (thinking text, visible text).
    def flush(self) -> tuple[str, str]:
        pending = self._pending
        self._pending = ""
        if not pending:
            return "", ""
        if self._in_reasoning:
            return pending, ""
        self._remember_visible_text(pending)
        return "", pending


# Native chat and streaming

# Upload one base64 image to LM Studio and return its file handle. With a cache, identical images (by
# SHA-256 of the base64 text) are uploaded once and the handle is reused.
def _prepare_image_handle(
    client: Any,
    image_base64: str,
    name: str,
    cache: dict[str, Any] | None = None,
) -> Any:
    digest = hashlib.sha256(str(image_base64 or "").encode("utf-8")).hexdigest()
    if cache is not None and digest in cache:
        return cache[digest]

    image_bytes = base64.b64decode(image_base64)
    handle = client.prepare_image(io.BytesIO(image_bytes), name=name)
    if cache is not None:
        cache[digest] = handle
    return handle


# Convert generic chat messages (system, assistant, tool and user roles) into an LM Studio SDK chat
# history. Images of user messages are uploaded once and attached as file handles.
def _build_native_chat_history(client: Any, messages: list[dict[str, Any]]):
    lms = _get_sdk()
    chat = lms.Chat()
    image_handle_cache: dict[str, Any] = {}

    for message_index, message in enumerate(messages, start=1):
        role = str(message.get("role", "user")).lower()
        content = str(message.get("content", "") or "")
        images = [str(item) for item in (message.get("images") or []) if str(item).strip()]

        # System, assistant and tool messages map directly onto SDK calls.
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


# Run one native LM Studio SDK prediction and yield {"message": {...}} chunks with visible content and
# optional thinking text (one chunk per fragment when streaming, a single chunk otherwise). When no chunk
# was yielded but text exists, a final assembled message is yielded instead. Streaming stops early when an
# abort was requested.
def _stream_native_response(model: Any, chat: Any, options: dict[str, Any], stream: bool):
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

    # Emit whatever the parser still held back at the end of the output.
    tail_thinking, tail_content = parser.flush()
    if tail_thinking or tail_content:
        visible_content += tail_content
        thinking_content += tail_thinking
        yielded_chunk = True
        payload = {"role": "assistant", "content": tail_content}
        if tail_thinking:
            payload["thinking"] = tail_thinking
        yield {"message": payload}

    # Defensive fallback: yield the assembled message when text exists but no chunk was yielded.
    assistant_message: dict[str, Any] = {"role": "assistant", "content": visible_content}
    if thinking_content:
        assistant_message["thinking"] = thinking_content

    if not yielded_chunk and (visible_content or thinking_content):
        yield {"message": assistant_message}


# Stream one model round through the OpenAI-compatible endpoint. The generator yields {"message": {...}}
# chunks with visible and thinking text as they arrive and returns the assembled assistant message
# (content, thinking and reconstructed tool calls). The tool choice is forced to forced_tool_name when it is
# available, "required" when require_tool_call is set, otherwise "auto".
def _stream_openai_round(
    client,
    model_name: str,
    conversation: list[dict[str, Any]],
    options: dict[str, Any],
    tools: list[dict[str, Any]] | None = None,
    forced_tool_name: str = "",
    require_tool_call: bool = False,
):
    # Build the request options and pick the tool_choice mode.
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

    # Accumulators for the assembled message: texts, tool calls keyed by their stream index, and the parser.
    assistant_content = ""
    assistant_thinking = ""
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    parser = _ReasoningTextParser()

    # Start the streaming request.
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

    # Emit whatever the parser still held back at the end of the stream.
    tail_thinking, tail_content = parser.flush()
    if tail_thinking or tail_content:
        assistant_content += tail_content
        assistant_thinking += tail_thinking
        payload = {"content": tail_content}
        if tail_thinking:
            payload["thinking"] = tail_thinking
        yield {"message": payload}

    # Assemble the final assistant message with the tool calls in stream order.
    assembled_tool_calls = [tool_calls_by_index[idx] for idx in sorted(tool_calls_by_index)]
    assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant_content}
    if assistant_thinking:
        assistant_message["thinking"] = assistant_thinking
    if assembled_tool_calls:
        assistant_message["tool_calls"] = assembled_tool_calls

    return assistant_message


# Tool loop

# Forward every chunk of a round stream and return the round's final assistant message (an empty assistant
# message when the stream returned none). Used with "yield from" to keep the return value of the stream.
def _yield_stream_round(round_stream):
    while True:
        try:
            yield next(round_stream)
        except StopIteration as stop:
            return stop.value or {"role": "assistant", "content": ""}


# Run a generation with local tools through LM Studio's OpenAI-compatible tool calling. The generator yields
# message chunks plus "transcript_message", "tool_events" and "tool_result" items. Each round streams a model
# answer; tool calls it contains are executed locally (with preflight, cooldown, duplicate and quota checks)
# and their results are fed back until the model answers without tools. A supervisor re-prompts when
# user-required tools were ignored, forces a final answer after cancelled or repeatedly blocked tools, and
# the loop ends with an error message when the round limit is exceeded.
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
    # Resolve the tool definitions available for the selected tool servers.
    tools, tool_lookup = tool_registry.build_ollama_tools(
        tool_server_ids,
        engine="lms",
        model_name=model_name,
        tool_source_map=tool_context.get("tool_source_map") if isinstance(tool_context, dict) else None,
        allowed_tool_aliases=tool_context.get("allowed_tool_aliases") if isinstance(tool_context, dict) else None,
        instant_mode=bool(tool_context.get("instant_mode")) if isinstance(tool_context, dict) else False,
        instant_search_batch_size=tool_context.get("instant_search_batch_size") if isinstance(tool_context, dict) else None,
    )

    # Without any usable tool this is a plain single-round chat.
    if not tools:
        request_messages = conversation if conversation is not None else _build_openai_messages(messages)
        yield from _yield_stream_round(
            _stream_openai_round(client, model_name, request_messages, options)
        )
        return

    # Prepare the conversation and the per-generation bookkeeping: tool quotas, duplicate detection, blocked
    # result streak and the required-tool supervisor state.
    conversation = conversation if conversation is not None else _build_openai_messages(messages)
    conversation = tool_registry.with_available_tools_prompt(conversation, tool_lookup)
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

    # Decide for each round whether a specific tool is forced or any required tool call is demanded.
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

        # No tool calls: the model answered. Re-prompt when required tools were ignored, otherwise finish.
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

        # Convert OpenAI-compatible tool payloads into the internal dispatch
        # shape used by the shared MCP tool bridge.
        tool_calls = []
        for tc in raw_tool_calls:
            fn = tc.get("function", {})
            name = fn.get("name", "")
            try:
                arguments = json.loads(fn.get("arguments", "{}"))
            except (json.JSONDecodeError, TypeError):
                # Malformed arguments from the model are treated as an empty argument set.
                arguments = {}
            tool_calls.append(
                {
                    "name": name,
                    "arguments": arguments,
                    "id": tc.get("id", f"call_{round_index}_{name}"),
                }
            )

        # Normalize and limit the calls, then record which required tool servers they satisfy.
        tool_calls = tool_registry.prepare_tool_calls(tool_lookup, tool_calls)
        tool_calls = tool_registry.limit_tool_calls_from_context(tool_context, tool_calls, tool_lookup)

        for tool_call in tool_calls:
            satisfied_server_id = tool_registry.tool_server_id_for_alias(tool_lookup, tool_call["name"])
            if satisfied_server_id:
                satisfied_tool_server_ids.add(satisfied_server_id)

        # Replace the raw assistant message with its canonical form in the conversation and the transcript.
        canonical_assistant_message = tool_registry.canonicalize_transcript_tool_calls(
            assistant_message, tool_calls
        )
        conversation[-1] = canonical_assistant_message
        yield {"transcript_message": canonical_assistant_message}

        # Announce the calls to the UI; the index suffix keeps aliases unique within the round.
        tool_events = [_build_tool_event(tool_lookup, tool_call) for tool_call in tool_calls]
        for _i, _ev in enumerate(tool_events):
            _ev["alias"] = f"{_ev['alias']}__{_i}"
        yield {"tool_events": tool_events}

        # Execute the calls one by one and append each result to the conversation.
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

            # Pick the result: a preflight error, a cooldown, duplicate or quota error, or the real tool run.
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

            # Track cancelled results and the streak of blocking results for the checks after the round.
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

        # Stop on a user abort. After a cancelled tool or two blocked results in a row, force a final answer
        # without further tool calls.
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

        # Keep the conversation within budget before the next round.
        conversation = tool_registry.maybe_compact_tool_conversation(
            tool_context, conversation, provider_format="standard"
        )

    # Every round ended with tool calls: give up and report it.
    yield {"message": {"content": "[Error during generation: tool loop exceeded the safety limit.]"}}


# Public adapter API

# Return whether the conversation already contains tool state (a tool result or an assistant tool call).
def _conversation_uses_tools(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if str(message.get("role", "")).lower() == "tool":
            return True
        if message.get("tool_calls"):
            return True
    return False


# Return the names of the models available in the configured LM Studio server, or an empty list when the
# server cannot be reached. LM Studio can serve any downloaded model through just-in-time loading, so every
# downloaded LLM is listed rather than only the currently loaded ones (a running server with no model loaded
# would otherwise report an empty list). The loaded list is the fallback when the downloaded query is
# unavailable.
def get_models() -> list[Any]:
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


# Release runtime resources. Nothing to do here: the LM Studio model lifecycle is managed outside this adapter.
def cleanup_runtime() -> None:
    return None


# Reject a download request, because LM Studio downloads are managed by LM Studio itself.
def download_model(model_name: str, **kwargs: Any) -> Any:
    raise NotImplementedError("LM Studio model downloads are managed by LM Studio.")


# Generate a streamed or non-streamed response through LM Studio and yield its chunks. Conversations that
# use tools go through the OpenAI-compatible endpoint (tool loop); plain chats use the native SDK.
# Recognized keyword arguments: stream, options, think, think_level, think_param_name,
# think_level_param_name, sync_operation_defaults, tool_server_ids (or the legacy tool_server_id and
# tool_id) and tool_context.
def generate(model_name: str, messages: list[dict[str, Any]], **kwargs: Any):
    # Normalize legacy single-id inputs into the shared multi-server format.
    raw_ids = kwargs.pop("tool_server_ids", None) or kwargs.pop("tool_server_id", None) or kwargs.pop("tool_id", None)
    if isinstance(raw_ids, str):
        tool_server_ids = [raw_ids] if raw_ids.strip() else []
    elif isinstance(raw_ids, list):
        tool_server_ids = [str(s) for s in raw_ids if str(s).strip()]
    else:
        tool_server_ids = []

    # Read the generation settings.
    tool_context = dict(kwargs.pop("tool_context", {}) or {})
    stream = bool(kwargs.get("stream", False))
    raw_options = dict(kwargs.get("options", {}) or {})
    sync_operation_defaults = dict(kwargs.get("sync_operation_defaults", {}) or {})
    think = kwargs.get("think")
    think_level = kwargs.get("think_level")
    think_param_name = str(kwargs.get("think_param_name", "think") or "think")
    think_level_param_name = str(kwargs.get("think_level_param_name", "reasoning_effort") or "reasoning_effort")

    # Reset the abort flag of the previous generation and sync custom operation fields to LM Studio.
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

        # Internal chat keys are application settings and must not reach the model request.
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

    # Plain chat: stream the response through the native SDK.
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


# Build the settings and capabilities payload of one model for the settings panel by combining the
# downloaded-model catalog with the settings of the loaded instance, the local model index and the user
# defaults saved by LM Studio. The result holds the context length, default operation config, detected
# thinking toggle and level parameters, vision, tool and file support, runtime hardware limits and the raw
# model info.
def get_model_settings(model_name: str) -> dict[str, Any]:
    # Read the catalog entry first; loaded-instance info is only fetched through the SDK when the catalog
    # has no entry or the model is loaded. An SDK failure is tolerated as long as the catalog exists.
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

    # Read the local model index and the user defaults saved on disk.
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

    # Collect the capability sources and the operation defaults (model config merged with the disk defaults).
    metadata_overrides = _normalize_model_config(raw_info.get("metadataOverrides") or {})
    catalog_capabilities = catalog_info.get("capabilities", {})
    if not isinstance(catalog_capabilities, dict):
        catalog_capabilities = {}
    catalog_reasoning = catalog_capabilities.get("reasoning")
    config_payload = _normalize_model_config(raw_info.get("config") or {})
    operation_defaults = _merge_nested_dicts(_normalize_model_config(config_payload.get("operation") or {}), disk_operation_defaults)

    # Detect reasoning toggle and level custom fields and seed their defaults into the operation defaults.
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

    # Resolve the context length (falling back to 8192) and the vision, tool and thinking capabilities.
    # The context length prefers loaded-instance info and metadata overrides over the catalog; vision and
    # tool support prefer the catalog.
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

    # Choose the request parameter names for thinking and a default level list when none was declared.
    think_param_name = custom_toggle_name or "think"
    think_level_param_name = custom_level_name or "reasoning_effort"
    if not think_level_options and supports_think_level:
        if custom_level_name or think_level_param_name == "reasoning_effort":
            think_level_options = ["low", "medium", "high"]
        elif think_level_param_name in {"think_level", "thinking_level"}:
            think_level_options = ["low", "medium", "high"]

    # Probe local hardware and list the capability labels shown in the UI.
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
