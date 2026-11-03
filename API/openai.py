# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import re
import threading
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from API import mcp as tool_registry
from Settings import settings
from Settings.proxy_policy import urlopen_with_loopback_bypass

logger = logging.getLogger(__name__)

# Process-wide abort flag: set by abort_generation, cleared when a new generation starts and checked
# while a response is streamed.
_abort_event = threading.Event()

# Cache of companion model catalogs by URL, guarded by its lock because requests run on several threads.
_metadata_cache_lock = threading.Lock()
_supplemental_models_cache: dict[str, list[dict[str, Any]]] = {}

# Default cap on model rounds in one tool loop; the request context may lower or raise it.
MAX_TOOL_ROUNDS = 100

# Opening and closing tags that wrap reasoning text inside streamed content.
REASONING_TAG_PAIRS = (
    ("<think>", "</think>"),
    ("<thinking>", "</thinking>"),
    ("<reasoning>", "</reasoning>"),
)

# Service control tokens (harmony and ChatML style) that some backends leak into generated text.
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

# Provider parameter names that switch thinking on or off, and that choose the reasoning level.
THINK_PARAM_NAMES = {"think", "thinking", "reasoning"}
THINK_LEVEL_PARAM_NAMES = {"think_level", "thinking_level", "reasoning_effort"}

# Normalized capability, tag or flag names that providers use to advertise a model feature
# (tool calling, image input, audio input, video input and reasoning).
TOOL_CAPABILITY_NAMES = {
    "tool",
    "tools",
    "tool_calling",
    "tool_use",
    "function_calling",
    "functions",
    "supports_tools",
    "supports_tool_calling",
    "trained_for_tool_use",
}
VISION_CAPABILITY_NAMES = {
    "vision",
    "image",
    "images",
    "image_input",
    "image_inputs",
    "multimodal",
    "multimodal_input",
    "supports_vision",
}
AUDIO_INPUT_CAPABILITY_NAMES = {"audio", "audio_input", "input_audio", "supports_audio_input"}
VIDEO_INPUT_CAPABILITY_NAMES = {"video", "video_input", "input_video", "supports_video_input"}
REASONING_CAPABILITY_NAMES = {
    "think",
    "thinking",
    "reasoning",
    "chain_of_thought",
    "supports_reasoning",
    "reasoning_enabled",
}

# Generic option names that are renamed to their OpenAI equivalent before the request is built.
DIRECT_OPTION_ALIASES = {
    "num_predict": "max_tokens",
}

# Options that the OpenAI client accepts as top-level keyword arguments; every other option goes
# through extra_body.
DIRECT_OPTION_KEYS = {
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
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "verbosity",
}

# Endpoint hosts (including their subdomains) that expect reasoning controls as a single
# "reasoning" object instead of separate flat options.
OBJECT_REASONING_ENDPOINT_HOSTS = {
    "openrouter.ai",
}

# Option names that carry the reasoning effort and are folded into the reasoning object.
REASONING_EFFORT_KEYS = {
    "reasoning_effort",
    "think_level",
    "thinking_level",
}

# Metadata keys whose nested values hold default parameter values.
DEFAULT_CONTAINER_KEYS = {
    "defaults",
    "default_parameters",
    "default_settings",
    "parameters",
    "settings",
}

# Metadata keys whose nested values list the parameters a model supports.
PARAMETER_CONTAINER_KEYS = {
    "supported_parameters",
    "parameter_schema",
    "parameters",
}


# Shared helpers

# Signal the active OpenAI-compatible generation to stop; the streaming code checks the flag.
def abort_generation() -> None:
    _abort_event.set()


# Close a client instance, ignoring every error.
def _close_client(client: Any) -> None:
    try:
        client.close()
    except Exception:
        # Closing is best-effort cleanup; a failing close must not hide the real result or error.
        pass


# Return a lower-case identifier for one key or token: every run of characters other than letters and
# digits becomes a single underscore, and leading and trailing underscores are removed.
def _normalize_key_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


# Return a boolean for a JSON-like value. None gives the default, and the usual true/false words
# ("on", "yes", "enabled", "supported" and so on) are understood; other strings fall back to bool().
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


# Convert a scalar into a positive integer, or return None for booleans, non-numbers and values below 1.
def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return None
    return numeric_value if numeric_value > 0 else None


# Recursively merge two dictionaries without mutating either input; on conflicts the override wins.
def _merge_nested_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_nested_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


# Drop empty nested config values ({}, [], "" and None) from dicts and lists. Falsy scalars such as 0
# and False are kept, so real default values survive.
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


# Convert SDK objects (pydantic models and similar) recursively into plain Python containers.
# Unknown objects fall back to their public attributes, and finally to str().
def _to_plain_data(value: Any) -> Any:
    # Scalars pass through; dicts and sequences are converted element by element.
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {str(key): _to_plain_data(child) for key, child in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_plain_data(child) for child in value]

    # Try the serializers that SDK models commonly expose, in order of preference.
    serialized: Any = None
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
            # The serializer rejected the keyword arguments, so retry without them.
            try:
                serialized = serializer()
            except Exception:
                # This serializer does not work for the object; try the next one.
                continue
        except Exception:
            # This serializer failed; try the next one.
            continue
        break

    # Use the serialized dict and add the extra fields that pydantic keeps outside the declared fields.
    if isinstance(serialized, dict):
        normalized = {str(key): _to_plain_data(child) for key, child in serialized.items()}
        model_extra = getattr(value, "model_extra", None)
        if isinstance(model_extra, dict):
            for key, child in model_extra.items():
                normalized.setdefault(str(key), _to_plain_data(child))
        return normalized

    if isinstance(serialized, (list, tuple)):
        return [_to_plain_data(child) for child in serialized]

    # No serializer worked: read the public instance attributes instead.
    if hasattr(value, "__dict__"):
        serialized_dict = {
            str(key): _to_plain_data(child)
            for key, child in vars(value).items()
            if not str(key).startswith("_")
        }
        if serialized_dict:
            model_extra = getattr(value, "model_extra", None)
            if isinstance(model_extra, dict):
                for key, child in model_extra.items():
                    serialized_dict.setdefault(str(key), _to_plain_data(child))
            return serialized_dict

    return str(value)


# Model metadata inspection

# Return a model identifier trimmed, stripped of surrounding slashes and lower-cased for comparisons.
def _normalize_model_identifier(value: Any) -> str:
    return str(value or "").strip().strip("/").lower()


# Return whether two model identifiers likely refer to the same model: they are equal after
# normalization, or they share the last path segment (so "org/model" matches "model").
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


# Yield every nested value, at any depth of dicts and lists, whose normalized key is one of the
# target keys. A matching value is yielded before its own children are searched.
def _iter_values_for_keys(source: Any, target_keys: set[str]):
    if isinstance(source, dict):
        for key, value in source.items():
            if _normalize_key_name(key) in target_keys:
                yield value
            yield from _iter_values_for_keys(value, target_keys)
        return

    if isinstance(source, list):
        for item in source:
            yield from _iter_values_for_keys(item, target_keys)


# Collect normalized names from capability-like metadata: strings, the keys of dicts whose value is
# true, and the values of descriptive keys such as "name" or "type", searched recursively.
def _extract_named_values(value: Any) -> set[str]:
    # A plain string is one name.
    if isinstance(value, str):
        normalized = _normalize_key_name(value)
        return {normalized} if normalized else set()

    # In a dict, a true flag contributes its key and a descriptive key contributes its value.
    if isinstance(value, dict):
        normalized_values: set[str] = set()
        for key, child in value.items():
            normalized_key = _normalize_key_name(key)
            if isinstance(child, bool):
                if child and normalized_key:
                    normalized_values.add(normalized_key)
                continue

            if isinstance(child, (str, int, float)) and normalized_key in {
                "ability",
                "capability",
                "feature",
                "id",
                "kind",
                "modality",
                "name",
                "parameter",
                "type",
                "value",
            }:
                normalized_child = _normalize_key_name(child)
                if normalized_child:
                    normalized_values.add(normalized_child)

            normalized_values.update(_extract_named_values(child))
        return normalized_values

    # A list contributes the names of all its elements.
    if isinstance(value, list):
        normalized_values: set[str] = set()
        for child in value:
            normalized_values.update(_extract_named_values(child))
        return normalized_values

    return set()


# Return the normalized feature tokens a model payload advertises under keys such as "capabilities",
# "features", "modalities" or "tags".
def _extract_capability_tokens(raw_model: dict[str, Any]) -> set[str]:
    capability_tokens: set[str] = set()
    for container in _iter_values_for_keys(
        raw_model,
        {
            "abilities",
            "ability",
            "capabilities",
            "features",
            "feature_flags",
            "input_modalities",
            "modalities",
            "output_modalities",
            "supported_modalities",
            "tags",
            "traits",
        },
    ):
        capability_tokens.update(_extract_named_values(container))
    return capability_tokens


# Return an explicit boolean capability flag when the payload exposes one under any of the feature
# names, or None when it does not. The result is true when any matching entry says the feature is on.
def _extract_feature_flag(raw_model: dict[str, Any], feature_names: set[str]) -> bool | None:
    normalized_targets = {_normalize_key_name(name) for name in feature_names}
    matched_values: list[bool] = []

    # Walk all nested metadata and record the value of every matching flag.
    def visit(source: Any) -> None:
        if isinstance(source, dict):
            for key, value in source.items():
                normalized_key = _normalize_key_name(key)
                if normalized_key in normalized_targets:
                    if isinstance(value, dict):
                        # Providers expose the same capability in several shapes,
                        # so prefer explicit support flags before falling back to
                        # looser truthy heuristics.
                        availability_flags = [
                            _bool_from_value(value.get(flag_key), default=False)
                            for flag_key in ("supported", "available")
                            if flag_key in value
                        ]
                        if availability_flags:
                            matched_values.append(any(availability_flags))
                        elif any(
                            _normalize_key_name(child_key) not in {"enabled", "value", "default", "default_value"}
                            for child_key in value.keys()
                        ):
                            matched_values.append(True)
                        elif "enabled" in value:
                            matched_values.append(_bool_from_value(value.get("enabled"), default=False))
                        elif "value" in value:
                            matched_values.append(_bool_from_value(value.get("value"), default=False))
                        else:
                            matched_values.append(bool(value))
                    else:
                        matched_values.append(_bool_from_value(value, default=False))
                visit(value)
            return

        if isinstance(source, list):
            for item in source:
                visit(item)

    visit(raw_model)

    # No matching key at all means the payload says nothing about this feature.
    if not matched_values:
        return None
    return any(matched_values)


# Runtime defaults and supported parameters

# Extract parameter defaults from one config or schema container. A parameter given as a dict
# contributes its "default", "defaultValue" or plain "value"; a scalar or list value is a default itself.
def _extract_defaults_from_container(container: Any) -> dict[str, Any]:
    if not isinstance(container, dict):
        return {}

    defaults: dict[str, Any] = {}
    properties = container.get("properties")
    if isinstance(properties, dict):
        defaults = _merge_nested_dicts(defaults, _extract_defaults_from_container(properties))

    for key, value in container.items():
        normalized_key = _normalize_key_name(key)

        # Skip schema keywords; they describe parameters but are not parameters.
        if normalized_key in {
            "description",
            "enum",
            "items",
            "options",
            "properties",
            "required",
            "schema",
            "title",
            "type",
            "values",
        }:
            continue

        # A parameter definition: take its explicit default, or its value when it is not a schema node.
        if isinstance(value, dict):
            if "default" in value:
                defaults[str(key)] = _to_plain_data(value.get("default"))
                continue
            if "defaultValue" in value:
                defaults[str(key)] = _to_plain_data(value.get("defaultValue"))
                continue
            if "value" in value and not any(
                _normalize_key_name(child_key) in {"description", "enum", "options", "title", "type", "values"}
                for child_key in value.keys()
            ):
                defaults[str(key)] = _to_plain_data(value.get("value"))
                continue
            continue

        # A plain value is the default itself.
        if isinstance(value, (str, int, float, bool, list)):
            defaults[str(key)] = _to_plain_data(value)

    cleaned_defaults = _clean_nested_config(defaults)
    return cleaned_defaults if isinstance(cleaned_defaults, dict) else {}


# Extract the enumerated option values of one parameter definition as unique lower-case strings in
# their original order. The definition may be a list itself or a dict with an "enum", "options" or
# similar key.
def _extract_option_values(definition: Any) -> list[str]:
    # Find the raw option list under the first key that holds one.
    raw_options: Any = []
    if isinstance(definition, dict):
        for key in ("enum", "options", "values", "allowed_values", "allowedValues"):
            candidate = definition.get(key)
            if isinstance(candidate, list):
                raw_options = candidate
                break
    elif isinstance(definition, list):
        raw_options = definition

    normalized_options: list[str] = []
    if not isinstance(raw_options, list):
        return normalized_options

    # Options may be plain values or objects carrying the value under value, id, name or key.
    for option in raw_options:
        if isinstance(option, dict):
            option_value = option.get("value", option.get("id", option.get("name", option.get("key"))))
        else:
            option_value = option
        normalized_value = str(option_value or "").strip().lower()
        if not normalized_value or normalized_value in normalized_options:
            continue
        normalized_options.append(normalized_value)

    return normalized_options


# Return the reasoning toggle default, the reasoning level default and the supported level options
# found under any "reasoning" or "thinking" entry of the model payload. The first value found for
# each of them wins.
def _extract_reasoning_metadata(raw_model: dict[str, Any]) -> tuple[Any | None, Any | None, list[str]]:
    toggle_default = None
    level_default = None
    level_options: list[str] = []

    for container in _iter_values_for_keys(raw_model, {"reasoning", "thinking"}):
        if not isinstance(container, dict):
            continue

        # First collect the on/off default, then look for a separate effort
        # field if the provider exposes reasoning levels independently.
        for key in ("enabled", "available", "supported", "value"):
            if toggle_default is None and key in container:
                toggle_default = container.get(key)
                break

        # Without an explicit flag, read a textual on/off default.
        if toggle_default is None:
            for key in ("default", "defaultValue"):
                if key not in container:
                    continue
                default_value = container.get(key)
                normalized_default = str(default_value or "").strip().lower()
                if normalized_default in {"on", "off", "true", "false"}:
                    toggle_default = normalized_default in {"on", "true"}
                    break

        # The level default and options come from an effort-like field, which is either a plain value
        # or a dict with its own default and options.
        for key in ("effort", "level", "reasoning_effort", "thinking_level"):
            candidate = container.get(key)
            if candidate is None:
                continue

            if isinstance(candidate, dict):
                if level_default is None:
                    for default_key in ("default", "defaultValue", "value"):
                        if default_key in candidate:
                            level_default = candidate.get(default_key)
                            break
                if not level_options:
                    level_options = _extract_option_values(candidate)
            else:
                if level_default is None:
                    level_default = candidate

        # Fall back to a dedicated list of levels next to the effort field.
        if not level_options:
            for key in ("levels", "options", "values", "allowed_options", "allowedValues"):
                candidate = container.get(key)
                if candidate is None:
                    continue
                level_options = _extract_option_values(candidate)
                if level_options:
                    break

        # Check whether the reported levels are really just on/off values.
        normalized_level_options = [str(option).strip().lower() for option in level_options if str(option).strip()]
        if normalized_level_options and set(normalized_level_options).issubset({"on", "off", "true", "false"}):
            # Some backends report boolean-style options in the level field.
            # Treat those as toggle metadata instead of fake effort levels.
            if toggle_default is None:
                normalized_default = str(container.get("default", container.get("defaultValue", "")) or "").strip().lower()
                if normalized_default in {"on", "true"}:
                    toggle_default = True
                elif normalized_default in {"off", "false"}:
                    toggle_default = False
            level_options = []
            level_default = None

    return toggle_default, level_default, level_options


# Return the default runtime options a model exposes: the merged defaults of every config container,
# plus the reasoning toggle ("think") and level ("reasoning_effort") defaults when they are not set yet.
def _extract_defaults(raw_model: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}

    # Merge the defaults of every container in the payload.
    for container in _iter_values_for_keys(raw_model, DEFAULT_CONTAINER_KEYS):
        extracted = _extract_defaults_from_container(container)
        if extracted:
            defaults = _merge_nested_dicts(defaults, extracted)

    # Add the reasoning defaults without overriding explicit ones.
    reasoning_toggle_default, reasoning_level_default, _reasoning_level_options = _extract_reasoning_metadata(raw_model)
    if reasoning_toggle_default is not None and "think" not in defaults:
        defaults["think"] = reasoning_toggle_default
    if reasoning_level_default is not None and "reasoning_effort" not in defaults:
        defaults["reasoning_effort"] = reasoning_level_default

    cleaned_defaults = _clean_nested_config(defaults)
    return cleaned_defaults if isinstance(cleaned_defaults, dict) else {}


# Extract the parameter names declared by one schema-like container. The container is either a dict
# (a schema or a name-to-definition map) or a list of names and parameter objects.
def _extract_parameter_names_from_container(container: Any) -> set[str]:
    parameter_names: set[str] = set()

    # Dict container: keys are parameter names unless they are schema keywords.
    if isinstance(container, dict):
        properties = container.get("properties")
        if isinstance(properties, dict):
            parameter_names.update(_extract_parameter_names_from_container(properties))

        for key, value in container.items():
            normalized_key = _normalize_key_name(key)

            # Schema keywords are not parameters; only "schema" and "items" can wrap real parameters.
            if normalized_key in {
                "default",
                "default_value",
                "description",
                "enum",
                "items",
                "options",
                "properties",
                "required",
                "schema",
                "title",
                "type",
                "values",
            }:
                if normalized_key in {"schema", "items"}:
                    parameter_names.update(_extract_parameter_names_from_container(value))
                continue

            if isinstance(value, dict):
                # Nested dicts may be either real parameter definitions or just
                # more structural schema nodes, so inspect their keys first.
                nested_keys = {_normalize_key_name(child_key) for child_key in value.keys()}
                if nested_keys & {"default", "default_value", "description", "enum", "options", "properties", "type", "values"}:
                    parameter_names.add(str(key))
                    parameter_names.update(_extract_parameter_names_from_container(value.get("properties")))
                    continue
                parameter_names.update(_extract_parameter_names_from_container(value))
                continue

            # A list value is a parameter when it holds options, and its elements may declare more names.
            if isinstance(value, list):
                if value and all(isinstance(item, (str, dict)) for item in value):
                    parameter_names.add(str(key))
                for item in value:
                    parameter_names.update(_extract_parameter_names_from_container(item))
                continue

            # A scalar value means the key itself is a parameter.
            if isinstance(value, (str, int, float, bool)):
                parameter_names.add(str(key))

        return parameter_names

    # List container: plain items are names, dict items carry a name and may nest more parameters.
    if isinstance(container, list):
        for item in container:
            if isinstance(item, (str, int, float)):
                parameter_name = str(item).strip()
                if parameter_name:
                    parameter_names.add(parameter_name)
                continue
            if not isinstance(item, dict):
                continue
            parameter_name = item.get("name", item.get("id", item.get("key", item.get("parameter"))))
            if parameter_name:
                parameter_names.add(str(parameter_name))
            for key in ("properties", "schema", "items"):
                parameter_names.update(_extract_parameter_names_from_container(item.get(key)))

    return parameter_names


# Return the runtime parameter names a model supports: the keys of its defaults plus every name
# declared in a supported-parameters container. Names that normalize to an empty key are dropped.
def _extract_supported_parameter_names(raw_model: dict[str, Any], defaults: dict[str, Any]) -> set[str]:
    parameter_names = {str(key) for key in defaults.keys()}
    for container in _iter_values_for_keys(raw_model, PARAMETER_CONTAINER_KEYS):
        parameter_names.update(_extract_parameter_names_from_container(container))
    return {name for name in parameter_names if _normalize_key_name(name)}


# Collect the supported option values of the given parameters from anywhere in the model metadata,
# without duplicates and in discovery order. A parameter appears either as a dict key or as an item
# with a name.
def _extract_parameter_option_values(raw_model: dict[str, Any], parameter_names: set[str]) -> list[str]:
    normalized_targets = {_normalize_key_name(name) for name in parameter_names if _normalize_key_name(name)}
    collected_options: list[str] = []

    # Keep option values unique while preserving discovery order.
    def register_options(definition: Any) -> None:
        for option in _extract_option_values(definition):
            if option not in collected_options:
                collected_options.append(option)

    # Recurse through dict and list payloads to find matching parameters.
    def visit(source: Any) -> None:
        if isinstance(source, dict):
            for key, value in source.items():
                if _normalize_key_name(key) in normalized_targets:
                    register_options(value)
                visit(value)
            return

        if isinstance(source, list):
            for item in source:
                if isinstance(item, dict):
                    parameter_name = item.get("name", item.get("id", item.get("key", item.get("parameter"))))
                    if _normalize_key_name(parameter_name) in normalized_targets:
                        register_options(item)
                visit(item)

    visit(raw_model)
    return collected_options


# Find which of the provider's parameter names stands for a candidate concept (for example the
# thinking toggle). An exact match on the normalized name wins, then a name that ends with a
# candidate; otherwise the default name is returned.
def _resolve_parameter_name(
    parameter_names: set[str],
    candidate_names: set[str],
    default_name: str,
) -> str:
    normalized_map = {
        _normalize_key_name(parameter_name): str(parameter_name)
        for parameter_name in parameter_names
        if _normalize_key_name(parameter_name)
    }

    for candidate_name in candidate_names:
        resolved = normalized_map.get(_normalize_key_name(candidate_name))
        if resolved:
            return resolved

    for normalized_name, original_name in normalized_map.items():
        for candidate_name in candidate_names:
            normalized_candidate = _normalize_key_name(candidate_name)
            if normalized_candidate and normalized_name.endswith(normalized_candidate):
                return original_name

    return default_name


# Return the largest context length found in the model metadata, or 8192 when it states none.
def _extract_context_length(raw_model: dict[str, Any]) -> int:
    # Gather every positive number under a context-size key; a dict value may nest it under
    # limit, max or value.
    context_values: list[int] = []
    for value in _iter_values_for_keys(
        raw_model,
        {
            "context_length",
            "context_window",
            "input_token_limit",
            "max_context_length",
            "max_context_window",
            "max_input_tokens",
            "max_tokens",
        },
    ):
        if isinstance(value, dict):
            for key in ("limit", "max", "value"):
                numeric_value = _coerce_positive_int(value.get(key))
                if numeric_value is not None:
                    context_values.append(numeric_value)
        else:
            numeric_value = _coerce_positive_int(value)
            if numeric_value is not None:
                context_values.append(numeric_value)

    return max(context_values) if context_values else 8192


# Provider-specific reasoning controls

# Return whether a hostname is the expected host or one of its subdomains, ignoring case.
def _endpoint_host_matches(hostname: str, expected_host: str) -> bool:
    normalized_hostname = str(hostname or "").strip().lower()
    normalized_expected = str(expected_host or "").strip().lower()
    return normalized_hostname == normalized_expected or normalized_hostname.endswith(f".{normalized_expected}")


# Return whether the endpoint expects reasoning controls as one "reasoning" object. The endpoint is
# the given base URL, or the configured OpenAI engine URL when none is given.
def _uses_object_reasoning_controls(base_url: Any = None) -> bool:
    configured_base_url = str(base_url if base_url is not None else settings.get_engine_url("openai") or "").strip()
    if not configured_base_url:
        return False

    try:
        parsed = urllib.parse.urlsplit(configured_base_url)
    except ValueError:
        return False

    hostname = parsed.hostname or ""
    return any(_endpoint_host_matches(hostname, expected_host) for expected_host in OBJECT_REASONING_ENDPOINT_HOSTS)


# Convert a reasoning control (dict, boolean, on/off word, effort name or None) into the object shape
# used by some providers: {"enabled": ...} and/or {"effort": ...}. The effort is added unless the
# control explicitly disables reasoning.
def _coerce_reasoning_control_object(value: Any, effort: Any = None) -> dict[str, Any]:
    if isinstance(value, dict):
        normalized = dict(value)
        if effort is not None and normalized.get("enabled") is not False:
            normalized.setdefault("effort", effort)
        return normalized

    if isinstance(value, bool):
        if value and effort is not None:
            return {"effort": effort}
        return {"enabled": value}

    if value is None:
        return {"effort": effort} if effort is not None else {}

    normalized_value = str(value).strip().lower()
    if normalized_value in {"true", "1", "yes", "on", "enabled"}:
        return {"effort": effort} if effort is not None else {"enabled": True}
    if normalized_value in {"false", "0", "no", "off", "disabled"}:
        return {"enabled": False}

    return {"effort": value}


# Rewrite the reasoning controls of the request options in place for endpoints that expect the
# object shape: flat effort options and any "reasoning" option are removed from both dicts and
# replaced by one "reasoning" object in extra_body. Other endpoints are left untouched.
def _normalize_reasoning_request_options(
    direct_options: dict[str, Any],
    extra_body: dict[str, Any],
    *,
    base_url: Any = None,
) -> None:
    if not _uses_object_reasoning_controls(base_url):
        return

    # Take the first effort value found and remove every flat effort option.
    effort_value = None
    for options in (direct_options, extra_body):
        for key in list(options.keys()):
            if _normalize_key_name(key) not in REASONING_EFFORT_KEYS:
                continue
            if effort_value is None:
                effort_value = options.get(key)
            options.pop(key, None)

    # Take the first existing "reasoning" option and remove every copy of it.
    reasoning_value = None
    has_reasoning_value = False
    for options in (direct_options, extra_body):
        if "reasoning" not in options:
            continue
        if not has_reasoning_value:
            reasoning_value = options.get("reasoning")
            has_reasoning_value = True
        options.pop("reasoning", None)

    # Build the single reasoning object and send it through extra_body when it is not empty.
    reasoning_object = _coerce_reasoning_control_object(
        reasoning_value if has_reasoning_value else None,
        effort=effort_value,
    )
    cleaned_reasoning = _clean_nested_config(reasoning_object)
    if isinstance(cleaned_reasoning, dict) and cleaned_reasoning:
        extra_body["reasoning"] = cleaned_reasoning


# Companion metadata

# Return the candidate roots of the companion catalog API: the configured OpenAI base URL and, when
# it ends with a version segment such as /v1, the same URL without that segment.
def _iter_companion_metadata_roots() -> list[str]:
    configured_base_url = str(settings.get_engine_url("openai") or "").strip().rstrip("/")
    if not configured_base_url:
        return []

    roots = [configured_base_url]
    try:
        parsed = urllib.parse.urlsplit(configured_base_url)
    except ValueError:
        return roots

    path = parsed.path.rstrip("/")
    if re.search(r"/v\d+$", path, flags=re.IGNORECASE):
        trimmed_path = re.sub(r"/v\d+$", "", path, flags=re.IGNORECASE)
        companion_root = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, trimmed_path or "/", "", "")).rstrip("/")
        if companion_root and companion_root not in roots:
            roots.append(companion_root)

    return roots


# Fetch one URL and return its JSON object, or None on a network error, a bad response or a
# non-object payload. Loopback URLs bypass the configured proxy.
def _fetch_json_url(url: str) -> dict[str, Any] | None:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urlopen_with_loopback_bypass(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (OSError, ValueError, urllib.error.URLError, urllib.error.HTTPError):
        return None

    return payload if isinstance(payload, dict) else None


# Convert one companion catalog payload ("models" or "data" list) into a list of plain model dicts.
def _extract_model_catalog_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    raw_items: Any = payload.get("models", payload.get("data", []))
    if not isinstance(raw_items, list):
        return []

    normalized_items: list[dict[str, Any]] = []
    for raw_item in raw_items:
        item = _to_plain_data(raw_item)
        if isinstance(item, dict):
            normalized_items.append(item)
    return normalized_items


# Load the model catalog of one companion endpoint, serving it from the process-wide cache when
# possible. A failed fetch gives an empty list and is not cached.
def _load_companion_model_catalog(url: str) -> list[dict[str, Any]]:
    with _metadata_cache_lock:
        if url in _supplemental_models_cache:
            return list(_supplemental_models_cache[url])

    payload = _fetch_json_url(url)
    if payload is None:
        return []

    models = _extract_model_catalog_items(payload)

    with _metadata_cache_lock:
        _supplemental_models_cache[url] = list(models)
    return models


# Return the record of a model from the companion catalogs (supplemental, read-only metadata), or
# an empty dict when no catalog lists it.
def _get_companion_model_payload(model_name: str) -> dict[str, Any]:
    for root in _iter_companion_metadata_roots():
        for relative_path in ("/api/v1/models", "/api/v0/models"):
            models = _load_companion_model_catalog(f"{root}{relative_path}")
            for payload in models:
                candidate_name = payload.get("id", payload.get("key", payload.get("model", payload.get("name"))))
                if _model_identifiers_match(model_name, candidate_name):
                    return payload

    return {}


# Return the richest metadata available for a model by merging three sources: the model list of the
# endpoint, the companion catalogs and the direct model retrieval. When retrieval fails and the model
# is not listed either, the retrieval error is raised.
def _get_model_payload(client: Any, model_name: str) -> dict[str, Any]:
    # Look the model up in the endpoint's model list.
    listed_match: dict[str, Any] = {}
    try:
        # Listing is usually cheaper and often returns extra catalog metadata
        # even when direct retrieval is partially implemented.
        listed_models = list(getattr(client.models.list(), "data", []) or [])
    except Exception:
        # Not every endpoint supports listing; retrieval below is the fallback.
        listed_models = []

    for listed_model in listed_models:
        payload = _to_plain_data(listed_model)
        if not isinstance(payload, dict):
            continue

        candidate_name = payload.get("id", payload.get("model", payload.get("name")))
        if _model_identifiers_match(model_name, candidate_name):
            listed_match = payload
            break

    # Retrieve the model directly and keep the error for the decision below.
    retrieved_payload: dict[str, Any] = {}
    retrieve_error: Exception | None = None
    try:
        # Retrieval is authoritative when it works, so merge it last.
        raw_model = client.models.retrieve(model_name)
        payload = _to_plain_data(raw_model)
        if isinstance(payload, dict):
            retrieved_payload = payload
    except Exception as exc:
        retrieve_error = exc

    if retrieve_error is not None and not listed_match:
        logger.error("[OpenAI API] Error fetching settings for %s: %s", model_name, retrieve_error)
        raise retrieve_error

    # Merge from lowest to highest confidence so provider-specific catalog
    # hints can fill gaps without overriding explicit model data.
    companion_payload = _get_companion_model_payload(model_name)
    merged_payload = _merge_nested_dicts(listed_match, companion_payload)
    merged_payload = _merge_nested_dicts(merged_payload, retrieved_payload)
    if not merged_payload:
        merged_payload = {"id": model_name, "model": model_name}
    return merged_payload


# Requests, responses and text extraction

# Create an OpenAI client for the configured base URL. Raise ImportError when the openai package is
# missing; a local endpoint without a key gets a placeholder key because the SDK requires one.
def _get_client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("The 'openai' package is required for OpenAI-compatible support.") from exc

    return OpenAI(
        base_url=settings.get_engine_url("openai"),
        api_key=settings.get_openai_api_key() or "not-needed",
    )


# Remove service control tokens that should never reach the chat UI from generated text.
def _sanitize_generated_text(text: str) -> str:
    cleaned = str(text or "")
    for pattern in CONTROL_TOKEN_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


# Extract visible text (reasoning=False) or reasoning text (reasoning=True) from one payload
# fragment, which can be a string, a list of parts or a typed dict. Control tokens are removed.
def _extract_text_fragment(value: Any, *, reasoning: bool) -> str:
    # Strings are the base case; other values are converted to plain data first.
    if value is None:
        return ""

    if isinstance(value, str):
        return _sanitize_generated_text(value)

    plain_value = _to_plain_data(value)
    if isinstance(plain_value, str):
        return _sanitize_generated_text(plain_value)

    if isinstance(plain_value, list):
        return "".join(_extract_text_fragment(item, reasoning=reasoning) for item in plain_value)

    if not isinstance(plain_value, dict):
        return ""

    # Typed item: decide from its type whether it is reasoning or visible content.
    item_type = _normalize_key_name(
        plain_value.get("type", plain_value.get("content_type", plain_value.get("kind", "")))
    )
    is_reasoning_item = "reasoning" in item_type or "thinking" in item_type

    if reasoning and not is_reasoning_item:
        # Reasoning text is often nested under dedicated fields even when the
        # outer item looks like a generic content wrapper.
        for key in (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "reasoning_text",
            "thinking",
            "thinking_text",
        ):
            if key not in plain_value:
                continue
            extracted = _extract_text_fragment(plain_value[key], reasoning=True)
            if extracted:
                return extracted
        return ""

    # Visible text never comes from a reasoning item.
    if not reasoning and is_reasoning_item:
        return ""

    # Look for text under the preferred keys; reasoning requests also accept the dedicated reasoning keys.
    preferred_keys = (
        (
            "reasoning",
            "reasoning_content",
            "reasoning_details",
            "reasoning_text",
            "thinking",
            "thinking_text",
            "text",
            "content",
            "value",
        )
        if reasoning
        else ("text", "content", "value")
    )

    for key in preferred_keys:
        if key not in plain_value:
            continue
        if reasoning and key in {"text", "content", "value"} and item_type and not is_reasoning_item:
            continue
        extracted = _extract_text_fragment(plain_value[key], reasoning=reasoning)
        if extracted:
            return extracted

    # Finally descend into wrapper keys that nest the real content.
    for key in ("parts", "message", "delta"):
        if key not in plain_value:
            continue
        extracted = _extract_text_fragment(plain_value[key], reasoning=reasoning)
        if extracted:
            return extracted

    return ""


# Return the visible content text of one choice delta or message payload, taken from the first of
# "content", "text" or "output_text" that yields text.
def _extract_content_text_from_payload(payload: dict[str, Any]) -> str:
    for key in ("content", "text", "output_text"):
        if key not in payload:
            continue
        extracted = _extract_text_fragment(payload.get(key), reasoning=False)
        if extracted:
            return extracted
    return ""


# Return the reasoning text of one choice delta or message payload. Backends use different keys for
# it, and some put reasoning items inside a structured "content" value.
def _extract_reasoning_text_from_payload(payload: dict[str, Any]) -> str:
    for key in (
        "reasoning",
        "reasoning_content",
        "reasoning_details",
        "reasoning_text",
        "thinking",
        "thinking_text",
    ):
        if key not in payload:
            continue
        extracted = _extract_text_fragment(payload.get(key), reasoning=True)
        if extracted:
            return extracted

    raw_content = payload.get("content")
    if isinstance(raw_content, (dict, list)):
        return _extract_text_fragment(raw_content, reasoning=True)
    return ""


# Return the reasoning-fragment type ("reasoning", "reasoning_start_tag" or "reasoning_end_tag")
# when the backend marks its fragments with one, otherwise None.
def _extract_reasoning_type(payload: dict[str, Any]) -> str | None:
    for key in ("reasoning_type", "reasoningType", "type"):
        raw_value = payload.get(key)
        normalized = _normalize_key_name(raw_value)
        if not normalized:
            continue
        if normalized in {"reasoning", "reasoning_end_tag", "reasoning_start_tag"}:
            return str(raw_value)
    return None


# Return tool-call arguments as a dict. A JSON string is parsed; text that is not valid JSON, or JSON
# that is not an object, is wrapped as {"value": ...}.
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


# Merge one streamed tool-call fragment into the accumulated calls, which are keyed by the fragment
# index. The function name and argument text arrive in pieces and are concatenated; id and type are
# overwritten when present.
def _merge_tool_call_delta(tool_calls_by_index: dict[int, dict[str, Any]], raw_tool_call: dict[str, Any]) -> None:
    # A missing or invalid index starts a new call after the existing ones.
    try:
        index = int(raw_tool_call.get("index", len(tool_calls_by_index)))
    except (TypeError, ValueError):
        index = len(tool_calls_by_index)

    entry = tool_calls_by_index.setdefault(
        index,
        {
            "id": "",
            "type": "function",
            "function": {"name": "", "arguments": ""},
        },
    )

    # Update the identity of the call.
    tool_id = raw_tool_call.get("id")
    if tool_id:
        entry["id"] = str(tool_id)

    tool_type = raw_tool_call.get("type")
    if tool_type:
        entry["type"] = str(tool_type)

    # Append the name and argument pieces; dict arguments are serialized to JSON text first.
    function_payload = raw_tool_call.get("function")
    if not isinstance(function_payload, dict):
        function_payload = {}

    function_name = function_payload.get("name")
    if function_name:
        entry["function"]["name"] += str(function_name)

    raw_arguments = function_payload.get("arguments")
    if isinstance(raw_arguments, dict):
        raw_arguments = json.dumps(raw_arguments, ensure_ascii=False)
    if raw_arguments:
        entry["function"]["arguments"] += str(raw_arguments)


# Convert ASLM chat messages into OpenAI-compatible message payloads. Tool results become tool-role
# messages, messages with images or media become a list of content parts, and the tool calls and
# stored thinking of assistant messages are carried along.
def _build_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []

    for message in messages:
        role = str(message.get("role", "user")).lower()
        content = message.get("content", "") or ""
        images = message.get("images") or []
        image_mime_types = message.get("image_mime_types") or []
        media = message.get("media") if isinstance(message.get("media"), list) else []

        # A tool result keeps only the call id and its content.
        if role == "tool":
            payload.append(
                {
                    "role": "tool",
                    "tool_call_id": message.get("tool_call_id", message.get("name", "call")),
                    "content": content,
                }
            )
            continue

        # Attachments turn the content into multi-part form: the text, then images as data URLs
        # (JPEG unless a MIME type is given), then audio and other files.
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

            # MP3 and WAV audio goes as input_audio; every other media item is sent as a file data URL.
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

        # Keep the tool calls the assistant made earlier so the tool results that follow still match them.
        if role == "assistant" and message.get("tool_calls"):
            entry["tool_calls"] = message["tool_calls"]

        if role == "assistant" and message.get("thinking"):
            # Forward stored thinking from previous turns so reasoning models
            # retain their own chain-of-thought across multi-turn conversations.
            entry["thinking"] = message["thinking"]

        payload.append(entry)

    return payload


# Split generic generation options into direct client keyword arguments and an "extra_body" dict for
# fields that only compatible backends understand. The thinking switch and level from kwargs
# (think, think_level and their parameter names) are added as options unless already set.
def _build_openai_request_options(
    options: dict[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    direct_options: dict[str, Any] = {}
    extra_body: dict[str, Any] = {}

    # Keep known OpenAI kwargs at the top level and forward everything else
    # through ``extra_body`` for compatible backends with custom fields.
    for raw_key, raw_value in (options or {}).items():
        normalized_key = DIRECT_OPTION_ALIASES.get(raw_key, raw_key)
        if normalized_key in DIRECT_OPTION_KEYS:
            direct_options[normalized_key] = raw_value
        else:
            extra_body[raw_key] = raw_value

    # top_logprobs is only valid together with logprobs, so drop it otherwise.
    if not _bool_from_value(direct_options.get("logprobs")):
        direct_options.pop("top_logprobs", None)

    # Add the thinking switch and level under the parameter names that the model uses.
    think = kwargs.get("think")
    think_level = kwargs.get("think_level")
    think_param_name = str(kwargs.get("think_param_name", "think") or "think")
    think_level_param_name = str(kwargs.get("think_level_param_name", "reasoning_effort") or "reasoning_effort")

    if think is not None and think_param_name:
        normalized_think_key = DIRECT_OPTION_ALIASES.get(think_param_name, think_param_name)
        target_options = direct_options if normalized_think_key in DIRECT_OPTION_KEYS else extra_body
        target_options.setdefault(normalized_think_key, think)

    if think_level is not None and think_level_param_name:
        normalized_level_key = DIRECT_OPTION_ALIASES.get(think_level_param_name, think_level_param_name)
        target_options = direct_options if normalized_level_key in DIRECT_OPTION_KEYS else extra_body
        target_options.setdefault(normalized_level_key, think_level)

    # Convert the reasoning options for endpoints that need the object shape.
    _normalize_reasoning_request_options(direct_options, extra_body)

    # Merge into an extra_body the caller may already have passed as an option.
    if extra_body:
        merged_extra_body = dict(direct_options.get("extra_body", {}) or {})
        merged_extra_body.update(extra_body)
        direct_options["extra_body"] = merged_extra_body

    return direct_options


# Tool feedback payloads

# Serialize one tool invocation into the event dict that the UI renders while the response streams.
def _build_tool_event(tool_lookup: dict[str, dict[str, Any]], tool_call: dict[str, Any]) -> dict[str, Any]:
    return tool_registry.build_tool_event(tool_lookup, tool_call)


# Build the tool-role message for one tool result: the text for the model plus the extra fields
# split off the result and, when an event is given, the display details of the call.
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

    # Copy the display details of the call, falling back to the tool name for missing ones.
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


# Reasoning stream parsing

# Incremental parser that splits streamed text into reasoning and visible content. It follows inline
# reasoning tags (such as <think>) across chunk boundaries and also honors the reasoning type that
# some backends attach to each fragment.
class _ReasoningTextParser:
    # Store the tag pairs and start in visible mode with nothing buffered.
    def __init__(self, tag_pairs: tuple[tuple[str, str], ...] = REASONING_TAG_PAIRS) -> None:
        # Opening and closing tags to look for.
        self._start_tags = [start for start, _end in tag_pairs]
        self._end_tags = [end for _start, end in tag_pairs]

        # Parsing state: whether the text is currently inside a reasoning block, the tail of the last
        # chunk that may be the beginning of a tag, and the last visible character seen so far.
        self._in_reasoning = False
        self._pending = ""
        self._visible_boundary_char: str | None = None

    # Return the position and text of the earliest of the given tags in the source, or None. Outside
    # reasoning, an opening tag only counts at a reasoning start boundary.
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

    # Return whether an opening tag at the given index may start reasoning. That is the case at the
    # start of the text or after a clear separator (a line break, ">" or punctuation); spaces and tabs
    # before the tag are skipped. This keeps a tag mentioned in the middle of a sentence visible.
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

    # Split the source into the text that is safe to emit and a trailing part that could be the
    # beginning of one of the tags. The trailing part is held back until the next chunk arrives.
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

    # Parse one streamed fragment and return its (thinking, visible_content) parts. A reasoning type
    # reported by the backend takes precedence: start and end tag markers switch the mode, and a plain
    # reasoning fragment is thinking as a whole. Otherwise inline tags in the text switch the mode.
    def feed(self, content: str, reasoning_type: str | None = None) -> tuple[str, str]:
        thinking_parts: list[str] = []
        content_parts: list[str] = []

        # Handle fragments that the backend has already classified.
        normalized_reasoning_type = _normalize_key_name(reasoning_type)
        if normalized_reasoning_type == "reasoning_start_tag":
            self._in_reasoning = True
            return "", ""
        if normalized_reasoning_type == "reasoning_end_tag":
            self._in_reasoning = False
            return "", ""
        if normalized_reasoning_type == "reasoning":
            return str(content or ""), ""

        # Scan the text, including the held-back tail of the previous chunk, tag by tag. The text before
        # each tag goes to thinking or visible content depending on the mode, and the tag flips the mode.
        source = f"{self._pending}{content or ''}"
        self._pending = ""

        while source:
            tags = self._end_tags if self._in_reasoning else self._start_tags
            next_tag = self._find_next_tag(source, tags)
            if next_tag is None:
                # No complete tag left: emit the text and keep a possible partial tag for the next chunk.
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

    # Return the held-back partial fragment as a (thinking, visible_content) pair at the end of the
    # stream, so that a trailing text which only looked like a tag start is not lost.
    def flush(self) -> tuple[str, str]:
        pending = self._pending
        self._pending = ""
        if not pending:
            return "", ""
        if self._in_reasoning:
            return pending, ""
        self._remember_visible_text(pending)
        return "", pending


# Streaming and the tool loop

# Run the reasoning and the content text of one choice payload through the parser and return the
# combined (thinking, visible_content) pair.
def _parse_payload_fragments(
    parser: _ReasoningTextParser,
    payload: dict[str, Any],
) -> tuple[str, str]:
    thinking_parts: list[str] = []
    content_parts: list[str] = []

    # Dedicated reasoning fields are thinking as a whole.
    reasoning_fragment = _extract_reasoning_text_from_payload(payload)
    if reasoning_fragment:
        thinking_part, content_part = parser.feed(reasoning_fragment, reasoning_type="reasoning")
        if thinking_part:
            thinking_parts.append(thinking_part)
        if content_part:
            content_parts.append(content_part)

    # The content text may still carry inline reasoning tags or a backend-reported reasoning type.
    content_fragment = _extract_content_text_from_payload(payload)
    if content_fragment:
        thinking_part, content_part = parser.feed(
            content_fragment,
            reasoning_type=_extract_reasoning_type(payload),
        )
        if thinking_part:
            thinking_parts.append(thinking_part)
        if content_part:
            content_parts.append(content_part)

    return "".join(thinking_parts), "".join(content_parts)


# Run one model request and yield {"message": ...} chunks as text arrives (streamed or not). The
# generator returns the assembled assistant message, with its thinking and tool calls, as its
# StopIteration value. A tool choice is sent only when tools are given: a forced tool when it is
# among them, "required" when a call is needed, otherwise "auto".
def _stream_openai_round(
    client: Any,
    model_name: str,
    conversation: list[dict[str, Any]],
    options: dict[str, Any],
    *,
    tools: list[dict[str, Any]] | None = None,
    forced_tool_name: str = "",
    require_tool_call: bool = False,
    stream: bool = True,
):
    # Build the request arguments, including the tool definitions and the tool choice.
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

    # Accumulators for the assistant message, filled while the response is consumed.
    assistant_content = ""
    assistant_thinking = ""
    tool_calls_by_index: dict[int, dict[str, Any]] = {}
    parser = _ReasoningTextParser()
    yielded_chunk = False

    # Send the request; a backend that rejects a forced tool choice gets one retry with "auto".
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=conversation,
            stream=stream,
            **request_kwargs,
        )
    except Exception as exc:
        # Not every OpenAI-compatible backend accepts forced tool_choice payloads;
        # degrade to auto so the request still runs and the loop supervisor can nudge.
        if request_kwargs.get("tool_choice") in (None, "auto"):
            raise
        logger.warning("[OpenAI API] Forced tool_choice rejected, retrying with auto: %s", exc)
        request_kwargs["tool_choice"] = "auto"
        response = client.chat.completions.create(
            model=model_name,
            messages=conversation,
            stream=stream,
            **request_kwargs,
        )

    # Consume the response. Both paths yield the text parts as they are parsed and collect tool calls.
    if stream:
        # Streaming deltas arrive fragment-by-fragment, so accumulate both the
        # visible text and partially-built tool calls.
        for raw_chunk in response:
            if _abort_event.is_set():
                break

            chunk_payload = _to_plain_data(raw_chunk)
            choices = chunk_payload.get("choices", []) if isinstance(chunk_payload, dict) else []
            for choice in choices if isinstance(choices, list) else []:
                if not isinstance(choice, dict):
                    continue

                delta_payload = _to_plain_data(choice.get("delta", {}))
                if not isinstance(delta_payload, dict):
                    delta_payload = {}

                thinking_part, content_part = _parse_payload_fragments(parser, delta_payload)
                if thinking_part or content_part:
                    yielded_chunk = True
                    assistant_content += content_part
                    assistant_thinking += thinking_part
                    payload = {"role": "assistant", "content": content_part}
                    if thinking_part:
                        payload["thinking"] = thinking_part
                    yield {"message": payload}

                raw_tool_calls = delta_payload.get("tool_calls", delta_payload.get("toolCalls", []))
                if isinstance(raw_tool_calls, list):
                    for raw_tool_call in raw_tool_calls:
                        normalized_tool_call = _to_plain_data(raw_tool_call)
                        if isinstance(normalized_tool_call, dict):
                            _merge_tool_call_delta(tool_calls_by_index, normalized_tool_call)
    else:
        # Non-streaming responses still reuse the same parsers so both code
        # paths produce the same assistant payload shape.
        response_payload = _to_plain_data(response)
        choices = response_payload.get("choices", []) if isinstance(response_payload, dict) else []
        for choice in choices if isinstance(choices, list) else []:
            if not isinstance(choice, dict):
                continue

            message_payload = _to_plain_data(choice.get("message", {}))
            if not isinstance(message_payload, dict):
                message_payload = {}

            thinking_part, content_part = _parse_payload_fragments(parser, message_payload)
            if thinking_part or content_part:
                yielded_chunk = True
                assistant_content += content_part
                assistant_thinking += thinking_part
                payload = {"role": "assistant", "content": content_part}
                if thinking_part:
                    payload["thinking"] = thinking_part
                yield {"message": payload}

            raw_tool_calls = message_payload.get("tool_calls", message_payload.get("toolCalls", []))
            if isinstance(raw_tool_calls, list):
                for index, raw_tool_call in enumerate(raw_tool_calls):
                    normalized_tool_call = _to_plain_data(raw_tool_call)
                    if isinstance(normalized_tool_call, dict) and "index" not in normalized_tool_call:
                        normalized_tool_call["index"] = index
                    if isinstance(normalized_tool_call, dict):
                        _merge_tool_call_delta(tool_calls_by_index, normalized_tool_call)

    # Flush any trailing partial tag fragment that was held back between chunks.
    tail_thinking, tail_content = parser.flush()
    if tail_thinking or tail_content:
        yielded_chunk = True
        assistant_content += tail_content
        assistant_thinking += tail_thinking
        payload = {"role": "assistant", "content": tail_content}
        if tail_thinking:
            payload["thinking"] = tail_thinking
        yield {"message": payload}

    # Assemble the final assistant message with the tool calls in index order.
    assistant_message: dict[str, Any] = {"role": "assistant", "content": assistant_content}
    if assistant_thinking:
        assistant_message["thinking"] = assistant_thinking

    assembled_tool_calls = [tool_calls_by_index[index] for index in sorted(tool_calls_by_index)]
    if assembled_tool_calls:
        assistant_message["tool_calls"] = assembled_tool_calls

    # Emit the whole message once if no text chunk was sent along the way.
    if not yielded_chunk and (assistant_content or assistant_thinking):
        yield {"message": assistant_message}

    return assistant_message


# Re-yield every chunk of a round generator and return its final assistant message, so a caller can
# use "yield from" and still receive the return value. An empty assistant message is returned when
# the round has no return value.
def _yield_stream_round(round_stream):
    while True:
        try:
            yield next(round_stream)
        except StopIteration as stop:
            return stop.value or {"role": "assistant", "content": ""}


# Run a conversation with local tools through OpenAI-compatible function calling. Each round streams
# a model response; when it asks for tools they are executed and their results are appended to the
# conversation for the next round. The loop ends when the model answers without tool calls, after a
# forced final answer, on an abort, or when the round limit is reached. Besides message chunks it
# yields transcript_message, tool_events and tool_result items for the caller.
def _run_tool_loop(
    client: Any,
    model_name: str,
    messages: list[dict[str, Any]],
    options: dict[str, Any],
    tool_server_ids: list[str],
    tool_context: dict[str, Any],
    *,
    stream: bool = True,
    conversation: list[dict[str, Any]] | None = None,
):
    # Resolve the tools of the selected servers; without any, this is a plain single round.
    tools, tool_lookup = tool_registry.build_ollama_tools(
        tool_server_ids,
        engine="openai",
        model_name=model_name,
        tool_source_map=tool_context.get("tool_source_map") if isinstance(tool_context, dict) else None,
        allowed_tool_aliases=tool_context.get("allowed_tool_aliases") if isinstance(tool_context, dict) else None,
        instant_mode=bool(tool_context.get("instant_mode")) if isinstance(tool_context, dict) else False,
        instant_search_batch_size=tool_context.get("instant_search_batch_size") if isinstance(tool_context, dict) else None,
    )

    if not tools:
        request_messages = conversation if conversation is not None else _build_openai_messages(messages)
        yield from _yield_stream_round(
            _stream_openai_round(
                client,
                model_name,
                request_messages,
                options,
                stream=stream,
            )
        )
        return

    # Prepare the conversation and the loop state: tool limits and duplicate tracking, the blocked
    # result streak, and which user-required tools are still unmet together with how many reminders
    # the supervisor may still send.
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

    max_tool_rounds = tool_registry.tool_round_limit_from_context(tool_context, MAX_TOOL_ROUNDS)
    for round_index in range(max_tool_rounds):
        # Decide whether this round must call a specific tool or any tool: a forced tool first, then
        # any user-required tool server that has not been used yet.
        round_forced_tool_name = ""
        round_requires_tool_call = False
        if enforce_required_tools_this_round:
            forced_server_id = tool_registry.tool_server_id_for_alias(tool_lookup, forced_tool_name)
            if forced_tool_name and forced_server_id not in satisfied_tool_server_ids:
                round_forced_tool_name = forced_tool_name
            elif any(server_id not in satisfied_tool_server_ids for server_id in required_tool_server_ids):
                round_requires_tool_call = True
        enforce_required_tools_this_round = False

        # Each round lets the model either finish or request another batch of
        # local tool calls to continue the conversation.
        assistant_message = yield from _yield_stream_round(
            _stream_openai_round(
                client,
                model_name,
                conversation,
                options,
                tools=tools,
                forced_tool_name=round_forced_tool_name,
                require_tool_call=round_requires_tool_call,
                stream=stream,
            )
        )
        conversation.append(assistant_message)

        # No tool calls ends the loop, unless required tools were ignored and a reminder is still allowed.
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

        # Normalize tool-call payloads into one internal shape before dispatch.
        tool_calls: list[dict[str, Any]] = []
        for raw_tool_call in raw_tool_calls:
            if not isinstance(raw_tool_call, dict):
                continue
            function_payload = raw_tool_call.get("function", {})
            if not isinstance(function_payload, dict):
                function_payload = {}
            tool_calls.append(
                {
                    "id": str(raw_tool_call.get("id") or f"call_{round_index}_{len(tool_calls)}"),
                    "name": str(function_payload.get("name") or ""),
                    "arguments": _normalize_tool_call_arguments(function_payload.get("arguments")),
                }
            )

        # Prepare the calls (parallel searches may be coalesced) and apply the limit on parallel calls.
        tool_calls = tool_registry.prepare_tool_calls(tool_lookup, tool_calls)
        tool_calls = tool_registry.limit_tool_calls_from_context(tool_context, tool_calls, tool_lookup)

        # Remember which tool servers were used, for the required-tool checks.
        for tool_call in tool_calls:
            satisfied_server_id = tool_registry.tool_server_id_for_alias(tool_lookup, tool_call["name"])
            if satisfied_server_id:
                satisfied_tool_server_ids.add(satisfied_server_id)

        # Replace the assistant message with its canonical form, whose tool arguments match the
        # prepared calls, then announce the tool events to the UI with a unique alias per call.
        canonical_assistant_message = tool_registry.canonicalize_transcript_tool_calls(
            assistant_message, tool_calls
        )
        conversation[-1] = canonical_assistant_message
        yield {"transcript_message": canonical_assistant_message}

        tool_events = [_build_tool_event(tool_lookup, tool_call) for tool_call in tool_calls]
        for _i, _ev in enumerate(tool_events):
            _ev["alias"] = f"{_ev['alias']}__{_i}"
        yield {"tool_events": tool_events}

        # Execute the calls one by one and feed each result back into the conversation.
        cancelled_tool_result_seen = False
        for tool_call_index, (tool_call, tool_event) in enumerate(zip(tool_calls, tool_events), start=1):
            # Build the per-call context and log the request.
            call_context = dict(tool_context or {})
            call_context.update(
                {
                    "engine": "openai",
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

            # Get the result: a preflight error, cooldown, duplicate call or quota error replaces the
            # execution, checked in that order; otherwise the tool runs and its cooldown is recorded.
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

            # Track cancelled and consecutive blocked results; they trigger a forced final answer below.
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

            # Feed the tool result back as a standard tool-role message so the
            # next model round can continue from the resolved state.
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

        # After an abort, stop at once. After a cancelled tool or two blocked results in a row, ask the
        # model for a final answer without tools instead of continuing the loop.
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
                _stream_openai_round(
                    client,
                    model_name,
                    conversation,
                    options,
                    stream=stream,
                )
            )
            yield {"transcript_message": assistant_message}
            return

        # Compact the completed tool history before the next round when a compactor is configured and
        # the conversation is large.
        conversation = tool_registry.maybe_compact_tool_conversation(
            tool_context, conversation, provider_format="standard"
        )

    # The round limit was reached without a final answer.
    yield {"message": {"content": "[Error during generation: tool loop exceeded the safety limit.]"}}


# Public adapter API

# Return whether the conversation already contains tool state: a tool-role message or an assistant
# message with tool calls.
def _conversation_uses_tools(messages: list[dict[str, Any]]) -> bool:
    for message in messages:
        if str(message.get("role", "")).lower() == "tool":
            return True
        if message.get("tool_calls"):
            return True
    return False


# Return the models exposed by the configured OpenAI-compatible endpoint as plain dicts. An error
# is logged and gives an empty list.
def get_models() -> list[Any]:
    client = _get_client()
    try:
        response = client.models.list()
        return [_to_plain_data(item) for item in list(getattr(response, "data", []) or [])]
    except Exception as exc:
        logger.error("[OpenAI API] Error listing models: %s", exc)
        return []
    finally:
        _close_client(client)


# Raise NotImplementedError unconditionally: OpenAI-compatible endpoints expose remote models only, so
# there is nothing to download locally.
def download_model(model_name: str, **kwargs: Any) -> Any:
    raise NotImplementedError("OpenAI-compatible models are remote and cannot be downloaded locally.")


# Build the capability metadata of one model from its endpoint payload: defaults, supported
# parameters, thinking toggle and level controls, input modalities, tool calling, context length and
# the raw payload. Providers describe models unevenly, so every capability is inferred from several hints.
def get_model_settings(model_name: str) -> dict[str, Any]:
    client = _get_client()
    try:
        # Build defaults and supported parameter names first, then layer the
        # higher-level capabilities inferred from those raw hints.
        raw_model = _get_model_payload(client, model_name)
        defaults = _extract_defaults(raw_model)
        supported_parameter_names = _extract_supported_parameter_names(raw_model, defaults)
        capability_tokens = _extract_capability_tokens(raw_model)

        reasoning_toggle_default, reasoning_level_default, reasoning_level_options = _extract_reasoning_metadata(raw_model)
        parameter_names = set(supported_parameter_names) | {str(key) for key in defaults.keys()}

        # Find the provider's own names for the thinking toggle and level, and record the discovered
        # reasoning defaults under them.
        think_param_name = _resolve_parameter_name(parameter_names, THINK_PARAM_NAMES, "think")
        think_level_param_name = _resolve_parameter_name(parameter_names, THINK_LEVEL_PARAM_NAMES, "reasoning_effort")

        if reasoning_toggle_default is not None and think_param_name not in defaults:
            defaults[think_param_name] = reasoning_toggle_default
        if reasoning_level_default is not None and think_level_param_name not in defaults:
            defaults[think_level_param_name] = reasoning_level_default

        # Without levels from the reasoning metadata, read them from the options of the level parameter.
        if not reasoning_level_options:
            reasoning_level_options = _extract_parameter_option_values(
                raw_model,
                {think_level_param_name, *THINK_LEVEL_PARAM_NAMES},
            )

        # Read the explicit feature flags (None when the payload does not state them).
        explicit_tool_support = _extract_feature_flag(raw_model, TOOL_CAPABILITY_NAMES)
        explicit_vision_support = _extract_feature_flag(raw_model, VISION_CAPABILITY_NAMES)
        explicit_audio_input_support = _extract_feature_flag(raw_model, AUDIO_INPUT_CAPABILITY_NAMES)
        explicit_video_input_support = _extract_feature_flag(raw_model, VIDEO_INPUT_CAPABILITY_NAMES)
        explicit_reasoning_support = _extract_feature_flag(raw_model, REASONING_CAPABILITY_NAMES)

        # Capability inference combines explicit booleans, capability tokens,
        # and parameter availability because providers expose metadata unevenly.
        supports_tool_calling = (
            bool(explicit_tool_support)
            or bool(capability_tokens & TOOL_CAPABILITY_NAMES)
            or any(
                _normalize_key_name(parameter_name) in {"parallel_tool_calls", "tool_choice", "tools"}
                for parameter_name in supported_parameter_names
            )
        )
        supports_vision = (
            bool(explicit_vision_support)
            or bool(capability_tokens & VISION_CAPABILITY_NAMES)
        )
        supports_audio_input = (
            bool(explicit_audio_input_support)
            or bool(capability_tokens & AUDIO_INPUT_CAPABILITY_NAMES)
        )
        supports_video_input = (
            bool(explicit_video_input_support)
            or bool(capability_tokens & VIDEO_INPUT_CAPABILITY_NAMES)
        )

        # A tool-capable model supports the tool request parameters. Thinking is supported when the
        # model has a toggle or level parameter, reasoning metadata, or an explicit reasoning flag.
        if supports_tool_calling:
            supported_parameter_names.update({"tools", "tool_choice"})
        supports_think_toggle = (
            think_param_name in defaults
            or any(_normalize_key_name(parameter_name) == _normalize_key_name(think_param_name) for parameter_name in supported_parameter_names)
            or isinstance(reasoning_toggle_default, bool)
        )
        supports_think_level = (
            think_level_param_name in defaults
            or any(_normalize_key_name(parameter_name) == _normalize_key_name(think_level_param_name) for parameter_name in supported_parameter_names)
            or bool(reasoning_level_options)
            or (
                reasoning_level_default is not None
                and str(reasoning_level_default or "").strip().lower() not in {"on", "off", "true", "false"}
            )
        )
        supports_thinking = (
            bool(explicit_reasoning_support)
            or bool(capability_tokens & REASONING_CAPABILITY_NAMES)
            or supports_think_toggle
            or supports_think_level
        )

        # List the thinking parameters as supported and make sure the level default is recorded.
        if supports_think_toggle:
            supported_parameter_names.add(think_param_name)
        if supports_think_level:
            supported_parameter_names.add(think_level_param_name)
            if think_level_param_name not in defaults and reasoning_level_default is not None:
                defaults[think_level_param_name] = reasoning_level_default

        # Keep the exported capability list small and UI-oriented.
        capabilities: list[str] = []
        if supports_vision:
            capabilities.append("vision")
        if supports_audio_input:
            capabilities.append("audio_input")
        if supports_video_input:
            capabilities.append("video_input")
        if supports_tool_calling:
            capabilities.append("tools")
        if supports_thinking:
            capabilities.append("thinking")
        capabilities.append("files")

        # Assemble the settings payload for the UI.
        return {
            "model": model_name,
            "context_length": _extract_context_length(raw_model),
            "defaults": defaults,
            "supported_parameters": sorted(supported_parameter_names, key=str.casefold),
            "supports_thinking": supports_thinking,
            "supports_think_toggle": supports_think_toggle,
            "supports_think_level": supports_think_level,
            "think_param_name": think_param_name,
            "think_level_param_name": think_level_param_name,
            "think_level_options": reasoning_level_options,
            "supports_vision": supports_vision,
            "supports_audio_input": supports_audio_input,
            "supports_video_input": supports_video_input,
            "supports_tool_calling": supports_tool_calling,
            # ASLM-Chat file attachments are serialized into universal text context.
            "supports_files": True,
            "capabilities": capabilities,
            "runtime_limits": {},
            "custom_fields": [],
            "raw": raw_model,
        }
    finally:
        _close_client(client)


# Generate a chat response through the OpenAI-compatible API and yield its chunks. Recognized kwargs
# are stream, options, think, think_level, their parameter names, tool_context and the tool server
# ids (tool_server_ids, or the older tool_server_id and tool_id). With tool servers the response comes
# from the tool loop; otherwise it is a single round. The client is always closed at the end.
def generate(model_name: str, messages: list[dict[str, Any]], **kwargs: Any):
    # Accept historical single-id inputs and normalize them into one list.
    raw_ids = kwargs.pop("tool_server_ids", None) or kwargs.pop("tool_server_id", None) or kwargs.pop("tool_id", None)
    if isinstance(raw_ids, str):
        tool_server_ids = [raw_ids] if raw_ids.strip() else []
    elif isinstance(raw_ids, list):
        tool_server_ids = [str(server_id) for server_id in raw_ids if str(server_id).strip()]
    else:
        tool_server_ids = []

    # Read the request settings and convert the generic options into client arguments.
    tool_context = dict(kwargs.pop("tool_context", {}) or {})
    stream = bool(kwargs.get("stream", False))
    request_options = _build_openai_request_options(
        dict(kwargs.get("options", {}) or {}),
        think=kwargs.get("think"),
        think_level=kwargs.get("think_level"),
        think_param_name=kwargs.get("think_param_name", "think"),
        think_level_param_name=kwargs.get("think_level_param_name", "reasoning_effort"),
    )
    client = _get_client()
    try:
        # A new generation starts without a pending abort request.
        _abort_event.clear()

        # Tool-enabled conversations stay in the tool loop so the adapter can
        # execute local tools between model rounds.
        if tool_server_ids:
            yield from _run_tool_loop(
                client,
                model_name,
                messages,
                request_options,
                tool_server_ids,
                tool_context,
                stream=stream,
            )
            return

        conversation = _build_openai_messages(messages)

        # Conversations that already contain tool state must stay on the same
        # OpenAI-compatible message format even without new tool servers.
        if _conversation_uses_tools(messages):
            yield from _yield_stream_round(
                _stream_openai_round(
                    client,
                    model_name,
                    conversation,
                    request_options,
                    stream=stream,
                )
            )
            return

        yield from _yield_stream_round(
            _stream_openai_round(
                client,
                model_name,
                conversation,
                request_options,
                stream=stream,
            )
        )
    except Exception as exc:
        logger.error("[OpenAI API] Error generating response from %s: %s", model_name, exc)
        raise
    finally:
        _close_client(client)
