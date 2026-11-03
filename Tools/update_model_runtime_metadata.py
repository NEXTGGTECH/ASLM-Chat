# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request


# Tools folder and module root. The module root goes first on sys.path so the Settings package
# is importable when this file runs as a standalone script.
TOOLS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TOOLS_DIR.parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from Settings.proxy_policy import apply_loopback_proxy_bypass, urlopen_direct

# Inputs (module settings and manifest) and the generated metadata file this script writes.
SETTINGS_PATH = MODULE_DIR / "Settings" / "settings.json"
MODULE_MANIFEST_PATH = MODULE_DIR / "ASLM_Module.json"
METADATA_PATH = TOOLS_DIR / "model_runtime_metadata.json"

# Route of the module's own UI server that reports the active engine, model, capabilities and limits.
ASLM_INFERENCE_INFO_ROUTE = "/api/inference_info/"


# Read a JSON object from a file; a missing, unreadable or non-object file yields an empty dict.
def _read_json_file(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


# Return a setting's current value (or its default) from the module manifest settings list, or None.
def _manifest_setting(manifest: dict[str, Any], key: str) -> Any:
    settings_list = manifest.get("settings")
    if not isinstance(settings_list, list):
        return None
    for item in settings_list:
        if not isinstance(item, dict) or item.get("key") != key:
            continue
        if "value" in item:
            return item.get("value")
        return item.get("default")
    return None


# Resolve one setting by priority: the environment variable (when env_key is given), then the
# module manifest, then settings.json, then the default. Empty values are skipped.
def _runtime_setting(
    settings: dict[str, Any],
    manifest: dict[str, Any],
    key: str,
    default: Any = None,
    *,
    env_key: str | None = None,
) -> Any:
    if env_key:
        env_value = os.environ.get(env_key)
        if env_value not in (None, ""):
            return env_value
    manifest_value = _manifest_setting(manifest, key)
    if manifest_value not in (None, ""):
        return manifest_value
    return settings.get(key, default)


# Convert a value to a valid TCP port number, or None when it is not one.
def _coerce_port(value: Any) -> int | None:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return None
    return port if 0 < port < 65536 else None


# Build the loopback URL of a route on the given port; an unknown port yields an empty string.
def _local_route_url(port: int | None, route: str) -> str:
    if port is None:
        return ""
    return f"http://127.0.0.1:{port}{route}"


# Build the metadata entry that tells readers where a piece of information comes from: a local
# HTTP route and the setting that holds its port.
def _source_descriptor(name: str, route: str, port_key: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "local_http_route",
        "route": route,
        "host": "127.0.0.1",
        "port_setting": port_key,
        "port_sources": [
            "ASLM_Module.json settings[].value/default",
            "Settings/settings.json",
            "environment override when available",
        ],
    }


# Request a JSON object from a local URL without any proxy, using a short timeout; any network or
# parse failure yields an empty dict.
def _fetch_json(url: str, *, method: str = "GET", body: dict[str, Any] | None = None) -> dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = Request(url, data=data, method=method, headers=headers)
    try:
        with urlopen_direct(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
    except (OSError, ValueError, HTTPError, URLError):
        return {}
    return payload if isinstance(payload, dict) else {}


# Return the base URL of the local Ollama service from its configured port, or an empty string.
def _ollama_base_url(settings: dict[str, Any], manifest: dict[str, Any]) -> str:
    port = _coerce_port(_runtime_setting(settings, manifest, "ollama-service_port"))
    if port is None:
        return ""
    return f"http://127.0.0.1:{port}"


# Find the entry for the given model in Ollama's loaded models list, or an empty dict.
def _find_loaded_ollama_model(models: list[Any], model_name: str) -> dict[str, Any]:
    for item in models:
        if not isinstance(item, dict):
            continue
        if item.get("model") == model_name or item.get("name") == model_name:
            return item
    return {}


# Return the key under which a model's record is stored in the metadata: "<engine>:<model>".
def _active_model_key(engine: str, model_name: str) -> str:
    return f"{engine}:{model_name}"


# Build the runtime metadata document for the active engine and model. ASLM's inference info route
# is the primary source; for Ollama, its loaded models and model metadata endpoints add runtime
# details. Unavailable sources only leave fields empty.
def _build_metadata() -> dict[str, Any]:
    # Locate the module's own UI server from the configured port and ask it for the inference info.
    settings = _read_json_file(SETTINGS_PATH)
    manifest = _read_json_file(MODULE_MANIFEST_PATH)
    ui_port = _coerce_port(_runtime_setting(settings, manifest, "ui-port", env_key="ASLM_UI_PORT"))
    inference_info_url = _local_route_url(ui_port, ASLM_INFERENCE_INFO_ROUTE)
    inference_info = _fetch_json(inference_info_url) if inference_info_url else {}

    # Active engine and model, plus the capabilities and limits blocks (ignored when not objects).
    engine = str(inference_info.get("engine") or settings.get("llm-engine") or "").strip()
    engine_label = str(inference_info.get("engine_label") or engine).strip()
    model_name = str(inference_info.get("model") or inference_info.get("model_name") or "").strip()
    source = inference_info.get("source", {})
    model_source = source.get("model") if isinstance(source, dict) else ""

    capabilities = inference_info.get("capabilities", {})
    if not isinstance(capabilities, dict):
        capabilities = {}
    limits = inference_info.get("limits", {})
    if not isinstance(limits, dict):
        limits = {}

    # Runtime details and the list of sources the metadata was gathered from.
    runtime: dict[str, Any] = {}
    sources = [
        {
            **_source_descriptor("ASLM inference info", ASLM_INFERENCE_INFO_ROUTE, "ui-port"),
            "notes": "Unified ASLM view of the active engine, selected model, capabilities and active limits.",
        }
    ]

    # Ollama corroboration: whether the model is loaded (with its context length and quantization)
    # and the capabilities the provider reports for it.
    if engine == "ollama-service" and model_name:
        ollama_base = _ollama_base_url(settings, manifest)
        ps_url = f"{ollama_base}/api/ps" if ollama_base else ""
        show_url = f"{ollama_base}/api/show" if ollama_base else ""
        ps_payload = _fetch_json(ps_url) if ps_url else {}
        loaded_model = _find_loaded_ollama_model(ps_payload.get("models", []), model_name)
        show_payload = _fetch_json(show_url, method="POST", body={"model": model_name}) if show_url else {}
        details = loaded_model.get("details", {}) if isinstance(loaded_model.get("details"), dict) else {}

        runtime = {
            "loaded": bool(loaded_model),
            "loaded_context_length": loaded_model.get("context_length"),
            "parameter_size": details.get("parameter_size"),
            "quantization_level": details.get("quantization_level"),
        }
        raw_capabilities = show_payload.get("capabilities")
        if isinstance(raw_capabilities, list):
            capabilities.setdefault("items", raw_capabilities)

        sources.extend(
            [
                {
                    **_source_descriptor("Ollama loaded models", "/api/ps", "ollama-service_port"),
                    "notes": "Confirms whether the selected model is loaded and reports loaded context_length.",
                },
                {
                    **_source_descriptor("Ollama model metadata", "/api/show", "ollama-service_port"),
                    "notes": "Confirms provider model capabilities and metadata.",
                },
            ]
        )

    # Record of the active model: normalized capability flags and context/output limits.
    model_record = {
        "engine": engine,
        "model": model_name,
        "capabilities": {
            "vision": bool(capabilities.get("supports_vision", False)),
            "tools": bool(capabilities.get("supports_tool_calling", False)),
            "thinking": bool(capabilities.get("supports_thinking", False)),
            "files": bool(capabilities.get("supports_files", False)),
        },
        "limits": {
            "context_window": limits.get("context_window", inference_info.get("context_window")),
            "model_context_limit": limits.get("model_context_limit", inference_info.get("model_context_limit")),
            "max_output_tokens": limits.get("max_output_tokens", inference_info.get("max_output_tokens")),
            "output_token_limit": limits.get("output_token_limit", inference_info.get("output_token_limit")),
        },
        "runtime": runtime,
        "sources": sources,
    }

    # The document: active model summary, and the model record only when an engine and model are known.
    return {
        "schema_version": 1,
        "_comment": (
            "Runtime model metadata consumed by local tools. Primary source: ASLM /api/inference_info/. "
            "Provider corroboration may come from engine-specific endpoints such as Ollama /api/ps and /api/show."
        ),
        "updated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "active": {
            "engine": engine,
            "engine_label": engine_label,
            "model": model_name,
            "source": model_source or "unknown",
            "primary_source": {
                **_source_descriptor("ASLM inference info", ASLM_INFERENCE_INFO_ROUTE, "ui-port"),
                "fields": [
                    "engine",
                    "model",
                    "capabilities.supports_vision",
                    "limits.context_window",
                    "limits.model_context_limit",
                ],
            },
        },
        "models": {
            _active_model_key(engine, model_name): model_record,
        }
        if engine and model_name
        else {},
    }


# Write JSON through a temporary file in the same folder and replace the target with it, so
# readers never see a half-written file.
def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


# Regenerate model_runtime_metadata.json and print a one-line JSON summary (path, engine, model,
# vision support).
def main() -> int:
    # Local requests must bypass any system proxy.
    apply_loopback_proxy_bypass()
    payload = _build_metadata()
    _write_json_atomic(METADATA_PATH, payload)

    # Summarize the written metadata for the caller.
    active = payload.get("active", {})
    print(
        json.dumps(
            {
                "metadata_path": str(METADATA_PATH),
                "engine": active.get("engine"),
                "model": active.get("model"),
                "vision": next(iter(payload.get("models", {}).values()), {})
                .get("capabilities", {})
                .get("vision"),
            },
            ensure_ascii=False,
        )
    )
    return 0


# Run: python Tools/update_model_runtime_metadata.py
if __name__ == "__main__":
    raise SystemExit(main())
