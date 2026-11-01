# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

SANDBOX_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_JSON_FILE = SANDBOX_ROOT / "sandbox.json"
SANDBOX_ENV_FILE = SANDBOX_ROOT / "sandbox.env"

SANDBOX_CONFIG_KEYS = (
    "SANDBOX_CONTAINER_NAME",
    "SANDBOX_IMAGE",
    "SANDBOX_IMAGE_SOURCE",
    "SANDBOX_CPU_LIMIT",
    "SANDBOX_MEMORY_LIMIT",
    "SANDBOX_MEMORY_SWAP_LIMIT",
    "SANDBOX_PIDS_LIMIT",
    "SANDBOX_STORAGE_LIMIT",
    "SANDBOX_NETWORK_ENABLED",
    "SANDBOX_NETWORK_LIMIT_MBIT",
    "SANDBOX_DEFAULT_TIMEOUT",
    "SANDBOX_MAX_OUTPUT_BYTES",
    "SANDBOX_OUTPUT_HEAD_RATIO",
    "SANDBOX_MAX_READ_BYTES",
    "SANDBOX_MAX_CAT_FILE_BYTES",
    "SANDBOX_MAX_CAT_LINE_THRESHOLD",
    "SANDBOX_MAX_IMAGE_PREVIEW_BYTES",
    "SANDBOX_MAX_LS_ENTRIES",
    "SANDBOX_MAX_FIND_RESULTS",
    "SANDBOX_MAX_GREP_RESULTS",
    "SANDBOX_BACKGROUND_TIMEOUT_THRESHOLD",
    "SANDBOX_THREAD_LIMIT",
    "SANDBOX_DEFAULT_TASK_DIR",
    "SANDBOX_WORKSPACE_CLEANUP_ENABLED",
    "SANDBOX_WORKSPACE_CLEANUP_IDLE_SECONDS",
    "SANDBOX_WORKSPACE_CLEANUP_RECYCLE_SECONDS",
    "SANDBOX_WORKSPACE_CLEANUP_INTERVAL_SECONDS",
    "SANDBOX_MAX_FILE_MAP_SYMBOLS",
    "SANDBOX_DOCKER_START_TIMEOUT_SECONDS",
    "SANDBOX_AUTO_START_DOCKER",
)


# Load and whitelist the generated sandbox JSON document.
def load_sandbox_json(path: Path = SANDBOX_JSON_FILE) -> dict[str, Any] | None:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read sandbox config %s: %s", path, exc)
        return None

    if not isinstance(raw, dict):
        logger.warning("Sandbox config %s must contain a JSON object", path)
        return None

    return {key: raw[key] for key in SANDBOX_CONFIG_KEYS if key in raw}


# Serialize one JSON scalar to the line-oriented sandbox.env representation.
def _serialize_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if value is None:
        return ""
    return str(value).replace("\r", "").replace("\n", "")


# Render deterministic sandbox.env content from one validated JSON configuration.
def _render_sandbox_env(config: dict[str, Any]) -> str:
    lines = [
        "# sandbox.env - generated from sandbox.json. Do not edit manually.",
        "",
    ]
    lines.extend(
        f"{key}={_serialize_env_value(config[key])}"
        for key in SANDBOX_CONFIG_KEYS
        if key in config
    )
    return "\n".join(lines) + "\n"


# Atomically replace sandbox.env only when its generated content changed.
def _write_env_if_changed(path: Path, content: str) -> None:
    try:
        if path.read_text(encoding="utf-8-sig") == content:
            return
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not compare generated sandbox env %s: %s", path, exc)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        temporary_path.write_text(content, encoding="utf-8")
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# Synchronize sandbox.env from sandbox.json.
def sync_sandbox_env(
    json_path: Path = SANDBOX_JSON_FILE,
    env_path: Path = SANDBOX_ENV_FILE,
) -> None:
    config = load_sandbox_json(json_path)
    if config is None:
        try:
            env_path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove stale sandbox env %s: %s", env_path, exc)
        return

    _write_env_if_changed(env_path, _render_sandbox_env(config))
