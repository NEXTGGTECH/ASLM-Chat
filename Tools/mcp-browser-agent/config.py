# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Base paths: the agent folder, the shared sandbox, the download/workspace folder (overridable through
# ASLM_BROWSER_WORKSPACE_DIR) and the generated config file.
BROWSER_AGENT_ROOT = Path(__file__).resolve().parent
SANDBOX_DIR = BROWSER_AGENT_ROOT.parent / "mcp-sandbox" / "_sandbox"
DOWNLOADS_DIR = Path(os.getenv("ASLM_BROWSER_WORKSPACE_DIR", SANDBOX_DIR)).resolve()
CONFIG_FILE = BROWSER_AGENT_ROOT / "config.json"

# Fallback value for every setting that config.json may override.
DEFAULT_CONFIG: dict[str, Any] = {
    "browser_width": 1280,
    "browser_height": 800,
    "browser_headless": False,
    "max_a11y_depth": 15,
    "max_elements": 200,
    "max_main_interactive": 60,
    "auto_text_preview_length": 1500,
}


# Load the generated browser config and fall back per field when it is missing or invalid.
def _load_browser_config(path: Path = CONFIG_FILE) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raw = {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read browser-agent config %s: %s", path, exc)
        raw = {}

    if not isinstance(raw, dict):
        raw = {}

    config = dict(DEFAULT_CONFIG)
    for key in DEFAULT_CONFIG:
        if key in raw:
            config[key] = raw[key]
    return config


# Read one positive integer browser setting without allowing invalid runtime constants.
def _positive_int(config: dict[str, Any], key: str) -> int:
    default = int(DEFAULT_CONFIG[key])
    try:
        value = int(config.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


# Read one Boolean browser setting from JSON-compatible scalar values.
def _bool_value(config: dict[str, Any], key: str) -> bool:
    value = config.get(key, DEFAULT_CONFIG[key])
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


# Settings loaded once at import time; the constants below are derived from them.
_CONFIG = _load_browser_config()

# Browser window settings.
BROWSER_WIDTH = _positive_int(_CONFIG, "browser_width")
BROWSER_HEIGHT = _positive_int(_CONFIG, "browser_height")
BROWSER_HEADLESS = _bool_value(_CONFIG, "browser_headless")

# Accessibility snapshot limits.
MAX_A11Y_DEPTH = _positive_int(_CONFIG, "max_a11y_depth")
MAX_ELEMENTS = _positive_int(_CONFIG, "max_elements")
MAX_MAIN_INTERACTIVE = _positive_int(_CONFIG, "max_main_interactive")

# Snapshot text preview.
AUTO_TEXT_PREVIEW_LEN = _positive_int(_CONFIG, "auto_text_preview_length")
