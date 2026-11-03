# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Project root and the JSON file that stores the last theme snapshot received from the ASLM host.
BASE_DIR = Path(__file__).resolve().parent.parent
HOST_THEME_FILE = BASE_DIR / "Settings" / "host_theme.json"


# Write JSON atomically via a temporary file and replace on success.
def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
            file.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                # The leftover temporary file is harmless; the original write error is re-raised below.
                pass
        raise


# Persist the ASLM host theme snapshot next to module settings.
def save_host_theme_payload(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise TypeError("host theme payload must be a dict")
    atomic_write_json(HOST_THEME_FILE, data)


# Load the last persisted host theme snapshot, or None when missing or invalid.
def load_host_theme() -> dict[str, Any] | None:
    if not HOST_THEME_FILE.exists():
        return None
    try:
        text = HOST_THEME_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read host theme file %s: %s", HOST_THEME_FILE, exc)
        return None
    text = text.lstrip("\ufeff").strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse host theme file %s: %s", HOST_THEME_FILE, exc)
        return None
    return raw if isinstance(raw, dict) else None
