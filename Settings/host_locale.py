# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Project root and the JSON file that stores the last locale snapshot received from the ASLM host.
BASE_DIR = Path(__file__).resolve().parent.parent
HOST_LOCALE_FILE = BASE_DIR / "Settings" / "host_locale.json"


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


# Persist the ASLM host locale snapshot next to module settings.
def save_host_locale_payload(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise TypeError("host locale payload must be a dict")
    atomic_write_json(HOST_LOCALE_FILE, data)


# Load the last persisted host locale snapshot, or None when missing or invalid.
def load_host_locale() -> dict[str, Any] | None:
    if not HOST_LOCALE_FILE.exists():
        return None
    try:
        text = HOST_LOCALE_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read host locale file %s: %s", HOST_LOCALE_FILE, exc)
        return None
    text = text.lstrip("\ufeff").strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse host locale file %s: %s", HOST_LOCALE_FILE, exc)
        return None
    return raw if isinstance(raw, dict) else None


# Language codes supported by ASLM AppPersonalizationConfig / AppLocalizationService.
HOST_SUPPORTED_LANGUAGE_CODES: frozenset[str] = frozenset(
    {
        "en",
        "zh-Hans",
        "es",
        "ar",
        "hi",
        "pt-BR",
        "ru",
        "ja",
        "de",
        "fr",
        "ko",
        "it",
        "zh-Hant",
        "pt",
        "tr",
        "pl",
        "uk",
        "id",
        "vi",
        "nl",
    }
)


# Normalize a host language code to a supported value, defaulting to English.
def normalize_host_language(value: str | None) -> str:
    if not value or not str(value).strip():
        return "en"
    trimmed = str(value).strip()
    for code in HOST_SUPPORTED_LANGUAGE_CODES:
        if code.lower() == trimmed.lower():
            return code
    return "en"


# Return the BCP-47 language code from the host locale snapshot.
def get_language() -> str:
    payload = load_host_locale()
    if payload:
        return normalize_host_language(str(payload.get("language", "en")))
    return "en"


# Return the host-provided display name for the current language, when available.
def get_display_name() -> str | None:
    payload = load_host_locale()
    if not payload:
        return None
    name = payload.get("displayName")
    if name is None:
        return None
    text = str(name).strip()
    return text or None
