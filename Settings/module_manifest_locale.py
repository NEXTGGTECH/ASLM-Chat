# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from Settings.host_locale import atomic_write_json, normalize_host_language

logger = logging.getLogger(__name__)

# Project root, the module manifest that gets patched in place and the folder with one
# translated catalog per language (<language>.json).
BASE_DIR = Path(__file__).resolve().parent.parent
MANIFEST_FILE = BASE_DIR / "ASLM_Module.json"
MANIFEST_LOCALES_DIR = Path(__file__).resolve().parent / "module_manifest_locales"

# Language whose catalog is the fallback when the requested one is missing.
BASE_LOCALE = "en"


# Load one manifest locale catalog from disk, falling back to English.
def _load_manifest_locale(language: str | None) -> dict[str, Any]:
    normalized = normalize_host_language(language)
    path = MANIFEST_LOCALES_DIR / f"{normalized}.json"
    if not path.is_file():
        path = MANIFEST_LOCALES_DIR / f"{BASE_LOCALE}.json"
    if not path.is_file():
        logger.warning("Manifest locale catalog not found for %s", normalized)
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not load manifest locale catalog %s: %s", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


# Apply localized name/description fields to one command list by index.
def _patch_command_list(commands: list[dict[str, Any]], localized: list[dict[str, Any]] | None) -> None:
    if not isinstance(localized, list):
        return
    for index, entry in enumerate(commands):
        if not isinstance(entry, dict) or index >= len(localized):
            continue
        overlay = localized[index]
        if not isinstance(overlay, dict):
            continue
        if "name" in entry and "name" in overlay:
            entry["name"] = overlay["name"]
        if "description" in entry and "description" in overlay:
            entry["description"] = overlay["description"]


# Apply localized fields to manifest settings entries by setting key.
def _patch_settings(settings: list[dict[str, Any]], localized: dict[str, Any] | None) -> None:
    if not isinstance(localized, dict):
        return
    for entry in settings:
        if not isinstance(entry, dict):
            continue
        key = entry.get("key")
        if not key or key not in localized:
            continue
        overlay = localized[key]
        if not isinstance(overlay, dict):
            continue
        if "name" in entry and "name" in overlay:
            entry["name"] = overlay["name"]
        if "description" in entry and "description" in overlay:
            entry["description"] = overlay["description"]


# Apply localized fields to setting categories by category id.
def _patch_setting_categories(
    categories: list[dict[str, Any]],
    localized: list[dict[str, Any]] | None,
) -> None:
    if not isinstance(localized, list):
        return

    by_id = {
        str(item.get("id")): item
        for item in localized
        if isinstance(item, dict) and item.get("id") is not None
    }
    for entry in categories:
        if not isinstance(entry, dict):
            continue
        overlay = by_id.get(str(entry.get("id", "")))
        if not isinstance(overlay, dict):
            continue
        if "name" in entry and "name" in overlay:
            entry["name"] = overlay["name"]
        if "description" in entry and "description" in overlay:
            entry["description"] = overlay["description"]


# Apply localized fields to downloads bridge categories by category id.
def _patch_download_categories(categories: list[dict[str, Any]], localized: list[dict[str, Any]] | None) -> None:
    if not isinstance(localized, list):
        return
    by_id = {
        str(item.get("id")): item
        for item in localized
        if isinstance(item, dict) and item.get("id") is not None
    }
    for entry in categories:
        if not isinstance(entry, dict):
            continue
        category_id = str(entry.get("id", ""))
        overlay = by_id.get(category_id)
        if not isinstance(overlay, dict):
            continue
        if "title" in overlay:
            entry["title"] = overlay["title"]
        if "description" in overlay:
            entry["description"] = overlay["description"]


# Patch ASLM_Module.json with localized manifest strings for the given language.
def apply_manifest_locale(language: str | None) -> None:
    # Read the current manifest; a missing, unreadable or non-object manifest is left untouched.
    if not MANIFEST_FILE.is_file():
        logger.warning("Module manifest not found at %s", MANIFEST_FILE)
        return

    try:
        manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read module manifest %s: %s", MANIFEST_FILE, exc)
        return
    if not isinstance(manifest, dict):
        logger.warning("Module manifest must be a JSON object.")
        return

    # Load the translated catalog; an empty catalog means there is nothing to apply.
    locale = _load_manifest_locale(language)
    if not locale:
        return

    # Module name and description.
    if "name" in locale:
        manifest["name"] = locale["name"]
    if "description" in locale:
        manifest["description"] = locale["description"]

    # Command lists (first run and run) are patched by position.
    commands = manifest.get("commands")
    localized_commands = locale.get("commands")
    if isinstance(commands, dict) and isinstance(localized_commands, dict):
        _patch_command_list(
            commands.get("firstRun") if isinstance(commands.get("firstRun"), list) else [],
            localized_commands.get("firstRun") if isinstance(localized_commands.get("firstRun"), list) else None,
        )
        _patch_command_list(
            commands.get("run") if isinstance(commands.get("run"), list) else [],
            localized_commands.get("run") if isinstance(localized_commands.get("run"), list) else None,
        )

    # Settings entries are matched by key.
    settings = manifest.get("settings")
    if isinstance(settings, list):
        _patch_settings(settings, locale.get("settings") if isinstance(locale.get("settings"), dict) else None)

    # Setting categories are matched by id.
    setting_categories = manifest.get("settingCategories")
    localized_setting_categories = locale.get("settingCategories")
    if isinstance(setting_categories, list):
        _patch_setting_categories(
            setting_categories,
            localized_setting_categories if isinstance(localized_setting_categories, list) else None,
        )

    # Downloads bridge categories are matched by id.
    downloads_bridge = manifest.get("downloadsBridge")
    localized_bridge = locale.get("downloadsBridge")
    if isinstance(downloads_bridge, dict) and isinstance(localized_bridge, dict):
        categories = downloads_bridge.get("categories")
        localized_categories = localized_bridge.get("categories")
        if isinstance(categories, list):
            _patch_download_categories(
                categories,
                localized_categories if isinstance(localized_categories, list) else None,
            )

    # Persist the patched manifest atomically.
    atomic_write_json(MANIFEST_FILE, manifest)
    logger.info("Applied manifest locale %s to %s", normalize_host_language(language), MANIFEST_FILE)
