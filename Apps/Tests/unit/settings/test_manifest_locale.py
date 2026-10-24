# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging

import pytest

from Settings import module_manifest_locale


pytestmark = [pytest.mark.unit]


# Build a minimal manifest with the sections the localizer patches.
def _write_manifest(path) -> None:
    manifest = {
        "id": "aslm-chat",
        "name": "ASLM Chat",
        "description": "Chat module",
        "commands": {
            "firstRun": [{"name": "First Run Setup", "description": "Prepare", "exec": "main.py first_run"}],
            "run": [{"name": "UI Server", "description": "Serve UI", "exec": "main.py runserver"}],
        },
        "settings": [
            {"key": "generate-chat-titles", "name": "Generate chat titles", "value": True},
            {"key": "ui-port", "name": "UI port", "value": 20000},
        ],
        "settingCategories": [
            {"id": "engines", "name": "Engines", "description": "Engine switches"},
            {"id": "unknown", "name": "Unknown", "description": "Not localized"},
        ],
        "downloadsBridge": {
            "categories": [
                {"id": "ollama-library", "title": "Ollama models", "description": "Official models"}
            ]
        },
    }
    path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")


# Build a locale catalog with overlays for every manifest section.
def _write_locale_catalog(directory, language, name) -> None:
    catalog = {
        "name": f"{name} Chat",
        "description": f"{name} chat module",
        "commands": {
            "firstRun": [{"name": f"{name} First Run", "description": f"{name} prepare"}],
            "run": [{"name": f"{name} Server", "description": f"{name} serve"}],
        },
        "settings": {
            "generate-chat-titles": {"name": f"{name} titles", "description": f"{name} titles desc"},
            "absent-key": {"name": "never applied"},
        },
        "settingCategories": [
            {"id": "engines", "name": f"{name} Engines", "description": f"{name} engines desc"},
            {"id": "missing-elsewhere", "name": "unused"},
        ],
        "downloadsBridge": {
            "categories": [
                {"id": "ollama-library", "title": f"{name} models", "description": f"{name} models desc"},
                {"id": "not-in-manifest", "title": "ignored"},
            ]
        },
    }
    directory.joinpath(f"{language}.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8"
    )


@pytest.fixture
def localized_manifest(tmp_path, monkeypatch):
    manifest_path = tmp_path / "ASLM_Module.json"
    locales_dir = tmp_path / "locales"
    locales_dir.mkdir()
    monkeypatch.setattr(module_manifest_locale, "MANIFEST_FILE", manifest_path)
    monkeypatch.setattr(module_manifest_locale, "MANIFEST_LOCALES_DIR", locales_dir)
    return manifest_path, locales_dir


# Applying a language.


# Applying a language rewrites every localized section in place.
def test_apply_manifest_locale_patches_all_sections(localized_manifest):
    manifest_path, locales_dir = localized_manifest
    _write_manifest(manifest_path)
    _write_locale_catalog(locales_dir, "de", "German")

    module_manifest_locale.apply_manifest_locale("de")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "German Chat"
    assert manifest["description"] == "German chat module"
    assert manifest["commands"]["firstRun"][0]["name"] == "German First Run"
    assert manifest["commands"]["run"][0]["description"] == "German serve"

    settings_by_key = {item["key"]: item for item in manifest["settings"]}
    assert settings_by_key["generate-chat-titles"]["name"] == "German titles"
    # Settings without a locale overlay keep their original values.
    assert settings_by_key["ui-port"]["name"] == "UI port"

    categories = {item["id"]: item for item in manifest["settingCategories"]}
    assert categories["engines"]["name"] == "German Engines"
    assert categories["unknown"]["name"] == "Unknown"

    bridge_category = manifest["downloadsBridge"]["categories"][0]
    assert bridge_category["title"] == "German models"


# Command overlays apply by index and skip missing entries.
def test_command_overlay_applies_by_index(localized_manifest):
    manifest_path, locales_dir = localized_manifest
    _write_manifest(manifest_path)
    locales_dir.joinpath("en.json").write_text(
        json.dumps({"commands": {"run": [{"name": "Renamed"}]}}),
        encoding="utf-8",
    )

    module_manifest_locale.apply_manifest_locale("en")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["commands"]["run"][0]["name"] == "Renamed"
    # firstRun is untouched because the catalog has no run-length overlay.
    assert manifest["commands"]["firstRun"][0]["name"] == "First Run Setup"


# Unknown languages fall back to the English catalog.
def test_unknown_language_falls_back_to_english(localized_manifest):
    manifest_path, locales_dir = localized_manifest
    _write_manifest(manifest_path)
    _write_locale_catalog(locales_dir, "en", "English")

    module_manifest_locale.apply_manifest_locale("klingon")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["name"] == "English Chat"


# Failure modes.


# A missing manifest is a warning no-op instead of a crash.
def test_missing_manifest_is_a_noop(localized_manifest, caplog):
    _manifest_path, locales_dir = localized_manifest
    _write_locale_catalog(locales_dir, "en", "English")

    with caplog.at_level(logging.WARNING, logger="Settings.module_manifest_locale"):
        module_manifest_locale.apply_manifest_locale("en")

    assert any("Module manifest not found" in record.getMessage() for record in caplog.records)


# A corrupt manifest is left untouched on disk.
def test_corrupt_manifest_is_a_noop(localized_manifest):
    manifest_path, locales_dir = localized_manifest
    manifest_path.write_text("{broken", encoding="utf-8")
    _write_locale_catalog(locales_dir, "en", "English")

    module_manifest_locale.apply_manifest_locale("en")

    assert manifest_path.read_text(encoding="utf-8") == "{broken"
