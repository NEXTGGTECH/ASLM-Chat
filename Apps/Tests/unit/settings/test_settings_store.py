# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging

import pytest

from Settings import settings as project_settings


pytestmark = [pytest.mark.unit]


# Loading.


# A fresh install without settings.json boots on the documented defaults.
def test_load_without_settings_file_returns_defaults(isolated_settings):
    data = project_settings.load_settings()

    assert data["ui-port"] == 20000
    assert data["llm-engine"] == "ollama-service"
    assert data["ollama-service_port"] == 20003
    assert data["sandbox-memory-limit"] == 6


# Stored values win over defaults, missing keys fall back to defaults.
def test_load_merges_stored_payload_over_defaults(isolated_settings):
    isolated_settings.joinpath("settings_file.json").write_text(
        json.dumps({"ui-port": 30123, "web-search-max-results": 25}),
        encoding="utf-8",
    )

    data = project_settings.load_settings()

    assert data["ui-port"] == 30123
    assert data["web-search-max-results"] == 25
    assert data["debug"] is True


# A corrupt settings file degrades to defaults instead of crashing boot.
def test_corrupt_settings_file_falls_back_to_defaults(isolated_settings):
    isolated_settings.joinpath("settings_file.json").write_text("{not json", encoding="utf-8")

    data = project_settings.load_settings()

    assert data["ui-port"] == 20000


# A settings file holding a JSON array (not an object) is ignored.
def test_non_object_settings_file_is_ignored(isolated_settings):
    isolated_settings.joinpath("settings_file.json").write_text("[1, 2]", encoding="utf-8")

    assert project_settings.load_settings()["ui-port"] == 20000


# Removed legacy tool keys are silently dropped from loaded snapshots.
def test_removed_tool_setting_keys_are_dropped(isolated_settings):
    isolated_settings.joinpath("settings_file.json").write_text(
        json.dumps({"sandbox-image-source": "local", "ui-port": 20001}),
        encoding="utf-8",
    )

    data = project_settings.load_settings()

    assert "sandbox-image-source" not in data
    assert data["ui-port"] == 20001


# Environment overrides.


# ASLM_* environment variables override the persisted values with coercion.
def test_environment_overrides_apply_with_value_normalization(isolated_settings, monkeypatch):
    monkeypatch.setenv("ASLM_UI_PORT", "31234")
    monkeypatch.setenv("ASLM_WEB_SEARCH_TOR_ENABLED", "true")

    data = project_settings.load_settings()

    assert data["ui-port"] == 31234
    assert data["web-search-tor-enabled"] is True


# Host-injected identity variables never become settings keys.
def test_host_identity_environment_keys_are_ignored(isolated_settings, monkeypatch):
    monkeypatch.setenv("ASLM_MODULE_ID", "aslm-chat")
    monkeypatch.setenv("ASLM_MODULE_DIR", "C:/somewhere")

    data = project_settings.load_settings()

    assert "module_id" not in data
    assert "module_dir" not in data


# Historical mixed-separator keys resolve from environment variable names.
def test_environment_key_normalization_matches_historical_keys(isolated_settings, monkeypatch):
    monkeypatch.setenv("ASLM_OLLAMA_SERVICE_PORT", "31111")

    data = project_settings.load_settings()

    assert data["ollama-service_port"] == 31111


# Saving.


# save_settings persists and reloads through the cache.
def test_save_settings_roundtrip(isolated_settings):
    project_settings.save_settings({"ui-port": 30500})

    data = project_settings.load_settings()

    assert data["ui-port"] == 30500
    assert json.loads(isolated_settings.joinpath("settings_file.json").read_text(encoding="utf-8"))["ui-port"] == 30500


# Saving settings regenerates the derived tool configuration documents.
def test_save_settings_syncs_generated_tool_configs(isolated_settings):
    payload = {"web-search-tavily-api-key": "tvly-secret-123", "browser-agent-headless": True}

    project_settings.save_settings(payload)

    api_keys_doc = json.loads(
        isolated_settings.joinpath("web_search_api_keys_file.json").read_text(encoding="utf-8")
    )
    assert api_keys_doc["search"]["hosted_api"]["tavily_api_key"] == "tvly-secret-123"

    browser_doc = json.loads(
        isolated_settings.joinpath("browser_agent_config_file.json").read_text(encoding="utf-8")
    )
    assert browser_doc["browser_headless"] is True


# set() persists the value and mirrors it into the module manifest.
def test_set_persists_value_and_mirrors_manifest(isolated_settings, isolated_manifest):
    project_settings.set("ui-port", 30001, sync_runtime=False)

    stored = json.loads(isolated_settings.joinpath("settings_file.json").read_text(encoding="utf-8"))
    assert stored["ui-port"] == 30001

    manifest = json.loads(isolated_manifest.path.read_text(encoding="utf-8"))
    mirrored = {item["key"]: item.get("value") for item in manifest["settings"]}
    assert mirrored["ui-port"] == 30001


# Host account keys never leave the local settings file.
def test_host_key_settings_are_not_mirrored_to_manifest(isolated_settings, isolated_manifest):
    # The manifest advertises the key so only the guard, not its absence,
    # can make this test pass.
    manifest = json.loads(isolated_manifest.path.read_text(encoding="utf-8"))
    manifest["settings"].append({"key": "key-gh", "value": None})
    isolated_manifest.path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")

    project_settings.set("key-gh", "gh-token", sync_runtime=False)

    manifest = json.loads(isolated_manifest.path.read_text(encoding="utf-8"))
    mirrored = {item["key"]: item.get("value") for item in manifest["settings"]}
    assert mirrored["key-gh"] is None

    stored = json.loads(isolated_settings.joinpath("settings_file.json").read_text(encoding="utf-8"))
    assert stored["key-gh"] == "gh-token"


# Sharing one TCP port between two services produces a collision warning.
def test_port_collision_logs_warning(isolated_settings, caplog):
    isolated_settings.joinpath("settings_file.json").write_text(
        json.dumps({"ui-port": 30000, "ollama-service_port": 30000}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="Settings.settings"):
        project_settings.load_settings()

    assert any("port collision" in record.message.lower() for record in caplog.records)


# Engine resolution.


# Requesting an engine that is not enabled falls back to the first enabled one.
def test_resolve_enabled_engine_falls_back_to_first_enabled(isolated_settings):
    project_settings.save_settings({"openai": True}, sync_tool_configs=False)

    assert project_settings.resolve_enabled_engine("google-genai") == "openai"
    assert project_settings.resolve_enabled_engine("gemini") == "openai"


# The active engine is preserved when it is enabled.
def test_resolve_enabled_engine_keeps_enabled_engine(isolated_settings):
    project_settings.save_settings({"openai": True, "llm-engine": "openai"}, sync_tool_configs=False)

    assert project_settings.get_llm_engine() == "openai"


# Engine URLs are built from the configured endpoints.
def test_get_engine_url_builds_endpoints(isolated_settings):
    project_settings.save_settings(
        {
            "ollama-service_port": 30100,
            "openai_url": "https://api.example.com/v1/",
            "google_genai_url": "generativelanguage.googleapis.com",
        },
        sync_tool_configs=False,
    )

    assert project_settings.get_engine_url("ollama-service") == "http://127.0.0.1:30100"
    assert project_settings.get_engine_url("openai") == "https://api.example.com/v1"
    assert project_settings.get_engine_url("google-genai") == "https://generativelanguage.googleapis.com"


# The OpenAI key getter prefers the stored key and falls back to the env var.
def test_openai_api_key_falls_back_to_environment(isolated_settings, monkeypatch):
    assert project_settings.get_openai_api_key() == ""

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert project_settings.get_openai_api_key() == "env-key"

    project_settings.save_settings({"openai_api_key": "stored-key"}, sync_tool_configs=False)
    assert project_settings.get_openai_api_key() == "stored-key"


# Value normalization.


# Address normalization strips schemes and trailing slashes for storage.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://api.example.com/v1/", "api.example.com/v1"),
        ("http://127.0.0.1:8000/v1", "127.0.0.1:8000/v1"),
        ("127.0.0.1:1234", "127.0.0.1:1234"),
        ("", ""),
    ],
)
def test_normalize_engine_address(raw, expected):
    assert project_settings.normalize_engine_address(raw) == expected


# Engine aliases resolve to canonical identifiers.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ollama", "ollama-service"),
        ("lm-studio", "lms"),
        ("openai-api", "openai"),
        ("gemini", "google-genai"),
        ("google", "google-genai"),
        (None, "ollama-service"),
    ],
)
def test_normalize_engine_name(raw, expected):
    assert project_settings.normalize_engine_name(raw) == expected


# Scalar coercion covers the value formats arriving from environment variables.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("42", 42),
        ("3.14", 3.14),
        ("3,14", 3.14),
        ("true", True),
        ("false", False),
        ("null", None),
        ('{"a": 1}', {"a": 1}),
        ("hello", "hello"),
    ],
)
def test_normalize_setting_value(raw, expected):
    assert project_settings.normalize_setting_value(raw) == expected


# Gigabyte settings reject non-positive and non-numeric values.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (8, 8),
        ("12", 12),
        (0, 6),
        (-3, 6),
        ("huge", 6),
        (True, 6),
    ],
)
def test_normalize_gigabyte_value(raw, expected):
    assert project_settings._normalize_gigabyte_value(raw, 6) == expected
