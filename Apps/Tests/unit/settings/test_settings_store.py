# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging

import pytest

from Settings import settings as project_settings


pytestmark = [pytest.mark.unit]


# Loading

# Verify that a fresh install without a settings file boots on the documented defaults.
def test_load_without_settings_file_returns_defaults(isolated_settings):
    data = project_settings.load_settings()

    assert data["ui-port"] == 20000
    assert data["llm-engine"] == "ollama-service"
    assert data["ollama-service_port"] == 20003
    assert data["sandbox-memory-limit"] == 6


# Verify that stored values win over the defaults while keys missing from the file fall back to the defaults.
def test_load_merges_stored_payload_over_defaults(isolated_settings):
    isolated_settings.joinpath("settings_file.json").write_text(
        json.dumps({"ui-port": 30123, "web-search-max-results": 25}),
        encoding="utf-8",
    )

    data = project_settings.load_settings()

    assert data["ui-port"] == 30123
    assert data["web-search-max-results"] == 25
    assert data["debug"] is True


# Verify that a corrupt settings file falls back to the defaults instead of crashing the boot.
def test_corrupt_settings_file_falls_back_to_defaults(isolated_settings):
    isolated_settings.joinpath("settings_file.json").write_text("{not json", encoding="utf-8")

    data = project_settings.load_settings()

    assert data["ui-port"] == 20000


# Verify that a settings file holding a JSON array instead of an object is ignored and the defaults are used.
def test_non_object_settings_file_is_ignored(isolated_settings):
    isolated_settings.joinpath("settings_file.json").write_text("[1, 2]", encoding="utf-8")

    assert project_settings.load_settings()["ui-port"] == 20000


# Verify that removed legacy tool keys are dropped from the loaded settings, even when the file or an environment
# variable still provides them, while other keys are kept.
@pytest.mark.parametrize("key", [
    "sandbox-image-source",
    "web-search-query-rewriter-endpoint",
    "web-search-query-rewriter-model",
])
def test_removed_tool_setting_keys_are_dropped(isolated_settings, monkeypatch, key):
    isolated_settings.joinpath("settings_file.json").write_text(
        json.dumps({key: "legacy", "ui-port": 20001}),
        encoding="utf-8",
    )
    monkeypatch.setenv(project_settings._to_env_var_name(key), "legacy-override")

    data = project_settings.load_settings()

    assert key not in data
    assert data["ui-port"] == 20001


# Environment overrides

# Verify that ASLM_* environment variables override the settings and their string values are coerced to the
# proper types (integer, boolean).
def test_environment_overrides_apply_with_value_normalization(isolated_settings, monkeypatch):
    monkeypatch.setenv("ASLM_UI_PORT", "31234")
    monkeypatch.setenv("ASLM_WEB_SEARCH_TOR_ENABLED", "true")

    data = project_settings.load_settings()

    assert data["ui-port"] == 31234
    assert data["web-search-tor-enabled"] is True


# Verify that the identity variables injected by the host (ASLM_MODULE_ID, ASLM_MODULE_DIR) do not become
# settings keys.
def test_host_identity_environment_keys_are_ignored(isolated_settings, monkeypatch):
    monkeypatch.setenv("ASLM_MODULE_ID", "aslm-chat")
    monkeypatch.setenv("ASLM_MODULE_DIR", "C:/somewhere")

    data = project_settings.load_settings()

    assert "module_id" not in data
    assert "module_dir" not in data


# Verify that an environment variable name maps onto a historical key that mixes separators
# (ollama-service_port).
def test_environment_key_normalization_matches_historical_keys(isolated_settings, monkeypatch):
    monkeypatch.setenv("ASLM_OLLAMA_SERVICE_PORT", "31111")

    data = project_settings.load_settings()

    assert data["ollama-service_port"] == 31111


# Saving

# Verify that save_settings persists the values so load_settings returns them and the settings file contains
# them.
def test_save_settings_roundtrip(isolated_settings):
    project_settings.save_settings({"ui-port": 30500})

    data = project_settings.load_settings()

    assert data["ui-port"] == 30500
    assert json.loads(isolated_settings.joinpath("settings_file.json").read_text(encoding="utf-8"))["ui-port"] == 30500


# Verify that saving settings regenerates the derived web search API key and browser agent configuration files.
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


# Verify that saving settings removes the old endpoint and model overrides of the query rewriter section while
# keeping the options that exist only in the JSON file.
def test_save_settings_removes_legacy_query_rewriter_overrides(isolated_settings):
    path = project_settings.WEB_SEARCH_CONFIG_FILE
    path.write_text(json.dumps({"query_rewriter": {
        "endpoint": "http://legacy.invalid/chat/completions",
        "model": "legacy-model",
        "timeout_seconds": 42,
        "max_queries": 2,
    }}), encoding="utf-8")

    project_settings.save_settings({"web-search-query-rewriter-enabled": True})

    section = json.loads(path.read_text(encoding="utf-8"))["query_rewriter"]
    assert "endpoint" not in section
    assert "model" not in section
    assert section["enabled"] is True
    assert section["timeout_seconds"] == 42
    assert section["max_queries"] == 2


# Verify that set() persists the value in the settings file and mirrors it into the module manifest.
def test_set_persists_value_and_mirrors_manifest(isolated_settings, isolated_manifest):
    project_settings.set("ui-port", 30001, sync_runtime=False)

    stored = json.loads(isolated_settings.joinpath("settings_file.json").read_text(encoding="utf-8"))
    assert stored["ui-port"] == 30001

    manifest = json.loads(isolated_manifest.path.read_text(encoding="utf-8"))
    mirrored = {item["key"]: item.get("value") for item in manifest["settings"]}
    assert mirrored["ui-port"] == 30001


# Verify that a host account key is stored in the local settings file but never mirrored into the module
# manifest.
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


# Verify that two services configured with the same TCP port produce a port collision warning on load.
def test_port_collision_logs_warning(isolated_settings, caplog):
    isolated_settings.joinpath("settings_file.json").write_text(
        json.dumps({"ui-port": 30000, "ollama-service_port": 30000}),
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING, logger="Settings.settings"):
        project_settings.load_settings()

    assert any("port collision" in record.message.lower() for record in caplog.records)


# Engine resolution

# Verify that requesting an engine that is not enabled (also through an alias) resolves to the first enabled
# engine.
def test_resolve_enabled_engine_falls_back_to_first_enabled(isolated_settings):
    project_settings.save_settings({"openai": True}, sync_tool_configs=False)

    assert project_settings.resolve_enabled_engine("google-genai") == "openai"
    assert project_settings.resolve_enabled_engine("gemini") == "openai"


# Verify that the configured LLM engine is kept when it is enabled.
def test_resolve_enabled_engine_keeps_enabled_engine(isolated_settings):
    project_settings.save_settings({"openai": True, "llm-engine": "openai"}, sync_tool_configs=False)

    assert project_settings.get_llm_engine() == "openai"


# Verify that engine URLs are built from the configured port or address: loopback with the port for the managed
# Ollama service, no trailing slash for OpenAI and an added https scheme for Google GenAI.
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


# Verify that the OpenAI API key is empty by default, comes from the OPENAI_API_KEY environment variable when no
# key is stored, and that a stored key wins over the environment.
def test_openai_api_key_falls_back_to_environment(isolated_settings, monkeypatch):
    assert project_settings.get_openai_api_key() == ""

    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    assert project_settings.get_openai_api_key() == "env-key"

    project_settings.save_settings({"openai_api_key": "stored-key"}, sync_tool_configs=False)
    assert project_settings.get_openai_api_key() == "stored-key"


# Value normalization

# Verify that address normalization strips the scheme and trailing slash for storage and keeps a bare host and
# port.
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


# Verify that engine aliases resolve to their canonical identifiers and a missing name resolves to the managed
# Ollama service.
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


# Verify that string values from environment variables are coerced to numbers (also with a decimal comma),
# booleans, null and JSON objects, and plain text is kept.
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


# Verify that a gigabyte value is accepted as a positive number or numeric string, while zero, negative,
# non-numeric and boolean values are replaced by the given default.
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
