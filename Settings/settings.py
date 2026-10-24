# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
SETTINGS_FILE = BASE_DIR / "Settings" / "settings.json"
WEB_SEARCH_API_KEYS_FILE = BASE_DIR / "Tools" / "mcp-web-search" / "core" / "config" / "api_keys.json"
WEB_SEARCH_CONFIG_FILE = BASE_DIR / "Tools" / "mcp-web-search" / "core" / "config" / "search_config.json"
BROWSER_AGENT_CONFIG_FILE = BASE_DIR / "Tools" / "mcp-browser-agent" / "config.json"
SANDBOX_CONFIG_FILE = BASE_DIR / "Tools" / "mcp-sandbox" / "sandbox.json"

ENGINE_LABELS = {
    "ollama-service": "Ollama",
    "lms": "LM Studio",
    "openai": "OpenAI-Compatible",
    "google-genai": "Google GenAI",
}

ENGINE_IDS = ("ollama-service", "lms", "openai", "google-genai")

ENGINE_ALIASES = {
    "ollama": "ollama-service",
    "ollama-service": "ollama-service",
    "lms": "lms",
    "lm-studio": "lms",
    "openai": "openai",
    "openai-api": "openai",
    "google-genai": "google-genai",
    "google_genai": "google-genai",
    "google": "google-genai",
    "gemini": "google-genai",
}

ENGINE_URL_KEYS = {
    "ollama-service": None,
    "lms": "lms_url",
    "openai": "openai_url",
    "google-genai": "google_genai_url",
}

ENGINE_API_KEY_KEYS = {
    "openai": "openai_api_key",
    "google-genai": "google_genai_api_key",
}

HOST_KEY_SETTING_KEYS = frozenset({"key-aslm", "key-gh"})

WEB_SEARCH_API_KEY_SETTINGS = {
    "web-search-tavily-api-key": "tavily_api_key",
    "web-search-firecrawl-api-key": "firecrawl_api_key",
    "web-search-brave-api-key": "brave_api_key",
    "web-search-serpapi-api-key": "serpapi_api_key",
}

WEB_SEARCH_CONFIG_SETTINGS = {
    "web-search-max-results": ("search", "max_results"),
    "web-search-total-context-budget": ("search", "total_context_budget"),
    "web-search-prefetch-timeout": ("search", "prefetch_fetch_timeout"),
    "web-search-preview-max-chars": ("search", "preview_max_chars"),
    "web-search-extraction-timeout": ("extraction", "timeout_seconds"),
    "web-search-max-page-chars": ("extraction", "max_page_chars"),
    "web-search-min-content-length": ("extraction", "min_content_length"),
    "web-search-compress-read-pages": ("extraction", "enable_read_page_compress"),
    "web-search-compress-threshold": ("extraction", "read_page_compress_threshold_chars"),
    "web-search-compress-target": ("extraction", "read_page_compress_target_chars"),
    "web-search-cache-ttl": ("cache", "search_ttl_seconds"),
    "web-search-negative-cache-ttl": ("cache", "search_negative_ttl_seconds"),
    "web-search-page-cache-ttl": ("cache", "page_ttl_seconds"),
    "web-search-repeat-block-window": ("cache", "repeat_block_window_seconds"),
    "web-search-seen-source-window": ("cache", "seen_source_window_seconds"),
    "web-search-prefetch-max-urls": ("cache", "prefetch_max_urls"),
    "web-search-tor-enabled": ("tor", "enabled"),
    "web-search-engine-google": ("engines", "google"),
    "web-search-engine-duckduckgo": ("engines", "duckduckgo"),
    "web-search-engine-startpage": ("engines", "startpage"),
    "web-search-engine-qwant": ("engines", "qwant"),
    "web-search-engine-brave": ("engines", "brave"),
    "web-search-engine-yandex": ("engines", "yandex"),
    "web-search-engine-legacy-yandex": ("engines", "legacy_yandex"),
    "web-search-image-engine-bing": ("image_engines", "bing"),
    "web-search-image-engine-brave": ("image_engines", "brave"),
    "web-search-image-engine-duckduckgo": ("image_engines", "duckduckgo"),
    "web-search-image-engine-openverse": ("image_engines", "openverse"),
    "web-search-query-rewriter-enabled": ("query_rewriter", "enabled"),
    "web-search-query-rewriter-endpoint": ("query_rewriter", "endpoint"),
    "web-search-query-rewriter-model": ("query_rewriter", "model"),
    "web-search-query-rewriter-mode": ("query_rewriter", "mode"),
    "web-search-localization-enabled": ("localization", "enabled"),
    "web-search-localization-penalty": ("localization", "language_penalty"),
    "web-search-rich-cards-enabled": ("rich_cards", "enabled"),
    "web-search-rich-cards-wikipedia": ("rich_cards", "wikipedia"),
    "web-search-rich-cards-stackexchange": ("rich_cards", "stackexchange"),
    "web-search-rich-cards-reddit": ("rich_cards", "reddit"),
    "web-search-profile-import-enabled": ("profile_import", "enabled"),
    "web-search-profile-import-all-profiles": ("profile_import", "all_profiles"),
    "web-search-profile-import-refresh-hours": ("profile_import", "refresh_hours"),
    "web-search-profile-import-purge-on-disable": ("profile_import", "purge_on_disable"),
}

WEB_SEARCH_STATIC_CONFIG = {
    "profile_import": {
        "browsers": ["chrome", "edge", "brave", "firefox"],
        "domains": [
            "google.com",
            "bing.com",
            "duckduckgo.com",
            "startpage.com",
            "yandex.com",
            "yandex.ru",
            "qwant.com",
            "brave.com",
            "search.brave.com",
            "reddit.com",
        ],
    }
}

BROWSER_AGENT_CONFIG_SETTINGS = {
    "browser-agent-width": "browser_width",
    "browser-agent-height": "browser_height",
    "browser-agent-headless": "browser_headless",
    "browser-agent-max-a11y-depth": "max_a11y_depth",
    "browser-agent-max-elements": "max_elements",
    "browser-agent-max-main-interactive": "max_main_interactive",
    "browser-agent-auto-text-preview-length": "auto_text_preview_length",
}

SANDBOX_CONFIG_SETTINGS = {
    "sandbox-container-name": "SANDBOX_CONTAINER_NAME",
    "sandbox-image": "SANDBOX_IMAGE",
    "sandbox-cpu-limit": "SANDBOX_CPU_LIMIT",
    "sandbox-memory-limit": "SANDBOX_MEMORY_LIMIT",
    "sandbox-memory-swap-limit": "SANDBOX_MEMORY_SWAP_LIMIT",
    "sandbox-pids-limit": "SANDBOX_PIDS_LIMIT",
    "sandbox-storage-limit": "SANDBOX_STORAGE_LIMIT",
    "sandbox-network-enabled": "SANDBOX_NETWORK_ENABLED",
    "sandbox-network-limit-mbit": "SANDBOX_NETWORK_LIMIT_MBIT",
    "sandbox-default-timeout": "SANDBOX_DEFAULT_TIMEOUT",
    "sandbox-max-output-bytes": "SANDBOX_MAX_OUTPUT_BYTES",
    "sandbox-output-head-ratio": "SANDBOX_OUTPUT_HEAD_RATIO",
    "sandbox-max-read-bytes": "SANDBOX_MAX_READ_BYTES",
    "sandbox-max-cat-file-bytes": "SANDBOX_MAX_CAT_FILE_BYTES",
    "sandbox-max-cat-line-threshold": "SANDBOX_MAX_CAT_LINE_THRESHOLD",
    "sandbox-max-image-preview-bytes": "SANDBOX_MAX_IMAGE_PREVIEW_BYTES",
    "sandbox-max-ls-entries": "SANDBOX_MAX_LS_ENTRIES",
    "sandbox-max-find-results": "SANDBOX_MAX_FIND_RESULTS",
    "sandbox-max-grep-results": "SANDBOX_MAX_GREP_RESULTS",
    "sandbox-background-timeout-threshold": "SANDBOX_BACKGROUND_TIMEOUT_THRESHOLD",
    "sandbox-thread-limit": "SANDBOX_THREAD_LIMIT",
    "sandbox-default-task-dir": "SANDBOX_DEFAULT_TASK_DIR",
    "sandbox-workspace-cleanup-enabled": "SANDBOX_WORKSPACE_CLEANUP_ENABLED",
    "sandbox-workspace-cleanup-idle-seconds": "SANDBOX_WORKSPACE_CLEANUP_IDLE_SECONDS",
    "sandbox-workspace-cleanup-recycle-seconds": "SANDBOX_WORKSPACE_CLEANUP_RECYCLE_SECONDS",
    "sandbox-workspace-cleanup-interval-seconds": "SANDBOX_WORKSPACE_CLEANUP_INTERVAL_SECONDS",
    "sandbox-max-file-map-symbols": "SANDBOX_MAX_FILE_MAP_SYMBOLS",
    "sandbox-docker-start-timeout-seconds": "SANDBOX_DOCKER_START_TIMEOUT_SECONDS",
}

SANDBOX_STATIC_CONFIG = {
    "SANDBOX_IMAGE_SOURCE": "registry",
    "SANDBOX_AUTO_START_DOCKER": True,
}

SANDBOX_GIGABYTE_SETTINGS = frozenset(
    {
        "sandbox-memory-limit",
        "sandbox-memory-swap-limit",
        "sandbox-storage-limit",
    }
)

DECIMAL_SETTING_KEYS = frozenset(
    {
        "web-search-prefetch-timeout",
        "web-search-extraction-timeout",
        "web-search-profile-import-refresh-hours",
        "sandbox-cpu-limit",
        "sandbox-output-head-ratio",
    }
)

REMOVED_TOOL_SETTING_KEYS = frozenset(
    {
        "web-search-profile-import-browsers",
        "web-search-profile-import-domains",
        "sandbox-image-source",
        "sandbox-auto-start-docker",
    }
)

DEFAULTS: dict[str, Any] = {
    "ui-port": 20000,
    "debug": True,
    "console_log_level": "debug",
    "generate-chat-titles": True,
    "secret_key": "",
    "allowed_hosts": ["127.0.0.1", "localhost"],
    "llm-engine": "ollama-service",
    "ollama-service_port": 20003,
    "browser-daemon-port": 20010,
    "ollama-service": False,
    "ollama-service_path": None,
    "ollama-service_data": None,
    "ollama-service_models": None,
    "lms": False,
    "lms_url": "127.0.0.1:1234",
    "openai": False,
    "openai_url": "127.0.0.1:8000/v1",
    "openai_api_key": "",
    "google-genai": False,
    "google_genai_url": "generativelanguage.googleapis.com",
    "google_genai_api_key": "",
    "key-aslm": None,
    "key-gh": None,
    "web-search-tavily-api-key": "",
    "web-search-firecrawl-api-key": "",
    "web-search-brave-api-key": "",
    "web-search-serpapi-api-key": "",
    "web-search-max-results": 10,
    "web-search-total-context-budget": 40000,
    "web-search-prefetch-timeout": 8.0,
    "web-search-preview-max-chars": 4000,
    "web-search-extraction-timeout": 25.0,
    "web-search-max-page-chars": 20000,
    "web-search-min-content-length": 800,
    "web-search-compress-read-pages": True,
    "web-search-compress-threshold": 10000,
    "web-search-compress-target": 10000,
    "web-search-cache-ttl": 21600,
    "web-search-negative-cache-ttl": 300,
    "web-search-page-cache-ttl": 86400,
    "web-search-repeat-block-window": 30,
    "web-search-seen-source-window": 30,
    "web-search-prefetch-max-urls": 4,
    "web-search-tor-enabled": False,
    "web-search-engine-google": True,
    "web-search-engine-duckduckgo": True,
    "web-search-engine-startpage": True,
    "web-search-engine-qwant": True,
    "web-search-engine-brave": True,
    "web-search-engine-yandex": False,
    "web-search-engine-legacy-yandex": False,
    "web-search-image-engine-bing": True,
    "web-search-image-engine-brave": True,
    "web-search-image-engine-duckduckgo": False,
    "web-search-image-engine-openverse": True,
    "web-search-query-rewriter-enabled": False,
    "web-search-query-rewriter-endpoint": "",
    "web-search-query-rewriter-model": "",
    "web-search-query-rewriter-mode": "heuristic",
    "web-search-localization-enabled": True,
    "web-search-localization-penalty": 0.12,
    "web-search-rich-cards-enabled": False,
    "web-search-rich-cards-wikipedia": True,
    "web-search-rich-cards-stackexchange": True,
    "web-search-rich-cards-reddit": True,
    "web-search-profile-import-enabled": False,
    "web-search-profile-import-all-profiles": False,
    "web-search-profile-import-refresh-hours": 12.0,
    "web-search-profile-import-purge-on-disable": True,
    "browser-agent-width": 1280,
    "browser-agent-height": 800,
    "browser-agent-headless": False,
    "browser-agent-max-a11y-depth": 15,
    "browser-agent-max-elements": 200,
    "browser-agent-max-main-interactive": 60,
    "browser-agent-auto-text-preview-length": 1500,
    "sandbox-container-name": "aslm-chat-sandbox",
    "sandbox-image": "nggtlightkeeper/aslm-chat-sandbox:latest",
    "sandbox-cpu-limit": 8,
    "sandbox-memory-limit": 6,
    "sandbox-memory-swap-limit": 8,
    "sandbox-pids-limit": 1024,
    "sandbox-storage-limit": 12,
    "sandbox-network-enabled": True,
    "sandbox-network-limit-mbit": 0,
    "sandbox-default-timeout": 60,
    "sandbox-max-output-bytes": 60000,
    "sandbox-output-head-ratio": 0.5,
    "sandbox-max-read-bytes": 200000,
    "sandbox-max-cat-file-bytes": 30720,
    "sandbox-max-cat-line-threshold": 300,
    "sandbox-max-image-preview-bytes": 2000000,
    "sandbox-max-ls-entries": 500,
    "sandbox-max-find-results": 200,
    "sandbox-max-grep-results": 200,
    "sandbox-background-timeout-threshold": 10,
    "sandbox-thread-limit": 8,
    "sandbox-default-task-dir": "_sandbox",
    "sandbox-workspace-cleanup-enabled": True,
    "sandbox-workspace-cleanup-idle-seconds": 5400,
    "sandbox-workspace-cleanup-recycle-seconds": 10800,
    "sandbox-workspace-cleanup-interval-seconds": 5,
    "sandbox-max-file-map-symbols": 50,
    "sandbox-docker-start-timeout-seconds": 60,
}

CONSOLE_LOG_LEVELS = {"basic", "debug", "trace"}
NORMALIZED_ADDRESS_KEYS = {"lms_url", "openai_url", "google_genai_url"}
IGNORED_ENV_KEYS = {"ASLM_MODULE_ID", "ASLM_MODULE_DIR"}

_settings_cache_lock = threading.RLock()
_settings_cache: dict[str, Any] | None = None
_settings_cache_mtime_ns: int | None = None


# Return the current settings file mtime.
def _get_settings_mtime_ns() -> int | None:
    try:
        return SETTINGS_FILE.stat().st_mtime_ns
    except OSError:
        return None


# Store one effective settings snapshot in memory.
def _store_settings_cache(data: dict[str, Any], mtime_ns: int | None) -> None:
    global _settings_cache, _settings_cache_mtime_ns

    with _settings_cache_lock:
        _settings_cache = dict(data)
        _settings_cache_mtime_ns = mtime_ns


# Invalidate the in-memory settings snapshot.
def _invalidate_settings_cache() -> None:
    global _settings_cache, _settings_cache_mtime_ns

    with _settings_cache_lock:
        _settings_cache = None
        _settings_cache_mtime_ns = None


# Normalize one raw settings value.
def normalize_setting_value(value: Any) -> Any:
    if not isinstance(value, str):
        return value

    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    if value.count(",") == 1 and "." not in value:
        try:
            return float(value.replace(",", "."))
        except ValueError:
            pass

    if value.startswith(("{", "[")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    return value


# Normalize one raw settings key.
def normalize_setting_key(raw_key: str) -> str:
    key = raw_key.strip().lower()
    if key in DEFAULTS:
        return key

    dashed = key.replace("_", "-")
    if dashed in DEFAULTS:
        return dashed

    # A few historical keys intentionally mix dashes and underscores (for
    # example ``ollama-service_port``). Environment variable names cannot
    # preserve that distinction, so compare a separator-normalized form too.
    for canonical_key in DEFAULTS:
        if canonical_key.replace("_", "-") == dashed:
            return canonical_key

    underscored = key.replace("-", "_")
    if underscored in DEFAULTS:
        return underscored

    return key


# Normalize one engine address for storage.
def normalize_engine_address(value: Any) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""

    parsed = urlparse(raw_value)
    if parsed.scheme and parsed.netloc:
        return f"{parsed.netloc}{parsed.path}".rstrip("/")

    if parsed.scheme and parsed.path:
        return parsed.path.rstrip("/")

    return raw_value.rstrip("/")


# Normalize one engine identifier.
def normalize_engine_name(engine: str | None) -> str:
    if not engine:
        return ENGINE_ALIASES["ollama-service"]

    normalized = str(engine).strip().lower()
    return ENGINE_ALIASES.get(normalized, normalized)


# List enabled engine identifiers from one settings snapshot.
def _get_enabled_engine_ids_from_settings(settings_data: dict[str, Any]) -> list[str]:
    return [engine_id for engine_id in ENGINE_IDS if bool(settings_data.get(engine_id, False))]


# Resolve one engine against the enabled engine list.
def _resolve_enabled_engine_from_settings(
    settings_data: dict[str, Any],
    engine: str | None,
    default: str = "ollama-service",
) -> str:
    canonical = normalize_engine_name(engine or default)
    enabled_engine_ids = _get_enabled_engine_ids_from_settings(settings_data)

    if canonical in enabled_engine_ids:
        return canonical
    if enabled_engine_ids:
        return enabled_engine_ids[0]

    return canonical



# List engines supported by the UI.
def get_supported_engines() -> list[dict[str, str]]:
    from Apps.UI.locale_catalog import translate

    return [
        {
            "id": engine_id,
            "label": translate(f"engines.{engine_id}", fallback=ENGINE_LABELS[engine_id]),
        }
        for engine_id in get_enabled_engine_ids()
    ]


# List enabled engine identifiers from the effective settings.
def get_enabled_engine_ids() -> list[str]:
    return _get_enabled_engine_ids_from_settings(load_settings())


# Resolve one requested engine against the current enabled engine list.
def resolve_enabled_engine(engine: str | None, default: str = "ollama-service") -> str:
    return _resolve_enabled_engine_from_settings(load_settings(), engine, default)



# Read the settings payload from disk.
def _load_settings_from_disk() -> dict[str, Any]:
    if not SETTINGS_FILE.exists():
        return {}

    try:
        with SETTINGS_FILE.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", SETTINGS_FILE, exc)
        return {}

    return data if isinstance(data, dict) else {}


# Apply environment overrides to one settings snapshot.
def _apply_environment_overrides(data: dict[str, Any]) -> dict[str, Any]:
    updated = dict(data)

    for env_key, env_value in os.environ.items():
        if not env_key.startswith("ASLM_") or env_key in IGNORED_ENV_KEYS:
            continue

        setting_key = normalize_setting_key(env_key[5:])
        updated[setting_key] = normalize_setting_value(env_value)

    return updated


_PORT_SETTING_KEYS = ("ui-port", "ollama-service_port", "browser-daemon-port")


# Log a warning when two services share the same TCP port.
def _warn_port_collisions(settings: dict[str, Any]) -> None:
    by_port: dict[int, list[str]] = {}
    for key in _PORT_SETTING_KEYS:
        raw = settings.get(key)
        try:
            port = int(raw)
        except (TypeError, ValueError):
            continue
        if port <= 0 or port > 65535:
            continue
        by_port.setdefault(port, []).append(key)
    for port, keys in sorted(by_port.items()):
        if len(keys) > 1:
            logger.warning(
                "Settings port collision on %s: %s — assign unique ports in Settings/settings.json",
                port,
                ", ".join(keys),
            )


# Normalize a loaded settings snapshot (addresses, active engine, port checks).
def _normalize_loaded_settings(data: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(data)
    normalized.pop("lms_load_config", None)

    for key in REMOVED_TOOL_SETTING_KEYS:
        normalized.pop(key, None)

    for key in NORMALIZED_ADDRESS_KEYS:
        normalized[key] = normalize_engine_address(normalized.get(key, DEFAULTS.get(key, "")))

    for key in SANDBOX_GIGABYTE_SETTINGS:
        normalized[key] = _normalize_gigabyte_value(normalized.get(key), DEFAULTS[key])

    for key in DECIMAL_SETTING_KEYS:
        normalized[key] = _normalize_decimal_setting_value(normalized.get(key), DEFAULTS[key])

    normalized["llm-engine"] = _resolve_enabled_engine_from_settings(
        normalized,
        normalized.get("llm-engine"),
        DEFAULTS["llm-engine"],
    )
    _warn_port_collisions(normalized)
    return normalized


# Load the effective settings snapshot.
def load_settings() -> dict[str, Any]:
    mtime_ns = _get_settings_mtime_ns()
    with _settings_cache_lock:
        if _settings_cache is not None and _settings_cache_mtime_ns == mtime_ns:
            return dict(_settings_cache)

    settings_data = dict(DEFAULTS)
    settings_data.update(_load_settings_from_disk())
    settings_data = _apply_environment_overrides(settings_data)
    settings_data = _normalize_loaded_settings(settings_data)
    _store_settings_cache(settings_data, mtime_ns)
    return dict(settings_data)


# Serialize and atomically replace one generated JSON configuration when its content changed.
def _write_generated_json(path: Path, data: dict[str, Any]) -> None:
    serialized = json.dumps(data, indent=4, ensure_ascii=False) + "\n"
    try:
        if path.read_text(encoding="utf-8-sig") == serialized:
            return
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning("Could not compare generated config %s: %s", path, exc)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    try:
        temporary_path.write_text(serialized, encoding="utf-8")
        os.replace(temporary_path, path)
    except Exception:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise


# Convert a stored optional API key to the nullable value expected by tool configs.
def _nullable_api_key(value: Any) -> str | None:
    text = str(value or "").strip()
    return text or None


# Normalize a positive integer gigabyte setting.
def _normalize_gigabyte_value(value: Any, default: int) -> int:
    normalized = normalize_setting_value(value)
    if isinstance(normalized, bool) or not isinstance(normalized, int):
        return default
    if normalized <= 0:
        return default
    return normalized


# Format a numeric gigabyte setting for Docker and sandbox.env.
def _format_gigabytes(value: Any, default: int) -> str:
    gigabytes = _normalize_gigabyte_value(value, default)
    return f"{gigabytes}G"


# Normalize a decimal setting received with either dot or comma separator.
def _normalize_decimal_setting_value(value: Any, default: int | float) -> float:
    normalized = normalize_setting_value(value)
    if isinstance(normalized, bool) or not isinstance(normalized, (int, float)):
        return float(default)
    return float(normalized)


# Generate the hosted web-search API key document from the effective module settings.
def _write_web_search_api_keys(settings_data: dict[str, Any]) -> None:
    hosted_api = {
        config_key: _nullable_api_key(settings_data.get(setting_key))
        for setting_key, config_key in WEB_SEARCH_API_KEY_SETTINGS.items()
    }
    _write_generated_json(
        WEB_SEARCH_API_KEYS_FILE,
        {"search": {"hosted_api": hosted_api}},
    )


# Read the existing web-search configuration so JSON-only sections survive synchronization.
def _load_web_search_config_payload() -> dict[str, Any]:
    try:
        payload = json.loads(WEB_SEARCH_CONFIG_FILE.read_text(encoding="utf-8-sig"))
    except (FileNotFoundError, json.JSONDecodeError, OSError) as exc:
        if not isinstance(exc, FileNotFoundError):
            logger.warning("Could not read %s: %s", WEB_SEARCH_CONFIG_FILE, exc)
        return {}

    return payload if isinstance(payload, dict) else {}


# Generate managed search_config.json values without replacing JSON-only sections.
def _write_web_search_config(settings_data: dict[str, Any]) -> None:
    payload = _load_web_search_config_payload()
    for section_name, section_values in WEB_SEARCH_STATIC_CONFIG.items():
        payload.setdefault(section_name, {}).update(section_values)

    for setting_key, (section_name, config_key) in WEB_SEARCH_CONFIG_SETTINGS.items():
        value = settings_data.get(setting_key, DEFAULTS[setting_key])
        payload.setdefault(section_name, {})[config_key] = value

    _write_generated_json(WEB_SEARCH_CONFIG_FILE, payload)


# Generate the browser-agent runtime configuration from the effective module settings.
def _write_browser_agent_config(settings_data: dict[str, Any]) -> None:
    payload = {
        config_key: settings_data.get(setting_key, DEFAULTS[setting_key])
        for setting_key, config_key in BROWSER_AGENT_CONFIG_SETTINGS.items()
    }
    _write_generated_json(BROWSER_AGENT_CONFIG_FILE, payload)


# Generate the sandbox JSON bridge consumed before sandbox.env is loaded.
def _write_sandbox_config(settings_data: dict[str, Any]) -> None:
    payload: dict[str, Any] = dict(SANDBOX_STATIC_CONFIG)
    for setting_key, config_key in SANDBOX_CONFIG_SETTINGS.items():
        value = settings_data.get(setting_key, DEFAULTS[setting_key])
        if setting_key in SANDBOX_GIGABYTE_SETTINGS:
            value = _format_gigabytes(value, DEFAULTS[setting_key])
        payload[config_key] = value
    _write_generated_json(SANDBOX_CONFIG_FILE, payload)


# Synchronize generated tool configs affected by one setting, or every config during bootstrap.
def _sync_tool_configs(settings_data: dict[str, Any], setting_key: str | None = None) -> None:
    effective = dict(DEFAULTS)
    effective.update(settings_data)
    effective = _apply_environment_overrides(effective)
    effective = _normalize_loaded_settings(effective)

    if setting_key is None or setting_key in WEB_SEARCH_API_KEY_SETTINGS:
        _write_web_search_api_keys(effective)
    if setting_key is None or setting_key in WEB_SEARCH_CONFIG_SETTINGS:
        _write_web_search_config(effective)
    if setting_key is None or setting_key in BROWSER_AGENT_CONFIG_SETTINGS:
        _write_browser_agent_config(effective)
    if setting_key is None or setting_key in SANDBOX_CONFIG_SETTINGS:
        _write_sandbox_config(effective)


# Save the settings snapshot and optionally refresh all generated tool configs.
def save_settings(data: dict[str, Any], *, sync_tool_configs: bool = True) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with SETTINGS_FILE.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, ensure_ascii=False)
    if sync_tool_configs:
        _sync_tool_configs(data)
    settings_data = dict(DEFAULTS)
    settings_data.update(data)
    settings_data = _apply_environment_overrides(settings_data)
    _store_settings_cache(_normalize_loaded_settings(settings_data), _get_settings_mtime_ns())



# Build one runtime environment variable name.
def _to_env_var_name(key: str) -> str:
    return f"ASLM_{key.replace('-', '_').upper()}"


# Serialize one value for environment storage.
def _serialize_env_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)

    return str(value)


# Apply one runtime setting to the current process environment.
def _apply_process_environment_value(key: str, value: Any) -> None:
    env_key = _to_env_var_name(key)
    serialized = _serialize_env_value(value)
    if serialized:
        os.environ[env_key] = serialized
    elif env_key in os.environ:
        del os.environ[env_key]


# Load stored settings without applying ASLM_ environment overrides.
def _load_stored_settings_snapshot() -> dict[str, Any]:
    settings_data = dict(DEFAULTS)
    settings_data.update(_load_settings_from_disk())
    return _normalize_loaded_settings(settings_data)


# Locate the ASLM module manifest when available.
def _get_module_manifest_path() -> Path | None:
    module_dir = os.environ.get("ASLM_MODULE_DIR", "").strip()
    if module_dir:
        manifest_path = Path(module_dir) / "ASLM_Module.json"
        if manifest_path.exists():
            return manifest_path

    manifest_path = BASE_DIR / "ASLM_Module.json"
    return manifest_path if manifest_path.exists() else None


# Mirror one runtime setting into the module manifest.
def _sync_module_manifest_setting(key: str, value: Any) -> None:
    manifest_path = _get_module_manifest_path()
    if manifest_path is None:
        return

    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read %s: %s", manifest_path, exc)
        return

    settings_list = manifest.get("settings")
    if not isinstance(settings_list, list):
        return

    changed = False
    for setting_item in settings_list:
        if not isinstance(setting_item, dict):
            continue
        if setting_item.get("key") != key:
            continue

        if setting_item.get("value") == value:
            return

        setting_item["value"] = value
        changed = True
        break

    if not changed:
        return

    try:
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=4, ensure_ascii=False)
            file.write("\n")
    except OSError as exc:
        logger.warning("Could not write %s: %s", manifest_path, exc)



# Read one setting value.
def get(key: str, default: Any = None) -> Any:
    return load_settings().get(key, default)


# Persist one setting value.
def set(key: str, value: Any, *, sync_runtime: bool = True) -> None:
    if key == "llm-engine":
        value = resolve_enabled_engine(str(value) if value is not None else None)

    if key in NORMALIZED_ADDRESS_KEYS:
        value = normalize_engine_address(value)

    if key in SANDBOX_GIGABYTE_SETTINGS:
        value = _normalize_gigabyte_value(value, DEFAULTS[key])

    if key in DECIMAL_SETTING_KEYS:
        value = _normalize_decimal_setting_value(value, DEFAULTS[key])

    stored_raw_data = _load_settings_from_disk()
    stored_data = dict(DEFAULTS)
    stored_data.update(stored_raw_data)
    stored_data = _normalize_loaded_settings(stored_data)
    if key in stored_raw_data and stored_data.get(key, DEFAULTS.get(key)) == value:
        _sync_tool_configs(stored_data, key)
        _apply_process_environment_value(key, value)
        _store_settings_cache(_apply_environment_overrides(stored_data), _get_settings_mtime_ns())
        return

    # Save the normalized value to disk first.
    data = load_settings()
    data[key] = value
    save_settings(data, sync_tool_configs=False)
    _sync_tool_configs(data, key)

    # Keep the current process environment in sync with the saved value.
    _apply_process_environment_value(key, value)
    _invalidate_settings_cache()

    # Host account keys belong only to the local settings file, not the module manifest.
    if key not in HOST_KEY_SETTING_KEYS:
        _sync_module_manifest_setting(key, value)

    if sync_runtime and key in ENGINE_IDS:
        try:
            from API import llm_api

            llm_api.sync_enabled_engine_runtimes()
        except Exception as exc:
            logger.warning("Failed to sync engine runtimes after setting %s: %s", key, exc)



# Read the active LLM engine name.
def get_llm_engine(default: str = "ollama-service") -> str:
    configured = get("llm-engine", default)
    return normalize_engine_name(configured)


# Resolve the settings key for one engine URL.
def get_engine_url_key(engine: str | None) -> str | None:
    canonical = normalize_engine_name(engine)
    return ENGINE_URL_KEYS.get(canonical)


# Infer a scheme for remote endpoints without one.
def _infer_remote_scheme(value: str) -> str:
    endpoint = str(value or "").strip()
    if not endpoint:
        return "http"

    host_part = endpoint.split("/", 1)[0].strip()
    host_name = host_part.split(":", 1)[0].strip().lower()
    if host_name in {"localhost", "127.0.0.1", "::1"}:
        return "http"

    return "https"


# Build the effective engine URL.
def get_engine_url(engine: str | None) -> str:
    canonical = normalize_engine_name(engine)

    if canonical == "ollama-service":
        port = int(get("ollama-service_port", DEFAULTS["ollama-service_port"]))
        return f"http://127.0.0.1:{port}"

    url_key = get_engine_url_key(canonical)
    if not url_key:
        return ""

    value = normalize_engine_address(get(url_key, DEFAULTS.get(url_key, "")) or "")
    if canonical in {"openai", "google-genai"} and value and "://" not in value:
        return f"{_infer_remote_scheme(value)}://{value}"

    return value


# Read the OpenAI-compatible API key.
def get_openai_api_key() -> str:
    configured = get("openai_api_key", "") or os.environ.get("OPENAI_API_KEY", "")
    return str(configured).strip()


# Read the Google GenAI API key.
def get_google_genai_api_key() -> str:
    configured = (
        get("google_genai_api_key", "")
        or os.environ.get("GOOGLE_API_KEY", "")
        or os.environ.get("GEMINI_API_KEY", "")
    )
    return str(configured).strip()


# Resolve the settings key for one engine API key.
def get_engine_api_key_key(engine: str | None) -> str | None:
    canonical = normalize_engine_name(engine)
    return ENGINE_API_KEY_KEYS.get(canonical)


# Read the configured API key for one engine.
def get_engine_api_key(engine: str | None) -> str:
    canonical = normalize_engine_name(engine)
    if canonical == "openai":
        return get_openai_api_key()
    if canonical == "google-genai":
        return get_google_genai_api_key()
    return ""


# Build the runtime settings payload for the UI.
def get_runtime_engine_settings() -> dict[str, Any]:
    openai_api_key = get_openai_api_key()
    google_genai_api_key = get_google_genai_api_key()
    active_engine = get_llm_engine()
    engine_api_keys = {
        "ollama-service": False,
        "lms": False,
        "openai": bool(openai_api_key),
        "google-genai": bool(google_genai_api_key),
    }

    return {
        "llm-engine": active_engine,
        "console_log_level": get_console_log_level(),
        "lms_url": normalize_engine_address(get("lms_url", DEFAULTS["lms_url"])),
        "openai_url": normalize_engine_address(get("openai_url", DEFAULTS["openai_url"])),
        "google_genai_url": normalize_engine_address(get("google_genai_url", DEFAULTS["google_genai_url"])),
        "has_openai_api_key": bool(openai_api_key),
        "has_google_genai_api_key": bool(google_genai_api_key),
        "active_has_api_key": bool(engine_api_keys.get(active_engine, False)),
        "engine_api_keys": engine_api_keys,
        "engine_api_key_keys": dict(ENGINE_API_KEY_KEYS),
        "engine_urls": {
            "ollama-service": get_engine_url("ollama-service"),
            "lms": get_engine_url("lms"),
            "openai": get_engine_url("openai"),
            "google-genai": get_engine_url("google-genai"),
        },
    }



# Read the console log level.
def get_console_log_level(default: str = "debug") -> str:
    configured = str(get("console_log_level", default) or default).strip().lower()
    return configured if configured in CONSOLE_LOG_LEVELS else default


# Check whether debug console output is enabled.
def is_console_debug_enabled() -> bool:
    return get_console_log_level() in {"debug", "trace"}


# Check whether trace console output is enabled.
def is_console_trace_enabled() -> bool:
    return get_console_log_level() == "trace"



# Check whether one engine is enabled.
def is_engine_enabled(engine: str | None) -> bool:
    canonical = normalize_engine_name(engine)
    return bool(get(canonical, False))


# Check whether the engine uses the Ollama adapter path.
def is_ollama_engine(engine: str | None) -> bool:
    return normalize_engine_name(engine) == "ollama-service"
