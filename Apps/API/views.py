# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import base64
import binascii
import copy
import hashlib
import json
import logging
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from django.http import JsonResponse, StreamingHttpResponse
from django.utils import timezone

from API import llm_api, mcp as tool_registry
from API import ollama as ollama_api
from Apps.Data import model_capabilities
from Apps.Data.ollama_presets import get_ollama_preset_payload
from Apps.Data.lms_presets import get_lms_preset_payload
from Apps.Data.openai_presets import get_openai_preset_payload
from Apps.Data.google_genai_presets import get_google_genai_preset_payload
from Apps.Data.models import (
    Chat,
    Message,
    MessageAttachment,
    MessageAttachmentKind,
    MessageImage,
)
from Services.upload_storage import (
    load_upload_manifest,
    model_upload_payload,
    public_upload_payload,
    resolve_uploaded_file_host_path,
)
from Settings import settings
from Settings import skills as skills_config
from Tools.system_prompts import get_system_prompt, is_instant_generation

logger = logging.getLogger(__name__)


TOOLS_PATH = settings.BASE_DIR / "Tools"
if str(TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(TOOLS_PATH))


from context_compression.history_compressor import (  # noqa: E402
    build_structured_history_summary,
    decide_compression,
    fit_summary_text,
    resolve_context_window_tokens,
)

THINK_PARAM_NAMES = {"think", "thinking", "reasoning"}
THINK_LEVEL_PARAM_NAMES = {"think_level", "thinking_level", "reasoning_effort"}
TOOL_CAPABILITY_NAMES = {"tools", "tool", "tool-calling", "tool_calling"}
OLLAMA_TOOL_TEMPLATE_MARKERS = (
    ".Tools",
    ".ToolCalls",
    "tool_calls",
    "tool_call_id",
    "<tool_call",
    "[TOOL_CALLS]",
    "available_tools",
    "function_call",
)
TEXT_ATTACHMENT_EXTENSIONS = {
    ".adoc", ".ahk", ".asm", ".asciidoc", ".bash", ".bat", ".bib", ".c", ".cc", ".cfg",
    ".clj", ".cljc", ".cljs", ".cmd", ".cmake", ".conf", ".config", ".cpp", ".cs",
    ".cshtml", ".csproj", ".css", ".csv", ".cts", ".cue", ".cxx", ".dart", ".diff",
    ".dockerfile", ".edn", ".ejs", ".env", ".erb", ".erl", ".ex", ".exs", ".fish",
    ".fs", ".fsi", ".fsx", ".geojson", ".go", ".gql", ".gradle", ".graphql", ".groovy",
    ".gvy", ".h", ".handlebars", ".haml", ".hbs", ".hpp", ".hrl", ".hs", ".htm", ".html",
    ".http", ".hxx", ".ini", ".java", ".jinja", ".jinja2", ".jl", ".js", ".json",
    ".json5", ".jsonc", ".jsonl", ".jsx", ".kt", ".kts", ".ksh", ".latex", ".less",
    ".liquid", ".log", ".lua", ".m", ".mako", ".markdown", ".md", ".mdown", ".mdx",
    ".mjml", ".mkd", ".mm", ".mod", ".mts", ".mustache", ".nix", ".njk", ".patch",
    ".php", ".pl", ".pm", ".pod", ".properties", ".proto", ".ps1", ".ps1xml", ".psd1",
    ".psm1", ".pug", ".py", ".pyi", ".r", ".rb", ".rego", ".rest", ".rst", ".rs",
    ".sass", ".scala", ".scss", ".service", ".sh", ".shtml", ".sol", ".sql", ".srt",
    ".styl", ".svelte", ".sum", ".svg", ".swift", ".tcl", ".templ", ".tex", ".tf",
    ".tfvars", ".toml", ".ts", ".tsv", ".tsx", ".twig", ".txt", ".vb", ".vbs", ".vue",
    ".vtt", ".xhtml", ".xml", ".xsd", ".xsl", ".yaml", ".yml", ".zsh",
}


# Build dynamic runtime context injected into every system prompt.
def _build_runtime_context() -> str:
    from datetime import datetime
    now = datetime.now()
    return f"Today is {now.strftime('%A, %Y-%m-%d')}."


# Return whether the chat has not yet persisted a user message.
def _chat_is_first_user_turn(chat: Chat | None) -> bool:
    if chat is None:
        return True
    return not chat.messages.filter(role="user").exists()


# Merge the project prompt with per-request user instructions.
def _compose_system_prompt(
    user_system_prompt: str,
    *,
    consume_skill_notifications: bool = True,
    include_skills_baseline: bool = False,
    forced_skill_names: list[str] | None = None,
    instant_mode: bool = False,
) -> str:
    parts: list[str] = []

    default_prompt = get_system_prompt(instant_mode=instant_mode)
    if default_prompt:
        parts.append(default_prompt)

    runtime_context = _build_runtime_context()
    if runtime_context:
        parts.append(runtime_context)

    try:
        skill_delta = skills_config.build_system_prompt_skills_section(
            consume=consume_skill_notifications,
            include_baseline=include_skills_baseline,
        )
    except Exception:
        logger.exception("Failed to build skills prompt delta")
        skill_delta = ""
    if skill_delta:
        parts.append(skill_delta)

    forced_skill_section = _build_forced_skill_prompt_section(forced_skill_names or [])
    if forced_skill_section:
        parts.append(forced_skill_section)

    user_prompt = str(user_system_prompt or "").strip()
    if user_prompt:
        parts.append(f"Additional instructions:\n{user_prompt}")

    return "\n\n".join(parts)


# Build a deterministic system prompt block for skills explicitly selected by a slash command.
def _build_forced_skill_prompt_section(skill_names: list[str]) -> str:
    unique_names: list[str] = []
    seen: set[str] = set()
    for raw_name in skill_names or []:
        name = str(raw_name or "").strip()
        key = name.casefold()
        if name and key not in seen:
            unique_names.append(name)
            seen.add(key)
    if not unique_names:
        return ""

    try:
        payload = skills_config.list_skills()
    except Exception:
        logger.exception("Failed to list skills for forced injection")
        return ""

    root = Path(str(payload.get("root") or skills_config.SKILLS_DIR))
    folder_by_key = {
        str(folder.get("name") or "").casefold(): folder
        for folder in payload.get("folders", [])
        if isinstance(folder, dict)
    }
    sections = [
        "Forced skill instructions:",
        "",
        "The user explicitly selected these skill(s) for this request. Follow the included instructions.",
    ]

    for name in unique_names:
        folder = folder_by_key.get(name.casefold())
        if not folder:
            sections.append(f"\nSkill '{name}' was requested but is not installed.")
            continue
        if folder.get("enabled") is False:
            sections.append(f"\nSkill '{name}' was requested but is disabled.")
            continue

        folder_name = str(folder.get("name") or name)
        primary = str(folder.get("primary_file") or "SKILL.md").strip() or "SKILL.md"
        skill_root = (root / folder_name).resolve()
        skill_path = (skill_root / primary).resolve()
        try:
            skill_path.relative_to(skill_root)
            content = skill_path.read_text(encoding="utf-8").strip()
        except Exception:
            logger.exception("Failed to inject forced skill %s", name)
            content = ""
        if content:
            sections.append(f"\n--- Skill: {folder_name}/{primary} ---\n{content}")

    return "\n".join(sections).strip()


TEXT_ATTACHMENT_FILENAMES = {
    ".bash_profile",
    ".bashrc",
    ".dockerignore",
    ".editorconfig",
    ".env",
    ".env.example",
    ".eslintignore",
    ".eslintrc",
    ".gitattributes",
    ".gitconfig",
    ".gitignore",
    ".gitkeep",
    ".gitmodules",
    ".npmrc",
    ".nvmrc",
    ".prettierignore",
    ".prettierrc",
    ".python-version",
    ".stylelintrc",
    ".tool-versions",
    ".yamllint",
    ".zshenv",
    ".zshrc",
    "brewfile",
    "cmakelists.txt",
    "containerfile",
    "dockerfile",
    "gemfile",
    "jenkinsfile",
    "justfile",
    "makefile",
    "procfile",
    "rakefile",
    "tiltfile",
    "vagrantfile",
}
ATTACHMENT_TEXT_CHAR_LIMIT = 24000
LLM_HISTORY_MAX_MESSAGES = 40
LLM_HISTORY_MIN_MESSAGES = 8
LLM_HISTORY_DEFAULT_CHAR_BUDGET = 48000
LLM_HISTORY_COMPRESSION_TRIGGER_RATIO = 0.80
LLM_HISTORY_COMPRESSION_MAX_CHARS = 90000
LLM_HISTORY_COMPRESSION_MAX_LINES = 120
LLM_HISTORY_COMPRESSION_MAX_ITEMS = 120
LLM_HISTORY_COMPRESSION_RECENT_USER_MESSAGES = 5
LLM_HISTORY_COMPRESSION_DIRECTIVE_MESSAGES = 20
STREAMING_ASSISTANT_SNAPSHOT_INTERVAL_SECONDS = 1.0
STREAMING_ASSISTANT_SNAPSHOT_MIN_CHAR_DELTA = 256
MODEL_INFO_CACHE_SCHEMA_VERSION = "reasoning-level-options-v2"
MODEL_INFO_CACHE_TTL_SECONDS = 300
MODEL_LIST_CACHE_TTL_SECONDS = 45
MODEL_RUNTIME_METADATA_PATH = settings.BASE_DIR / "Tools" / "model_runtime_metadata.json"
LLM_CONTROL_TOKEN_PATTERNS = (
    re.compile(
        r"<\|start\|>\s*(?:assistant|user|system)?\s*(?:<\|channel\|>\s*(?:final|analysis|commentary))?\s*(?:<\|message\|>)?",
        flags=re.IGNORECASE,
    ),
    re.compile(r"<\|start\|>", flags=re.IGNORECASE),
    re.compile(r"<\|channel\|>\s*(?:final|analysis|commentary)", flags=re.IGNORECASE),
    re.compile(r"<\|message\|>", flags=re.IGNORECASE),
    re.compile(r"<\|return\|>", flags=re.IGNORECASE),
    re.compile(r"<\|startoftext\|>", flags=re.IGNORECASE),
    re.compile(r"<\|im_(?:start|end)\|>", flags=re.IGNORECASE),
    re.compile(r"<\|(?:assistant|user|system|endoftext)\|>", flags=re.IGNORECASE),
)
SLASH_SEARCH_COMMAND_PATTERN = re.compile(r"(?:(?<=\s)|^)/search(?![\w-])", flags=re.IGNORECASE)
SLASH_TOOL_COMMAND_PATTERN = re.compile(r"(?:(?<=\s)|^)/tool[ \t]+(?P<id>[\w][\w.-]*)", flags=re.IGNORECASE)
SLASH_SKILL_COMMAND_PATTERN = re.compile(r"(?:(?<=\s)|^)/skill[ \t]+(?P<name>[\w][\w.-]*)", flags=re.IGNORECASE)
SLASH_SEARCH_DIRECTIVE = (
    "The user selected web search for this request. Use the web_search tool first, "
    "then answer from the result. If forced tool choice is unavailable, still call web_search when possible."
)
INSTANT_SLASH_SEARCH_BATCH_DIRECTIVE = (
    "Instant mode is active for this slash-command search. Every web_search call must contain "
    "exactly three complementary query strings in the query array. Send all three at once; never "
    "send a single query or a one/two-item batch. At most two web_search calls are allowed."
)
SLASH_INLINE_MARKER_DIRECTIVE = (
    "The user's message contains inline '[user command: ...]' markers. Each marker shows the exact "
    "place where the user invoked a command, so satisfy it for that part of the message."
)


@dataclass(frozen=True)
class ChatCommandPlan:
    message: str
    directives: list[str]
    tool_server_ids: list[str]
    required_tool_server_ids: list[str]
    forced_tool_name: str
    forced_skill_names: list[str]


# Parse slash commands anywhere in the message into deterministic generation controls.
# Each recognized command is rewritten in place into an inline marker so the model sees
# the request at the exact position where the user typed it.
def _parse_chat_command_plan(
    message: str,
    existing_tool_ids: list[str] | None = None,
    *,
    is_known_tool_id: Callable[[str], bool] | None = None,
) -> ChatCommandPlan:
    text = str(message or "")
    tool_ids = [str(item).strip() for item in existing_tool_ids or [] if str(item).strip()]
    directives: list[str] = []
    required_tool_ids: list[str] = []
    forced_tool_name = ""
    forced_skill_names: list[str] = []
    known_tool_id_cache: dict[str, bool] = {}

    def _tool_id_known(tool_id: str) -> bool:
        if is_known_tool_id is None:
            return True
        if tool_id not in known_tool_id_cache:
            try:
                known_tool_id_cache[tool_id] = bool(is_known_tool_id(tool_id))
            except Exception:
                logger.exception("Failed to validate slash-command tool id %s", tool_id)
                known_tool_id_cache[tool_id] = False
        return known_tool_id_cache[tool_id]

    def _require_tool(tool_id: str) -> None:
        if tool_id not in tool_ids:
            tool_ids.append(tool_id)
        if tool_id not in required_tool_ids:
            required_tool_ids.append(tool_id)

    def _replace_search(match: re.Match) -> str:
        nonlocal forced_tool_name
        # Unknown on this engine: leave the literal text so the request still succeeds.
        if not _tool_id_known("web_search"):
            return match.group(0)
        _require_tool("web_search")
        forced_tool_name = "web_search"
        if SLASH_SEARCH_DIRECTIVE not in directives:
            directives.append(SLASH_SEARCH_DIRECTIVE)
        return "[user command: run a web search (web_search tool) for this part of the request]"

    def _replace_tool(match: re.Match) -> str:
        tool_id = str(match.group("id") or "").strip()
        if not tool_id or not _tool_id_known(tool_id):
            return match.group(0)
        _require_tool(tool_id)
        directive = (
            f"The user selected tool server '{tool_id}' for this request. "
            "Actually call its tools where they apply instead of answering from memory."
        )
        if directive not in directives:
            directives.append(directive)
        return f"[user command: use the '{tool_id}' tools for this part of the request]"

    def _replace_skill(match: re.Match) -> str:
        skill_name = str(match.group("name") or "").strip()
        if not skill_name:
            return match.group(0)
        if skill_name not in forced_skill_names:
            forced_skill_names.append(skill_name)
        return f"[user command: apply the skill '{skill_name}' to this request]"

    text = SLASH_TOOL_COMMAND_PATTERN.sub(_replace_tool, text)
    text = SLASH_SKILL_COMMAND_PATTERN.sub(_replace_skill, text)
    text = SLASH_SEARCH_COMMAND_PATTERN.sub(_replace_search, text)

    if directives or forced_skill_names:
        directives.insert(0, SLASH_INLINE_MARKER_DIRECTIVE)

    return ChatCommandPlan(
        message=text.strip(),
        directives=directives,
        tool_server_ids=tool_ids,
        required_tool_server_ids=required_tool_ids,
        forced_tool_name=forced_tool_name,
        forced_skill_names=forced_skill_names,
    )


# Append one-shot slash-command directives to a system prompt.
def _append_command_directives_to_system_prompt(system_prompt: str, directives: list[str]) -> str:
    cleaned_directives = [str(item or "").strip() for item in directives or [] if str(item or "").strip()]
    if not cleaned_directives:
        return system_prompt
    directive_block = "Request command directives:\n" + "\n".join(f"- {item}" for item in cleaned_directives)
    return "\n\n".join(part for part in [str(system_prompt or "").strip(), directive_block] if part).strip()


def _command_directives_for_mode(command_plan: ChatCommandPlan, *, instant_mode: bool) -> list[str]:
    """Add request-scoped mode rules without changing the baseline prompt."""

    directives = list(command_plan.directives)
    if instant_mode and command_plan.forced_tool_name == "web_search":
        directives.append(INSTANT_SLASH_SEARCH_BATCH_DIRECTIVE)
    return directives


def _apply_command_tool_context(
    tool_context: dict[str, Any],
    command_plan: ChatCommandPlan,
    *,
    instant_mode: bool,
) -> None:
    """Apply deterministic slash-command controls to one adapter tool context."""

    if command_plan.forced_tool_name:
        tool_context["forced_tool_name"] = command_plan.forced_tool_name
    if command_plan.required_tool_server_ids:
        tool_context["required_tool_server_ids"] = command_plan.required_tool_server_ids
    if instant_mode and command_plan.forced_tool_name == "web_search":
        tool_context["instant_search_batch_size"] = tool_registry.INSTANT_SEARCH_BATCH_LIMIT

_metadata_cache_lock = threading.RLock()
_model_info_cache: dict[tuple[str, str, str, str], tuple[float, dict[str, Any]]] = {}
_model_list_cache: dict[tuple[str, str, str], tuple[float, list[str]]] = {}
_tool_server_cache: dict[tuple[str, str, str, str], tuple[float, list[dict[str, Any]]]] = {}
_active_model_lock = threading.RLock()
_active_model_by_engine: dict[str, str] = {}
_chat_usage_lock = threading.RLock()
_chat_usage_by_chat_id: dict[str, dict[str, Any]] = {}


_generation_state_lock = threading.RLock()
_active_generation_id_by_engine: dict[str, str] = {}
_active_generation_id_by_chat_id: dict[str, str] = {}


UPLOADED_FILE_CONTEXT_ENTRY_TYPE = "uploaded_file_context"
UPLOADED_JSON_CONTEXT_MAX_BYTES = 8 * 1024 * 1024
UPLOADED_MEDIA_CONTEXT_MAX_BYTES = 20 * 1024 * 1024
JSON_UPLOAD_MIME_TYPES = {
    "application/geo+json",
    "application/json",
    "application/ld+json",
    "application/vnd.api+json",
    "application/x-ndjson",
}
JSON_UPLOAD_SUFFIXES = {".geojson", ".json", ".jsonl", ".ndjson"}

CONTEXT_WINDOW_KEYS = (
    "num_ctx",
    "context_length",
    "contextLength",
    "context_window",
    "contextWindow",
    "max_context_length",
    "maxContextLength",
    "max_context_window",
    "maxContextWindow",
    "input_token_limit",
    "inputTokenLimit",
)
OUTPUT_TOKEN_KEYS = (
    "num_predict",
    "maxTokens",
    "max_tokens",
    "max_completion_tokens",
    "max_output_tokens",
    "output_token_limit",
    "outputTokenLimit",
)


# Return a stable runtime scope for model metadata caches.
def _engine_metadata_scope(engine: str) -> tuple[str, str]:
    try:
        endpoint = settings.get_engine_url(engine)
    except Exception:
        endpoint = ""

    try:
        api_key = settings.get_engine_api_key(engine)
    except Exception:
        api_key = ""

    api_key_hash = hashlib.sha256(str(api_key or "").encode("utf-8")).hexdigest()[:16]
    return endpoint, api_key_hash


# Return a defensive deep copy of a cacheable payload.
def _clone_metadata_payload(payload: Any) -> Any:
    return copy.deepcopy(payload)


# Clear cached model metadata.
def _clear_model_metadata_caches() -> None:
    with _metadata_cache_lock:
        _model_info_cache.clear()
        _model_list_cache.clear()
        _tool_server_cache.clear()


# Drop cached tool server lists only (e.g. after ``mcp.json`` changes).
def _clear_tool_server_cache() -> None:
    with _metadata_cache_lock:
        _tool_server_cache.clear()


# Drop the cached model list for one engine's current scope (force re-probe).
def _clear_cached_model_list(engine: str) -> None:
    endpoint, api_key_hash = _engine_metadata_scope(engine)
    cache_key = (engine, endpoint, api_key_hash)
    with _metadata_cache_lock:
        _model_list_cache.pop(cache_key, None)


# Remember the latest selected model for one engine.
def _remember_active_model(engine: str, model_name: str) -> None:
    normalized_engine = settings.normalize_engine_name(engine)
    normalized_model = str(model_name or "").strip()
    if not normalized_model:
        return

    with _active_model_lock:
        _active_model_by_engine[normalized_engine] = normalized_model


# Read the latest selected model for one engine.
def _get_remembered_active_model(engine: str) -> str:
    normalized_engine = settings.normalize_engine_name(engine)
    with _active_model_lock:
        return _active_model_by_engine.get(normalized_engine, "")


# Convert one value into a positive integer when possible.
def _coerce_positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None

    try:
        number = int(value)
    except (TypeError, ValueError):
        return None

    return number if number > 0 else None


# Read the first positive integer from a mapping.
def _first_positive_int(mapping: dict[str, Any], keys: tuple[str, ...]) -> int | None:
    if not isinstance(mapping, dict):
        return None

    for key in keys:
        number = _coerce_positive_int(mapping.get(key))
        if number is not None:
            return number

    return None


# Resolve the model name represented by an inference-info request.
def _resolve_inference_model(engine: str, requested_model: str | None = None) -> tuple[str, str]:
    model_name = str(requested_model or "").strip()
    if model_name:
        return model_name, "request"

    model_name = _get_remembered_active_model(engine)
    if model_name:
        return model_name, "runtime_selection"

    models = _load_models_for_engine(engine)
    if models:
        return models[0], "model_list"

    return "", "none"


# Return cached model info when it is still fresh.
def _get_cached_model_info(engine: str, model_name: str) -> dict[str, Any] | None:
    endpoint, api_key_hash = _engine_metadata_scope(engine)
    cache_key = (engine, model_name, endpoint, f"{api_key_hash}:{MODEL_INFO_CACHE_SCHEMA_VERSION}")
    now = time.monotonic()

    with _metadata_cache_lock:
        cached = _model_info_cache.get(cache_key)
        if cached is None:
            return None
        cached_at, payload = cached
        if now - cached_at > MODEL_INFO_CACHE_TTL_SECONDS:
            _model_info_cache.pop(cache_key, None)
            return None
        return _clone_metadata_payload(payload)


# Store model info in the runtime cache.
def _set_cached_model_info(engine: str, model_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    endpoint, api_key_hash = _engine_metadata_scope(engine)
    cache_key = (engine, model_name, endpoint, f"{api_key_hash}:{MODEL_INFO_CACHE_SCHEMA_VERSION}")
    cached_payload = _clone_metadata_payload(payload)

    with _metadata_cache_lock:
        _model_info_cache[cache_key] = (time.monotonic(), cached_payload)

    return _clone_metadata_payload(cached_payload)


# Return cached model names when still fresh.
def _get_cached_model_list(engine: str) -> list[str] | None:
    endpoint, api_key_hash = _engine_metadata_scope(engine)
    cache_key = (engine, endpoint, api_key_hash)
    now = time.monotonic()

    with _metadata_cache_lock:
        cached = _model_list_cache.get(cache_key)
        if cached is None:
            return None
        cached_at, models = cached
        if now - cached_at > MODEL_LIST_CACHE_TTL_SECONDS:
            _model_list_cache.pop(cache_key, None)
            return None
        return list(models)


# Store model names in the runtime cache.
def _set_cached_model_list(engine: str, models: list[str]) -> list[str]:
    endpoint, api_key_hash = _engine_metadata_scope(engine)
    cache_key = (engine, endpoint, api_key_hash)
    cached_models = list(models)

    with _metadata_cache_lock:
        _model_list_cache[cache_key] = (time.monotonic(), cached_models)

    return list(cached_models)


# Return cached tool servers when still fresh.
def _list_tool_servers_cached(engine: str, model_name: str | None = None) -> list[dict[str, Any]]:
    endpoint, api_key_hash = _engine_metadata_scope(engine)
    normalized_model = str(model_name or "")
    cache_key = (engine, normalized_model, endpoint, api_key_hash)
    now = time.monotonic()

    with _metadata_cache_lock:
        cached = _tool_server_cache.get(cache_key)
        if cached is not None:
            cached_at, servers = cached
            if now - cached_at <= MODEL_LIST_CACHE_TTL_SECONDS:
                return _stamp_docker_availability([
                    server
                    for server in _clone_metadata_payload(servers)
                    if str(server.get("id") or "").strip().lower() != "deep_research"
                ])
            _tool_server_cache.pop(cache_key, None)

    # Deep Research has its own chat protocol and must never be offered to the
    # model (or slash/MCP menus) as a selectable tool server.
    servers = [
        server
        for server in tool_registry.list_servers(engine, model_name)
        if str(server.get("id") or "").strip().lower() != "deep_research"
    ]
    with _metadata_cache_lock:
        _tool_server_cache[cache_key] = (time.monotonic(), _clone_metadata_payload(servers))
    return _stamp_docker_availability(_clone_metadata_payload(servers))


# Stamp live Docker availability onto servers that require it (e.g. the sandbox).
# Kept outside the metadata cache so the flag always reflects the latest probe.
def _stamp_docker_availability(servers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not any(server.get("requires_docker") for server in servers):
        return servers

    from Services import docker_status

    available = docker_status.is_available()
    for server in servers:
        if server.get("requires_docker"):
            server["docker_available"] = available
    return servers


# Emit one concise runtime event for the ASLM console.
def _print_runtime_event(message: str) -> None:
    print(f"[ASLM-Chat] {message}", flush=True)


# Return whether the exception is an expected runtime/connectivity failure.
def _is_expected_runtime_error(exc: Exception) -> bool:
    exc_name = type(exc).__name__
    if exc_name in {"ConnectError", "ConnectionError", "ReadTimeout", "TimeoutException"}:
        return True

    message = str(exc).lower()
    expected_markers = (
        "failed to connect to ollama",
        "connection refused",
        "connection error",
        "connecterror",
        "timed out",
        "timeout",
        "winerror 10061",
    )
    return any(marker in message for marker in expected_markers)


# Return a user-facing runtime error string without noisy transport details.
def _format_runtime_error(engine: str, exc: Exception) -> str:
    message = str(exc).strip()
    normalized_message = message.lower()

    if settings.is_ollama_engine(engine) and _is_expected_runtime_error(exc):
        return (
            "Failed to connect to Ollama. Please check that Ollama is downloaded, "
            "running and accessible. https://ollama.com/download"
        )

    if engine == "openai" and _is_expected_runtime_error(exc):
        return "Failed to connect to the configured OpenAI-compatible endpoint."

    if engine == "google-genai" and _is_expected_runtime_error(exc):
        return "Failed to connect to the configured Google GenAI endpoint."

    if engine == "lms" and _is_expected_runtime_error(exc):
        return "Failed to connect to LM Studio."

    if engine == "google-genai":
        if any(marker in normalized_message for marker in ("api key", "permission denied", "unauthenticated", "unauthorized")):
            return "Google GenAI authentication failed. Check the configured API key."

    if engine == "lms":
        if "v cache quantization requires flash attention" in normalized_message:
            return (
                "The current LM Studio load settings are incompatible. "
                "Enable Flash Attention or set V Cache Quantization Type to f16."
            )
        if "model get/load error" in normalized_message or "failed to load model" in normalized_message:
            return "LM Studio could not load the selected model with the current load settings."

    return message


# Remove assistant-control tokens that should never be shown to the user.
def _strip_llm_control_tokens(content: str) -> str:
    cleaned = str(content or "")
    for pattern in LLM_CONTROL_TOKEN_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    return cleaned


# Return a short, readable summary of option keys.
def _summarize_option_keys(options: dict[str, Any] | None, max_keys: int = 6) -> str:
    if not isinstance(options, dict) or not options:
        return "none"

    option_keys = sorted({str(key).strip() for key in options if str(key).strip()}, key=str.casefold)
    if len(option_keys) <= max_keys:
        return ", ".join(option_keys)
    return f"{', '.join(option_keys[:max_keys])}, +{len(option_keys) - max_keys} more"


# Count image attachments present in the current outbound request.
def _count_request_images(messages: list[dict[str, Any]]) -> int:
    image_count = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        raw_images = message.get("images")
        if isinstance(raw_images, list):
            image_count += len(raw_images)
    return image_count


# Return decoded bytes for one base64 payload or data URL.
def _decode_base64_payload(raw_value: Any) -> bytes:
    if raw_value is None:
        return b""

    payload = str(raw_value).strip()
    if not payload:
        return b""

    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    try:
        return base64.b64decode(payload, validate=True)
    except (ValueError, binascii.Error):
        return base64.b64decode(payload)


# Estimate decoded byte size without materializing the full payload.
def _estimate_base64_payload_size(raw_value: Any) -> int:
    payload = str(raw_value or "").strip()
    if not payload:
        return 0

    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    payload = "".join(payload.split())
    if not payload:
        return 0

    padding = len(payload) - len(payload.rstrip("="))
    return max((len(payload) * 3) // 4 - padding, 0)


# Return whether one payload is structurally valid base64.
def _is_valid_base64_payload(raw_value: Any) -> bool:
    payload = str(raw_value or "").strip()
    if not payload:
        return False

    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]

    payload = "".join(payload.split())
    if not payload or len(payload) % 4 == 1:
        return False

    return re.fullmatch(r"[A-Za-z0-9+/]*={0,2}", payload) is not None


# Split a data URL into MIME type and base64 payload.
def _parse_data_url(raw_value: Any) -> tuple[str, str]:
    payload = str(raw_value or "").strip()
    if not payload.startswith("data:") or "," not in payload:
        return "", payload

    header, encoded = payload.split(",", 1)
    mime_type = "application/octet-stream"
    if ";" in header:
        mime_type = header[5:].split(";", 1)[0].strip() or mime_type
    return mime_type, encoded


# Return the normalized attachment kind for the payload.
def _guess_attachment_kind(mime_type: str, name: str = "") -> str:
    normalized_mime = str(mime_type or "").strip().lower()
    if normalized_mime.startswith("image/"):
        return MessageAttachmentKind.IMAGE

    guessed_mime, _encoding = mimetypes.guess_type(name or "")
    if guessed_mime and guessed_mime.startswith("image/"):
        return MessageAttachmentKind.IMAGE

    return MessageAttachmentKind.FILE


# Normalize one incoming attachment payload into the storage shape.
def _normalize_attachment_payload(raw_attachment: Any, order: int) -> dict[str, Any] | None:
    if isinstance(raw_attachment, str):
        mime_type = _detect_image_mime(raw_attachment)
        raw_data = raw_attachment
        name = f"image-{order + 1}"
    elif isinstance(raw_attachment, dict):
        name = str(
            raw_attachment.get("name")
            or raw_attachment.get("filename")
            or raw_attachment.get("title")
            or ""
        ).strip()
        mime_type = str(raw_attachment.get("mime_type") or raw_attachment.get("mimeType") or "").strip()
        raw_data = raw_attachment.get("data")
        if raw_data is None:
            raw_data = raw_attachment.get("base64")
        data_url = raw_attachment.get("data_url") or raw_attachment.get("dataUrl")
        if data_url:
            parsed_mime, parsed_data = _parse_data_url(data_url)
            if parsed_mime and not mime_type:
                mime_type = parsed_mime
            raw_data = parsed_data
        if not mime_type and name:
            mime_type = mimetypes.guess_type(name)[0] or ""
        if not mime_type:
            mime_type = "application/octet-stream"
    else:
        return None

    # URL attachments (pasted/dropped links). No data required. Handled specially:
    # keep original URL text in input, add attachment, fetch content via read_page after send.
    url_val = ""
    if isinstance(raw_attachment, dict):
        url_val = str(raw_attachment.get("url") or raw_attachment.get("href") or "").strip()
    if url_val and (url_val.lower().startswith("http://") or url_val.lower().startswith("https://")):
        return {
            "kind": "url",
            "name": url_val,
            "mime_type": "text/x-url",
            "data": "",
            "size_bytes": 0,
            "url": url_val,
            "order": order,
        }

    encoded = str(raw_data or "").strip()
    if not encoded:
        return None

    if encoded.startswith("data:"):
        parsed_mime, parsed_data = _parse_data_url(encoded)
        if parsed_mime:
            mime_type = parsed_mime
        encoded = parsed_data

    size_bytes = _estimate_base64_payload_size(encoded)
    if size_bytes <= 0 or not _is_valid_base64_payload(encoded):
        return None

    kind = _guess_attachment_kind(mime_type, name)
    if not name:
        extension = mimetypes.guess_extension(mime_type or "") or ""
        base_name = "image" if kind == MessageAttachmentKind.IMAGE else "file"
        name = f"{base_name}-{order + 1}{extension}"

    return {
        "kind": kind,
        "name": name,
        "mime_type": mime_type,
        "data": encoded,
        "size_bytes": size_bytes,
        "order": order,
    }


# Return a normalized list of request attachments.
def _normalize_request_attachments(data: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    raw_attachments = data.get("attachments", []) or []
    raw_images = data.get("images", []) or []
    for raw_attachment in list(raw_attachments) + list(raw_images):
        attachment = _normalize_attachment_payload(raw_attachment, len(normalized))
        if attachment is not None:
            normalized.append(attachment)
    return normalized


# Build an attachment content endpoint path.
def _attachment_content_url(record_type: str, record_id: int) -> str:
    return f"/api/attachment/{record_type}/{record_id}/content/"


# Convert a persisted attachment-like object into the frontend payload.
def _serialize_attachment_record(attachment: Any, *, include_data: bool = True) -> dict[str, Any]:
    if isinstance(attachment, MessageAttachment):
        payload = {
            "id": attachment.id,
            "record_type": "attachment",
            "kind": attachment.kind,
            "name": attachment.name,
            "mime_type": attachment.mime_type,
            "size_bytes": attachment.size_bytes,
            "order": attachment.order,
            "content_url": _attachment_content_url("attachment", attachment.id),
            "url": attachment.name if attachment.kind == "url" else "",
        }
        if include_data:
            payload["data_url"] = attachment.data_url()
            payload["extracted_text"] = attachment.extracted_text
            payload["extracted_text_ready"] = attachment.extracted_text_ready
        return payload

    if isinstance(attachment, MessageImage):
        payload = {
            "id": attachment.id,
            "record_type": "image",
            "kind": MessageAttachmentKind.IMAGE,
            "name": f"image-{attachment.order + 1}",
            "mime_type": attachment.mime_type,
            "size_bytes": _estimate_base64_payload_size(attachment.data),
            "order": attachment.order,
            "content_url": _attachment_content_url("image", attachment.id),
        }
        if include_data:
            payload["data_url"] = attachment.data_url()
        return payload

    return {}


# Return all persisted attachments for a message in a shared shape.
def _get_message_attachments(message: Message, *, include_data: bool = True) -> list[dict[str, Any]]:
    attachments = [_serialize_attachment_record(item, include_data=include_data) for item in message.attachments.all()]
    legacy_images = [_serialize_attachment_record(item, include_data=include_data) for item in message.images.all()]
    combined = [item for item in attachments + legacy_images if item]
    combined.sort(key=lambda item: (int(item.get("order") or 0), item.get("name", "")))
    return combined


# Decode one serialized attachment payload into bytes.
def _attachment_data_to_bytes(attachment: dict[str, Any]) -> bytes:
    try:
        return _decode_base64_payload(attachment.get("data_url") or attachment.get("data"))
    except (ValueError, binascii.Error):
        return b""


# Return whether the attachment should be decoded as text.
def _is_text_attachment(mime_type: str, name: str) -> bool:
    normalized_mime = str(mime_type or "").strip().lower()
    normalized_name = Path(name or "").name.lower()
    attachment_path = Path(name or "")
    if normalized_mime.startswith("text/"):
        return True
    if normalized_mime in {
        "application/ecmascript",
        "application/json",
        "application/ld+json",
        "application/javascript",
        "application/sql",
        "application/toml",
        "application/vnd.api+json",
        "application/xml",
        "application/x-httpd-php",
        "application/x-javascript",
        "application/x-ndjson",
        "application/x-sh",
        "application/x-shellscript",
        "application/x-toml",
        "application/x-yaml",
        "application/yaml",
        "image/svg+xml",
    }:
        return True

    if normalized_name in TEXT_ATTACHMENT_FILENAMES:
        return True

    return any(suffix.lower() in TEXT_ATTACHMENT_EXTENSIONS for suffix in attachment_path.suffixes)


# Trim attachment text so prompts stay bounded.
def _truncate_attachment_text(text: str, limit: int = ATTACHMENT_TEXT_CHAR_LIMIT) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[:limit].rstrip()}\n...[truncated]"


# Persist extracted text for one stored file attachment.
def _cache_attachment_text(attachment: dict[str, Any], extracted_text: str) -> str:
    attachment["extracted_text"] = extracted_text
    attachment["extracted_text_ready"] = True

    if attachment.get("record_type") != "attachment":
        return extracted_text

    try:
        attachment_id = int(attachment.get("id") or 0)
    except (TypeError, ValueError):
        attachment_id = 0

    if attachment_id <= 0:
        return extracted_text

    MessageAttachment.objects.filter(id=attachment_id).update(
        extracted_text=extracted_text,
        extracted_text_ready=True,
    )
    return extracted_text


# Extract prompt-friendly text from a file attachment when possible.
def _extract_attachment_text(attachment: dict[str, Any]) -> str:
    if attachment.get("extracted_text_ready"):
        return _truncate_attachment_text(str(attachment.get("extracted_text") or ""))

    # For URL attachments the page content is pre-fetched into extracted_text by chat_api.
    if str(attachment.get("kind") or "").lower() == "url":
        # If not ready yet (should not happen), return stub.
        return _cache_attachment_text(attachment, str(attachment.get("extracted_text") or ""))

    attachment_name = str(attachment.get("name") or "file").strip() or "file"
    mime_type = str(attachment.get("mime_type") or "application/octet-stream").strip()
    file_bytes = _attachment_data_to_bytes(attachment)
    if not file_bytes:
        return _cache_attachment_text(attachment, "")

    suffix = Path(attachment_name).suffix.lower()
    if mime_type == "application/pdf" or suffix == ".pdf":
        try:
            import fitz

            with fitz.open(stream=file_bytes, filetype="pdf") as document:
                pages = [page.get_text("text") for page in document]
            return _cache_attachment_text(attachment, _truncate_attachment_text("\n".join(pages)))
        except Exception:
            return _cache_attachment_text(attachment, "")

    if _is_text_attachment(mime_type, attachment_name):
        for encoding in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
            try:
                return _cache_attachment_text(attachment, _truncate_attachment_text(file_bytes.decode(encoding)))
            except UnicodeDecodeError:
                continue

    return _cache_attachment_text(attachment, "")


# Serialize one non-image attachment into universal text context.
def _build_file_attachment_prompt_block(attachment: dict[str, Any]) -> str:
    attachment_name = str(attachment.get("name") or "file").strip() or "file"
    mime_type = str(attachment.get("mime_type") or "application/octet-stream").strip()
    size_bytes = int(attachment.get("size_bytes") or 0)
    extracted_text = _extract_attachment_text(attachment)

    # Special friendly block for URL attachments (fetched via read_page).
    if str(attachment.get("kind") or "").lower() == "url" or mime_type == "text/x-url":
        url = attachment.get("url") or attachment_name
        if extracted_text and not str(extracted_text).lower().startswith("error"):
            return (
                f"[Attached URL: {url}]\n"
                f"Content:\n{extracted_text}\n"
                f"[/Attached URL]"
            )
        return f"[Attached URL: {url} — content could not be fetched]"

    if extracted_text:
        return (
            f"[Attached file: {attachment_name}]\n"
            f"MIME: {mime_type}\n"
            f"Size: {size_bytes} bytes\n"
            f"Content:\n{extracted_text}\n"
            f"[/Attached file]"
        )

    return (
        f"[Attached file: {attachment_name}]\n"
        f"MIME: {mime_type}\n"
        f"Size: {size_bytes} bytes\n"
        "Content could not be extracted automatically.\n"
        "[/Attached file]"
    )


# Return whether the resolved tool selection includes sandbox file access.
def _selected_tools_include_sandbox(selected_tool_servers: list[dict[str, Any]]) -> bool:
    selected_ids = {
        str(server.get("id") or "").strip()
        for server in selected_tool_servers
    }
    return "sandbox" in selected_ids


# Return uploaded file ids referenced by a chat request.
def _normalize_uploaded_file_ids(data: dict[str, Any]) -> list[str]:
    raw_values: list[Any] = []
    for key in ("uploaded_file_ids", "uploaded_files", "file_ids"):
        value = data.get(key)
        if isinstance(value, list):
            raw_values.extend(value)
        elif value:
            raw_values.append(value)

    for attachment in data.get("attachments", []) or []:
        if isinstance(attachment, dict):
            file_id = attachment.get("file_id")
            if file_id:
                raw_values.append(file_id)

    seen: set[str] = set()
    normalized: list[str] = []
    for raw_value in raw_values:
        if isinstance(raw_value, dict):
            raw_value = raw_value.get("file_id") or raw_value.get("id")
        file_id = str(raw_value or "").strip()
        if file_id and file_id not in seen:
            seen.add(file_id)
            normalized.append(file_id)
    return normalized


# Load model-facing upload manifests for the selected tool state.
def _load_model_upload_manifests(file_ids: list[str], *, sandbox_enabled: bool) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    for file_id in file_ids:
        manifest = load_upload_manifest(file_id)
        if manifest:
            manifests.append(model_upload_payload(manifest, sandbox_enabled=sandbox_enabled))
    return manifests


# Return stable file ids from loaded upload manifests.
def _upload_manifest_file_ids(manifests: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    file_ids: list[str] = []
    for manifest in manifests or []:
        file_id = str((manifest or {}).get("file_id") or "").strip()
        if file_id and file_id not in seen:
            seen.add(file_id)
            file_ids.append(file_id)
    return file_ids


# Build a stored user-message entry that keeps upload context replayable.
def _build_uploaded_file_context_entry(file_ids: list[str]) -> dict[str, Any] | None:
    normalized_ids = []
    seen: set[str] = set()
    for file_id in file_ids or []:
        clean_id = str(file_id or "").strip()
        if clean_id and clean_id not in seen:
            seen.add(clean_id)
            normalized_ids.append(clean_id)
    if not normalized_ids:
        return None
    return {
        "type": UPLOADED_FILE_CONTEXT_ENTRY_TYPE,
        "uploaded_file_ids": normalized_ids,
    }


# Return upload ids persisted on a user message.
def _extract_uploaded_file_ids_from_message(message: Message) -> list[str]:
    raw_entries = getattr(message, "llm_transcript", None)
    if not isinstance(raw_entries, list):
        return []

    seen: set[str] = set()
    file_ids: list[str] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        entry_type = str(raw_entry.get("type") or raw_entry.get("kind") or "").strip()
        if entry_type != UPLOADED_FILE_CONTEXT_ENTRY_TYPE:
            continue

        raw_values: list[Any] = []
        for key in ("uploaded_file_ids", "uploaded_files", "file_ids"):
            value = raw_entry.get(key)
            if isinstance(value, list):
                raw_values.extend(value)
            elif value:
                raw_values.append(value)

        for raw_value in raw_values:
            if isinstance(raw_value, dict):
                raw_value = raw_value.get("file_id") or raw_value.get("id")
            file_id = str(raw_value or "").strip()
            if file_id and file_id not in seen:
                seen.add(file_id)
                file_ids.append(file_id)
    return file_ids


# Load persisted upload manifests for one stored user message.
def _load_message_upload_manifests(message: Message, *, sandbox_enabled: bool) -> list[dict[str, Any]]:
    return _load_model_upload_manifests(
        _extract_uploaded_file_ids_from_message(message),
        sandbox_enabled=sandbox_enabled,
    )


# Serialize one uploaded file manifest into private model context.
def _build_uploaded_file_prompt_block(manifest: dict[str, Any]) -> str:
    name = str(manifest.get("name") or "file").strip() or "file"
    mime = str(manifest.get("mime") or "application/octet-stream").strip()
    size_bytes = int(manifest.get("size_bytes") or 0)
    lines = [
        f"[Uploaded file: {name}]",
        f"File ID: {manifest.get('file_id') or ''}",
        f"MIME: {mime}",
        f"Size: {size_bytes} bytes",
    ]

    sandbox_path = manifest.get("sandbox_path")
    if sandbox_path:
        lines.append(f"Sandbox path: {sandbox_path}")

    archive_tree = manifest.get("archive_tree")
    if isinstance(archive_tree, list) and archive_tree:
        lines.append(
            "Archive preview (file names read from the compressed archive; "
            "the archive has not been extracted):"
        )
        lines.extend(f"- {item}" for item in archive_tree[:250])

    table_preview = str(manifest.get("table_preview") or "").strip()
    if table_preview:
        lines.append(f"Table preview:\n{table_preview}")

    json_content = manifest.get("json_content")
    if isinstance(json_content, str):
        lines.append(f"JSON content:\n```json\n{json_content}\n```")
        if manifest.get("json_content_truncated"):
            lines.append("JSON content was truncated to the model-context size limit.")

    text_preview = str(manifest.get("text_preview") or "").strip()
    if text_preview and not isinstance(json_content, str):
        lines.append(f"Text preview:\n{text_preview}")
    elif manifest.get("media_context_attached"):
        lines.append("Media content is attached as a native multimodal input.")
    elif not isinstance(json_content, str) and not sandbox_path:
        lines.append("Readable content is not available for this model request.")

    lines.append("[/Uploaded file]")
    return "\n".join(lines)


# Return media kinds accepted by the selected model, based on normalized adapter metadata.
def _supported_model_media_kinds(
    engine: str,
    model_info_payload: dict[str, Any] | None,
) -> set[str]:
    payload = model_info_payload if isinstance(model_info_payload, dict) else {}
    supported: set[str] = set()
    if bool(payload.get("supports_vision", False)):
        supported.add("image")
    if bool(payload.get("supports_audio_input", False)):
        supported.add("audio")
    if bool(payload.get("supports_video_input", False)):
        supported.add("video")

    # Ollama and LM Studio currently expose image handles only. Never infer
    # audio/video support for them from a broad "multimodal" label.
    if settings.is_ollama_engine(engine) or settings.normalize_engine_name(engine) == "lms":
        supported.intersection_update({"image"})
    return supported


# Return the uploaded file bytes without exposing the host path in model text.
def _read_uploaded_context_bytes(manifest: dict[str, Any], *, max_bytes: int) -> tuple[bytes, bool]:
    file_id = str(manifest.get("file_id") or "").strip()
    storage_manifest = load_upload_manifest(file_id) if file_id else None
    source_manifest = storage_manifest or manifest
    try:
        target = resolve_uploaded_file_host_path(source_manifest)
        with target.open("rb") as handle:
            payload = handle.read(max_bytes + 1)
    except (FileNotFoundError, OSError, ValueError):
        return b"", False
    return payload[:max_bytes], len(payload) > max_bytes


# Decode JSON-family uploads for direct text context instead of a short manifest preview.
def _uploaded_json_context(manifest: dict[str, Any]) -> tuple[str | None, bool]:
    name = str(manifest.get("name") or "").strip()
    mime = str(manifest.get("mime") or "").split(";", 1)[0].strip().lower()
    if mime not in JSON_UPLOAD_MIME_TYPES and Path(name).suffix.lower() not in JSON_UPLOAD_SUFFIXES:
        return None, False

    raw_bytes, truncated = _read_uploaded_context_bytes(
        manifest,
        max_bytes=UPLOADED_JSON_CONTEXT_MAX_BYTES,
    )
    if not raw_bytes:
        return None, truncated
    for encoding in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return raw_bytes.decode(encoding).strip(), truncated
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace").strip(), truncated


# Build one inline media part only when the selected model supports its modality.
def _uploaded_media_context(
    manifest: dict[str, Any],
    supported_media_kinds: set[str],
) -> dict[str, Any] | None:
    mime = str(manifest.get("mime") or "application/octet-stream").split(";", 1)[0].strip().lower()
    kind = next((candidate for candidate in ("image", "audio", "video") if mime.startswith(f"{candidate}/")), "")
    if not kind or kind not in supported_media_kinds:
        return None

    raw_bytes, truncated = _read_uploaded_context_bytes(
        manifest,
        max_bytes=UPLOADED_MEDIA_CONTEXT_MAX_BYTES,
    )
    if not raw_bytes or truncated:
        return None
    return {
        "kind": kind,
        "name": str(manifest.get("name") or f"uploaded-{kind}"),
        "mime_type": mime,
        "data": base64.b64encode(raw_bytes).decode("ascii"),
    }


# Attach uploaded file manifests to the current user entry.
def _apply_uploaded_file_manifests_to_llm_entry(
    entry: dict[str, Any],
    manifests: list[dict[str, Any]],
    *,
    supported_media_kinds: set[str] | None = None,
) -> dict[str, Any]:
    allowed_media = set(supported_media_kinds or set())
    hydrated_manifests: list[dict[str, Any]] = []
    media_parts: list[dict[str, Any]] = []
    for manifest in manifests:
        if not manifest:
            continue
        hydrated = dict(manifest)
        json_content, json_truncated = _uploaded_json_context(hydrated)
        if json_content is not None:
            hydrated["json_content"] = json_content
            hydrated["json_content_truncated"] = json_truncated
        media_part = _uploaded_media_context(hydrated, allowed_media)
        if media_part:
            hydrated["media_context_attached"] = True
            media_parts.append(media_part)
        hydrated_manifests.append(hydrated)

    blocks = [_build_uploaded_file_prompt_block(manifest) for manifest in hydrated_manifests]
    if not blocks:
        return entry

    content = str(entry.get("content") or "").strip()
    upload_context = "\n\n".join(blocks)
    entry["content"] = f"{content}\n\n{upload_context}".strip() if content else upload_context

    for media_part in media_parts:
        if media_part["kind"] == "image":
            images = entry.setdefault("images", [])
            image_mime_types = entry.setdefault("image_mime_types", [])
            if media_part["data"] not in images:
                images.append(media_part["data"])
                image_mime_types.append(media_part["mime_type"])
            continue
        media = entry.setdefault("media", [])
        if not any(
            item.get("data") == media_part["data"] and item.get("mime_type") == media_part["mime_type"]
            for item in media
            if isinstance(item, dict)
        ):
            media.append(media_part)
    return entry


# Attach images and file context to one outbound LLM message.
def _apply_attachments_to_llm_entry(entry: dict[str, Any], attachments: list[dict[str, Any]]) -> dict[str, Any]:
    image_payloads = []
    image_mime_types = []
    file_blocks = []

    for attachment in attachments:
        if attachment.get("kind") == MessageAttachmentKind.IMAGE:
            raw_payload = str(attachment.get("data_url") or attachment.get("data") or "")
            mime_type, encoded = _parse_data_url(raw_payload)
            image_payloads.append(encoded or raw_payload)
            image_mime_types.append(str(attachment.get("mime_type") or mime_type or "image/jpeg"))
        else:
            file_blocks.append(_build_file_attachment_prompt_block(attachment))

    if image_payloads:
        entry["images"] = image_payloads
        entry["image_mime_types"] = image_mime_types

    if file_blocks:
        content = str(entry.get("content") or "").strip()
        blocks = "\n\n".join(block for block in file_blocks if block)
        entry["content"] = f"{content}\n\n{blocks}".strip() if content else blocks

    return entry


# Read local GPU devices
def _get_local_gpu_devices() -> list[dict[str, Any]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []

    if result.returncode != 0:
        return []

    devices: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",", 1)]
        if len(parts) != 2 or not parts[0]:
            continue
        try:
            device_id = int(parts[0])
        except ValueError:
            continue
        devices.append({"id": device_id, "name": parts[1] or f"GPU {device_id}"})

    return devices


# Raised when an explicit request targets a disabled engine.
class RequestEngineResolutionError(ValueError):
    """The requested engine is not enabled in settings."""


# Resolve the default active engine without request context.
def _get_active_engine(requested_engine: str | None = None) -> str:
    return settings.normalize_engine_name(requested_engine or settings.get_llm_engine())


# Resolve one engine from an HTTP request body and/or query string.
def _resolve_request_engine(request, data: dict[str, Any] | None = None) -> str:
    explicit_engine = None
    if isinstance(data, dict) and data.get("engine"):
        explicit_engine = data.get("engine")
    elif request.GET.get("engine"):
        explicit_engine = request.GET.get("engine")

    if explicit_engine:
        canonical_engine = settings.normalize_engine_name(str(explicit_engine))
        if canonical_engine not in settings.ENGINE_IDS:
            raise RequestEngineResolutionError(
                f"Unsupported engine '{canonical_engine}'."
            )
        if not settings.is_engine_enabled(canonical_engine):
            raise RequestEngineResolutionError(
                f"Engine '{canonical_engine}' is not enabled."
            )
        return canonical_engine

    return settings.get_llm_engine()


# Resolve one request engine or return a JSON error response.
def _resolve_request_engine_or_response(request, data: dict[str, Any] | None = None):
    try:
        return _resolve_request_engine(request, data), None
    except RequestEngineResolutionError as exc:
        return None, JsonResponse({"error": str(exc)}, status=400)


# Extract model name
def _extract_model_name(model_entry: Any) -> str:
    if isinstance(model_entry, str):
        return model_entry
    if isinstance(model_entry, dict):
        for key in ("model", "id", "model_key", "identifier", "name"):
            value = model_entry.get(key)
            if value:
                return str(value)
        return ""
    for attr in ("model", "id", "model_key", "identifier", "name"):
        value = getattr(model_entry, attr, None)
        if value:
            return str(value)
    return ""


# Load engine models
def _load_models_for_engine(engine: str) -> list[str]:
    cached_models = _get_cached_model_list(engine)
    if cached_models is not None:
        return cached_models

    try:
        raw_models = llm_api.get_models(engine)
    except NotImplementedError:
        logger.info("Model listing is not implemented for engine %s", engine)
        _print_runtime_event(f"Models not supported for engine={engine}.")
        return []
    except Exception as exc:
        logger.warning("Failed to load models for engine %s: %s", engine, exc)
        _print_runtime_event(f"Model list failed: engine={engine}, error={exc}")
        return []

    model_names = []
    for entry in raw_models or []:
        model_name = _extract_model_name(entry)
        if model_name:
            model_names.append(model_name)

    sorted_model_names = sorted(set(model_names), key=str.casefold)
    if not sorted_model_names and settings.is_ollama_engine(engine):
        return []

    return _set_cached_model_list(engine, sorted_model_names)


# Detect image MIME type
def _detect_image_mime(base64_data: str) -> str:
    if base64_data.startswith("/9j/"):
        return "image/jpeg"
    if base64_data.startswith("iVBOR"):
        return "image/png"
    if base64_data.startswith("R0lGO"):
        return "image/gif"
    if base64_data.startswith("UklGR"):
        return "image/webp"
    return "image/jpeg"


# Strip legacy markup
def _strip_llm_markup(content: str) -> str:
    source = _strip_llm_control_tokens(str(content or ""))
    source = re.sub(r"<think>.*?</think>", "", source, flags=re.DOTALL)
    source = re.sub(r"<tool_call>.*?</tool_call>", "", source, flags=re.DOTALL)
    return source.strip()


# Normalize transcript entries
def _normalize_transcript_entries(raw_entries: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_entries, list):
        return []

    entries: list[dict[str, Any]] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            continue
        role = str(raw_entry.get("role", "") or "").strip().lower()
        if not role and (
            raw_entry.get("alias")
            or raw_entry.get("tool_id")
            or raw_entry.get("tool_name")
            or raw_entry.get("tool_display_name")
        ):
            role = "tool"
        if role not in {"assistant", "tool"}:
            continue

        entry: dict[str, Any] = {
            "role": role,
            "content": str(raw_entry.get("content", "") or ""),
        }
        if role == "assistant":
            thinking = str(raw_entry.get("thinking", "") or "")
            if thinking:
                entry["thinking"] = thinking
            google_parts = raw_entry.get("google_parts")
            if isinstance(google_parts, list):
                entry["google_parts"] = google_parts
            tool_calls = raw_entry.get("tool_calls")
            if isinstance(tool_calls, list):
                entry["tool_calls"] = tool_calls
        else:
            for key in ("alias", "name", "tool_name", "tool_call_id", "server_id", "server_name", "tool_id", "tool_display_name"):
                value = raw_entry.get(key)
                if value is not None:
                    entry[key] = value
            arguments = raw_entry.get("arguments")
            if isinstance(arguments, dict):
                entry["arguments"] = arguments
            tool_ui = raw_entry.get("tool_ui")
            if isinstance(tool_ui, dict):
                entry["tool_ui"] = tool_ui
            structured_content = raw_entry.get("structured_content")
            if isinstance(structured_content, dict):
                entry["structured_content"] = structured_content
        entries.append(entry)

    return entries


# Convert one assistant transcript into LLM history entries.
def _llm_entries_from_assistant_transcript(
    transcript_entries: list[dict[str, Any]],
    *,
    content_fallback: str = "",
) -> list[dict[str, Any]]:
    if transcript_entries:
        llm_entries: list[dict[str, Any]] = []
        for entry in transcript_entries:
            if entry["role"] == "tool" and str(entry.get("alias") or "") == "context_compression_summary":
                summary_content = str(entry.get("content") or "").strip()
                if summary_content:
                    llm_entries.append({"role": "system", "content": summary_content})
                continue
            payload = {
                "role": entry["role"],
                "content": entry.get("content", ""),
            }
            if entry["role"] == "assistant":
                if entry.get("thinking"):
                    payload["thinking"] = entry["thinking"]
                if isinstance(entry.get("google_parts"), list):
                    payload["google_parts"] = entry["google_parts"]
                if isinstance(entry.get("tool_calls"), list):
                    payload["tool_calls"] = entry["tool_calls"]
            else:
                if entry.get("tool_call_id"):
                    payload["tool_call_id"] = entry["tool_call_id"]
                if entry.get("name"):
                    payload["name"] = entry["name"]
                if entry.get("tool_name"):
                    payload["tool_name"] = entry["tool_name"]
            llm_entries.append(payload)
        return llm_entries

    stripped_content = _strip_llm_markup(content_fallback)
    if not stripped_content:
        return []
    return [{"role": "assistant", "content": stripped_content}]


# Normalize inline attachments from one request mapping.
def _normalize_attachments_from_mapping(data: dict[str, Any]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for raw_attachment in list(data.get("attachments") or []) + list(data.get("images") or []):
        attachment = _normalize_attachment_payload(raw_attachment, len(normalized))
        if attachment is not None:
            normalized.append(attachment)
    return normalized


# Build LLM history entries from one request-side history message.
def _build_llm_entries_from_request_message(
    message: dict[str, Any],
    *,
    sandbox_enabled: bool = False,
    supported_media_kinds: set[str] | None = None,
) -> list[dict[str, Any]]:
    role = str(message.get("role") or "").strip().lower()
    if role == "assistant":
        transcript_entries = _normalize_transcript_entries(message.get("llm_transcript"))
        return _llm_entries_from_assistant_transcript(
            transcript_entries,
            content_fallback=str(message.get("content") or ""),
        )

    if role not in {"user", "system", "tool"}:
        return []

    if role in {"user", "system"}:
        payload = {"role": role, "content": str(message.get("content") or "")}
        attachments = _normalize_attachments_from_mapping(message)
        payload = _apply_attachments_to_llm_entry(payload, attachments)
        upload_manifests = _load_model_upload_manifests(
            _normalize_uploaded_file_ids(message),
            sandbox_enabled=sandbox_enabled,
        )
        payload = _apply_uploaded_file_manifests_to_llm_entry(
            payload,
            upload_manifests or [],
            supported_media_kinds=supported_media_kinds,
        )
        if not str(payload.get("content") or "").strip() and not attachments and not upload_manifests:
            return []
        return [payload]

    tool_payload: dict[str, Any] = {
        "role": "tool",
        "content": str(message.get("content") or ""),
    }
    for key in ("tool_call_id", "name", "tool_name", "alias", "tool_id"):
        if message.get(key):
            tool_payload[key] = message[key]
    return [tool_payload]


# Build LLM history entries
def _build_llm_history_entries(
    message: Message,
    *,
    sandbox_enabled: bool = False,
    supported_media_kinds: set[str] | None = None,
) -> list[dict[str, Any]]:
    if message.role != "assistant":
        payload = {"role": message.role, "content": message.content}
        payload = _apply_attachments_to_llm_entry(payload, _get_message_attachments(message))
        payload = _apply_uploaded_file_manifests_to_llm_entry(
            payload,
            _load_message_upload_manifests(message, sandbox_enabled=sandbox_enabled),
            supported_media_kinds=supported_media_kinds,
        )
        return [payload]

    transcript_entries = _normalize_transcript_entries(message.llm_transcript)
    return _llm_entries_from_assistant_transcript(
        transcript_entries,
        content_fallback=message.content,
    )


# Return whether one stored assistant message represents a compression marker.
def _message_has_context_compression_summary(message: Message) -> bool:
    if getattr(message, "role", "") != "assistant":
        return False
    for entry in _normalize_transcript_entries(getattr(message, "llm_transcript", None)):
        if entry.get("role") == "tool" and str(entry.get("alias") or "") == "context_compression_summary":
            return bool(str(entry.get("content") or "").strip())
    return False


# Build chronological non-compression entries represented by a new boundary.
def _build_context_compression_source_entries(
    history_records: list[Message],
    *,
    sandbox_enabled: bool = False,
) -> list[dict[str, Any]]:
    boundary_records: list[Message] = []
    for historical_message in history_records:
        if _message_has_context_compression_summary(historical_message):
            break
        boundary_records.append(historical_message)

    entries: list[dict[str, Any]] = []
    for historical_message in reversed(boundary_records):
        entries.extend(_build_llm_history_entries(historical_message, sandbox_enabled=sandbox_enabled))
    return entries


# Build activity segments
def _build_activity_segments(message: Message) -> list[dict[str, Any]]:
    transcript_entries = _normalize_transcript_entries(message.llm_transcript)
    if not transcript_entries:
        return []

    def canonical_alias(value: Any) -> str:
        return re.sub(r"__\d+$", "", str(value or "").strip())

    # Index tool results by alias while preserving order. Some runtimes can
    # emit repeated aliases (for example "sandbox__share_file__0" for several
    # different files in one answer), so a simple dict overwrite would keep
    # only the last file after reload.
    tool_results: dict[str, list[dict[str, Any]]] = {}
    for entry in transcript_entries:
        if entry.get("role") == "tool":
            # Prefer the full alias stored by _build_tool_message, fall back to tool_id/name.
            alias = str(entry.get("alias") or entry.get("tool_id") or entry.get("name") or "")
            if alias:
                tool_results.setdefault(alias, []).append({
                    "content": str(entry.get("content") or ""),
                    "toolUi": entry.get("tool_ui") if isinstance(entry.get("tool_ui"), dict) else None,
                    "structuredContent": entry.get("structured_content") if isinstance(entry.get("structured_content"), dict) else None,
                    "entry": entry,
                })

    def pop_tool_result(alias: str) -> tuple[str, dict[str, Any] | None]:
        queue = tool_results.get(alias) or []
        if queue:
            return alias, queue.pop(0)
        normalized = canonical_alias(alias)
        for candidate_alias, candidate_queue in tool_results.items():
            if candidate_queue and canonical_alias(candidate_alias) == normalized:
                return candidate_alias, candidate_queue.pop(0)
        return alias, None

    def split_alias(alias: str) -> tuple[str, str]:
        normalized = canonical_alias(alias)
        if "__" not in normalized:
            return "", normalized
        return tuple(normalized.split("__", 1))  # type: ignore[return-value]

    def pending_deep_research_ui(alias: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        server_id, tool_id = split_alias(alias)
        if server_id != "deep_research" and tool_id != "deep_research":
            return None
        try:
            rounds = min(25, max(2, int(arguments.get("max_rounds") or 6)))
        except (TypeError, ValueError):
            rounds = 6
        return {
            "kind": "deep_research",
            "status": "planning",
            "topic": str(arguments.get("topic") or "").strip(),
            "session_id": str(arguments.get("session_id") or "").strip(),
            "plan_version": 0,
            "checklist": [],
            "queries_used": 0,
            "query_budget": rounds * 2,
            "can_approve": False,
            "can_edit": False,
            "can_stop": True,
        }

    segments: list[dict[str, Any]] = []
    consumed_result_entries: dict[str, int] = {}
    for entry in transcript_entries:
        if entry["role"] == "assistant":
            thinking = str(entry.get("thinking", "") or "").strip()
            content = str(entry.get("content", "") or "").strip()
            if thinking:
                segments.append({"type": "thought", "content": thinking})
            if content:
                segments.append({"type": "text", "content": content})
            for raw_call in entry.get("tool_calls") if isinstance(entry.get("tool_calls"), list) else []:
                if not isinstance(raw_call, dict):
                    continue
                function_payload = raw_call.get("function")
                if not isinstance(function_payload, dict):
                    continue
                alias = str(function_payload.get("name") or "").strip()
                if not alias:
                    continue
                raw_arguments = function_payload.get("arguments")
                if isinstance(raw_arguments, str):
                    try:
                        decoded_arguments = json.loads(raw_arguments)
                    except json.JSONDecodeError:
                        decoded_arguments = {}
                    arguments = decoded_arguments if isinstance(decoded_arguments, dict) else {}
                else:
                    arguments = raw_arguments if isinstance(raw_arguments, dict) else {}
                result_alias, result_payload = pop_tool_result(alias)
                server_id, tool_id = split_alias(alias)
                result_entry = result_payload.get("entry") if isinstance(result_payload, dict) else {}
                segment = {
                    "type": "tool",
                    "alias": result_alias or alias,
                    "serverId": str(result_entry.get("server_id") or server_id),
                    "serverName": str(result_entry.get("server_name") or server_id.replace("_", " ").title()),
                    "toolId": str(result_entry.get("tool_id") or tool_id),
                    "toolName": str(
                        result_entry.get("tool_display_name")
                        or result_entry.get("tool_name")
                        or tool_id.replace("_", " ").title()
                    ),
                    "arguments": arguments,
                    "result": result_payload.get("content") if isinstance(result_payload, dict) else None,
                }
                tool_ui = result_payload.get("toolUi") if isinstance(result_payload, dict) else None
                if not isinstance(tool_ui, dict):
                    tool_ui = pending_deep_research_ui(alias, arguments)
                if isinstance(tool_ui, dict):
                    segment["toolUi"] = tool_ui
                structured_content = (
                    result_payload.get("structuredContent") if isinstance(result_payload, dict) else None
                )
                if isinstance(structured_content, dict):
                    segment["structuredContent"] = structured_content
                segments.append(segment)
                if result_payload is not None:
                    normalized_result_alias = canonical_alias(result_alias)
                    consumed_result_entries[normalized_result_alias] = (
                        consumed_result_entries.get(normalized_result_alias, 0) + 1
                    )
            continue

        seg_alias = str(entry.get("alias") or entry.get("tool_id", entry.get("name", "")) or "")
        normalized_seg_alias = canonical_alias(seg_alias)
        if consumed_result_entries.get(normalized_seg_alias, 0) > 0:
            consumed_result_entries[normalized_seg_alias] -= 1
            continue
        payload_queue = tool_results.get(seg_alias) or []
        if payload_queue:
            result_payload = payload_queue.pop(0)
        else:
            result_payload = {
                "content": str(entry.get("content") or ""),
                "toolUi": entry.get("tool_ui") if isinstance(entry.get("tool_ui"), dict) else None,
                "structuredContent": entry.get("structured_content") if isinstance(entry.get("structured_content"), dict) else None,
            }
        segment = {
            "type": "tool",
            "alias": seg_alias,
            "serverId": str(entry.get("server_id", "") or ""),
            "serverName": str(entry.get("server_name", "") or ""),
            "toolId": str(entry.get("tool_id", entry.get("name", "")) or ""),
            "toolName": str(entry.get("tool_display_name", entry.get("tool_name", entry.get("name", ""))) or ""),
            "arguments": entry.get("arguments") if isinstance(entry.get("arguments"), dict) else {},
            "result": result_payload.get("content") if isinstance(result_payload, dict) else None,
        }
        if isinstance(result_payload, dict):
            if isinstance(result_payload.get("toolUi"), dict):
                segment["toolUi"] = result_payload["toolUi"]
            if isinstance(result_payload.get("structuredContent"), dict):
                segment["structuredContent"] = result_payload["structuredContent"]
        segments.append(segment)

    return segments


# Return whether the stored transcript contains model reasoning.
def _message_has_reasoning_segments(message: Message) -> bool:
    return any(
        entry.get("role") == "assistant" and str(entry.get("thinking", "") or "").strip()
        for entry in _normalize_transcript_entries(message.llm_transcript)
    )


# Serialize message
def _serialize_message(message: Message, *, include_attachment_data: bool = True) -> dict[str, Any]:
    payload = {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
    }
    activity_segments = _build_activity_segments(message)
    if activity_segments:
        payload["activity_segments"] = activity_segments
        payload["reasoning_mode"] = _message_has_reasoning_segments(message)
    attachments = _get_message_attachments(message, include_data=include_attachment_data)
    if message.role == "user":
        for manifest in _load_message_upload_manifests(message, sandbox_enabled=True):
            try:
                uploaded_payload = public_upload_payload(manifest)
            except Exception:
                continue
            if uploaded_payload:
                uploaded_payload["kind"] = "file"
                attachments.append(uploaded_payload)
    if attachments:
        payload["attachments"] = attachments
        payload["images"] = [
            item.get("data_url") or item.get("content_url")
            for item in attachments
            if item.get("kind") == MessageAttachmentKind.IMAGE and (item.get("data_url") or item.get("content_url"))
        ]
    return payload


# Extract streamed message parts
def _extract_stream_message_parts(chunk: Any) -> tuple[str, str]:
    raw_message = chunk.get("message", {}) if isinstance(chunk, dict) else getattr(chunk, "message", {})
    if isinstance(raw_message, dict):
        thinking_part = raw_message.get("thinking", "") or ""
        text_part = raw_message.get("content", "") or ""
    else:
        thinking_part = getattr(raw_message, "thinking", "") or ""
        text_part = getattr(raw_message, "content", "") or ""
    return _strip_llm_control_tokens(str(thinking_part)), _strip_llm_control_tokens(str(text_part))


# Return transcript entries safe to persist while a response is streaming.
def _copy_transcript_entries_for_storage(transcript_entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in transcript_entries:
        if isinstance(entry, dict):
            entries.append(copy.deepcopy(entry))
    return entries


# Overlay streamed assistant text onto machine transcript entries.
def _build_streaming_assistant_transcript(
    transcript_entries: list[dict[str, Any]],
    *,
    visible_content: str,
    thinking_content: str,
) -> list[dict[str, Any]]:
    entries = _copy_transcript_entries_for_storage(transcript_entries)
    has_assistant_content = False
    has_assistant_thinking = False
    for entry in entries:
        if str(entry.get("role") or "").strip().lower() != "assistant":
            continue
        if _strip_llm_control_tokens(str(entry.get("content") or "")).strip():
            has_assistant_content = True
        if _strip_llm_control_tokens(str(entry.get("thinking") or "")).strip():
            has_assistant_thinking = True

    buffered_entry: dict[str, Any] = {"role": "assistant", "content": ""}
    if visible_content and not has_assistant_content:
        buffered_entry["content"] = visible_content
    if thinking_content and not has_assistant_thinking:
        buffered_entry["thinking"] = thinking_content
    if buffered_entry.get("content") or buffered_entry.get("thinking"):
        entries.append(buffered_entry)
    return entries


# Serialize tool marker
def _serialize_tool_call_marker(tool_event: dict[str, Any]) -> str:
    payload = {
        "alias": str(tool_event.get("alias", "") or "").strip(),
        "server_id": str(tool_event.get("server_id", "") or "").strip(),
        "server_name": str(tool_event.get("server_name", "") or "").strip(),
        "tool_id": str(tool_event.get("tool_id", "") or "").strip(),
        "tool_name": str(tool_event.get("tool_name", "") or "").strip(),
        "arguments": tool_event.get("arguments") or {},
    }
    if isinstance(tool_event.get("tool_ui"), dict):
        payload["tool_ui"] = tool_event["tool_ui"]
    return f'<tool_call>{json.dumps(payload, ensure_ascii=False)}</tool_call>'


# Serialize tool result marker
def _serialize_tool_result_marker(
    alias: str,
    content: str,
    *,
    tool_ui: dict[str, Any] | None = None,
    structured_content: dict[str, Any] | None = None,
) -> str:
    payload = {"alias": alias, "content": content}
    if isinstance(tool_ui, dict):
        payload["tool_ui"] = tool_ui
    if isinstance(structured_content, dict):
        payload["structured_content"] = structured_content
    return f'<tool_result>{json.dumps(payload, ensure_ascii=False)}</tool_result>'


# Encode one realtime tool activity event for the frontend event timeline.
def _serialize_tool_activity_marker(activity: dict[str, Any]) -> str:
    return f'<tool_activity>{json.dumps(activity, ensure_ascii=False)}</tool_activity>'


# Encode a context-compression boundary without pretending it is a model tool.
def _serialize_context_compression_marker(compression_event: dict[str, Any]) -> str:
    payload = {
        "alias": str(compression_event.get("alias") or "context_compression_summary").strip(),
        "server_id": str(compression_event.get("server_id") or "system").strip(),
        "server_name": str(compression_event.get("server_name") or "System").strip(),
        "tool_id": str(compression_event.get("tool_id") or "context_compression_summary").strip(),
        "tool_name": str(compression_event.get("tool_display_name") or compression_event.get("tool_name") or "Context Compression").strip(),
        "arguments": compression_event.get("arguments") if isinstance(compression_event.get("arguments"), dict) else {},
        "content": str(compression_event.get("content") or ""),
    }
    tool_ui = compression_event.get("tool_ui")
    if isinstance(tool_ui, dict):
        payload["tool_ui"] = tool_ui
    structured_content = compression_event.get("structured_content")
    if isinstance(structured_content, dict):
        payload["structured_content"] = structured_content
    return f'<context_compression>{json.dumps(payload, ensure_ascii=False)}</context_compression>'


# Extract Ollama model info
def _normalize_capability_tokens(capabilities: Any) -> set[str]:
    if capabilities is None:
        return set()

    raw_items: list[Any]
    if isinstance(capabilities, dict):
        raw_items = [key for key, value in capabilities.items() if bool(value)]
    elif isinstance(capabilities, str):
        raw_items = re.split(r"[\s,;|]+", capabilities)
    elif isinstance(capabilities, (list, tuple, set)):
        raw_items = list(capabilities)
    else:
        raw_items = [capabilities]

    tokens: set[str] = set()
    for item in raw_items:
        value = str(item or "").strip().lower()
        if not value:
            continue
        tokens.add(value)
        if "." in value:
            tokens.add(value.rsplit(".", 1)[-1])
    return tokens


# Return whether one Ollama chat template can serialize tools.
def _ollama_template_supports_tool_calling(template: str) -> bool:
    normalized_template = str(template or "")
    if not normalized_template:
        return False

    folded_template = normalized_template.lower()
    return any(marker.lower() in folded_template for marker in OLLAMA_TOOL_TEMPLATE_MARKERS)


# Return whether Ollama metadata is strong enough to expose local tools.
def _ollama_metadata_supports_tool_calling(
    capabilities: Any,
    template: str,
) -> bool:
    if capabilities is not None:
        return bool(_normalize_capability_tokens(capabilities) & TOOL_CAPABILITY_NAMES)

    # Older/custom Ollama responses may not expose `capabilities`. In that
    # case, fall back to whether the model template can serialize tool calls.
    return _ollama_template_supports_tool_calling(template)


# Parse Ollama-specific model metadata into a frontend-friendly payload.
def _extract_ollama_model_info(settings_data: Any) -> dict[str, Any]:
    context_length = 8192
    model_layers = 0
    defaults: dict[str, Any] = {}

    # Read raw metadata from dict-like or SDK responses.
    if isinstance(settings_data, dict):
        modelinfo = settings_data.get("modelinfo", settings_data.get("model_info", {})) or {}
        parameters_str = settings_data.get("parameters", "") or ""
        template_str = settings_data.get("template", "") or ""
        capabilities = settings_data.get("capabilities")
    else:
        modelinfo = getattr(settings_data, "modelinfo", {}) or {}
        parameters_str = getattr(settings_data, "parameters", "") or ""
        template_str = getattr(settings_data, "template", "") or ""
        capabilities = getattr(settings_data, "capabilities", None)

    # Extract numeric limits from Ollama's flat metadata keys.
    for key, value in modelinfo.items():
        if key.endswith(".context_length"):
            try:
                context_length = int(value)
            except (TypeError, ValueError):
                pass
        if key.endswith(".block_count"):
            try:
                model_layers = int(value)
            except (TypeError, ValueError):
                pass

    # Parse default runtime parameters from the Modelfile-like payload.
    if parameters_str:
        for line in parameters_str.strip().splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            if parts[0].strip().lower() == "parameter":
                if len(parts) < 3:
                    continue
                key = parts[1].strip().lower()
                value = " ".join(parts[2:]).strip()
            else:
                key = parts[0].strip().lower()
                value = " ".join(parts[1:]).strip()
            if not ollama_api.is_supported_runtime_option_key(key):
                continue
            normalized = settings.normalize_setting_value(value)
            if key == "stop":
                existing = defaults.get("stop")
                if isinstance(existing, list):
                    existing.append(normalized)
                else:
                    defaults["stop"] = [normalized]
            else:
                defaults[key] = normalized

    think_param_name = "think"
    think_level_param_name = "think_level"
    normalized_capabilities = _normalize_capability_tokens(capabilities)

    # Detect supported features from the template, defaults, and capabilities.
    supports_thinking = any(
        marker in template_str
        for marker in (".Think ", ".Think\n", ".ThinkLevel", ".Reasoning", ".Reason ")
    )
    supports_think_level = any(
        marker in template_str for marker in (".ThinkLevel", ".ReasoningEffort")
    )
    supports_think_toggle = False

    if "thinking" in normalized_capabilities:
        supports_thinking = True

    for candidate in THINK_PARAM_NAMES:
        if candidate in defaults:
            think_param_name = candidate
            supports_think_toggle = True
            supports_thinking = True
            break

    for candidate in THINK_LEVEL_PARAM_NAMES:
        if candidate in defaults:
            think_level_param_name = candidate
            supports_think_level = True
            break

    # Ollama exposes plain thinking as a boolean `think` request field, but
    # level-based models such as gpt-oss use the same field for low/medium/high.
    # Do not add a fake Off option when the model advertises think levels.
    if supports_thinking and not supports_think_level:
        supports_think_toggle = True

    supports_vision = "vision" in normalized_capabilities
    supports_audio_input = bool(normalized_capabilities & {"audio", "audio_input", "input_audio"})
    supports_video_input = bool(normalized_capabilities & {"video", "video_input", "input_video"})
    supports_tool_calling = _ollama_metadata_supports_tool_calling(capabilities, template_str)

    # Runtime limits are used by the frontend controls.
    cpu_threads = max(int(os.cpu_count() or 1), 1)
    gpu_devices = _get_local_gpu_devices()
    gpu_count = len(gpu_devices)

    return {
        "context_length": context_length,
        "model_layers": model_layers,
        "defaults": defaults,
        "supports_thinking": supports_thinking,
        "supports_think_toggle": supports_think_toggle,
        "supports_think_level": supports_think_level,
        "think_param_name": think_param_name,
        "think_level_param_name": think_level_param_name,
        "supports_vision": supports_vision,
        "supports_audio_input": supports_audio_input,
        "supports_video_input": supports_video_input,
        "supports_tool_calling": supports_tool_calling,
        "supports_files": False,
        "runtime_limits": {
            "cpu_threads": cpu_threads,
            "gpu_count": gpu_count,
            "gpu_devices": gpu_devices,
            "main_gpu_max": max(gpu_count - 1, 0),
            "model_layers": model_layers,
        },
    }


# Missing layer metadata alone is not evidence that a model runs in the cloud.
def _ollama_model_is_cloud(model_name: str, settings_data: Any) -> bool:
    name = str(model_name or "").strip().lower()
    if name.endswith("-cloud") or name.endswith(":cloud") or "-cloud:" in name:
        return True
    if isinstance(settings_data, dict):
        return bool(settings_data.get("remote_model") or settings_data.get("remote_host"))
    return bool(getattr(settings_data, "remote_model", None) or getattr(settings_data, "remote_host", None))


# Extract generic model info
def _extract_generic_model_info(settings_data: Any) -> dict[str, Any]:
    if not isinstance(settings_data, dict):
        return {
            "context_length": 8192,
            "defaults": {},
            "supports_thinking": False,
            "supports_think_toggle": False,
            "supports_think_level": False,
            "think_param_name": "think",
            "think_level_param_name": "think_level",
            "think_level_options": [],
            "supports_vision": False,
            "supports_audio_input": False,
            "supports_video_input": False,
            "supports_tool_calling": False,
            "supports_files": False,
            "runtime_limits": {},
            "custom_fields": [],
        }

    capabilities = settings_data.get("capabilities", []) or []
    normalized_capabilities = {str(item).strip().lower() for item in capabilities}
    defaults = settings_data.get("defaults", settings_data.get("parameters", {})) or {}
    if not isinstance(defaults, dict):
        defaults = {}

    supports_think_level = bool(settings_data.get("supports_think_level", False))
    think_level_param_name = str(settings_data.get("think_level_param_name", "think_level") or "think_level")
    think_level_options = (
        settings_data.get("think_level_options", [])
        if isinstance(settings_data.get("think_level_options", []), list)
        else []
    )
    if supports_think_level and not think_level_options:
        if think_level_param_name == "reasoning_effort":
            think_level_options = ["minimal", "low", "medium", "high", "xhigh"]
        elif think_level_param_name == "thinking_level":
            think_level_options = ["minimal", "low", "medium", "high"]
        else:
            think_level_options = ["low", "medium", "high"]

    context_length = (
        settings_data.get("context_length")
        or settings_data.get("max_context_window")
        or settings_data.get("max_tokens")
        or 8192
    )

    return {
        "context_length": int(context_length),
        "defaults": defaults,
        "supports_thinking": bool(settings_data.get("supports_thinking", False)),
        "supports_think_toggle": bool(settings_data.get("supports_think_toggle", False)),
        "supports_think_level": supports_think_level,
        "think_param_name": settings_data.get("think_param_name", "think"),
        "think_level_param_name": think_level_param_name,
        "think_level_options": think_level_options,
        "supported_parameters": settings_data.get("supported_parameters", []) if isinstance(settings_data.get("supported_parameters", []), list) else [],
        "supports_vision": "vision" in normalized_capabilities or bool(settings_data.get("supports_vision", False)),
        "supports_audio_input": bool(
            settings_data.get("supports_audio_input", False)
            or normalized_capabilities & {"audio", "audio_input", "input_audio"}
        ),
        "supports_video_input": bool(
            settings_data.get("supports_video_input", False)
            or normalized_capabilities & {"video", "video_input", "input_video"}
        ),
        "supports_tool_calling": bool(settings_data.get("supports_tool_calling", False)),
        "supports_files": bool(settings_data.get("supports_files", False)),
        "capabilities_source": settings_data.get("capabilities_source", ""),
        "metadata_fallback": bool(settings_data.get("metadata_fallback", False)),
        "runtime_limits": settings_data.get("runtime_limits", {}) if isinstance(settings_data.get("runtime_limits", {}), dict) else {},
        "custom_fields": settings_data.get("custom_fields", []) if isinstance(settings_data.get("custom_fields", []), list) else [],
    }


# Build model info payload
def _build_fallback_model_info_payload(engine: str, model_name: str) -> dict[str, Any]:
    payload = _extract_generic_model_info({})
    payload["available_tool_servers"] = []
    payload["model"] = model_name
    payload["engine"] = engine
    payload["metadata_fallback"] = True
    return payload


# Load adapter metadata and normalize it for the frontend.
def _extract_model_info(engine: str, model_name: str, settings_data: Any) -> dict[str, Any]:
    if settings.is_ollama_engine(engine):
        payload = _extract_ollama_model_info(settings_data)
        payload["is_cloud"] = _ollama_model_is_cloud(model_name, settings_data)
        return payload
    return _extract_generic_model_info(settings_data)


def _build_model_info_payload(
    engine: str,
    model_name: str,
    *,
    allow_fallback: bool = False,
) -> dict[str, Any]:
    cached_payload = _get_cached_model_info(engine, model_name)
    if cached_payload is not None:
        _sync_runtime_model_metadata(
            engine,
            model_name,
            cached_payload,
            source="model_info_cache",
            route="/api/model_info/",
        )
        return cached_payload

    endpoint = model_capabilities.get_endpoint(engine)
    try:
        settings_data = llm_api.get_model_settings(engine, model_name)
    except Exception:
        if not allow_fallback:
            raise
        return _build_fallback_model_info_payload(engine, model_name)
    payload = _extract_model_info(engine, model_name, settings_data)
    # Keep only capability facts on disk, not presets, API keys or runtime settings.
    if endpoint != model_capabilities.get_endpoint(engine):
        raise RuntimeError("Engine endpoint changed while loading model metadata")
    model_capabilities.store(engine, model_name, endpoint, payload)
    if settings.is_ollama_engine(engine):
        preset_payload = get_ollama_preset_payload(model_name)
        payload["defaults"] = {**payload.get("defaults", {}), **preset_payload["active_config"]}
        payload["ollama_presets"] = preset_payload
    elif engine == "lms":
        preset_payload = get_lms_preset_payload(model_name)
        active_config = preset_payload.get("active_config", {}) if isinstance(preset_payload, dict) else {}
        if isinstance(active_config, dict):
            payload["defaults"] = {
                **payload.get("defaults", {}),
                **(active_config.get("operation", {}) if isinstance(active_config.get("operation", {}), dict) else {}),
            }
        payload["lms_presets"] = preset_payload
    elif engine == "openai":
        # Scope presets by endpoint URL so the same model name from different
        # providers keeps independent presets.
        preset_payload = get_openai_preset_payload(settings.get_engine_url("openai"), model_name)
        active_config = preset_payload.get("active_config", {}) if isinstance(preset_payload, dict) else {}
        if isinstance(active_config, dict):
            payload["defaults"] = {**payload.get("defaults", {}), **active_config}
        payload["openai_presets"] = preset_payload
    elif engine == "google-genai":
        preset_payload = get_google_genai_preset_payload(model_name)
        active_config = preset_payload.get("active_config", {}) if isinstance(preset_payload, dict) else {}
        if isinstance(active_config, dict):
            payload["defaults"] = {**payload.get("defaults", {}), **active_config}
        payload["google_genai_presets"] = preset_payload
    payload["available_tool_servers"] = (
        _list_tool_servers_cached(engine, model_name) if payload.get("supports_tool_calling") else []
    )
    payload["model"] = model_name
    payload["engine"] = engine
    if endpoint != model_capabilities.get_endpoint(engine):
        raise RuntimeError("Engine endpoint changed while loading model metadata")
    cached_payload = _set_cached_model_info(engine, model_name, payload)
    _sync_runtime_model_metadata(
        engine,
        model_name,
        cached_payload,
        source="model_info",
        route="/api/model_info/",
    )
    return cached_payload


# Build a stable engine label for API payloads.
def _get_engine_label(engine: str) -> str:
    return getattr(settings, "ENGINE_LABELS", {}).get(engine, engine)


# Normalize model metadata into a compact runtime-inference payload.
def _build_inference_info_payload(
    engine: str,
    model_name: str,
    model_info_payload: dict[str, Any],
    model_source: str,
) -> dict[str, Any]:
    defaults = model_info_payload.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}

    runtime_limits = model_info_payload.get("runtime_limits", {})
    if not isinstance(runtime_limits, dict):
        runtime_limits = {}

    model_context_limit = _coerce_positive_int(model_info_payload.get("context_length"))
    current_context_window = _first_positive_int(defaults, CONTEXT_WINDOW_KEYS) or model_context_limit
    current_output_tokens = _first_positive_int(defaults, OUTPUT_TOKEN_KEYS)
    output_token_limit = (
        _first_positive_int(runtime_limits, OUTPUT_TOKEN_KEYS)
        or _coerce_positive_int(model_info_payload.get("output_token_limit"))
    )

    available_tool_servers = model_info_payload.get("available_tool_servers", [])
    if not isinstance(available_tool_servers, list):
        available_tool_servers = []

    capabilities = model_info_payload.get("capabilities", [])
    if not isinstance(capabilities, list):
        capabilities = []

    supported_parameters = model_info_payload.get("supported_parameters", [])
    if not isinstance(supported_parameters, list):
        supported_parameters = []

    return {
        "ok": True,
        "engine": engine,
        "engine_label": _get_engine_label(engine),
        "model": model_name,
        "model_name": model_name,
        "context_window": current_context_window,
        "model_context_limit": model_context_limit,
        "max_output_tokens": current_output_tokens,
        "output_token_limit": output_token_limit,
        "limits": {
            "context_window": current_context_window,
            "model_context_limit": model_context_limit,
            "max_output_tokens": current_output_tokens,
            "output_token_limit": output_token_limit,
        },
        "capabilities": {
            "items": capabilities,
            "supports_thinking": bool(model_info_payload.get("supports_thinking", False)),
            "supports_think_toggle": bool(
                model_info_payload.get("supports_think_toggle", False)
            ),
            "supports_think_level": bool(model_info_payload.get("supports_think_level", False)),
            "supports_vision": bool(model_info_payload.get("supports_vision", False)),
            "supports_audio_input": bool(model_info_payload.get("supports_audio_input", False)),
            "supports_video_input": bool(model_info_payload.get("supports_video_input", False)),
            "supports_tool_calling": bool(model_info_payload.get("supports_tool_calling", False)),
            "supports_files": bool(model_info_payload.get("supports_files", False)),
        },
        "generation_defaults": defaults,
        "supported_parameters": supported_parameters,
        "runtime_limits": runtime_limits,
        "tool_servers": available_tool_servers,
        "source": {
            "model": model_source,
        },
    }


# Describe one local metadata source without freezing dynamic ports.
def _runtime_metadata_source(name: str, route: str, port_setting: str) -> dict[str, Any]:
    return {
        "name": name,
        "type": "local_http_route",
        "route": route,
        "host": "127.0.0.1",
        "port_setting": port_setting,
        "port_sources": [
            "ASLM_Module.json settings[].value/default",
            "Settings/settings.json",
            "environment override when available",
        ],
    }


# Return runtime metadata file.
def _read_runtime_metadata_file() -> dict[str, Any]:
    try:
        with MODEL_RUNTIME_METADATA_PATH.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


# Write runtime metadata file.
def _write_runtime_metadata_file(payload: dict[str, Any]) -> None:
    MODEL_RUNTIME_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = MODEL_RUNTIME_METADATA_PATH.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temp_path, MODEL_RUNTIME_METADATA_PATH)


# Persist active model metadata for local tools in real time.
def _sync_runtime_model_metadata(
    engine: str,
    model_name: str,
    model_info_payload: dict[str, Any],
    *,
    source: str,
    route: str,
) -> None:
    if not engine or not model_name or not isinstance(model_info_payload, dict):
        return
    if model_info_payload.get("metadata_fallback"):
        return

    defaults = model_info_payload.get("defaults", {})
    if not isinstance(defaults, dict):
        defaults = {}
    runtime_limits = model_info_payload.get("runtime_limits", {})
    if not isinstance(runtime_limits, dict):
        runtime_limits = {}

    model_context_limit = _coerce_positive_int(model_info_payload.get("context_length"))
    context_window = _first_positive_int(defaults, CONTEXT_WINDOW_KEYS) or model_context_limit
    max_output_tokens = _first_positive_int(defaults, OUTPUT_TOKEN_KEYS)
    output_token_limit = (
        _first_positive_int(runtime_limits, OUTPUT_TOKEN_KEYS)
        or _coerce_positive_int(model_info_payload.get("output_token_limit"))
    )

    payload = _read_runtime_metadata_file()
    models = payload.get("models", {})
    if not isinstance(models, dict):
        models = {}

    model_key = f"{engine}:{model_name}"
    existing = models.get(model_key, {})
    runtime = existing.get("runtime", {}) if isinstance(existing, dict) else {}
    if not isinstance(runtime, dict):
        runtime = {}

    sources = [
        {
            **_runtime_metadata_source("ASLM model info", route, "ui-port"),
            "notes": "ASLM-normalized active model metadata written during model selection or generation.",
        }
    ]
    if settings.is_ollama_engine(engine):
        sources.extend(
            [
                {
                    **_runtime_metadata_source("Ollama loaded models", "/api/ps", "ollama-service_port"),
                    "notes": "Provider runtime source for loaded model state and loaded context_length.",
                },
                {
                    **_runtime_metadata_source("Ollama model metadata", "/api/show", "ollama-service_port"),
                    "notes": "Provider model metadata source for capabilities and model limits.",
                },
            ]
        )

    models[model_key] = {
        "engine": engine,
        "model": model_name,
        "capabilities": {
            "vision": bool(model_info_payload.get("supports_vision", False)),
            "tools": bool(model_info_payload.get("supports_tool_calling", False)),
            "thinking": bool(model_info_payload.get("supports_thinking", False)),
            "files": bool(model_info_payload.get("supports_files", False)),
        },
        "limits": {
            "context_window": context_window,
            "model_context_limit": model_context_limit,
            "max_output_tokens": max_output_tokens,
            "output_token_limit": output_token_limit,
        },
        "runtime": runtime,
        "sources": sources,
    }

    payload.update(
        {
            "schema_version": 1,
            "_comment": (
                "Runtime model metadata consumed by local tools. Ports are dynamic; "
                "resolve local_http_route sources through ASLM_Module.json and Settings/settings.json."
            ),
            "updated_at": timezone.now().replace(microsecond=0).isoformat().replace("+00:00", "Z"),
            "active": {
                "engine": engine,
                "engine_label": _get_engine_label(engine),
                "model": model_name,
                "source": source,
                "primary_source": {
                    **_runtime_metadata_source("ASLM model info", route, "ui-port"),
                    "fields": [
                        "engine",
                        "model",
                        "supports_vision",
                        "context_length",
                        "defaults",
                    ],
                },
            },
            "models": models,
        }
    )

    try:
        _write_runtime_metadata_file(payload)
    except OSError as exc:
        logger.debug("Failed to write runtime model metadata: %s", exc)


# Read JSON body
def _read_json_request_body(request) -> dict[str, Any]:
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON format") from exc

    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object")
    return data


# Resolve selected tool servers
def _build_tool_source_map(
    tool_sources: list[dict[str, Any]] | None,
    selected_tool_servers: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, str]]:
    tool_source_map: dict[str, dict[str, str]] = {}

    for source in tool_sources or []:
        if not isinstance(source, dict):
            continue
        module_dir = str(source.get("module_dir") or "").strip()
        if not module_dir:
            continue
        tools_dir = str(source.get("tools_dir") or (Path(module_dir) / "Tools"))
        raw_ids = source.get("tool_server_ids") or source.get("tool_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids] if raw_ids.strip() else []
        for raw_id in raw_ids:
            server_id = str(raw_id or "").strip()
            if server_id:
                tool_source_map[server_id] = {
                    "module_dir": module_dir,
                    "tools_dir": tools_dir,
                }

    for server in selected_tool_servers or []:
        if not isinstance(server, dict):
            continue
        server_id = str(server.get("id") or "").strip()
        if not server_id or server_id in tool_source_map:
            continue
        module_dir = str(server.get("source_module_dir") or "").strip()
        if not module_dir:
            continue
        tools_dir = str(server.get("source_tools_dir") or (Path(module_dir) / "Tools"))
        tool_source_map[server_id] = {
            "module_dir": module_dir,
            "tools_dir": tools_dir,
        }

    return tool_source_map


# Parse external tool source declarations from one request payload.
def _parse_tool_sources(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw_sources = data.get("tool_sources")
    if not isinstance(raw_sources, list):
        return []

    parsed: list[dict[str, Any]] = []
    for entry in raw_sources:
        if not isinstance(entry, dict):
            continue
        module_dir = str(entry.get("module_dir") or "").strip()
        if not module_dir:
            continue
        raw_ids = entry.get("tool_server_ids") or entry.get("tool_ids") or []
        if isinstance(raw_ids, str):
            raw_ids = [raw_ids] if raw_ids.strip() else []
        tool_server_ids = [str(item).strip() for item in raw_ids if str(item).strip()]
        parsed.append(
            {
                "module_dir": module_dir,
                "tools_dir": str(entry.get("tools_dir") or (Path(module_dir) / "Tools")),
                "tool_server_ids": tool_server_ids,
            }
        )
    return parsed


# Resolve selected tool servers
def _resolve_tool_servers(
    engine: str,
    model_name: str,
    tool_server_ids: list[str],
    *,
    tool_sources: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    resolved: list[dict[str, Any]] = []
    seen: set[str] = set()

    for source in tool_sources or []:
        module_dir = str(source.get("module_dir") or "").strip()
        tools_dir = source.get("tools_dir") or (Path(module_dir) / "Tools")
        for raw_id in source.get("tool_server_ids") or []:
            normalized = str(raw_id or "").strip()
            if not normalized or normalized in seen:
                continue
            server = tool_registry.get_server(
                normalized,
                engine=engine,
                model_name=model_name,
                tools_dir=tools_dir,
                module_dir=module_dir,
            )
            if server is None:
                raise ValueError(f"Unknown or unsupported tool server: {normalized}")
            resolved.append(server)
            seen.add(normalized)

    for raw_id in tool_server_ids:
        normalized = str(raw_id or "").strip()
        if not normalized or normalized in seen:
            continue
        server = tool_registry.get_server(normalized, engine=engine, model_name=model_name)
        if server is None:
            raise ValueError(f"Unknown or unsupported tool server: {normalized}")
        resolved.append(server)
        seen.add(normalized)
    return resolved


# Raise when tools are requested for a model that should not call tools.
def _validate_tool_server_support(
    engine: str,
    model_name: str,
    tool_server_ids: list[str],
    payload: dict[str, Any] | None = None,
) -> None:
    if not tool_server_ids:
        return

    payload = payload or _build_model_info_payload(engine, model_name, allow_fallback=True)
    if payload.get("supports_tool_calling"):
        return

    raise ValueError(f"Model does not support tool calling: {model_name}")

# Parse stored tool slugs
def _parse_active_tool_slugs(slug: str) -> list[str]:
    import json as _json
    if not slug:
        return []
    try:
        parsed = _json.loads(slug)
        if isinstance(parsed, list):
            return [str(s) for s in parsed if str(s).strip()]
    except (ValueError, TypeError):
        pass
    # Legacy: single plain string
    return [slug] if slug.strip() else []


# Resolve a bounded history budget from model metadata.
def _resolve_history_char_budget(
    model_info_payload: dict[str, Any] | None,
    *,
    active_engine: str = "",
    active_model: str = "",
    observed_chars_per_token: float | None = None,
) -> int:
    context_length = _context_window_tokens_from_model_info(model_info_payload)
    return _history_char_budget_from_context_window(
        context_length,
        model_info_payload=model_info_payload,
        active_engine=active_engine,
        active_model=active_model,
        observed_chars_per_token=observed_chars_per_token,
        minimum_chars=16000,
        fallback_chars=LLM_HISTORY_DEFAULT_CHAR_BUDGET,
    )


# Estimate the prompt cost of one normalized LLM entry.
def _estimate_llm_entry_chars(entry: dict[str, Any]) -> int:
    if not isinstance(entry, dict):
        return 0

    cost = len(str(entry.get("content") or ""))
    cost += len(str(entry.get("thinking") or ""))
    if isinstance(entry.get("tool_calls"), list):
        cost += len(json.dumps(entry["tool_calls"], ensure_ascii=False))
    if isinstance(entry.get("google_parts"), list):
        cost += len(json.dumps(entry["google_parts"], ensure_ascii=False))

    images = entry.get("images") if isinstance(entry.get("images"), list) else []
    cost += len(images) * 4096
    media = entry.get("media") if isinstance(entry.get("media"), list) else []
    for item in media:
        if isinstance(item, dict):
            cost += max(4096, len(str(item.get("data") or "")) // 4)
    return cost


# Approximate token count from UTF-8 character count.
def _estimate_tokens_from_chars(char_count: int) -> int:
    try:
        chars = max(0, int(char_count))
    except (TypeError, ValueError):
        chars = 0
    if chars <= 0:
        return 0
    # Backward-compatible fallback for legacy call sites.
    return max(1, int(round(chars / 2.4)))


# Return one base chars/token hint from model family and engine metadata.
def _model_chars_per_token_hint(
    *,
    model_info_payload: dict[str, Any] | None,
    active_engine: str,
    active_model: str,
) -> float:
    model_name = str(active_model or (model_info_payload or {}).get("model") or "").strip().lower()
    engine_name = str(active_engine or "").strip().lower()

    # Conservative defaults that are still closer to real behavior than a fixed
    # chars/1.5 estimate. Lower value => more tokens estimated.
    base_ratio = 2.4
    if "qwen" in model_name:
        base_ratio = 2.0
    elif "deepseek" in model_name:
        base_ratio = 2.2
    elif "llama" in model_name or "mistral" in model_name or "mixtral" in model_name:
        base_ratio = 2.7
    elif "gemma" in model_name:
        base_ratio = 2.5
    elif "gpt" in model_name or "o3" in model_name or "o4" in model_name:
        base_ratio = 2.8

    if engine_name in {"google-genai", "google_genai", "gemini"}:
        base_ratio = min(base_ratio + 0.2, 3.0)
    return base_ratio


# Return the chars/token ratio shared by usage telemetry and compression.
def _effective_chars_per_token_hint(
    *,
    model_info_payload: dict[str, Any] | None,
    active_engine: str,
    active_model: str,
    observed_chars_per_token: float | None = None,
) -> float:
    base_ratio = _model_chars_per_token_hint(
        model_info_payload=model_info_payload,
        active_engine=active_engine,
        active_model=active_model,
    )
    if isinstance(observed_chars_per_token, (int, float)) and observed_chars_per_token > 0:
        base_ratio = (base_ratio * 0.35) + (float(observed_chars_per_token) * 0.65)
    return float(max(1.4, min(4.0, base_ratio)))


# Convert a token context window into the same char budget the UI estimates.
def _history_char_budget_from_context_window(
    context_window_tokens: int,
    *,
    model_info_payload: dict[str, Any] | None,
    active_engine: str,
    active_model: str,
    observed_chars_per_token: float | None = None,
    minimum_chars: int = 16000,
    fallback_chars: int = LLM_HISTORY_DEFAULT_CHAR_BUDGET,
) -> int:
    try:
        tokens = max(0, int(context_window_tokens))
    except (TypeError, ValueError):
        tokens = 0
    if tokens <= 0:
        return fallback_chars

    chars_per_token = _effective_chars_per_token_hint(
        model_info_payload=model_info_payload,
        active_engine=active_engine,
        active_model=active_model,
        observed_chars_per_token=observed_chars_per_token,
    )
    return max(int(minimum_chars), int(tokens * chars_per_token))


# Estimate tokens using model hints + optional observed prompt telemetry.
def _estimate_tokens_adaptive(
    *,
    char_count: int,
    model_info_payload: dict[str, Any] | None,
    active_engine: str,
    active_model: str,
    observed_chars_per_token: float | None = None,
) -> int:
    try:
        chars = max(0, int(char_count))
    except (TypeError, ValueError):
        chars = 0
    if chars <= 0:
        return 0

    base_ratio = _effective_chars_per_token_hint(
        model_info_payload=model_info_payload,
        active_engine=active_engine,
        active_model=active_model,
        observed_chars_per_token=observed_chars_per_token,
    )

    # Keep estimator sane and stable across outlier payloads.
    return max(1, int(round(chars / base_ratio)))


# Context window tokens from model info.
def _context_window_tokens_from_model_info(model_info_payload: dict[str, Any] | None) -> int:
    if not isinstance(model_info_payload, dict):
        return 0

    defaults = model_info_payload.get("defaults", {})
    if isinstance(defaults, dict):
        from_defaults = _first_positive_int(defaults, CONTEXT_WINDOW_KEYS)
        if from_defaults:
            return from_defaults

    limits = model_info_payload.get("limits", {})
    if isinstance(limits, dict):
        current_limit = _first_positive_int(limits, ("context_window", "contextWindow", "num_ctx"))
        if current_limit:
            return current_limit

    direct_current = _first_positive_int(
        model_info_payload,
        ("context_window", "contextWindow", "num_ctx", "input_token_limit", "inputTokenLimit"),
    )
    if direct_current:
        return direct_current

    direct_max = _coerce_positive_int(model_info_payload.get("context_length"))
    if direct_max:
        return direct_max
    return 0


# Estimate current context usage for UI telemetry.
def _estimate_context_usage(
    *,
    chat: Chat | None,
    system_prompt: str,
    draft_text: str,
    model_info_payload: dict[str, Any] | None,
    active_engine: str = "",
    active_model: str = "",
) -> dict[str, Any]:
    context_window_tokens = resolve_context_window_tokens(
        model_info_payload,
        runtime_metadata_path=MODEL_RUNTIME_METADATA_PATH,
        active_engine=active_engine,
        active_model=active_model,
    )
    if context_window_tokens <= 0:
        context_window_tokens = _context_window_tokens_from_model_info(model_info_payload)
    if context_window_tokens <= 0:
        context_window_tokens = 32768

    used_chars = len(str(system_prompt or ""))
    compressed_context_active = False
    observed_chars_per_token: float | None = None
    if chat is not None:
        history_qs = (
            chat.messages
            .prefetch_related("attachments", "images")
            .order_by("-created_at", "-id")[:LLM_HISTORY_MAX_MESSAGES]
        )
        for historical_message in history_qs:
            entries = _build_llm_history_entries(historical_message)
            used_chars += sum(_estimate_llm_entry_chars(entry) for entry in entries)
            if _message_has_context_compression_summary(historical_message):
                compressed_context_active = True
                break
        with _chat_usage_lock:
            observed = dict(_chat_usage_by_chat_id.get(str(chat.id), {}))
        raw_ratio = observed.get("observed_chars_per_token")
        if isinstance(raw_ratio, (int, float)) and raw_ratio > 0:
            observed_chars_per_token = float(raw_ratio)

    if draft_text:
        used_chars += len(str(draft_text))

    base_chars_per_token = _model_chars_per_token_hint(
        model_info_payload=model_info_payload,
        active_engine=active_engine,
        active_model=active_model,
    )
    effective_chars_per_token = _effective_chars_per_token_hint(
        model_info_payload=model_info_payload,
        active_engine=active_engine,
        active_model=active_model,
        observed_chars_per_token=observed_chars_per_token,
    )
    estimated_used_tokens = _estimate_tokens_adaptive(
        char_count=used_chars,
        model_info_payload=model_info_payload,
        active_engine=active_engine,
        active_model=active_model,
        observed_chars_per_token=observed_chars_per_token,
    )
    ratio = min(1.0, (estimated_used_tokens / max(1, context_window_tokens)))

    return {
        "context_window_tokens": context_window_tokens,
        "estimated_used_tokens": estimated_used_tokens,
        "estimated_used_chars": used_chars,
        "ratio": ratio,
        "compressed_context_active": compressed_context_active,
        "base_chars_per_token": float(base_chars_per_token),
        "effective_chars_per_token": float(effective_chars_per_token),
        "observed_chars_per_token": float(observed_chars_per_token) if isinstance(observed_chars_per_token, (int, float)) and observed_chars_per_token > 0 else None,
    }


# Build one compression event payload for manual/auto UI-triggered compression.
def _build_manual_compression_event(
    *,
    chat: Chat,
    system_prompt: str,
    engine: str,
    model_name: str,
    model_info_payload: dict[str, Any] | None,
    force: bool,
    draft_text: str = "",
    exclude_message_ids: set[int | str] | None = None,
    summarize_with_model_enabled: bool = True,
) -> dict[str, Any] | None:
    observed_chars_per_token: float | None = None
    with _chat_usage_lock:
        observed = dict(_chat_usage_by_chat_id.get(str(chat.id), {}))
    raw_observed_chars_per_token = observed.get("observed_chars_per_token")
    if isinstance(raw_observed_chars_per_token, (int, float)) and raw_observed_chars_per_token > 0:
        observed_chars_per_token = float(raw_observed_chars_per_token)

    history_budget = _resolve_history_char_budget(
        model_info_payload,
        active_engine=engine,
        active_model=model_name,
        observed_chars_per_token=observed_chars_per_token,
    )
    runtime_context_tokens = resolve_context_window_tokens(
        model_info_payload,
        runtime_metadata_path=MODEL_RUNTIME_METADATA_PATH,
        active_engine=engine,
        active_model=model_name,
    )
    if runtime_context_tokens > 0:
        runtime_budget = _history_char_budget_from_context_window(
            runtime_context_tokens,
            model_info_payload=model_info_payload,
            active_engine=engine,
            active_model=model_name,
            observed_chars_per_token=observed_chars_per_token,
            minimum_chars=12000,
            fallback_chars=history_budget,
        )
        history_budget = min(history_budget, runtime_budget)

    used_history_chars = len(str(system_prompt or "")) + len(str(draft_text or ""))
    selected_history: list[list[dict[str, Any]]] = []
    overflow_entries: list[dict[str, Any]] = []
    history_qs = chat.messages.prefetch_related("attachments", "images")
    excluded_ids = {
        str(value)
        for value in (exclude_message_ids or set())
        if value is not None and str(value).strip()
    }
    if excluded_ids:
        history_qs = history_qs.exclude(id__in=excluded_ids)
    history_records = list(history_qs.order_by("-created_at", "-id")[:LLM_HISTORY_MAX_MESSAGES])
    for index, historical_message in enumerate(history_records):
        entries = _build_llm_history_entries(historical_message)
        if not entries:
            continue
        entry_cost = sum(_estimate_llm_entry_chars(entry) for entry in entries)
        if (
            selected_history
            and len(selected_history) >= LLM_HISTORY_MIN_MESSAGES
            and used_history_chars + entry_cost > history_budget
        ):
            overflow_entries.extend(entries)
            for older_message in history_records[index + 1:]:
                if _message_has_context_compression_summary(older_message):
                    overflow_entries.extend(_build_llm_history_entries(older_message))
                    break
                overflow_entries.extend(_build_llm_history_entries(older_message))
            break
        selected_history.append(entries)
        used_history_chars += entry_cost
        if _message_has_context_compression_summary(historical_message):
            break

    if force and not overflow_entries and len(history_records) > 1:
        keep_recent_count = min(LLM_HISTORY_MIN_MESSAGES, max(1, len(history_records) - 1))
        for older_message in history_records[keep_recent_count:]:
            entries = _build_llm_history_entries(older_message)
            if entries:
                overflow_entries.extend(entries)
            if _message_has_context_compression_summary(older_message):
                break

    compression = decide_compression(
        used_history_chars=used_history_chars,
        history_budget_chars=history_budget,
        model_info_payload=model_info_payload,
        runtime_metadata_path=MODEL_RUNTIME_METADATA_PATH,
        active_engine=engine,
        active_model=model_name,
        debug_force_4k=False,
        trigger_ratio=LLM_HISTORY_COMPRESSION_TRIGGER_RATIO,
    )
    if (force or compression.enabled) and not overflow_entries and len(history_records) > 1:
        keep_recent_count = min(LLM_HISTORY_MIN_MESSAGES, max(1, len(history_records) - 1))
        for older_message in history_records[keep_recent_count:]:
            entries = _build_llm_history_entries(older_message)
            if entries:
                overflow_entries.extend(entries)
            if _message_has_context_compression_summary(older_message):
                break

    summary_entries = _build_context_compression_source_entries(history_records)
    # This event always compresses summary_entries (the span up to the previous
    # summary boundary); without them there is nothing to compress and emitting
    # an event would replace history with an empty summary.
    if not summary_entries:
        return None
    if not overflow_entries and (compression.enabled or force):
        overflow_entries = list(summary_entries)
    should_compress = bool(overflow_entries) and (compression.enabled or force)
    if not should_compress:
        return None

    recent_user_messages = [
        _strip_llm_control_tokens(str(message.content or "")).strip()[:1200]
        for message in chat.messages.filter(role="user").order_by("-created_at", "-id")[:LLM_HISTORY_COMPRESSION_RECENT_USER_MESSAGES]
        if str(message.content or "").strip()
    ]
    direct_user_directives = _collect_direct_user_directives(chat, exclude_message_id=0)
    summarize_with_model_callback = None
    if summarize_with_model_enabled:
        summarize_with_model_callback = lambda prompt_messages: _generate_history_summary_with_model(
            engine=engine,
            model_name=model_name,
            prompt_messages=prompt_messages,
            model_info_payload=model_info_payload,
        )

    try:
        summary_text, summary_payload = build_structured_history_summary(
            overflow_entries=summary_entries,
            recent_user_messages=recent_user_messages,
            direct_user_directives=direct_user_directives,
            summarize_with_model=summarize_with_model_callback,
            max_overflow_entries=LLM_HISTORY_COMPRESSION_MAX_ITEMS,
        )
    except Exception as exc:
        logger.warning("History compression summary failed: %s", exc)
        summary_text, summary_payload = build_structured_history_summary(
            overflow_entries=summary_entries,
            recent_user_messages=recent_user_messages,
            direct_user_directives=direct_user_directives,
            summarize_with_model=None,
            max_overflow_entries=LLM_HISTORY_COMPRESSION_MAX_ITEMS,
        )

    summary_text, summary_payload = fit_summary_text(summary_payload, LLM_HISTORY_COMPRESSION_MAX_CHARS)
    return {
        "role": "tool",
        "alias": "context_compression_summary",
        "name": "context_compression_summary",
        "tool_name": "context_compression_summary",
        "tool_id": "context_compression_summary",
        "tool_display_name": "Context Compression",
        "server_id": "system",
        "server_name": "System",
        "arguments": {
            "reason": compression.reason,
            "history_budget_chars": history_budget,
            "used_history_chars": used_history_chars,
            "context_window_tokens": compression.context_window_tokens,
            "force": bool(force),
        },
        "content": summary_text,
        "tool_ui": {
            "status": "done",
            "label": "Context compressed",
            "kind": "context_compression",
        },
        "structured_content": {
            "kind": "context_compression",
            "summary": summary_payload,
        },
    }


# Build one compression timeline event from stateless overflow data (for external modules).
def _build_stateless_compression_event(
    *,
    engine: str,
    model_name: str,
    model_info_payload: dict[str, Any] | None,
    force: bool,
    used_history_chars: int,
    history_budget_chars: int,
    overflow_entries: list[dict[str, Any]],
    summary_source_entries: list[dict[str, Any]],
    recent_user_messages: list[str],
    direct_user_directives: list[str],
    summarize_with_model_enabled: bool = True,
    debug_force_4k: bool = False,
    trigger_ratio: float = LLM_HISTORY_COMPRESSION_TRIGGER_RATIO,
    compression_mode: str = "manual",
) -> dict[str, Any] | None:
    compression = decide_compression(
        used_history_chars=used_history_chars,
        history_budget_chars=history_budget_chars,
        model_info_payload=model_info_payload,
        runtime_metadata_path=MODEL_RUNTIME_METADATA_PATH,
        active_engine=engine,
        active_model=model_name,
        debug_force_4k=debug_force_4k,
        trigger_ratio=trigger_ratio,
    )

    entries_to_summarize = list(overflow_entries)
    summary_entries = list(summary_source_entries)
    normalized_mode = str(compression_mode or "manual").strip().lower()

    if normalized_mode == "auto":
        if entries_to_summarize and compression.enabled:
            if not summary_entries:
                entries_to_summarize = []
            else:
                entries_to_summarize = list(summary_entries)
        if compression.enabled and not entries_to_summarize and summary_entries:
            entries_to_summarize = list(summary_entries)
        should_compress = bool(entries_to_summarize) and compression.enabled
    else:
        if compression.enabled and not entries_to_summarize and summary_entries:
            entries_to_summarize = list(summary_entries)
        if force and not entries_to_summarize:
            entries_to_summarize = list(summary_entries)
        if force and not summary_entries:
            return None
        should_compress = bool(entries_to_summarize) and (compression.enabled or force)

    if not should_compress:
        return None

    summarize_entries = list(summary_entries) if normalized_mode != "auto" else list(entries_to_summarize)
    if not summarize_entries:
        return None

    summarize_with_model_callback = None
    if summarize_with_model_enabled:
        summarize_with_model_callback = lambda prompt_messages: _generate_history_summary_with_model(
            engine=engine,
            model_name=model_name,
            prompt_messages=prompt_messages,
            model_info_payload=model_info_payload,
        )

    try:
        summary_text, summary_payload = build_structured_history_summary(
            overflow_entries=summarize_entries,
            recent_user_messages=recent_user_messages,
            direct_user_directives=direct_user_directives,
            summarize_with_model=summarize_with_model_callback,
            max_overflow_entries=LLM_HISTORY_COMPRESSION_MAX_ITEMS,
        )
    except Exception as exc:
        logger.warning("History compression summary failed: %s", exc)
        summary_text, summary_payload = build_structured_history_summary(
            overflow_entries=summarize_entries,
            recent_user_messages=recent_user_messages,
            direct_user_directives=direct_user_directives,
            summarize_with_model=None,
            max_overflow_entries=LLM_HISTORY_COMPRESSION_MAX_ITEMS,
        )

    if not summary_text and normalized_mode == "auto":
        return None

    summary_text, summary_payload = fit_summary_text(summary_payload, LLM_HISTORY_COMPRESSION_MAX_CHARS)
    return {
        "role": "tool",
        "alias": "context_compression_summary",
        "name": "context_compression_summary",
        "tool_name": "context_compression_summary",
        "tool_id": "context_compression_summary",
        "tool_display_name": "Context Compression",
        "server_id": "system",
        "server_name": "System",
        "arguments": {
            "reason": compression.reason,
            "history_budget_chars": history_budget_chars,
            "used_history_chars": used_history_chars,
            "context_window_tokens": compression.context_window_tokens,
            "force": bool(force),
        },
        "content": summary_text,
        "tool_ui": {
            "status": "done",
            "label": "Context compressed",
            "kind": "context_compression",
        },
        "structured_content": {
            "kind": "context_compression",
            "summary": summary_payload,
        },
    }


# Collect recent user messages.
def _collect_recent_user_messages(chat: Chat, exclude_message_id: int) -> list[str]:
    messages = (
        chat.messages
        .filter(role="user")
        .exclude(id=exclude_message_id)
        .order_by("-created_at", "-id")[:LLM_HISTORY_COMPRESSION_RECENT_USER_MESSAGES]
    )
    result: list[str] = []
    for message in messages:
        text = _strip_llm_control_tokens(str(message.content or "")).strip()
        if text:
            result.append(text[:1200])
    return result


# Collect direct user directives.
def _collect_direct_user_directives(chat: Chat, exclude_message_id: int) -> list[str]:
    return []


# Generate one non-streamed summary payload with the active model.
def _generate_history_summary_with_model(
    *,
    engine: str,
    model_name: str,
    prompt_messages: list[dict[str, str]],
    model_info_payload: dict[str, Any] | None = None,
) -> str:
    options = {
        "temperature": 0.0,
        "max_tokens": 8192,
    }
    chunks = llm_api.generate(
        engine=engine,
        model_name=model_name,
        messages=prompt_messages,
        stream=False,
        options=options,
        think=False,
        think_param_name=str((model_info_payload or {}).get("think_param_name", "think") or "think"),
        think_level_param_name=str((model_info_payload or {}).get("think_level_param_name", "reasoning_effort") or "reasoning_effort"),
    )
    parts: list[str] = []
    for chunk in chunks:
        thinking_part, text_part = _extract_stream_message_parts(chunk)
        if text_part:
            parts.append(text_part)
        elif isinstance(chunk, str):
            parts.append(chunk)
    summary = _strip_llm_control_tokens("".join(parts)).strip()
    if not summary:
        logger.warning(
            "History summary model returned empty output (engine=%s model=%s); "
            "compression will fall back to raw context harvest",
            engine,
            model_name,
        )
    return summary


# Build LLM message history
def _build_chat_history(
    chat: Chat,
    user_message_record: Message,
    user_message: str,
    system_prompt: str,
    engine: str,
    model_name: str,
    model_info_payload: dict[str, Any] | None = None,
    upload_manifests: list[dict[str, Any]] | None = None,
    sandbox_enabled: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    llm_messages: list[dict[str, Any]] = []
    supported_media_kinds = _supported_model_media_kinds(engine, model_info_payload)
    if system_prompt:
        llm_messages.append({"role": "system", "content": system_prompt})

    observed_chars_per_token: float | None = None
    with _chat_usage_lock:
        observed = dict(_chat_usage_by_chat_id.get(str(chat.id), {}))
    raw_observed_chars_per_token = observed.get("observed_chars_per_token")
    if isinstance(raw_observed_chars_per_token, (int, float)) and raw_observed_chars_per_token > 0:
        observed_chars_per_token = float(raw_observed_chars_per_token)

    history_budget = _resolve_history_char_budget(
        model_info_payload,
        active_engine=engine,
        active_model=model_name,
        observed_chars_per_token=observed_chars_per_token,
    )
    debug_force_4k = str(os.getenv("ASLM_DEBUG_CONTEXT_COMPRESSION_4K", "")).strip().lower() in {"1", "true", "yes", "on"}
    runtime_context_tokens = resolve_context_window_tokens(
        model_info_payload,
        runtime_metadata_path=MODEL_RUNTIME_METADATA_PATH,
        active_engine=engine,
        active_model=model_name,
    )
    if debug_force_4k:
        runtime_context_tokens = 4096
    if runtime_context_tokens > 0:
        runtime_budget = _history_char_budget_from_context_window(
            runtime_context_tokens,
            model_info_payload=model_info_payload,
            active_engine=engine,
            active_model=model_name,
            observed_chars_per_token=observed_chars_per_token,
            minimum_chars=12000,
            fallback_chars=history_budget,
        )
        history_budget = min(history_budget, runtime_budget)
    current_entry: dict[str, Any] = {"role": "user", "content": user_message}
    current_attachments = _get_message_attachments(user_message_record)
    current_entry = _apply_attachments_to_llm_entry(current_entry, current_attachments)
    current_upload_manifests = (
        upload_manifests
        if upload_manifests is not None
        else _load_message_upload_manifests(user_message_record, sandbox_enabled=sandbox_enabled)
    )
    current_entry = _apply_uploaded_file_manifests_to_llm_entry(
        current_entry,
        current_upload_manifests or [],
        supported_media_kinds=supported_media_kinds,
    )

    used_history_chars = len(str(system_prompt or "")) + _estimate_llm_entry_chars(current_entry)
    selected_history: list[list[dict[str, Any]]] = []
    overflow_entries: list[dict[str, Any]] = []
    history_records = list((
        chat.messages
        .exclude(id=user_message_record.id)
        .prefetch_related("attachments", "images")
        .order_by("-created_at", "-id")[:LLM_HISTORY_MAX_MESSAGES]
    ))
    for index, historical_message in enumerate(history_records):
        entries = _build_llm_history_entries(
            historical_message,
            sandbox_enabled=sandbox_enabled,
            supported_media_kinds=supported_media_kinds,
        )
        if not entries:
            continue

        entry_cost = sum(_estimate_llm_entry_chars(entry) for entry in entries)
        if (
            selected_history
            and len(selected_history) >= LLM_HISTORY_MIN_MESSAGES
            and used_history_chars + entry_cost > history_budget
        ):
            # Collect the current and older entries for compression.
            overflow_entries.extend(entries)
            for older_message in history_records[index + 1:]:
                if _message_has_context_compression_summary(older_message):
                    overflow_entries.extend(_build_llm_history_entries(older_message, sandbox_enabled=sandbox_enabled))
                    break
                overflow_entries.extend(_build_llm_history_entries(older_message, sandbox_enabled=sandbox_enabled))
            break

        selected_history.append(entries)
        used_history_chars += entry_cost
        if _message_has_context_compression_summary(historical_message):
            break

    compression = decide_compression(
        used_history_chars=used_history_chars,
        history_budget_chars=history_budget,
        model_info_payload=model_info_payload,
        runtime_metadata_path=MODEL_RUNTIME_METADATA_PATH,
        active_engine=engine,
        active_model=model_name,
        debug_force_4k=debug_force_4k,
        trigger_ratio=LLM_HISTORY_COMPRESSION_TRIGGER_RATIO,
    )
    compression_event: dict[str, Any] | None = None
    summary_entries = _build_context_compression_source_entries(
        history_records,
        sandbox_enabled=sandbox_enabled,
    )
    if overflow_entries and compression.enabled:
        if not summary_entries:
            overflow_entries = []
        else:
            overflow_entries = summary_entries
    if compression.enabled and not overflow_entries and summary_entries:
        overflow_entries = list(summary_entries)
        selected_history = []
    if overflow_entries and compression.enabled:
        recent_user_messages = _collect_recent_user_messages(chat, user_message_record.id)
        direct_user_directives = _collect_direct_user_directives(chat, user_message_record.id)
        try:
            summary_text, _summary_payload = build_structured_history_summary(
                overflow_entries=overflow_entries,
                recent_user_messages=recent_user_messages,
                direct_user_directives=direct_user_directives,
                summarize_with_model=lambda prompt_messages: _generate_history_summary_with_model(
                    engine=engine,
                    model_name=model_name,
                    prompt_messages=prompt_messages,
                    model_info_payload=model_info_payload,
                ),
                max_overflow_entries=LLM_HISTORY_COMPRESSION_MAX_ITEMS,
            )
        except Exception as exc:
            logger.warning("History compression summary failed: %s", exc)
            summary_text, _summary_payload = build_structured_history_summary(
                overflow_entries=overflow_entries,
                recent_user_messages=recent_user_messages,
                direct_user_directives=direct_user_directives,
                summarize_with_model=None,
                max_overflow_entries=LLM_HISTORY_COMPRESSION_MAX_ITEMS,
            )

        if summary_text:
            summary_text, _summary_payload = fit_summary_text(_summary_payload, LLM_HISTORY_COMPRESSION_MAX_CHARS)
            llm_messages.append({"role": "system", "content": summary_text})
            compression_event = {
                "role": "tool",
                "alias": "context_compression_summary",
                "name": "context_compression_summary",
                "tool_name": "context_compression_summary",
                "tool_id": "context_compression_summary",
                "tool_display_name": "Context Compression",
                "server_id": "system",
                "server_name": "System",
                "arguments": {
                    "reason": compression.reason,
                    "history_budget_chars": history_budget,
                    "used_history_chars": used_history_chars,
                    "context_window_tokens": compression.context_window_tokens,
                },
                "content": summary_text,
                "tool_ui": {
                    "status": "done",
                    "label": "Context compressed",
                    "kind": "context_compression",
                },
                "structured_content": {
                    "kind": "context_compression",
                    "summary": _summary_payload,
                },
            }
            logger.info(
                "History compression applied: engine=%s model=%s %s debug_force_4k=%s",
                engine,
                model_name,
                compression.reason,
                debug_force_4k,
            )

    for entries in reversed(selected_history):
        llm_messages.extend(entries)

    llm_messages.append(current_entry)

    return llm_messages, compression_event

# Split generation options
def _split_generation_options(
    options: dict[str, Any],
    think_param_name: str = "think",
    think_level_param_name: str = "think_level",
) -> tuple[Any, Any, dict[str, Any]]:
    think_value = None
    think_level_value = None
    clean_options: dict[str, Any] = {}
    think_param_names = set(THINK_PARAM_NAMES)
    think_level_param_names = set(THINK_LEVEL_PARAM_NAMES)
    if think_param_name:
        think_param_names.add(str(think_param_name))
    if think_level_param_name:
        think_level_param_names.add(str(think_level_param_name))

    for key, value in options.items():
        if key in think_param_names:
            think_value = value
        elif key in think_level_param_names:
            think_level_value = value
        else:
            clean_options[key] = value

    return think_value, think_level_value, clean_options


@dataclass(frozen=True)
class PreparedGenerationRequest:
    engine: str
    model_name: str
    options: dict[str, Any]
    selected_tool_servers: list[dict[str, Any]]
    model_info_payload: dict[str, Any]
    sandbox_enabled: bool
    attachments: list[dict[str, Any]]
    uploaded_file_ids: list[str]
    upload_manifests: list[dict[str, Any]]
    think_value: Any
    think_level_value: Any
    instant_mode: bool
    clean_options: dict[str, Any]
    sync_operation_defaults: dict[str, Any] | None
    tool_sources: list[dict[str, Any]]
    external_tool_context: dict[str, Any]


# Build LMS sync defaults for one generation request.
def _build_lms_sync_operation_defaults(
    engine: str,
    model_info_payload: dict[str, Any],
    think_value: Any,
    think_level_value: Any,
) -> dict[str, Any] | None:
    if engine != "lms":
        return None

    payload: dict[str, Any] = {}
    defaults = model_info_payload.get("defaults")
    if isinstance(defaults, dict):
        payload.update(defaults)

    think_param = str(model_info_payload.get("think_param_name", "think") or "think")
    think_level_param = str(model_info_payload.get("think_level_param_name", "think_level") or "think_level")
    if think_value is not None and think_param.startswith("ext."):
        payload[think_param] = think_value
    if think_level_value is not None and think_level_param.startswith("ext."):
        payload[think_level_param] = think_level_value
    return payload or None


# Validate and normalize one shared generation request payload.
def _prepare_generation_request(
    request,
    data: dict[str, Any],
    *,
    route: str,
    require_user_input: bool = True,
    user_message: str = "",
    attachments: list[dict[str, Any]] | None = None,
    uploaded_file_ids: list[str] | None = None,
) -> PreparedGenerationRequest | JsonResponse:
    model_name = str(data.get("model", "") or "").strip()
    options = data.get("options", {}) or {}
    resolved_attachments = attachments if attachments is not None else _normalize_request_attachments(data)
    resolved_upload_ids = uploaded_file_ids if uploaded_file_ids is not None else _normalize_uploaded_file_ids(data)

    engine, engine_error = _resolve_request_engine_or_response(request, data)
    if engine_error is not None:
        return engine_error

    raw_tool_ids = data.get("tool_server_ids") or data.get("tool_server_id") or data.get("tool_id") or []
    if isinstance(raw_tool_ids, str):
        raw_tool_ids = [raw_tool_ids] if raw_tool_ids.strip() else []
    tool_server_ids = [str(s).strip() for s in raw_tool_ids if str(s).strip()]
    tool_sources = _parse_tool_sources(data)
    external_tool_context = data.get("tool_context") if isinstance(data.get("tool_context"), dict) else {}

    if not model_name:
        return JsonResponse({"error": "Missing model parameter"}, status=400)
    if require_user_input and not str(user_message or "").strip() and not resolved_attachments and not resolved_upload_ids:
        return JsonResponse({"error": "Missing message or attachments"}, status=400)

    _remember_active_model(engine, model_name)
    selected_tool_servers = _resolve_tool_servers(
        engine,
        model_name,
        tool_server_ids,
        tool_sources=tool_sources,
    )
    model_info_payload = _build_model_info_payload(engine, model_name, allow_fallback=True)
    _sync_runtime_model_metadata(
        engine,
        model_name,
        model_info_payload,
        source="generation",
        route=route,
    )

    has_image_attachments = any(
        attachment.get("kind") == MessageAttachmentKind.IMAGE
        for attachment in resolved_attachments
        if isinstance(attachment, dict)
    )
    vision_supported = bool(model_info_payload.get("supports_vision", False))
    vision_source = str(model_info_payload.get("supports_vision_source") or "").strip().lower()
    if has_image_attachments and not vision_supported and vision_source == "explicit_false":
        return JsonResponse(
            {
                "error": (
                    f"Model '{model_name}' does not support image input on '{engine}'. "
                    "Choose a vision-capable model."
                )
            },
            status=400,
        )
    _validate_tool_server_support(
        engine,
        model_name,
        [server["id"] for server in selected_tool_servers],
        payload=model_info_payload,
    )

    sandbox_enabled = _selected_tools_include_sandbox(selected_tool_servers)
    upload_manifests = _load_model_upload_manifests(resolved_upload_ids, sandbox_enabled=sandbox_enabled)
    think_value, think_level_value, clean_options = _split_generation_options(
        options,
        think_param_name=str(model_info_payload.get("think_param_name", "think") or "think"),
        think_level_param_name=str(model_info_payload.get("think_level_param_name", "think_level") or "think_level"),
    )

    return PreparedGenerationRequest(
        engine=engine,
        model_name=model_name,
        options=options,
        selected_tool_servers=selected_tool_servers,
        model_info_payload=model_info_payload,
        sandbox_enabled=sandbox_enabled,
        attachments=resolved_attachments,
        uploaded_file_ids=resolved_upload_ids,
        upload_manifests=upload_manifests,
        think_value=think_value,
        think_level_value=think_level_value,
        instant_mode=is_instant_generation(think_value, think_level_value),
        clean_options=clean_options,
        sync_operation_defaults=_build_lms_sync_operation_defaults(
            engine,
            model_info_payload,
            think_value,
            think_level_value,
        ),
        tool_sources=tool_sources,
        external_tool_context=external_tool_context,
    )


# Resolve whether the skills inventory should be injected into the system prompt.
def _resolve_include_skills_baseline(data: dict[str, Any], history_messages: list[dict[str, Any]]) -> bool:
    if "include_skills_baseline" in data:
        return bool(data.get("include_skills_baseline"))
    return len(history_messages) == 0


# Normalize request-side conversation history.
def _normalize_request_history_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list):
        return []

    normalized: list[dict[str, Any]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            continue
        role = str(raw_message.get("role") or "").strip().lower()
        if role not in {"user", "assistant", "system", "tool"}:
            continue
        normalized.append(raw_message)
    return normalized


# Return whether one request history message stores a compression boundary.
def _request_message_has_context_compression_summary(message: dict[str, Any]) -> bool:
    if str(message.get("role") or "").strip().lower() != "assistant":
        return False
    for entry in _normalize_transcript_entries(message.get("llm_transcript")):
        if entry.get("role") == "tool" and str(entry.get("alias") or "") == "context_compression_summary":
            return bool(str(entry.get("content") or "").strip())
    return False


# Collect recent user messages from request-side history.
def _collect_recent_user_messages_from_history(
    history_messages: list[dict[str, Any]],
    current_user_text: str,
) -> list[str]:
    result: list[str] = []
    current_text = _strip_llm_control_tokens(str(current_user_text or "")).strip()
    if current_text:
        result.append(current_text[:1200])

    for message in reversed(history_messages):
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        text = _strip_llm_control_tokens(str(message.get("content") or "")).strip()
        if text:
            result.append(text[:1200])
        if len(result) >= LLM_HISTORY_COMPRESSION_RECENT_USER_MESSAGES:
            break
    return result


# Build chronological compression source entries from request history.
def _build_context_compression_source_entries_from_request(
    history_records_newest_first: list[dict[str, Any]],
    *,
    sandbox_enabled: bool = False,
) -> list[dict[str, Any]]:
    boundary_records: list[dict[str, Any]] = []
    for historical_message in history_records_newest_first:
        if _request_message_has_context_compression_summary(historical_message):
            break
        boundary_records.append(historical_message)

    entries: list[dict[str, Any]] = []
    for historical_message in reversed(boundary_records):
        entries.extend(
            _build_llm_entries_from_request_message(historical_message, sandbox_enabled=sandbox_enabled)
        )
    return entries


# Split request history and the current user turn.
def _split_request_conversation(
    data: dict[str, Any],
    history_messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, list[dict[str, Any]], list[str]]:
    user_message = str(data.get("message", "") or "")
    attachments = _normalize_request_attachments(data)
    uploaded_file_ids = _normalize_uploaded_file_ids(data)

    if user_message or attachments or uploaded_file_ids or not history_messages:
        return history_messages, user_message, attachments, uploaded_file_ids

    last_message = history_messages[-1]
    if str(last_message.get("role") or "").strip().lower() != "user":
        return history_messages, user_message, attachments, uploaded_file_ids

    promoted_attachments = _normalize_attachments_from_mapping(last_message)
    promoted_upload_ids = _normalize_uploaded_file_ids(last_message)
    promoted_text = str(last_message.get("content") or "")
    if not promoted_text and not promoted_attachments and not promoted_upload_ids:
        return history_messages, user_message, attachments, uploaded_file_ids

    merged_upload_ids = list(dict.fromkeys([*promoted_upload_ids, *uploaded_file_ids]))
    return history_messages[:-1], promoted_text, promoted_attachments, merged_upload_ids


# Build the current user LLM entry for stateless generation.
def _build_current_user_llm_entry(
    user_message: str,
    attachments: list[dict[str, Any]],
    upload_manifests: list[dict[str, Any]],
    *,
    supported_media_kinds: set[str] | None = None,
) -> dict[str, Any]:
    current_entry: dict[str, Any] = {"role": "user", "content": user_message}
    current_entry = _apply_attachments_to_llm_entry(current_entry, attachments)
    current_entry = _apply_uploaded_file_manifests_to_llm_entry(
        current_entry,
        upload_manifests or [],
        supported_media_kinds=supported_media_kinds,
    )
    return current_entry


# Build LLM messages for stateless generation from request payload.
def _build_generate_llm_messages(
    history_messages: list[dict[str, Any]],
    current_entry: dict[str, Any],
    system_prompt: str,
    engine: str,
    model_name: str,
    model_info_payload: dict[str, Any] | None,
    *,
    session_id: str,
    sandbox_enabled: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    llm_messages: list[dict[str, Any]] = []
    supported_media_kinds = _supported_model_media_kinds(engine, model_info_payload)
    if system_prompt:
        llm_messages.append({"role": "system", "content": system_prompt})

    observed_chars_per_token: float | None = None
    with _chat_usage_lock:
        observed = dict(_chat_usage_by_chat_id.get(str(session_id), {}))
    raw_observed_chars_per_token = observed.get("observed_chars_per_token")
    if isinstance(raw_observed_chars_per_token, (int, float)) and raw_observed_chars_per_token > 0:
        observed_chars_per_token = float(raw_observed_chars_per_token)

    history_budget = _resolve_history_char_budget(
        model_info_payload,
        active_engine=engine,
        active_model=model_name,
        observed_chars_per_token=observed_chars_per_token,
    )
    debug_force_4k = str(os.getenv("ASLM_DEBUG_CONTEXT_COMPRESSION_4K", "")).strip().lower() in {"1", "true", "yes", "on"}
    runtime_context_tokens = resolve_context_window_tokens(
        model_info_payload,
        runtime_metadata_path=MODEL_RUNTIME_METADATA_PATH,
        active_engine=engine,
        active_model=model_name,
    )
    if debug_force_4k:
        runtime_context_tokens = 4096
    if runtime_context_tokens > 0:
        runtime_budget = _history_char_budget_from_context_window(
            runtime_context_tokens,
            model_info_payload=model_info_payload,
            active_engine=engine,
            active_model=model_name,
            observed_chars_per_token=observed_chars_per_token,
            minimum_chars=12000,
            fallback_chars=history_budget,
        )
        history_budget = min(history_budget, runtime_budget)

    used_history_chars = len(str(system_prompt or "")) + _estimate_llm_entry_chars(current_entry)
    selected_history: list[list[dict[str, Any]]] = []
    overflow_entries: list[dict[str, Any]] = []
    history_records = list(history_messages[-LLM_HISTORY_MAX_MESSAGES:])
    history_newest_first = list(reversed(history_records))

    for index, historical_message in enumerate(history_newest_first):
        entries = _build_llm_entries_from_request_message(
            historical_message,
            sandbox_enabled=sandbox_enabled,
            supported_media_kinds=supported_media_kinds,
        )
        if not entries:
            continue

        entry_cost = sum(_estimate_llm_entry_chars(entry) for entry in entries)
        if (
            selected_history
            and len(selected_history) >= LLM_HISTORY_MIN_MESSAGES
            and used_history_chars + entry_cost > history_budget
        ):
            overflow_entries.extend(entries)
            for older_message in history_newest_first[index + 1:]:
                if _request_message_has_context_compression_summary(older_message):
                    overflow_entries.extend(
                        _build_llm_entries_from_request_message(older_message, sandbox_enabled=sandbox_enabled)
                    )
                    break
                overflow_entries.extend(
                    _build_llm_entries_from_request_message(older_message, sandbox_enabled=sandbox_enabled)
                )
            break

        selected_history.append(entries)
        used_history_chars += entry_cost
        if _request_message_has_context_compression_summary(historical_message):
            break

    compression = decide_compression(
        used_history_chars=used_history_chars,
        history_budget_chars=history_budget,
        model_info_payload=model_info_payload,
        runtime_metadata_path=MODEL_RUNTIME_METADATA_PATH,
        active_engine=engine,
        active_model=model_name,
        debug_force_4k=debug_force_4k,
        trigger_ratio=LLM_HISTORY_COMPRESSION_TRIGGER_RATIO,
    )
    compression_event: dict[str, Any] | None = None
    summary_entries = _build_context_compression_source_entries_from_request(
        history_newest_first,
        sandbox_enabled=sandbox_enabled,
    )
    if overflow_entries and compression.enabled:
        if not summary_entries:
            overflow_entries = []
        else:
            overflow_entries = summary_entries
    if compression.enabled and not overflow_entries and summary_entries:
        overflow_entries = list(summary_entries)
        selected_history = []
    if overflow_entries and compression.enabled:
        recent_user_messages = _collect_recent_user_messages_from_history(
            history_messages,
            str(current_entry.get("content") or ""),
        )
        direct_user_directives = _collect_direct_user_directives(None, 0)
        try:
            summary_text, summary_payload = build_structured_history_summary(
                overflow_entries=overflow_entries,
                recent_user_messages=recent_user_messages,
                direct_user_directives=direct_user_directives,
                summarize_with_model=lambda prompt_messages: _generate_history_summary_with_model(
                    engine=engine,
                    model_name=model_name,
                    prompt_messages=prompt_messages,
                    model_info_payload=model_info_payload,
                ),
                max_overflow_entries=LLM_HISTORY_COMPRESSION_MAX_ITEMS,
            )
        except Exception as exc:
            logger.warning("History compression summary failed: %s", exc)
            summary_text, summary_payload = build_structured_history_summary(
                overflow_entries=overflow_entries,
                recent_user_messages=recent_user_messages,
                direct_user_directives=direct_user_directives,
                summarize_with_model=None,
                max_overflow_entries=LLM_HISTORY_COMPRESSION_MAX_ITEMS,
            )

        if summary_text:
            summary_text, summary_payload = fit_summary_text(summary_payload, LLM_HISTORY_COMPRESSION_MAX_CHARS)
            llm_messages.append({"role": "system", "content": summary_text})
            compression_event = {
                "role": "tool",
                "alias": "context_compression_summary",
                "name": "context_compression_summary",
                "tool_name": "context_compression_summary",
                "tool_id": "context_compression_summary",
                "tool_display_name": "Context Compression",
                "server_id": "system",
                "server_name": "System",
                "arguments": {
                    "reason": compression.reason,
                    "history_budget_chars": history_budget,
                    "used_history_chars": used_history_chars,
                    "context_window_tokens": compression.context_window_tokens,
                },
                "content": summary_text,
                "tool_ui": {
                    "status": "done",
                    "label": "Context compressed",
                    "kind": "context_compression",
                },
                "structured_content": {
                    "kind": "context_compression",
                    "summary": summary_payload,
                },
            }
            logger.info(
                "History compression applied: engine=%s model=%s %s debug_force_4k=%s",
                engine,
                model_name,
                compression.reason,
                debug_force_4k,
            )

    for entries in reversed(selected_history):
        llm_messages.extend(entries)

    llm_messages.append(current_entry)
    return llm_messages, compression_event


# Insert a one-off system notice after the main system prompt, without persisting a message.
def _inject_ephemeral_system_notice(llm_messages: list[dict[str, Any]], notice: str) -> None:
    cleaned = str(notice or "").strip()
    if not cleaned:
        return
    entry = {"role": "system", "content": cleaned}
    if llm_messages and llm_messages[0].get("role") == "system":
        llm_messages.insert(1, entry)
        return
    llm_messages.insert(0, entry)


# Build generation kwargs
def _build_generate_kwargs(
    engine: str,
    model_name: str,
    llm_messages: list[dict[str, Any]],
    think_value: Any,
    think_level_value: Any,
    clean_options: dict[str, Any],
    session_id: str,
    selected_tool_servers: list[dict[str, Any]],
    think_param_name: str = "think",
    think_level_param_name: str = "think_level",
    sync_operation_defaults: dict[str, Any] | None = None,
    tool_sources: list[dict[str, Any]] | None = None,
    external_tool_context: dict[str, Any] | None = None,
    instant_mode: bool = False,
) -> dict[str, Any]:
    generate_kwargs: dict[str, Any] = {
        "engine": engine,
        "model_name": model_name,
        "messages": llm_messages,
        "stream": True,
    }

    if think_value is not None:
        generate_kwargs["think"] = think_value
    if think_level_value is not None:
        generate_kwargs["think_level"] = think_level_value
    if engine in {"lms", "openai", "google-genai"} and think_param_name:
        generate_kwargs["think_param_name"] = think_param_name
    if engine in {"lms", "openai", "google-genai"} and think_level_param_name:
        generate_kwargs["think_level_param_name"] = think_level_param_name
    if clean_options:
        generate_kwargs["options"] = clean_options
    if engine == "lms" and isinstance(sync_operation_defaults, dict) and sync_operation_defaults:
        generate_kwargs["sync_operation_defaults"] = sync_operation_defaults

    if selected_tool_servers:
        generate_kwargs["tool_server_ids"] = [s["id"] for s in selected_tool_servers]
        tool_source_map = _build_tool_source_map(tool_sources, selected_tool_servers)
        tool_context = {
            "chat_id": str(session_id),
            "engine": engine,
            "model_name": model_name,
            "module_dir": str(settings.BASE_DIR),
            "project_dir": str(settings.BASE_DIR),
            "selected_tool_server_ids": [s["id"] for s in selected_tool_servers],
            "sandbox_enabled": _selected_tools_include_sandbox(selected_tool_servers),
            "tool_source_map": tool_source_map,
        }
        if instant_mode:
            tool_context["instant_mode"] = True
        if isinstance(external_tool_context, dict):
            for key, value in external_tool_context.items():
                # Tuple membership compares with ==, so unhashable values (lists) are fine.
                if value not in (None, ""):
                    tool_context[key] = value
        generate_kwargs["tool_context"] = tool_context

    return generate_kwargs

# Stream and save assistant response
def _stream_chat_response(
    engine: str,
    generate_kwargs: dict[str, Any],
    generation_id: str,
    *,
    chat: Chat | None = None,
    assistant_message_record: Message | None = None,
    session_id: str = "",
    compression_event: dict[str, Any] | None = None,
    model_info_payload: dict[str, Any] | None = None,
    system_prompt: str = "",
    current_user_message_id: int | str | None = None,
    persist_messages: bool = True,
):
    visible_parts: list[str] = []
    thinking_parts: list[str] = []
    transcript_entries: list[dict[str, Any]] = []
    usage_snapshot: dict[str, Any] = {}
    is_thinking = False
    failed = False
    restart_after_stream_compression = False
    started_at = time.perf_counter()
    model_name = str(generate_kwargs.get("model_name", "") or "")
    llm_messages = generate_kwargs.get("messages", [])
    prompt_chars_estimate = sum(
        _estimate_llm_entry_chars(entry)
        for entry in llm_messages
        if isinstance(entry, dict)
    ) if isinstance(llm_messages, list) else 0
    image_count = _count_request_images(llm_messages if isinstance(llm_messages, list) else [])
    selected_tool_server_ids = generate_kwargs.get("tool_server_ids", []) or []
    raw_options = generate_kwargs.get("options", {})
    raw_think_value = generate_kwargs.get("think")
    emit_thinking = raw_think_value is not False and str(raw_think_value).strip().lower() not in {"false", "0", "off", "no"}
    last_snapshot_at = 0.0
    last_snapshot_chars = 0
    stream_compression_applied = isinstance(compression_event, dict)

    # Return current in-flight assistant text for compression estimates.
    def current_stream_context_text() -> str:
        parts = [
            _strip_llm_control_tokens("".join(thinking_parts)).strip(),
            _strip_llm_control_tokens("".join(visible_parts)).strip(),
        ]
        for entry in transcript_entries:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "").strip().lower()
            if role == "assistant":
                parts.append(_strip_llm_control_tokens(str(entry.get("thinking") or "")).strip())
                parts.append(_strip_llm_control_tokens(str(entry.get("content") or "")).strip())
            elif role == "tool" and str(entry.get("alias") or "") != "context_compression_summary":
                parts.append(_strip_llm_control_tokens(str(entry.get("content") or "")).strip())
        return "\n".join(part for part in parts if part)

    # Persist the in-flight assistant buffer so reloads/compression see it.
    def persist_stream_snapshot(*, force: bool = False) -> None:
        nonlocal last_snapshot_at, last_snapshot_chars

        if not persist_messages or assistant_message_record is None or chat is None:
            return

        visible_content = _strip_llm_control_tokens("".join(visible_parts)).strip()
        thinking_content = _strip_llm_control_tokens("".join(thinking_parts)).strip()
        transcript_snapshot = _build_streaming_assistant_transcript(
            transcript_entries,
            visible_content=visible_content,
            thinking_content=thinking_content,
        )
        snapshot_chars = len(visible_content) + len(thinking_content) + sum(
            len(str(entry.get("content") or "")) + len(str(entry.get("thinking") or ""))
            for entry in transcript_snapshot
            if isinstance(entry, dict)
        )
        if not snapshot_chars:
            return

        now = time.perf_counter()
        if (
            not force
            and last_snapshot_chars > 0
            and now - last_snapshot_at < STREAMING_ASSISTANT_SNAPSHOT_INTERVAL_SECONDS
            and snapshot_chars - last_snapshot_chars < STREAMING_ASSISTANT_SNAPSHOT_MIN_CHAR_DELTA
        ):
            return

        try:
            assistant_message_record.content = visible_content
            assistant_message_record.llm_transcript = transcript_snapshot
            assistant_message_record.save(update_fields=["content", "llm_transcript"])
            Chat.objects.filter(pk=chat.pk).update(updated_at=timezone.now())
            last_snapshot_at = now
            last_snapshot_chars = snapshot_chars
        except Exception:
            logger.exception("Failed to persist streaming assistant snapshot")

    # Build and persist one automatic compression marker at safe stream points.
    def maybe_apply_stream_context_compression(trigger: str, extra_context_text: str = "") -> str:
        nonlocal restart_after_stream_compression, stream_compression_applied

        if not persist_messages or stream_compression_applied:
            return ""

        draft_text = "\n".join(
            part
            for part in (
                current_stream_context_text(),
                _strip_llm_control_tokens(str(extra_context_text or "")).strip(),
            )
            if part
        )

        excluded_message_ids: set[int | str] = set()
        if assistant_message_record is not None:
            excluded_message_ids.add(assistant_message_record.id)
        if current_user_message_id is not None:
            excluded_message_ids.add(current_user_message_id)
        event = _build_manual_compression_event(
            chat=chat,
            system_prompt=system_prompt,
            draft_text=draft_text,
            engine=engine,
            model_name=model_name,
            model_info_payload=model_info_payload,
            force=False,
            exclude_message_ids=excluded_message_ids,
            summarize_with_model_enabled=False,
        )
        if not event:
            return ""

        arguments = event.get("arguments")
        if isinstance(arguments, dict):
            arguments["auto_trigger"] = trigger
            arguments["streaming"] = True
            arguments["restart_generation"] = True
        Message.objects.create(
            chat=chat,
            role="assistant",
            content="",
            llm_transcript=[event],
        )
        stream_compression_applied = True
        try:
            llm_api.abort_generation(engine)
        except Exception:
            logger.exception("Failed to abort generation after streaming compression")
        restart_after_stream_compression = True
        return _serialize_context_compression_marker(event)

    _print_runtime_event(
        "Chat started: "
        f"engine={engine}, "
        f"model={model_name}, "
        f"messages={len(llm_messages) if isinstance(llm_messages, list) else 0}, "
        f"images={image_count}, "
        f"tools={len(selected_tool_server_ids) if isinstance(selected_tool_server_ids, list) else 0}, "
        f"options={_summarize_option_keys(raw_options)}"
    )
    if settings.is_console_trace_enabled():
        _print_runtime_event(
            "Chat trace: "
            f"tool_servers={selected_tool_server_ids if isinstance(selected_tool_server_ids, list) else []}, "
            f"options_payload={json.dumps(raw_options, ensure_ascii=False, sort_keys=True) if isinstance(raw_options, dict) else raw_options}"
        )

    tracking_id = str(session_id or (chat.id if chat is not None else "") or "")
    try:
        with _generation_state_lock:
            _active_generation_id_by_engine[str(engine)] = str(generation_id or "")
            if tracking_id:
                _active_generation_id_by_chat_id[tracking_id] = str(generation_id or "")
        tool_context = generate_kwargs.get("tool_context")
        if isinstance(tool_context, dict):
            tool_context["generation_id"] = str(generation_id or "")
        if isinstance(compression_event, dict):
            transcript_entries.append(compression_event)
            persist_stream_snapshot(force=True)
            yield _serialize_context_compression_marker(compression_event)

        llm_api.prepare_runtime(engine)
        response_iterator = llm_api.generate(**generate_kwargs)
        for chunk in response_iterator:
            if isinstance(chunk, dict) and isinstance(chunk.get("tool_progress"), dict):
                if is_thinking:
                    is_thinking = False
                    yield "\n</think>\n"
                yield _serialize_tool_activity_marker(chunk["tool_progress"])
                continue

            # Save assistant/tool transcript entries that are not meant
            # to be rendered directly as plain chat text.
            if isinstance(chunk, dict) and chunk.get("transcript_message"):
                transcript_message = chunk["transcript_message"]
                if isinstance(transcript_message, dict):
                    transcript_thinking = _strip_llm_control_tokens(str(transcript_message.get("thinking") or "")).strip()
                    if not emit_thinking and transcript_message.get("thinking"):
                        transcript_message = {**transcript_message}
                        transcript_message.pop("thinking", None)
                    transcript_entries.append(transcript_message)
                    persist_stream_snapshot()
                    transcript_trigger = ""
                    transcript_trigger_context = transcript_thinking
                    if transcript_thinking:
                        transcript_trigger = "reasoning"
                    elif transcript_message.get("tool_calls"):
                        transcript_trigger = "tool_call"
                        transcript_trigger_context = json.dumps(transcript_message.get("tool_calls"), ensure_ascii=False)
                    if transcript_trigger:
                        compression_marker = maybe_apply_stream_context_compression(
                            transcript_trigger,
                            extra_context_text=transcript_trigger_context,
                        )
                        if compression_marker:
                            yield compression_marker
                            break
                continue

            if isinstance(chunk, dict) and chunk.get("tool_result"):
                tool_message = chunk["tool_result"]
                if isinstance(tool_message, dict):
                    transcript_entries.append(tool_message)
                    persist_stream_snapshot()
                    alias = str(tool_message.get("alias") or tool_message.get("tool_id") or tool_message.get("name") or "")
                    content = str(tool_message.get("content") or "")
                    tool_ui = tool_message.get("tool_ui") if isinstance(tool_message.get("tool_ui"), dict) else None
                    structured_content = (
                        tool_message.get("structured_content")
                        if isinstance(tool_message.get("structured_content"), dict)
                        else None
                    )
                    yield _serialize_tool_result_marker(
                        alias,
                        content,
                        tool_ui=tool_ui,
                        structured_content=structured_content,
                    )
                    compression_marker = maybe_apply_stream_context_compression("tool_result")
                    if compression_marker:
                        yield compression_marker
                        break
                continue

            if isinstance(chunk, dict) and isinstance(chunk.get("tool_events"), list):
                if is_thinking:
                    is_thinking = False
                    yield "\n</think>\n"
                markers = [
                    _serialize_tool_call_marker(tool_event)
                    for tool_event in chunk["tool_events"]
                    if isinstance(tool_event, dict)
                ]
                if markers:
                    yield "".join(markers)
                compression_marker = maybe_apply_stream_context_compression(
                    "tool_call",
                    extra_context_text=json.dumps(chunk["tool_events"], ensure_ascii=False),
                )
                if compression_marker:
                    yield compression_marker
                    break
                continue

            # Tool events are sent as inline markers for the frontend.
            if isinstance(chunk, dict) and chunk.get("tool_event"):
                if is_thinking:
                    is_thinking = False
                    yield "\n</think>\n"
                yield _serialize_tool_call_marker(chunk["tool_event"])
                compression_marker = maybe_apply_stream_context_compression(
                    "tool_call",
                    extra_context_text=json.dumps(chunk["tool_event"], ensure_ascii=False),
                )
                if compression_marker:
                    yield compression_marker
                    break
                continue

            if isinstance(chunk, dict):
                prompt_tokens = _coerce_positive_int(
                    chunk.get("prompt_eval_count")
                    or chunk.get("prompt_tokens")
                    or chunk.get("input_tokens")
                )
                completion_tokens = _coerce_positive_int(
                    chunk.get("eval_count")
                    or chunk.get("completion_tokens")
                    or chunk.get("output_tokens")
                )
                total_tokens = _coerce_positive_int(
                    chunk.get("total_tokens")
                )
                usage = chunk.get("usage")
                if isinstance(usage, dict):
                    prompt_tokens = prompt_tokens or _coerce_positive_int(
                        usage.get("prompt_tokens") or usage.get("input_tokens")
                    )
                    completion_tokens = completion_tokens or _coerce_positive_int(
                        usage.get("completion_tokens") or usage.get("output_tokens")
                    )
                    total_tokens = total_tokens or _coerce_positive_int(usage.get("total_tokens"))
                if prompt_tokens is not None:
                    usage_snapshot["prompt_tokens"] = int(prompt_tokens)
                if completion_tokens is not None:
                    usage_snapshot["completion_tokens"] = int(completion_tokens)
                if total_tokens is not None:
                    usage_snapshot["total_tokens"] = int(total_tokens)

            thinking_part, text_part = _extract_stream_message_parts(chunk)

            # Thinking fragments are wrapped in custom tags.
            if thinking_part:
                if emit_thinking:
                    thinking_parts.append(thinking_part)
                    persist_stream_snapshot()
                    if not is_thinking:
                        is_thinking = True
                        yield "<think>\n"
                    yield thinking_part
                compression_marker = maybe_apply_stream_context_compression(
                    "reasoning",
                    extra_context_text="" if emit_thinking else thinking_part,
                )
                if compression_marker:
                    if is_thinking:
                        is_thinking = False
                        yield "\n</think>\n"
                    yield compression_marker
                    break

            if restart_after_stream_compression:
                break

            # Visible text is streamed and stored separately.
            if text_part:
                if is_thinking:
                    is_thinking = False
                    yield "\n</think>\n"
                visible_parts.append(text_part)
                persist_stream_snapshot()
                yield text_part
    except Exception as exc:
        failed = True
        formatted_error = _format_runtime_error(engine, exc)
        if _is_expected_runtime_error(exc):
            logger.warning("Error during streaming generation: %s", formatted_error)
        else:
            logger.exception("Error during streaming generation")
        if is_thinking:
            yield "\n</think>\n"
        yield f"\n[Error during generation: {formatted_error}]"
    finally:
        with _generation_state_lock:
            active_id = str(_active_generation_id_by_engine.get(str(engine)) or "")
            if active_id == str(generation_id or ""):
                _active_generation_id_by_engine.pop(str(engine), None)
            if tracking_id:
                chat_active_id = str(_active_generation_id_by_chat_id.get(tracking_id) or "")
                if chat_active_id == str(generation_id or ""):
                    _active_generation_id_by_chat_id.pop(tracking_id, None)
        if is_thinking:
            yield "\n</think>\n"

        visible_content = _strip_llm_control_tokens("".join(visible_parts)).strip()
        thinking_content = _strip_llm_control_tokens("".join(thinking_parts)).strip()
        transcript_assistant_visible_parts: list[str] = []
        transcript_has_renderable_payload = False
        for entry in transcript_entries:
            if not isinstance(entry, dict):
                continue
            role = str(entry.get("role") or "").lower()
            content = _strip_llm_control_tokens(str(entry.get("content") or "")).strip()
            thinking = _strip_llm_control_tokens(str(entry.get("thinking") or "")).strip()
            if role == "assistant":
                if content:
                    transcript_assistant_visible_parts.append(content)
                    transcript_has_renderable_payload = True
                if thinking:
                    transcript_has_renderable_payload = True
            elif role == "tool":
                alias = str(entry.get("alias") or entry.get("tool_id") or "").strip().lower()
                # Compression markers alone should not become an "empty model reply".
                if alias and alias != "context_compression_summary":
                    transcript_has_renderable_payload = True
                elif content and not alias:
                    transcript_has_renderable_payload = True

        if not visible_content and transcript_assistant_visible_parts:
            visible_content = _strip_llm_control_tokens("".join(transcript_assistant_visible_parts)).strip()

        transcript_entries = _build_streaming_assistant_transcript(
            transcript_entries,
            visible_content=visible_content,
            thinking_content=thinking_content,
        )

        should_persist_message = bool(visible_content or thinking_content or transcript_has_renderable_payload)
        if persist_messages and assistant_message_record is not None and chat is not None:
            if should_persist_message:
                assistant_message_record.content = visible_content
                assistant_message_record.llm_transcript = transcript_entries
                assistant_message_record.save(update_fields=["content", "llm_transcript"])
                # Bump chat ordering so sidebar reflects the latest activity.
                Chat.objects.filter(pk=chat.pk).update(updated_at=timezone.now())
            else:
                # Nothing was generated; drop the empty placeholder we pre-created.
                assistant_message_record.delete()

        duration_seconds = time.perf_counter() - started_at
        if usage_snapshot and tracking_id:
            prompt_tokens_observed = _coerce_positive_int(usage_snapshot.get("prompt_tokens"))
            if prompt_tokens_observed and prompt_chars_estimate > 0:
                usage_snapshot["prompt_chars_estimate"] = int(prompt_chars_estimate)
                usage_snapshot["observed_chars_per_token"] = float(prompt_chars_estimate / max(1, prompt_tokens_observed))
            with _chat_usage_lock:
                _chat_usage_by_chat_id[tracking_id] = {
                    **usage_snapshot,
                    "engine": engine,
                    "model": model_name,
                    "updated_at": timezone.now().isoformat(),
                }
        _print_runtime_event(
            "Chat completed: "
            f"engine={engine}, "
            f"model={model_name}, "
            f"status={'failed' if failed else 'ok'}, "
            f"took={duration_seconds:.2f}s, "
            f"visible_chars={len(visible_content)}, "
            f"transcript_entries={len(transcript_entries)}"
        )


# Handle a stateless generation request for external modules.
def generate_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        history_messages = _normalize_request_history_messages(data.get("messages") or [])
        history_messages, user_message, attachments, uploaded_file_ids = _split_request_conversation(
            data,
            history_messages,
        )

        prepared = _prepare_generation_request(
            request,
            data,
            route="/api/generate/",
            user_message=user_message,
            attachments=attachments,
            uploaded_file_ids=uploaded_file_ids,
        )
        if isinstance(prepared, JsonResponse):
            return prepared

        source_tool_ids = {
            str(raw_id or "").strip()
            for source in prepared.tool_sources or []
            for raw_id in source.get("tool_server_ids") or []
        }
        command_plan = _parse_chat_command_plan(
            user_message,
            [server["id"] for server in prepared.selected_tool_servers],
            is_known_tool_id=lambda tool_id: (
                tool_id in source_tool_ids
                or tool_registry.get_server(tool_id, engine=prepared.engine, model_name=prepared.model_name) is not None
            ),
        )
        selected_tool_servers = prepared.selected_tool_servers
        if command_plan.tool_server_ids != [server["id"] for server in prepared.selected_tool_servers]:
            selected_tool_servers = _resolve_tool_servers(
                prepared.engine,
                prepared.model_name,
                command_plan.tool_server_ids,
                tool_sources=prepared.tool_sources,
            )
            _validate_tool_server_support(
                prepared.engine,
                prepared.model_name,
                [server["id"] for server in selected_tool_servers],
                payload=prepared.model_info_payload,
            )

        session_id = str(data.get("session_id") or "").strip() or str(uuid.uuid4())
        consume_skill_notifications = data.get("consume_skill_notifications", True)
        if not isinstance(consume_skill_notifications, bool):
            consume_skill_notifications = bool(consume_skill_notifications)
        system_prompt = _compose_system_prompt(
            data.get("system_prompt", ""),
            consume_skill_notifications=consume_skill_notifications,
            include_skills_baseline=_resolve_include_skills_baseline(data, history_messages),
            forced_skill_names=command_plan.forced_skill_names,
            instant_mode=prepared.instant_mode,
        )
        system_prompt = _append_command_directives_to_system_prompt(
            system_prompt,
            _command_directives_for_mode(command_plan, instant_mode=prepared.instant_mode),
        )
        current_entry = _build_current_user_llm_entry(
            command_plan.message,
            prepared.attachments,
            prepared.upload_manifests,
            supported_media_kinds=_supported_model_media_kinds(
                prepared.engine,
                prepared.model_info_payload,
            ),
        )
        llm_messages, compression_event = _build_generate_llm_messages(
            history_messages,
            current_entry,
            system_prompt,
            prepared.engine,
            prepared.model_name,
            prepared.model_info_payload,
            session_id=session_id,
            sandbox_enabled=_selected_tools_include_sandbox(selected_tool_servers),
        )
        model_info_payload = prepared.model_info_payload
        external_tool_context = dict(prepared.external_tool_context)
        _apply_command_tool_context(
            external_tool_context,
            command_plan,
            instant_mode=prepared.instant_mode,
        )
        generate_kwargs = _build_generate_kwargs(
            prepared.engine,
            prepared.model_name,
            llm_messages,
            prepared.think_value,
            prepared.think_level_value,
            prepared.clean_options,
            session_id,
            selected_tool_servers,
            think_param_name=str(model_info_payload.get("think_param_name", "think") or "think"),
            think_level_param_name=str(model_info_payload.get("think_level_param_name", "think_level") or "think_level"),
            sync_operation_defaults=prepared.sync_operation_defaults,
            tool_sources=prepared.tool_sources,
            external_tool_context=external_tool_context,
            instant_mode=prepared.instant_mode,
        )

        generation_id = uuid.uuid4().hex
        response = StreamingHttpResponse(
            _stream_chat_response(
                prepared.engine,
                generate_kwargs,
                generation_id,
                session_id=session_id,
                compression_event=compression_event,
                model_info_payload=prepared.model_info_payload,
                system_prompt=system_prompt,
                persist_messages=False,
            ),
            content_type="text/plain; charset=utf-8",
        )
        response["X-LLM-Engine"] = prepared.engine
        response["X-Session-ID"] = session_id
        response["X-Generation-ID"] = generation_id
        return response

    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Unhandled exception in generate_api")
        return JsonResponse({"error": str(exc)}, status=500)


# Abort active generation.
def abort_generation_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        engine, engine_error = _resolve_request_engine_or_response(request, data)
        if engine_error is not None:
            return engine_error
        generation_id = str(data.get("generation_id") or "").strip()
        if generation_id:
            with _generation_state_lock:
                active_id = str(_active_generation_id_by_engine.get(str(engine)) or "")
            if not active_id or active_id != generation_id:
                return JsonResponse({"ok": True, "ignored": True, "reason": "generation_mismatch"})
        llm_api.abort_generation(engine, generation_id=generation_id or None)
        return JsonResponse({"ok": True})
    except Exception as exc:
        logger.exception("Failed to abort generation")
        return JsonResponse({"error": str(exc)}, status=500)


# Return persistent capability markers independently of model presets.
def get_model_capabilities_api(request):
    """Serve cached markers, resolving only visible uncached/stale models on demand."""
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)
    engine, engine_error = _resolve_request_engine_or_response(request)
    if engine_error is not None:
        return engine_error
    endpoint = model_capabilities.get_endpoint(engine)
    if "endpoint" in request.GET and str(request.GET["endpoint"]).strip().rstrip("/") != endpoint:
        return JsonResponse({"error": "Engine endpoint changed"}, status=409)

    records = model_capabilities.list_cached(engine, endpoint)
    model_name = str(request.GET.get("model", "")).strip()
    if not model_name:
        return JsonResponse({"engine": engine, "records": records, "ttl": model_capabilities.CACHE_TTL_SECONDS})
    cached = next((row for row in records if row["model"] == model_name), None)
    if cached and model_capabilities.is_fresh(cached):
        return JsonResponse(cached)
    try:
        payload = _get_cached_model_info(engine, model_name)
        if payload is None:
            raw = llm_api.get_model_settings(engine, model_name)
            payload = _extract_model_info(engine, model_name, raw)
        if endpoint != model_capabilities.get_endpoint(engine):
            return JsonResponse({"error": "Engine endpoint changed"}, status=409)
        record = model_capabilities.store(engine, model_name, endpoint, payload)
        if record is None:
            raise RuntimeError("Model capability metadata is unavailable")
        return JsonResponse(record)
    except Exception as exc:
        logger.warning("Could not refresh model capabilities for %s on %s: %s", model_name, engine, exc)
        if cached:
            return JsonResponse({**cached, "stale": True})
        return JsonResponse({"error": _format_runtime_error(engine, exc)}, status=503)


# Return model metadata for the selected engine.
def get_model_info_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    model_name = request.GET.get("model", "")
    if not model_name:
        return JsonResponse({"error": "Model parameter is required"}, status=400)

    engine, engine_error = _resolve_request_engine_or_response(request)
    if engine_error is not None:
        return engine_error
    started_at = time.perf_counter()

    try:
        payload = _build_model_info_payload(engine, model_name)
        _remember_active_model(engine, model_name)
        _print_runtime_event(
            "Model info loaded: "
            f"engine={engine}, "
            f"model={model_name}, "
            f"tools={len(payload.get('available_tool_servers', []) or [])}, "
            f"options={_summarize_option_keys(payload.get('defaults', {}))}, "
            f"took={(time.perf_counter() - started_at):.2f}s"
        )
        return JsonResponse(payload)
    except NotImplementedError as exc:
        logger.info("Model info is not implemented for engine %s: %s", engine, exc)
        _print_runtime_event(f"Model info not supported: engine={engine}, model={model_name}")
        return JsonResponse({"error": str(exc)}, status=501)
    except Exception as exc:
        formatted_error = _format_runtime_error(engine, exc)
        if _is_expected_runtime_error(exc):
            logger.warning("Error getting model info for %s on engine %s: %s", model_name, engine, formatted_error)
        else:
            logger.exception("Error getting model info for %s on engine %s", model_name, engine)
        _print_runtime_event(f"Model info failed: engine={engine}, model={model_name}, error={formatted_error}")
        return JsonResponse({"error": formatted_error}, status=500)


# Return unified runtime inference metadata for the active engine/model.
def get_inference_info_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    engine, engine_error = _resolve_request_engine_or_response(request)
    if engine_error is not None:
        return engine_error
    model_name, model_source = _resolve_inference_model(engine, request.GET.get("model"))
    if not model_name:
        return JsonResponse(
            {
                "ok": False,
                "engine": engine,
                "engine_label": _get_engine_label(engine),
                "model": "",
                "model_name": "",
                "context_window": None,
                "model_context_limit": None,
                "max_output_tokens": None,
                "output_token_limit": None,
                "limits": {
                    "context_window": None,
                    "model_context_limit": None,
                    "max_output_tokens": None,
                    "output_token_limit": None,
                },
                "capabilities": {
                    "items": [],
                    "supports_thinking": False,
                    "supports_think_toggle": False,
                    "supports_think_level": False,
                    "supports_vision": False,
                    "supports_tool_calling": False,
                    "supports_files": False,
                },
                "generation_defaults": {},
                "supported_parameters": [],
                "runtime_limits": {},
                "tool_servers": [],
                "source": {"model": model_source},
                "error": "No active model is available for the selected engine.",
            },
            status=404,
        )

    started_at = time.perf_counter()

    try:
        model_info_payload = _build_model_info_payload(engine, model_name)
        if model_source == "request":
            _remember_active_model(engine, model_name)
        payload = _build_inference_info_payload(engine, model_name, model_info_payload, model_source)
        _sync_runtime_model_metadata(
            engine,
            model_name,
            model_info_payload,
            source=model_source,
            route="/api/inference_info/",
        )
        _print_runtime_event(
            "Inference info loaded: "
            f"engine={engine}, "
            f"model={model_name}, "
            f"context={payload.get('context_window')}, "
            f"output={payload.get('max_output_tokens')}, "
            f"took={(time.perf_counter() - started_at):.2f}s"
        )
        return JsonResponse(payload)
    except NotImplementedError as exc:
        logger.info("Inference info is not implemented for engine %s: %s", engine, exc)
        _print_runtime_event(f"Inference info not supported: engine={engine}, model={model_name}")
        return JsonResponse({"ok": False, "engine": engine, "model": model_name, "error": str(exc)}, status=501)
    except Exception as exc:
        formatted_error = _format_runtime_error(engine, exc)
        if _is_expected_runtime_error(exc):
            logger.warning("Error getting inference info for %s on engine %s: %s", model_name, engine, formatted_error)
        else:
            logger.exception("Error getting inference info for %s on engine %s", model_name, engine)
        _print_runtime_event(f"Inference info failed: engine={engine}, model={model_name}, error={formatted_error}")
        return JsonResponse({"ok": False, "engine": engine, "model": model_name, "error": formatted_error}, status=500)


# Return the model list for the selected engine.
def get_models_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    engine, engine_error = _resolve_request_engine_or_response(request)
    if engine_error is not None:
        return engine_error

    # Polling requests force a fresh probe so newly loaded/downloaded models and
    # recovered connections surface without waiting for the cache TTL to expire.
    if str(request.GET.get("refresh", "")).strip().lower() in {"1", "true", "yes", "on"}:
        _clear_cached_model_list(engine)

    started_at = time.perf_counter()
    models = _load_models_for_engine(engine)
    _print_runtime_event(
        f"Models loaded: engine={engine}, count={len(models)}, took={(time.perf_counter() - started_at):.2f}s"
    )
    return JsonResponse({"engine": engine, "models": models})


# Return estimated/observed context usage for the current chat and model.
def get_context_usage_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        engine, engine_error = _resolve_request_engine_or_response(request)
        if engine_error is not None:
            return engine_error
        model_name = str(request.GET.get("model") or _get_remembered_active_model(engine) or "").strip()
        chat_id = str(request.GET.get("chat_id") or "").strip()
        draft_text = str(request.GET.get("draft") or "")

        model_info_payload = _build_model_info_payload(engine, model_name, allow_fallback=True)
        chat: Chat | None = None
        if chat_id:
            try:
                chat = Chat.objects.get(id=chat_id)
            except Chat.DoesNotExist:
                chat = None

        system_prompt = _compose_system_prompt(
            str(request.GET.get("system_prompt") or ""),
            consume_skill_notifications=False,
            include_skills_baseline=_chat_is_first_user_turn(chat),
            instant_mode=str(request.GET.get("instant_mode") or "").strip().lower() in {"1", "true", "yes", "on"},
        )

        estimate = _estimate_context_usage(
            chat=chat,
            system_prompt=system_prompt,
            draft_text=draft_text,
            model_info_payload=model_info_payload,
            active_engine=engine,
            active_model=model_name,
        )
        with _chat_usage_lock:
            observed = dict(_chat_usage_by_chat_id.get(chat_id, {})) if chat_id else {}

        response_payload = {
            "engine": engine,
            "model": model_name,
            "chat_id": chat_id,
            "context_window_tokens": estimate["context_window_tokens"],
            "estimated_used_tokens": estimate["estimated_used_tokens"],
            "estimated_used_chars": estimate["estimated_used_chars"],
            "ratio": estimate["ratio"],
            "threshold_ratio": LLM_HISTORY_COMPRESSION_TRIGGER_RATIO,
            "compressed_context_active": bool(estimate.get("compressed_context_active")),
            "observed_usage": observed,
            "token_estimator": {
                "mode": "adaptive",
                "base_chars_per_token": estimate.get("base_chars_per_token"),
                "effective_chars_per_token": estimate.get("effective_chars_per_token"),
                "chat_observed_chars_per_token": estimate.get("observed_chars_per_token"),
                "observed_chars_per_token": observed.get("observed_chars_per_token"),
                "prompt_chars_estimate": observed.get("prompt_chars_estimate"),
            },
        }
        return JsonResponse(response_payload)
    except Exception as exc:
        logger.exception("Failed to build context usage payload")
        return JsonResponse({"error": str(exc)}, status=500)


# Return whether history compression should run for one stateless usage snapshot.
def context_compression_decide_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        engine, engine_error = _resolve_request_engine_or_response(request, data)
        if engine_error is not None:
            return engine_error
        model_name = str(data.get("model") or _get_remembered_active_model(engine) or "").strip()
        debug_force_4k = str(os.getenv("ASLM_DEBUG_CONTEXT_COMPRESSION_4K", "")).strip().lower() in {"1", "true", "yes", "on"}
        model_info_payload = data.get("model_info")
        if not isinstance(model_info_payload, dict):
            model_info_payload = _build_model_info_payload(engine, model_name, allow_fallback=True)

        trigger_ratio = data.get("trigger_ratio")
        try:
            parsed_trigger_ratio = float(trigger_ratio) if trigger_ratio is not None else LLM_HISTORY_COMPRESSION_TRIGGER_RATIO
        except (TypeError, ValueError):
            parsed_trigger_ratio = LLM_HISTORY_COMPRESSION_TRIGGER_RATIO

        compression = decide_compression(
            used_history_chars=int(data.get("used_history_chars") or 0),
            history_budget_chars=int(data.get("history_budget_chars") or 0),
            model_info_payload=model_info_payload,
            runtime_metadata_path=MODEL_RUNTIME_METADATA_PATH,
            active_engine=engine,
            active_model=model_name,
            debug_force_4k=debug_force_4k,
            trigger_ratio=parsed_trigger_ratio,
        )
        return JsonResponse({
            "ok": True,
            "enabled": compression.enabled,
            "context_window_tokens": compression.context_window_tokens,
            "history_budget_chars": compression.history_budget_chars,
            "reason": compression.reason,
        })
    except Exception as exc:
        logger.exception("Failed to decide context compression")
        return JsonResponse({"error": str(exc)}, status=500)


# Build one compression event from stateless overflow data for external module consumers.
def context_compression_build_event_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        engine, engine_error = _resolve_request_engine_or_response(request, data)
        if engine_error is not None:
            return engine_error
        model_name = str(data.get("model") or _get_remembered_active_model(engine) or "").strip()
        force = bool(data.get("force"))
        debug_force_4k = str(os.getenv("ASLM_DEBUG_CONTEXT_COMPRESSION_4K", "")).strip().lower() in {"1", "true", "yes", "on"}
        model_info_payload = data.get("model_info")
        if not isinstance(model_info_payload, dict):
            model_info_payload = _build_model_info_payload(engine, model_name, allow_fallback=True)

        overflow_entries = data.get("overflow_entries") or []
        summary_source_entries = data.get("summary_source_entries") or []
        if not isinstance(overflow_entries, list):
            overflow_entries = []
        if not isinstance(summary_source_entries, list):
            summary_source_entries = []

        recent_user_messages = data.get("recent_user_messages") or []
        direct_user_directives = data.get("direct_user_directives") or []
        if not isinstance(recent_user_messages, list):
            recent_user_messages = []
        if not isinstance(direct_user_directives, list):
            direct_user_directives = []

        trigger_ratio = data.get("trigger_ratio")
        try:
            parsed_trigger_ratio = float(trigger_ratio) if trigger_ratio is not None else LLM_HISTORY_COMPRESSION_TRIGGER_RATIO
        except (TypeError, ValueError):
            parsed_trigger_ratio = LLM_HISTORY_COMPRESSION_TRIGGER_RATIO

        event = _build_stateless_compression_event(
            engine=engine,
            model_name=model_name,
            model_info_payload=model_info_payload,
            force=force,
            used_history_chars=int(data.get("used_history_chars") or 0),
            history_budget_chars=int(data.get("history_budget_chars") or 0),
            overflow_entries=overflow_entries,
            summary_source_entries=summary_source_entries,
            recent_user_messages=[str(value) for value in recent_user_messages],
            direct_user_directives=[str(value) for value in direct_user_directives],
            summarize_with_model_enabled=bool(data.get("summarize_with_model", True)),
            debug_force_4k=debug_force_4k,
            trigger_ratio=parsed_trigger_ratio,
            compression_mode=str(data.get("compression_mode") or "manual"),
        )
        if not event:
            return JsonResponse({"ok": True, "event": None, "reason": "below_threshold"})
        return JsonResponse({"ok": True, "event": event})
    except Exception as exc:
        logger.exception("Failed to build stateless compression event")
        return JsonResponse({"error": str(exc)}, status=500)


# Force or opportunistically run context compression and persist a timeline marker.
def context_compress_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        engine, engine_error = _resolve_request_engine_or_response(request, data)
        if engine_error is not None:
            return engine_error
        model_name = str(data.get("model") or _get_remembered_active_model(engine) or "").strip()
        chat_id = str(data.get("chat_id") or "").strip()
        if not chat_id:
            return JsonResponse({"ok": True, "applied": False, "reason": "missing_chat_id"})
        force = bool(data.get("force"))
        draft_text = str(data.get("draft") or data.get("draft_text") or "")

        try:
            chat = Chat.objects.get(id=chat_id)
        except Chat.DoesNotExist:
            return JsonResponse({"ok": True, "applied": False, "reason": "chat_not_found"})

        system_prompt = _compose_system_prompt(
            str(data.get("system_prompt") or ""),
            consume_skill_notifications=False,
            include_skills_baseline=False,
        )

        with _generation_state_lock:
            active_generation_id = str(_active_generation_id_by_chat_id.get(str(chat.id)) or "")
        if active_generation_id:
            return JsonResponse({
                "ok": True,
                "applied": False,
                "reason": "generation_active",
            }, status=409)

        model_info_payload = _build_model_info_payload(engine, model_name, allow_fallback=True)
        event = _build_manual_compression_event(
            chat=chat,
            system_prompt=system_prompt,
            draft_text=draft_text,
            engine=engine,
            model_name=model_name,
            model_info_payload=model_info_payload,
            force=force,
        )
        if not event:
            return JsonResponse({"ok": True, "applied": False, "reason": "below_threshold"})

        message = Message.objects.create(
            chat=chat,
            role="assistant",
            content="",
            llm_transcript=[event],
        )
        Chat.objects.filter(pk=chat.pk).update(updated_at=timezone.now())
        return JsonResponse({
            "ok": True,
            "applied": True,
            "message": _serialize_message(message, include_attachment_data=False),
        })
    except Exception as exc:
        logger.exception("Failed to run context compression")
        return JsonResponse({"error": str(exc)}, status=500)
