# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

# Character and window limits for compression prompts and stored summaries.
COMPRESSION_ENTRY_MAX_CHARS = 2400
ANALYTIC_ENTRY_MAX_CHARS = 360
MEMORY_ENTRY_MAX_CHARS = 1200
COMPRESSION_FIRST_CONTEXT_ENTRIES = 3
COMPRESSION_LAST_CONTEXT_ENTRIES = 5
TOOL_OBSERVATION_MAX_CHARS = 700

# Hosts and markers stripped or down-ranked during memory sanitization.
NOISY_URL_HOSTS = {
    "127.0.0.1",
    "localhost",
    "www.bing.com",
    "bing.com",
}

# Lowercased line prefixes of search-tool boilerplate (citation instructions, result headers) that is dropped.
RAW_TOOL_NOISE_MARKERS = (
    "citation rules",
    "cite search evidence",
    "use the exact citation",
    "put the citation handle",
    "search results for:",
)


# Compression decision

# Snapshot returned when evaluating whether history compression should run.
@dataclass
class CompressionDecision:
    enabled: bool
    context_window_tokens: int
    history_budget_chars: int
    reason: str


# Resolve context window size from model info, then from runtime metadata when available.
def resolve_context_window_tokens(
    model_info_payload: dict[str, Any] | None,
    *,
    runtime_metadata_path: Path | None = None,
    active_engine: str = "",
    active_model: str = "",
) -> int:
    # Coerce a value to a positive int, returning 0 for bools, None, non-numbers and non-positive numbers.
    def _positive_int(value: Any) -> int:
        if isinstance(value, bool) or value is None:
            return 0
        try:
            number = int(value)
        except (TypeError, ValueError):
            return 0
        return number if number > 0 else 0

    # Prefer explicit limits from the active model payload.
    payload_model_limit = 0
    if isinstance(model_info_payload, dict):
        defaults = model_info_payload.get("defaults", {})
        if isinstance(defaults, dict):
            for key in ("num_ctx", "context_window", "contextWindow", "input_token_limit", "inputTokenLimit"):
                number = _positive_int(defaults.get(key))
                if number:
                    return number

        limits = model_info_payload.get("limits", {})
        if isinstance(limits, dict):
            for key in ("context_window", "contextWindow", "num_ctx"):
                number = _positive_int(limits.get(key))
                if number:
                    return number

        for key in ("context_window", "contextWindow", "num_ctx", "input_token_limit", "inputTokenLimit"):
            number = _positive_int(model_info_payload.get(key))
            if number:
                return number

        payload_model_limit = _positive_int(model_info_payload.get("context_length"))

    if runtime_metadata_path is None:
        return payload_model_limit

    # Fall back to the on-disk runtime metadata registry.
    try:
        payload = json.loads(runtime_metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return payload_model_limit

    if not isinstance(payload, dict):
        return payload_model_limit

    models = payload.get("models", {})
    if not isinstance(models, dict):
        return payload_model_limit

    model_key = f"{active_engine}:{active_model}" if active_engine and active_model else ""
    targets: list[dict[str, Any]] = []

    # Search the requested model first, then the model marked active in the registry, then every registered model.
    if model_key and isinstance(models.get(model_key), dict):
        targets.append(models[model_key])
    active = payload.get("active", {})
    if isinstance(active, dict):
        active_key = f"{active.get('engine')}:{active.get('model')}"
        if isinstance(models.get(active_key), dict):
            targets.append(models[active_key])

    for item in models.values():
        if isinstance(item, dict):
            targets.append(item)

    # The configured context window of the first target that defines one wins over any model limit.
    for target in targets:
        limits = target.get("limits", {})
        if not isinstance(limits, dict):
            continue
        context_window = _positive_int(limits.get("context_window"))
        if context_window:
            return context_window

    # No configured window anywhere: use the first model_context_limit.
    for target in targets:
        limits = target.get("limits", {})
        if not isinstance(limits, dict):
            continue
        model_limit = _positive_int(limits.get("model_context_limit"))
        if model_limit:
            return model_limit

    return payload_model_limit


# Decide whether history compression should run for the current character budget.
def decide_compression(
    *,
    used_history_chars: int,
    history_budget_chars: int,
    model_info_payload: dict[str, Any] | None,
    runtime_metadata_path: Path | None,
    active_engine: str,
    active_model: str,
    debug_force_4k: bool = False,
    trigger_ratio: float = 0.80,
) -> CompressionDecision:
    # Resolve the context window, which debug mode overrides with a fixed 4096 tokens.
    context_window_tokens = resolve_context_window_tokens(
        model_info_payload,
        runtime_metadata_path=runtime_metadata_path,
        active_engine=active_engine,
        active_model=active_model,
    )
    if debug_force_4k:
        context_window_tokens = 4096

    # Compression is enabled once the used history reaches trigger_ratio (clamped to 0..1) of the budget.
    threshold = int(max(0, history_budget_chars) * max(0.0, min(1.0, trigger_ratio)))
    enabled = history_budget_chars > 0 and used_history_chars >= threshold
    reason = f"used={used_history_chars}, threshold={threshold}, budget={history_budget_chars}, ctx={context_window_tokens}"
    return CompressionDecision(
        enabled=enabled,
        context_window_tokens=context_window_tokens,
        history_budget_chars=history_budget_chars,
        reason=reason,
    )


# Transcript text cleanup

# Remove model control tokens and thinking wrappers from transcript text.
def _strip_control_tokens(text: str) -> str:
    cleaned = str(text or "")
    cleaned = re.sub(r"<\|start\|>.*?<\|message\|>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = re.sub(r"<\|[^>]+\|>", "", cleaned)
    return cleaned.strip()


# Format one history entry as role-prefixed text for generic compression prompts.
def _entry_text(entry: dict[str, Any], max_chars: int = COMPRESSION_ENTRY_MAX_CHARS) -> str:
    role = str(entry.get("role") or "unknown")
    content = _strip_control_tokens(str(entry.get("content") or ""))
    thinking = _strip_control_tokens(str(entry.get("thinking") or ""))
    parts = [part for part in (content, thinking) if part]
    raw = "\n".join(parts).strip()
    if len(raw) > max_chars:
        raw = raw[:max_chars].rstrip() + "..."
    return f"{role}: {raw}" if raw else f"{role}:"


# URL and path validation

# Return True when a URL points at localhost, Bing, or other low-value hosts.
def _is_noisy_url(url: str) -> bool:
    raw = str(url or "").strip().strip("\"'`<>").rstrip(".,;:)]}")
    if not raw:
        return True
    match = re.match(r"^https?://([^/?#]+)", raw, flags=re.IGNORECASE)
    host = match.group(1).lower() if match else ""
    host = host.split(":", 1)[0]
    if host in NOISY_URL_HOSTS:
        return True
    if host.endswith(".bing.com"):
        return True
    return False


# Normalize a URL string and drop noisy or empty values.
def _clean_url(url: str) -> str:
    raw = str(url or "").strip().strip("\"'`<>").rstrip(".,;:)]}")
    if not raw or _is_noisy_url(raw):
        return ""
    return raw


# Detect host/path strings that should be treated as URLs rather than file paths.
def _looks_like_web_host_path(value: str) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text.startswith(("http://", "https://", "//")):
        return True
    return bool(re.match(r"^(?:www\.)?[^/\\]+\.(?:com|org|net|ru|io|ai|dev|gov|edu|co|tv)(?:[/\\].*)?$", text))


# Method/attribute tails from source code that must not be treated as file extensions.
_CODE_TOKEN_EXTENSIONS = frozenset(
    {
        "strip",
        "match",
        "split",
        "join",
        "find",
        "read",
        "write",
        "append",
        "extend",
        "lower",
        "upper",
        "format",
        "draw",
        "add",
        "get",
        "set",
        "empty",
        "multiline",
        "findall",
        "startswith",
        "endswith",
        "enqueue",
        "isenabled",
        "visible",
        "width",
        "opacity",
        "count",
        "take",
        "text",
    }
)


# Return the final path segment from a Windows or POSIX path string.
def _file_basename(value: str) -> str:
    return str(value or "").rsplit("/", 1)[-1].rsplit("\\", 1)[-1]


# Return the extension after the last dot, matching Path.suffix semantics.
def _file_extension(value: str) -> str:
    return Path(_file_basename(value)).suffix


# Return the extension body without the leading dot.
def _extension_body(value: str) -> str:
    return _file_extension(value).lstrip(".")


# Reject identifier.token shapes produced by the file regex over source code.
def _looks_like_code_fragment(value: str) -> bool:
    name = _file_basename(value)
    if "/" in value or "\\" in value:
        return False
    if "." not in name:
        return True

    stem, _, ext = name.rpartition(".")
    if not stem or not ext:
        return True
    if ext.isupper() and len(ext) > 4:
        return True
    if len(stem) <= 2 and stem.isascii() and stem.islower():
        return True
    if len(ext) == 1:
        return True
    if ext != ext.lower() and ext != ext.upper():
        return True
    return ext.lower() in _CODE_TOKEN_EXTENSIONS


# Validate a candidate file path extracted from chat text.
def _looks_like_valid_path(value: str) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if re.search(r"\\[nr]\s", text):
        return False
    if "=" in text or "(" in text or ")" in text:
        return False
    if re.fullmatch(r"\d+(\.\d+)?", text):
        return False
    if re.search(r"\s{2,}", text):
        return False
    if len(text.split()) > 4:
        return False
    if "/" not in text and "\\" not in text:
        extension = _file_extension(text)
        if not re.fullmatch(r"\.[A-Za-z0-9]{1,12}", extension):
            return False
        if _looks_like_code_fragment(text):
            return False
    return True


# Memory sanitization

# Lowercased assistant phrases that indicate navigation filler, not durable facts.
_ASSISTANT_NAV_PREFIXES = (
    "assistant: now let me",
    "assistant: let me",
    "assistant: let me now",
    "assistant: let me check",
    "assistant: let me look",
    "assistant: let me read",
    "assistant: let me verify",
    "assistant: let me do",
    "assistant: let me continue",
    "assistant: let me get",
    "assistant: now i",
)


# Return True when text matches known assistant navigation openers.
def _is_assistant_navigation(text: str) -> bool:
    lowered = str(text or "").lower().strip()
    return any(lowered.startswith(prefix) for prefix in _ASSISTANT_NAV_PREFIXES)


# Require minimum length and reject title-case-only heading fragments.
def _passes_semantic_threshold(text: str) -> bool:
    if len(text) < 15:
        return False
    bare = text.rstrip(":").strip()

    # Title-cased text with digits is usually a real fact ("Django 5.2 REST API"),
    # not a heading fragment — only digit-free title case is treated as a heading.
    if bare and bare == bare.title() and not any(ch.isdigit() for ch in bare):
        return False
    return True


# Remove repeated tool boilerplate while preserving useful facts.
def _clean_memory_text(text: str) -> str:
    lines: list[str] = []
    skip_citation_block = False
    for raw_line in _strip_control_tokens(text).splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        if not line:
            skip_citation_block = False
            continue
        lowered = line.lower()
        if lowered.startswith(RAW_TOOL_NOISE_MARKERS):
            skip_citation_block = True
            continue
        if skip_citation_block and lowered.startswith(("-", "*")):
            continue
        skip_citation_block = False
        if lowered in {"---", "done", "running"}:
            continue
        lines.append(line)
    return "\n".join(lines).strip()


# Build a sanitized role-prefixed memory line for one transcript entry.
def _entry_memory_text(entry: dict[str, Any], max_chars: int = MEMORY_ENTRY_MAX_CHARS) -> str:
    role = str(entry.get("role") or "unknown").lower()
    content = _clean_memory_text(str(entry.get("content") or ""))
    thinking = "" if role == "assistant" else _clean_memory_text(str(entry.get("thinking") or ""))
    raw = "\n".join(part for part in (content, thinking) if part).strip()
    if not raw:
        return f"{role}:"

    # Collapse web-search tool dumps into a short structured observation.
    if role == "tool":
        site_match = re.search(r"\*\*Site:\*\*\s*([^\n]+)", raw)
        url_match = re.search(r"\*\*URL:\*\*\s*([^\n]+)", raw)
        title_match = re.search(r"^\s*#\s+(.+)$", raw, flags=re.MULTILINE)
        query_match = re.search(r"Search results for:\s*([^\n]+)", raw, flags=re.IGNORECASE)
        parts: list[str] = []
        if query_match:
            parts.append(f"Search query: {query_match.group(1).strip()}")
        if title_match:
            parts.append(f"Source title: {title_match.group(1).strip()}")
        if site_match:
            parts.append(f"Site: {site_match.group(1).strip()}")
        if url_match and not _is_noisy_url(url_match.group(1).strip()):
            parts.append(f"URL: {url_match.group(1).strip()}")
        for line in _line_candidates(raw):
            lowered = line.lower()
            if lowered.startswith(RAW_TOOL_NOISE_MARKERS):
                continue
            if lowered.startswith(("site:**", "url:**", "**site", "**url")):
                continue
            if lowered.count("http://") + lowered.count("https://") > 1:
                continue
            if any(_is_noisy_url(url) for url in re.findall(r"https?://[^\s)>\]\"']+", line, flags=re.IGNORECASE)):
                continue
            if len(parts) >= 8:
                break
            parts.append(line[:260])
        raw = "\n".join(parts) if parts else raw

    if len(raw) > max_chars:
        raw = raw[:max_chars].rstrip() + "..."
    return f"{role}: {raw}" if raw else f"{role}:"


# Return a compact observation from a tool result without raw tool-call logs.
def _tool_observation_text(entry: dict[str, Any], max_chars: int = TOOL_OBSERVATION_MAX_CHARS) -> str:
    text = _clean_memory_text(str(entry.get("content") or ""))
    if not text:
        return ""

    # Locate the tool name and the labeled fields (title, site, URL, error signal) in the cleaned output.
    tool_name = str(entry.get("tool_name") or entry.get("name") or entry.get("alias") or entry.get("tool_id") or "tool").strip()
    title_match = re.search(r"^\s*#\s+(.+)$", text, flags=re.MULTILINE)
    site_match = re.search(r"\*\*Site:\*\*\s*([^\n]+)", text)
    url_match = re.search(r"\*\*URL:\*\*\s*([^\n]+)", text)
    error_match = re.search(r"(Traceback[\s\S]{0,500}|[A-Za-z_]*Error:\s*[^\n]+|exit_code['\"]?\s*:\s*[1-9]\d*)", text)

    # Lead with the labeled fields so they survive the length cap.
    parts: list[str] = []
    if tool_name:
        parts.append(f"Tool outcome from {tool_name}.")
    if title_match:
        parts.append(f"Source/title: {title_match.group(1).strip()}.")
    if site_match:
        parts.append(f"Site: {site_match.group(1).strip()}.")
    if url_match and not _is_noisy_url(url_match.group(1).strip()):
        parts.append(f"URL: {url_match.group(1).strip()}.")
    if error_match:
        parts.append(f"Error signal: {error_match.group(1).strip()}.")

    # Fill the remaining slots (up to 6 parts) with content lines, skipping boilerplate,
    # link lists, noisy URLs and JSON objects.
    for line in _line_candidates(text):
        lowered = line.lower()
        if lowered.startswith(RAW_TOOL_NOISE_MARKERS):
            continue
        if lowered.startswith(("site:**", "url:**", "**site", "**url")):
            continue
        if lowered.count("http://") + lowered.count("https://") > 1:
            continue
        if any(_is_noisy_url(url) for url in re.findall(r"https?://[^\s)>\]\"']+", line, flags=re.IGNORECASE)):
            continue
        if re.match(r"^\{.*\}$", line):
            continue
        parts.append(line[:260])
        if len(parts) >= 6:
            break

    # Join into one line and clamp to the character budget.
    observation = " ".join(part for part in parts if part).strip()
    if len(observation) > max_chars:
        observation = observation[:max_chars].rstrip() + "..."
    return observation


# Choose the best compact text representation for one compression-prompt entry.
def _compression_prompt_entry_text(entry: dict[str, Any]) -> str:
    role = str(entry.get("role") or "").lower()
    if role == "tool":
        observation = _tool_observation_text(entry)
        return f"tool_observation: {observation}" if observation else ""
    return _entry_memory_text(entry, max_chars=1000)


# Keep the first and last entries around the compressed span for model context.
def _scoped_context_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(entries) <= COMPRESSION_FIRST_CONTEXT_ENTRIES + COMPRESSION_LAST_CONTEXT_ENTRIES:
        return list(entries)
    first = entries[:COMPRESSION_FIRST_CONTEXT_ENTRIES]
    last = entries[-COMPRESSION_LAST_CONTEXT_ENTRIES:]
    return [*first, *last]


# Detect text that still looks like an unprocessed tool or search dump.
def _looks_like_raw_tool_dump(text: str) -> bool:
    lowered = str(text or "").lower()
    if any(marker in lowered for marker in RAW_TOOL_NOISE_MARKERS):
        return True
    if lowered.startswith(("**site:**", "**url:**", "site:**", "url:**")):
        return True
    if lowered.count("http://") + lowered.count("https://") >= 2:
        return True
    if any(_is_noisy_url(url) for url in re.findall(r"https?://[^\s)>\]\"']+", lowered, flags=re.IGNORECASE)):
        return True
    if len(lowered) > 600 and ("**site:**" in lowered or "**url:**" in lowered):
        return True
    return False


# Normalize, filter, and dedupe semantic list fields on a summary payload.
def _sanitize_semantic_items(
    values: list[Any],
    *,
    limit: int,
    max_chars: int,
    allow_tool_memory: bool = False,
    apply_semantic_threshold: bool = False,
) -> list[str]:
    cleaned: list[str] = []
    for value in values:
        text = _clean_memory_text(str(value or ""))
        if not text:
            continue
        if text.strip().lower() in {"assistant:", "assistant: []"}:
            continue
        if not allow_tool_memory and _looks_like_raw_tool_dump(text):
            continue
        if _is_assistant_navigation(text):
            continue
        if apply_semantic_threshold and not _passes_semantic_threshold(text):
            continue
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        cleaned.append(text)
    return _dedupe_strings(cleaned, limit=limit)


# Sanitize open_tasks with stricter semantic thresholds than generic lists.
def _sanitize_open_tasks(values: list[Any], *, limit: int = 12) -> list[str]:
    tasks: list[str] = []
    for value in values:
        text = _clean_memory_text(str(value or ""))
        if not text:
            continue
        if not _passes_semantic_threshold(text):
            continue
        if len(text) > ANALYTIC_ENTRY_MAX_CHARS:
            text = text[:ANALYTIC_ENTRY_MAX_CHARS].rstrip() + "..."
        tasks.append(text)
    return _dedupe_strings(tasks, limit=limit)


# Deduplicate strings case-insensitively while preserving first-seen order.
def _dedupe_strings(values: list[str], *, limit: int = 64) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = re.sub(r"\s+", " ", str(value or "")).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(text)
        if len(result) >= limit:
            break
    return result


# Extract non-trivial lines suitable for fact or observation harvesting.
def _line_candidates(text: str) -> list[str]:
    raw = _strip_control_tokens(text)
    candidates: list[str] = []
    for line in raw.splitlines():
        cleaned = re.sub(r"^[\s>*#\-\d.)•]+", "", line).strip()
        if len(cleaned) < 18:
            continue
        if cleaned.lower().startswith(RAW_TOOL_NOISE_MARKERS):
            continue
        candidates.append(cleaned)
    return candidates


# Summary payload

# Return the canonical empty structured summary object.
def _empty_summary_payload() -> dict[str, Any]:
    return {
        "summary_version": 1,
        "session_goal": "",
        "current_focus": "",
        "work_summary": "",
        "reflection_summary": "",
        "recent_user_messages": [],
        "key_facts": [],
        "artifacts": {"files": [], "urls": [], "tools_used": []},
        "open_tasks": [],
        "risk_flags": [],
        "source_memory": [],
    }


# Build a fallback summary payload by scanning overflow entries without a model.
def _raw_context_payload(
    overflow_entries: list[dict[str, Any]],
    recent_user_messages: list[str],
    *,
    warning: str = "",
    raw_model_output: str = "",
) -> dict[str, Any]:
    # Collected artifacts and memory lines.
    urls: list[str] = []
    files: list[str] = []
    tools_used: list[str] = []
    source_memory: list[str] = []

    # Patterns for URLs, and for Windows absolute paths or any name with a short extension.
    url_pattern = re.compile(r"https?://[^\s)>\]\"']+", flags=re.IGNORECASE)
    file_pattern = re.compile(r"(?:[A-Za-z]:\\[^\n\r\t*?\"<>|]+|[\w./\\-]+\.[A-Za-z0-9]{1,12})")

    # Harvest URLs, paths, tool names, and memory lines from each overflow turn.
    for entry in overflow_entries:
        # Assistant thinking is not harvested; only content is.
        role = str(entry.get("role") or "").strip().lower()
        content = _strip_control_tokens(str(entry.get("content") or ""))
        thinking = "" if role == "assistant" else _strip_control_tokens(str(entry.get("thinking") or ""))
        text = "\n".join(part for part in (content, thinking) if part).strip()
        if not text:
            continue

        # Keep non-noisy URLs and plausible file paths (web host paths and code fragments are rejected).
        urls.extend(url for url in (_clean_url(url) for url in url_pattern.findall(text)) if url)
        files.extend(
            file_name
            for file_name in (match.group(0).strip(".,;:`'\"") for match in file_pattern.finditer(text))
            if file_name
            and not _looks_like_web_host_path(file_name)
            and _looks_like_valid_path(file_name)
        )

        # Tool entries only contribute their tool name; every other role contributes a sanitized memory line.
        if role == "tool":
            tool_name = str(entry.get("tool_name") or entry.get("name") or entry.get("alias") or entry.get("tool_id") or "").strip()
            if tool_name:
                tools_used.append(tool_name)

        if role != "tool":
            memory_text = _entry_memory_text(entry, max_chars=MEMORY_ENTRY_MAX_CHARS)
            if memory_text and memory_text.strip().lower() not in {"assistant:", "user:"}:
                source_memory.append(memory_text)

    # The latest non-empty user message becomes the goal and focus; a warning becomes a risk flag.
    latest_user = next((msg for msg in recent_user_messages if str(msg or "").strip()), "")
    risk_flags = [warning] if warning else []

    # Keep the unparsed model output (capped at 6000 chars) as the work summary so no information is lost.
    cleaned_raw_output = _clean_memory_text(raw_model_output)
    if len(cleaned_raw_output) > 6000:
        cleaned_raw_output = cleaned_raw_output[:6000].rstrip() + "..."
    work_summary = f"{{{cleaned_raw_output}}}" if cleaned_raw_output else "Raw compressed context was preserved without semantic extraction."

    # Assemble the payload with per-field size limits.
    return {
        "summary_version": 1,
        "session_goal": _clean_memory_text(latest_user)[:900],
        "current_focus": _clean_memory_text(latest_user)[:900],
        "work_summary": work_summary,
        "reflection_summary": "",
        "recent_user_messages": _dedupe_strings(recent_user_messages[:12], limit=12),
        "key_facts": [],
        "artifacts": {
            "files": _dedupe_strings(files, limit=32),
            "urls": _dedupe_strings(urls, limit=24),
            "tools_used": _dedupe_strings(tools_used, limit=32),
        },
        "open_tasks": [],
        "risk_flags": _sanitize_semantic_items(risk_flags, limit=4, max_chars=ANALYTIC_ENTRY_MAX_CHARS),
        "source_memory": _sanitize_semantic_items(source_memory, limit=96, max_chars=MEMORY_ENTRY_MAX_CHARS, allow_tool_memory=True),
    }


# Clamp and sanitize every field on a model-produced summary payload.
def _sanitize_summary_payload(model_payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(model_payload) if isinstance(model_payload, dict) else {}
    merged["summary_version"] = 1
    merged.pop("history_highlights", None)

    # Normalize long narrative fields first.
    work_summary = _clean_memory_text(str(merged.get("work_summary") or ""))
    if len(work_summary) > 6000:
        work_summary = work_summary[:6000].rstrip() + "..."
    merged["work_summary"] = work_summary

    reflection_summary = _clean_memory_text(str(merged.get("reflection_summary") or ""))
    if len(reflection_summary) > 2000:
        reflection_summary = reflection_summary[:2000].rstrip() + "..."
    merged["reflection_summary"] = reflection_summary

    for key, max_chars in (("session_goal", 900), ("current_focus", 900)):
        text = _clean_memory_text(str(merged.get(key) or ""))
        if len(text) > max_chars:
            text = text[:max_chars].rstrip() + "..."
        merged[key] = text

    # Sanitize list-shaped semantic fields with per-field limits.
    semantic_limits = {
        "recent_user_messages": (12, 900, False),
        "key_facts": (48, 420, False),
        "open_tasks": (12, ANALYTIC_ENTRY_MAX_CHARS, False),
        "risk_flags": (12, ANALYTIC_ENTRY_MAX_CHARS, False),
        "source_memory": (96, MEMORY_ENTRY_MAX_CHARS, True),
    }
    for key, (limit, max_chars, allow_tool_memory) in semantic_limits.items():
        values = merged.get(key) if isinstance(merged.get(key), list) else []
        if key == "open_tasks":
            merged[key] = _sanitize_open_tasks(values, limit=limit)
            continue
        merged[key] = _sanitize_semantic_items(
            values,
            limit=limit,
            max_chars=max_chars,
            allow_tool_memory=allow_tool_memory,
            apply_semantic_threshold=(key == "key_facts"),
        )

    # Keep only plausible file paths and non-noisy URLs, deduplicated, in the artifacts block.
    artifacts = merged.get("artifacts") if isinstance(merged.get("artifacts"), dict) else {}
    merged["artifacts"] = {
        "files": _dedupe_strings(
            [
                str(file_name)
                for file_name in (artifacts.get("files") if isinstance(artifacts.get("files"), list) else [])
                if not _looks_like_web_host_path(str(file_name)) and _looks_like_valid_path(str(file_name))
            ],
            limit=32,
        ),
        "urls": _dedupe_strings([url for url in (_clean_url(str(url)) for url in (artifacts.get("urls") if isinstance(artifacts.get("urls"), list) else [])) if url], limit=24),
        "tools_used": _dedupe_strings(artifacts.get("tools_used") if isinstance(artifacts.get("tools_used"), list) else [], limit=32),
    }
    return merged


# Merge model summary fields with deterministic raw-context harvest results.
def _merge_model_summary_with_raw_context(model_payload: dict[str, Any], raw_payload: dict[str, Any]) -> dict[str, Any]:
    merged = _sanitize_summary_payload(model_payload)

    # Append the harvested user messages and memory lines to the model's lists, then re-sanitize and re-cap them.
    for key in ("recent_user_messages", "source_memory"):
        max_chars = 900 if key == "recent_user_messages" else MEMORY_ENTRY_MAX_CHARS
        limit = 12 if key == "recent_user_messages" else 96
        values = merged.get(key) if isinstance(merged.get(key), list) else []
        raw_values = raw_payload.get(key) if isinstance(raw_payload.get(key), list) else []
        merged[key] = _sanitize_semantic_items(
            [*values, *raw_values],
            limit=limit,
            max_chars=max_chars,
            allow_tool_memory=(key == "source_memory"),
        )

    # Union the model's artifacts with the harvested ones, filtered and deduplicated the same way.
    artifacts = merged.get("artifacts") if isinstance(merged.get("artifacts"), dict) else {}
    raw_artifacts = raw_payload.get("artifacts") if isinstance(raw_payload.get("artifacts"), dict) else {}
    merged["artifacts"] = {
        "files": _dedupe_strings(
            [
                str(file_name)
                for file_name in [
                    *(artifacts.get("files") if isinstance(artifacts.get("files"), list) else []),
                    *(raw_artifacts.get("files") or []),
                ]
                if not _looks_like_web_host_path(str(file_name)) and _looks_like_valid_path(str(file_name))
            ],
            limit=32,
        ),
        "urls": _dedupe_strings([url for url in (_clean_url(str(url)) for url in [*(artifacts.get("urls") if isinstance(artifacts.get("urls"), list) else []), *(raw_artifacts.get("urls") or [])]) if url], limit=24),
        "tools_used": _dedupe_strings([*(artifacts.get("tools_used") if isinstance(artifacts.get("tools_used"), list) else []), *(raw_artifacts.get("tools_used") or [])], limit=32),
    }
    return merged


# Serialize a summary payload into the stored compression marker text.
def _summary_text_from_payload(payload: dict[str, Any]) -> str:
    return "[Conversation History Summary Base]\n" + json.dumps(payload, ensure_ascii=False, indent=2)


# Fit a summary into a character budget while preserving valid JSON.
def fit_summary_text(summary_payload: dict[str, Any], max_chars: int) -> tuple[str, dict[str, Any]]:
    # A missing or non-positive budget means no limit: return the payload as it is.
    try:
        budget = int(max_chars)
    except (TypeError, ValueError):
        budget = 0
    if budget <= 0:
        return _summary_text_from_payload(summary_payload), summary_payload

    # Work on a deep copy so the caller's payload is never mutated; a payload that already fits is returned as is.
    fitted = json.loads(json.dumps(summary_payload, ensure_ascii=False))
    if not isinstance(fitted, dict):
        fitted = {}

    if len(_summary_text_from_payload(fitted)) <= budget:
        return _summary_text_from_payload(fitted), fitted

    # Record in the payload that content may have been dropped.
    risk_flags = fitted.get("risk_flags") if isinstance(fitted.get("risk_flags"), list) else []
    risk_flags.append("Compression summary was size-fitted; low-priority excerpts may be omitted.")
    fitted["risk_flags"] = risk_flags

    # Drop or truncate lower-priority fields until the serialized summary fits: narrative fields are cut to
    # a fixed length, list fields lose trailing items down to 8 entries.
    shrink_order = [
        "reflection_summary",
        "work_summary",
        "source_memory",
        "key_facts",
        "open_tasks",
        "recent_user_messages",
    ]
    for key in shrink_order:
        if key == "work_summary":
            text = str(fitted.get("work_summary") or "")
            if len(text) > 2400 and len(_summary_text_from_payload(fitted)) > budget:
                fitted["work_summary"] = text[:2400].rstrip() + "..."
            continue
        if key == "reflection_summary":
            text = str(fitted.get("reflection_summary") or "")
            if len(text) > 1200 and len(_summary_text_from_payload(fitted)) > budget:
                fitted["reflection_summary"] = text[:1200].rstrip() + "..."
            continue
        values = fitted.get(key)
        if not isinstance(values, list):
            continue
        while len(values) > 8 and len(_summary_text_from_payload(fitted)) > budget:
            values.pop()

    # Still too large: trim the artifact lists the same way.
    artifacts = fitted.get("artifacts") if isinstance(fitted.get("artifacts"), dict) else {}
    for key in ("urls", "files", "tools_used"):
        values = artifacts.get(key)
        if not isinstance(values, list):
            continue
        while len(values) > 8 and len(_summary_text_from_payload(fitted)) > budget:
            values.pop()

    # Still too large: truncate individual long memory and fact entries.
    for key in ("source_memory", "key_facts"):
        values = fitted.get(key)
        if not isinstance(values, list):
            continue
        for index, value in enumerate(list(values)):
            if len(_summary_text_from_payload(fitted)) <= budget:
                break
            text = str(value)
            if len(text) > 700:
                values[index] = text[:700].rstrip() + "..."

    # Last resort: drop source_memory entries one by one until the text fits or none are left.
    while len(_summary_text_from_payload(fitted)) > budget:
        values = fitted.get("source_memory")
        if isinstance(values, list) and values:
            values.pop()
            continue
        break

    return _summary_text_from_payload(fitted), fitted


# Model output parsing

# Parse a JSON object from raw model text, including fenced or prose-wrapped payloads.
def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = str(text or "").strip()
    if not raw:
        return None

    # Unwrap a fenced code block first, then try the whole text as JSON.
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw, flags=re.IGNORECASE)
    if fence_match:
        raw = fence_match.group(1).strip()
    try:
        payload = json.loads(raw)
        return payload if isinstance(payload, dict) else None
    except ValueError:
        # Not pure JSON; fall through to extracting the outermost braces below.
        pass

    # Retry on the span from the first "{" to the last "}" to cope with prose around the object.
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        try:
            payload = json.loads(raw[start:end + 1])
            return payload if isinstance(payload, dict) else None
        except ValueError:
            return None
    return None


# Return True when a Markdown field value is effectively empty.
def _empty_markdown_value(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return normalized in {"", "none", "n/a", "na", "-", "(none)", "empty"}


# Parse bullet list lines from a Markdown section body.
def _markdown_list_items(lines: list[str]) -> list[str]:
    items: list[str] = []
    for line in lines:
        text = re.sub(r"^\s*(?:[-*+]|\d+[.)])\s*", "", str(line or "")).strip()
        if _empty_markdown_value(text):
            continue
        items.append(text)
    return items


# Join non-empty Markdown section lines into a single block string.
def _markdown_block(lines: list[str]) -> str:
    cleaned = [str(line or "").strip() for line in lines]
    text = "\n".join(line for line in cleaned if not _empty_markdown_value(line)).strip()
    return text


# Normalize a Markdown heading or label for canonical section lookup.
def _normalize_markdown_heading(heading: str) -> str:
    text = str(heading or "").strip().lower().strip("*_`:")
    text = re.sub(r"[_-]+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# Canonical Markdown section titles for model compression output.
MARKDOWN_SECTION_HEADINGS = {
    "session_goal": "Session Goal",
    "current_focus": "Current Focus",
    "work_summary": "Work Summary",
    "reflection_summary": "Reflection Summary",
    "recent_user_messages": "Recent User Messages",
    "key_facts": "Key Facts",
    "files": "Files",
    "urls": "URLs",
    "tools_used": "Tools Used",
    "open_tasks": "Open Tasks",
    "risk_flags": "Risk Flags",
    "source_memory": "Source Memory",
}


# Map normalized headings and aliases back to internal summary field names (filled once at import time).
NORMALIZED_MARKDOWN_SECTIONS = {}
for field_name, heading in MARKDOWN_SECTION_HEADINGS.items():
    NORMALIZED_MARKDOWN_SECTIONS[_normalize_markdown_heading(field_name)] = field_name
    NORMALIZED_MARKDOWN_SECTIONS[_normalize_markdown_heading(heading)] = field_name


# Resolve a heading string to an internal summary field name, if recognized.
def _canonical_markdown_section(heading: str) -> str:
    return NORMALIZED_MARKDOWN_SECTIONS.get(_normalize_markdown_heading(heading), "")


# Parse fixed-section Markdown model output into a summary payload dict.
def _extract_markdown_summary(text: str) -> dict[str, Any] | None:
    # Split the text into sections keyed by field name. A "#" heading or a "Label:" line starts a section;
    # every other line belongs to the current section. Lines before the first recognized heading are ignored.
    sections: dict[str, list[str]] = {}
    current = ""
    for raw_line in str(text or "").splitlines():
        heading_match = re.match(r"^\s{0,3}#{1,4}\s+(.+?)\s*$", raw_line)
        if heading_match:
            current = _canonical_markdown_section(heading_match.group(1))
            if current:
                sections.setdefault(current, [])
            continue
        label_match = re.match(r"^\s*(?:[-*+]\s*)?(?:\*\*)?([A-Za-z0-9_ ][A-Za-z0-9_ /-]{1,80}?)(?::\*\*|\*\*:|:)\s*(.*)$", raw_line)
        if label_match:
            candidate = _canonical_markdown_section(label_match.group(1))
            if candidate:
                current = candidate
                sections.setdefault(current, [])
                inline_value = label_match.group(2).strip()
                if inline_value:
                    sections[current].append(inline_value)
                continue
        if current:
            sections.setdefault(current, []).append(raw_line)

    if not sections:
        return None

    # Return the collected lines of one section, or an empty list when the model omitted it.
    def section(field_name: str) -> list[str]:
        return sections.get(field_name, [])

    # Fill the payload: narrative fields become text blocks, list fields become bullet items.
    payload = _empty_summary_payload()
    payload["session_goal"] = _markdown_block(section("session_goal"))
    payload["current_focus"] = _markdown_block(section("current_focus"))
    payload["work_summary"] = _markdown_block(section("work_summary"))
    payload["reflection_summary"] = _markdown_block(section("reflection_summary"))
    payload["recent_user_messages"] = _markdown_list_items(section("recent_user_messages"))
    payload["key_facts"] = _markdown_list_items(section("key_facts"))
    payload["open_tasks"] = _markdown_list_items(section("open_tasks"))
    payload["risk_flags"] = _markdown_list_items(section("risk_flags"))
    payload["source_memory"] = _markdown_list_items(section("source_memory"))
    payload["artifacts"] = {
        "files": _markdown_list_items(section("files")),
        "urls": _markdown_list_items(section("urls")),
        "tools_used": _markdown_list_items(section("tools_used")),
    }

    # Reject output that parsed into sections but carries no content at all.
    has_semantic_content = any(
        payload.get(key)
        for key in ("session_goal", "current_focus", "work_summary", "reflection_summary", "recent_user_messages", "key_facts", "open_tasks", "risk_flags", "source_memory")
    )
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    has_artifacts = any(artifacts.get(key) for key in ("files", "urls", "tools_used"))
    return payload if has_semantic_content or has_artifacts else None


# Parse model output as JSON first, then as contract Markdown sections.
def _extract_summary_payload(text: str) -> dict[str, Any] | None:
    return _extract_json_object(text) or _extract_markdown_summary(text)


# Summary construction

# Build the structured history compression block and parsed payload metadata.
def build_structured_history_summary(
    *,
    overflow_entries: list[dict[str, Any]],
    recent_user_messages: list[str],
    direct_user_directives: list[str],
    summarize_with_model: Callable[[list[dict[str, str]]], str] | None,
    max_overflow_entries: int = 40,
) -> tuple[str, dict[str, Any]]:
    # Limit the span to the most recent overflow entries and render the first and last ones
    # as the model transcript.
    scoped_full = overflow_entries[-max(1, max_overflow_entries):]
    scoped = _scoped_context_entries(scoped_full)
    transcript_lines = []
    for idx, entry in enumerate(scoped, 1):
        entry_text = _compression_prompt_entry_text(entry)
        if entry_text:
            transcript_lines.append(f"[{idx}] {entry_text}")
    transcript = "\n".join(transcript_lines)

    # The raw harvest is always computed: it backs up and enriches whatever the model returns.
    summary_payload: dict[str, Any] | None = None
    raw_payload = _raw_context_payload(scoped_full, recent_user_messages)
    model_text = ""

    # Ask the model for contract Markdown when a callback is available.
    if summarize_with_model is not None and transcript.strip():
        # System message sets the rules; the user message lists the required headings, field rules and the context.
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You compress chat history into a fixed-section Markdown memory base.\n"
                    "Return only the requested Markdown sections. Do not wrap the output in code fences.\n"
                    "Preserve concrete facts, paths, URLs, explicit user directives, tool outcomes, and open tasks.\n"
                    "Do not include apologies or filler.\n"
                    "Never copy raw tool calls, terminal logs, JSON tool payloads, diffs, tracebacks, citation boilerplate, or search-result lists into the output.\n"
                    "Instead, infer concise outcomes and durable facts from tool observations."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Build a structured history base using exactly these Markdown headings, in this order:\n"
                    "## Session Goal\n"
                    "## Current Focus\n"
                    "## Work Summary\n"
                    "## Reflection Summary\n"
                    "## Recent User Messages\n"
                    "## Key Facts\n"
                    "## Files\n"
                    "## URLs\n"
                    "## Tools Used\n"
                    "## Open Tasks\n"
                    "## Risk Flags\n"
                    "## Source Memory\n\n"
                    "For list sections, use one '- ' bullet per item. For empty sections, write '- None'.\n\n"
                    "IMPORTANT:\n"
                    "- This is a large memory base, not a tiny summary.\n"
                    "- Be exhaustive about durable session state: user goal, constraints, completed actions, files changed, errors fixed, current answer facts, and remaining work.\n"
                    "- work_summary must be a detailed narrative block (multi-sentence, chronological) describing what happened in the dialogue, which tools were used, what failed, what was fixed, and what remains.\n"
                    "- reflection_summary must be a concise analytical reflection: what is reliable, what is uncertain, key risks, and the best next-step strategy.\n"
                    "- Fill each field by semantics, not by keyword matching.\n"
                    "- Treat assistant reasoning/thinking as unreliable scratchpad; do not copy it into open_tasks, key_facts, or risk_flags unless the final visible answer or user explicitly confirms it.\n"
                    "- key_facts must contain task parameters and extracted domain facts only; never copy user messages verbatim into key_facts.\n"
                    "- open_tasks must contain model-generated next actions, not copied reasoning. If the latest user goal is not explicitly confirmed complete by the user, write at least one concrete verification or continuation task.\n"
                    "- Leave open_tasks empty only when the user explicitly confirmed the goal is complete or there is truly no actionable next step.\n"
                    "- Never put partial thought fragments in open_tasks (examples: \"We need...\", \"Now combine...\", \"Wait...\", \"Check balancing...\").\n"
                    "- risk_flags must contain only warnings that affect future work; leave it empty if none exist.\n"
                    "- Do not put raw search snippets, citation instructions, terminal logs, tool JSON, diffs, or page dumps into any field.\n"
                    "- source_memory must preserve curated detailed excerpts from important user/assistant turns only; never store entries beginning with tool:.\n"
                    "- For assistant entries in source_memory, summarize actions in square brackets (example: assistant: [Applied edit and fixed import]); do not quote long assistant outputs.\n"
                    "- Do not duplicate the same long text twice in different fields.\n"
                    "- Include direct user instructions verbatim when possible.\n"
                    "- Keep file paths and URLs exact.\n"
                    "- recent_user_messages must include no more than the latest 5 user turns.\n"
                    "- Raw context below contains at most the first 3 and last 5 compressed-span entries; infer the rest from directives, artifacts, and tool observations.\n\n"
                    f"Direct user directives:\n{json.dumps(direct_user_directives[:24], ensure_ascii=False)}\n\n"
                    f"Recent user messages:\n{json.dumps(recent_user_messages[:5], ensure_ascii=False)}\n\n"
                    f"Compressed-span context:\n{transcript}"
                ),
            },
        ]

        # Run the model and parse its reply (JSON or Markdown sections).
        model_text = summarize_with_model(prompt_messages)
        summary_payload = _extract_summary_payload(model_text)

    # Fall back to raw harvest when the model output cannot be parsed.
    if not isinstance(summary_payload, dict):
        # Warn only when a model was actually asked and gave unusable output.
        warning = ""
        if summarize_with_model is not None and transcript.strip():
            if _clean_memory_text(model_text):
                warning = "Model summary output could not be parsed; raw compressed context was preserved."
            else:
                warning = "Model returned empty summary output; raw compressed context was preserved."
        summary_payload = _raw_context_payload(scoped_full, recent_user_messages, warning=warning, raw_model_output=model_text)
    else:
        # Parsed model output is sanitized and merged with the raw harvest.
        summary_payload = _merge_model_summary_with_raw_context(summary_payload, raw_payload)

    summary_payload.setdefault("summary_version", 1)
    summary_text = _summary_text_from_payload(summary_payload)
    return summary_text, summary_payload
