# Copyright NEXTGGTECH. Elastic License 2.0.

"""Optional LLM query rewriter: turn conversational model "thoughts" into search queries.

Models routinely write conversational requests into the search field ("where can I
see how people make those charts") instead of effective query terms. When enabled,
each incoming query string is treated as raw intent and rewritten by a subagent call:
a local model with a CLEAN context — only the machine-format instruction — returns one
small JSON object with plain search terms.

The subagent always follows the active chat model: engine + model come from
Tools/model_runtime_metadata.json (written by the chat on model selection) and the
endpoint is resolved by the module's shared settings reader (including environment
overrides and defaults).

Design constraints:
- Fail-open: any error, timeout or malformed reply returns the input unchanged. The
  rewriter must never break a search that would have worked without it.
- Heuristic gate: only conversational-looking strings pay the rewrite latency; a clean
  3-word query passes through untouched.
- TTL cache: identical thoughts are rewritten once.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import time
from pathlib import Path
from functools import lru_cache
from typing import Any, NamedTuple

import httpx

from core.config import load_search_config

logger = logging.getLogger("mcp.web_search.rewriter")

REWRITER_SYSTEM_PROMPT = """You convert a researcher's raw thoughts into web-search queries.
The thought may be rambling, conversational, or mixed-language. Extract the actual
information need and return 1-3 effective search queries.
Rules: plain search terms only (no questions, no explanations), each under 140 chars,
prefer the thought's language but add an English variant when it likely helps,
pick the vertical honestly.
Output ONLY one JSON object, nothing else:
{"queries": ["...", "..."], "vertical": "web|academic|shopping", "language": "ru|en|..."}"""

_CACHE_TTL_SECONDS = 3600.0
_cache: dict[tuple[RewriterTarget, str], tuple[list[str], float]] = {}
_cache_lock = asyncio.Lock()
_warned_no_target = False

_TOOL_ROOT = Path(__file__).resolve().parents[2]  # .../Tools/mcp-web-search
_RUNTIME_METADATA_FILE = _TOOL_ROOT.parent / "model_runtime_metadata.json"


@lru_cache(maxsize=1)
def _project_settings():
    """Load the shared settings reader even in the isolated tool environment."""
    path = _TOOL_ROOT.parents[1] / "Settings" / "settings.py"
    spec = importlib.util.spec_from_file_location("aslm_rewriter_settings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_engine_settings() -> dict[str, Any]:
    settings = _project_settings()
    values = dict(settings.load_settings())
    # Stored addresses omit schemes; the shared resolver restores HTTP/HTTPS.
    for engine in ("lms", "openai", "google-genai"):
        values[f"{engine.replace('-', '_')}_url"] = settings.get_engine_url(engine)
    return values


class RewriterConfigSnapshot:
    """Cheap per-call view of the rewriter config (loaded fresh each call)."""

    __slots__ = ("enabled", "timeout_seconds", "max_queries", "mode")

    def __init__(self) -> None:
        section = load_search_config().query_rewriter
        self.enabled = bool(section.enabled)
        self.timeout_seconds = max(5.0, float(section.timeout_seconds))
        self.max_queries = max(1, int(section.max_queries))
        self.mode = section.mode


class RewriterTarget(NamedTuple):
    """One resolved subagent endpoint: how and where to call the model."""

    flavor: str  # "ollama" native /api/chat or "openai" /chat/completions
    url: str
    model: str
    api_key: str


def looks_conversational(query: str) -> bool:
    """Heuristic: does this string read like a thought rather than a search query?"""
    text = str(query or "").strip()
    words = text.split()
    if len(words) >= 6:
        return True
    if "?" in text:
        return True
    lowered = text.casefold()
    return any(
        marker in lowered
        for marker in (
            "хочу ", "найди ", "где ", "как ", "помню ", "кажется", "вроде",
            "i need", "i want", "find me", "i saw", "i remember", "looking for",
            "what is", "how do ", "how to ",
        )
    ) and len(words) >= 3


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _http_base(raw: Any) -> str:
    base = str(raw or "").strip().rstrip("/")
    if not base:
        return ""
    if "://" not in base:
        base = "http://" + base
    return base


def _chat_completions_url(base: str) -> str:
    return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")


def _runtime_target() -> RewriterTarget | None:
    """Resolve the active chat model into a callable endpoint. None when unknown."""
    try:
        metadata = _read_json(_RUNTIME_METADATA_FILE)
        active = metadata.get("active") if isinstance(metadata, dict) else None
    except (OSError, ValueError):
        return None
    if not isinstance(active, dict):
        return None
    engine = str(active.get("engine") or "").strip()
    model = str(active.get("model") or "").strip()
    if not engine or not model:
        return None
    try:
        engine_settings = _load_engine_settings()
    except (OSError, ValueError):
        engine_settings = {}

    if engine == "ollama-service":
        try:
            port = int(engine_settings.get("ollama-service_port") or 20003)
        except (TypeError, ValueError):
            port = 20003
        return RewriterTarget("ollama", f"http://127.0.0.1:{port}/api/chat", model, "")
    if engine == "lms":
        base = _http_base(engine_settings.get("lms_url") or "127.0.0.1:1234")
        return RewriterTarget("openai", _chat_completions_url(base), model, "")
    if engine == "openai":
        base = _http_base(engine_settings.get("openai_url"))
        if not base:
            return None
        api_key = str(engine_settings.get("openai_api_key") or "").strip()
        return RewriterTarget("openai", _chat_completions_url(base), model, api_key)
    if engine == "google-genai":
        api_key = str(engine_settings.get("google_genai_api_key") or "").strip()
        if not api_key:
            return None
        base = str(engine_settings.get("google_genai_url") or "https://generativelanguage.googleapis.com").strip().rstrip("/")
        if "://" not in base:
            base = "https://" + base
        if base.endswith("/v1beta") or base.endswith("/v1"):
            base = base.rsplit("/", 1)[0]
        return RewriterTarget(
            "openai",
            f"{base}/v1beta/openai/chat/completions",
            model,
            api_key,
        )
    return None


def _parse_model_reply(content: Any) -> dict[str, Any]:
    """Extract the JSON object from a model reply that may wrap it in prose or fences."""
    text = str(content or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("model reply contains no JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("model reply JSON is not an object")
    return parsed


async def _call_model(target: RewriterTarget, timeout_seconds: float, thought: str) -> list[str]:
    payload: dict[str, Any] = {
        "model": target.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": REWRITER_SYSTEM_PROMPT},
            {"role": "user", "content": thought},
        ],
    }
    headers = {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}
    async with httpx.AsyncClient(trust_env=False, timeout=timeout_seconds, headers=headers) as client:
        response = await client.post(target.url, json=payload)
        response.raise_for_status()
        data = response.json()
    if target.flavor == "ollama":
        content = ((data.get("message") or {}).get("content")) if isinstance(data, dict) else None
    else:
        choices = data.get("choices") if isinstance(data, dict) else None
        message = (choices[0].get("message") if choices else None) or {}
        content = message.get("content")
    parsed = _parse_model_reply(content)
    raw_queries = parsed.get("queries")
    if not isinstance(raw_queries, list):
        raise ValueError("model reply queries must be an array")
    queries = [
        " ".join(str(q).split())
        for q in raw_queries
        if isinstance(q, str) and q.strip()
    ]
    return queries


async def rewrite_query(query: str) -> tuple[str, bool]:
    """Rewrite one thought; returns (query_to_search, rewritten_flag). Fail-open."""
    global _warned_no_target
    config = RewriterConfigSnapshot()
    text = " ".join(str(query or "").split())
    if not config.enabled or not text:
        return text or query, False
    if config.mode == "heuristic" and not looks_conversational(text):
        return text, False

    target = _runtime_target()
    if target is None:
        if not _warned_no_target:
            _warned_no_target = True
            logger.warning(
                "query rewriter enabled but no active runtime model in %s; "
                "queries pass through unchanged",
                _RUNTIME_METADATA_FILE.name,
            )
        return text, False

    cache_key = (target, text)
    async with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and cached[1] > time.monotonic():
            return cached[0][0], True

    started = time.perf_counter()
    try:
        rewritten = (await _call_model(target, config.timeout_seconds, text))[: config.max_queries]
    except Exception as exc:  # noqa: BLE001 — fail-open is the contract
        logger.warning(
            "query rewriter failed (%s: %s) — passing the original query through",
            type(exc).__name__, str(exc)[:120],
        )
        return text, False
    if not rewritten:
        return text, False

    async with _cache_lock:
        _cache[cache_key] = (rewritten, time.monotonic() + _CACHE_TTL_SECONDS)
    logger.info(
        "query rewriter: %r -> %r via %s/%s in %.1fs",
        text[:80], rewritten, target.flavor, target.model,
        time.perf_counter() - started,
    )
    # The first clean variant replaces the thought; multi-variant fan-out for a
    # single incoming query is handled by expanded_queries() at the call site.
    return rewritten[0], True


async def rewrite_queries(queries: list[str]) -> tuple[list[str], bool]:
    """Rewrite a whole query list concurrently; returns (queries, any_rewritten)."""
    if not queries:
        return queries, False
    results = await asyncio.gather(*(rewrite_query(q) for q in queries))
    return [item[0] for item in results], any(item[1] for item in results)


async def expanded_queries(query: str) -> list[str] | None:
    """Return rewritten variants, including a single variant, or None on skip/failure."""
    global _warned_no_target
    config = RewriterConfigSnapshot()
    text = " ".join(str(query or "").split())
    if not config.enabled or not text:
        return None
    if config.mode == "heuristic" and not looks_conversational(text):
        return None
    target = _runtime_target()
    if target is None:
        if not _warned_no_target:
            _warned_no_target = True
            logger.warning(
                "query rewriter enabled but no model to call — queries pass through unchanged",
            )
        return None
    cache_key = (target, text)
    async with _cache_lock:
        cached = _cache.get(cache_key)
        if cached and cached[1] > time.monotonic():
            return list(cached[0])
    try:
        rewritten = (await _call_model(target, config.timeout_seconds, text))[: config.max_queries]
    except Exception as exc:  # noqa: BLE001 — fail-open is the contract
        logger.warning(
            "query rewriter failed (%s: %s) — passing the original query through",
            type(exc).__name__, str(exc)[:120],
        )
        return None
    if not rewritten:
        return None
    async with _cache_lock:
        _cache[cache_key] = (rewritten, time.monotonic() + _CACHE_TTL_SECONDS)
    logger.info("query rewriter fan-out: %r -> %r", text[:80], rewritten)
    return rewritten
