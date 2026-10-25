# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import copy
import logging
import sys
import time
from pathlib import Path
from typing import Any

SERVER_ROOT = Path(__file__).resolve().parent
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

# Force UTF-8 on the process stdout/stderr. The ASLM tool worker re-execs this module
# in-process on every call, then prints the JSON result envelope with ensure_ascii=False.
# Search results routinely contain emoji/CJK from page content; if the worker's stdout
# codec is not UTF-8 (Windows default is cp125x), that print raises UnicodeEncodeError
# AFTER the tool already finished — the "finishes then crashes" failure. Reconfiguring
# here guarantees the envelope encodes regardless of the parent's environment.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 — older/odd streams without reconfigure: best-effort
        pass

from core.logging_setup import setup_logging
from core.mcp_contract import (
    IMAGE_SEARCH_TOOL_DESCRIPTION,
    MCP_SERVER_DESCRIPTION,
    READ_PAGE_TOOL_DESCRIPTION,
    build_search_description,
    build_search_schema,
    coerce_search_academic,
    coerce_search_effort,
    coerce_search_onion,
    coerce_search_queries,
    coerce_search_shopping,
    prepare_search_arguments,
)
from core.read import run_read_page
from core.search.web_search import run_web_search, run_web_search_batch, run_web_search_plan
from core.search_io_logger import write_search_io_event
from core.query.rewriter import expanded_queries, rewrite_queries
from urllib.parse import urlparse

# Wire rotating file logs (web_search.log / read_page.log / core.log / mcp_trace.log)
# and the model IO log before anything runs, exactly like the legacy adapter.
setup_logging()
logger = logging.getLogger("mcp.server")

_evicted = False


# Reclaim disk from expired cache entries once per process, on first tool call.
def _evict_caches_once() -> None:
    global _evicted
    if _evicted:
        return
    _evicted = True
    try:
        from core.cache import get_page_cache
        from core.cache.hosted_cache import get_hosted_cache

        get_hosted_cache().evict_expired()
        get_page_cache().evict_stale()
    except Exception as exc:  # noqa: BLE001 — housekeeping must never block a search
        logger.debug("cache eviction skipped: %s", exc)

MCP_SERVER = {
    "id": "web_search",
    "name": "Web Search",
    "description": MCP_SERVER_DESCRIPTION,
}

TOOLS = [
    {
        "id": "web_search",
        "name": "Web Search",
        "description": build_search_description(),
        # Built from config so the onion opt-in is advertised only while tor.enabled.
        "parameters": build_search_schema(),
    },
    {
        "id": "image_search",
        "name": "Image Search",
        "description": IMAGE_SEARCH_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "description": "Plain search terms for the images (the user's language, no questions).",
                    "type": "string",
                    "minLength": 2,
                },
                "limit": {
                    "description": "How many images to return (1-24, default 12).",
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 24,
                },
            },
            "required": ["query"],
        },
    },
    {
        "id": "read_page",
        "name": "Read Page",
        "description": READ_PAGE_TOOL_DESCRIPTION,
        "parameters": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "url": {
                    "description": (
                        "One exact non-empty URL copied from web_search results, or an array "
                        "of exact URLs for a batch read. Never use an object, search topic, "
                        "or empty string."
                    ),
                    "oneOf": [
                        {"type": "string", "minLength": 8, "format": "uri"},
                        {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 8, "format": "uri"},
                        },
                    ],
                },
            },
            "required": ["url"],
        },
    },
]

# Optional worker preflight. The API calls this before publishing the tool event so the
# frontend receives canonical search data without waiting for network I/O.
TOOL_PREPARERS = {"web_search": prepare_search_arguments}


# Return whether this server supports the given engine or model.
def supports(engine: str | None = None, model_name: str | None = None) -> bool:
    return engine in (None, "ollama-service", "lms", "openai", "google-genai")


# Attach rich cards (wikipedia/discussions) to a finished web_search result.
async def _attach_rich_cards(result: dict[str, Any], queries: list[str]) -> dict[str, Any]:
    try:
        from core.config import load_search_config
        config = load_search_config().rich_cards
        if not config.enabled:
            return result
        from core.cards import RichCardService

        sources = result.get("sources") or []
        language = ""
        for query in queries:
            if isinstance(query, str) and query.strip():
                from core.search.query_locales import analyze_query
                language = analyze_query(query).language
                break
        service = RichCardService()
        cards = await service.build(sources, config, language=language or "en")
        discussions = service.build_discussions(sources, config)
        ui = result.setdefault("ui", {})
        if cards:
            ui["rich_cards"] = cards
        if discussions:
            ui["discussions"] = discussions
        logger.info(
            "mcp.web_search.rich_cards cards=%d discussions=%d",
            len(cards), len(discussions.get("items", [])),
        )
    except Exception as exc:  # noqa: BLE001 — enrichment must never fail a search
        logger.warning("rich cards skipped: %s", exc)
    return result


# Rewrite raw model-supplied query text before the plan compiler runs.
async def _rewrite_search_args(args: dict[str, Any]) -> dict[str, Any]:
    """Return args with query text replaced by rewriter output when it fires.

    The model composes its own query strings, so the subagent must see them
    BEFORE prepare_search_arguments compiles the advanced schema — rewritten
    text then flows through the same operator extraction, tool_ui build, and
    execution pipeline in every schema mode. Fail-open: any non-string shape
    or disabled config returns the args untouched.
    """
    try:
        if not isinstance(args, dict) or "query" not in args:
            return args
        raw = args["query"]
        if isinstance(raw, str):
            queries = [raw]
        elif isinstance(raw, list) and raw and all(isinstance(item, str) for item in raw):
            queries = list(raw)
        else:
            return args
        if not any(str(item).strip() for item in queries):
            return args
        if len(queries) == 1:
            fan_out = await expanded_queries(queries[0])
            if fan_out:
                vertical = str(args.get("vertical") or "web")
                from core.query.search_plan import VERTICAL_QUERY_LIMITS
                limit = 1 if coerce_search_effort(args) == "high" else VERTICAL_QUERY_LIMITS.get(vertical, 3)
                queries = fan_out[:limit]
            else:
                return args
        else:
            queries, changed = await rewrite_queries(queries)
            if not changed:
                return args
        rewritten_args = dict(args)
        rewritten_args["query"] = queries[0] if len(queries) == 1 else queries
        if not prepare_search_arguments(rewritten_args).get("ok"):
            logger.warning("query rewriter produced an invalid search plan; using original arguments")
            return args
        return rewritten_args
    except Exception as exc:  # noqa: BLE001 — the rewriter must never break a search
        logger.warning("query rewriter args pass-through failed: %s", exc)
        return args


# Run the ranked web_search pipeline (the model-facing default tool).
#
# The model routes query strings through the vertical fields and chooses effort. Recency
# is parsed from the query, while region and safe-search remain internal routing details.
async def _call_web_search(
    args: dict[str, Any], *, preflight_warnings: list[dict[str, Any]] | None = None,
    instant_mode: bool = False,
) -> dict[str, Any]:
    # Instant mode is the latency-critical path and keeps its own contract.
    if not instant_mode:
        args = await _rewrite_search_args(args)
    prepared = prepare_search_arguments(args, instant_mode=instant_mode)
    if not prepared.get("ok"):
        return dict(prepared.get("error_result") or {})

    canonical_args = dict(prepared.get("arguments") or {})
    search_request = dict(prepared.get("search_request") or {})
    if search_request.get("schema_mode") in {"advanced", "instant"}:
        queries = [
            str(item.get("compiled_query") or "")
            for item in search_request.get("queries", [])
            if isinstance(item, dict)
        ]
        query = queries[0] if len(queries) == 1 else list(queries)
        query_preview = " | ".join(queries)[:320]
        effort = str(search_request.get("effort") or "medium")
        shopping = any(item.get("vertical") == "shopping" for item in search_request.get("queries", []))
        academic = any(item.get("vertical") == "academic" for item in search_request.get("queries", []))
        onion = any(item.get("vertical") == "onion" for item in search_request.get("queries", []))
    else:
        queries = coerce_search_queries(canonical_args.get("query", ""))
        query = queries[0] if len(queries) == 1 else list(queries)
        query_preview = " | ".join(queries)[:320]
        effort = coerce_search_effort(canonical_args)
        shopping = coerce_search_shopping(canonical_args)
        academic = coerce_search_academic(canonical_args)
        onion = coerce_search_onion(canonical_args)

    write_search_io_event(
        {
            "layer": "mcp_adapter",
            "phase": "web_search.request",
            "tool_id": "web_search",
            "query": query,
            "queries": list(queries),
            "search_request": search_request,
            "batch": len(queries) > 1,
            "effort": effort,
            "shopping": shopping,
            "academic": academic,
            "onion": onion,
        }
    )
    logger.info(
        "mcp.web_search.start effort=%s shopping=%s academic=%s onion=%s queries=%d query_preview=%r",
        effort, shopping, academic, onion, len(queries), query_preview,
    )

    started = time.perf_counter()
    try:
        if search_request.get("schema_mode") in {"advanced", "instant"}:
            result = await run_web_search_plan(search_request)
        elif len(queries) > 1:
            result = await run_web_search_batch(
                queries, effort=effort, shopping=shopping, academic=academic, onion=onion,
            )
        else:
            result = await run_web_search(
                queries[0] if queries else "",
                effort=effort,
                shopping=shopping,
                academic=academic,
                onion=onion,
            )
            result["search_request"] = search_request
            result.setdefault("ui", {})["search_request"] = search_request
            result["ui"]["query_count"] = len(queries)
    except Exception:
        logger.exception("mcp.web_search.failed query_preview=%r", query_preview)
        raise
    warnings = [
        dict(warning)
        for warning in prepared.get("warnings", [])
        if isinstance(warning, dict)
    ]
    for warning in preflight_warnings or []:
        if not isinstance(warning, dict):
            continue
        normalized_warning = dict(warning)
        if normalized_warning not in warnings:
            warnings.append(normalized_warning)
    if warnings:
        warning_context = "\n".join(
            f"{warning.get('code', 'SEARCH_WARNING')}: {warning.get('message', '')}".strip()
            for warning in warnings
        )
        existing_context = str(result.get("model_context") or "").strip()
        result["model_context"] = (
            f"{warning_context}\n\n{existing_context}" if existing_context else warning_context
        )
        result["warnings"] = warnings
        result.setdefault("ui", {})["warnings"] = warnings
    elapsed = time.perf_counter() - started
    logger.info(
        "mcp.web_search.done effort=%s queries=%d sources=%d elapsed=%.3fs",
        effort, len(queries), len(result.get("sources", [])), elapsed,
    )
    # Rich cards (config-gated, default OFF): wikipedia summary card + forum
    # discussion block for the top results. Fail-open on any error.
    result = await _attach_rich_cards(result, queries)
    write_search_io_event(
        {
            "layer": "mcp_adapter",
            "phase": "web_search.result",
            "tool_id": "web_search",
            "query": query,
            "queries": list(queries),
            "batch": len(queries) > 1,
            "result": result,
            "elapsed_seconds": elapsed,
        }
    )
    return result


# Parse a JSON-encoded list passed as a string tool argument (some callers stringify).
def _maybe_parse_list(val: Any) -> Any:
    if isinstance(val, str) and val.strip().startswith("["):
        import json
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return parsed
        except Exception:  # noqa: BLE001
            logger.debug("failed to parse list-like tool argument: %r", val[:200])
    return val


# Bare registrable host for a URL (no scheme/userinfo/port/www.).
def _source_domain(url: str) -> str:
    host = urlparse(url or "").netloc.lower()
    if "@" in host:
        host = host.rsplit("@", 1)[-1]
    if ":" in host:
        host = host.split(":", 1)[0]
    return host.removeprefix("www.")


# Short human label from a domain (example.com -> "Example").
def _display_domain(domain: str) -> str:
    parts = [p for p in (domain or "").split(".") if p]
    if len(parts) >= 2:
        label = parts[-2]
    elif parts:
        label = parts[0]
    else:
        return ""
    return label.replace("-", " ").title()


# DuckDuckGo favicon URL for a source chip.
def _favicon_url(domain: str) -> str:
    return f"https://icons.duckduckgo.com/ip3/{domain}.ico" if domain else ""


# One read_page source record (UI chip metadata + ok flag from the result text).
def _read_page_source(url: str, rank: int, result_text: str = "") -> dict[str, Any]:
    domain = _source_domain(url)
    ok = not str(result_text or "").lstrip().lower().startswith("error:")
    return {
        "rank": rank,
        "url": (url or "").strip(),
        "domain": domain,
        "display_domain": _display_domain(domain) or domain,
        "favicon_url": _favicon_url(domain),
        "ok": ok,
    }


# Assemble the structured read_page payload for one or many URLs (legacy bridge shape).
def _read_page_payload(urls: list[str], results: list[str]) -> dict[str, Any]:
    sources = [
        _read_page_source(url, i, results[i - 1] if i - 1 < len(results) else "")
        for i, url in enumerate(urls, 1)
    ]
    ok_count = sum(1 for s in sources if s.get("ok"))
    status = "done" if ok_count == len(sources) else ("partial" if ok_count else "error")
    return {
        "query": ", ".join(urls),
        "sources": sources,
        "model_context": "\n\n".join(str(r or "") for r in results).strip(),
        "ui": {
            "kind": "read_page",
            "status": status,
            "result_count": len(sources),
            "sources": sources,
        },
    }


# Fetch one or many URLs as markdown and wrap them in the structured payload.
# Run the image-search vertical and shape a gallery-friendly payload.
# In-memory TTL cache for image results: reopening a chat re-resolves every
# [[images:]] token, and re-running engines for the same query is pure latency.
_IMAGE_CACHE_TTL_SECONDS = 1800.0
_IMAGE_CACHE_MAX_ENTRIES = 100
_image_cache: dict[str, tuple[float, dict[str, Any]]] = {}


def _image_cache_key(query: str, limit: int) -> str:
    return f"{query.casefold()}|{limit}"


def _cached_image_result(query: str, limit: int) -> dict[str, Any] | None:
    entry = _image_cache.get(_image_cache_key(query, limit))
    if entry is None or entry[0] <= time.monotonic():
        return None
    return copy.deepcopy(entry[1])


def _store_image_result(query: str, limit: int, payload: dict[str, Any]) -> None:
    if len(_image_cache) >= _IMAGE_CACHE_MAX_ENTRIES:
        now = time.monotonic()
        for key in [k for k, (expires, _) in _image_cache.items() if expires <= now]:
            _image_cache.pop(key, None)
        while len(_image_cache) >= _IMAGE_CACHE_MAX_ENTRIES:
            _image_cache.pop(next(iter(_image_cache)))
    _image_cache[_image_cache_key(query, limit)] = (
        time.monotonic() + _IMAGE_CACHE_TTL_SECONDS,
        copy.deepcopy(payload),
    )


async def _call_image_search(args: dict[str, Any]) -> dict[str, Any]:
    from core.search.image_search import ImageSearchService

    query = " ".join(str(args.get("query") or "").split())
    if not query:
        return {
            "ui": {"kind": "image_search", "status": "error", "images": [], "error": "empty query"},
            "tool_ui": {"kind": "image_search", "status": "error", "images": []},
            "result": "Image search failed: query must be a non-empty string.",
        }
    try:
        limit = max(1, min(24, int(args.get("limit") or 12)))
    except (TypeError, ValueError):
        limit = 12

    cached = _cached_image_result(query, limit)
    if cached is not None:
        logger.info("mcp.image_search.cache hit query=%r", query[:120])
        return cached

    started = time.perf_counter()
    try:
        payload = await ImageSearchService().search(
            query, limit=limit, timeout_seconds=8.0
        )
    except Exception as exc:  # noqa: BLE001 — the tool must fail soft
        logger.warning("mcp.image_search.failed query=%r error=%s", query[:120], exc)
        return {
            "ui": {"kind": "image_search", "status": "error", "images": [], "error": str(exc)[:200]},
            "tool_ui": {"kind": "image_search", "status": "error", "images": []},
            "result": f"Image search failed: {exc}",
        }

    images = [
        {
            "id": f"img{index}",
            "title": str(source.get("title") or ""),
            "image_url": str(source.get("image_url") or ""),
            "thumbnail_url": str(source.get("thumbnail_url") or source.get("image_url") or ""),
            "page_url": str(source.get("url") or ""),
            "source_domain": str(source.get("source_domain") or ""),
            "width": source.get("width"),
            "height": source.get("height"),
            "engines": source.get("engines") or [],
        }
        for index, source in enumerate(payload.get("sources") or [], 1)
        if source.get("image_url")
    ]
    ui = {
        "kind": "image_search",
        "status": "done" if images else "empty",
        "query": payload.get("query") or query,
        "language": payload.get("language") or "",
        "result_count": len(images),
        "images": images,
    }
    write_search_io_event(
        {"layer": "mcp_adapter", "phase": "image_search.result", "tool_id": "image_search",
         "query": query, "result_count": len(images)}
    )
    logger.info(
        "mcp.image_search.done status=%s count=%d took=%.0fms",
        ui["status"], len(images), (time.perf_counter() - started) * 1000,
    )
    # The model-facing result stays compact: it sees URLs + metadata and can fetch
    # any image itself when it is multimodal; the rich gallery is UI-only.
    result_lines = [
        f'{item["id"]}: {item["title"][:80]} | {item["image_url"]}'
        + (f' | page: {item["page_url"]}' if item["page_url"] else "")
        for item in images
    ]
    model_context = (
        f'Images for "{query}" ({len(images)}):\n' + "\n".join(result_lines)
        if images
        else f'No images found for "{query}".'
    )
    if images:
        model_context += (
            "\n\nTo display these images, include the following service token in your final "
            "answer on its own line, outside code blocks, inline code, quotes, links, and lists:\n"
            f"[[images: {query[:140]}]]"
        )
    # model_context is the MCP contract's compact model-facing text: with it (plus
    # the ui dict) _extract_structured_tool_result keeps the gallery metadata and
    # the model sees the clean listing instead of a full JSON dump.
    if images:
        _store_image_result(query, limit, {
            "ui": ui,
            "tool_ui": ui,
            "model_context": model_context,
            "result": model_context,
        })
    return {
        "ui": ui,
        "tool_ui": ui,
        "model_context": model_context,
        "result": model_context,
    }


async def _call_read_page(args: dict[str, Any]) -> dict[str, Any]:
    import asyncio

    from core.config import load_search_config

    batch_limit = max(1, int(load_search_config().search.batch_query_limit))
    url = _maybe_parse_list(args.get("url", ""))
    write_search_io_event(
        {"layer": "mcp_adapter", "phase": "read_page.request", "tool_id": "read_page", "url": url}
    )

    if isinstance(url, list):
        urls = [u.strip() for u in url if isinstance(u, str) and u.strip()][:batch_limit]
        logger.info("mcp.read_page.start batch=True urls=%d", len(urls))
        results = await asyncio.gather(*[run_read_page(u) for u in urls], return_exceptions=True)
        texts = [r if isinstance(r, str) else f"Error: {r}" for r in results]
        payload = _read_page_payload(urls, texts)
    else:
        url_text = str(url).strip()
        logger.info("mcp.read_page.start batch=False url=%r", url_text[:220])
        text = await run_read_page(url_text)
        payload = _read_page_payload([url_text], [text])

    write_search_io_event(
        {"layer": "mcp_adapter", "phase": "read_page.result", "tool_id": "read_page",
         "result": payload}
    )
    logger.info("mcp.read_page.done status=%s count=%d",
                payload["ui"]["status"], payload["ui"]["result_count"])
    return payload


# Cancel outstanding background work (prefetch) and release this process's HTTP resources
# at server shutdown. The warm browser daemon is intentionally left running so it stays warm
# for the next tool call; it self-terminates on its own idle timeout (daemon_idle_shutdown_sec).
async def shutdown() -> None:
    from core.fetch.browser.client import shutdown_browser
    from core.search import serp_api
    from core.search.prefetch import shutdown_prefetch

    await shutdown_prefetch()
    await shutdown_browser()
    if serp_api._shared_transport is not None:
        await serp_api._shared_transport.close()


# Dispatch an ASLM tool call to the matching search implementation.
async def call_tool(
    tool_id: str,
    arguments: dict[str, Any] | None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args = dict(arguments or {})
    _evict_caches_once()
    if tool_id == "web_search":
        instant_mode = bool((context or {}).get("instant_mode"))
        inherited_warnings: list[dict[str, Any]] = []
        raw_arguments = (context or {}).get("raw_tool_arguments")
        if isinstance(raw_arguments, dict):
            raw_preflight = prepare_search_arguments(raw_arguments, instant_mode=instant_mode)
            if raw_preflight.get("ok"):
                inherited_warnings = [
                    dict(warning)
                    for warning in raw_preflight.get("warnings", [])
                    if isinstance(warning, dict)
                ]
        return await _call_web_search(
            args,
            preflight_warnings=inherited_warnings,
            instant_mode=instant_mode,
        )
    if tool_id == "image_search":
        return await _call_image_search(args)
    if tool_id == "read_page":
        return await _call_read_page(args)
    raise ValueError(f"Unknown tool: {tool_id}")
