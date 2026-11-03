# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import contextlib
import html
import logging
import re
import time
from collections.abc import AsyncIterator
from typing import Any
from urllib.parse import urlparse

import httpx

from core.query.provider_compiler import compile_provider_query

from .hosted_providers import HostedProvider, HostedResult, available_providers, sanitize_query_for_api

logger = logging.getLogger("services.web_search")
trace_logger = logging.getLogger("trace.web_search")

# Paragraph boundary: a blank line, possibly containing whitespace.
_PARA_SPLIT_RE = re.compile(r"\n\s*\n")


# Return the lowercased host of a URL, or an empty string.
def _host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ""


# Wrap provider text (plain or markdown) in block HTML so trafilatura/normalize_page
# keeps paragraph structure when the compaction pipeline runs on the cache hit.
def _wrap_as_html(text: str) -> str:
    blocks = [html.escape(b.strip()) for b in _PARA_SPLIT_RE.split(text or "") if b.strip()]
    body = "".join(f"<p>{b}</p>" for b in blocks)
    return f"<html><body><article>{body}</article></body></html>"


# Pre-populate SourceCache so read_page cache-hits this URL and compacts the provider
# content without a re-fetch. Uses the same key read_page looks up. Best-effort.
def _prepopulate_cache(results: list[HostedResult]) -> int:
    content_rows = [r for r in results if r.content and r.content.strip()]
    if not content_rows:
        return 0
    try:
        from core.cache import get_page_cache
        from core.read.service import _cache_key_for_read, _variant_label
    except Exception as exc:  # noqa: BLE001 — cache feed is optional
        logger.debug("hosted cache pre-populate unavailable: %s", exc)
        return 0

    cache = get_page_cache()
    cached = 0
    for r in content_rows:
        try:
            key = _cache_key_for_read(r.url, variant=_variant_label(r.url))
            existing = cache.get_cached(key)
            if existing and cache.is_fresh(key) and (existing.raw_html or existing.clean_text):
                continue
            cache.cache_page(key, r.title, clean_text="", raw_html=_wrap_as_html(r.content))
            cached += 1
        except Exception:  # noqa: BLE001 — one bad row must not break the feed
            continue
    return cached


# Build the per-provider status payload (mirrors serp_api's engine payload shape).
def _engine_payload(provider_name: str, family: str, results: list[HostedResult], fetch_ms: float) -> dict[str, Any]:
    return {
        "engine": f"hosted:{provider_name}",
        "provider_family": family,
        "status": "success" if results else "empty",
        "fetch_ms": round(fetch_ms, 2),
        "parse_ms": 0.0,
        "sources": [
            {"url": r.url, "title": r.title, "snippet": r.snippet} for r in results
        ],
    }


# Stream hosted-provider sources/votes/engine events, pre-populating content as it lands.
# Emits the exact event shape of `serp_api.search_stream` ({"type": "source"|"vote"|"engine"}), so
# `web_search` consumes scrape and hosted sources through one triage with no special-casing. As
# content-bearing providers (Tavily, Firecrawl) return, their full page text is pre-populated into
# SourceCache under the read_page cache key BEFORE the source event is emitted, so when the
# orchestrator later parses that URL, read_page gets a cache hit and runs the normal
# extraction/compaction pipeline with no network fetch.
# Hosted is strictly a supplement: with no providers available it yields nothing and the search
# stays pure scrape.
async def hosted_search_stream(
    query: str,
    *,
    region: str = "us-en",
    max_results: int = 5,
    deadline_seconds: float = 8.0,
    providers: list[HostedProvider] | None = None,
    query_text: str = "",
    operators: dict[str, Any] | None = None,
    timelimit: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    providers = providers if providers is not None else available_providers()
    if not providers:
        return

    # Shared state: the event queue drained by the consumer below, plus the URLs already emitted
    # as sources (guarded by a lock) so a repeat from another provider becomes a vote.
    queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
    seen: set[str] = set()
    seen_lock = asyncio.Lock()

    # Query one provider, feed its content into the cache and push its source, vote and engine events.
    async def run(provider: HostedProvider, client: httpx.AsyncClient) -> None:
        family = provider.provider_family

        # Compile the provider-specific query and sanitize it for the hosted API.
        compiled = compile_provider_query(
            query_text or query,
            operators,
            provider.name,
            fallback_query=query,
            timelimit=timelimit,
        )
        sanitized = sanitize_query_for_api(compiled.query)

        # Run the search and time it; a provider failure yields an empty result list.
        started = time.perf_counter()
        try:
            results = await provider.search(client, sanitized, max_results=max_results)
        except Exception as exc:  # noqa: BLE001 — provider error never sinks the stream
            logger.warning("hosted provider %s failed: %s", provider.name, exc)
            results = []
        fetch_ms = (time.perf_counter() - started) * 1000

        # Content into the cache first, so a later parse cache-hits and compacts it.
        if results:
            n = await asyncio.get_running_loop().run_in_executor(None, _prepopulate_cache, results)
            if n:
                trace_logger.info("hosted.cache_feed provider=%s urls=%d", provider.name, n)

        # Emit one event per row: the first sighting of a URL is a source, a repeat is a vote.
        for rank, r in enumerate(results, 1):
            if not r.url:
                continue
            async with seen_lock:
                is_dup = r.url in seen
                if not is_dup:
                    seen.add(r.url)
            event_type = "vote" if is_dup else "source"
            item: dict[str, Any] = {
                "type": event_type,
                "engine": f"hosted:{provider.name}",
                "provider_family": family,
                "rank": rank,
                "url": {"url": r.url, "host": _host_of(r.url)},
            }
            if event_type == "source":
                item["serp"] = {
                    "title": r.title, "snippet": r.snippet,
                    "fetch_ms": fetch_ms, "parse_ms": 0.0,
                }
            await queue.put(item)

        # Close the provider with its engine status event, carrying the compiled query details.
        payload = _engine_payload(provider.name, family, results, fetch_ms)
        payload["query"] = sanitized
        payload["timelimit"] = compiled.timelimit
        payload["omitted_operators"] = list(compiled.omitted_operators)
        await queue.put({"type": "engine", "engine": f"hosted:{provider.name}",
                         "payload": payload})

    # Run all providers concurrently and always finish with a None sentinel so the consumer stops.
    async def produce() -> None:
        try:
            async with httpx.AsyncClient(timeout=deadline_seconds, follow_redirects=True) as client:
                async with asyncio.TaskGroup() as group:
                    for provider in providers:
                        group.create_task(run(provider, client), name=f"hosted:{provider.name}")
        finally:
            await queue.put(None)

    # Consume events as they arrive until the sentinel or the overall deadline.
    producer = asyncio.create_task(produce())
    try:
        async with asyncio.timeout(deadline_seconds):
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item
    except TimeoutError:
        # Deadline reached: keep whatever was already yielded and let the search move on.
        pass
    finally:
        # Stop any provider still running and swallow its cancellation or failure.
        producer.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await producer
