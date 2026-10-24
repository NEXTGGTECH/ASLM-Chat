# Copyright NEXTGGTECH. Elastic License 2.0.

"""Concurrent image search with cross-provider deduplication.

Ported from ASLM-Search, adapted to this core's SerpApi (no page routing: the
tool only serves first pages) and query_locales (the query's own script picks
the region, so a Russian query searches Yandex-friendly ru-ru image indexes).
"""

from __future__ import annotations

import re
from typing import Any

from core.config import load_search_config
from core.engines.images import IMAGE_ENGINE_TYPES, BingImagesParser
from core.fetch.transport import AdaptiveTransport

from .query_locales import analyze_query
from .serp_api import SerpApi
from .web_search import canonicalize_url


def enabled_image_engine_types(
    enabled: dict[str, bool] | None = None,
) -> tuple[type, ...]:
    engine_map = (
        enabled
        if enabled is not None
        else load_search_config().image_engines.as_map()
    )
    selected = tuple(
        parser
        for name, parser in IMAGE_ENGINE_TYPES.items()
        if bool(engine_map.get(name, False))
    )
    return selected or (BingImagesParser,)


def _image_key(source: dict[str, Any]) -> str:
    image_url = str(source.get("image_url") or "").strip()
    return canonicalize_url(image_url) if image_url else ""


def _merge_duplicate(keep: dict[str, Any], other: dict[str, Any], engine: str) -> None:
    engines = keep.setdefault("engines", [])
    if engine not in engines:
        engines.append(engine)
    for field in (
        "thumbnail_url",
        "source",
        "source_domain",
        "width",
        "height",
        "image_format",
        "filesize",
        "author",
        "license",
    ):
        if not keep.get(field) and other.get(field):
            keep[field] = other[field]


def _query_relevance(query: str, source: dict[str, Any]) -> float:
    terms = {
        token.casefold()
        for token in re.findall(r"[^\W_]+", query, re.UNICODE)
        if len(token) >= 2
    }
    if not terms:
        return 0.0
    haystack = " ".join(
        str(source.get(field) or "")
        for field in ("title", "snippet", "url", "image_url", "source")
    ).casefold()
    return sum(term in haystack for term in terms) / len(terms)


class ImageSearchService:
    """Run the configured image adapters and interleave their ranked results."""

    async def search(
        self,
        query: str,
        *,
        limit: int = 12,
        region: str = "",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        timeout_seconds: float = 6.0,
        enabled_engines: dict[str, bool] | None = None,
    ) -> dict[str, Any]:
        normalized_query = " ".join(str(query or "").split()).strip()
        if not normalized_query:
            raise ValueError("query must not be empty")
        # The query's own script picks the region (query_locales): image indexes
        # are far more language-sensitive than web SERPs.
        analysis = analyze_query(normalized_query)
        resolved_region = region or analysis.region
        selected = enabled_image_engine_types(enabled_engines)
        per_engine_limit = min(50, max(20, int(limit) * 2))
        transport = AdaptiveTransport(timeout_seconds=timeout_seconds)
        api = SerpApi(
            transport=transport,
            timeout_seconds=timeout_seconds,
            source_limit=per_engine_limit,
            engines=selected,
        )
        engine_payloads: dict[str, dict[str, Any]] = {}
        try:
            async for event in api.search_stream(
                normalized_query,
                region=resolved_region,
                safesearch=safesearch,
                timelimit=timelimit,
                deadline_seconds=timeout_seconds,
            ):
                if event["type"] == "engine":
                    payload = event["payload"]
                    engine_payloads[str(payload["engine"])] = payload
        finally:
            await transport.close()

        merged: list[dict[str, Any]] = []
        seen: dict[str, dict[str, Any]] = {}
        longest = max(
            (
                len((engine_payloads.get(parser.name) or {}).get("sources") or [])
                for parser in selected
            ),
            default=0,
        )
        for rank in range(longest):
            for parser in selected:
                sources = (engine_payloads.get(parser.name) or {}).get("sources") or []
                if rank >= len(sources):
                    continue
                source = dict(sources[rank])
                key = _image_key(source)
                if not key:
                    continue
                existing = seen.get(key)
                if existing is not None:
                    _merge_duplicate(existing, source, parser.name)
                    continue
                source["engine"] = parser.name
                source["engines"] = [parser.name]
                source["rank"] = rank + 1
                seen[key] = source
                merged.append(source)

        for source in merged:
            position = 1.0 / max(1, int(source.get("rank") or 1))
            relevance = _query_relevance(normalized_query, source)
            consensus = 0.1 * max(0, len(source.get("engines") or []) - 1)
            source["score"] = round(position * 0.5 + relevance * 0.5 + consensus, 4)
        merged.sort(key=lambda source: float(source.get("score") or 0), reverse=True)

        return {
            "query": normalized_query,
            "region": resolved_region,
            "language": analysis.language,
            "sources": merged[: max(1, int(limit))],
            "engines": engine_payloads,
        }


__all__ = ["ImageSearchService", "enabled_image_engine_types"]
