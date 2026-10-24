# Copyright NEXTGGTECH. Elastic License 2.0.

"""Fail-open rich cards for known result types (port of ASLM-Search's cards service).

Two card kinds, both strictly optional enrichment for the top SERP results:
- wikipedia: a localized summary card (REST summary API with langlink resolution),
  fetched under a short timeout and cached (positive and negative TTLs).
- discussions: a metadata-only block that groups StackExchange/Reddit/GitHub
  Discussions/HackerNews threads already present in the results — no network.

Everything fails open: any error returns an empty card and the search result is
untouched. The whole feature is config-gated and DEFAULT OFF in search_config.json
because a card costs an extra API roundtrip per query.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote, unquote, urlsplit

import aiohttp

logger = logging.getLogger("mcp.web_search.rich_cards")

_MAX_RESPONSE_BYTES = 1_048_576
# Wikimedia's API User-Agent policy requires a descriptive agent with contact
# info; bare product names and raw browser UAs both draw 403s on the REST API.
_USER_AGENT = "ASLM-Chat-WebSearch/0.1 (https://github.com/Di120078/ASLM-Chat)"
_STACK_EXCHANGE_HOSTS = frozenset(
    {
        "stackoverflow.com",
        "superuser.com",
        "serverfault.com",
        "askubuntu.com",
        "mathoverflow.net",
    }
)
_QUESTION_RE = re.compile(r"/(?:questions|q)/(\d+)(?:/|$)")
_REDDIT_POST_RE = re.compile(r"/r/([^/]+)/comments/([a-z0-9]+)(?:/|$)", re.IGNORECASE)
_GITHUB_DISCUSSION_RE = re.compile(
    r"^/([^/]+)/([^/]+)/discussions/(\d+)(?:/|$)", re.IGNORECASE
)
_HACKER_NEWS_RE = re.compile(r"^/item(?:/|$)", re.IGNORECASE)
_DISCUSSION_INITIAL_LIMIT = 4
_DISCUSSION_TOTAL_LIMIT = 8


class _PlainTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data)


def _plain_text(value: object, *, maximum: int) -> str:
    if value is None:
        return ""
    parser = _PlainTextParser()
    try:
        parser.feed(str(value))
        parser.close()
    except Exception:  # noqa: BLE001 — malformed markup must never break a card
        return ""
    text = " ".join(" ".join(parser.parts).split())
    return text[:maximum]


def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


def _provider_for(url: str) -> str:
    host = _host(url)
    if host.endswith(".wikipedia.org"):
        return "wikipedia"
    return ""


def _discussion_source(url: str) -> dict[str, str]:
    host = _host(url)
    path = urlsplit(url).path
    if host in _STACK_EXCHANGE_HOSTS and _QUESTION_RE.search(path):
        return {"kind": "stackexchange", "provider": host.split(".", 1)[0]}
    if host == "github.com" and _GITHUB_DISCUSSION_RE.search(path):
        return {"kind": "github", "provider": "GitHub"}
    if host == "news.ycombinator.com" and _HACKER_NEWS_RE.search(path):
        return {"kind": "hackernews", "provider": "Hacker News"}
    if host == "reddit.com" or host.endswith(".reddit.com"):
        if _REDDIT_POST_RE.search(path):
            return {"kind": "reddit", "provider": "Reddit"}
    return {}


def _discussion_title(value: object, kind: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return ""
    if kind == "stackexchange" and not text.endswith("?"):
        text = f"{text}?"
    return text[:160]


def _discussion_key(url: str) -> str:
    match = _REDDIT_POST_RE.search(urlsplit(url).path)
    if match:
        return f"reddit:{match.group(2)}"
    match = _QUESTION_RE.search(urlsplit(url).path)
    if match and _host(url) in _STACK_EXCHANGE_HOSTS:
        return f"se:{match.group(1)}"
    match = _GITHUB_DISCUSSION_RE.search(urlsplit(url).path)
    if match:
        return f"gh:{match.group(1)}:{match.group(3)}"
    if _host(url) == "news.ycombinator.com" and _HACKER_NEWS_RE.search(urlsplit(url).path):
        return f"hn:{urlsplit(url).query}"
    return ""


class _TtlCache:
    """Tiny in-memory TTL cache; cards are cheap to rebuild after a restart."""

    def __init__(self, max_entries: int = 512) -> None:
        self._entries: dict[str, tuple[Any, float]] = {}
        self._max_entries = max_entries

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        value, expires = entry
        if expires <= time.monotonic():
            self._entries.pop(key, None)
            return None
        return value

    def set(self, key: str, value: Any, *, ttl: float) -> None:
        if len(self._entries) >= self._max_entries:
            oldest = min(self._entries.items(), key=lambda item: item[1][1])[0]
            self._entries.pop(oldest, None)
        self._entries[key] = (value, time.monotonic() + max(1.0, ttl))


def _cache_key(provider: str, url: str, language: str = "") -> str:
    digest = hashlib.sha256(f"{provider}|{language}|{url}".encode("utf-8")).hexdigest()
    return f"rich-card:v1:{digest}"


async def _fetch_json(
    url: str, *, params: dict[str, str], timeout: float
) -> dict[str, Any]:
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}
    async with aiohttp.ClientSession(timeout=client_timeout, headers=headers) as session:
        async with session.get(url, params=params, allow_redirects=False) as response:
            if response.status != 200:
                return {}
            chunks: list[bytes] = []
            received = 0
            async for chunk in response.content.iter_chunked(65_536):
                received += len(chunk)
                if received > _MAX_RESPONSE_BYTES:
                    return {}
                chunks.append(chunk)
            raw = b"".join(chunks)
            try:
                value = json.loads(raw.decode(response.charset or "utf-8"))
            except (LookupError, UnicodeDecodeError, ValueError):
                return {}
            return value if isinstance(value, dict) else {}


async def _wikipedia_card(result: dict[str, Any], timeout: float) -> dict[str, Any]:
    url = str(result.get("url") or "")
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()
    path_marker = "/wiki/"
    if not host.endswith(".wikipedia.org") or path_marker not in parsed.path:
        return {}
    source_language = host.removeprefix("www.").split(".", 1)[0]
    requested_language = str(result.get("_card_language") or source_language)
    target_language = requested_language.lower().split("-", 1)[0]
    if not re.fullmatch(r"[a-z]{2,3}", target_language):
        target_language = source_language
    page_title = unquote(parsed.path.split(path_marker, 1)[1]).strip()
    if not source_language or not page_title or "/" in page_title:
        return {}

    localized_title = page_title
    if target_language != source_language:
        langlinks_payload = await _fetch_json(
            f"https://{source_language}.wikipedia.org/w/api.php",
            params={
                "action": "query",
                "prop": "langlinks",
                "lllang": target_language,
                "lllimit": "1",
                "titles": page_title.replace("_", " "),
                "redirects": "1",
                "format": "json",
                "formatversion": "2",
            },
            timeout=timeout,
        )
        query = (
            langlinks_payload.get("query")
            if isinstance(langlinks_payload.get("query"), dict)
            else {}
        )
        pages = query.get("pages") if isinstance(query.get("pages"), list) else []
        page = pages[0] if pages and isinstance(pages[0], dict) else {}
        links = page.get("langlinks") if isinstance(page.get("langlinks"), list) else []
        link = links[0] if links and isinstance(links[0], dict) else {}
        localized_title = str(link.get("title") or link.get("*") or page_title)

    async def summary(language: str, title: str) -> dict[str, Any]:
        endpoint = (
            f"https://{language}.wikipedia.org/api/rest_v1/page/summary/"
            f"{quote(title, safe='()_,-')}"
        )
        return await _fetch_json(endpoint, params={"redirect": "true"}, timeout=timeout)

    payload = await summary(target_language, localized_title)
    resolved_language = target_language
    if not payload and target_language != source_language:
        payload = await summary(source_language, page_title)
        resolved_language = source_language
    if (
        not payload
        or payload.get("type")
        == "https://mediawiki.org/wiki/HyperSwitch/errors/not_found"
    ):
        return {}
    content_urls = (
        payload.get("content_urls") if isinstance(payload.get("content_urls"), dict) else {}
    )
    desktop = content_urls.get("desktop") if isinstance(content_urls.get("desktop"), dict) else {}
    thumbnail = payload.get("thumbnail") if isinstance(payload.get("thumbnail"), dict) else {}
    return {
        "kind": "wikipedia",
        "provider": "Wikipedia",
        "url": str(desktop.get("page") or url),
        "title": _plain_text(payload.get("title") or result.get("title"), maximum=160),
        "description": _plain_text(payload.get("description"), maximum=120),
        "summary": _plain_text(payload.get("extract") or result.get("snippet"), maximum=520),
        "image_url": str(thumbnail.get("source") or ""),
        "image_width": int(thumbnail.get("width") or 0),
        "image_height": int(thumbnail.get("height") or 0),
        "language": resolved_language,
    }


class RichCardService:
    """Build cached cards only for the top visible SERP candidates."""

    _shared_cache = _TtlCache()

    def __init__(self) -> None:
        self.cache = self._shared_cache

    async def build(
        self,
        results: list[dict[str, Any]],
        config: Any,
        *,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        if not getattr(config, "enabled", False) or not results:
            return []
        candidates: list[tuple[int, str, dict[str, Any], str]] = []
        seen: set[str] = set()
        for index, result in enumerate(results):
            url = str(result.get("url") or "")
            provider = _provider_for(url)
            if provider != "wikipedia" or not getattr(config, "wikipedia", True):
                continue
            if provider in seen:
                continue
            seen.add(provider)
            candidate = {**result, "_card_language": language}
            candidates.append(
                (
                    index,
                    provider,
                    candidate,
                    _cache_key(provider, url, language.lower()),
                )
            )
            if len(candidates) >= int(getattr(config, "max_cards", 3)):
                break
        if not candidates:
            return []

        async def resolve(
            index: int, provider: str, result: dict[str, Any], key: str
        ) -> tuple[int, dict[str, Any]]:
            cached = self.cache.get(key)
            if cached is not None:
                logger.debug("rich card cache hit provider=%s", provider)
                return index, cached.get("card") if isinstance(cached.get("card"), dict) else {}
            started = time.perf_counter()
            try:
                async with asyncio.timeout(float(getattr(config, "timeout_seconds", 2.0))):
                    card = await _wikipedia_card(
                        result, float(getattr(config, "timeout_seconds", 2.0))
                    )
            except (aiohttp.ClientError, asyncio.TimeoutError, ValueError, TypeError) as exc:
                logger.debug("rich card fetch failed provider=%s error=%s", provider, exc)
                card = {}
            logger.info(
                "rich card resolved provider=%s url=%r hit=%s in %.0fms",
                provider, str(result.get("url") or "")[:80], bool(card),
                (time.perf_counter() - started) * 1000,
            )
            ttl = float(getattr(config, "ttl_seconds", 86400)) if card else float(
                getattr(config, "negative_ttl_seconds", 900)
            )
            self.cache.set(key, {"card": card}, ttl=ttl)
            return index, card

        resolved = await asyncio.gather(*(resolve(*candidate) for candidate in candidates))
        resolved.sort(key=lambda item: item[0])
        return [card for _index, card in resolved if card]

    def build_discussions(
        self,
        results: list[dict[str, Any]],
        config: Any,
    ) -> dict[str, Any]:
        """Build a scriptless forum block from SERP metadata only (no network)."""

        if not getattr(config, "enabled", False) or not results:
            return {}
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for result in results:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "")
            source = _discussion_source(url)
            kind = source.get("kind", "")
            if not source or (kind == "reddit" and not getattr(config, "reddit", True)):
                continue
            if kind == "stackexchange" and not getattr(config, "stackexchange", True):
                continue
            key = _discussion_key(url)
            title = _discussion_title(result.get("title"), kind)
            if not key or key in seen or not title:
                continue
            seen.add(key)
            domain = _host(url)
            items.append(
                {
                    **source,
                    "url": url,
                    "title": title,
                    "summary": _plain_text(result.get("snippet"), maximum=420),
                    "domain": domain,
                    "favicon_url": str(
                        result.get("favicon_url")
                        or (f"https://{domain}/favicon.ico" if domain else "")
                    ),
                }
            )
            if len(items) >= _DISCUSSION_TOTAL_LIMIT:
                break
        if not items:
            return {}
        logger.info("rich card discussions built count=%d", len(items))
        return {
            "items": items[:_DISCUSSION_INITIAL_LIMIT],
            "extra_items": items[_DISCUSSION_INITIAL_LIMIT:],
        }


__all__ = ["RichCardService"]
