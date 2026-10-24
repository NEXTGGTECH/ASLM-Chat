# Copyright NEXTGGTECH. Elastic License 2.0.

"""Hosted supplement layer: provider clients, key gating, content→cache feed, consensus."""

from __future__ import annotations

import asyncio

import httpx

from core.config.api_keys import ApiKeysConfig, HostedSearchApiKeysSection, SearchApiKeysSection
from core.search import hosted_providers as hp
from core.search import hosted_stream as hs
from core.search.hosted_providers import (
    BraveClient,
    FirecrawlClient,
    HostedResult,
    SerpApiClient,
    TavilyClient,
)


def _keys(**kw) -> ApiKeysConfig:
    return ApiKeysConfig(search=SearchApiKeysSection(hosted_api=HostedSearchApiKeysSection(**kw)))


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- provider parsing ------------------------------------------------------------

def test_tavily_returns_content(monkeypatch):
    monkeypatch.setattr(hp, "load_api_keys", lambda: _keys(tavily_api_key="k"))

    def handler(req):
        assert req.url.host == "api.tavily.com"
        return httpx.Response(200, json={"results": [
            {"url": "https://a.com", "title": "A", "content": "short",
             "raw_content": "FULL PAGE TEXT", "published_date": "2026-01-01"},
        ]})

    async def go():
        async with _client(handler) as c:
            return await TavilyClient().search(c, "q", max_results=5)

    out = asyncio.run(go())
    assert len(out) == 1
    assert out[0].content == "FULL PAGE TEXT"
    assert out[0].provider_family == "tavily"
    assert out[0].snippet == "short"


def test_firecrawl_content_mode_returns_markdown(monkeypatch):
    monkeypatch.setattr(hp, "load_api_keys", lambda: _keys(firecrawl_api_key="k"))
    monkeypatch.setattr(hp, "provider_mode", lambda name: "content")

    def handler(req):
        assert req.headers.get("authorization") == "Bearer k"
        assert b"scrapeOptions" in req.content  # content mode pays for the scrape
        return httpx.Response(200, json={"data": [
            {"url": "https://b.com", "title": "B", "description": "desc",
             "markdown": "# Heading\n\nbody text"},
        ]})

    async def go():
        async with _client(handler) as c:
            return await FirecrawlClient().search(c, "q", max_results=5)

    out = asyncio.run(go())
    assert out[0].content == "# Heading\n\nbody text"
    assert out[0].provider_family == "firecrawl"


def test_firecrawl_default_serp_mode_skips_scrape(monkeypatch):
    # Default mode is "serp": no scrapeOptions in the request (no credits burned on
    # scrapes the hosted deadline would then discard) and no stub content for the cache.
    monkeypatch.setattr(hp, "load_api_keys", lambda: _keys(firecrawl_api_key="k"))

    def handler(req):
        assert b"scrapeOptions" not in req.content
        return httpx.Response(200, json={"data": [
            {"url": "https://b.com", "title": "B", "description": "desc",
             "markdown": "# should be ignored in serp mode"},
        ]})

    async def go():
        async with _client(handler) as c:
            return await FirecrawlClient().search(c, "q", max_results=5)

    out = asyncio.run(go())
    assert out[0].content == ""
    assert out[0].snippet == "desc"


def test_serpapi_family_is_google_and_no_content(monkeypatch):
    monkeypatch.setattr(hp, "load_api_keys", lambda: _keys(serpapi_api_key="k"))

    def handler(req):
        return httpx.Response(200, json={"organic_results": [
            {"link": "https://c.com", "title": "C", "snippet": "snip", "date": "1d"},
        ]})

    async def go():
        async with _client(handler) as c:
            return await SerpApiClient().search(c, "q", max_results=5)

    out = asyncio.run(go())
    assert out[0].provider_family == "google"  # votes with the Google scrape parser
    assert out[0].content == ""


def test_provider_soft_fails_on_http_error(monkeypatch):
    monkeypatch.setattr(hp, "load_api_keys", lambda: _keys(brave_api_key="k"))

    async def go():
        async with _client(lambda req: httpx.Response(429, json={})) as c:
            return await BraveClient().search(c, "q", max_results=5)

    assert asyncio.run(go()) == []


# --- key gating ------------------------------------------------------------------

def test_available_providers_gated_by_keys(monkeypatch):
    monkeypatch.setattr(hp, "load_api_keys", lambda: _keys())
    assert hp.available_providers() == []
    monkeypatch.setattr(hp, "load_api_keys", lambda: _keys(tavily_api_key="k", serpapi_api_key="k"))
    names = [p.name for p in hp.available_providers()]
    assert names == ["tavily", "serpapi"]


# --- stream + content→cache feed -------------------------------------------------

class _FakeProvider:
    def __init__(self, results, name="tavily", family="tavily"):
        self._results = results
        self.name = name
        self.provider_family = family
        self.returns_content = True

    def key(self, keys):
        return "k"

    async def search(self, client, query, *, max_results):
        return list(self._results)


def test_stream_emits_events_and_feeds_cache(monkeypatch, tmp_path):
    from core.cache.source_cache import SourceCache
    from core.read.service import _cache_key_for_read, _variant_label

    # Cache safety checks resolve DNS; pin them to a public address so the
    # test stays offline (fake-IP proxies resolve x.com to private ranges).
    monkeypatch.setattr(
        "core.fetch.url_utils._resolve_host_ips", lambda host: {"93.184.216.34"}
    )

    cache = SourceCache(str(tmp_path / "src.db"))
    monkeypatch.setattr("core.cache.get_page_cache", lambda: cache)

    provider = _FakeProvider([
        HostedResult(url="https://x.com", title="X", snippet="s",
                     provider="tavily", provider_family="tavily", content="DEEP CONTENT")
    ])

    async def go():
        return [e async for e in hs.hosted_search_stream(
            "q", providers=[provider], deadline_seconds=5.0)]

    events = asyncio.run(go())
    kinds = [e["type"] for e in events]
    assert "source" in kinds and "engine" in kinds
    src = next(e for e in events if e["type"] == "source")
    assert src["url"]["url"] == "https://x.com"
    assert src["provider_family"] == "tavily"

    key = _cache_key_for_read("https://x.com", variant=_variant_label("https://x.com"))
    cached = cache.get_cached(key)
    assert cached is not None and "DEEP CONTENT" in cached.raw_html


def test_stream_dedup_emits_vote(monkeypatch, tmp_path):
    from core.cache.source_cache import SourceCache

    monkeypatch.setattr("core.cache.get_page_cache", lambda: SourceCache(str(tmp_path / "s.db")))
    r = HostedResult(url="https://dup.com", title="D", snippet="s",
                     provider="brave", provider_family="brave")
    provider = _FakeProvider([r, r], name="brave", family="brave")
    provider.returns_content = False

    async def go():
        return [e async for e in hs.hosted_search_stream(
            "q", providers=[provider], deadline_seconds=5.0)]

    kinds = [e["type"] for e in asyncio.run(go())]
    assert kinds.count("source") == 1
    assert kinds.count("vote") == 1
