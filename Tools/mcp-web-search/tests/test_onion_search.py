# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio

import core.fetch.onion.search as osearch
from core.fetch.onion.search import (
    OnionResult,
    _discover_for_service,
    _is_article_path,
)
from core.fetch.onion.models import OnionService
from core.fetch.onion.transport import OnionFetch


# Verify that the article-path heuristic accepts article ids and dated news paths and rejects section
# pages, listing roots and paths that are too shallow.
def test_article_path_heuristic():
    # Accepted: a DW article id and a dated news path.
    assert _is_article_path("/en/some-press-freedom-story/a-77698587")
    assert _is_article_path("/world/2026/jun/25/some-slug")

    # Rejected: a section page, a listing root and a path that is too shallow.
    assert not _is_article_path("/en/top-stories/s-9097")
    assert not _is_article_path("/search")
    assert not _is_article_path("/en")


# Onion service entry for Deutsche Welle, the provider used by the discovery tests.
_DW = OnionService(
    name="dw", category="media",
    clearnet_anchor="https://www.dw.com/en/",
    onion="http://www.dwnews.onion/en/",
)

# A SERP result dict for site:dw.com. It mixes article hits, a nav/section link (dropped), an
# off-domain stray (dropped) and a duplicate article across engines.
_FAKE_SERP = {
    "engines": {
        "ddg": {"sources": [
            {"url": "https://www.dw.com/en/press-freedom-under-pressure/a-77698587"},
            {"url": "https://www.dw.com/en/top-stories/s-9097"},
            {"url": "https://www.dw.com/en/free-media-matters/a-77587540"},
            {"url": "https://evil.example/dw.com/phish/a-1"},
        ]},
        "brave": {"sources": [
            {"url": "https://www.dw.com/en/how-big-tech-changes-journalism/a-77696474"},
            {"url": "https://www.dw.com/en/press-freedom-under-pressure/a-77698587"},
        ]},
    }
}

# Article page served by the faked onion fetch: enough body text for the extractor and BM25 to keep.
_ARTICLE_HTML = """
<html><body><article><h1>Press freedom under pressure</h1>
<p>Press freedom is under growing pressure worldwide as governments tighten control over
independent media. Journalists face detention, surveillance and legal threats in many
countries, watchdog groups report. Analysts say the trend threatens democratic accountability
and the public's access to reliable information about press freedom and censorship.</p>
<p>This second paragraph adds further detail so the extractor has real body content to work
with, discussing press freedom, journalists, and censorship at sufficient length.</p>
</article></body></html>
"""


# Verify that discovery runs a site:-scoped SERP query, rewrites the clearnet article URLs to the onion
# host, drops section and off-domain links, and de-duplicates articles across engines.
def test_discovery_rewrites_to_onion_and_filters(monkeypatch):
    # Return the canned SERP and check that the query is scoped to the service domain.
    async def fake_serp(query, **kw):
        assert query.startswith("site:dw.com ")
        return _FAKE_SERP

    monkeypatch.setattr(osearch, "run_serp_search", fake_serp, raising=False)
    monkeypatch.setattr(osearch, "resolve_onion", lambda svc, **k: "http://www.dwnews.onion/en/")

    pairs = asyncio.run(_discover_for_service(_DW, "press freedom", limit=10, serp_timeout=8))
    urls = [u for _, u in pairs]

    # Three unique articles remain, all on the onion host, sections and the off-domain stray dropped.
    assert len(urls) == 3
    assert all("dwnews.onion" in u for u in urls)
    assert all("/a-" in u for u in urls)
    assert not any("/s-" in u for u in urls)
    assert not any("evil.example" in u for u in urls)
    assert all(name == "dw" for name, _ in pairs)


# Patch the onion search module to answer every SERP query with the canned DW result.
def _patch_serp(monkeypatch):
    # Return the canned SERP for any query.
    async def fake_serp(query, **kw):
        return _FAKE_SERP
    monkeypatch.setattr(osearch, "run_serp_search", fake_serp, raising=False)


# Verify the full flow (discover, warm Tor, scrape, BM25 rank) returns OnionResult items for the DW
# provider with onion URLs and content that kept the query topic.
def test_onion_search_flow(monkeypatch):
    # Serve the article page for every onion URL.
    async def fake_fetch(url, *, timeout=None, impersonate="chrome124"):
        return OnionFetch(url=url, status="ok", ok=True, http_status=200, text=_ARTICLE_HTML)

    _patch_serp(monkeypatch)
    monkeypatch.setattr(osearch, "onion_fetch", fake_fetch)
    monkeypatch.setattr(osearch, "resolve_onion", lambda svc, **k: "http://www.dwnews.onion/en/")

    # Report Tor as available without spawning anything real.
    async def fake_warm(budget):
        return True
    monkeypatch.setattr(osearch, "_warm_tor", fake_warm)

    results = asyncio.run(osearch.onion_search(
        "press freedom", limit=3, per_link_timeout=5, max_chars=2000,
        concurrency=3, providers=("dw",),
    ))
    assert results and all(isinstance(r, OnionResult) for r in results)
    assert all(r.provider == "dw" and "dwnews.onion" in r.url for r in results)

    # BM25 kept the query topic in the content.
    assert any("press freedom" in r.content.lower() for r in results)
    assert all(r.content for r in results)


# Verify that the search returns no results when Tor cannot be warmed up, without scraping anything.
def test_onion_search_empty_when_tor_unavailable(monkeypatch):
    _patch_serp(monkeypatch)
    monkeypatch.setattr(osearch, "resolve_onion", lambda svc, **k: "http://www.dwnews.onion/en/")

    # Report Tor as down.
    async def fake_warm(budget):
        return False
    monkeypatch.setattr(osearch, "_warm_tor", fake_warm)
    results = asyncio.run(osearch.onion_search("x", providers=("dw",), per_link_timeout=5))
    assert results == []


# Verify that the search returns no results when the SERP only contains a section link and so yields no
# article candidates.
def test_onion_search_empty_when_no_candidates(monkeypatch):
    # Return a SERP with a single section link.
    async def empty_serp(query, **kw):
        return {"engines": {"ddg": {"sources": [{"url": "https://www.dw.com/en/x/s-1"}]}}}
    monkeypatch.setattr(osearch, "run_serp_search", empty_serp, raising=False)
    monkeypatch.setattr(osearch, "resolve_onion", lambda svc, **k: "http://www.dwnews.onion/en/")
    results = asyncio.run(osearch.onion_search("x", providers=("dw",), per_link_timeout=5))
    assert results == []
