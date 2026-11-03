# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import logging
import re as _re
from dataclasses import dataclass
from urllib.parse import urlparse

from core.search.serp_api import run_serp_search

from .registry import service_for, services_in
from .resolver import resolve_onion
from .transport import onion_fetch

logger = logging.getLogger("core.fetch.onion.search")

# Categories whose onion mirrors are article sites worth a deep content search. Other vetted
# services (mail/infosec/whistleblow portals) don't publish searchable articles.
_SEARCHABLE_CATEGORIES = ("media", "rights")

# Path segments that mark a listing/nav page, not an article.
_LISTING_SEGMENTS = frozenset({
    "search", "s", "tag", "tags", "topic", "topics", "category", "categories",
    "latest", "live", "section", "sections", "author", "profile", "newsletter",
})

# A path is an article only with a strong marker — a date segment (/YYYY/…, most news), an
# article-id suffix (/a-12345678 on DW etc.), or a long numeric id. This rejects nav/section
# links (e.g. DW /en/top-stories/s-9097) that a bare SERP sweep would otherwise admit.
_ARTICLE_RE = _re.compile(r"/(19|20)\d{2}/|/a-?\d{4,}\b|/\d{7,}\b")


# One scraped onion article: its onion URL, host, extracted title, BM25-compressed content and
# the name of the service (provider) it came from.
@dataclass(slots=True)
class OnionResult:
    url: str
    host: str
    title: str
    content: str
    provider: str


# Registrable host of a URL/anchor, www-stripped ("https://www.dw.com/en/" -> "dw.com").
def _anchor_host(url: str) -> str:
    return (urlparse(url).netloc or "").lower().removeprefix("www.")


# Heuristic: does this path look like an article (vs nav/listing)? Generic across news sites:
# at least two segments, not a known listing root, and a date/id marker.
def _is_article_path(path: str) -> bool:
    segs = [s for s in path.split("/") if s]
    if len(segs) < 2 or segs[0].lower() in _LISTING_SEGMENTS:
        return False
    return bool(_ARTICLE_RE.search(path))


# Return the services to search: the explicit provider names (resolved to services, unknown names
# dropped) or, by default, every vetted service in a searchable category.
def _resolve_providers(providers: tuple[str, ...] | None):
    if providers is not None:
        out = [service_for(n) for n in providers]
        return [s for s in out if s is not None]
    services: list = []
    for cat in _SEARCHABLE_CATEGORIES:
        services.extend(services_in(cat))
    return services


# Discover article URLs for one service via the clearnet SERP, rewritten to its onion mirror.
# Returns (provider_name, onion_url) pairs. Clearnet-only (no Tor); soft-fails to [].
async def _discover_for_service(svc, query: str, *, limit: int, serp_timeout: float
                                ) -> list[tuple[str, str]]:
    clearnet_host = _anchor_host(svc.clearnet_anchor)
    if not clearnet_host:
        return []

    # Current onion URL (cached or seeded); without a host there is nothing to rewrite to.
    onion = resolve_onion(svc)
    onion_parsed = urlparse(onion)
    onion_host, onion_scheme = onion_parsed.netloc, (onion_parsed.scheme or "http")
    if not onion_host:
        return []

    # Search the clearnet host through the SERP; discovery is best-effort and soft-fails.
    try:
        result = await run_serp_search(
            f"site:{clearnet_host} {query}", timeout_seconds=serp_timeout, source_limit=limit
        )
    except Exception:  # noqa: BLE001 — discovery is best-effort
        logger.debug("onion discovery SERP failed for site:%s", clearnet_host, exc_info=True)
        return []

    # Collect unique article URLs of this service from every engine, rewritten to the onion
    # mirror (it serves the same paths as the clearnet site), up to the limit.
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for engine in (result.get("engines") or {}).values():
        for source in engine.get("sources") or []:
            u = str(source.get("url") or "")
            if not u:
                continue
            p = urlparse(u)
            host = p.netloc.lower().removeprefix("www.")

            # Keep only results actually on this service's domain (site: is advisory, not law).
            if host != clearnet_host and not host.endswith("." + clearnet_host):
                continue
            if not _is_article_path(p.path):
                continue
            onion_url = p._replace(scheme=onion_scheme, netloc=onion_host).geturl()
            key = onion_url.split("#", 1)[0]
            if key in seen:
                continue
            seen.add(key)
            out.append((svc.name, onion_url))
            if len(out) >= limit:
                return out
    return out


# Fetch one result page over Tor and return its BM25-compressed content (None on any failure).
async def _scrape_one(url: str, query: str, provider: str, *, timeout: float, max_chars: int):
    from core.extract.content_processor import compress_to_budget
    from core.extract.page_normalizer import has_extractable_content, normalize_page
    from core.fetch.antibot import is_antibot
    from core.fetch.thread_pool import io_pool as _io_pool

    # Fetch over Tor with a hard outer bound slightly above the fetch timeout.
    try:
        r = await asyncio.wait_for(onion_fetch(url, timeout=timeout), timeout=timeout + 5)
    except asyncio.TimeoutError:
        logger.warning("onion fetch timed out provider=%s url=%r", provider, url)
        return None
    except Exception:  # noqa: BLE001 — one failed source must not sink the search
        logger.warning("onion fetch raised provider=%s url=%r", provider, url, exc_info=True)
        return None

    # Reject failed responses and anti-bot walls.
    if not r.ok or not r.text:
        logger.warning(
            "onion fetch failed provider=%s url=%r http_status=%d error=%s",
            provider,
            url,
            r.http_status,
            r.error or r.status,
        )
        return None
    if is_antibot(r.text):
        logger.warning("onion fetch returned an antibot wall provider=%s url=%r", provider, url)
        return None

    # Convert the HTML to markdown in the io pool (CPU-bound), bounded to 10 seconds.
    loop = asyncio.get_running_loop()
    try:
        md = await asyncio.wait_for(
            loop.run_in_executor(
                _io_pool,
                lambda: normalize_page(url, r.text, favor_recall=True),
            ),
            timeout=10,
        )
    except Exception:  # noqa: BLE001 — one failed extraction must not sink the search
        logger.warning("onion extraction failed provider=%s url=%r", provider, url, exc_info=True)
        return None

    # Reject pages without extractable content, then compress to the character budget.
    if not has_extractable_content(md):
        logger.warning(
            "onion extraction returned no page content provider=%s url=%r html_len=%d",
            provider,
            url,
            len(r.text),
        )
        return None
    content = compress_to_budget(md, query, max_chars)
    if not content.strip():
        logger.warning("onion compression returned no content provider=%s url=%r", provider, url)
        return None

    # The title is the first non-empty line of the content, stripped of markdown heading marks.
    title = next((ln.lstrip("# ").strip() for ln in (content or md).splitlines() if ln.strip()), url)
    return OnionResult(url=url, host=urlparse(url).netloc, title=title[:200],
                       content=content, provider=provider)


# Resolve the tor SOCKS once before the parallel scrapes (probes a running tor; we never spawn).
# Returns True if tor is usable. Runs the blocking probe in the io pool, bounded so it can't hang.
async def _warm_tor(budget: float) -> bool:
    from core.fetch.onion.tor_proxy import resolve_socks
    from core.fetch.thread_pool import io_pool as _io_pool

    loop = asyncio.get_running_loop()
    try:
        socks = await asyncio.wait_for(
            loop.run_in_executor(_io_pool, resolve_socks), timeout=budget
        )
    except Exception:  # noqa: BLE001
        # A failed or timed-out probe means tor is unusable, which is reported as False.
        return False
    return socks is not None


# Deep onion search: discover article URLs on vetted services via the hardened clearnet SERP
# (`site:<host> <query>`), rewrite the clearnet host to its onion mirror, scrape those pages over
# Tor and return the BM25-compressed content directly (no bare-snippet SERP). Discovery and
# scraping each run in parallel and per-link bounded, so one slow circuit cannot stall the batch.
# SERP-scoped discovery is used instead of each site's internal search because per-site search
# paths rot (SPA rewrites, 404/403, silent changes). The onion mirror serves the SAME paths as
# the clearnet site, so discovery stays clearnet (fast, reliable, like the resolver reaching the
# clearnet anchor) and only the article fetch rides Tor.
async def onion_search(
    query: str, *, limit: int = 5, per_link_timeout: float = 20.0,
    max_chars: int = 4000, concurrency: int = 4, providers: tuple[str, ...] | None = None,
) -> list[OnionResult]:
    services = _resolve_providers(providers)
    if not services:
        return []

    # Discovery is clearnet, so a normal SERP timeout suffices — no Tor cold-start budget needed.
    serp_timeout = min(max(per_link_timeout, 8.0), 12.0)

    # 1. Discover candidate article URLs per provider, in parallel (clearnet SERP, no Tor).
    discovered = await asyncio.gather(
        *(_discover_for_service(s, query, limit=limit, serp_timeout=serp_timeout) for s in services),
        return_exceptions=True,
    )
    candidates: list[tuple[str, str]] = []
    for d in discovered:
        if isinstance(d, list):
            candidates.extend(d)
    candidates = _round_robin(candidates)[:limit]
    if not candidates:
        logger.info("onion_search: no candidates for %r", query[:120])
        return []

    # 2. Warm tor once (cold start) before the parallel scrapes; bail if tor is unavailable.
    if not await _warm_tor(budget=max(per_link_timeout * 2.0, 45.0)):
        logger.info("onion_search: tor unavailable, skipping %d candidates", len(candidates))
        return []

    # 3. Scrape + compress the candidates in parallel, bounded by a semaphore.
    sem = asyncio.Semaphore(max(1, concurrency))

    # Scrape one candidate while holding a concurrency slot.
    async def _guarded(name: str, url: str):
        async with sem:
            return await _scrape_one(url, query, name, timeout=per_link_timeout, max_chars=max_chars)

    scraped = await asyncio.gather(*(_guarded(n, u) for n, u in candidates), return_exceptions=True)
    out = [r for r in scraped if isinstance(r, OnionResult)]
    logger.info("onion_search: %d/%d results for %r", len(out), len(candidates), query[:120])
    return out


# Round-robin interleave (name, url) pairs across providers for source diversity.
def _round_robin(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    by_name: dict[str, list[tuple[str, str]]] = {}
    for name, url in pairs:
        by_name.setdefault(name, []).append((name, url))
    out: list[tuple[str, str]] = []
    while any(by_name.values()):
        for name in list(by_name):
            if by_name[name]:
                out.append(by_name[name].pop(0))
    return out
