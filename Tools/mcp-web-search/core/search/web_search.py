# Copyright NEXTGGTECH. Elastic License 2.0.

"""Web-search orchestrator: stream → triage → bounded eager parse scheduler.

Wires the live SERP stream to incremental triage and an eager parse scheduler:

    search_stream → source event → triage.ingest (~0.1 ms)
        ├─ PARSE  → fetch/parse starts immediately (bounded slots)
        ├─ QUEUE  → held; consensus votes may upgrade it mid-stream
        └─ SKIP   → dropped

Engine selection is tier-based (low/medium/high) and gated by the per-engine
circuit breaker. Parsing of early winners overlaps the tail of slow engines, so
parse latency hides inside SERP latency.

Process discipline (non-negotiable): every parse task is tracked and cancelled
at the deadline. No fire-and-forget tasks. The persistent warm browser lives in
its own daemon, so a cancelled search never leaves a browser process behind.
"""

from __future__ import annotations

import asyncio
import contextlib
import itertools
import logging
import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from ..engines import (
    BraveParser,
    DuckDuckGoParser,
    GoogleParser,
    QwantParser,
    StartpageParser,
    YandexParser,
)
from core.engines.yandex import LegacyYandexParser
from core.cache.source_cache import canonicalize_url
from .health import EngineHealthTracker, get_health_tracker
from .quality import markdown_meta_date
from .query_locales import analyze_query
from .serp_api import SerpApi, _get_transport
from .triage import TriageAction, TriageSession

logger = logging.getLogger("services.web_search")
trace_logger = logging.getLogger("trace.web_search")

# Engines take the parser's max; final cut happens after triage, not at the engine
# (the parser scans the whole page either way — trimming early just wastes results).
_SOURCE_LIMIT = 20

# A domain the runtime profile store has learned to fetch slower than this is left
# snippet-only in search (still readable via read_page). Tighter than read_page's own
# avoid bar — a wide search cannot afford a multi-second page on the hot path.
_INLINE_PARSE_SKIP_MS = 6_000.0

# These hosts are always snippet-only during web_search. Exact hostname/subdomain
# matching avoids path heuristics and cannot match an unrelated domain such as
# not-reddit.com. Their dedicated read_page behavior, if any, remains unchanged.
_INLINE_PARSE_BLOCKED_DOMAINS = frozenset({"reddit.com", "redd.it"})

# Per-link timeout for onion fetches during a web_search — tighter than read_page's onion
# path (which uses the full tor.fetch_timeout). A search must stay responsive even when a
# Tor circuit is slow; a slow onion link is dropped rather than allowed to stall the batch.
_ONION_WEB_SEARCH_LINK_TIMEOUT = 20.0

# Per-provider cap for the hosted supplement layer. Low by design — hosted credits cost
# money and diversity beats depth; the scrape engines carry recall.
_HOSTED_MAX_RESULTS = 5

# Sentinel marking one merged sub-stream as drained.
_STREAM_DONE = object()


# Interleave several event streams into one, yielding items as any sub-stream produces
# them and finishing only when all are drained. Cancels stragglers on exit.
async def _merge_streams(*streams):
    queue: asyncio.Queue = asyncio.Queue()

    async def drain(stream) -> None:
        try:
            async for item in stream:
                await queue.put(item)
        finally:
            await queue.put(_STREAM_DONE)

    tasks = [asyncio.create_task(drain(s)) for s in streams]
    remaining = len(tasks)
    try:
        while remaining:
            item = await queue.get()
            if item is _STREAM_DONE:
                remaining -= 1
                continue
            yield item
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


# Whether a source may be parsed inline during a search, or should stay snippet-only.
# Two reasons to skip: a read_page-only custom-domain handler (reddit/x/ebay/youtube —
# browser/slow APIs), or a domain the runtime store has learned is too slow to parse.
# Either way the source still ranks and is returned; only its page is left for read_page.
def _inline_parse_allowed(url: str) -> bool:
    try:
        host = (urlsplit(str(url or "")).hostname or "").lower().rstrip(".")
    except ValueError:
        host = ""
    if any(host == domain or host.endswith(f".{domain}") for domain in _INLINE_PARSE_BLOCKED_DOMAINS):
        return False
    try:
        import custom_domains

        if custom_domains.is_read_page_only(url):
            return False
    except Exception:  # noqa: BLE001 — a handler lookup must never sink a search
        pass
    try:
        from core.profiles import domain_of, get_runtime_profiles

        hint = get_runtime_profiles().best_method(domain_of(url))
    except Exception:  # noqa: BLE001 — a profile lookup must never sink a search
        return True
    return not (hint and hint.expected_fetch_ms > _INLINE_PARSE_SKIP_MS)


# Collapse same-host near-duplicates from a score-sorted list. Triage dedups exact
# URLs, but engines also return slug/redirect/anchor variants of one page (e.g. two
# programming-helper.com URLs for the same article, or a "Redirecting to…" stub). Keep
# the highest-scored variant per host whose title is near-identical to an already-kept
# one; distinct pages on the same host (e.g. two different kubernetes.io docs) survive.
def _dedupe_near_duplicates(ranked: list[_Source]) -> list[_Source]:
    import re as _re

    def toks(t: str) -> frozenset[str]:
        return frozenset(_re.findall(r"\w+", (t or "").lower()))

    kept: list[_Source] = []
    seen_by_host: dict[str, list[frozenset[str]]] = {}
    for s in ranked:
        t = toks(s.title)
        prior = seen_by_host.get(s.host, [])
        is_dup = False
        for pt in prior:
            if not t or not pt:
                continue
            # Containment: the shorter title's tokens almost fully inside the other →
            # same article (handles "X" vs "X - Site Name" and redirect stubs).
            overlap = len(t & pt) / max(1, min(len(t), len(pt)))
            if overlap >= 0.85:
                is_dup = True
                break
        if is_dup:
            continue
        kept.append(s)
        seen_by_host.setdefault(s.host, []).append(t)
    return kept


# Resolve a direct PDF URL for a result (already a PDF, or an arXiv abs → pdf link).
# Ported from the legacy _infer_pdf_url so the model still gets direct PDF links.
def _infer_pdf_url(url: str) -> str:
    from core.extract.pdf_extractor import looks_like_pdf_url

    u = (url or "").strip()
    if not u:
        return ""
    if looks_like_pdf_url(u):
        return u
    if "arxiv.org/abs/" in u:
        return u.replace("/abs/", "/pdf/", 1)
    return ""


# Per-effort contracts. Each tier is a distinct deal, not just a bigger budget:
#
#   low    — SERP-only. Pure HTTP, NO page parse, no browser, no model. Sub-2s.
#   medium — HTTP page parsing only (no browser, no model). Hard-bounded to ~6-8s total:
#            tight per-page parse cap and a short overall deadline.
#   high   — deeper HTTP parse + limited warm-browser escalation.
#
# parse_budget=0 means SERP-only. parse_timeout is the hard per-page cap (6s ceiling —
# a wide search cannot afford a slow page; slow domains are left snippet-only anyway).
@dataclass(frozen=True, slots=True)
class EffortProfile:
    name: str
    parse_budget: int  # total pages parsed per search
    parse_concurrency: int  # parallel parse slots
    parse_reserve: int  # slots held back until the stream ends (anti-greed)
    parse_timeout: float  # hard cap per page parse
    parse_max_chars: int
    deadline: float  # hard cap for the whole search
    max_results: int  # sources returned after ranking
    allow_browser: bool  # high only: limited warm-browser escalation


EFFORT_PROFILES: dict[str, EffortProfile] = {
    "low": EffortProfile("low", 0, 0, 0, 0.0, 0, 8.0, 8, False),
    "medium": EffortProfile("medium", 3, 3, 1, 6.0, 8_000, 8.0, 10, False),
    "high": EffortProfile("high", 8, 4, 1, 8.0, 20_000, 22.0, 16, True),
}


# Process-local monotonic namespace. Four hex digits provide 65,536 compact IDs; the
# representation grows instead of wrapping once that range is exhausted.
_SEARCH_ID_SEQUENCE = itertools.count()


# Opaque per-search id; seeds the citation handles the model is told to cite with.
def _make_search_id() -> str:
    return f"srch_{next(_SEARCH_ID_SEQUENCE):04x}"


# Stable citation handle for a source's rank within a search. New searches use at least a
# four-character namespace; a longer monotonic search id is never truncated or wrapped.
def _citation_id(search_id: str, rank: int) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", (search_id or "").lower()).removeprefix("srch")
    namespace = (compact or "src").ljust(4, "0")
    return f"c{namespace}-{rank}"


# Build the model-visible text block the chat bridge reads from result["model_context"]:
# citation handles + Title/Domain/URL/(parsed Content | search Preview), capped to a
# total budget. Operates on the serialised source dicts so the live and seen-suppressed
# paths stay consistent. Empty results return the explicit "No results found" sentinel the
# model and UI both understand (mirrors API/mcp.py::_serialize_tool_result).
def _build_model_context(
    query: str, sources: list[dict[str, Any]], *, total_budget: int, per_source_chars: int,
) -> str:
    if not sources:
        return f"No results found for: {query}"
    first_handle = sources[0].get("id") or "id-1"
    lines: list[str] = [
        f"Search results for: {query}",
        "",
        f"Cite sources only with the exact handles below, e.g. [{first_handle}]. "
        "Put the handle right after the claim it supports; do not invent or renumber handles.",
        "",
        "Sources:",
    ]
    total = len("\n".join(lines))
    for s in sources:
        excerpt = str(s.get("markdown") or s.get("snippet") or "").strip()
        if per_source_chars and len(excerpt) > per_source_chars:
            excerpt = excerpt[:per_source_chars].rstrip() + " …"
        label = "Content" if (s.get("parsed_ok") and s.get("markdown")) else "Preview"
        block = [
            f"Citation handle: [{s.get('id', '')}]",
            f"Title: {s.get('title', '')}",
            f"Domain: {s.get('host', '')}",
            f"URL: {s.get('url', '')}",
        ]
        if excerpt:
            block.append(f"{label}: {excerpt}")
        block.append("")
        btext = "\n".join(block)
        if total_budget and total + len(btext) > total_budget:
            lines.append("[...additional sources omitted: context budget reached]")
            break
        lines.extend(block)
        total += len(btext)
    return "\n".join(lines).strip()


# Adapt one shopping product into a citable source dict (price/seller/rating in the
# snippet so it shows in model_context; structured fields kept for richer UI).
def _shopping_product_dict(product: Any, *, citation_id: str, rank: int) -> dict[str, Any]:
    price = str(getattr(product, "price_text", "") or "").strip()
    price_value = getattr(product, "price_value", None)
    currency = str(getattr(product, "currency", "") or "").strip().upper()
    if not price and price_value is not None:
        price = f"{price_value:g} {currency}".strip()
    converted_prices = [
        dict(item)
        for item in (getattr(product, "converted_prices", []) or [])
        if isinstance(item, dict)
    ]
    converted_summary = " / ".join(
        str(item.get("price_text") or "").strip()
        for item in converted_prices[:3]
        if str(item.get("price_text") or "").strip()
    )
    rating = getattr(product, "rating", None)
    bits = [b for b in (price, (f"≈ {converted_summary}" if converted_summary else ""),
                        str(getattr(product, "seller", "") or ""),
                        (f"rating {rating}" if rating else "")) if b]
    source = str(getattr(product, "source", "") or "shopping")
    return {
        "id": citation_id,
        "rank": rank,
        "kind": "shopping",
        "url": str(getattr(product, "url", "") or ""),
        "host": str(getattr(product, "source_domain", "") or source),
        "title": str(getattr(product, "title", "") or ""),
        "snippet": " · ".join(bits),
        "engine": f"shopping:{source}",
        "score": float(getattr(product, "confidence", 0.0) or 0.0),
        "consensus_families": [],
        "price": {
            "value": price_value,
            "currency": currency,
            "price_text": price,
        },
        "price_text": price,
        "price_value": price_value,
        "currency": currency,
        "converted_prices": converted_prices,
        "display_currency_count": (1 + len(converted_prices)) if currency else len(converted_prices),
        "rating": rating,
        "seller": str(getattr(product, "seller", "") or ""),
        "availability": str(getattr(product, "availability", "") or ""),
        "source": source,
        "favicon_url": str(getattr(product, "favicon_url", "") or ""),
    }


# Adapt one academic paper into a citable source dict. Authors/year/venue/DOI go into the
# snippet (model_context-visible) and the abstract feeds the markdown body so the model can
# cite without a page fetch; structured fields are kept for richer UI.
def _academic_paper_dict(paper: Any, *, citation_id: str, rank: int) -> dict[str, Any]:
    authors = list(getattr(paper, "authors", []) or [])
    author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
    year = getattr(paper, "year", None)
    venue = str(getattr(paper, "venue", "") or "")
    doi = str(getattr(paper, "doi", "") or "")
    cites = getattr(paper, "citations", None)
    head = " · ".join(b for b in (
        author_str, str(year) if year else "", venue,
        f"{cites} citations" if cites else "",
    ) if b)
    abstract = str(getattr(paper, "abstract", "") or "")
    source = str(getattr(paper, "source", "") or "academic")
    return {
        "id": citation_id,
        "rank": rank,
        "kind": "academic",
        "url": str(getattr(paper, "url", "") or ""),
        "host": str(getattr(paper, "source_domain", "") or source),
        "title": str(getattr(paper, "title", "") or ""),
        "snippet": (f"{head}\n{abstract}" if head else abstract)[:600],
        "engine": f"academic:{source}",
        "score": float(getattr(paper, "confidence", 0.0) or 0.0),
        "consensus_families": [],
        "authors": authors,
        "year": year,
        "venue": venue,
        "doi": doi,
        "citations": cites,
        "open_access": bool(getattr(paper, "open_access", False)),
        "abstract": abstract,
        "source": source,
        **({"pdf_url": pdf} if (pdf := str(getattr(paper, "pdf_url", "") or "")) else {}),
    }


# UI metadata block (source chips) the chat bridge surfaces alongside model_context.
def _build_ui(sources: list[dict[str, Any]]) -> dict[str, Any]:
    chips = [
        {
            "rank": s.get("rank"),
            "id": s.get("id"),
            "url": s.get("url"),
            "domain": s.get("host"),
            "favicon_url": f"https://icons.duckduckgo.com/ip3/{s.get('host')}.ico" if s.get("host") else "",
            "parsed": bool(s.get("parsed_ok")),
        }
        for s in sources
    ]
    return {
        "kind": "web_search",
        "status": "done" if chips else "empty",
        "result_count": len(chips),
        "sources": chips,
    }


# Pick engines for a tier, honoring the per-engine config switch and the circuit breaker.
# Config is policy (an engine the user turned off never enters a tier — Yandex engines default
# off, one flip re-enables them); the breaker is health. Never returns an empty list: low's
# leads fall back to Startpage, then to the emergency buffer (the legacy Yandex widget,
# buffer_engine.enabled), and as a last resort the first ENABLED lead is forced even
# through an open breaker. `enabled`/`buffer` override the config maps (tests).
def select_engines(
    effort: str,
    tracker: EngineHealthTracker,
    *,
    enabled: dict[str, bool] | None = None,
    buffer: bool | None = None,
) -> list[type]:
    from core.config import load_search_config

    cfg = load_search_config()
    if enabled is None:
        enabled = cfg.engines.as_map()
    if buffer is None:
        buffer = cfg.buffer_engine.enabled

    def on(parser: type) -> bool:
        return bool(enabled.get(parser.name, True))

    selected: list[type] = []

    # low core: DDG (lead) + native Yandex (opt-in), breaker-gated. The legacy
    # Yandex widget is NOT part of rotation — it is the emergency buffer below.
    if on(YandexParser) and tracker.allow(YandexParser.name):
        selected.append(YandexParser)
    if on(DuckDuckGoParser) and tracker.allow(DuckDuckGoParser.name):
        selected.append(DuckDuckGoParser)
    if not selected:
        # Leads are cooling down or switched off — pull the reserve.
        if on(StartpageParser) and tracker.allow(StartpageParser.name):
            selected.append(StartpageParser)
        elif buffer and tracker.allow(LegacyYandexParser.name):
            # Emergency buffer: every regular engine is off in config or blocked by
            # its breaker — the legacy Yandex widget takes the load so the search
            # still returns results instead of none.
            selected.append(LegacyYandexParser)
        else:
            # Forced: low must never be empty. First enabled lead wins; an all-off
            # config is a misconfiguration and still gets DDG rather than nothing.
            forced = next(
                (p for p in (DuckDuckGoParser, StartpageParser, YandexParser, GoogleParser) if on(p)),
                DuckDuckGoParser,
            )
            selected.append(forced)

    if effort == "low":
        return selected

    # medium: the google-family slot. Google and Startpage serve the SAME index; Startpage
    # is more scrape-stable, so it is Google's standby. The substitution is keyed on Google
    # being *solidly healthy* (breaker CLOSED), not merely on it being pickable this instant:
    #   • Google CLOSED           → Google alone (Startpage stays a warm standby).
    #   • Google OPEN (cooling)   → Startpage substitutes for the whole cooldown.
    #   • Google HALF_OPEN probe  → Google probes AND Startpage rides along, so a failed
    #                               probe (a still-blocked IP) doesn't cost the search its
    #                               google-family result. Same family → triage dedups the
    #                               overlap into one consensus vote, never a double count.
    # This is the "remember Google is bad" fix: while Google is unproven, Startpage is a
    # first-class member of the tier, not a fill-in that vanishes the moment a probe fires.
    google_fired = on(GoogleParser) and tracker.allow(GoogleParser.name)
    if google_fired:
        selected.append(GoogleParser)
    if (
        StartpageParser not in selected
        and on(StartpageParser)
        and (not google_fired or not tracker.is_healthy(GoogleParser.name))
        and tracker.allow(StartpageParser.name)
    ):
        selected.append(StartpageParser)
    # …plus Qwant as a health-gated helper.
    if on(QwantParser) and tracker.allow(QwantParser.name):
        selected.append(QwantParser)

    if effort == "medium":
        return selected

    # high: Brave (rate-governed by its breaker).
    if on(BraveParser) and tracker.allow(BraveParser.name):
        selected.append(BraveParser)
    return selected


# One aggregated source row under construction.
@dataclass(slots=True)
class _Source:
    url: str
    host: str
    title: str
    snippet: str
    engine: str  # first engine that surfaced it
    rank: int
    score: float
    families: list[str]
    parsed_markdown: str = ""
    parsed_ok: bool = False
    parse_ms: float = 0.0


# Orchestrates one search: stream → triage → bounded eager parsing → ranking.
class WebSearchService:

    def __init__(
        self,
        *,
        tracker: EngineHealthTracker | None = None,
        read_page=None,
    ) -> None:
        self._tracker = tracker or get_health_tracker()
        # Injectable for tests; default import is deferred so SERP-only paths
        # never pay for the read service's heavy imports.
        self._read_page = read_page

    # Resolve the page reader lazily.
    def _reader(self):
        if self._read_page is None:
            from core.read.service import run_read_page

            self._read_page = run_read_page
        return self._read_page

    # Parse one URL under its own hard timeout; never raises. The search query is passed
    # as the compaction focus so chunk selection ranks against the actual query.
    async def _parse_one(
        self, source: _Source, profile: EffortProfile, *, query: str = ""
    ) -> None:
        started = time.perf_counter()
        try:
            async with asyncio.timeout(profile.parse_timeout):
                markdown = await self._reader()(
                    source.url,
                    timeout=profile.parse_timeout,
                    max_chars=profile.parse_max_chars,
                    focus=query,
                    # Only the high profile may escalate blocked or weak pages to the
                    # warm browser; low/medium remain cheap and HTTP-only.
                    allow_browser=profile.allow_browser,
                )
            source.parsed_ok = bool(markdown) and not markdown.startswith("Error:")
            # A terminal domain handler may intentionally reject a page (for example,
            # Telegram group roots have no public HTTP message feed).  Keep the SERP
            # snippet as the preview instead of serialising the parser's diagnostic as
            # if it were useful source content.
            source.parsed_markdown = (markdown or "") if source.parsed_ok else ""
        except (TimeoutError, asyncio.CancelledError):
            source.parsed_markdown = ""
            source.parsed_ok = False
            raise
        except Exception as exc:  # noqa: BLE001 — a failed parse must not sink the search
            logger.debug("parse failed for %s: %s", source.url, exc)
            source.parsed_ok = False
        finally:
            source.parse_ms = (time.perf_counter() - started) * 1000

    # Build the hosted supplement stream when API keys are configured; None otherwise
    # (no keys → pure scrape, baseline unchanged). Imports are deferred so the SERP-only
    # and key-less paths never pay for httpx provider code.
    def _hosted_stream(
        self,
        query: str,
        *,
        region: str,
        deadline: float,
        query_text: str = "",
        operators: dict[str, Any] | None = None,
        timelimit: str | None = None,
    ):
        try:
            from core.search.hosted_providers import available_providers
            from core.search.hosted_stream import hosted_search_stream
        except Exception as exc:  # noqa: BLE001 — hosted layer is optional
            logger.debug("hosted layer unavailable: %s", exc)
            return None
        if not available_providers():
            return None
        return hosted_search_stream(
            query,
            region=region,
            max_results=_HOSTED_MAX_RESULTS,
            deadline_seconds=deadline,
            query_text=query_text,
            operators=operators,
            timelimit=timelimit,
        )

    # Run one full search. Returns the aggregated, ranked payload.
    async def search(
        self,
        query: str,
        *,
        effort: str = "low",
        region: str = "",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        shopping: bool = False,
        academic: bool = False,
        onion: bool = False,
        vertical_only: bool = False,
        dates_resolved: bool = False,
        query_text: str = "",
        operators: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        profile = EFFORT_PROFILES.get(effort, EFFORT_PROFILES["low"])
        started = time.perf_counter()

        # Fold the query's own date intent in: a trailing/comma year or freshness word
        # derives a timelimit (stricter of it and any explicit one) and the year token is
        # stripped so it doesn't skew the lexical match. Governed by the `query` config.
        from core.config import load_search_config
        from core.search.query_dates import resolve_query_dates

        if not dates_resolved:
            query, timelimit = resolve_query_dates(query, load_search_config().query, timelimit)

        # Region routing (§3.2): explicit region wins; otherwise the query's own
        # script/stop-word locale (query_locales, 20 profiles) picks the home region —
        # Latin queries also resolve via distinctive diacritics and stop-word hints,
        # so French/Spanish/Turkish/... search their home region instead of us-en.
        analysis = analyze_query(query)
        language = analysis.language
        if not region:
            region = analysis.region

        # Advanced vertical fields are routes, not boolean supplements. A dedicated
        # shopping query must return products with prices instead of spending the
        # deadline on a normal web SERP and timing out the shopping merge afterward.
        if vertical_only and shopping:
            search_id = _make_search_id()
            source_dicts = await self._shopping_sources(
                query, profile, language, search_id, start_rank=1
            )
            from core.config import load_search_config

            search_cfg = load_search_config().search
            model_context = _build_model_context(
                query,
                source_dicts,
                total_budget=int(search_cfg.total_context_budget or 0),
                per_source_chars=int(search_cfg.preview_max_chars or 0),
            )
            return {
                "query": query,
                "search_id": search_id,
                "effort": profile.name,
                "shopping": True,
                "academic": False,
                "onion": False,
                "language": language,
                "region": region,
                "engines_used": ["shopping"],
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
                "model_context": model_context,
                "sources": source_dicts,
                "ui": _build_ui(source_dicts),
                "engines": {
                    "shopping": {
                        "status": "success" if source_dicts else "empty",
                        "count": len(source_dicts),
                    }
                },
                "health": self._tracker.snapshot(),
            }

        engines = select_engines(profile.name, self._tracker)
        from core.query.provider_compiler import compile_provider_query

        provider_queries = {
            engine.name: compile_provider_query(
                query_text or query,
                operators,
                engine.name,
                fallback_query=query,
                timelimit=timelimit,
            )
            for engine in engines
        }
        logger.info(
            "web_search.start effort=%s region=%s language=%s engines=%s query=%r",
            profile.name, region, language, [e.name for e in engines], query[:160],
        )
        api = SerpApi(
            transport=_get_transport(8.0),
            timeout_seconds=8.0,
            source_limit=_SOURCE_LIMIT,
            engines=tuple(engines),
        )

        # Learned domain trust: one snapshot per search (single DB read), so triage
        # itself stays I/O-free. A missing/failed store degrades to a neutral session.
        try:
            from core.profiles import get_runtime_profiles

            reputation = get_runtime_profiles().reputation_snapshot()
        except Exception as exc:  # noqa: BLE001 — reputation must never sink a search
            logger.debug("reputation snapshot unavailable: %s", exc)
            reputation = None

        triage = TriageSession(query, reputation=reputation)
        sources: dict[str, _Source] = {}
        queue: list[str] = []  # urls in QUEUE state
        engine_payloads: dict[str, dict[str, Any]] = {}
        parse_tasks: dict[str, asyncio.Task[None]] = {}
        parse_started_count = 0
        slots = asyncio.Semaphore(max(1, profile.parse_concurrency)) if profile.parse_budget else None
        budget_during_stream = max(0, profile.parse_budget - profile.parse_reserve)

        # Spawn a tracked parse task for a triaged source.
        def spawn_parse(url: str) -> None:
            nonlocal parse_started_count
            if profile.parse_budget == 0 or url in parse_tasks:
                return
            if parse_started_count >= profile.parse_budget:
                return
            source = sources[url]
            # Snippet-only sources (read_page-only handler / learned-slow domain) never
            # spend a parse slot — they stay in the ranked output without a page fetch.
            if not _inline_parse_allowed(url):
                trace_logger.info("parse.skip host=%s url=%r (snippet-only)", source.host, url)
                return
            parse_started_count += 1

            async def run() -> None:
                assert slots is not None
                async with slots:
                    await self._parse_one(source, profile, query=query)

            parse_tasks[url] = asyncio.create_task(run(), name=f"parse:{source.host}")

        # Baseline scrape stream; hosted providers (content + SERP) join the same triage
        # as a supplement when keys exist. Hosted is gated to effort tiers that parse
        # (low stays SERP-only and pays no hosted credits).
        event_stream = api.search_stream(
            query, region=region, safesearch=safesearch, timelimit=timelimit,
            engine_queries={name: item.query for name, item in provider_queries.items()},
            engine_timelimits={name: item.timelimit for name, item in provider_queries.items()},
            engine_omitted_operators={
                name: item.omitted_operators for name, item in provider_queries.items()
            },
            # Captcha-retry budget (native Yandex): only the high tier pays for identity
            # re-rolls — a BLOCKED SERP is usually a sticky cookie branch, and a
            # cookieless retry recovers it cheaply (SerpApi._run_engine).
            engine_blocked_retries=(
                {YandexParser.name: 3}
                if profile.name == "high" and YandexParser in engines
                else None
            ),
            deadline_seconds=profile.deadline * 0.8,
        )
        if profile.parse_budget:
            hosted = self._hosted_stream(
                query,
                region=region,
                deadline=profile.deadline * 0.8,
                query_text=query_text,
                operators=operators,
                timelimit=timelimit,
            )
            if hosted is not None:
                event_stream = _merge_streams(event_stream, hosted)

        # Fold a triage re-score of an already-seen source (a consensus vote, or a full
        # revisit that may also improve the positional view) back into its live state.
        def apply_rescore(url: str, family: str, decision) -> None:
            source = sources.get(url)
            if source is not None and family not in source.families:
                source.families.append(family)
            if source is not None:
                source.score = triage.score_of(url)
            if (
                decision is not None
                and decision.upgraded
                and parse_started_count < budget_during_stream
            ):
                with contextlib.suppress(ValueError):
                    queue.remove(url)
                spawn_parse(url)

        def apply_vote(url: str, family: str) -> None:
            apply_rescore(url, family, triage.ingest_vote(provider_family=family, url=url))

        try:
            async with asyncio.timeout(profile.deadline):
                async for event in event_stream:
                    kind = event["type"]
                    if kind == "source":
                        raw_url = event["url"]["url"]
                        # Dedup/consensus key is the canonical URL, so the same page from
                        # another family (http/https, www, trailing slash, tracking params)
                        # merges into one source and casts a vote instead of a duplicate.
                        url = canonicalize_url(raw_url)
                        # The same URL arriving from another stream/provider is a
                        # consensus vote plus a chance at a better positional view
                        # (rank/snippet from this engine), never an overwrite.
                        if url in sources:
                            apply_rescore(
                                url,
                                event["provider_family"],
                                triage.ingest_revisit(
                                    engine=event["engine"],
                                    provider_family=event["provider_family"],
                                    rank=event["rank"],
                                    url=url,
                                    title=event["serp"]["title"],
                                    snippet=event["serp"]["snippet"],
                                ),
                            )
                            continue
                        decision = triage.ingest_source(
                            engine=event["engine"],
                            provider_family=event["provider_family"],
                            rank=event["rank"],
                            url=url,
                            title=event["serp"]["title"],
                            snippet=event["serp"]["snippet"],
                        )
                        if decision.action == TriageAction.SKIP:
                            continue
                        sources[url] = _Source(
                            url=raw_url,
                            host=event["url"]["host"],
                            title=event["serp"]["title"],
                            snippet=event["serp"]["snippet"],
                            engine=event["engine"],
                            rank=event["rank"],
                            score=decision.score,
                            families=[event["provider_family"]],
                        )
                        trace_logger.info(
                            "source engine=%s family=%s rank=%d action=%s score=%.3f url=%r",
                            event["engine"], event["provider_family"], event["rank"],
                            decision.action.name, decision.score, url,
                        )
                        if decision.action == TriageAction.PARSE and parse_started_count < budget_during_stream:
                            spawn_parse(url)
                        else:
                            queue.append(url)
                    elif kind == "vote":
                        apply_vote(canonicalize_url(event["url"]["url"]), event["provider_family"])
                    elif kind == "engine":
                        payload = event["payload"]
                        engine_payloads[payload["engine"]] = payload
                        self._tracker.record(
                            payload["engine"],
                            status=payload["status"],
                            fetch_ms=payload["fetch_ms"],
                            results=len(payload["sources"]),
                        )

                # Stream done: spend remaining budget (incl. reserve) on the best
                # queued sources at full-picture scores.
                if profile.parse_budget:
                    queue.sort(key=triage.score_of, reverse=True)
                    for url in queue:
                        if parse_started_count >= profile.parse_budget:
                            break
                        spawn_parse(url)
                    if parse_tasks:
                        await asyncio.gather(*parse_tasks.values(), return_exceptions=True)
        except TimeoutError:
            logger.warning("web_search deadline %.0fs hit for query=%r", profile.deadline, query)
        finally:
            # Hard rule: no task survives the search. Cancellation propagates into
            # read_page; the warm browser runs out-of-process in its own daemon.
            pending = [task for task in parse_tasks.values() if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        # A parsed page's declared date beats the snippet estimate: swap it into the
        # triage score before the final sort, so freshness ranks on what the page
        # actually says about itself, not on what the engine guessed.
        for url, source in sources.items():
            if source.parsed_ok and source.parsed_markdown:
                if page_date := markdown_meta_date(source.parsed_markdown):
                    triage.apply_page_date(url, page_date)
                    source.score = triage.score_of(url)

        ranked = _dedupe_near_duplicates(
            sorted(sources.values(), key=lambda s: s.score, reverse=True)
        )
        top = ranked[: profile.max_results]
        parsed_ok = sum(1 for s in top if s.parsed_ok)
        logger.info(
            "web_search.done effort=%s sources=%d parsed=%d/%d elapsed_ms=%.0f query=%r",
            profile.name, len(top), parsed_ok, parse_started_count,
            (time.perf_counter() - started) * 1000, query[:160],
        )

        # The chat bridge (API/mcp.py) reads result["model_context"] as the model-visible
        # text and pairs it with "ui"/"sources" for chips; without model_context it cannot
        # surface results at all. Build the serialised sources first, then derive both from
        # them so this and the seen-suppressed path in run_web_search stay consistent.
        from core.config import load_search_config

        search_cfg = load_search_config().search
        search_id = _make_search_id()
        source_dicts: list[dict[str, Any]] = [
            {
                "id": _citation_id(search_id, rank),
                "url": s.url,
                "host": s.host,
                "title": s.title,
                "snippet": s.snippet,
                "engine": s.engine,
                "rank": rank,
                "score": round(s.score, 4),
                "consensus_families": s.families,
                **({"pdf_url": pdf} if (pdf := _infer_pdf_url(s.url)) else {}),
                **(
                    {
                        "parsed_ok": s.parsed_ok,
                        "parse_ms": round(s.parse_ms, 1),
                        "markdown": s.parsed_markdown,
                    }
                    if s.parse_ms or s.parsed_markdown
                    else {}
                ),
            }
            for rank, s in enumerate(top, 1)
        ]

        # Vertical merges run AFTER the SERP deadline block above, so without their own
        # cap they were unbounded — a slow scholarly/shopping API stretched the search
        # far past its declared budget with no timeout owning it (seen live: high with
        # academic=true ran minutes while the deadline had long "fired"). Half the
        # profile deadline is plenty for a supplement and bounds the worst-case total
        # at ~1.5x the declared deadline. A timed-out vertical is dropped, not fatal —
        # same soft-failure contract the verticals already had.
        async def merge_vertical(name: str, coro) -> list[dict[str, Any]]:
            try:
                return await asyncio.wait_for(coro, timeout=profile.deadline * 0.5)
            except (TimeoutError, asyncio.TimeoutError):
                logger.warning("%s merge timed out after %.1fs for query=%r",
                               name, profile.deadline * 0.5, query[:120])
                return []

        # Shopping opt-in: merge structured product results (price/seller/rating) in as
        # additional citable sources, continuing the citation-handle sequence. Off by
        # default; failure is soft (web results still stand).
        if shopping:
            source_dicts.extend(
                await merge_vertical("shopping", self._shopping_sources(
                    query, profile, language, search_id, start_rank=len(source_dicts) + 1
                ))
            )

        # Academic opt-in: merge structured scholarly results (paper/authors/DOI/abstract)
        # in as additional citable sources. Off by default; failure is soft.
        if academic:
            source_dicts.extend(
                await merge_vertical("academic", self._academic_sources(
                    query, profile, search_id, start_rank=len(source_dicts) + 1
                ))
            )

        # Onion opt-in: surface vetted censorship-resistant onion sources over Tor as
        # additional citable sources. Off by default; failure is soft (web results stand).
        if onion:
            source_dicts.extend(
                await merge_vertical("onion", self._onion_sources(
                    query, profile, search_id, start_rank=len(source_dicts) + 1
                ))
            )

        model_context = _build_model_context(
            query, source_dicts,
            total_budget=int(search_cfg.total_context_budget or 0),
            per_source_chars=int(search_cfg.preview_max_chars or 0),
        )
        return {
            "query": query,
            "search_id": search_id,
            "effort": profile.name,
            "shopping": shopping,
            "academic": academic,
            "onion": onion,
            "language": language,
            "region": region,
            "engines_used": [engine.name for engine in engines],
            "provider_queries": {
                name: {
                    "query": item.query,
                    "timelimit": item.timelimit,
                    "omitted_operators": list(item.omitted_operators),
                }
                for name, item in provider_queries.items()
            },
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "model_context": model_context,
            "sources": source_dicts,
            "ui": _build_ui(source_dicts),
            "engines": engine_payloads,
            "health": self._tracker.snapshot(),
        }

    # Run the shopping engine and adapt products into citable source dicts (continuing the
    # citation-handle sequence). Soft-fails to [] so a shopping error never sinks a search.
    async def _shopping_sources(
        self, query: str, profile: EffortProfile, language: str, search_id: str, *, start_rank: int
    ) -> list[dict[str, Any]]:
        try:
            from core.fetch.shopping import search_shopping

            result = await search_shopping(
                query, effort=profile.name, limit=profile.max_results, language=language
            )
        except Exception:  # noqa: BLE001 — shopping is supplemental; never fail the search
            logger.warning("shopping search failed for %r", query[:160], exc_info=True)
            return []
        dicts = [
            _shopping_product_dict(p, citation_id=_citation_id(search_id, start_rank + i),
                                   rank=start_rank + i)
            for i, p in enumerate(result.products)
        ]
        if dicts:
            logger.info("shopping.merged products=%d query=%r", len(dicts), query[:120])
        return dicts

    # Run the academic engine and adapt papers into citable source dicts (continuing the
    # citation-handle sequence). Soft-fails to [] so a scholarly error never sinks a search.
    async def _academic_sources(
        self, query: str, profile: EffortProfile, search_id: str, *, start_rank: int
    ) -> list[dict[str, Any]]:
        try:
            from core.fetch.academic import search_academic

            result = await search_academic(
                query, effort=profile.name, limit=profile.max_results
            )
        except Exception:  # noqa: BLE001 — academic is supplemental; never fail the search
            logger.warning("academic search failed for %r", query[:160], exc_info=True)
            return []
        dicts = [
            _academic_paper_dict(p, citation_id=_citation_id(search_id, start_rank + i),
                                 rank=start_rank + i)
            for i, p in enumerate(result.papers)
        ]
        if dicts:
            logger.info("academic.merged papers=%d query=%r", len(dicts), query[:120])
        return dicts

    # Deep onion search: per-site search over Tor → parallel scrape of top results → BM25-
    # compressed content returned directly as citable sources (not bare snippets). Per-link
    # timeout is kept tighter here than read_page's onion path — a web_search must stay snappy
    # even when an onion circuit is slow. Soft-fails to [].
    async def _onion_sources(
        self, query: str, profile: EffortProfile, search_id: str, *, start_rank: int
    ) -> list[dict[str, Any]]:
        try:
            from core.fetch.onion.search import onion_search

            results = await onion_search(
                query, limit=profile.max_results,
                per_link_timeout=_ONION_WEB_SEARCH_LINK_TIMEOUT, max_chars=4000,
            )
        except Exception:  # noqa: BLE001 — onion is supplemental; never fail the search
            logger.warning("onion search failed for %r", query[:160], exc_info=True)
            return []
        out: list[dict[str, Any]] = []
        for i, r in enumerate(results):
            rank = start_rank + i
            out.append({
                "id": _citation_id(search_id, rank),
                "url": r.url,
                "host": r.host,
                "title": r.title,
                "snippet": (r.content or "")[:300],
                "engine": f"onion:{r.provider}",
                "rank": rank,
                "score": 0.6,
                "consensus_families": [f"onion:{r.provider}"],
                "parsed_ok": True,
                "parse_ms": 0.0,
                "markdown": r.content,
            })
        if out:
            logger.info("onion.merged sources=%d query=%r", len(out), query[:120])
        return out


# True when at least one engine produced a real verdict (success/partial/genuine empty),
# vs every engine failing (error/blocked/timeout/changed). Distinguishes a genuine empty
# SERP — which is worth negative-caching — from a transient outage, which is not.
def _had_productive_engine(payload: dict[str, Any]) -> bool:
    productive = {"success", "partial", "empty"}
    engines = payload.get("engines") or {}
    return any((p or {}).get("status") in productive for p in engines.values())


# Build the payload returned when an identical query is hard-blocked as a repeat.
def _repeat_block_payload(query: str, effort: str, age: float) -> dict[str, Any]:
    note = (
        f"You already ran this exact search {age:.0f}s ago — the results were just "
        "shown above. Re-issuing the same query is suppressed; reuse the previous "
        "results, refine the query, or read_page one of those sources."
    )
    return {
        "query": query,
        "effort": effort,
        "blocked": True,
        "block_reason": "repeat",
        "note": note,
        "model_context": note,  # bridge reads model_context as the model-visible text
        "sources": [],
    }


# Cached/blocked/deduped entry point used by the MCP tool (run_web_search).
#
# Wraps the pure WebSearchService.search with the short-horizon "search memory":
#   1. identical query within the block window → hard block (no engines hit);
#   2. otherwise serve from the query-results cache when fresh, else run live;
#   3. drop result URLs already shown to the model within the suppression window;
#   4. record what was served, then warm the top unparsed URLs in the background.
async def run_web_search(
    query: str,
    *,
    effort: str = "low",
    region: str = "",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    shopping: bool = False,
    academic: bool = False,
    onion: bool = False,
    vertical_only: bool = False,
    dates_resolved: bool = False,
    query_text: str = "",
    operators: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from core.cache.hosted_cache import get_hosted_cache
    from core.config import load_search_config

    from .prefetch import get_prefetch_manager
    from .recent_tracker import get_recent_tracker

    cache_cfg = load_search_config().cache
    tracker = get_recent_tracker()
    cache = get_hosted_cache()
    # shopping/academic are part of the cache/repeat key: vertical and plain runs differ.
    key_args = dict(region=region, safesearch=safesearch, timelimit=timelimit,
                    effort=effort, shopping=shopping, academic=academic)
    qkey = tracker.query_key(query, **key_args)

    # An onion run is an explicit, rare, slow opt-in whose source set differs — bypass the
    # hosted cache, the repeat-block, AND the shared recency tracker entirely. The tracker is
    # keyed without the onion flag, so writing onion runs into it would (a) false-block the next
    # PLAIN search of the same text as a "repeat" and (b) leak .onion URLs into the suppression
    # map. Return immediately so an onion run always runs fresh and never pollutes plain state.
    if onion:
        payload = await WebSearchService().search(
            query, effort=effort, region=region, safesearch=safesearch,
            timelimit=timelimit, shopping=shopping, academic=academic, onion=True,
            vertical_only=vertical_only,
            dates_resolved=dates_resolved,
            query_text=query_text, operators=operators,
        )
        payload["cached"] = False
        return payload

    # 1. Identical query just served → hard block, no engines.
    age = tracker.repeat_age(qkey, cache_cfg.repeat_block_window_seconds)
    if age is not None:
        logger.info("web_search.repeat_block age=%.0fs query=%r", age, query[:160])
        return _repeat_block_payload(query, effort, age)

    # 2. Fresh cache hit, else run the live pipeline and cache it.
    payload = cache.get(query, **key_args)
    if payload is not None:
        payload = {**payload, "cached": True}
    else:
        payload = await WebSearchService().search(
            query, effort=effort, region=region, safesearch=safesearch,
            timelimit=timelimit, shopping=shopping, academic=academic,
            vertical_only=vertical_only,
            dates_resolved=dates_resolved,
            query_text=query_text, operators=operators,
        )
        payload["cached"] = False
        empty = not payload.get("sources")
        # A negative cache must reflect a genuine empty SERP, not a transient outage. If the
        # result is empty only because every engine errored/blocked/timed out, don't cache it
        # at all — caching would freeze a momentary failure into "nothing exists" for the TTL.
        if empty and not _had_productive_engine(payload):
            logger.info("web_search.skip_cache transient empty (no productive engine) query=%r", query[:160])
        else:
            cache.set(query, payload, is_empty=empty, **key_args)

    # 3. Drop sources the model was already shown within the suppression window.
    sources = list(payload.get("sources") or [])
    seen = tracker.recently_seen(
        [s.get("url", "") for s in sources], cache_cfg.seen_source_window_seconds
    )
    if seen:
        kept = [s for s in sources if s.get("url") not in seen]
        # Rebuild the model-visible text and chips so handles match the kept sources.
        search_cfg = load_search_config().search
        payload = {
            **payload,
            "sources": kept,
            "suppressed_seen": len(seen),
            "model_context": _build_model_context(
                query, kept,
                total_budget=int(search_cfg.total_context_budget or 0),
                per_source_chars=int(search_cfg.preview_max_chars or 0),
            ),
            "ui": _build_ui(kept),
        }
        sources = kept

    # 4. Remember what we served, then warm the top unparsed URLs for likely read_page.
    served_urls = [s.get("url", "") for s in sources]
    horizon = max(cache_cfg.repeat_block_window_seconds, cache_cfg.seen_source_window_seconds, 60)
    tracker.record(qkey, served_urls, horizon=float(horizon))

    if cache_cfg.prefetch_max_urls > 0:
        targets = [s.get("url", "") for s in sources if not s.get("markdown")]
        get_prefetch_manager().schedule(targets[: cache_cfg.prefetch_max_urls])

    return payload


def _combine_search_results(
    plans: list[dict[str, Any]],
    results: list[dict[str, Any]],
    *,
    effort: str,
    schema_mode: str,
    started: float,
    description: str = "",
    legacy_flags: dict[str, bool] | None = None,
) -> dict[str, Any]:
    from core.config import load_search_config

    search_cfg = load_search_config().search
    search_id = _make_search_id()
    all_sources: list[dict[str, Any]] = []
    sources_by_query: list[list[dict[str, Any]]] = []
    next_rank = 1

    for query_index, (plan, result) in enumerate(zip(plans, results), 1):
        query_sources: list[dict[str, Any]] = []
        for raw_source in result.get("sources") or []:
            source = dict(raw_source)
            source["id"] = _citation_id(search_id, next_rank)
            source["rank"] = next_rank
            source["batch_query"] = plan["compiled_query"]
            source["batch_query_index"] = query_index
            source["batch_query_vertical"] = plan.get("vertical", "web")
            source["query_index"] = query_index
            source["vertical"] = plan.get("vertical", "web")
            query_sources.append(source)
            all_sources.append(source)
            next_rank += 1
        sources_by_query.append(query_sources)

    total_budget = int(search_cfg.total_context_budget or 0)
    query_budget = total_budget // len(plans) if total_budget else 0
    per_source_chars = int(search_cfg.preview_max_chars or 0)
    context_sections: list[str] = []
    for plan, result, query_sources in zip(plans, results, sources_by_query):
        query = plan["compiled_query"]
        if query_sources:
            context_sections.append(_build_model_context(
                query, query_sources,
                total_budget=query_budget,
                per_source_chars=per_source_chars,
            ))
        else:
            context_sections.append(str(result.get("model_context") or f"No results found for: {query}"))

    request_queries = [
        {
            "vertical": str(plan.get("vertical") or "web"),
            "compiled_query": str(plan.get("compiled_query") or ""),
            "operators": dict(plan.get("operators") or {}),
        }
        for plan in plans
    ]
    search_request = {
        "schema_mode": schema_mode,
        "effort": effort,
        "queries": request_queries,
    }
    if description:
        search_request["description"] = description
    ui = _build_ui(all_sources)
    queries_with_sources = sum(bool(sources) for sources in sources_by_query)
    if queries_with_sources == 0:
        ui["status"] = "empty"
    elif queries_with_sources < len(plans):
        ui["status"] = "partial"
    else:
        ui["status"] = "done"
    ui["query_count"] = len(plans)
    ui["search_request"] = search_request
    if description:
        ui["description"] = description

    queries = [plan["compiled_query"] for plan in plans]
    payload: dict[str, Any] = {
        "query": queries[0] if len(queries) == 1 else queries,
        "queries": queries,
        "batch": len(queries) > 1,
        "search_id": search_id,
        "search_ids": [result.get("search_id") for result in results if result.get("search_id")],
        "effort": effort,
        "search_request": search_request,
        "languages": list(dict.fromkeys(
            str(result.get("language") or "") for result in results if result.get("language")
        )),
        "regions": list(dict.fromkeys(
            str(result.get("region") or "") for result in results if result.get("region")
        )),
        "engines_used": list(dict.fromkeys(
            engine for result in results for engine in (result.get("engines_used") or [])
        )),
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "cached": all(bool(result.get("cached")) for result in results),
        "model_context": (
            f"Batch search results for {len(plans)} queries run concurrently:\n\n"
            + "\n\n---\n\n".join(context_sections)
            if len(plans) > 1
            else context_sections[0]
        ),
        "sources": all_sources,
        "ui": ui,
        "query_results": [
            {
                "index": query_index,
                "query": plan["compiled_query"],
                "vertical": plan.get("vertical", "web"),
                "search_id": result.get("search_id"),
                "source_count": len(query_sources),
                "blocked": bool(result.get("blocked")),
                "cached": bool(result.get("cached")),
            }
            for query_index, (plan, result, query_sources) in enumerate(
                zip(plans, results, sources_by_query), 1
            )
        ],
    }
    payload.update(legacy_flags or {})
    return payload


# Run up to three independent legacy searches concurrently.
async def run_web_search_batch(
    queries: list[str],
    *,
    effort: str = "low",
    region: str = "",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    shopping: bool = False,
    academic: bool = False,
    onion: bool = False,
) -> dict[str, Any]:
    if not queries:
        return await run_web_search(
            "", effort=effort, region=region, safesearch=safesearch,
            timelimit=timelimit, shopping=shopping, academic=academic, onion=onion,
        )
    if len(queries) == 1:
        return await run_web_search(
            queries[0], effort=effort, region=region, safesearch=safesearch,
            timelimit=timelimit, shopping=shopping, academic=academic, onion=onion,
        )

    started = time.perf_counter()
    results = await asyncio.gather(*(
        run_web_search(
            query, effort=effort, region=region, safesearch=safesearch,
            timelimit=timelimit, shopping=shopping, academic=academic, onion=onion,
        )
        for query in queries
    ))

    vertical = "shopping" if shopping else ("academic" if academic else ("onion" if onion else "web"))
    plans = [
        {"vertical": vertical, "compiled_query": query, "operators": {}}
        for query in queries
    ]
    return _combine_search_results(
        plans,
        list(results),
        effort=effort,
        schema_mode="legacy",
        started=started,
        legacy_flags={"shopping": shopping, "academic": academic, "onion": onion},
    )


async def run_web_search_plan(search_request: dict[str, Any]) -> dict[str, Any]:
    """Run a prepared mixed-vertical advanced plan concurrently."""

    plans = [dict(item) for item in search_request.get("queries", []) if isinstance(item, dict)]
    effort = str(search_request.get("effort") or "medium")
    started = time.perf_counter()

    async def run_item(plan: dict[str, Any]) -> dict[str, Any]:
        vertical = str(plan.get("vertical") or "web")
        return await run_web_search(
            str(plan.get("compiled_query") or ""),
            effort=effort,
            timelimit=plan.get("timelimit"),
            shopping=vertical == "shopping",
            academic=vertical == "academic",
            onion=vertical == "onion",
            vertical_only=vertical == "shopping",
            dates_resolved=True,
            query_text=str(plan.get("text") or ""),
            operators=dict(plan.get("operators") or {}),
        )

    results = await asyncio.gather(*(run_item(plan) for plan in plans))
    return _combine_search_results(
        plans,
        list(results),
        effort=effort,
        schema_mode="advanced",
        started=started,
        description=str(search_request.get("description") or ""),
    )
