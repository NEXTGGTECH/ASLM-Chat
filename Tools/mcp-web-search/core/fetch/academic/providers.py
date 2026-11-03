# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable
from urllib.parse import quote, quote_plus

from .registry import AcademicDomain, domain_for


# A wired scholarly provider: how to build the request URL and how to read the reply.
# Each provider maps to one `json_api` (or arxiv `atom`) row in academic_registry.json: the
# registry supplies pacing/host/tier, while the URL shape and response format differ per API
# and are pinned here.
@dataclass(frozen=True, slots=True)
class AcademicProvider:
    name: str
    # Registry pattern this provider maps to.
    host: str
    # Response format: "json", "atom", "html" or "serpapi".
    fmt: str
    # Build the request URL from (query, limit).
    url_builder: Callable[[str, int], str]
    # Base ranking weight, also the starting point of the paper confidence score.
    weight: float = 1.0
    # Per-request header overrides (e.g. a browser UA for HTML-scrape providers that reject
    # the polite scholarly UA). Merged over the client defaults for this provider only.
    extra_headers: tuple[tuple[str, str], ...] = ()

    # Return the per-request header overrides, or None when the provider has none.
    def headers(self) -> dict[str, str] | None:
        return dict(self.extra_headers) or None

    # Return the registry row (pacing/tier) for this provider's host, or None when it is absent.
    @property
    def domain(self) -> AcademicDomain | None:
        return domain_for(self.host)


# URL-encode a query for a query-string value (spaces become "+").
def _q(query: str) -> str:
    return quote_plus(query.strip())


# URL-encode a query for a path segment (everything escaped, including "/").
def _path_q(query: str) -> str:
    return quote(query.strip(), safe="")


# Polite identifier: OpenAlex/Crossref/Europe PMC give faster, more reliable service to
# requests that announce a contact in the User-Agent / mailto. Not auth — just etiquette.
MAILTO = "aslm-chat@users.noreply.github.com"


# Every wired provider. Covers the no-auth, free-text-search APIs: OpenAlex, Crossref, Europe PMC,
# DOAJ (JSON) and arXiv (Atom XML), plus Google Scholar (HTML scrape or SerpApi). Hardened
# SPA/browser tiers (pubmed, semanticscholar) and DOI-only lookups (unpaywall) are described in
# the registry but not wired here.
PROVIDERS: tuple[AcademicProvider, ...] = (
    AcademicProvider(
        name="openalex",
        host="openalex.org",
        fmt="json",
        url_builder=lambda q, n: (
            f"https://api.openalex.org/works?search={_q(q)}"
            f"&per_page={max(1, n)}&mailto={MAILTO}"
        ),
        weight=1.0,
    ),
    AcademicProvider(
        name="crossref",
        host="crossref.org",
        fmt="json",
        url_builder=lambda q, n: (
            f"https://api.crossref.org/works?query={_q(q)}"
            f"&rows={max(1, n)}&select=DOI,title,author,abstract,issued,"
            f"container-title,is-referenced-by-count,URL,link&mailto={MAILTO}"
        ),
        weight=0.9,
    ),
    AcademicProvider(
        name="europepmc",
        host="europepmc.org",
        fmt="json",
        url_builder=lambda q, n: (
            "https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
            f"query={_q(q)}&format=json&resultType=core&pageSize={max(1, n)}"
        ),
        weight=0.85,
    ),
    AcademicProvider(
        name="doaj",
        host="doaj.org",
        fmt="json",
        url_builder=lambda q, n: (
            f"https://doaj.org/api/search/articles/{_path_q(q)}?pageSize={max(1, n)}"
        ),
        weight=0.7,
    ),
    AcademicProvider(
        name="arxiv",
        host="arxiv.org",
        fmt="atom",
        url_builder=lambda q, n: (
            f"https://export.arxiv.org/api/query?search_query=all:{_q(q)}"
            f"&start=0&max_results={max(1, n)}&sortBy=relevance"
        ),
        weight=0.8,
    ),
    # Google Scholar: no API, HTML scrape. Strong cross-publisher aggregator with excellent
    # relevance + "Cited by N", but snippet-only (no clean DOI/abstract) and antibot-prone —
    # it answers plain httpx only when paced (registry rps 0.25) and returns a 200 CAPTCHA
    # page when hammered, so the engine treats a CAPTCHA body as an antibot block (cooldown).
    AcademicProvider(
        name="scholar",
        host="scholar.google.com",
        fmt="html",
        url_builder=lambda q, n: (
            f"https://scholar.google.com/scholar?q={_q(q)}&hl=en&num={min(20, max(1, n))}"
        ),
        weight=0.82,
        extra_headers=(
            ("User-Agent",
             "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
             "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"),
            ("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"),
            ("Accept-Language", "en-US,en;q=0.9"),
        ),
    ),
    # Block-free Scholar via SerpApi's google_scholar engine (key-gated). SerpApi solves
    # the antibot on their backend, so this never blocks our IP — strictly preferred over
    # the direct scrape whenever a SerpApi key is configured. The URL is `search.json`; the
    # query/engine/key go in params (added by the engine, which holds the secret).
    AcademicProvider(
        name="scholar_serpapi",
        host="scholar.google.com",
        fmt="serpapi",
        url_builder=lambda q, n: "https://serpapi.com/search.json",
        weight=0.9,
    ),
)


# Return True when a SerpApi key is configured (the block-free Scholar path is available).
def _has_serpapi_key() -> bool:
    try:
        from core.config.api_keys import load_api_keys

        return bool((load_api_keys().search.hosted_api.serpapi_api_key or "").strip())
    except Exception:  # noqa: BLE001 — no key / config issue → fall back to direct scrape
        return False


# Return the active provider set: exactly one Scholar path is used. With a SerpApi key, the block-free
# `scholar_serpapi` replaces the antibot-prone direct `scholar` scrape; without it, the
# direct scrape stands (and leans on pacing + cooldown).
def active_providers() -> tuple[AcademicProvider, ...]:
    drop = "scholar" if _has_serpapi_key() else "scholar_serpapi"
    return tuple(p for p in PROVIDERS if p.name != drop)


# Return the active providers ranked by static weight (registry pacing could refine this later).
def ranked_providers() -> list[AcademicProvider]:
    return sorted(active_providers(), key=lambda p: p.weight, reverse=True)
