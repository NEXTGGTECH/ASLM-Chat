# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import re
import time

import httpx

from core.fetch.arxiv_api import arxiv_api_slot

from .health import get_provider_health
from .models import AcademicPaper, AcademicProviderAttempt, AcademicSearchResult
from .parse import PARSERS, is_scholar_block, parse_arxiv, parse_scholar, parse_serpapi_scholar
from .providers import AcademicProvider, ranked_providers

# Per-effort fetch breadth and overall wall-clock ceiling (regional has no effect here —
# the scholarly APIs are global English-first endpoints).
EFFORT_LIMIT = {"low": 6, "medium": 10, "high": 16}
EFFORT_HARD_TIMEOUT_MS = {"low": 3500, "medium": 6000, "high": 9000}

# Default request headers for every provider call (polite contact UA, JSON or Atom accepted).
_HEADERS = {
    "User-Agent": (
        "ASLM-Chat-academic/1.0 (mailto:aslm-chat@users.noreply.github.com) "
        "httpx scholarly-aggregator"
    ),
    "Accept": "application/json, application/atom+xml;q=0.9, */*;q=0.5",
}

# Matches runs of non-alphanumerics; replaced with a space to normalize titles for dedup.
_TITLE_NORM_RE = re.compile(r"[^a-z0-9]+")

# Ranking diversity knobs. Consensus rewards cross-index agreement (quality signal, folded
# into the score). Saturation penalises the Nth pick from the same provider (ordering only)
# so a citation-rich index leads the top-N without owning all of it. ~0.13 is tuned so one
# provider's tail drops below another's head while a genuinely better source still leads.
_CONSENSUS_BONUS = 0.10
_SOURCE_SATURATION = 0.13


# Polite minimum seconds between fires for a provider, from its registry rps budget.
def _min_interval(provider: AcademicProvider) -> float:
    dom = provider.domain
    rps = dom.rps if dom and dom.rps > 0 else 0.0
    return 1.0 / rps if rps > 0 else 0.0


# Academic vertical engine: fans out to keyless scholarly REST APIs under a hard budget.
# Mirrors the shopping engine's discipline: providers run concurrently under one hard deadline
# per effort tier, and whatever finished by the deadline is merged (the rest is cancelled and
# the result is marked partial). Every provider soft-fails independently: a 4xx, timeout or
# parse miss drops that provider, never the search. Results are deduped across providers (DOI
# first, then a normalized title) so the same paper indexed by OpenAlex and Crossref counts once.
class AcademicSearchEngine:
    # Run one academic search: fire the available providers concurrently, collect what finishes
    # before the hard deadline, then rank and dedupe the papers into a result with attempt records.
    async def search(
        self,
        query: str,
        *,
        effort: str = "medium",
        limit: int | None = None,
        hard_timeout_ms: int | None = None,
    ) -> AcademicSearchResult:
        # Resolve the effort tier into the result cap and the hard deadline. Providers are asked
        # for a little more than the cap so dedup still fills it.
        started = time.perf_counter()
        effort = effort if effort in EFFORT_LIMIT else "medium"
        cap = limit if limit is not None else EFFORT_LIMIT[effort]
        per_provider = max(3, cap)
        timeout_ms = hard_timeout_ms if hard_timeout_ms is not None else EFFORT_HARD_TIMEOUT_MS[effort]

        # Per-search accumulators: attempt records, raw papers and the partial-result flag.
        health = get_provider_health()
        attempts: list[AcademicProviderAttempt] = []
        collected: list[AcademicPaper] = []
        partial = False
        partial_reason = ""

        # Bench providers still cooling down from repeated antibot/errors, or paced out by
        # their polite rps budget (registry rps → min-interval) — they record a `skip`
        # attempt and are not hit. Preemptive pacing keeps antibot-prone sources (Scholar)
        # under their rate limit instead of getting blocked then cooled down.
        providers = []
        for p in ranked_providers():
            if health.available(p.name, _min_interval(p)):
                providers.append(p)
                health.note_fired(p.name)
            elif health.cooldown_remaining(p.name) > 0:
                attempts.append(AcademicProviderAttempt(
                    provider=p.name, method=p.fmt, url="", ok=False, elapsed_ms=0,
                    error=f"cooldown {health.cooldown_remaining(p.name):.0f}s",
                ))
            else:
                attempts.append(AcademicProviderAttempt(
                    provider=p.name, method=p.fmt, url="", ok=False, elapsed_ms=0,
                    error="paced (rps budget)",
                ))

        # Fire the selected providers concurrently and harvest completions until the deadline.
        timeout = httpx.Timeout(timeout_ms / 1000, connect=min(3.0, timeout_ms / 1000))
        async with httpx.AsyncClient(headers=_HEADERS, follow_redirects=True, timeout=timeout) as client:
            tasks = {
                asyncio.create_task(self._provider(client, p, query, per_provider)): p
                for p in providers
            }
            deadline = started + timeout_ms / 1000
            pending = set(tasks)
            try:
                while pending:
                    wait = max(0.0, deadline - time.perf_counter())
                    if wait <= 0:
                        break
                    done, pending = await asyncio.wait(
                        pending, timeout=wait, return_when=asyncio.FIRST_COMPLETED
                    )
                    if not done:
                        break
                    for task in done:
                        papers, attempt = task.result()
                        attempts.append(attempt)
                        collected.extend(papers)

                # Deadline reached: cancel the unfinished providers and mark the result partial.
                if pending:
                    partial = True
                    partial_reason = "hard_timeout"
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
            except asyncio.CancelledError:
                # The caller cancelled the whole search: stop every provider task, then propagate.
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        # Merge duplicates across providers, rank, and report timings plus per-provider health.
        papers = self._rank_and_dedupe(collected, cap)
        return AcademicSearchResult(
            query=query,
            effort=effort,
            papers=papers,
            attempts=attempts,
            timings={
                "total_elapsed_ms": int((time.perf_counter() - started) * 1000),
                "hard_timeout_ms": timeout_ms,
                "providers_ok": sum(1 for a in attempts if a.ok),
                "providers_fired": len(providers),
                "providers_total": len(ranked_providers()),
                "raw_papers": len(collected),
                "cooldowns": health.snapshot(),
            },
            partial=partial,
            partial_reason=partial_reason,
        )

    # One provider call: fetch + parse, never raises (soft-fails into an attempt record).
    async def _provider(
        self, client: httpx.AsyncClient, provider: AcademicProvider, query: str, limit: int
    ) -> tuple[list[AcademicPaper], AcademicProviderAttempt]:
        url = provider.url_builder(query, limit)
        health = get_provider_health()

        # SerpApi google_scholar: the query/engine/key ride in params (the engine holds the
        # secret, not the provider). No key → not selected, but guard anyway.
        params = None
        if provider.fmt == "serpapi":
            from core.config.api_keys import load_api_keys

            key = (load_api_keys().search.hosted_api.serpapi_api_key or "").strip()
            if not key:
                return [], AcademicProviderAttempt(
                    provider=provider.name, method=provider.fmt, url=url, ok=False,
                    elapsed_ms=0, error="no serpapi key",
                )
            params = {"engine": "google_scholar", "q": query, "num": min(20, max(1, limit)),
                      "api_key": key, "output": "json", "hl": "en"}

        # Fetch the page; arXiv calls go through the shared arXiv API slot (its own rate limit).
        started = time.perf_counter()
        try:
            if provider.name == "arxiv":
                async with arxiv_api_slot():
                    resp = await client.get(url, params=params, headers=provider.headers())
            else:
                resp = await client.get(url, params=params, headers=provider.headers())
            elapsed = int((time.perf_counter() - started) * 1000)

            # A non-200 response counts against the provider's health and ends the attempt.
            if resp.status_code != 200:
                health.record(provider.name, ok=False, status_code=resp.status_code)
                return [], AcademicProviderAttempt(
                    provider=provider.name, method=provider.fmt, url=url, ok=False,
                    elapsed_ms=elapsed, status_code=resp.status_code,
                    error=f"http {resp.status_code}",
                )

            # Parse by response format: atom (arXiv), SerpApi JSON, scraped HTML, or per-provider JSON.
            if provider.fmt == "atom":
                papers = parse_arxiv(resp.text)
            elif provider.fmt == "serpapi":
                papers = parse_serpapi_scholar(resp.json())
            elif provider.fmt == "html":
                # HTML-scrape (Scholar): a 200 may be a CAPTCHA wall, not results — count it
                # as antibot (synthetic 429) so the cooldown benches it, exactly like a 403.
                if is_scholar_block(resp.text):
                    health.record(provider.name, ok=False, status_code=429)
                    return [], AcademicProviderAttempt(
                        provider=provider.name, method=provider.fmt, url=url, ok=False,
                        elapsed_ms=elapsed, status_code=200, error="antibot captcha",
                    )
                papers = parse_scholar(resp.text)
            else:
                papers = PARSERS[provider.name](resp.json())

            # Drop untitled records and score the rest.
            papers = [p for p in papers if p.title]
            for p in papers:
                p.confidence = self._score(p, provider)

            # A clean 200 (even with zero papers) is healthy — it resets the streak.
            health.record(provider.name, ok=True, status_code=200)
            return papers, AcademicProviderAttempt(
                provider=provider.name, method=provider.fmt, url=url, ok=bool(papers),
                elapsed_ms=elapsed, status_code=200, papers=len(papers),
            )
        except Exception as exc:  # noqa: BLE001 — a provider failure must not sink the search
            health.record(provider.name, ok=False, error=type(exc).__name__)
            return [], AcademicProviderAttempt(
                provider=provider.name, method=provider.fmt, url=url, ok=False,
                elapsed_ms=int((time.perf_counter() - started) * 1000),
                error=f"{type(exc).__name__}: {exc}",
            )

    # Lightweight relevance/quality blend — provider weight, citation pull, abstract+OA bonus.
    # The citation bonus is deliberately capped low (0.3): it's a soft authority nudge, not a
    # dominant axis — a large cap let citation-rich indexes (OpenAlex/Scholar) open an
    # unbridgeable gap over citation-less ones (arXiv/Europe PMC) and monopolise the top-N.
    def _score(self, paper: AcademicPaper, provider: AcademicProvider) -> float:
        score = provider.weight
        if paper.citations:
            score += min(0.3, paper.citations / 3000.0)
        if paper.abstract:
            score += 0.15
        if paper.open_access or paper.pdf_url:
            score += 0.1
        return round(score, 4)

    # Cross-provider dedup + diversity-aware ranking.
    #   1. dedup: a DOI is authoritative; else a normalized title. Highest score wins, the
    #      loser's pdf/abstract/citations backfill the winner and mark it in `also_in`.
    #   2. consensus bonus: a paper several indexes agree on is a stronger signal — fold it
    #      into the score (real quality, so it shows in the score field).
    #   3. source saturation: a per-provider diminishing penalty applied to *ordering only*
    #      (not the score) so one rich index (OpenAlex/Scholar) can lead but not monopolise
    #      the top-N — its long tail sinks below other providers' best.
    def _rank_and_dedupe(self, papers: list[AcademicPaper], cap: int) -> list[AcademicPaper]:
        # Dual-key dedup: match on DOI OR a normalized title, so the same paper indexed by
        # OpenAlex (its DOI) and Scholar (URL + title, no DOI) collapses into one entry
        # instead of appearing twice. A title key is only trusted when it's distinctive
        # enough (>=12 chars) to avoid merging different short-titled works.
        best: dict[str, AcademicPaper] = {}
        order: list[str] = []
        by_doi: dict[str, str] = {}
        by_title: dict[str, str] = {}
        for paper in sorted(papers, key=lambda p: p.confidence, reverse=True):
            tnorm = _TITLE_NORM_RE.sub(" ", paper.title.lower()).strip()
            tkey = tnorm if len(tnorm) >= 12 else ""
            ckey = (by_doi.get(paper.doi) if paper.doi else None) or (by_title.get(tkey) if tkey else None)
            if ckey is not None:
                self._backfill(best[ckey], paper)
            else:
                ckey = paper.doi or tkey
                if not ckey:
                    continue
                best[ckey] = paper
                order.append(ckey)
            if paper.doi:
                by_doi[paper.doi] = ckey
            if tkey:
                by_title[tkey] = ckey

        # Consensus bonus: reward papers several providers agree on (capped at 3 extra sources).
        kept = [best[k] for k in order]
        for paper in kept:
            agree = len(paper.meta.get("also_in", []))
            if agree:
                paper.confidence = round(paper.confidence + _CONSENSUS_BONUS * min(agree, 3), 4)

        # Saturation: the Nth pick from one source loses N * _SOURCE_SATURATION in ordering only.
        seen: dict[str, int] = {}
        adjusted: list[tuple[float, AcademicPaper]] = []
        for paper in sorted(kept, key=lambda p: p.confidence, reverse=True):
            n = seen.get(paper.source, 0)
            seen[paper.source] = n + 1
            adjusted.append((paper.confidence - _SOURCE_SATURATION * n, paper))
        adjusted.sort(key=lambda t: t[0], reverse=True)
        return [paper for _, paper in adjusted][:cap]

    # Fill gaps in the kept paper (abstract, PDF, DOI, citations, open access) from a duplicate
    # and record the duplicate's source in `meta["also_in"]`.
    @staticmethod
    def _backfill(keep: AcademicPaper, other: AcademicPaper) -> None:
        if not keep.abstract and other.abstract:
            keep.abstract = other.abstract
        if not keep.pdf_url and other.pdf_url:
            keep.pdf_url = other.pdf_url
        if not keep.doi and other.doi:
            keep.doi = other.doi
        if keep.citations is None and other.citations is not None:
            keep.citations = other.citations
        keep.open_access = keep.open_access or other.open_access
        if other.source not in keep.meta.get("also_in", []):
            keep.meta.setdefault("also_in", []).append(other.source)


# Run one academic search with a fresh engine (convenience entry point).
async def search_academic(
    query: str, *, effort: str = "medium", limit: int | None = None
) -> AcademicSearchResult:
    return await AcademicSearchEngine().search(query, effort=effort, limit=limit)
