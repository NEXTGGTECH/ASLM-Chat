# Copyright NEXTGGTECH. Elastic License 2.0.

"""Incremental, model-free SERP triage.

Sources are scored the moment they arrive from search_stream. The session decides
per source: parse now, hold in queue, or skip. Consensus votes (the same URL
surfacing from another provider family) re-score an already-seen source and may
upgrade it from the queue into a parse slot.

No registries, no models, no I/O — pure functions over SERP fields, so a decision
costs well under a millisecond. Learned domain trust enters as a point-in-time
ReputationSnapshot the caller loads up front (one DB read per search); per-source
scoring stays a dict lookup.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from core.profiles import ReputationSnapshot, domain_of

from core.extract.scoring import query_terms
from core.config import load_search_config
from .query_locales import analyze_query, detect_result_language, language_mismatch_penalty

from .quality import (
    emd_penalty,
    extract_date,
    hub_penalty,
    insecure_scheme_penalty,
    is_established_tld,
    is_skip_title,
    page_date_score,
    query_years,
    seo_slug_penalty,
    suspicious_url_penalty,
)

# Signal weights. Position dominates BY POLICY: the engines already solved authority
# ranking, and our job is triage + junk suppression, not re-ranking their SERPs. The
# date weight pays for a REAL extracted date (see quality.extract_date), not year
# tokens in the title.
# There is NO lexical term-matching channel — deleted outright, BY MEASUREMENT
# (ablation 2026-07-05, 9 live-captured cases incl. two real-world failures): removing
# it beat every weighting tried (expected top1 7/9 vs 6/9 at lex=0.15, mean rank 1.67
# vs 1.89, no case worse). Flat term matching only ever paid SEO titles for filler
# words — "GPT-4 Technical Report" outranked the asked-for GPT-4o System Card because
# it matched "technical report". IDF-weighted and version-token-weighted variants were
# also measured and did NOT outperform plain removal, so no cleverness was kept.
# Relevance discrimination is consensus + position + earned signals; the next lever
# for cases those can't split (premier engines disagreeing at rank 1) is post-parse
# content rescore, not term matching.
_W_POSITION = 0.60
_W_SNIPPET_LEN = 0.05
_W_DATE = 0.10

# Engine classes drive how much a SERP position is worth:
# premier indexes earn a strong positional prior, recall helpers a weak one.
# Hosted API providers sit ABOVE the google family by design (Dima's call): they are
# paid, curated, relevance-reranked SERPs immune to our scrape-blocking noise — when
# the user pays for them, their placement is trusted the most.
# Yandex is deliberately floored: its organic ranking has proven unreliable across the
# board, so its placement carries less prior than any recall helper — its finds must
# earn their parse slot through other families' consensus or leftover budget.
_ENGINE_POSITION_WEIGHT = {
    "hosted:tavily": 1.05,
    "hosted:firecrawl": 1.05,
    "hosted:brave": 1.05,
    "hosted:serpapi": 1.05,
    "google": 1.00,
    "startpage": 1.00,  # same family, same prior
    "duckduckgo": 0.85,
    "brave": 0.70,
    "qwant": 0.60,
    "yandex": 0.35,
    "legacy_yandex": 0.35,
}
_DEFAULT_POSITION_WEIGHT = 0.60

# Expected SERP depth used to decay position into [0,1].
_POSITION_DEPTH = 10

# Consensus: votes from distinct provider families. Primary trust signal — it
# replaces the legacy curated trust registry (deleted by design). A vote is worth
# what its family's judgement is worth (same rationale as the position prior — and
# the same Yandex floor); the strongest family present is treated as the baseline
# view and the REST vote, so the bonus does not depend on arrival order.
_CONSENSUS_STEP = 0.18
_CONSENSUS_CAP = 0.36
# NOTE: vote weight is NOT the position weight. Position weight prices how much we
# trust a family's ORDERING; vote weight prices how much its listing counts as
# INDEPENDENT EVIDENCE that the page matters. Hence brave votes above ddg despite the
# weaker positional prior (own index → more independent evidence; ddg is bing-flavored
# and correlates with everyone), and firecrawl (google-proxying) votes at google grade
# while tavily's reranked feed sits just under it. Hand-tuned 2026-07-05.
_FAMILY_VOTE_WEIGHT = {
    "tavily": 0.95,
    "firecrawl": 1.00,
    "google": 1.00,
    "duckduckgo": 0.80,
    "brave": 0.85,
    "qwant": 0.65,
    "yandex": 0.35,
    "legacy_yandex": 0.35,
}
_DEFAULT_VOTE_WEIGHT = 0.70

# Decision thresholds on the [0,1] soft score.
_SKIP_THRESHOLD = 0.10
_PARSE_THRESHOLD = 0.50

# E-E-A-T, reworked: trust is earned, not declared. A domain on an unproven TLD with no
# positive runtime history and only ONE provider family behind it must clear a slightly
# higher parse bar. A second family's vote or earned history removes the margin — the
# unknown is never punished for being unknown, only asked for more at the boundary.
_UNPROVEN_PARSE_MARGIN = 0.05

# Snippet shorter than this gets a soft penalty (legacy hard-skipped — too harsh:
# engines legitimately emit short snippets).
_SHORT_SNIPPET_CHARS = 30
_SHORT_SNIPPET_PENALTY = 0.08


class TriageAction(StrEnum):
    PARSE = "parse"  # score cleared the bar — fetch/parse immediately
    QUEUE = "queue"  # usable, waits for a free slot or a consensus upgrade
    SKIP = "skip"  # not worth a fetch


@dataclass(slots=True)
class TriageDecision:
    action: TriageAction
    score: float
    url: str
    # True when this decision upgrades a previously queued source (consensus vote).
    upgraded: bool = False


# Internal per-URL state for incremental consensus rescoring.
@dataclass(slots=True)
class _SourceState:
    base_score: float
    families: set[str] = field(default_factory=set)
    action: TriageAction = TriageAction.SKIP
    # Unproven-TLD domain without earned history: stricter parse bar while it has
    # only a single provider family behind it.
    unproven: bool = False
    # The date component's value currently folded into base_score, kept so a better
    # date source (the parsed page's own published_time) can swap it out later.
    date_value: float = 0.0


# Incremental triage over the live source stream.
class TriageSession:

    def __init__(self, query: str, reputation: ReputationSnapshot | None = None) -> None:
        self.query = query
        self._years = query_years(query)
        self._terms = tuple(query_terms(query))  # once per session, for the EMD tell
        self._query_language = analyze_query(query).language
        localization = load_search_config().localization
        self._language_weight = localization.language_penalty if localization.enabled else 0.0
        self._states: dict[str, _SourceState] = {}
        self._penalties = reputation.penalties if reputation else {}
        self._proven = reputation.proven if reputation else frozenset()

    # Score one source from its SERP fields alone (no consensus component).
    # Returns (score, date_value) — the date component is tracked separately so the
    # parsed page's own published_time can replace the snippet estimate post-parse.
    def _soft_score(
        self,
        *,
        engine: str,
        rank: int,
        title: str,
        snippet: str,
        url: str,
    ) -> tuple[float, float]:
        position = max(0.0, 1.0 - (max(1, rank) - 1) / _POSITION_DEPTH)
        position_weight = _ENGINE_POSITION_WEIGHT.get(engine, _DEFAULT_POSITION_WEIGHT)
        hub = hub_penalty(url, title, snippet)
        snip_len = min(1.0, len(snippet) / 300)
        # Only the snippet may carry the date: engines prefix real publication dates
        # there, while a title year is a stuffing tell, not evidence.
        date_value = page_date_score(extract_date(snippet), self._years)

        score = (
            _W_POSITION * position * position_weight
            + _W_SNIPPET_LEN * snip_len
            + _W_DATE * date_value
        )
        if len(snippet) < _SHORT_SNIPPET_CHARS:
            score -= _SHORT_SNIPPET_PENALTY
        score -= 0.25 * hub
        # Identity-blind SEO trim: gently down-weight year-stuffed farm slugs by URL shape
        # only. No domain favouritism — authority is earned via consensus, not declared.
        score += seo_slug_penalty(url)
        score += insecure_scheme_penalty(url)
        # Bad-site tells: structural marks of phishing/farm URLs (IP hosts, embedded
        # gTLD costumes, hyphen/year-stamped names, exact-match-domain squatting).
        score += suspicious_url_penalty(url)
        score += emd_penalty(url, self._terms)
        # Language-of-result scoring (query_locales): a page written in a script the
        # query did not use pays a small, capped toll — never a ban, consensus can
        # always outvote it, and English results never pay at all.
        if self._language_weight:
            result_language = detect_result_language(title, snippet, url)
            score -= language_mismatch_penalty(
                self._query_language, result_language, self._language_weight
            )
        # Earned negative reputation (TLS failures, systematic empty parses). Magnitude
        # is capped at write-out; consensus votes can still outvote it — two independent
        # families saying "this page answers the query" beat an old grudge.
        score -= self._penalties.get(domain_of(url), 0.0)
        return max(0.0, min(1.0, score)), date_value

    # Map a score to an action. strict raises the parse bar for unproven single-family
    # domains; SKIP is never affected — reputation nudges, it does not execute.
    @staticmethod
    def _action_for(score: float, *, strict: bool = False) -> TriageAction:
        if score < _SKIP_THRESHOLD:
            return TriageAction.SKIP
        bar = _PARSE_THRESHOLD + (_UNPROVEN_PARSE_MARGIN if strict else 0.0)
        return TriageAction.PARSE if score >= bar else TriageAction.QUEUE

    # Whether the stricter parse bar applies to this source right now.
    @staticmethod
    def _strict_bar(state: _SourceState) -> bool:
        return state.unproven and len(state.families) <= 1

    # Total score = base soft score + consensus bonus for extra families. The bonus is
    # order-independent: the strongest family present is the baseline, the rest vote at
    # their class weight.
    @staticmethod
    def _total(state: _SourceState) -> float:
        if len(state.families) <= 1:
            return min(1.0, state.base_score)
        weights = sorted(
            (_FAMILY_VOTE_WEIGHT.get(f, _DEFAULT_VOTE_WEIGHT) for f in state.families),
            reverse=True,
        )
        bonus = min(_CONSENSUS_CAP, _CONSENSUS_STEP * sum(weights[1:]))
        return min(1.0, state.base_score + bonus)

    # Ingest a new source event; returns the decision for this URL.
    def ingest_source(
        self,
        *,
        engine: str,
        provider_family: str,
        rank: int,
        url: str,
        title: str,
        snippet: str,
    ) -> TriageDecision:
        if is_skip_title(title):
            self._states[url] = _SourceState(base_score=0.0, action=TriageAction.SKIP)
            return TriageDecision(action=TriageAction.SKIP, score=0.0, url=url)

        base, date_value = self._soft_score(
            engine=engine, rank=rank, title=title, snippet=snippet, url=url
        )
        domain = domain_of(url)
        state = _SourceState(
            base_score=base,
            unproven=not is_established_tld(domain) and domain not in self._proven,
            date_value=date_value,
        )
        state.families.add(provider_family)
        score = self._total(state)
        state.action = self._action_for(score, strict=self._strict_bar(state))
        self._states[url] = state
        return TriageDecision(action=state.action, score=score, url=url)

    # Re-ingest an already-seen URL surfacing as a full SERP entry from another stream.
    # Two effects: the base becomes the BEST single-engine view of the page (a rank-1
    # Google listing must not stay priced at the low-trust-rank-9 view that arrived first —
    # essential now that position dominates the score), and the new family votes.
    def ingest_revisit(
        self,
        *,
        engine: str,
        provider_family: str,
        rank: int,
        url: str,
        title: str,
        snippet: str,
    ) -> TriageDecision | None:
        state = self._states.get(url)
        if state is None or state.action == TriageAction.SKIP:
            return None
        base, date_value = self._soft_score(
            engine=engine, rank=rank, title=title, snippet=snippet, url=url
        )
        if base > state.base_score:
            state.base_score = base
            state.date_value = date_value
        return self._rescore(state, url, provider_family)

    # Ingest a consensus vote for an already-seen URL. Returns an upgraded decision
    # when the extra vote lifts the source over a threshold, else None.
    def ingest_vote(self, *, provider_family: str, url: str) -> TriageDecision | None:
        state = self._states.get(url)
        if state is None or state.action == TriageAction.SKIP:
            return None
        return self._rescore(state, url, provider_family)

    # Add a family (when new), re-derive the action, and surface a queue→parse upgrade.
    def _rescore(self, state: _SourceState, url: str, provider_family: str) -> TriageDecision | None:
        state.families.add(provider_family)
        score = self._total(state)
        new_action = self._action_for(score, strict=self._strict_bar(state))
        upgraded = new_action == TriageAction.PARSE and state.action == TriageAction.QUEUE
        state.action = new_action
        if upgraded:
            return TriageDecision(action=new_action, score=score, url=url, upgraded=True)
        return None

    # Swap the snippet-estimated date component for the parsed page's own date (its
    # declared published_time is strictly better evidence). Called after a successful
    # parse, before the final ranking sort; a page with no declared date is left as-is.
    def apply_page_date(self, url: str, date: tuple[int, int, int]) -> None:
        state = self._states.get(url)
        if state is None:
            return
        new_value = page_date_score(date, self._years)
        state.base_score = max(
            0.0, min(1.0, state.base_score + _W_DATE * (new_value - state.date_value))
        )
        state.date_value = new_value

    # Current score for a URL (queue ordering for leftover slots).
    def score_of(self, url: str) -> float:
        state = self._states.get(url)
        return self._total(state) if state else 0.0
