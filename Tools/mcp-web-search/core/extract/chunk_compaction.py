# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
from dataclasses import dataclass

from core.extract.chunk_quality import (
    SEO_HARD_REJECT_THRESHOLD,
    seo_keyword_stuffing_penalty,
)
from core.extract.fact_signals import has_currency, has_measurement
from core.extract.content_processor import (
    _bm25_score_paragraphs,
    _bm25_tokenize,
    _clean_latex_for_index,
    _has_latex,
    _is_fenced_code_block,
    _split_blocks,
    _truncate_markdown_to_budget,
)


# Tunables of the single compaction policy: the character budget, the minimum hybrid score for
# a paragraph to be kept, the chunk cap (and its expanded cap when enough chunks score at least
# expand_score), and the weight of the SEO-stuffing penalty in the hybrid score.
@dataclass(frozen=True)
class _Policy:
    char_budget: int = 1_400
    min_score: float = 0.40
    max_chunks: int = 4
    max_chunks_expanded: int = 8
    expand_score: float = 0.52
    seo_weight: float = 0.45


# Policy used when the caller gives no usable character budget.
_DEFAULT_POLICY = _Policy()


# Return the single compaction policy, with the character budget replaced by the caller's one
# (at least 400) when it is positive; every other tunable stays at its default.
def _policy_for(char_budget: int | None) -> _Policy:
    if char_budget is None or char_budget <= 0:
        return _DEFAULT_POLICY
    return _Policy(
        char_budget=max(400, int(char_budget)),
        min_score=_DEFAULT_POLICY.min_score,
        max_chunks=_DEFAULT_POLICY.max_chunks,
        max_chunks_expanded=_DEFAULT_POLICY.max_chunks_expanded,
        expand_score=_DEFAULT_POLICY.expand_score,
        seo_weight=_DEFAULT_POLICY.seo_weight,
    )


# Return a 0..1 estimate of how sentence-like the text is: sentence-ending marks counted against
# roughly one sentence per 180 characters. Texts shorter than 40 characters score 0.
def _sentence_like_ratio(text: str) -> float:
    compact = (text or "").strip()
    if len(compact) < 40:
        return 0.0
    endings = len(re.findall(r"[.!?…]", compact))
    return min(1.0, endings / max(1, len(compact) / 180))


# Return a cheap 0..1 factual-density proxy (no entity model): vocabulary variety, digits,
# currency and measurement mentions, and sentence-likeness.
def _entity_heuristic_score(text: str) -> float:
    tokens = _bm25_tokenize(text)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    digit_hits = sum(1 for ch in text if ch.isdigit())

    # Add up the weighted signals; the cap on the total keeps the score in 0..1.
    score = 0.0
    score += min(0.35, unique_ratio * 0.5)
    score += 0.15 if digit_hits >= 2 else 0.05 if digit_hits else 0.0
    score += 0.20 if has_currency(text) else 0.0
    score += 0.10 if has_measurement(text) else 0.0
    score += 0.20 * _sentence_like_ratio(text)
    return min(1.0, score)


# Scale scores to [0, 1] by dividing by the peak value; all zeros when there is no positive peak.
def _normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    peak = max(values)
    if peak <= 0:
        return [0.0] * len(values)
    return [min(1.0, max(0.0, v / peak)) for v in values]


# Return (hybrid_score, seo_penalty) per paragraph. The hybrid score blends the normalized BM25
# relevance, the factual-density heuristic and the inverse SEO-stuffing penalty.
def _score_paragraphs(
    paragraphs: list[str], query_terms: list[str], policy: _Policy
) -> list[tuple[float, float]]:
    if not paragraphs:
        return []

    # Index LaTeX-bearing pages on their cleaned text, so markup commands do not skew BM25.
    if _has_latex("\n\n".join(paragraphs)):
        index_paras = [_clean_latex_for_index(p) for p in paragraphs]
    else:
        index_paras = paragraphs

    # Compute the three per-paragraph signals; fenced code is never penalized as SEO spam.
    bm25_raw = _bm25_score_paragraphs(index_paras, query_terms) if query_terms else [0.0] * len(paragraphs)
    bm25 = _normalize_scores(bm25_raw)
    entity = [_entity_heuristic_score(p) for p in paragraphs]
    seo = [
        0.0 if _is_fenced_code_block(p) else seo_keyword_stuffing_penalty(p, query_terms)
        for p in paragraphs
    ]

    # Split the weight left after the SEO share between BM25 and the entity heuristic (55:25).
    seo_w = max(0.0, min(1.0, policy.seo_weight))
    remainder = max(0.05, 1.0 - seo_w)
    bm25_w = remainder * (0.55 / 0.80)
    entity_w = remainder * (0.25 / 0.80)

    scored: list[tuple[float, float]] = []
    for b, e, s in zip(bm25, entity, seo):
        hybrid = bm25_w * b + entity_w * e + seo_w * (1.0 - s)
        scored.append((min(1.0, hybrid), s))
    return scored


# Return the cap on selected prose paragraphs: the expanded cap when at least 3 paragraphs (and at
# least max_chunks) are strong, the normal cap otherwise.
def _chunk_limit(policy: _Policy, strong_count: int) -> int:
    if strong_count >= policy.max_chunks and strong_count >= 3:
        return policy.max_chunks_expanded
    return policy.max_chunks


# Compact parsed-page content to the query-relevant part. Paragraphs are scored by BM25 relevance
# to the query plus a cheap factual-density heuristic, SEO-stuffed blocks are rejected, and the
# best are packed into a character budget with one policy for every query (no query
# classification, no models, no embeddings). Fenced code is pinned verbatim first, then relevant
# prose is selected within the remaining budget; protected code may make the result exceed the
# budget. Kept paragraphs are returned in document order.
def compress_chunks(text: str, query: str, *, char_budget: int | None = None) -> str:
    policy = _policy_for(char_budget)

    # Without paragraph structure there is nothing to rank, so fall back to plain truncation.
    paragraphs = _split_blocks(text)
    if not paragraphs:
        return _truncate_markdown_to_budget(text, policy.char_budget, preserve_fenced_code=True)

    # Pin fenced code blocks and reserve their cost (plus the blank-line separator) up front.
    pinned = {idx for idx, paragraph in enumerate(paragraphs) if _is_fenced_code_block(paragraph)}
    pinned_cost = sum(len(paragraphs[idx]) + 2 for idx in pinned)
    prose_budget = max(0, policy.char_budget - pinned_cost)

    query_terms = _bm25_tokenize(query)
    scored = _score_paragraphs(paragraphs, query_terms, policy)

    # Text that already fits the budget is only filtered, not ranked: drop SEO-stuffed and
    # low-score paragraphs, keep the rest in order.
    if len(text) <= policy.char_budget:
        kept = [
            idx
            for idx, ((hybrid, seo), _para) in enumerate(zip(scored, paragraphs))
            if idx in pinned or (seo < SEO_HARD_REJECT_THRESHOLD and hybrid >= policy.min_score)
        ]
        if kept:
            return "\n\n".join(paragraphs[i] for i in sorted(kept))
        return ""

    # Candidates: unpinned paragraphs that pass the SEO rejection and the minimum score.
    eligible: list[tuple[int, float, float]] = []
    for idx, ((hybrid, seo), _para) in enumerate(zip(scored, paragraphs)):
        if idx in pinned:
            continue
        if seo >= SEO_HARD_REJECT_THRESHOLD or hybrid < policy.min_score:
            continue
        eligible.append((idx, hybrid, seo))

    if not eligible:
        # Fallback when nothing qualifies: take the highest-scoring paragraphs that are not
        # near-certain spam (penalty below 0.92) until the budget or chunk cap is reached.
        ranked = sorted(range(len(paragraphs)), key=lambda i: (-scored[i][0], scored[i][1]))
        fallback: list[int] = []
        budget = prose_budget
        for idx in ranked:
            if idx in pinned:
                continue
            if scored[idx][1] >= 0.92:
                continue
            cost = len(paragraphs[idx]) + 2
            if cost <= budget:
                fallback.append(idx)
                budget -= cost
            if len(fallback) >= policy.max_chunks:
                break
        selected_idx = sorted(pinned | set(fallback))
    else:
        # Greedy pick by hybrid score: the cap grows when many strong chunks exist, and a
        # paragraph that does not fit the remaining budget is skipped in favour of smaller ones.
        strong = [item for item in eligible if item[1] >= policy.expand_score]
        chunk_cap = _chunk_limit(policy, len(strong))
        ranked = sorted(eligible, key=lambda x: -x[1])
        selected: set[int] = set(pinned)
        budget = prose_budget
        for idx, _hybrid, _seo in ranked:
            if len(selected - pinned) >= chunk_cap:
                break
            cost = len(paragraphs[idx]) + 2
            if cost <= budget:
                selected.add(idx)
                budget -= cost
        selected_idx = sorted(selected)

    # Nothing selected at all: return a plain truncation instead of an empty result.
    if not selected_idx:
        return _truncate_markdown_to_budget(text, policy.char_budget, preserve_fenced_code=True)

    joined = "\n\n".join(paragraphs[i] for i in selected_idx)
    return joined
