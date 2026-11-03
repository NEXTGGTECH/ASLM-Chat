# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from urllib.parse import urlparse

from core.cache.query_normalizer import QUERY_STOPWORDS as _QUERY_STOPWORDS, COMPOSITE_TOKENS as _COMPOSITE_TOKENS
from core.query.operators import CONTENT_OPERATOR_PREFIXES, NON_CONTENT_OPERATOR_PREFIXES


# Punctuation stripped from both ends of a query word.
_PUNCT_STRIP = "?!.,;:\"'()[]{}<>@#"


# Tokenise the query into meaningful lowercase terms longer than two characters, with search
# operators removed. Stopwords are dropped unless that would leave no terms at all.
def query_terms(query: str) -> list[str]:
    raw = []
    for t in (query or "").split():
        tl = t.lower()

        # Search operators are directives, not content terms: scoring against them only
        # dilutes the score of genuine results (no page contains "site:github.com" verbatim).
        if (
            tl == "or"
            or tl.startswith(NON_CONTENT_OPERATOR_PREFIXES)
            or (t.startswith("-") and len(t) > 1)
        ):
            continue

        # intitle:/inurl: constrain where a useful content term appears; keep the value for
        # lexical relevance while discarding only the operator prefix.
        if tl.startswith(CONTENT_OPERATOR_PREFIXES):
            t = t.split(":", 1)[1]
            tl = t.lower()

        # Composite tokens (e.g. ".NET", "C#") must be substituted before
        # punct-strip so ".NET".strip(…) → "net" false-match doesn't happen.
        if tl in _COMPOSITE_TOKENS:
            raw.append(_COMPOSITE_TOKENS[tl])
        else:
            stripped = tl.strip(_PUNCT_STRIP)
            if stripped:
                raw.append(stripped)
    raw = [t for t in raw if len(t) > 2]
    filtered = [t for t in raw if t not in _QUERY_STOPWORDS]
    return filtered or raw


# Return a BM25-lite relevance score in [0, 1]: the share of query terms found in the title
# (weight 0.6), the snippet (0.3) and the URL path (0.1). Return 0.0 when the query has no terms.
def lexical_score(query: str, title: str, snippet: str, url: str) -> float:
    terms = query_terms(query)
    if not terms:
        return 0.0
    url_path = (urlparse(url).path or "").lower()
    title_l = title.lower()
    snippet_l = snippet.lower()
    n = len(terms)
    return min(
        1.0,
        0.6 * sum(1 for t in terms if t in title_l) / n
        + 0.3 * sum(1 for t in terms if t in snippet_l) / n
        + 0.1 * sum(1 for t in terms if t in url_path) / n,
    )
