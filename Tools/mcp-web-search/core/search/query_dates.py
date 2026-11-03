# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import datetime as _dt
import re
from typing import Optional

from core.query.operators import DATE_OPERATOR_PREFIXES

# Timelimit windows, tightest first, for picking the more restrictive of two.
_TIMELIMIT_ORDER: dict[str, int] = {"d": 0, "w": 1, "m": 2, "y": 3}

# Year / year-range token shapes. A range may use a hyphen, en-dash, or space.
# Excludes version-like tokens (decimals / single digits) by requiring 4-digit 19xx/20xx.
_YEAR_RANGE_SEPARATOR = r"(?:\s*(?:-|–)\s*|\s+)"
# Comma-preceded trailing year: always a time tag, never a topic anchor.
_TRAILING_YEAR_COMMA_RE = re.compile(
    rf",\s+(?:(?:19|20)\d{{2}})(?:{_YEAR_RANGE_SEPARATOR}(?:19|20)\d{{2}})?\s*$"
)
# A standalone year / range anywhere in the query.
_YEAR_ANYWHERE_RE = re.compile(
    rf"\b(?:19|20)\d{{2}}(?:{_YEAR_RANGE_SEPARATOR}(?:19|20)\d{{2}})?\b"
)
_ANY_YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


# Return True when the year at `start` is the value of a date operator (such as "after:"), so it is
# part of an explicit date filter and not a year written in the query text.
def _is_date_operator_year(query: str, start: int) -> bool:
    prefix = query[max(0, start - 7):start].lower()
    return any(prefix.endswith(operator) for operator in DATE_OPERATOR_PREFIXES)


# Return the standalone four-digit years in the query, skipping years that belong to date operators.
def _query_years(query: str) -> list[str]:
    return [
        match.group(1)
        for match in _ANY_YEAR_RE.finditer(query or "")
        if not _is_date_operator_year(query, match.start())
    ]


# Words that mark any year in the query as a freshness hint, not a topic anchor.
_TRAILING_YEAR_FRESHNESS_HINTS: frozenset[str] = frozenset({
    "latest", "recent", "current", "now", "today", "yesterday",
    "update", "updates", "news", "breaking", "headline",
    # Russian / Ukrainian
    "новости", "последние", "сейчас", "сегодня", "новини", "зараз", "сьогодні",
    # German / French / Dutch
    "aktuell", "neueste", "récent", "actuels", "laatste",
})


# Return the more restrictive of two timelimits (None = no restriction).
def stricter_timelimit(a: Optional[str], b: Optional[str]) -> Optional[str]:
    if a is None:
        return b
    if b is None:
        return a
    return a if _TIMELIMIT_ORDER.get(a, 99) <= _TIMELIMIT_ORDER.get(b, 99) else b


# Return True when the query anchors to a year before last calendar year (a historical lookup,
# so freshness windows should not apply).
def _has_historical_year_anchor(query: str) -> bool:
    years = _query_years(query or "")
    if not years:
        return False
    return min(int(y) for y in years) < _dt.date.today().year - 1


# Map a trailing/comma/freshness-tagged year in the query to a timelimit: `current` when the year is
# this year or later, `prev` for last year and `older` for anything earlier. Return None when the
# query has no such tag or no year.
def _year_hint_timelimit(
    query: str, *, current: Optional[str], prev: Optional[str], older: Optional[str]
) -> Optional[str]:
    # A year only counts as a time hint when the query also has a freshness word or a comma-tagged year.
    lower = query.lower()
    has_freshness = any(hint in lower for hint in _TRAILING_YEAR_FRESHNESS_HINTS)
    has_comma_year = bool(_TRAILING_YEAR_COMMA_RE.search(query))
    if not has_freshness and not has_comma_year:
        return None
    years = _query_years(query)
    if not years:
        return None

    # Pick the timelimit by how far the newest year is from today.
    this_year = _dt.date.today().year
    year = max(int(y) for y in years)
    if year >= this_year:
        return current
    if year == this_year - 1:
        return prev
    return older


# Strip year tokens used as freshness hints (comma-trailing or with freshness words).
def _strip_trailing_year(query: str) -> str:
    # A comma-tagged year is always a time tag: remove it unless that would empty the query.
    if _TRAILING_YEAR_COMMA_RE.search(query):
        cleaned = _TRAILING_YEAR_COMMA_RE.sub("", query).strip()
        return cleaned or query

    # With a freshness word, remove only recent years.
    lower = query.lower()
    if any(hint in lower for hint in _TRAILING_YEAR_FRESHNESS_HINTS):
        # Strip only a year acting as a recency tag (this/last year). An older year next to
        # a freshness word is far more likely a topic anchor — a product/standard version
        # ("Windows Server 2022 latest CU", "ISO 27001 2022 current") — and must survive into
        # the engine query, or the search loses what it's actually about.
        cutoff = _dt.date.today().year - 1

        # Replacement callback: drop a matched year (or range, judged by its first year) that is
        # recent, keep older ones and the values of date operators.
        def drop_recent(match: re.Match[str]) -> str:
            if _is_date_operator_year(query, match.start()):
                return match.group(0)
            return "" if int(match.group(0)[:4]) >= cutoff else match.group(0)

        cleaned = " ".join(_YEAR_ANYWHERE_RE.sub(drop_recent, query).split())
        return cleaned if cleaned.strip() else query
    return query


# Apply the configured year-hint policy: optionally derive a timelimit from a year token
# in the query and strip that token. Return (possibly-cleaned query, derived timelimit).
# The policy is the `year_hint_mode` setting of the `query` config section, defaulting to
# "timelimit" (also for unknown values):
#   timelimit - derive a timelimit from the year and strip the token;
#   strip     - strip the year token but derive no timelimit;
#   none      - leave the query untouched.
# A query anchored to a historical year never gets a derived timelimit.
def _apply_year_hint_policy(query: str, qcfg: object) -> tuple[str, Optional[str]]:
    mode = str(getattr(qcfg, "year_hint_mode", "timelimit") or "timelimit").strip().lower()
    if mode not in {"timelimit", "strip", "none"}:
        mode = "timelimit"
    if mode == "none":
        return query, None

    # Derive the timelimit only in timelimit mode, then strip the year hint (timelimit and strip modes).
    year_tl = None
    if mode == "timelimit" and not _has_historical_year_anchor(query):
        year_tl = _year_hint_timelimit(
            query,
            current=getattr(qcfg, "year_hint_current", "m"),
            prev=getattr(qcfg, "year_hint_prev", "y"),
            older=getattr(qcfg, "year_hint_older", None),
        )
    return _strip_trailing_year(query), year_tl


# Public entry: fold the query's own date intent into the search. A query often encodes its own
# time intent: a trailing year ("AI agents 2025"), a comma-tagged year ("AI agents, 2024-2025") or a
# freshness word ("latest ..."). That becomes a DDGS-style timelimit (d/w/m/y) and the year token is
# stripped so it does not skew the lexical match.
#
# Return the query to actually send to engines (year token stripped per policy) and the
# timelimit to use — the stricter of any explicit caller timelimit and the one derived
# from the query's year/freshness tokens.
def resolve_query_dates(
    query: str, qcfg: object, explicit_timelimit: Optional[str] = None
) -> tuple[str, Optional[str]]:
    clean_query, year_tl = _apply_year_hint_policy(query, qcfg)
    return clean_query, stricter_timelimit(explicit_timelimit, year_tl)
