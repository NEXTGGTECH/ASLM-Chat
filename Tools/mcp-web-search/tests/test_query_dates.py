# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import datetime
from types import SimpleNamespace

from core.search.query_dates import resolve_query_dates, stricter_timelimit

# Current calendar year, used to build queries with a recent year tag.
_THIS_YEAR = datetime.date.today().year


# Build a stand-in query config: the year hint mode plus the time limits for the current, previous and
# older years.
def _cfg(mode: str = "timelimit"):
    return SimpleNamespace(
        year_hint_mode=mode, year_hint_current="m", year_hint_prev="y", year_hint_older=None
    )


# Verify that with the year hint mode "none" the query is returned unchanged and no time limit is set.
def test_mode_none_is_noop():
    q = f"openai latest news {_THIS_YEAR}"
    assert resolve_query_dates(q, _cfg("none")) == (q, None)


# Verify that a query without a year or freshness word is returned untouched with no time limit.
def test_plain_query_untouched():
    assert resolve_query_dates("python asyncio tutorial", _cfg()) == ("python asyncio tutorial", None)


# Verify that a freshness word plus the current year strips the year from the query and derives the
# current-year time limit ("m").
def test_freshness_plus_current_year_derives_and_strips():
    clean, tl = resolve_query_dates(f"gpt-5 release latest news {_THIS_YEAR}", _cfg())
    assert str(_THIS_YEAR) not in clean
    assert "latest news" in clean
    assert tl == "m"


# Verify that a freshness word plus last year derives the previous-year time limit ("y").
def test_last_year_freshness_derives_prev():
    _, tl = resolve_query_dates(f"foldable phones current {_THIS_YEAR - 1}", _cfg())
    assert tl == "y"


# Verify that a year range trailing a comma is stripped from the query.
def test_comma_trailing_year_is_stripped():
    clean, _ = resolve_query_dates("ai developments, 2023-2024", _cfg())
    assert clean == "ai developments"


# Verify that without a freshness word or a comma the year is a topic anchor: the query is left in place
# and no time limit is derived.
def test_topic_year_without_freshness_is_kept():
    clean, tl = resolve_query_dates("rust async 2019", _cfg())
    assert clean == "rust async 2019" and tl is None


# Verify that a freshness word anchored to an old year gets no freshness window.
def test_historical_anchor_suppresses_timelimit():
    _, tl = resolve_query_dates("history of computing news 2005", _cfg())
    assert tl is None


# Verify that an explicit time limit "d" is kept because it is stricter than the derived "m".
def test_explicit_timelimit_wins_when_stricter():
    _, tl = resolve_query_dates(f"news latest {_THIS_YEAR}", _cfg(), explicit_timelimit="d")
    assert tl == "d"


# Verify that stricter_timelimit picks the tighter of two limits and treats None as "no limit".
def test_stricter_timelimit_helper():
    assert stricter_timelimit("d", "y") == "d"
    assert stricter_timelimit(None, "w") == "w"
    assert stricter_timelimit("m", None) == "m"
    assert stricter_timelimit(None, None) is None


# Verify that an old year such as "2022" (a product version, not a recency tag) is not stripped just
# because a freshness word is present, or the engine query would lose what it is about (#1).
def test_old_entity_year_survives_freshness_word():
    clean, tl = resolve_query_dates("Windows Server 2022 latest CU", _cfg())
    assert "2022" in clean
    assert "latest CU" in clean

    # An old year next to a freshness word is a historical anchor, so no freshness time limit.
    assert tl is None


# Verify that the current year is a recency tag: it is stripped under a freshness word and gives "m".
def test_recent_year_still_stripped_under_freshness_word():
    clean, tl = resolve_query_dates(f"AI agents news {_THIS_YEAR}", _cfg())
    assert str(_THIS_YEAR) not in clean
    assert tl == "m"
