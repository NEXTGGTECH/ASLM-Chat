# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio

from core.search.health import (
    BreakerState,
    DEGRADATION_COOLDOWN,
    ERROR_COOLDOWN,
    PROBE_TIMEOUT,
    EngineHealthTracker,
)
from core.search.quality import (
    extract_date,
    hub_penalty,
    infer_query_language,
    is_skip_title,
    markdown_meta_date,
    page_date_score,
    query_years,
)
from core.search.triage import TriageAction, TriageSession
from core.search.web_search import (
    _citation_id,
    _inline_parse_allowed,
    _make_search_id,
    select_engines,
)


# Quality

# Verify that search ids increase linearly (each id is the previous one plus one in hex), that the
# numeric part is at least four digits long, and that citation ids keep the full search-id namespace
# without truncating it.
def test_search_ids_are_linear_and_citation_namespace_is_not_truncated():
    first = _make_search_id()
    second = _make_search_id()

    assert int(second.removeprefix("srch_"), 16) == int(first.removeprefix("srch_"), 16) + 1
    assert len(first.removeprefix("srch_")) >= 4
    assert _citation_id("srch_0abc", 3) == "c0abc-3"
    assert _citation_id("srch_10000", 1) == "c10000-1"


# Verify that a category/listing URL gets a hub penalty of at least 0.5 while a real article with a
# snippet gets none.
def test_hub_penalty_flags_category_pages():
    assert hub_penalty("https://site.com/category/news/", "All news", "") >= 0.5
    assert hub_penalty("https://site.com/blog/deep-dive-asyncio", "Deep dive", "long snippet here") == 0.0


# Verify that login and error-page titles are recognised as skip titles and a normal title is not.
def test_skip_title_patterns():
    assert is_skip_title("Login - example.com")
    assert is_skip_title("404 Not Found")
    assert not is_skip_title("Understanding asyncio")


# Verify the soft year policy: year alignment keys off a real extracted date (never a bare year token),
# a matching date scores 1.0, a mismatching one -0.3 and a missing date 0.0 (no signal is not a penalty).
def test_year_policy_soft_only():
    years = query_years("best gpu 2024")
    assert years == ["2024"]
    assert page_date_score(extract_date("Mar 12, 2024 — benchmark roundup"), years) == 1.0
    assert page_date_score(extract_date("15.06.2021 — обзор видеокарт"), years) == -0.3
    assert page_date_score(None, years) == 0.0


# Verify that date extraction needs a real date: a bare year in a title (the SEO-farm move) does not
# read as a date, while English, Russian, numeric, ISO and month-year dates and relative "ago" forms do.
def test_extract_date_requires_a_real_date_not_a_year_token():
    assert extract_date("GPT 5 Mini API Pricing 2026 - Costs, Performance & Providers") is None
    assert extract_date("Jul 3, 2026 — OpenAI updated its API pricing.") == (2026, 7, 3)
    assert extract_date("3 июл. 2026 г. — обновлены тарифы API.") == (2026, 7, 3)
    assert extract_date("03.07.2026 · новости") == (2026, 7, 3)
    assert extract_date("2026-07-03 release notes") == (2026, 7, 3)
    assert extract_date("July 2026 changelog") == (2026, 7, 1)
    import datetime as dt
    assert extract_date("2 days ago — thread", now=dt.date(2026, 7, 4)) == (2026, 7, 2)
    assert extract_date("5 часов назад — новость", now=dt.date(2026, 7, 4)) == (2026, 7, 4)


# Verify that date extraction handles localized month names, CJK numeric forms, relative forms in
# several languages, and that product names or year-adjacent words never read as dates.
def test_extract_date_multilingual():
    import datetime as dt

    # Textual dates with localized month names (day required outside en/ru).
    assert extract_date("3. Juli 2026 — Pressemitteilung") == (2026, 7, 3)
    assert extract_date("3 de julio de 2026 — anuncio oficial") == (2026, 7, 3)
    assert extract_date("5 października 2026 — aktualizacja") == (2026, 10, 5)
    assert extract_date("3 липня 2026 — новини") == (2026, 7, 3)
    assert extract_date("14 Şubat 2026 tarihinde yayınlandı") == (2026, 2, 14)
    assert extract_date("3 ottobre 2026 — comunicato") == (2026, 10, 3)

    # CJK numeric forms need no month names at all.
    assert extract_date("2026年7月3日に公開") == (2026, 7, 3)
    assert extract_date("2026년 7월 3일 발표") == (2026, 7, 3)
    assert extract_date("03/07/2026 mise à jour") == (2026, 7, 3)

    # Relative: prefix (de/fr/es) and CJK forms.
    assert extract_date("vor 3 Tagen aktualisiert", now=dt.date(2026, 7, 4)) == (2026, 7, 1)
    assert extract_date("il y a 2 jours — actu", now=dt.date(2026, 7, 4)) == (2026, 7, 2)
    assert extract_date("hace 2 días — noticia", now=dt.date(2026, 7, 4)) == (2026, 7, 2)
    assert extract_date("3天前更新", now=dt.date(2026, 7, 4)) == (2026, 7, 1)
    assert extract_date("2일 전 게시", now=dt.date(2026, 7, 4)) == (2026, 7, 2)

    # Anti-gaming: prefix+tail words and product names never read as dates.
    assert extract_date("Marketing 2026 trends report") is None
    assert extract_date("iPhone 17 Pro 2026 review roundup") is None
    assert extract_date("Top 10 LED 2026 TVs compared") is None

    # The Spanish "ago" (August) must not hijack English "years ago" next to a year.
    d = extract_date("updated 3 years ago. 2026 predictions", now=dt.date(2026, 7, 4))
    assert d is not None and d[0] == 2023
    assert extract_date("лучшие инструменты 2026") is None


# Verify that without query years the page date score decays with age: a page from a few days ago
# scores above 0.9, a two-year-old page between 0 and 0.2, and a missing date 0.0.
def test_page_date_freshness_decays_without_query_years():
    import datetime as dt
    now = dt.date(2026, 7, 4)
    fresh = page_date_score((2026, 7, 1), [], now=now)
    old = page_date_score((2024, 7, 1), [], now=now)
    assert fresh > 0.9
    assert 0.0 < old < 0.2
    assert page_date_score(None, [], now=now) == 0.0


# Verify that the date is read from the normalized markdown meta head, that text without a meta head
# gives None, and that a date equal to today or in the future is ignored.
def test_markdown_meta_date_reads_normalized_head():
    import datetime as dt
    now = dt.date(2026, 7, 5)
    md = "**Title:** Docs\n**Site:** openai.com\n**Date:** 2026-06-30\n\n---\n\nBody text."
    assert markdown_meta_date(md, now=now) == (2026, 6, 30)
    assert markdown_meta_date("no meta head here", now=now) is None

    # Templates that render the CURRENT date (fossil, dashboards) must not read as
    # eternally fresh; future dates are junk.
    today = "**Title:** X\n**Site:** sqlite.org\n**Date:** 2026-07-05\n\n---\n\nBody."
    future = "**Title:** X\n**Site:** ex.com\n**Date:** 2027-01-01\n\n---\n\nBody."
    assert markdown_meta_date(today, now=now) is None
    assert markdown_meta_date(future, now=now) is None


# Verify that the query language is inferred from the script: Latin gives en, Cyrillic ru, Japanese ja.
def test_infer_query_language_scripts():
    assert infer_query_language("how to use asyncio") == "en"
    assert infer_query_language("как настроить роутер") == "ru"
    assert infer_query_language("非同期処理 チュートリアル") == "ja"


# Triage

# Ingest one source into the triage session with default engine, rank, URL, title and a long snippet,
# and return the triage decision. Any default can be overridden per test.
def _ingest(session: TriageSession, *, engine="google", family="google", rank=1,
            url="https://ex.com/a", title="Python asyncio tutorial",
            snippet="A long, detailed walkthrough of asyncio coroutines and tasks in Python."):
    return session.ingest_source(
        engine=engine, provider_family=family, rank=rank, url=url, title=title, snippet=snippet
    )


# Verify that a relevant top-ranked Google result is parsed immediately.
def test_triage_relevant_top_google_parses_immediately():
    session = TriageSession("python asyncio tutorial")
    decision = _ingest(session)
    assert decision.action == TriageAction.PARSE


# Verify that a result with a skip title (a login page) is skipped.
def test_triage_skip_title_is_skipped():
    session = TriageSession("python asyncio")
    decision = _ingest(session, title="Login required")
    assert decision.action == TriageAction.SKIP


# Verify that revisiting a queued page from a better engine re-prices it to the best positional view
# and upgrades it to a parse.
def test_triage_revisit_takes_best_positional_view_and_upgrades():
    session = TriageSession("python asyncio tutorial")

    # Weak engine, mid rank: lands in the queue.
    decision = _ingest(session, engine="qwant", family="qwant", rank=5)
    assert decision.action == TriageAction.QUEUE

    # The same page then surfaces at Google rank 1: the base must be re-priced to the
    # strongest single-engine view instead of the initial Qwant rank 5; the second
    # family's vote lands on top, giving a PARSE upgrade.
    upgraded = session.ingest_revisit(
        engine="google", provider_family="google", rank=1,
        url="https://ex.com/a", title="Python asyncio tutorial",
        snippet="A long, detailed walkthrough of asyncio coroutines and tasks in Python.",
    )
    assert upgraded is not None and upgraded.upgraded
    assert upgraded.action == TriageAction.PARSE


# Verify that a vote's weight depends on the engine family: a Google vote raises the score of a queued
# page more than a Yandex vote, and a Yandex vote still raises it.
def test_triage_vote_weight_follows_family_class():
    # Return the score increase a vote from the given family gives to a queued Qwant result.
    def bump_from(family: str) -> float:
        session = TriageSession("python asyncio tutorial")
        _ingest(session, engine="qwant", family="qwant", rank=5)
        before = session.score_of("https://ex.com/a")
        session.ingest_vote(provider_family=family, url="https://ex.com/a")
        return session.score_of("https://ex.com/a") - before

    # A Google vote must be worth more than a Yandex vote (deliberately floored engine).
    assert bump_from("google") > bump_from("yandex") > 0.0


# Verify that engines of one family vote only once: the first Google-family vote raises the score of a
# queued page, a second vote from the same family (as Startpage shares Google's family) is ignored.
def test_triage_google_startpage_one_family_vote():
    session = TriageSession("python asyncio tutorial")
    decision = _ingest(session, engine="qwant", family="qwant", rank=5)
    assert decision.action == TriageAction.QUEUE
    score_before = session.score_of("https://ex.com/a")
    session.ingest_vote(provider_family="google", url="https://ex.com/a")
    score_after_google = session.score_of("https://ex.com/a")

    # Startpage shares Google's family, so there is no second bump.
    assert session.ingest_vote(provider_family="google", url="https://ex.com/a") is None
    assert score_after_google > score_before
    assert session.score_of("https://ex.com/a") == score_after_google


# Health and circuit breaker

# Manually advanced fake clock: calling it returns the current fake time in seconds.
class _Clock:
    # Start the fake time at zero.
    def __init__(self) -> None:
        self.now = 0.0

    # Return the current fake time.
    def __call__(self) -> float:
        return self.now


# Verify that a blocked result opens the breaker for the error cooldown, then lets exactly one
# half-open probe through.
def test_breaker_error_opens_for_five_minutes():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    assert tracker.allow("ddg")
    tracker.record("ddg", status="blocked", fetch_ms=200, results=0)
    assert not tracker.allow("ddg")
    clock.now = ERROR_COOLDOWN - 1
    assert not tracker.allow("ddg")
    clock.now = ERROR_COOLDOWN + 1

    # Half-open probe, and only one probe at a time.
    assert tracker.allow("ddg")
    assert not tracker.allow("ddg")


# Verify that a 5x latency spike trips the breaker for the shorter degradation cooldown, and that a
# healthy probe afterwards closes it again.
def test_breaker_degradation_short_cooldown_and_recovery():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)

    # Build a latency baseline.
    tracker.record("yandex", status="success", fetch_ms=500, results=10)

    # A 5x spike trips the degradation breaker.
    tracker.record("yandex", status="success", fetch_ms=2500, results=10)
    assert not tracker.allow("yandex")
    clock.now = DEGRADATION_COOLDOWN + 1
    assert tracker.allow("yandex")
    tracker.record("yandex", status="success", fetch_ms=520, results=10)

    # Move past the jittered Stage C pace gate set by the probe fire; the breaker is closed again.
    clock.now += 5
    assert tracker.allow("yandex")


# Verify that a failed half-open probe reopens the breaker with a doubled cooldown.
def test_breaker_failed_probe_backs_off_exponentially():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    tracker.record("brave", status="blocked", fetch_ms=100, results=0)
    clock.now = ERROR_COOLDOWN + 1

    # The probe is admitted and then fails.
    assert tracker.allow("brave")
    tracker.record("brave", status="blocked", fetch_ms=100, results=0)
    health = tracker._health("brave")
    assert health.cooldown == ERROR_COOLDOWN * 2
    assert health.state == BreakerState.OPEN


# Verify that a half-open probe whose outcome is never recorded (e.g. the search deadline dropped the
# engine's status event) is reclaimed after the probe timeout instead of locking the engine out forever.
def test_breaker_abandoned_probe_is_expired_not_wedged():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    tracker.record("qwant", status="blocked", fetch_ms=100, results=0)
    clock.now = ERROR_COOLDOWN + 1

    # The probe is admitted, then held back while it is still in flight.
    assert tracker.allow("qwant")
    assert not tracker.allow("qwant")

    # Outcome never arrives; once the probe ages past the timeout it is reclaimed.
    clock.now += PROBE_TIMEOUT + 1
    assert tracker.allow("qwant")


# Stage C pacing

# Verify that an engine is held back within its minimum interval (about 6 s) and allowed again once the
# jittered interval (at most 6 * 1.25 = 7.5 s) has elapsed.
def test_pacing_holds_engine_within_min_interval():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)

    # The first fire is admitted (and paced), the second one within the interval is held back.
    assert tracker.allow("brave")
    assert not tracker.allow("brave")

    # Past the jittered interval the engine is allowed again.
    clock.now += 8.0
    assert tracker.allow("brave")


# Verify that pacing does not apply to a tolerant engine: duckduckgo has a minimum interval of 0, so an
# immediate re-fire is allowed.
def test_pacing_skipped_for_tolerant_engines():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    assert tracker.allow("duckduckgo")
    assert tracker.allow("duckduckgo")


# Engine selection

# All engines enabled; this isolates the breaker behaviour from the config layer.
_ALL_ON = {n: True for n in ("google", "duckduckgo", "startpage", "qwant", "brave", "yandex")}


# Verify that low effort selects yandex and duckduckgo as the default pair.
def test_select_engines_low_default_pair():
    tracker = EngineHealthTracker(clock=_Clock())
    names = [e.name for e in select_engines("low", tracker, enabled=_ALL_ON, buffer=False)]
    assert names == ["yandex", "duckduckgo"]


# Verify that low effort falls back to startpage when both default engines are breaker-blocked.
def test_select_engines_low_falls_back_to_startpage():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    tracker.record("yandex", status="blocked", fetch_ms=100, results=0)
    tracker.record("duckduckgo", status="blocked", fetch_ms=100, results=0)
    names = [e.name for e in select_engines("low", tracker, enabled=_ALL_ON, buffer=False)]
    assert names == ["startpage"]


# Verify that low effort never returns an empty selection: with all candidates breaker-blocked the first
# enabled lead (duckduckgo) is forced through the open breaker.
def test_select_engines_low_never_empty():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    for engine in ("yandex", "duckduckgo", "startpage"):
        tracker.record(engine, status="blocked", fetch_ms=100, results=0)
    names = [e.name for e in select_engines("low", tracker, enabled=_ALL_ON, buffer=False)]
    assert names == ["duckduckgo"]


# Verify that the legacy Yandex widget, the emergency buffer, never appears in any tier while regular
# engines are alive, even when the buffer switch is on.
def test_select_engines_buffer_not_in_normal_rotation():
    tracker = EngineHealthTracker(clock=_Clock())
    names = [e.name for e in select_engines("high", tracker, enabled=_ALL_ON, buffer=True)]
    assert "legacy_yandex" not in names
    assert "duckduckgo" in names


# Verify that when every regular engine is breaker-blocked the legacy Yandex buffer takes the load.
def test_select_engines_buffer_fires_when_all_regular_dead():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    for engine in ("yandex", "duckduckgo", "startpage", "google", "qwant", "brave"):
        tracker.record(engine, status="blocked", fetch_ms=100, results=0)
    names = [e.name for e in select_engines("high", tracker, enabled=_ALL_ON, buffer=True)]
    assert names == ["legacy_yandex"]


# Verify that when every regular engine is switched off in config the legacy Yandex buffer takes the
# load.
def test_select_engines_buffer_fires_when_all_disabled():
    tracker = EngineHealthTracker(clock=_Clock())
    all_off = {n: False for n in _ALL_ON}
    names = [e.name for e in select_engines("low", tracker, enabled=all_off, buffer=True)]
    assert names == ["legacy_yandex"]


# Verify that when the buffer engine is itself breaker-blocked, selection falls through to the first
# enabled lead (duckduckgo), forced through its open breaker.
def test_select_engines_buffer_respects_own_breaker():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    for engine in ("yandex", "duckduckgo", "startpage", "legacy_yandex"):
        tracker.record(engine, status="blocked", fetch_ms=100, results=0)
    names = [e.name for e in select_engines("low", tracker, enabled=_ALL_ON, buffer=True)]
    assert names == ["duckduckgo"]


# Verify that while Google is solidly closed it owns the family slot alone and Startpage stays a warm
# standby (not selected) at medium effort.
def test_select_engines_medium_google_healthy_startpage_stays_standby():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    names = [e.name for e in select_engines("medium", tracker, enabled=_ALL_ON)]
    assert "google" in names
    assert "startpage" not in names


# Verify that when Google's breaker is open, medium effort drops Google and selects Startpage as the hot
# standby of the Google family.
def test_select_engines_medium_startpage_standby_when_google_open():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    tracker.record("google", status="blocked", fetch_ms=100, results=0)
    names = [e.name for e in select_engines("medium", tracker, enabled=_ALL_ON)]
    assert "google" not in names
    assert "startpage" in names


# Verify that when Google's cooldown has elapsed the next selection sends a half-open probe and
# Startpage rides along. Because recovery is unproven, a still-blocked probe must not cost the search its
# Google-family result (the "remember Google is bad" fix).
def test_select_engines_medium_startpage_rides_along_on_google_probe():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    tracker.record("google", status="blocked", fetch_ms=100, results=0)
    clock.now = ERROR_COOLDOWN + 1
    names = [e.name for e in select_engines("medium", tracker, enabled=_ALL_ON)]

    # The probe fires and the safety net rides along.
    assert "google" in names
    assert "startpage" in names
    assert tracker._health("google").state == BreakerState.HALF_OPEN


# Verify that is_healthy is a pure read: a never-seen engine is healthy, an open or unprobed one is
# not, and the call does not consume the half-open probe that allow() still admits.
def test_is_healthy_pure_read_no_side_effects():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)

    # Never seen: closed, hence healthy. Open: unhealthy.
    assert tracker.is_healthy("google")
    tracker.record("google", status="blocked", fetch_ms=100, results=0)
    assert not tracker.is_healthy("google")

    # Cooldown elapsed but unprobed: still unhealthy.
    clock.now = ERROR_COOLDOWN + 1
    assert not tracker.is_healthy("google")

    # Pure read must not have consumed the probe: allow() still admits it.
    assert tracker.allow("google")


# Per-engine config switches

# Verify that with the shipped defaults both Yandex engines are off and appear in no tier of the
# widest (high) fan-out, while the duckduckgo lead survives.
def test_select_engines_default_config_excludes_yandex_engines():
    from core.config.settings import EnginesSection

    tracker = EngineHealthTracker(clock=_Clock())
    names = [e.name for e in select_engines("high", tracker, enabled=EnginesSection().as_map())]
    assert "yandex" not in names
    assert "legacy_yandex" not in names
    assert "duckduckgo" in names


# Verify that a user who enables native Yandex gets it in the high tier, while legacy_yandex keeps its
# buffer-only role and never joins the rotation.
def test_select_engines_user_opts_native_yandex_back_in():
    from core.config.settings import EnginesSection

    enabled = EnginesSection(yandex=True).as_map()
    tracker = EngineHealthTracker(clock=_Clock())
    names = [e.name for e in select_engines("high", tracker, enabled=enabled, buffer=False)]
    assert "yandex" in names
    assert "legacy_yandex" not in names


# Verify that even the never-empty last resort respects the config: with yandex off and every other
# breaker open, the forced pick is the first enabled lead (duckduckgo), not yandex.
def test_select_engines_disabled_engine_never_forced():
    from core.config.settings import EnginesSection

    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    for engine in ("duckduckgo", "startpage"):
        tracker.record(engine, status="blocked", fetch_ms=100, results=0)
    names = [e.name for e in select_engines("low", tracker, enabled=EnginesSection().as_map(), buffer=False)]
    assert names == ["duckduckgo"]


# Verify the misconfiguration guard: with every engine disabled the selection still returns the
# duckduckgo lead instead of zero engines.
def test_select_engines_all_disabled_still_returns_lead():
    tracker = EngineHealthTracker(clock=_Clock())
    all_off = {n: False for n in _ALL_ON}
    names = [e.name for e in select_engines("low", tracker, enabled=all_off, buffer=False)]
    assert names == ["duckduckgo"]


# Verify that the engines and buffer_engine config sections load from a file: explicit values are
# honoured and absent ones keep their defaults (engines on, buffer on).
def test_engines_config_section_loads(tmp_path):
    import json as _json

    from core.config.settings import load_search_config

    cfg_path = tmp_path / "search_config.json"
    cfg_path.write_text(
        _json.dumps({"engines": {"yandex": True, "brave": False}}), encoding="utf-8"
    )
    cfg = load_search_config(cfg_path)

    # User opt-in and explicit off are honoured; absent entries keep their defaults.
    assert cfg.engines.yandex is True
    assert cfg.engines.brave is False
    assert cfg.engines.duckduckgo is True
    assert cfg.buffer_engine.enabled is True

    cfg_path.write_text(
        _json.dumps({"buffer_engine": {"enabled": False}}), encoding="utf-8"
    )
    cfg = load_search_config(cfg_path)

    # An explicit off for the buffer engine is honoured.
    assert cfg.buffer_engine.enabled is False


# Inline-parse policy: custom-domain scope and learned-slow domains

# Verify that browser-heavy custom domains (Reddit, YouTube) are marked read_page-only, while API-backed
# handlers (GitHub) and plain domains without a handler stay usable in web_search.
def test_custom_domain_scope_marks_browser_heavy_readpage_only():
    import custom_domains

    assert custom_domains.is_read_page_only("https://www.reddit.com/r/Python/comments/a/b/")
    assert custom_domains.is_read_page_only("https://www.youtube.com/watch?v=abc")

    # API-backed handlers stay usable in web_search.
    assert not custom_domains.is_read_page_only("https://github.com/python/cpython")

    # A plain domain with no handler is unaffected.
    assert not custom_domains.is_read_page_only("https://example.com/article")


# Verify that inline parsing is not allowed for a URL whose handler is read_page-only.
def test_inline_parse_blocks_readpage_only_handler():
    assert not _inline_parse_allowed("https://www.reddit.com/r/Python/comments/a/b/")


# Verify that inline parsing is blocked for every Reddit hostname (reddit.com, www, old, redd.it) but
# not for a lookalike domain.
def test_inline_parse_blocks_every_reddit_hostname_but_not_lookalikes():
    assert not _inline_parse_allowed("https://reddit.com/r/meshtastic/hot/")
    assert not _inline_parse_allowed("https://www.reddit.com/r/meshtastic/hot/")
    assert not _inline_parse_allowed("https://old.reddit.com/r/meshtastic/")
    assert not _inline_parse_allowed("https://redd.it/abc123")
    assert _inline_parse_allowed("https://not-reddit.com/article")


# Verify that a domain remembered as slow (9 s expected fetch) is not parsed inline and falls back to
# snippet-only, while a fast one (0.8 s) is parsed inline.
def test_inline_parse_skips_learned_slow_domain(monkeypatch):
    import core.profiles as profiles
    from core.profiles.models import ProfileHint

    # Fake runtime profiles that report the same expected fetch time for every domain.
    class _Profiles:
        # Store the expected fetch time in milliseconds.
        def __init__(self, ms):
            self._ms = ms

        # Return an httpx hint with the configured expected fetch time.
        def best_method(self, _domain):
            return ProfileHint(method="httpx", expected_fetch_ms=self._ms, confidence=0.9)

    monkeypatch.setattr(profiles, "get_runtime_profiles", lambda: _Profiles(9_000.0))
    assert not _inline_parse_allowed("https://slow.com/a")
    monkeypatch.setattr(profiles, "get_runtime_profiles", lambda: _Profiles(800.0))
    assert _inline_parse_allowed("https://fast.com/a")


# Verify that a domain with no handler and no profile data yet is parsed inline, so the profile store
# can learn from that parse.
def test_inline_parse_allows_unknown_domain():
    assert _inline_parse_allowed("https://brand-new-domain-xyz.com/a")


# Orchestrator with a synthetic stream and a fake reader

# Fake SerpApi replacement that replays the scripted events class attribute as its search stream.
class _FakeSerpApi:
    events: list[dict] = []

    # Accept and ignore the constructor arguments of the real SerpApi.
    def __init__(self, *args, **kwargs) -> None:
        pass

    # Yield the scripted events one by one.
    async def search_stream(self, *args, **kwargs):
        for event in self.events:
            yield event


# Fake page reader that returns a markdown stub naming the URL.
async def _fake_reader(
    url: str, *, timeout: float = 0, max_chars: int = 0, focus: str = "",
    allow_browser: bool = True,
) -> str:
    await asyncio.sleep(0)
    return f"# parsed {url}"


# Verify that a medium-effort search parses the winning source and ranks it first by consensus and rank:
# the top URL is voted for by both google and yandex families, is parsed to markdown, and the google
# breaker stays closed.
def test_web_search_medium_parses_winners_and_ranks(monkeypatch):
    import core.search.web_search as ws

    # Arrange: scripted events with a strong Google result, a weak Qwant result, a Yandex vote for the
    # Google result and the Google engine summary.
    _FakeSerpApi.events = [
        {
            "type": "source",
            "engine": "google",
            "provider_family": "google",
            "rank": 1,
            "url": {"url": "https://good.com/asyncio-guide", "host": "good.com"},
            "serp": {
                "title": "Python asyncio tutorial",
                "snippet": "A long, detailed walkthrough of asyncio coroutines and tasks in Python.",
                "fetch_ms": 400,
                "parse_ms": 2,
            },
        },
        {
            "type": "source",
            "engine": "qwant",
            "provider_family": "qwant",
            "rank": 9,
            "url": {"url": "https://meh.com/page", "host": "meh.com"},
            "serp": {"title": "asyncio", "snippet": "asyncio tutorial", "fetch_ms": 900, "parse_ms": 2},
        },
        {
            "type": "vote",
            "engine": "yandex",
            "provider_family": "yandex",
            "rank": 2,
            "url": {"url": "https://good.com/asyncio-guide", "host": "good.com"},
        },
        {
            "type": "engine",
            "engine": "google",
            "payload": {
                "engine": "google",
                "provider_family": "google",
                "status": "success",
                "fetch_ms": 400.0,
                "sources": [{"url": "https://good.com/asyncio-guide"}],
            },
        },
    ]
    monkeypatch.setattr(ws, "SerpApi", _FakeSerpApi)
    monkeypatch.setattr(ws, "_get_transport", lambda *_: None)

    service = ws.WebSearchService(tracker=EngineHealthTracker(clock=_Clock()), read_page=_fake_reader)
    result = asyncio.run(service.search("python asyncio tutorial", effort="medium"))

    # Consensus plus rank put the Google result on top.
    urls = [s["url"] for s in result["sources"]]
    assert urls[0] == "https://good.com/asyncio-guide"
    top = result["sources"][0]
    assert top["consensus_families"] == ["google", "yandex"]
    assert top["parsed_ok"] and top["markdown"].startswith("# parsed")
    assert result["health"]["google"]["state"] == "closed"


# Verify that the search query is passed to the page reader as the compaction focus, and that browser
# escalation is not allowed at medium effort.
def test_web_search_passes_query_as_focus_to_reader(monkeypatch):
    import core.search.web_search as ws

    _FakeSerpApi.events = [
        {
            "type": "source", "engine": "google", "provider_family": "google", "rank": 1,
            "url": {"url": "https://good.com/p", "host": "good.com"},
            "serp": {"title": "T", "snippet": "a detailed snippet about the topic at hand",
                     "fetch_ms": 1, "parse_ms": 1},
        },
        {
            "type": "engine", "engine": "google",
            "payload": {"engine": "google", "provider_family": "google", "status": "success",
                        "fetch_ms": 1.0, "sources": [{"url": "https://good.com/p"}]},
        },
    ]
    monkeypatch.setattr(ws, "SerpApi", _FakeSerpApi)
    monkeypatch.setattr(ws, "_get_transport", lambda *_: None)

    captured: dict = {}

    # Capture the focus and browser flag the service passes to the reader.
    async def capturing_reader(url, *, timeout=0, max_chars=0, focus="", allow_browser=True):
        captured["focus"] = focus
        captured["allow_browser"] = allow_browser
        return f"# parsed {url}"

    service = ws.WebSearchService(tracker=EngineHealthTracker(clock=_Clock()), read_page=capturing_reader)
    asyncio.run(service.search("attention mechanism", effort="medium"))

    # The query is the compaction focus.
    assert captured["focus"] == "attention mechanism"
    assert captured["allow_browser"] is False


# Verify that high effort allows the page reader to escalate to the browser.
def test_web_search_high_allows_browser_escalation(monkeypatch):
    import core.search.web_search as ws

    _FakeSerpApi.events = [
        {
            "type": "source", "engine": "google", "provider_family": "google", "rank": 1,
            "url": {"url": "https://good.com/p", "host": "good.com"},
            "serp": {"title": "T", "snippet": "a detailed snippet about the topic at hand",
                     "fetch_ms": 1, "parse_ms": 1},
        },
        {
            "type": "engine", "engine": "google",
            "payload": {"engine": "google", "provider_family": "google", "status": "success",
                        "fetch_ms": 1.0, "sources": [{"url": "https://good.com/p"}]},
        },
    ]
    monkeypatch.setattr(ws, "SerpApi", _FakeSerpApi)
    monkeypatch.setattr(ws, "_get_transport", lambda *_: None)
    captured: dict = {}

    # Capture the browser flag the service passes to the reader.
    async def capturing_reader(url, *, timeout=0, max_chars=0, focus="", allow_browser=False):
        captured["allow_browser"] = allow_browser
        return f"# parsed {url}"

    service = ws.WebSearchService(
        tracker=EngineHealthTracker(clock=_Clock()), read_page=capturing_reader
    )
    asyncio.run(service.search("attention mechanism", effort="high"))

    assert captured["allow_browser"] is True


# Verify that a hosted provider returning the same URL under a different family is merged as a
# consensus vote: one row remains, with both families listed.
def test_web_search_hosted_consensus_merges_not_overwrites(monkeypatch):
    import core.search.web_search as ws

    _FakeSerpApi.events = [
        {
            "type": "source", "engine": "google", "provider_family": "google", "rank": 1,
            "url": {"url": "https://good.com/x", "host": "good.com"},
            "serp": {"title": "T", "snippet": "a detailed snippet about the topic at hand",
                     "fetch_ms": 1, "parse_ms": 1},
        },
        {
            "type": "engine", "engine": "google",
            "payload": {"engine": "google", "provider_family": "google", "status": "success",
                        "fetch_ms": 1.0, "sources": [{"url": "https://good.com/x"}]},
        },
    ]
    monkeypatch.setattr(ws, "SerpApi", _FakeSerpApi)
    monkeypatch.setattr(ws, "_get_transport", lambda *_: None)

    # Stream a hosted provider result that surfaces the same URL under a different family: it must merge
    # as a consensus vote, not overwrite the original source or duplicate the row.
    async def fake_hosted():
        yield {
            "type": "source", "engine": "hosted:tavily", "provider_family": "tavily", "rank": 1,
            "url": {"url": "https://good.com/x", "host": "good.com"},
            "serp": {"title": "T", "snippet": "hosted snippet", "fetch_ms": 1, "parse_ms": 0},
        }
        yield {
            "type": "engine", "engine": "hosted:tavily",
            "payload": {"engine": "hosted:tavily", "provider_family": "tavily",
                        "status": "success", "fetch_ms": 1.0, "sources": [{"url": "https://good.com/x"}]},
        }

    service = ws.WebSearchService(tracker=EngineHealthTracker(clock=_Clock()), read_page=_fake_reader)
    monkeypatch.setattr(service, "_hosted_stream", lambda *a, **k: fake_hosted())
    result = asyncio.run(service.search("topic query example", effort="medium"))

    same = [s for s in result["sources"] if s["url"] == "https://good.com/x"]

    # Merged, not duplicated.
    assert len(same) == 1
    assert set(same[0]["consensus_families"]) == {"google", "tavily"}


# Verify that low effort is SERP-only: sources are returned without any page parsing (no markdown).
def test_web_search_low_is_serp_only(monkeypatch):
    import core.search.web_search as ws

    _FakeSerpApi.events = [
        {
            "type": "source",
            "engine": "yandex",
            "provider_family": "yandex",
            "rank": 1,
            "url": {"url": "https://good.com/a", "host": "good.com"},
            "serp": {
                "title": "Python asyncio tutorial",
                "snippet": "A long, detailed walkthrough of asyncio coroutines and tasks in Python.",
                "fetch_ms": 500,
                "parse_ms": 2,
            },
        },
    ]
    monkeypatch.setattr(ws, "SerpApi", _FakeSerpApi)
    monkeypatch.setattr(ws, "_get_transport", lambda *_: None)

    # Fail the test if the reader is called: low effort must never parse.
    async def explode(*a, **k):
        raise AssertionError("low effort must not parse pages")

    service = ws.WebSearchService(tracker=EngineHealthTracker(clock=_Clock()), read_page=explode)
    result = asyncio.run(service.search("python asyncio tutorial", effort="low"))
    assert result["sources"] and "markdown" not in result["sources"][0]


# Transient-failure negative cache guard (#4)

# Verify that a genuinely empty SERP counts as productive (cacheable), while a total outage where every
# engine failed does not (it must not be negative-cached).
def test_had_productive_engine_distinguishes_empty_from_outage():
    from core.search.web_search import _had_productive_engine

    # Genuine empty SERP: an engine worked and returned nothing, so it is productive.
    assert _had_productive_engine({"engines": {"yandex": {"status": "empty"}}}) is True
    assert _had_productive_engine({"engines": {"google": {"status": "success"}}}) is True

    # Total outage: every engine failed, so it is not productive.
    assert _had_productive_engine(
        {"engines": {"yandex": {"status": "timeout"}, "brave": {"status": "blocked"}}}
    ) is False
    assert _had_productive_engine({"engines": {}}) is False


# Captcha-burn retry (SerpApi blocked_retries)

from core.engines.models import EngineParseResult, EngineRequest, ParseStatus, SearchResult
from core.fetch._base import TransportResponse


# Fake engine parser that reports a block on its first two parse calls and a full SERP on the third.
# The call counter is a class attribute that tests reset.
class _RetryFakeParser:
    name = "retry_fake"
    provider_family = "retry_fake"
    calls = 0

    # Build a plain GET request carrying the identity key, so the retry can clear it.
    @staticmethod
    def build_request(query, *, region="us-en", safesearch="moderate", timelimit=None, page=1):
        return EngineRequest(
            method="GET", url="https://retry.test/search", identity_key="retry_fake"
        )

    # Count the call and return BLOCKED for the first two calls, then SUCCESS with one result.
    def parse(self, document):
        type(self).calls += 1
        if type(self).calls < 3:
            return EngineParseResult(engine=self.name, status=ParseStatus.BLOCKED)
        return EngineParseResult(
            engine=self.name,
            status=ParseStatus.SUCCESS,
            results=[SearchResult(title="ok", url="https://ok.test/a", snippet="s")],
        )


# Fake engine parser that reports a block on every parse call.
class _AlwaysBlockedParser(_RetryFakeParser):
    name = "retry_blocked"
    provider_family = "retry_blocked"

    # Count the call and return BLOCKED.
    def parse(self, document):
        type(self).calls += 1
        return EngineParseResult(engine=self.name, status=ParseStatus.BLOCKED)


# Fake transport that counts fetches (including those without an identity key) and identity client drops.
class _FakeTransport:
    # Start with all counters at zero.
    def __init__(self):
        self.fetches = 0
        self.drops = 0
        self.client_drops = 0
        self.naked_fetches = 0

    # Count the fetch, noting when the request carries no identity key, and return an empty 200 page.
    async def fetch(self, request):
        self.fetches += 1
        if not request.identity_key:
            self.naked_fetches += 1
        return TransportResponse(status=200, body=b"<html></html>", transport="fake")

    # Count the dropped pooled client for the host.
    def drop_identity_clients(self, host):
        self.client_drops += 1
        return 1

    # Close the transport; there is nothing to release.
    async def close(self):
        return None


# Run one engine through SerpApi with the given blocked_retries and return the engine payload and the
# fake transport, so tests can inspect the counters.
def _run_retry_case(monkeypatch, parser_type, blocked_retries):
    from core.search import serp_api

    transport = _FakeTransport()

    # Count the identity cookie drops instead of touching real cookies.
    def fake_drop(request):
        transport.drops += 1
        return 0

    monkeypatch.setattr(serp_api, "_drop_identity_cookies", fake_drop)
    api = serp_api.SerpApi(transport=transport, engines=(parser_type,))
    payload = asyncio.run(
        api._run_engine(
            parser_type, "q", region="us-en", safesearch="moderate",
            timelimit=None, blocked_retries=blocked_retries,
        )
    )
    return payload, transport


# Verify that a blocked engine recovers on the second retry after an identity reset: the cookie branch
# and the pooled client are dropped before each retry, and retries fetch with the identity key cleared.
def test_blocked_retry_recovers_after_identity_reset(monkeypatch):
    _RetryFakeParser.calls = 0
    payload, transport = _run_retry_case(monkeypatch, _RetryFakeParser, blocked_retries=3)
    assert payload["status"] == "success"
    assert payload["sources"][0]["url"] == "https://ok.test/a"

    # First attempt plus two retries; each retry resets the identity and fetches without the key.
    assert transport.fetches == 3
    assert transport.drops == 2
    assert transport.client_drops == 2
    assert transport.naked_fetches == 2
    assert "captcha retry 2/3" in payload["diagnostics"]


# Verify that an engine that stays blocked uses up all retries (first attempt plus three), reports
# "blocked" and records the last retry in the diagnostics.
def test_blocked_retry_exhausts_and_stays_blocked(monkeypatch):
    _AlwaysBlockedParser.calls = 0
    payload, transport = _run_retry_case(monkeypatch, _AlwaysBlockedParser, blocked_retries=3)
    assert payload["status"] == "blocked"
    assert transport.fetches == 4
    assert transport.drops == 3
    assert transport.client_drops == 3
    assert transport.naked_fetches == 3
    assert "captcha retry 3/3" in payload["diagnostics"]


# Verify that with blocked_retries=0 (the default) a blocked engine is fetched once and no identity is
# dropped.
def test_blocked_retry_off_by_default(monkeypatch):
    _AlwaysBlockedParser.calls = 0
    payload, transport = _run_retry_case(monkeypatch, _AlwaysBlockedParser, blocked_retries=0)
    assert payload["status"] == "blocked"
    assert transport.fetches == 1
    assert transport.drops == 0


# Adaptive Yandex UA mode

# Minimal Yandex SERP with one organic result, used to simulate a successful parse.
_GOOD_YANDEX_HTML = (
    '<html><body><div class="Organic"><h2><a class="OrganicTitle-Link" '
    'href="https://ok.test/x"><span class="OrganicTitle-LinkText">Title</span></a></h2>'
    '<div class="OrganicTextContentSpan">text</div></div></body></html>'
)


# Reset the Yandex module's adaptive UA state to the YaBrowser leg with no blocked streak.
def _reset_yandex_ua_state():
    from core.engines import yandex as yz

    yz._ua_mode = "yabrowser"
    yz._plain_until = 0.0
    yz._blocked_streak = 0


# Verify that the default Yandex leg sends a YaBrowser user agent with a pinned Chromium TLS target.
def test_yandex_default_leg_sends_yabrowser_ua():
    _reset_yandex_ua_state()
    from core.engines.yandex import YandexParser

    req = YandexParser.build_request("q", region="us-en")
    assert "YaBrowser" in req.headers.get("User-Agent", "")
    assert req.primp_target == "chrome"


# Verify that one blocked parse keeps the YaBrowser leg, a second consecutive block flips to the plain
# leg (the proven no-UA recipe), and the YaBrowser leg returns once the plain cooldown has elapsed.
def test_yandex_blocked_streak_flips_to_plain_then_recovers():
    _reset_yandex_ua_state()
    from core.engines import yandex as yz
    from core.engines.yandex import YandexParser

    parser = YandexParser()
    parser.parse("<html>smartcaptcha</html>")
    req = YandexParser.build_request("q", region="us-en")

    # One strike is not enough to flip.
    assert "YaBrowser" in req.headers.get("User-Agent", "")

    # The second consecutive block flips to the plain leg.
    parser.parse("<html>smartcaptcha</html>")
    req = YandexParser.build_request("q", region="us-en")
    assert "User-Agent" not in req.headers

    # Once the plain cooldown has elapsed the YaBrowser leg is used again.
    yz._plain_until = 0.0
    req = YandexParser.build_request("q", region="us-en")
    assert "YaBrowser" in req.headers.get("User-Agent", "")


# Verify that a successful parse resets the blocked streak, so a later single block does not flip the
# leg to plain.
def test_yandex_success_resets_blocked_streak():
    _reset_yandex_ua_state()
    from core.engines.yandex import YandexParser

    parser = YandexParser()

    # Streak 1, then a success resets it, then streak 1 again: no flip.
    parser.parse("<html>smartcaptcha</html>")
    assert parser.parse(_GOOD_YANDEX_HTML).status.value == "success"
    parser.parse("<html>smartcaptcha</html>")
    req = YandexParser.build_request("q", region="us-en")
    assert "YaBrowser" in req.headers.get("User-Agent", "")
