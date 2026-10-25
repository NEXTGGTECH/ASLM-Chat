# Copyright NEXTGGTECH. Elastic License 2.0.

"""Offline coverage for the new search core: quality, triage, health, orchestrator."""

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


# --- quality -----------------------------------------------------------------


def test_search_ids_are_linear_and_citation_namespace_is_not_truncated():
    first = _make_search_id()
    second = _make_search_id()

    assert int(second.removeprefix("srch_"), 16) == int(first.removeprefix("srch_"), 16) + 1
    assert len(first.removeprefix("srch_")) >= 4
    assert _citation_id("srch_0abc", 3) == "c0abc-3"
    assert _citation_id("srch_10000", 1) == "c10000-1"

def test_hub_penalty_flags_category_pages():
    assert hub_penalty("https://site.com/category/news/", "All news", "") >= 0.5
    assert hub_penalty("https://site.com/blog/deep-dive-asyncio", "Deep dive", "long snippet here") == 0.0


def test_skip_title_patterns():
    assert is_skip_title("Login - example.com")
    assert is_skip_title("404 Not Found")
    assert not is_skip_title("Understanding asyncio")


def test_year_policy_soft_only():
    # Year alignment now keys off a REAL extracted date, never a bare year token.
    years = query_years("best gpu 2024")
    assert years == ["2024"]
    assert page_date_score(extract_date("Mar 12, 2024 — benchmark roundup"), years) == 1.0
    assert page_date_score(extract_date("15.06.2021 — обзор видеокарт"), years) == -0.3
    assert page_date_score(None, years) == 0.0  # no signal ≠ penalty


def test_extract_date_requires_a_real_date_not_a_year_token():
    # The SEO-farm move — a bare year in the title — must NOT read as a date.
    assert extract_date("GPT 5 Mini API Pricing 2026 - Costs, Performance & Providers") is None
    assert extract_date("Jul 3, 2026 — OpenAI updated its API pricing.") == (2026, 7, 3)
    assert extract_date("3 июл. 2026 г. — обновлены тарифы API.") == (2026, 7, 3)
    assert extract_date("03.07.2026 · новости") == (2026, 7, 3)
    assert extract_date("2026-07-03 release notes") == (2026, 7, 3)
    assert extract_date("July 2026 changelog") == (2026, 7, 1)
    import datetime as dt
    assert extract_date("2 days ago — thread", now=dt.date(2026, 7, 4)) == (2026, 7, 2)
    assert extract_date("5 часов назад — новость", now=dt.date(2026, 7, 4)) == (2026, 7, 4)


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
    # es "ago" (August) must not hijack English "years ago" next to a year.
    d = extract_date("updated 3 years ago. 2026 predictions", now=dt.date(2026, 7, 4))
    assert d is not None and d[0] == 2023
    assert extract_date("лучшие инструменты 2026") is None


def test_page_date_freshness_decays_without_query_years():
    import datetime as dt
    now = dt.date(2026, 7, 4)
    fresh = page_date_score((2026, 7, 1), [], now=now)
    old = page_date_score((2024, 7, 1), [], now=now)
    assert fresh > 0.9
    assert 0.0 < old < 0.2
    assert page_date_score(None, [], now=now) == 0.0


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


def test_infer_query_language_scripts():
    assert infer_query_language("how to use asyncio") == "en"
    assert infer_query_language("как настроить роутер") == "ru"
    assert infer_query_language("非同期処理 チュートリアル") == "ja"


# --- triage ------------------------------------------------------------------

def _ingest(session: TriageSession, *, engine="google", family="google", rank=1,
            url="https://ex.com/a", title="Python asyncio tutorial",
            snippet="A long, detailed walkthrough of asyncio coroutines and tasks in Python."):
    return session.ingest_source(
        engine=engine, provider_family=family, rank=rank, url=url, title=title, snippet=snippet
    )


def test_triage_relevant_top_google_parses_immediately():
    session = TriageSession("python asyncio tutorial")
    decision = _ingest(session)
    assert decision.action == TriageAction.PARSE


def test_triage_skip_title_is_skipped():
    session = TriageSession("python asyncio")
    decision = _ingest(session, title="Login required")
    assert decision.action == TriageAction.SKIP


def test_triage_revisit_takes_best_positional_view_and_upgrades():
    session = TriageSession("python asyncio tutorial")
    # Weak engine, mid rank → lands in queue.
    decision = _ingest(session, engine="qwant", family="qwant", rank=5)
    assert decision.action == TriageAction.QUEUE
    # The same page then surfaces at Google rank 1: the base must be re-priced to the
    # strongest single-engine view instead of the initial Qwant rank 5; the second
    # family's vote lands on top → PARSE upgrade.
    upgraded = session.ingest_revisit(
        engine="google", provider_family="google", rank=1,
        url="https://ex.com/a", title="Python asyncio tutorial",
        snippet="A long, detailed walkthrough of asyncio coroutines and tasks in Python.",
    )
    assert upgraded is not None and upgraded.upgraded
    assert upgraded.action == TriageAction.PARSE


def test_triage_vote_weight_follows_family_class():
    def bump_from(family: str) -> float:
        session = TriageSession("python asyncio tutorial")
        _ingest(session, engine="qwant", family="qwant", rank=5)
        before = session.score_of("https://ex.com/a")
        session.ingest_vote(provider_family=family, url="https://ex.com/a")
        return session.score_of("https://ex.com/a") - before

    # A Google vote must be worth more than a Yandex vote (deliberately floored engine).
    assert bump_from("google") > bump_from("yandex") > 0.0


def test_triage_google_startpage_one_family_vote():
    session = TriageSession("python asyncio tutorial")
    decision = _ingest(session, engine="qwant", family="qwant", rank=5)
    assert decision.action == TriageAction.QUEUE
    score_before = session.score_of("https://ex.com/a")
    session.ingest_vote(provider_family="google", url="https://ex.com/a")
    score_after_google = session.score_of("https://ex.com/a")
    # Startpage shares Google's family → no second bump.
    assert session.ingest_vote(provider_family="google", url="https://ex.com/a") is None
    assert score_after_google > score_before
    assert session.score_of("https://ex.com/a") == score_after_google


# --- health / circuit breaker --------------------------------------------------

class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_breaker_error_opens_for_five_minutes():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    assert tracker.allow("ddg")
    tracker.record("ddg", status="blocked", fetch_ms=200, results=0)
    assert not tracker.allow("ddg")
    clock.now = ERROR_COOLDOWN - 1
    assert not tracker.allow("ddg")
    clock.now = ERROR_COOLDOWN + 1
    assert tracker.allow("ddg")  # half-open probe
    assert not tracker.allow("ddg")  # only one probe at a time


def test_breaker_degradation_short_cooldown_and_recovery():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    # Build a latency baseline.
    tracker.record("yandex", status="success", fetch_ms=500, results=10)
    # 5x spike → degradation trip.
    tracker.record("yandex", status="success", fetch_ms=2500, results=10)
    assert not tracker.allow("yandex")
    clock.now = DEGRADATION_COOLDOWN + 1
    assert tracker.allow("yandex")
    tracker.record("yandex", status="success", fetch_ms=520, results=10)
    clock.now += 5  # past the jittered Stage C pace gate set by the probe fire
    assert tracker.allow("yandex")  # closed again


def test_breaker_failed_probe_backs_off_exponentially():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    tracker.record("brave", status="blocked", fetch_ms=100, results=0)
    clock.now = ERROR_COOLDOWN + 1
    assert tracker.allow("brave")  # probe
    tracker.record("brave", status="blocked", fetch_ms=100, results=0)  # probe failed
    health = tracker._health("brave")
    assert health.cooldown == ERROR_COOLDOWN * 2
    assert health.state == BreakerState.OPEN


def test_breaker_abandoned_probe_is_expired_not_wedged():
    # A half-open probe whose outcome is never recorded (e.g. the search deadline
    # dropped the engine's status event) must not lock the engine out forever.
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    tracker.record("qwant", status="blocked", fetch_ms=100, results=0)
    clock.now = ERROR_COOLDOWN + 1
    assert tracker.allow("qwant")  # probe admitted...
    assert not tracker.allow("qwant")  # ...and held while still in flight
    # Outcome never arrives; once the probe ages past the timeout it is reclaimed.
    clock.now += PROBE_TIMEOUT + 1
    assert tracker.allow("qwant")


# --- Stage C pacing --------------------------------------------------------------

def test_pacing_holds_engine_within_min_interval():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    assert tracker.allow("brave")        # first fire admitted (and paced)
    assert not tracker.allow("brave")    # within the ~6s min-interval → held back
    clock.now += 8.0                      # past the jittered interval (max 6*1.25=7.5s)
    assert tracker.allow("brave")        # interval elapsed → allowed again


def test_pacing_skipped_for_tolerant_engines():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    assert tracker.allow("duckduckgo")          # duckduckgo min-interval is 0
    assert tracker.allow("duckduckgo")          # so an immediate re-fire is allowed


# --- engine selection ------------------------------------------------------------

# All engines enabled — isolates breaker behaviour from the config layer.
_ALL_ON = {n: True for n in ("google", "duckduckgo", "startpage", "qwant", "brave", "yandex", "legacy_yandex")}


def test_select_engines_low_default_pair():
    tracker = EngineHealthTracker(clock=_Clock())
    names = [e.name for e in select_engines("low", tracker, enabled=_ALL_ON)]
    assert names == ["yandex", "legacy_yandex", "duckduckgo"]


def test_select_engines_low_falls_back_to_startpage():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    tracker.record("yandex", status="blocked", fetch_ms=100, results=0)
    tracker.record("legacy_yandex", status="blocked", fetch_ms=100, results=0)
    tracker.record("duckduckgo", status="blocked", fetch_ms=100, results=0)
    names = [e.name for e in select_engines("low", tracker, enabled=_ALL_ON)]
    assert names == ["startpage"]


def test_select_engines_low_never_empty():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    for engine in ("yandex", "legacy_yandex", "duckduckgo", "startpage"):
        tracker.record(engine, status="blocked", fetch_ms=100, results=0)
    names = [e.name for e in select_engines("low", tracker, enabled=_ALL_ON)]
    assert names == ["duckduckgo"]  # first enabled lead, forced through an open breaker


def test_select_engines_medium_google_healthy_startpage_stays_standby():
    # Google solidly CLOSED → it owns the family slot alone; Startpage stays a warm standby.
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    names = [e.name for e in select_engines("medium", tracker, enabled=_ALL_ON)]
    assert "google" in names
    assert "startpage" not in names


def test_select_engines_medium_startpage_standby_when_google_open():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    tracker.record("google", status="blocked", fetch_ms=100, results=0)
    names = [e.name for e in select_engines("medium", tracker, enabled=_ALL_ON)]
    assert "google" not in names
    assert "startpage" in names  # hot standby of the google family


def test_select_engines_medium_startpage_rides_along_on_google_probe():
    # Google's cooldown elapsed → next selection sends a half-open PROBE. Because recovery
    # is unproven, Startpage must ride along so a still-blocked probe doesn't cost the
    # search its google-family result (the "remember Google is bad" fix).
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    tracker.record("google", status="blocked", fetch_ms=100, results=0)
    clock.now = ERROR_COOLDOWN + 1
    names = [e.name for e in select_engines("medium", tracker, enabled=_ALL_ON)]
    assert "google" in names      # the probe fires
    assert "startpage" in names   # …and the safety net rides along
    assert tracker._health("google").state == BreakerState.HALF_OPEN


def test_is_healthy_pure_read_no_side_effects():
    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    assert tracker.is_healthy("google")          # never-seen → CLOSED → healthy
    tracker.record("google", status="blocked", fetch_ms=100, results=0)
    assert not tracker.is_healthy("google")      # OPEN → unhealthy
    clock.now = ERROR_COOLDOWN + 1
    assert not tracker.is_healthy("google")       # cooldown elapsed but unprobed → unhealthy
    # Pure read must not have consumed the probe: allow() still admits it.
    assert tracker.allow("google")


# --- per-engine config switches (TODO §D) -----------------------------------------

def test_select_engines_default_config_excludes_yandex_engines():
    # Shipped defaults: both Yandex engines off. High tier is the widest fan-out — neither may appear.
    from core.config.settings import EnginesSection

    tracker = EngineHealthTracker(clock=_Clock())
    names = [e.name for e in select_engines("high", tracker, enabled=EnginesSection().as_map())]
    assert "yandex" not in names
    assert "legacy_yandex" not in names
    assert "duckduckgo" in names  # the lead survives


def test_select_engines_user_opts_yandex_engines_back_in():
    from core.config.settings import EnginesSection

    enabled = EnginesSection(yandex=True, legacy_yandex=True).as_map()
    tracker = EngineHealthTracker(clock=_Clock())
    names = [e.name for e in select_engines("high", tracker, enabled=enabled)]
    assert "yandex" in names
    assert "legacy_yandex" in names


def test_select_engines_disabled_engine_never_forced():
    # Even the never-empty last resort respects the config: with yandex off and every
    # breaker open, the forced pick is the first ENABLED lead, not yandex.
    from core.config.settings import EnginesSection

    clock = _Clock()
    tracker = EngineHealthTracker(clock=clock)
    for engine in ("duckduckgo", "startpage"):
        tracker.record(engine, status="blocked", fetch_ms=100, results=0)
    names = [e.name for e in select_engines("low", tracker, enabled=EnginesSection().as_map())]
    assert names == ["duckduckgo"]


def test_select_engines_all_disabled_still_returns_lead():
    tracker = EngineHealthTracker(clock=_Clock())
    all_off = {n: False for n in _ALL_ON}
    names = [e.name for e in select_engines("low", tracker, enabled=all_off)]
    assert names == ["duckduckgo"]  # misconfiguration guard: never return zero engines


def test_engines_config_section_loads(tmp_path):
    import json as _json

    from core.config.settings import load_search_config

    cfg_path = tmp_path / "search_config.json"
    cfg_path.write_text(_json.dumps({"engines": {"yandex": True, "brave": False}}), encoding="utf-8")
    cfg = load_search_config(cfg_path)
    assert cfg.engines.yandex is True      # user opt-in honoured
    assert cfg.engines.brave is False      # explicit off honoured
    assert cfg.engines.duckduckgo is True  # absent → default on
    assert cfg.engines.legacy_yandex is False  # absent → default off


# --- inline-parse policy (custom-domain scope + learned-slow) --------------------

def test_custom_domain_scope_marks_browser_heavy_readpage_only():
    import custom_domains

    assert custom_domains.is_read_page_only("https://www.reddit.com/r/Python/comments/a/b/")
    assert custom_domains.is_read_page_only("https://www.youtube.com/watch?v=abc")
    # API-backed handlers stay usable in web_search.
    assert not custom_domains.is_read_page_only("https://github.com/python/cpython")
    # A plain domain with no handler is unaffected.
    assert not custom_domains.is_read_page_only("https://example.com/article")


def test_inline_parse_blocks_readpage_only_handler():
    assert not _inline_parse_allowed("https://www.reddit.com/r/Python/comments/a/b/")


def test_inline_parse_blocks_every_reddit_hostname_but_not_lookalikes():
    assert not _inline_parse_allowed("https://reddit.com/r/meshtastic/hot/")
    assert not _inline_parse_allowed("https://www.reddit.com/r/meshtastic/hot/")
    assert not _inline_parse_allowed("https://old.reddit.com/r/meshtastic/")
    assert not _inline_parse_allowed("https://redd.it/abc123")
    assert _inline_parse_allowed("https://not-reddit.com/article")


def test_inline_parse_skips_learned_slow_domain(monkeypatch):
    import core.profiles as profiles
    from core.profiles.models import ProfileHint

    class _Profiles:
        def __init__(self, ms):
            self._ms = ms

        def best_method(self, _domain):
            return ProfileHint(method="httpx", expected_fetch_ms=self._ms, confidence=0.9)

    # A domain remembered as slow → snippet-only; a fast one → parsed inline.
    monkeypatch.setattr(profiles, "get_runtime_profiles", lambda: _Profiles(9_000.0))
    assert not _inline_parse_allowed("https://slow.com/a")
    monkeypatch.setattr(profiles, "get_runtime_profiles", lambda: _Profiles(800.0))
    assert _inline_parse_allowed("https://fast.com/a")


def test_inline_parse_allows_unknown_domain():
    # No handler and no profile data yet → parse it (the store learns from this parse).
    assert _inline_parse_allowed("https://brand-new-domain-xyz.com/a")


# --- orchestrator (synthetic stream, fake reader) --------------------------------

class _FakeSerpApi:
    """Replays a scripted event stream."""

    events: list[dict] = []

    def __init__(self, *args, **kwargs) -> None:
        pass

    async def search_stream(self, *args, **kwargs):
        for event in self.events:
            yield event


async def _fake_reader(
    url: str, *, timeout: float = 0, max_chars: int = 0, focus: str = "",
    allow_browser: bool = True,
) -> str:
    await asyncio.sleep(0)
    return f"# parsed {url}"


def test_web_search_medium_parses_winners_and_ranks(monkeypatch):
    import core.search.web_search as ws

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

    urls = [s["url"] for s in result["sources"]]
    assert urls[0] == "https://good.com/asyncio-guide"  # consensus + rank on top
    top = result["sources"][0]
    assert top["consensus_families"] == ["google", "yandex"]
    assert top["parsed_ok"] and top["markdown"].startswith("# parsed")
    assert result["health"]["google"]["state"] == "closed"


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

    async def capturing_reader(url, *, timeout=0, max_chars=0, focus="", allow_browser=True):
        captured["focus"] = focus
        captured["allow_browser"] = allow_browser
        return f"# parsed {url}"

    service = ws.WebSearchService(tracker=EngineHealthTracker(clock=_Clock()), read_page=capturing_reader)
    asyncio.run(service.search("attention mechanism", effort="medium"))

    assert captured["focus"] == "attention mechanism"  # query is the compaction focus
    assert captured["allow_browser"] is False


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

    async def capturing_reader(url, *, timeout=0, max_chars=0, focus="", allow_browser=False):
        captured["allow_browser"] = allow_browser
        return f"# parsed {url}"

    service = ws.WebSearchService(
        tracker=EngineHealthTracker(clock=_Clock()), read_page=capturing_reader
    )
    asyncio.run(service.search("attention mechanism", effort="high"))

    assert captured["allow_browser"] is True


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

    # Hosted provider surfaces the SAME url under a different family — must merge as a
    # consensus vote, not overwrite the original source or duplicate the row.
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
    assert len(same) == 1  # merged, not duplicated
    assert set(same[0]["consensus_families"]) == {"google", "tavily"}


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

    async def explode(*a, **k):  # low must never parse
        raise AssertionError("low effort must not parse pages")

    service = ws.WebSearchService(tracker=EngineHealthTracker(clock=_Clock()), read_page=explode)
    result = asyncio.run(service.search("python asyncio tutorial", effort="low"))
    assert result["sources"] and "markdown" not in result["sources"][0]


# --- #4: transient-failure negative cache guard ----------------------------------

def test_had_productive_engine_distinguishes_empty_from_outage():
    from core.search.web_search import _had_productive_engine

    # Genuine empty SERP: an engine worked and returned nothing → productive (cacheable).
    assert _had_productive_engine({"engines": {"yandex": {"status": "empty"}}}) is True
    assert _had_productive_engine({"engines": {"google": {"status": "success"}}}) is True
    # Total outage: every engine failed → not productive (must not negative-cache).
    assert _had_productive_engine(
        {"engines": {"yandex": {"status": "timeout"}, "brave": {"status": "blocked"}}}
    ) is False
    assert _had_productive_engine({"engines": {}}) is False
