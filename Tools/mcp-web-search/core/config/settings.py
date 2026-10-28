# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


logger = logging.getLogger("config.search")

_CONFIG_PATH = Path(__file__).parent / "search_config.json"

# Warm-browser daemon default port. ASLM assigns one in its port range and passes it via
# ASLM_BROWSER_DAEMON_PORT; standalone runs fall back to 20010 (the module's declared default).
# NB: 20004 is occupied by ASLM's Ollama in this deployment — keep the daemon clear of it.
_DEFAULT_DAEMON_PORT = 20010


def _default_daemon_url() -> str:
    raw = (os.environ.get("ASLM_BROWSER_DAEMON_PORT") or "").strip()
    try:
        port = int(raw) if raw else _DEFAULT_DAEMON_PORT
    except ValueError:
        port = _DEFAULT_DAEMON_PORT
    if not (0 < port <= 65535):
        port = _DEFAULT_DAEMON_PORT
    return f"http://127.0.0.1:{port}"


# Typed config dataclasses (loaded from search_config.json).
@dataclass
class SearchSection:
    # NOTE: the new streaming pipeline hardcodes per-effort budgets in EFFORT_PROFILES;
    # only the fields below are still read. The legacy DDGS/preview/quality-worker knobs
    # (ddgs_*, preview_fetch_*, quality_ddgs_*, routing_profile, auto_scrape_preview, …)
    # were removed 2026-06-20 — they had zero readers after the rewrite.
    tls_verify: bool = True        # set False only behind corporate MITM proxies
    max_results: int = 10
    batch_query_limit: int = 10    # read_page multi-URL batch cap (mcp-server)
    prefetch_fetch_timeout: float = 8.0
    preview_max_chars: int = 4_000     # per-source chars in model_context
    total_context_budget: int = 40_000  # max chars in total search output (0 = no limit)


@dataclass
class ExtractionSection:
    timeout_seconds: float = 25.0
    max_page_chars: int = 30_000
    min_content_length: int = 800
    enable_read_page_compress: bool = True
    read_page_compress_threshold_chars: int = 25_000
    read_page_compress_target_chars: int = 10_000


@dataclass
class CacheSection:
    search_ttl_seconds: int = 21_600        # 6 h — flat TTL for the query-results cache
    search_negative_ttl_seconds: int = 300  # 5 min — empty/failed result sets
    page_ttl_seconds: int = 86_400
    repeat_block_window_seconds: int = 30   # identical query within this → hard block
    seen_source_window_seconds: int = 30    # drop sources served to the model within this
    prefetch_max_urls: int = 4              # top uncached result URLs warmed per search (0 = off)


# Year tokens in queries: timelimit (default), strip, or none — see year_hint_* fields.
@dataclass
class QuerySection:
    schema_mode: str = "advanced"  # advanced (default) | legacy
    year_hint_mode: str = "timelimit"
    year_hint_current: Optional[str] = "m"  # year == this year  → last month
    year_hint_prev: Optional[str] = "y"     # year == last year  → last year
    year_hint_older: Optional[str] = None  # year < last year  → no restriction


# Warm-browser layer (cloakbrowser daemon). browser_fallback controls where the
# browser is allowed as a fallback; the backend is always the warm chromium daemon.
@dataclass
class BrowserSection:
    browser_fallback: str = "page"      # off | page (read_page only) | full (+ blocked SERP engines)
    daemon_url: str = field(default_factory=_default_daemon_url)
    engine: str = "chromium"            # warm backend is chromium-only by design
    autostart_daemon: bool = True       # spawn the daemon lazily on the first tool call
    # Daemon self-shuts-down after this many idle seconds (no fetch); 0 = eternal (run
    # until the task is killed). The daemon now OUTLIVES the tool-call process that spawned
    # it (the client no longer kills it on exit), so it stays warm across calls and this
    # idle timer — not the caller — bounds its life. Default 15 min: warm enough to serve a
    # follow-up search/read cheaply, short enough not to hold a browser's RAM for long.
    daemon_idle_shutdown_sec: float = 900.0
    headless: bool = True
    humanize: bool = False
    proxy: str = ""
    nav_timeout: float = 30.0           # per-page navigation timeout (seconds)
    wait: float = 3.0                   # post-load text-settle wait (seconds)
    fetch_timeout: float = 45.0         # client-side ceiling for one /fetch round-trip
    # Recycle thresholds (passed to the daemon; enforced daemon-side).
    max_requests: int = 40
    max_age_sec: float = 900.0
    max_rss_mb: int = 2048              # RSS of the browser process tree → checkpoint + respawn
    checkpoint_interval: float = 30.0   # idle storageState checkpoint cadence (seconds)


# Tor/onion access — the most optional thing in the search. OFF by default and zero-install:
# the tool never bundles, installs, or spawns tor. When enabled it REUSES a tor that is already
# running — a system daemon on 9050, an open Tor Browser on 9150, or an explicit socks_url. No
# running tor → the feature simply goes no-op, never an error.
@dataclass
class TorSection:
    enabled: bool = False               # master switch; off → onion paths are no-op
    socks_url: str = ""                 # explicit override, e.g. socks5h://127.0.0.1:9050
    fetch_timeout: float = 60.0         # per-request ceiling (Tor is slow; circuits add latency)
    # The onion allowlist is static and hand-vetted (the seed registry) — there is no runtime
    # discovery or persistence. (The old anchored auto-expansion + onion_registry.db store were
    # removed as more risk than value while the static link-search layer is still unfinished.)


# Import the user's REAL browser cookies/metadata into the identity layer so HTTP SERP
# engines and the warm browser replay a logged-in, human fingerprint. OFF by default and
# privacy-sensitive: reading a browser's cookie jar exposes the user's live sessions, so it
# is strictly opt-in. Cross-browser: Chrome/Edge/Brave (chromium family) + Firefox. `browsers`
# selects which to harvest; empty domains allowlist means every domain (narrow it to search
# engines to limit exposure). Only cookies whose domain matches the allowlist are imported.
@dataclass
class ProfileImportSection:
    enabled: bool = False                              # master switch — off → no profile is ever read
    browsers: list[str] = field(                       # which installed browsers to harvest
        default_factory=lambda: ["chrome", "edge", "brave", "firefox"]
    )
    domains: list[str] = field(                        # cookie-domain allowlist (empty = all domains)
        default_factory=lambda: [
            "google.com", "bing.com", "duckduckgo.com", "startpage.com",
            "yandex.com", "yandex.ru", "qwant.com", "brave.com", "search.brave.com",
            "reddit.com",
        ]
    )
    all_profiles: bool = False                         # False → default profile only; True → every profile
    refresh_hours: float = 12.0                        # re-harvest only after this long (0 = every start)
    purge_on_disable: bool = True                      # wipe imported cookies from the store when disabled


# Hosted (paid API) provider modes. Per provider: "content" | "serp" | "off".
#   content — SERP rows + full page text pre-fed into SourceCache (costs scrape credits)
#   serp    — SERP rows only: cheap consensus/coverage votes, no content fetch
#   off     — provider excluded even when its API key is configured
# Defaults are cost-aware: Firecrawl's content mode scrapes every result with a headless
# fetch (~9s measured) — slower than the medium-effort hosted deadline, so the credits
# were spent on results the deadline then discarded. SERP-only keeps its consensus value
# at near-zero latency; flip to "content" deliberately for high-effort/agent workloads.
@dataclass
class HostedApiSection:
    tavily: str = "content"     # fast even with raw content (~2.4s measured)
    firecrawl: str = "serp"     # content mode measured at ~9.4s — deadline-hostile
    brave: str = "serp"         # SERP-native providers: "content" behaves as "serp"
    serpapi: str = "serp"


# Per-engine kill switches for web_search's tiered selection (TODO §D). An engine set to
# False never enters a tier — the user opts weak engines in, they are not opted out by
# health at runtime (that's the breaker's job, this is policy). Both Yandex engines default
# OFF: the native ya.ru SERP is captcha-gated for scripted clients (it relies on the
# warm-browser fallback) and the legacy widget is Yandex's retired search product.
@dataclass
class RichCardsSection:
    """Fail-open rich cards for top SERP results (core/cards). DEFAULT OFF: each
    wikipedia card costs one extra API roundtrip per query; enable deliberately."""
    enabled: bool = False
    wikipedia: bool = True
    stackexchange: bool = True
    reddit: bool = True
    max_cards: int = 3
    timeout_seconds: float = 2.0
    ttl_seconds: float = 86400.0
    negative_ttl_seconds: float = 900.0


@dataclass
class ImageEnginesSection:
    """Kill switches for the image vertical (engines/images.py). Defaults follow the
    2026-10-28 live check: bing/brave/duckduckgo answer from residential networks
    (DDG once i.js stopped sending navigation-only headers); openverse times out
    on every request and would hold each image search until the deadline."""
    bing: bool = True
    brave: bool = True
    duckduckgo: bool = True
    openverse: bool = False

    def as_map(self) -> dict[str, bool]:
        return {"bing": self.bing, "brave": self.brave,
                "duckduckgo": self.duckduckgo, "openverse": self.openverse}


@dataclass
class QueryRewriterSection:
    """LLM subagent that judges and rewrites search queries.
    Disabled by default: each evaluated query pays one model call (tens of seconds
    on a 27B). In "always" mode (default) the model judges every query — an already
    clean query is approved unchanged, a conversational thought gets rewritten. In
    "heuristic" mode a local word-stem gate skips the model call for obviously clean
    queries. Always uses the active chat engine and model from
    Tools/model_runtime_metadata.json + Settings/settings.json."""
    enabled: bool = False
    timeout_seconds: float = 75.0
    max_queries: int = 3
    mode: str = "always"  # always (model judges) | heuristic (cheap gate first)
    # Word-stem overrides for the conversational heuristic (rewriter.looks_conversational).
    # NO built-in lists: empty (the default) disables the local gate entirely, so
    # heuristic mode behaves like always. A deployment that wants a cheap local gate
    # writes its own stems here (matched per word: equality, or prefix for stems
    # of 4+ chars).
    conversational_markers: tuple[str, ...] = ()


@dataclass
class LocalizationSection:
    """Language-aware region routing and result-language scoring (query_locales)."""
    enabled: bool = True
    # Triage toll for a result written in a language the query did not use. Capped at
    # 0.3 in code; 0.12 keeps one consensus vote (0.2+) above two foreign mismatches.
    language_penalty: float = 0.12


@dataclass
class EnginesSection:
    google: bool = True
    duckduckgo: bool = True
    startpage: bool = True
    qwant: bool = True
    brave: bool = True
    yandex: bool = False

    # {engine name: enabled} view for selection code.
    def as_map(self) -> dict[str, bool]:
        return {
            "google": self.google,
            "duckduckgo": self.duckduckgo,
            "startpage": self.startpage,
            "qwant": self.qwant,
            "brave": self.brave,
            "yandex": self.yandex,
        }


@dataclass
class BufferEngineSection:
    """Emergency reserve engine (the legacy Yandex site-search widget).

    The buffer never participates in normal tier rotation: select_engines fires it
    only when every regular engine is switched off in config or blocked by its
    circuit breaker, so a search still returns some results instead of none."""
    enabled: bool = True


@dataclass
class SearchConfig:
    search: SearchSection = field(default_factory=SearchSection)
    extraction: ExtractionSection = field(default_factory=ExtractionSection)
    cache: CacheSection = field(default_factory=CacheSection)
    query: QuerySection = field(default_factory=QuerySection)
    browser: BrowserSection = field(default_factory=BrowserSection)
    tor: TorSection = field(default_factory=TorSection)
    hosted_api: HostedApiSection = field(default_factory=HostedApiSection)
    profile_import: ProfileImportSection = field(default_factory=ProfileImportSection)
    query_rewriter: QueryRewriterSection = field(default_factory=QueryRewriterSection)
    image_engines: ImageEnginesSection = field(default_factory=ImageEnginesSection)
    rich_cards: RichCardsSection = field(default_factory=RichCardsSection)
    localization: LocalizationSection = field(default_factory=LocalizationSection)
    engines: EnginesSection = field(default_factory=EnginesSection)
    buffer_engine: BufferEngineSection = field(default_factory=BufferEngineSection)


_cached_config: SearchConfig | None = None
_cached_config_signature: tuple[int, int] | None = None

_MISSING = object()


# Return the file signature used to invalidate the config cache.
def _config_signature(path: Path) -> tuple[int, int] | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    return stat.st_mtime_ns, stat.st_size


# Coerce a value to one of an allowed set (case-insensitive), falling back on default.
def _one_of(value: object, allowed: set[str], default: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate in allowed:
        return candidate
    if value not in (None, ""):
        logger.warning("config: invalid value %r (allowed: %s) — using %r", value, sorted(allowed), default)
    return default


# Coerce a JSON value to a clean lowercased string list. None/non-list → default.
# When `allowed` is given, entries outside it are dropped (unknown browser names, etc.).
def _string_list(value: object, allowed: Optional[set[str]], default: list[str]) -> list[str]:
    if not isinstance(value, list):
        return list(default)
    out: list[str] = []
    for item in value:
        s = str(item or "").strip().lower()
        if not s:
            continue
        if allowed is not None and s not in allowed:
            logger.warning("config: ignoring unknown profile_import entry %r", item)
            continue
        if s not in out:
            out.append(s)
    return out


# Coerce JSON values to optional strings (empty string → None).
def _optional_string(value: object, default: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value == "":
        return None
    if value is _MISSING:
        return default
    return str(value)


# Load search_config.json and cache a SearchConfig singleton (custom path for tests only).
def load_search_config(path: Path | None = None) -> SearchConfig:
    global _cached_config, _cached_config_signature

    target = path or _CONFIG_PATH
    signature = _config_signature(target)
    if path is None and _cached_config is not None and _cached_config_signature == signature:
        return _cached_config

    try:
        raw = json.loads(target.read_text(encoding="utf-8"))
    except FileNotFoundError:
        logger.warning("search_config.json not found at %s — using defaults", target)
        config = SearchConfig()
        if path is None:
            _cached_config = config
            _cached_config_signature = signature
        return config
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s — using defaults", target, exc)
        config = SearchConfig()
        if path is None:
            _cached_config = config
            _cached_config_signature = signature
        return config

    s = raw.get("search", {})
    e = raw.get("extraction", {})
    c = raw.get("cache", {})
    q = raw.get("query", {})
    b = raw.get("browser", {})
    t = raw.get("tor", {})
    h = raw.get("hosted_api", {})
    p = raw.get("profile_import", {})
    w = raw.get("query_rewriter", {})
    l = raw.get("localization", {})
    im = raw.get("image_engines", {})
    rc = raw.get("rich_cards", {})
    g = raw.get("engines", {})
    buf = raw.get("buffer_engine", {})
    _hosted_modes = {"content", "serp", "off"}
    _known_browsers = {"chrome", "edge", "brave", "firefox"}
    if isinstance(g, dict):
        for key in g:
            if key not in EnginesSection().as_map():
                logger.warning("config: ignoring unknown engines entry %r", key)
    else:
        g = {}

    config = SearchConfig(
        search=SearchSection(
            tls_verify=bool(s.get("tls_verify", True)),
            max_results=int(s.get("max_results", 10)),
            batch_query_limit=int(s.get("batch_query_limit", 10)),
            prefetch_fetch_timeout=float(s.get("prefetch_fetch_timeout", 8.0)),
            preview_max_chars=int(s.get("preview_max_chars", 4_000)),
            total_context_budget=int(s.get("total_context_budget", 40_000)),
        ),
        extraction=ExtractionSection(
            timeout_seconds=float(e.get("timeout_seconds", 25.0)),
            max_page_chars=int(e.get("max_page_chars", 20_000)),
            min_content_length=int(e.get("min_content_length", 800)),
            enable_read_page_compress=bool(e.get("enable_read_page_compress", True)),
            read_page_compress_threshold_chars=int(
                e.get("read_page_compress_threshold_chars", 10_000)
            ),
            read_page_compress_target_chars=int(e.get("read_page_compress_target_chars", 10_000)),
        ),
        cache=CacheSection(
            search_ttl_seconds=int(c.get("search_ttl_seconds", 21_600)),
            search_negative_ttl_seconds=int(c.get("search_negative_ttl_seconds", 300)),
            page_ttl_seconds=int(c.get("page_ttl_seconds", 86_400)),
            repeat_block_window_seconds=int(c.get("repeat_block_window_seconds", 30)),
            seen_source_window_seconds=int(c.get("seen_source_window_seconds", 30)),
            prefetch_max_urls=int(c.get("prefetch_max_urls", 4)),
        ),
        query=QuerySection(
            schema_mode=_one_of(
                q.get("schema_mode", "advanced"), {"legacy", "advanced"}, "advanced"
            ),
            year_hint_mode=str(q.get("year_hint_mode", "timelimit")),
            year_hint_current=_optional_string(q.get("year_hint_current", _MISSING), "m"),
            year_hint_prev=_optional_string(q.get("year_hint_prev", _MISSING), "y"),
            year_hint_older=_optional_string(q.get("year_hint_older", _MISSING), None),
        ),
        browser=BrowserSection(
            browser_fallback=_one_of(
                b.get("browser_fallback", "page"), {"off", "page", "full"}, "page"
            ),
            daemon_url=str(b.get("daemon_url") or _default_daemon_url()),
            engine=str(b.get("engine", "chromium")),
            autostart_daemon=bool(b.get("autostart_daemon", True)),
            daemon_idle_shutdown_sec=float(b.get("daemon_idle_shutdown_sec", 900.0)),
            headless=bool(b.get("headless", True)),
            humanize=bool(b.get("humanize", False)),
            proxy=str(b.get("proxy", "")),
            nav_timeout=float(b.get("nav_timeout", 30.0)),
            wait=float(b.get("wait", 3.0)),
            fetch_timeout=float(b.get("fetch_timeout", 45.0)),
            max_requests=int(b.get("max_requests", 40)),
            max_age_sec=float(b.get("max_age_sec", 900.0)),
            max_rss_mb=int(b.get("max_rss_mb", 2048)),
            checkpoint_interval=float(b.get("checkpoint_interval", 30.0)),
        ),
        tor=TorSection(
            enabled=bool(t.get("enabled", False)),
            socks_url=str(t.get("socks_url", "")),
            fetch_timeout=float(t.get("fetch_timeout", 60.0)),
        ),
        hosted_api=HostedApiSection(
            tavily=_one_of(h.get("tavily", "content"), _hosted_modes, "content"),
            firecrawl=_one_of(h.get("firecrawl", "serp"), _hosted_modes, "serp"),
            brave=_one_of(h.get("brave", "serp"), _hosted_modes, "serp"),
            serpapi=_one_of(h.get("serpapi", "serp"), _hosted_modes, "serp"),
        ),
        profile_import=ProfileImportSection(
            enabled=bool(p.get("enabled", False)),
            browsers=_string_list(p.get("browsers"), _known_browsers, ["chrome", "edge", "brave", "firefox"]),
            domains=_string_list(p.get("domains"), None, ProfileImportSection().domains),
            all_profiles=bool(p.get("all_profiles", False)),
            refresh_hours=float(p.get("refresh_hours", 12.0)),
            purge_on_disable=bool(p.get("purge_on_disable", True)),
        ),
        query_rewriter=QueryRewriterSection(
            enabled=bool(w.get("enabled", False)),
            timeout_seconds=float(w.get("timeout_seconds", 75.0)),
            max_queries=int(w.get("max_queries", 3)),
            mode=_one_of(w.get("mode", "always"), {"heuristic", "always"}, "always"),
            conversational_markers=tuple(
                _string_list(w.get("conversational_markers"), None, [])
            ),
        ),
        image_engines=ImageEnginesSection(
            bing=bool(im.get("bing", True)),
            brave=bool(im.get("brave", True)),
            duckduckgo=bool(im.get("duckduckgo", True)),
            openverse=bool(im.get("openverse", False)),
        ),
        rich_cards=RichCardsSection(
            enabled=bool(rc.get("enabled", False)),
            wikipedia=bool(rc.get("wikipedia", True)),
            stackexchange=bool(rc.get("stackexchange", True)),
            reddit=bool(rc.get("reddit", True)),
            max_cards=int(rc.get("max_cards", 3)),
            timeout_seconds=float(rc.get("timeout_seconds", 2.0)),
            ttl_seconds=float(rc.get("ttl_seconds", 86400.0)),
            negative_ttl_seconds=float(rc.get("negative_ttl_seconds", 900.0)),
        ),
        localization=LocalizationSection(
            enabled=bool(l.get("enabled", True)),
            language_penalty=float(l.get("language_penalty", 0.12)),
        ),
        engines=EnginesSection(
            google=bool(g.get("google", True)),
            duckduckgo=bool(g.get("duckduckgo", True)),
            startpage=bool(g.get("startpage", True)),
            qwant=bool(g.get("qwant", True)),
            brave=bool(g.get("brave", True)),
            yandex=bool(g.get("yandex", False)),
        ),
        buffer_engine=BufferEngineSection(
            enabled=bool(buf.get("enabled", True)),
        ),
    )

    if path is None:
        _cached_config = config
        _cached_config_signature = signature
    return config
