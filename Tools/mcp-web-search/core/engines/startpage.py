# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import json
import re
import time
from html import unescape
from typing import Protocol
from urllib.parse import urlparse

from ..fetch.profiles import accept_language_for, build_nav_headers, for_engine
from .models import EngineParseResult, EngineRequest, ParseStatus, SearchResult
from .parsing import split_region

_HTML_TAG_RE = re.compile(r"<[^>]+>")

# Startpage safesearch level expected by the form ('qadf'/family-filter).
_SAFESEARCH = {"off": "none", "moderate": "moderate", "on": "heavy"}
_TIME_RANGE = {"d": "d", "w": "w", "m": "m", "y": "y"}

_BASE_URL = "https://www.startpage.com"
_SC_URL = f"{_BASE_URL}/"
_SEARCH_URL = f"{_BASE_URL}/sp/search"

# Startpage stamps a per-session 'sc' token on its search form; requests without
# it are treated as bots and captcha'd. The token rotates slowly, so it is cached
# process-wide and refreshed at most once per TTL. The lock prevents a stampede
# of homepage fetches when several searches start at once with a cold cache.
_SC_TTL = 3600.0
# After a failed scrape (cold/blocked homepage), back off this long before trying
# again — otherwise every Startpage search serializes a fresh homepage fetch under
# the global lock, turning one blocked homepage into a queue across all searches.
_SC_RETRY_COOLDOWN = 30.0
_SC_LOCK = asyncio.Lock()
_sc_code: str = ""
_sc_fetched_at: float = 0.0
_sc_failed_at: float = 0.0


# Collapse HTML markup and entities in a Startpage field into plain text.
def _to_text(value: object) -> str:
    return " ".join(unescape(_HTML_TAG_RE.sub(" ", str(value or ""))).split())


# Return the substring between the first 'start' marker and the next 'end' marker.
def _between(text: str, start: str, end: str) -> str:
    i = text.find(start)
    if i < 0:
        return ""
    i += len(start)
    j = text.find(end, i)
    if j < 0:
        return ""
    return text[i:j]


# Minimal transport surface needed to prefetch the sc token.
class _Transport(Protocol):
    async def fetch(self, request: EngineRequest): ...


# Scrape a fresh 'sc' token from Startpage's homepage form.
async def _fetch_sc_code(transport: _Transport) -> str:
    from .parsing import first_attribute, parse_html

    profile = for_engine("startpage")
    headers = build_nav_headers(profile, referer=f"{_BASE_URL}/", sec_fetch_site="same-origin")
    request = EngineRequest(
        method="GET",
        url=_SC_URL,
        headers=headers,
        primp_target=profile.primp_target,
        primp_os=profile.primp_os,
        identity_key="startpage",
    )
    response = await transport.fetch(request)
    if response.status >= 400:
        return ""
    tree = parse_html(response.text())
    node = tree
    return first_attribute(node, ('form#search input[name="sc"]', 'input[name="sc"]'), "value")


# Return a cached sc token, refreshing it through the transport when stale.
async def _get_sc_code(transport: _Transport) -> str:
    global _sc_code, _sc_fetched_at, _sc_failed_at
    now = time.monotonic()
    if _sc_code and (now - _sc_fetched_at) < _SC_TTL:
        return _sc_code
    async with _SC_LOCK:
        now = time.monotonic()
        if _sc_code and (now - _sc_fetched_at) < _SC_TTL:
            return _sc_code
        # A recent scrape failure is still cooling down — reuse the stale/empty
        # token instead of hammering the homepage on every search.
        # Apply backoff only after an actual failure, not to the first request after an OS startup.
        if _sc_failed_at and (now - _sc_failed_at) < _SC_RETRY_COOLDOWN:
            return _sc_code
        code = await _fetch_sc_code(transport)
        if code:
            _sc_code = code
            _sc_fetched_at = now
        else:
            _sc_failed_at = now
    return _sc_code


# Startpage parser. Startpage is a privacy frontend that serves Google's web
# index (provider family: google), so it gives Google-quality results through a
# path that stays available when Google's own SERP is rate-limiting. The request
# is two-phase (prefetch an sc token, then POST the search form); the parsed
# output is the same standard EngineParseResult every other engine returns.
class StartpageParser:
    name = "startpage"
    # Same Google index behind a different door — one consensus vote with Google.
    provider_family = "google"
    search_url = _SEARCH_URL

    _BLOCK_MARKERS = ("/sp/captcha", "verify you are human", "px-captcha")
    _EMPTY_MARKERS = ("no results found", "did not return any results")

    # Build the search request, prefetching the sc token through the transport.
    @staticmethod
    async def build_request_async(
        transport: _Transport,
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        page: int = 1,
    ) -> EngineRequest:
        country, language = split_region(region)
        sc_code = await _get_sc_code(transport)
        profile = for_engine("startpage")

        data: dict[str, str] = {
            "query": query,
            "cat": "web",
            "t": "device",
            "abd": "1",
            "abe": "1",
            "qsr": "all",
            "qadf": _SAFESEARCH.get(safesearch, "moderate"),
            "language": language,
            "lui": language,
        }
        if sc_code:
            data["sc"] = sc_code
        if timelimit and timelimit in _TIME_RANGE:
            data["with_date"] = _TIME_RANGE[timelimit]
        if page > 1:
            data["page"] = str(page)
            data["segment"] = "startpage.udog"

        # Startpage selects the region in the 'preferences' cookie and the language
        # in the POST body, mirroring its own search form.
        prefs = {
            "date_time": "world",
            "disable_family_filter": _SAFESEARCH.get(safesearch, "moderate"),
            "disable_open_in_new_window": "0",
            "enable_post_method": "1",
            "enable_stay_control": "1",
            "instant_answers": "1",
            "lang_homepage": "s/device/en/",
            "num_of_results": "10",
            "suggestions": "1",
            "wt_unit": "celsius",
            "language": language,
            "language_ui": language,
            "search_results_region": f"{language}-{country.upper()}",
        }
        preferences = "N1N".join(f"{key}EEE{value}" for key, value in prefs.items())

        extra = {"Origin": _BASE_URL}
        accept_language = accept_language_for(language, country)
        if accept_language:
            extra["Accept-Language"] = accept_language
        headers = build_nav_headers(
            profile,
            referer=f"{_BASE_URL}/",
            sec_fetch_site="same-origin",
            extra=extra,
        )
        return EngineRequest(
            method="POST",
            url=_SEARCH_URL,
            data=data,
            headers=headers,
            cookies={"preferences": preferences},
            primp_target=profile.primp_target,
            primp_os=profile.primp_os,
            identity_key="startpage",
        )

    # Parse Startpage's HTML, reading the embedded JSON the React app renders.
    def parse(self, document: str) -> EngineParseResult:
        lowered = document.lower()
        blocked = any(marker in lowered for marker in self._BLOCK_MARKERS)
        explicit_empty = any(marker in lowered for marker in self._EMPTY_MARKERS)

        raw = _between(document, "React.createElement(UIStartpage.AppSerpWeb, {", "}})")
        results: list[SearchResult] = []
        cards_seen = 0
        malformed = 0
        diagnostics: list[str] = []

        if raw:
            try:
                payload = json.loads("{" + raw + "}}")
            except json.JSONDecodeError as exc:
                payload = None
                diagnostics.append(f"Invalid embedded JSON: {exc}")
            if payload is not None:
                regions = (
                    payload.get("render", {}).get("presenter", {}).get("regions", {})
                )
                for block in regions.get("mainline", []) or []:
                    if block.get("display_type") != "web-google":
                        continue
                    for item in block.get("results", []) or []:
                        cards_seen += 1
                        title = _to_text(item.get("title"))
                        href = str(item.get("clickUrl") or "").strip()
                        snippet = _to_text(item.get("description"))
                        parsed = urlparse(href)
                        if not title or parsed.scheme not in {"http", "https"} or not parsed.netloc:
                            malformed += 1
                            continue
                        results.append(SearchResult(title=title, url=href, snippet=snippet))

        if blocked and not results:
            status = ParseStatus.BLOCKED
        elif results:
            status = ParseStatus.PARTIAL if malformed else ParseStatus.SUCCESS
        elif explicit_empty:
            status = ParseStatus.EMPTY
        else:
            status = ParseStatus.CHANGED
            diagnostics.append("No embedded web-google results block found.")

        return EngineParseResult(
            engine=self.name,
            status=status,
            results=results,
            parser_variant="json_appserp" if raw else "none",
            cards_seen=cards_seen,
            malformed_cards=malformed,
            diagnostics=diagnostics,
        )
