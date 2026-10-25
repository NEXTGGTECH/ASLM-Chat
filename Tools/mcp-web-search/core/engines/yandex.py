# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import time

from ..fetch.profiles import accept_language_for, build_nav_headers, for_engine
from .models import EngineParseResult, EngineRequest, ParseStatus, SearchResult
from .parsing import (
    classify_parse,
    first_attribute,
    first_cards,
    first_node_text,
    parse_html,
    split_region,
    valid_http_url,
)

# Languages Yandex's site-search widget accepts as the 'lang' argument.
_SUPPORTED_LANGS = frozenset(
    {"ru", "en", "be", "fr", "de", "id", "kk", "tt", "tr", "uk"}
)


# ── adaptive User-Agent mode (native ya.ru) ────────────────────────────────────
# Default leg: a Yandex Browser UA on top of the pinned chrome TLS — for ya.ru the
# most coherent client story (their own Chromium). Measured at parity with the
# cookieless "no UA" recipe, but the two legs' captcha dice are independent, so
# when the Yandex leg runs into a captcha streak the engine temporarily drops the
# explicit UA (the proven no-identity recipe) for a cooldown window, then returns
# to the default. Mode state is module-level: parse() feeds outcomes, build_request
# reads the mode (plain assignments only — no awaits, GIL is the lock).
_YB_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/146.0.0.0 YaBrowser/26.10.0.0 Safari/537.36"
)
_PLAIN_MODE_SECONDS = 15 * 60.0
_BLOCKED_STRIKES_TO_FLIP = 2

_ua_mode = "yabrowser"  # "yabrowser" | "plain"
_plain_until = 0.0
_blocked_streak = 0


# The User-Agent header value for the current mode ("" = send no UA header).
def _active_user_agent() -> str:
    global _ua_mode
    if _ua_mode == "plain":
        if time.monotonic() < _plain_until:
            return ""
        _ua_mode = "yabrowser"  # cooldown elapsed — back to the default leg
    return _YB_USER_AGENT


# Feed one parse outcome into the mode state machine.
def _note_parse_outcome(status: ParseStatus) -> None:
    global _blocked_streak, _ua_mode, _plain_until
    if status == ParseStatus.BLOCKED:
        _blocked_streak += 1
        if _blocked_streak >= _BLOCKED_STRIKES_TO_FLIP:
            _ua_mode = "plain"
            _plain_until = time.monotonic() + _PLAIN_MODE_SECONDS
    else:
        _blocked_streak = 0
        if _ua_mode == "plain" and time.monotonic() >= _plain_until:
            _ua_mode = "yabrowser"


# Yandex web parser using the native ya.ru SERP.
#
# The native endpoint serves Yandex's current search markup (div.Organic with
# OrganicTitle-Link anchors). It is aggressively captcha-gated for scripted
# clients, so the expected production path is the warm-browser fallback: a real
# Chromium render of the same URL passes where every HTTP fingerprint fails.
# Ad placements carry the action-counter class and are skipped so only the
# organic rank the user sees survives.
class YandexParser:
    name = "yandex"
    provider_family = "yandex"
    search_url = "https://ya.ru/search/"

    _BLOCK_MARKERS = ("showcaptcha", "smartcaptcha", "checkcaptcha", "robot check", "captcha")
    _EMPTY_MARKERS = ("nothing found", "there are no search results", "ничего не нашлось", "ничего не найдено")
    _VARIANTS = (
        ("native_organic", "div.Organic"),
        ("legacy_serp_items", "li[class*='serp-item']"),
    )

    # Build a native Yandex search request.
    @staticmethod
    def build_request(
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        page: int = 1,
    ) -> EngineRequest:
        del safesearch, timelimit
        _, language = split_region(region)
        params = {"text": query}
        if page > 1:
            params["p"] = str(page - 1)
        headers = {"Accept-Language": "en-US,en;q=0.9"}
        accept_language = accept_language_for(language)
        if accept_language:
            headers["Accept-Language"] = accept_language
        user_agent = _active_user_agent()
        if user_agent:
            headers["User-Agent"] = user_agent
        return EngineRequest(
            method="GET",
            url=YandexParser.search_url,
            params=params,
            headers=headers,
            # ya.ru captcha-gates primp's firefox fingerprint on sight (0/4 in testing)
            # while chrome/edge TLS pass consistently. The UA (or its absence — see
            # _active_user_agent) is the whole header identity Yandex sees — pin it
            # to chromium instead of the deterministic for_engine profile (which
            # hashes to a firefox one for this engine).
            primp_target="chrome",
            primp_os="windows",
            identity_key="yandex",
        )

    # Parse a raw native Yandex HTML document into normalized results.
    def parse(self, document: str) -> EngineParseResult:
        lowered = document.lower()
        blocked = any(marker in lowered for marker in self._BLOCK_MARKERS)
        explicit_empty = any(marker in lowered for marker in self._EMPTY_MARKERS)

        tree = parse_html(document)
        variant, cards = first_cards(tree, self._VARIANTS)

        results: list[SearchResult] = []
        malformed = 0
        for card in cards:
            native_link = card.css_first("a.OrganicTitle-Link")
            if native_link is not None and "action-counter" in (
                native_link.attributes.get("class") or ""
            ).split():
                # Paid placements use Yandex tracking links and the action-counter
                # class; they are not part of the organic rank the user sees.
                continue
            title = first_node_text(
                card,
                (
                    ".OrganicTitleContentSpan",
                    ".OrganicTitle-LinkText",
                    "h3.b-serp-item__title a.b-serp-item__title-link",
                    ".b-serp-item__title-link",
                    "h3",
                ),
            )
            href = first_attribute(
                card,
                (
                    "a.OrganicTitle-Link",
                    "a.b-serp-item__title-link",
                    "h3 a",
                    "a",
                ),
                "href",
            )
            snippet = first_node_text(
                card,
                (
                    ".OrganicTextContentSpan",
                    ".OrganicText",
                    ".b-serp-item__content .b-serp-item__text",
                    ".b-serp-item__text",
                ),
            )
            if not title or not valid_http_url(href):
                malformed += 1
                continue
            results.append(SearchResult(title=title, url=href, snippet=snippet))

        outcome = classify_parse(
            engine=self.name,
            results=results,
            parser_variant=variant,
            cards_seen=len(cards),
            malformed_cards=malformed,
            blocked=blocked and not results,
            explicit_empty=explicit_empty,
        )
        _note_parse_outcome(outcome.status)
        return outcome


# Legacy Yandex web parser using the retired site-search ("frame") widget.
#
# The widget endpoint is Yandex's outdated search product; it stays scrapeable
# long after the native SERP hardened, but its index and markup are frozen in
# the legacy product. Kept for networks where the native SERP will not render.
class LegacyYandexParser:
    name = "legacy_yandex"
    provider_family = "yandex"
    search_url = "https://yandex.com/search/site/"

    _BLOCK_MARKERS = ("showcaptcha", "smartcaptcha", "checkcaptcha", "robot check", "captcha")
    _EMPTY_MARKERS = ("nothing found", "there are no search results", "ничего не нашлось", "ничего не найдено")
    _VARIANTS = (
        ("serp_items", "li[class*='serp-item']"),
        ("organic_results", "li[data-cid]"),
    )

    # Build a Yandex site-search request with the fixed widget parameters.
    @staticmethod
    def build_request(
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        page: int = 1,
    ) -> EngineRequest:
        _, language = split_region(region)
        profile = for_engine("yandex")
        params = {
            "tmpl_version": "releases",
            "text": query,
            "web": "1",
            "frame": "1",
            "searchid": "3131712",
        }
        if language in _SUPPORTED_LANGS:
            params["lang"] = language
        if page > 1:
            params["p"] = str(page - 1)

        extra: dict[str, str] = {}
        accept_language = accept_language_for(language)
        if accept_language:
            extra["Accept-Language"] = accept_language
        headers = build_nav_headers(
            profile,
            referer="https://yandex.com/",
            sec_fetch_site="same-origin",
            extra=extra or None,
        )
        cookies = {"yp": "1716337604.sp.family%3A0#1685406411.szm.1:1920x1080:1920x999"}
        return EngineRequest(
            method="GET",
            url=LegacyYandexParser.search_url,
            params=params,
            headers=headers,
            cookies=cookies,
            primp_target=profile.primp_target,
            primp_os=profile.primp_os,
            identity_key="yandex",
        )

    # Parse a raw Yandex site-search HTML document into normalized results.
    def parse(self, document: str) -> EngineParseResult:
        lowered = document.lower()
        blocked = any(marker in lowered for marker in self._BLOCK_MARKERS)
        explicit_empty = any(marker in lowered for marker in self._EMPTY_MARKERS)

        tree = parse_html(document)
        variant, cards = first_cards(tree, self._VARIANTS)

        results: list[SearchResult] = []
        malformed = 0
        for card in cards:
            title = first_node_text(
                card,
                (
                    "h3.b-serp-item__title a.b-serp-item__title-link",
                    ".b-serp-item__title-link",
                    "h3",
                ),
            )
            href = first_attribute(
                card,
                (
                    "a.b-serp-item__title-link",
                    "h3 a",
                    "a",
                ),
                "href",
            )
            snippet = first_node_text(
                card,
                (
                    ".b-serp-item__content .b-serp-item__text",
                    ".b-serp-item__text",
                ),
            )
            if not title or not valid_http_url(href):
                malformed += 1
                continue
            results.append(SearchResult(title=title, url=href, snippet=snippet))

        return classify_parse(
            engine=self.name,
            results=results,
            parser_variant=variant,
            cards_seen=len(cards),
            malformed_cards=malformed,
            blocked=blocked and not results,
            explicit_empty=explicit_empty,
        )
