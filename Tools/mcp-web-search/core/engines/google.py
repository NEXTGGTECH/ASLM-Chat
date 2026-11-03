# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlparse

import random

from ..fetch.profiles import for_engine
from .models import EngineParseResult, EngineRequest, SearchResult
from .parsing import (
    classify_parse,
    first_attribute,
    first_cards,
    first_node_text,
    parse_html,
    split_region,
    valid_http_url,
)

# SOCS cookie deliberately omitted — it triggers consent-gate redirects on some IPs.
_GOOGLE_COOKIES = {
    "CONSENT": "YES+",
}

# Google Search App ("Google Go") user agents: Android Chrome + the " NSTNWV" GSA marker.
# Google serves a clean, parseable mobile SERP to the GSA UA (the SearXNG approach); a
# desktop Chrome UA gets a JS-gated/withheld page that yields zero organic results.
_GSA_USER_AGENTS = (
    "Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.6478.71 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 11; Pixel 5) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.6422.165 Mobile Safari/537.36",
    "Mozilla/5.0 (Linux; Android 12; SM-G991B) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.6367.179 Mobile Safari/537.36",
)


# A random Google Search App user agent (Android UA + the " NSTNWV" Google-Go marker).
def _gsa_user_agent() -> str:
    return random.choice(_GSA_USER_AGENTS) + " NSTNWV"


# Unwrap a Google redirect URL to the actual destination URL.
def _unwrap_url(value: str) -> str:
    if value.startswith("/url?"):
        value = f"https://www.google.com{value}"
    parsed = urlparse(value)
    if parsed.netloc.endswith("google.com") and parsed.path == "/url":
        target = parse_qs(parsed.query).get("q", [""])[0]
        return unquote(target) if target else ""
    return value


# Return True when the URL points to a Google-owned domain.
def _is_internal(value: str) -> bool:
    host = urlparse(value).netloc.lower()
    return host.endswith("google.com") or host.endswith("googleusercontent.com")


# Google SERP parser with structural fallbacks and degradation reporting.
class GoogleHtmlParser:
    name = "google"
    # Engines sharing a provider family serve the same underlying index; consensus
    # voting counts one vote per family, not per engine.
    provider_family = "google"
    search_url = "https://www.google.com/search"

    # Lowercase page fragments that mean Google served a consent gate, bot challenge or JavaScript wall.
    _BLOCK_MARKERS = (
        "/sorry/",
        "unusual traffic",
        "our systems have detected",
        "/httpservice/retry/enablejs",
        "before you continue to google",
        "consent.google.com/save",
    )

    # Lowercase page fragments that mean the query genuinely returned no results.
    _EMPTY_MARKERS = ("did not match any documents", "no results found for")

    # Known result card layouts as (variant name, CSS selector), tried in order until one matches.
    _VARIANTS = (
        ("result_containers", "div.MjjYud, div.Gx5Zad"),
        ("title_anchors", "a[href]:has(h3)"),
    )

    # Build the HTTP request for a Google search query using a random browser profile.
    @staticmethod
    def build_request(
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        page: int = 1,
    ) -> EngineRequest:
        country, language = split_region(region)
        profile = for_engine("google")

        # Clean param set (no num/cr): those push Google into a degraded/withheld layout.
        params = {
            "q": query,
            "filter": "0",
            "safe": {"on": "high", "moderate": "medium", "off": "off"}.get(safesearch, "medium"),
            "start": str(max(0, page - 1) * 10),
            "ie": "utf8",
            "oe": "utf8",
            "hl": f"{language}-{country.upper()}",
            "lr": f"lang_{language}",
        }
        if timelimit:
            params["tbs"] = f"qdr:{timelimit}"

        # Google Search App identity: a GSA UA + Accept:*/* gets the clean, parseable
        # mobile SERP. A desktop browser UA gets a JS-gated page with no organic results.
        headers = {
            "User-Agent": _gsa_user_agent(),
            "Accept": "*/*",
            "Accept-Language": f"{language}-{country.upper()},{language};q=0.9",
        }
        return EngineRequest(
            method="GET",
            url=GoogleHtmlParser.search_url,
            params=params,
            headers=headers,
            cookies=_GOOGLE_COOKIES,
            primp_target=profile.primp_target,
            primp_os=profile.primp_os,
            identity_key="google",
        )

    # Parse a raw Google SERP HTML document into an EngineParseResult.
    def parse(self, document: str) -> EngineParseResult:
        # Detect block and empty-result pages from the raw lowercase text.
        lowered = document.lower()
        blocked = any(marker in lowered for marker in self._BLOCK_MARKERS)
        explicit_empty = any(marker in lowered for marker in self._EMPTY_MARKERS)

        # Pick the first card layout that matches the page.
        tree = parse_html(document)
        variant, cards = first_cards(tree, self._VARIANTS)

        # Extract title, link and snippet per card; result links are unwrapped from Google redirects, and
        # cards without a title or a valid external http(s) link are counted as malformed.
        results: list[SearchResult] = []
        malformed = 0
        for card in cards:
            title = first_node_text(card, ("h3",))
            href = first_attribute(card, (":scope", "a:has(h3)"), "href")
            href = _unwrap_url(href)
            snippet = first_node_text(
                card,
                (
                    ".VwiC3b",
                    ".yXK7lf",
                    "div[data-sncf]",
                    # Snippet class of the current GSA/mobile SERP.
                    ".ilUpNd",
                ),
            )
            if not title or not valid_http_url(href) or _is_internal(href):
                malformed += 1
                continue
            results.append(SearchResult(title=title, url=href, snippet=snippet))

        # A block marker only counts when no results were parsed.
        return classify_parse(
            engine=self.name,
            results=results,
            parser_variant=variant,
            cards_seen=len(cards),
            malformed_cards=malformed,
            blocked=blocked and not results,
            explicit_empty=explicit_empty,
        )
