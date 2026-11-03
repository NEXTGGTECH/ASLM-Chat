# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlparse

from ..fetch.profiles import accept_language_for, build_nav_headers, for_engine
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


# Unwrap a DuckDuckGo redirect URL to the actual destination URL.
def _unwrap_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(target) if target else ""
    return value


# DuckDuckGo HTML SERP parser.
class DuckDuckGoParser:
    name = "duckduckgo"
    provider_family = "duckduckgo"
    search_url = "https://html.duckduckgo.com/html/"

    # Lowercase page fragments that mean DuckDuckGo served an anomaly or bot challenge page.
    _BLOCK_MARKERS = ("anomaly-modal", "challenge-form", "verify you are human", "captcha")

    # Lowercase page fragments that mean the query genuinely returned no results.
    _EMPTY_MARKERS = ("no results.", "no more results")

    # Known result card layouts as (variant name, CSS selector), tried in order until one matches.
    _VARIANTS = (
        ("result_cards", "div.result"),
        ("compact_body", "div.body"),
    )

    # Build the HTTP request for a DuckDuckGo search query using a random browser profile.
    @staticmethod
    def build_request(
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        page: int = 1,
    ) -> EngineRequest:
        profile = for_engine("duckduckgo")

        # Form fields: s is the result offset (first page 10, later pages step by 15), df the time limit and
        # kp the safe search level (-2 off, 1 strict).
        data: dict[str, str] = {"q": query, "b": "", "l": region}
        if page > 1:
            data["s"] = str(10 + (page - 2) * 15)
        if timelimit:
            data["df"] = timelimit
        if safesearch == "off":
            data["kp"] = "-2"
        elif safesearch == "on":
            data["kp"] = "1"

        # POST to html endpoint; Sec-Fetch-Site=same-site because form is on duckduckgo.com.
        _, language = split_region(region)
        extra: dict[str, str] = {}
        accept_language = accept_language_for(language)
        if accept_language:
            extra["Accept-Language"] = accept_language
        headers = build_nav_headers(
            profile,
            referer="https://duckduckgo.com/",
            sec_fetch_site="same-site",
            extra=extra or None,
        )
        return EngineRequest(
            method="POST",
            url=DuckDuckGoParser.search_url,
            data=data,
            headers=headers,
            primp_target=profile.primp_target,
            primp_os=profile.primp_os,
            identity_key="duckduckgo",
        )

    # Parse a raw DuckDuckGo SERP HTML document into an EngineParseResult.
    def parse(self, document: str) -> EngineParseResult:
        # Detect block and empty-result pages from the raw lowercase text.
        lowered = document.lower()
        blocked = any(marker in lowered for marker in self._BLOCK_MARKERS)
        explicit_empty = any(marker in lowered for marker in self._EMPTY_MARKERS)

        # Pick the first card layout that matches the page.
        tree = parse_html(document)
        variant, cards = first_cards(tree, self._VARIANTS)

        # Extract title, link and snippet per card; result links are unwrapped from DuckDuckGo redirects, and
        # cards without a title or a valid http(s) link are counted as malformed.
        results: list[SearchResult] = []
        malformed = 0
        for card in cards:
            title = first_node_text(card, (".result__title", "h2"))
            href = first_attribute(
                card,
                (
                    "a.result__a",
                    "h2 a",
                    "a",
                ),
                "href",
            )
            href = _unwrap_url(href)
            snippet = first_node_text(
                card,
                (
                    ".result__snippet",
                    ".result-snippet",
                ),
            )
            if not title or not valid_http_url(href):
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
