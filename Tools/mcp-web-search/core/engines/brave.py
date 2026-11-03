# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

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


# Brave Search SERP parser.
class BraveParser:
    name = "brave"
    provider_family = "brave"
    search_url = "https://search.brave.com/search"

    # Lowercase page fragments that mean Brave served a bot challenge or rate limit page.
    _BLOCK_MARKERS = ("captcha", "challenge-platform", "verify you are human", "rate limit")

    # Lowercase page fragments that mean the query genuinely returned no results.
    _EMPTY_MARKERS = ("no results found", "couldn't find any results")

    # Known result card layouts as (variant name, CSS selector), tried in order until one matches.
    _VARIANTS = (
        ("data_type_web", "div[data-type='web']"),
        ("result_class", ".result"),
    )

    # Build the HTTP request for a Brave search query using a random browser profile.
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
        profile = for_engine("brave")

        # Query parameters: the time limit maps to Brave's tf values and paging uses a page offset.
        params = {"q": query, "source": "web"}
        if timelimit:
            mapped = {"d": "pd", "w": "pw", "m": "pm", "y": "py"}.get(timelimit)
            if mapped:
                params["tf"] = mapped
        if page > 1:
            params["offset"] = str(page - 1)

        # Cookies carry the region and UI language; location and summarizer are switched off, and safe search
        # is only set when it differs from the default.
        cookies = {
            "country": country,
            "useLocation": "0",
            "summarizer": "0",
            "ui_lang": f"{language}-{country}",
        }
        if safesearch != "moderate":
            cookies["safesearch"] = "strict" if safesearch == "on" else "off"

        # Sec-Fetch-Site=same-origin: pretending to search from within brave.com.
        extra: dict[str, str] = {}
        accept_language = accept_language_for(language, country)
        if accept_language:
            extra["Accept-Language"] = accept_language
        headers = build_nav_headers(
            profile,
            referer="https://search.brave.com/",
            sec_fetch_site="same-origin",
            extra=extra or None,
        )
        return EngineRequest(
            method="GET",
            url=BraveParser.search_url,
            params=params,
            headers=headers,
            cookies=cookies,
            primp_target=profile.primp_target,
            primp_os=profile.primp_os,
            identity_key="brave",
        )

    # Parse a raw Brave SERP HTML document into an EngineParseResult.
    def parse(self, document: str) -> EngineParseResult:
        # Detect block and empty-result pages from the raw lowercase text.
        lowered = document.lower()
        blocked = any(marker in lowered for marker in self._BLOCK_MARKERS)
        explicit_empty = any(marker in lowered for marker in self._EMPTY_MARKERS)

        # Pick the first card layout that matches the page.
        tree = parse_html(document)
        variant, cards = first_cards(tree, self._VARIANTS)

        # Extract title, link and snippet per card; cards without a title or a valid http(s) link are
        # counted as malformed.
        results: list[SearchResult] = []
        malformed = 0
        for card in cards:
            title = first_node_text(
                card,
                (
                    ".snippet-title",
                    ".title",
                    "h2",
                ),
            )
            href = first_attribute(
                card,
                (
                    "a:has(.title)",
                    "a:has(.snippet-title)",
                    "h2 a",
                    "a",
                ),
                "href",
            )
            snippet = first_node_text(
                card,
                (
                    ".snippet .content",
                    ".snippet-description",
                    ".description",
                    "p",
                ),
            )
            if not title or not valid_http_url(href):
                malformed += 1
                continue
            results.append(SearchResult(title=title, url=href, snippet=snippet))

        # A block marker only counts when no results were parsed (result text may mention captchas).
        return classify_parse(
            engine=self.name,
            results=results,
            parser_variant=variant,
            cards_seen=len(cards),
            malformed_cards=malformed,
            blocked=blocked and not results,
            explicit_empty=explicit_empty,
        )
