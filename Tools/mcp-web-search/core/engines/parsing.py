# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
from collections.abc import Iterable
from urllib.parse import urlparse

from selectolax.lexbor import LexborHTMLParser, LexborNode

from .models import EngineParseResult, ParseStatus, SearchResult

# Matches any run of whitespace so it can be collapsed into a single space.
_SPACE_RE = re.compile(r"\s+")


# Parse an HTML document string into a LexborHTMLParser tree.
def parse_html(document: str) -> LexborHTMLParser:
    return LexborHTMLParser(document)


# Join and collapse whitespace from an iterable of text parts into one string.
def clean_text(parts: Iterable[object]) -> str:
    return _SPACE_RE.sub(" ", " ".join(str(part) for part in parts if part)).strip()


# Extract normalized text content from a node, or return an empty string.
def node_text(node: LexborNode | None) -> str:
    if node is None:
        return ""
    return _SPACE_RE.sub(" ", node.text(separator=" ", strip=True)).strip()


# Return the first non-empty text value matched by any of the given CSS selectors.
def first_node_text(node: LexborNode, selectors: Iterable[str]) -> str:
    for selector in selectors:
        value = node_text(node.css_first(selector))
        if value:
            return value
    return ""


# Return the first non-empty attribute value matched by any of the given selectors.
def first_attribute(node: LexborNode, selectors: Iterable[str], attribute: str) -> str:
    for selector in selectors:
        selected = node if selector == ":scope" else node.css_first(selector)
        if selected is not None:
            value = str(selected.attributes.get(attribute) or "").strip()
            if value:
                return value
    return ""


# Return the first variant name and node list whose CSS selector matches cards in the tree.
def first_cards(tree: LexborHTMLParser, variants: Iterable[tuple[str, str]]) -> tuple[str, list[LexborNode]]:
    for variant, selector in variants:
        cards = tree.css(selector)
        if cards:
            return variant, cards
    return "none", []


# Return True when the value is an absolute HTTP or HTTPS URL with a host.
def valid_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


# Split a region string like "us-en" into a (country, language) tuple.
def split_region(region: str) -> tuple[str, str]:
    normalized = (region or "us-en").replace("_", "-").lower()
    country, separator, language = normalized.partition("-")
    if not separator or not country or not language:
        return "us", "en"
    return country, language


# Remove duplicate URLs from a result list while preserving order.
def deduplicate(results: Iterable[SearchResult]) -> list[SearchResult]:
    output: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        if result.url in seen:
            continue
        seen.add(result.url)
        output.append(result)
    return output


# Classify a parse outcome and build an EngineParseResult from raw parse metrics. A block marker wins over
# results; results become PARTIAL when any card was malformed or fewer than 65% of the seen cards parsed;
# no results with an explicit empty marker is EMPTY, and no results without one means the page layout
# changed.
def classify_parse(
    *,
    engine: str,
    results: list[SearchResult],
    parser_variant: str,
    cards_seen: int,
    malformed_cards: int,
    blocked: bool = False,
    explicit_empty: bool = False,
    diagnostics: list[str] | None = None,
) -> EngineParseResult:
    # Copy the diagnostics so the caller's list is never mutated.
    diagnostics = list(diagnostics or [])

    # Pick the status in priority order: blocked, results, explicit empty, unknown layout.
    if blocked:
        status = ParseStatus.BLOCKED
    elif results:
        coverage = len(results) / max(cards_seen, len(results))
        status = ParseStatus.PARTIAL if malformed_cards or coverage < 0.65 else ParseStatus.SUCCESS
    elif explicit_empty:
        status = ParseStatus.EMPTY
    else:
        status = ParseStatus.CHANGED
        diagnostics.append("No usable results and no explicit empty-result marker.")

    # Deduplicate by URL so repeated cards do not inflate the result list.
    return EngineParseResult(
        engine=engine,
        status=status,
        results=deduplicate(results),
        parser_variant=parser_variant,
        cards_seen=cards_seen,
        malformed_cards=malformed_cards,
        diagnostics=diagnostics,
    )
