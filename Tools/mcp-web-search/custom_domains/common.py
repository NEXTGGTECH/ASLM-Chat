# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
from typing import Any

from core.extract.content_processor import _preclean_html
from core.extract.page_normalizer import (
    _build_markdown,
    _clean_content,
    _extract_content,
    _extract_meta,
    _extract_with_trafilatura_formatted,
    _fallback_segment,
)


# Chrome-like User-Agent for lightweight httpx fetches.
CHROME_BASIC_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}


# Collapse whitespace and cap string length.
def trim(text: str, limit: int = 240) -> str:
    return re.sub(r"\s+", " ", text or "").strip()[:limit]


# Parse document title from raw HTML.
def extract_title(html: str) -> str:
    match = re.search(r"<title[^>]*>(.*?)</title>", html or "", re.IGNORECASE | re.DOTALL)
    return trim(match.group(1) if match else "", 180)


# Return True when the page looks like a bot/captcha block rather than real content (heuristic).
def looks_blocked(title: str, html: str) -> bool:
    lower_title = (title or "").lower()
    lower_html = (html or "").lower()
    return any(
        marker in lower_title or marker in lower_html
        for marker in ("captcha", "pardon our interruption", "are you not a robot", "showcaptcha", "smartcaptcha")
    )


# Drop known Trafilatura noise lines from extracted markdown.
def clean_trafilatura_snapshot(text: str) -> str:
    lines = [line.rstrip() for line in (text or "").splitlines()]
    cleaned: list[str] = []

    # Lowercase lines (without markdown heading marks) that are boilerplate and get dropped.
    skip_markers = (
        "oops! looks like we're having trouble connecting to our server.",
        "refresh your browser window to try again.",
        "payments:",
    )

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower().lstrip("#").strip()
        if any(marker == lower for marker in skip_markers):
            continue
        cleaned.append(line)

    return "\n".join(cleaned).strip()


# Run normalize_page or trafilatura extraction and build markdown snapshot fields.
def extract_with_preferred_pipeline(url: str, raw_html: str, *, prefer_trafilatura: bool) -> dict[str, Any]:
    meta = _extract_meta(raw_html, None, url)
    cleaned_html = _preclean_html(raw_html)
    strategy = "normalize_page"

    # Extract the main text: trafilatura first when preferred (falling back to the normalizer if it
    # returns nothing), the normalizer otherwise. `strategy` records which path produced the text.
    if prefer_trafilatura:
        text = _extract_with_trafilatura_formatted(cleaned_html, url)
        strategy = "trafilatura"
        if not text.strip():
            text = _extract_content(raw_html, None, url)
            strategy = "trafilatura_fallback"
    else:
        text = _extract_content(raw_html, None, url)

    # Clean the text and build the markdown; a long single-block result is re-segmented first.
    if not text.strip():
        markdown = _build_markdown(meta, "*No content extracted.*")
    else:
        cleaned = clean_trafilatura_snapshot(text) if prefer_trafilatura else text
        cleaned = _clean_content(cleaned, strict=not prefer_trafilatura)
        block_count = len([block for block in cleaned.split("\n\n") if block.strip()])
        if block_count <= 1 and len(cleaned) > 300:
            cleaned = _fallback_segment(text)
            cleaned = _clean_content(cleaned, strict=not prefer_trafilatura)
        markdown = _build_markdown(meta, cleaned)

    # Snapshot fields; the preview is a short head of the markdown.
    return {
        "strategy": strategy,
        "markdown": markdown,
        "markdown_chars": len(markdown),
        "markdown_preview": trim(markdown, 400),
    }
