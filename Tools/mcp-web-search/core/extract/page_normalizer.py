# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import html as html_lib
import re
from typing import Optional
from urllib.parse import urlparse

# trafilatura and bs4 are optional: the flags record whether each is importable, and extraction
# falls back to simpler methods without them.
try:
    import trafilatura  # type: ignore
    _HAS_TRAFILATURA = True
except Exception:
    _HAS_TRAFILATURA = False

# Same for BeautifulSoup: _HAS_BS4 gates the DOM-based code paths.
try:
    from bs4 import BeautifulSoup  # type: ignore
    _HAS_BS4 = True
except Exception:
    _HAS_BS4 = False

from core.extract.content_processor import (
    _preclean_html,
    _extract_text_with_bs4,
    _regex_html_to_text,
    _normalize_text,
    _dedupe_blocks,
    _get_boilerplate_filter,
)

# Cap on the length of the final markdown, and the placeholder used when nothing was extracted.
_MAX_OUTPUT_CHARS = 50_000
_NO_CONTENT_MARKER = "*No content extracted.*"

# Runs of horizontal whitespace (newlines excluded), and an HTML tag.
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_TAG_RE = re.compile(r"<[^>]+>")

# Line starts that mark structured markdown: a heading (# to ######), a bullet or numbered list
# item, a block quote and a code fence.
_MD_HEADING_RE = re.compile(r"^#{1,6}\s")
_MD_LIST_RE = re.compile(r"^(\s*[-*•]\s|\s*\d+[.)]\s)")
_MD_QUOTE_RE = re.compile(r"^>\s?")
_MD_CODE_FENCE_RE = re.compile(r"^```")


# Return whether normalized markdown contains page content rather than metadata alone.
def has_extractable_content(markdown: str) -> bool:
    return bool(markdown and markdown.strip() and _NO_CONTENT_MARKER not in markdown)


# Extract title, author and date metadata using trafilatura (CPU only). Return whatever was found.
def _extract_meta_trafilatura(raw_html: str) -> dict[str, str]:
    meta: dict[str, str] = {}
    try:
        obj = trafilatura.extract_metadata(raw_html)
        if obj is None:
            return meta
        if getattr(obj, "title", None):
            meta["title"] = str(obj.title).strip()
        if getattr(obj, "author", None):
            meta["author"] = str(obj.author).strip()
        if getattr(obj, "date", None):
            meta["date"] = str(obj.date).strip()
    except Exception:
        # Metadata is optional; keep what was collected before trafilatura failed.
        pass
    return meta


# Extract title, author and date metadata from HTML with regular expressions (no parser needed).
def _extract_meta_fallback(raw_html: str) -> dict[str, str]:
    meta: dict[str, str] = {}

    # Title: the text of the <title> element.
    m = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.IGNORECASE | re.DOTALL)
    if m:
        meta["title"] = _normalize_text(m.group(1))

    # Author: the content of <meta name="author">.
    m = re.search(
        r'<meta[^>]+name=["\']author["\'][^>]+content=["\'](.*?)["\']',
        raw_html, re.IGNORECASE,
    )
    if m:
        meta["author"] = _normalize_text(m.group(1))

    # Date: the first match among article:published_time, name=date, name=publish(_/-)date
    # metas and a <time datetime> attribute; each pattern captures the attribute value.
    for pattern in (
        r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+name=["\']date["\'][^>]+content=["\'](.*?)["\']',
        r'<meta[^>]+name=["\']publish[_-]?date["\'][^>]+content=["\'](.*?)["\']',
        r'<time[^>]+datetime=["\'](.*?)["\']',
    ):
        m = re.search(pattern, raw_html, re.IGNORECASE)
        if m:
            meta["date"] = _normalize_text(m.group(1))
            break

    return meta


# Build a meta dict (title, author, date, site, url) from HTML, fallback text and URL.
def _extract_meta(raw_html: Optional[str], fallback_text: Optional[str], url: str) -> dict[str, str]:
    meta: dict[str, str] = {}

    # Prefer trafilatura metadata; use the regex fallback when there is no title, keeping any
    # non-empty trafilatura values on top of it.
    if raw_html:
        if _HAS_TRAFILATURA:
            meta = _extract_meta_trafilatura(raw_html)
        if not meta.get("title"):
            fb = _extract_meta_fallback(raw_html)
            meta = {**fb, **{k: v for k, v in meta.items() if v}}

    # Still no title: use the first line of the fallback text when it is short and does not end
    # like a sentence (so it looks like a heading).
    if not meta.get("title") and fallback_text:
        first_line = fallback_text.strip().split("\n", 1)[0].strip()
        if len(first_line) < 150 and first_line and not first_line.endswith("."):
            meta["title"] = first_line

    parsed = urlparse(url)
    meta["site"] = parsed.netloc.removeprefix("www.")
    meta["url"] = url
    return meta


# Protect preformatted blocks before an extractor normalizes the surrounding DOM. Return the
# (possibly rewritten) HTML and the markers for restore_pre_markers; the HTML is returned
# unchanged with no markers when bs4 is missing or the page has no <pre>.
def _protect_pre_blocks(source_html: str) -> tuple[str, list[tuple[str, str, str]]]:
    if not (_HAS_BS4 and "<pre" in source_html.lower()):
        return source_html, []

    from core.extract.markdown_code import wrap_pre_with_markers

    soup = BeautifulSoup(source_html, "lxml")
    markers = wrap_pre_with_markers(soup)
    return str(soup), markers


# Use trafilatura with include_formatting=True for markdown-like output. Recall mode is
# useful for already-isolated article HTML returned by structured APIs: there is no site
# chrome to reject, so retaining the whole article is safer than main-content guessing.
def _extract_with_trafilatura_formatted(
    cleaned_html: str,
    url: str = "",
    *,
    favor_recall: bool = False,
) -> str:
    if not cleaned_html:
        return ""

    # Extract with the code blocks protected; a trafilatura failure yields no text, not an error.
    source_html, code_markers = _protect_pre_blocks(cleaned_html)
    try:
        text = trafilatura.extract(
            source_html,
            url=url or None,
            include_comments=False,
            include_tables=True,
            include_formatting=True,
            include_links=True,
            favor_precision=not favor_recall,
            favor_recall=favor_recall,
            deduplicate=False,
            output_format="markdown" if favor_recall else "txt",
        )
    except Exception:
        return ""
    from core.extract.markdown_code import restore_pre_markers

    # Put the protected code blocks back as exact fenced markdown.
    return restore_pre_markers(text or "", code_markers)


# DOM block extraction with structural nav/UI rejection. Return (joined_text, stats), or
# ("", {}) on any failure so the caller falls back to bs4/regex extraction.
def _extract_with_dom_blocks(cleaned_html: str, url: str) -> tuple[str, dict]:
    try:
        from core.extract.dom_block_extractor import extract_dom_blocks
        from core.extract.markdown_code import restore_pre_markers

        source_html, code_markers = _protect_pre_blocks(cleaned_html)
        blocks, stats = extract_dom_blocks(source_html, url=url)
        text = restore_pre_markers("\n\n".join(blocks), code_markers)
        return text, dict(stats)
    except Exception:  # noqa: BLE001 — extraction refinement must never sink a read
        return "", {}


# Extract and return the main textual content from HTML or the fallback text. The extractors are
# tried in order of fidelity, and the first one that yields at least 200 normalized characters
# wins: trafilatura (formatted), DOM blocks, bs4, regex. Any non-empty regex text is accepted,
# and the stripped fallback text is used when there is no HTML.
def _extract_content(raw_html: Optional[str], fallback_text: Optional[str], url: str) -> str:
    if raw_html:
        cleaned_html = _preclean_html(raw_html)
        min_chars = 200

        # Keep the formatted result whenever it is substantial. A DOM pass's own reject
        # count cannot tell us whether this different extractor retained boilerplate.
        if _HAS_TRAFILATURA:
            text = _extract_with_trafilatura_formatted(cleaned_html, url)
            if len(_normalize_text(text)) >= min_chars:
                return text

        # Structural nav/UI rejection remains the fallback for thin formatted output.
        dom_text, _dom_stats = _extract_with_dom_blocks(cleaned_html, url)
        dom_ok = len(_normalize_text(dom_text)) >= min_chars
        if dom_ok:
            return dom_text

        if _HAS_BS4:
            text = _extract_text_with_bs4(cleaned_html)
            if len(_normalize_text(text)) >= min_chars:
                return text

        text = _regex_html_to_text(cleaned_html or raw_html)
        if text.strip():
            return text

    if fallback_text:
        return fallback_text.strip()

    return ""


# Split text into blocks at blank lines, preserving markdown structure: structured blocks
# (heading, list, quote, code fence) are kept verbatim, while prose blocks get HTML entities
# unescaped and horizontal whitespace collapsed.
def _split_blocks_structured(text: str) -> list[str]:
    pieces = re.split(r"\n\s*\n", text or "")
    blocks: list[str] = []
    for piece in pieces:
        stripped = piece.strip()
        if not stripped:
            continue
        if (
            _MD_HEADING_RE.match(stripped)
            or _MD_LIST_RE.match(stripped)
            or _MD_QUOTE_RE.match(stripped)
            or _MD_CODE_FENCE_RE.match(stripped)
        ):
            blocks.append(stripped)
        else:
            normalized = _WHITESPACE_RE.sub(
                " ", html_lib.unescape(stripped).replace("\u00a0", " ")
            ).strip()
            if normalized:
                blocks.append(normalized)
    return blocks


# Return True when the block starts like a markdown heading, list item, quote or code fence.
def _is_structured_block(block: str) -> bool:
    return bool(
        _MD_HEADING_RE.match(block)
        or _MD_LIST_RE.match(block)
        or _MD_QUOTE_RE.match(block)
        or _MD_CODE_FENCE_RE.match(block)
    )


# Clean and deduplicate blocks while preserving markdown structure. Structured blocks always
# stay. Prose blocks are dropped by the shared boilerplate filter in strict mode, and by a short
# list of cookie, newsletter, advertising and legal phrases otherwise.
def _clean_content(text: str, strict: bool = True) -> str:
    is_boilerplate = _get_boilerplate_filter()
    blocks = _split_blocks_structured(text)

    filtered: list[str] = []
    for block in blocks:
        if _is_structured_block(block):
            filtered.append(block)
            continue
        if strict:
            if not is_boilerplate(block):
                filtered.append(block)
        else:
            # Lenient mode: only obvious boilerplate phrases are removed.
            compact = _normalize_text(block).lower()
            if not any(m in compact for m in (
                "cookie", "consent", "gdpr", "newsletter", "subscribe",
                "sign up", "advert", "sponsored", "all rights reserved",
                "accept all", "privacy policy", "terms of service",
            )):
                filtered.append(block)

    deduped = _dedupe_blocks(filtered)
    return "\n\n".join(deduped)


# Whitespace after . ! or ? that is followed by an uppercase Latin or Cyrillic letter or a
# double quote, i.e. a likely sentence boundary.
_SENTENCE_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-ZА-ЯЁ"])')


# Segment a wall-of-text into reasonable blocks: keep existing paragraphs, else group sentences
# four per paragraph, else (a single sentence) turn lines into headings and paragraphs.
def _fallback_segment(text: str) -> str:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(parts) > 1:
        return "\n\n".join(parts)

    sentences = _SENTENCE_END_RE.split(text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    # At most one sentence: segment by lines instead.
    if len(sentences) <= 1:
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        if len(lines) > 1:
            return _segment_lines(lines)
        return text.strip()

    # Group the sentences four per paragraph.
    paragraphs: list[str] = []
    group: list[str] = []
    for s in sentences:
        group.append(s)
        if len(group) >= 4:
            paragraphs.append(" ".join(group))
            group = []
    if group:
        paragraphs.append(" ".join(group))
    return "\n\n".join(paragraphs)


# Turn short non-sentence lines (under 80 characters, no closing punctuation) into markdown
# headings; structured and longer lines are kept as paragraphs.
def _segment_lines(lines: list[str]) -> str:
    result: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if _MD_HEADING_RE.match(stripped) or _MD_LIST_RE.match(stripped) or _MD_QUOTE_RE.match(stripped):
            result.append(stripped)
        elif len(stripped) < 80 and not stripped.endswith((".", "!", "?", ",", ";", ":")):
            result.append(f"## {stripped}")
        else:
            result.append(stripped)
    return "\n\n".join(result)


# Assemble the final markdown document: title, a meta header (site, URL, date, author), a rule
# and the content. Tables are repaired, blank lines collapsed outside code fences, and the result
# truncated to _MAX_OUTPUT_CHARS.
def _build_markdown(meta: dict[str, str], content: str) -> str:
    parts: list[str] = []

    # Header: the title as H1 and one bold line per available meta field.
    title = meta.get("title", "").strip() or "Untitled"
    parts.append(f"# {title}")
    parts.append("")

    if meta.get("site"):
        parts.append(f"**Site:** {meta['site']}")
    if meta.get("url"):
        parts.append(f"**URL:** {meta['url']}")
    if meta.get("date"):
        parts.append(f"**Date:** {meta['date']}")
    if meta.get("author"):
        parts.append(f"**Author:** {meta['author']}")

    parts.append("")
    parts.append("---")
    parts.append("")
    parts.append(content)

    text = "\n".join(parts)

    # Repair GFM tables (trafilatura drops the header row's leading `|`, etc.) so every
    # renderer parses them. No-op on table-free pages.
    from core.extract.markdown_tables import normalize_markdown_tables

    text = normalize_markdown_tables(text)

    # Collapse runs of blank lines without touching the inside of code fences.
    from core.extract.markdown_code import collapse_blank_lines_preserving_fences

    text = collapse_blank_lines_preserving_fences(text)

    # Enforce the output size cap.
    if len(text) > _MAX_OUTPUT_CHARS:
        from core.extract.content_processor import _truncate_markdown_to_budget

        text = _truncate_markdown_to_budget(text, _MAX_OUTPUT_CHARS)

    return text


# Clean extraction below this many normalized characters is "too thin" — worth comparing against
# the full body.
_MIN_CLEAN_CHARS = 250


# Produce a clean, structured markdown representation of a web page: a meta header plus the
# extracted content, or a "no content" marker when nothing could be extracted.
def normalize_page(
    url: str,
    raw_html: Optional[str] = None,
    fallback_text: Optional[str] = None,
    *,
    favor_recall: bool = False,
) -> str:
    meta = _extract_meta(raw_html, fallback_text, url)

    # Trafilatura's formatted precision pass is both faster and more faithful on normal
    # articles and reference pages than flattening the same document into DOM blocks and
    # rebuilding structure afterwards. Keep the DOM/full-body machinery as the fallback
    # for genuinely thin extraction instead of running every successful page through it.
    if raw_html and _HAS_TRAFILATURA:
        formatted = _extract_with_trafilatura_formatted(raw_html, url, favor_recall=favor_recall)
        if len(_normalize_text(formatted)) >= 200:
            return _build_markdown(meta, formatted)

    # Slow path: extract with the fallback chain, then clean it (strict boilerplate filtering
    # only when the content came from HTML).
    content = _extract_content(raw_html, fallback_text, url)

    cleaned = ""
    if content.strip():
        strict = bool(raw_html)
        cleaned = _clean_content(content, strict=strict)

        # A long result without paragraph structure is re-segmented and cleaned again.
        block_count = len([b for b in cleaned.split("\n\n") if b.strip()])
        if block_count <= 1 and len(cleaned) > 300:
            cleaned = _fallback_segment(content)
            cleaned = _clean_content(cleaned, strict=strict)

    # Thin to full-body rescue: the clean pass strips everything it deems
    # boilerplate, which guts landing pages, doc indexes and download pages where the
    # "chrome" is the actual information. When the cleaned result is near-empty but the
    # page itself carried more visible text, return the whole readable body instead of a
    # husk — metadata keeps the clean pass's title/date. Bypasses _clean_content: block
    # filters would re-delete the short nav/CTA lines this path exists to preserve.
    if raw_html and len(_normalize_text(cleaned)) < _MIN_CLEAN_CHARS:
        from core.extract.content_processor import extract_full_body_text

        full = extract_full_body_text(raw_html)
        if len(_normalize_text(full)) > len(_normalize_text(cleaned)):
            return _build_markdown(meta, full)

    # Nothing usable: emit the metadata with the no-content marker.
    if not cleaned.strip():
        return _build_markdown(meta, _NO_CONTENT_MARKER)
    return _build_markdown(meta, cleaned)
