# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from typing import Any

import orjson
from selectolax.lexbor import LexborHTMLParser

# Below this DOM word count the page is "sparse" — worth trying data islands. Set high
# enough to cover marketing homepages with partial SSR (Notion SSR-renders ~300 words but
# carries ~800 in __NEXT_DATA__), matching webclaw's SPARSE_THRESHOLD.
_SPARSE_THRESHOLD = 500

# Cap total chunks so an adversarial payload can't blow up memory/CPU.
_MAX_CHUNKS = 1000

# Recursion depth guard for deeply nested JSON.
_MAX_DEPTH = 15

# JSON field names that carry a heading, and those that carry body text, for objects that are
# not a full CMS entry.
_HEADING_KEYS = ("heading", "title", "headline")
_BODY_KEYS = ("body", "description", "subheading", "eyebrow", "children")


# One recovered piece of text: an optional heading and a (possibly empty) body.
class _Chunk:
    __slots__ = ("heading", "body")

    # Store the heading (None when absent) and the body text.
    def __init__(self, heading: str | None, body: str) -> None:
        self.heading = heading
        self.body = body


# Recover text from JSON data islands embedded in <script type="application/json">. Modern SPAs
# (React, Next.js, Nuxt, Contentful-backed marketing sites) ship the page's real content as JSON
# in script tags and hydrate the visible DOM client-side, so a static fetch extracts almost
# nothing. This walks those JSON blobs as a FALLBACK, only when DOM extraction came back sparse.
# Return markdown of the genuinely-new text (deduped against `existing_markdown`), or None when
# the DOM already had enough content, no island parsed, or nothing survived dedup.
# Ported from webclaw-core's data_island.rs: the recognised shapes (Contentful rich-text nodes,
# CMS heading+body entries, quotes, orphan content fields, stat-string arrays) and the
# content-vs-identifier heuristic were tuned against real SPA payloads, so keep them as they are.
def try_extract_data_islands(
    html: str, dom_word_count: int, existing_markdown: str
) -> str | None:
    if dom_word_count >= _SPARSE_THRESHOLD:
        return None

    # Parse every JSON script block of useful size and collect text chunks from it. Invalid
    # JSON is skipped, and the total chunk count is capped.
    tree = LexborHTMLParser(html or "")
    chunks: list[_Chunk] = []
    for script in tree.css("script[type='application/json']"):
        if len(chunks) >= _MAX_CHUNKS:
            break
        json_text = script.text(deep=True) or ""
        if len(json_text) < 50:
            continue
        try:
            value = orjson.loads(json_text)
        except orjson.JSONDecodeError:
            # Not every application/json script is valid JSON (templates, partial payloads).
            continue
        _walk_json(value, chunks, 0)

    if not chunks:
        return None
    del chunks[_MAX_CHUNKS:]

    # Dedup: drop chunks whose key text already appears in the DOM markdown or repeats a
    # prior chunk. The body is the identity when present, else the heading.
    existing_lower = (existing_markdown or "").lower()
    seen: set[str] = set()
    kept: list[_Chunk] = []
    for c in chunks:
        key = c.body if c.body else (c.heading or "")
        if not key:
            continue
        key_lower = key.lower()
        if key_lower in seen or key_lower in existing_lower:
            continue
        seen.add(key_lower)
        kept.append(c)

    if not kept:
        return None

    # Render the surviving chunks as markdown: headings as "##", bodies as paragraphs.
    parts: list[str] = []
    for c in kept:
        if c.heading:
            parts.append(f"## {c.heading}")
        if c.body:
            parts.append(c.body)
    md = "\n\n".join(parts).strip()
    return md or None


# Recursively walk a JSON value, appending recognised text chunks. Object shapes are tried
# most-specific first (Contentful node → CMS entry → quote → orphan fields) and only then do
# we recurse into children; arrays check for a stat-string list before recursing.
def _walk_json(value: Any, chunks: list[_Chunk], depth: int) -> None:
    if depth > _MAX_DEPTH or len(chunks) >= _MAX_CHUNKS:
        return

    if isinstance(value, dict):
        node_type = value.get("nodeType")
        if isinstance(node_type, str):
            chunk = _extract_contentful_node(value, node_type)
            if chunk is not None:
                chunks.append(chunk)
                return

        if _is_cms_entry(value):
            chunk = _extract_cms_entry(value)
            if chunk is not None:
                chunks.append(chunk)
                return

        quote = _extract_quote(value)
        if quote is not None:
            chunks.append(quote)
            return

        # Orphan content fields (a lone body/heading) before recursing — the pattern
        # matchers above won't catch them.
        _extract_orphan_texts(value, chunks)

        for key, child in value.items():
            if _is_media_key(key):
                continue
            _walk_json(child, chunks, depth + 1)

    elif isinstance(value, list):
        # Stat-style string array, e.g. ["100M+ users", "#1 rated developer platform"].
        content_strings = [
            s for s in value if isinstance(s, str) and len(s) > 10 and " " in s
        ]
        if len(content_strings) >= 2:
            chunks.append(_Chunk(None, " | ".join(content_strings)))
            return
        for child in value:
            _walk_json(child, chunks, depth + 1)


# Extract text from a Contentful rich-text node (document / paragraph / heading-N /
# blockquote). Return None for shapes that carry no usable prose.
def _extract_contentful_node(node: dict, node_type: str) -> _Chunk | None:
    # A document is rendered by recursing into its children and joining their chunks.
    if node_type == "document":
        content = node.get("content")
        if not isinstance(content, list):
            return None
        parts: list[str] = []
        for child in content:
            if not isinstance(child, dict):
                continue
            child_type = child.get("nodeType")
            if not isinstance(child_type, str):
                continue
            chunk = _extract_contentful_node(child, child_type)
            if chunk is None:
                continue
            if chunk.heading:
                parts.append(f"## {chunk.heading}")
            if chunk.body:
                parts.append(chunk.body)
        return _Chunk(None, "\n\n".join(parts)) if parts else None

    # Leaf shapes: paragraphs and quotes must look like prose, headings are kept as they are.
    if node_type in ("paragraph", "text"):
        text = _collect_text_content(node)
        return _Chunk(None, text) if _is_content_text(text) else None

    if node_type.startswith("heading-"):
        text = _collect_text_content(node)
        return _Chunk(text, "") if text else None

    if node_type == "blockquote":
        text = _collect_text_content(node)
        return _Chunk(None, f"> {text}") if _is_content_text(text) else None

    return None


# Recursively gather plain text from a Contentful rich-text node tree (value + content[]).
def _collect_text_content(node: dict) -> str:
    text = ""
    value = node.get("value")
    if isinstance(value, str):
        text += value
    content = node.get("content")
    if isinstance(content, list):
        for child in content:
            if isinstance(child, dict):
                text += _collect_text_content(child)
    return text.strip()


# Return True when an object looks like a CMS entry: a heading-ish field paired with a body-ish one.
def _is_cms_entry(node: dict) -> bool:
    has_heading = any(k in node for k in ("heading", "title", "headline"))
    has_body = any(k in node for k in ("description", "subheading", "body", "text"))
    return has_heading and has_body


# Extract heading + body from a CMS-style entry, or None when the heading is missing, too short or
# an internal CMS label, or neither side is real content.
def _extract_cms_entry(node: dict) -> _Chunk | None:
    heading = (
        _extract_text_field(node, "heading")
        or _extract_text_field(node, "title")
        or _extract_text_field(node, "headline")
    )
    if not heading or _is_cms_internal_title(heading) or len(heading) <= 5:
        return None

    body = (
        _extract_text_field(node, "description")
        or _extract_text_field(node, "subheading")
        or _extract_text_field(node, "body")
        or _extract_text_field(node, "text")
        or ""
    )
    if not _is_content_text(heading) and not _is_content_text(body):
        return None
    return _Chunk(heading, body)


# Extract a quote/testimonial (quote + optional attribution), or None when not a quote.
def _extract_quote(node: dict) -> _Chunk | None:
    quote = _extract_text_field(node, "quote") or _extract_text_field(node, "quoteText")
    if not quote or not _is_content_text(quote):
        return None
    attribution = (
        _extract_text_field(node, "position")
        or _extract_text_field(node, "author")
        or _extract_text_field(node, "name")
        or ""
    )
    body = f"> {quote}" if not attribution else f"> {quote}\n> — {attribution}"
    return _Chunk(None, body)


# Append a standalone heading or body field from an object the pattern matchers skipped.
def _extract_orphan_texts(node: dict, chunks: list[_Chunk]) -> None:
    if _is_cms_entry(node):
        return
    for key in _HEADING_KEYS:
        text = _extract_text_field(node, key)
        if text and _is_content_text(text):
            chunks.append(_Chunk(text, ""))
            return
    for key in _BODY_KEYS:
        text = _extract_text_field(node, key)
        if text and _is_content_text(text):
            chunks.append(_Chunk(None, text))
            return


# Read a text field as either a plain string or a Contentful rich-text object. Return the stripped
# text, or None when the field is missing or empty.
def _extract_text_field(node: dict, key: str) -> str | None:
    value = node.get(key)
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, dict):
        text = _collect_text_content(value)
        return text or None
    return None


# JSON keys holding image/media/asset data — don't recurse in, or CMS alt text leaks as prose.
def _is_media_key(key: str) -> bool:
    k = key.lower()
    return (
        k in ("alt", "src", "url", "href")
        or "image" in k
        or "poster" in k
        or "video" in k
        or "thumbnail" in k
        or "icon" in k
        or "logo" in k
    )


# Return True for editorial/asset labels ("/home Customer Stories: Logo", "hero poster desktop")
# that are internal CMS titles, not user-facing text.
def _is_cms_internal_title(s: str) -> bool:
    if s.startswith("/home ") or s.startswith("/page "):
        return True
    words = s.split()
    if len(words) >= 3:
        labels = {"poster", "logo", "image", "icon", "asset", "thumbnail"}
        if any(w in labels for w in words):
            return True
    return False


# Heuristic: return True when the string is real prose, not an id/URL/class-name/hash. Prose is at
# least 15 characters, does not start like a URL, path or JSON, contains a space and is at
# least 60% alphanumeric.
def _is_content_text(s: str) -> bool:
    s = (s or "").strip()
    if len(s) < 15:
        return False
    if s.startswith(("http", "/", "{", "[")):
        return False

    # Prose has spaces; a lone token is technical.
    if " " not in s:
        return False
    alnum = sum(1 for c in s if c.isalnum())
    return alnum / len(s) >= 0.6
