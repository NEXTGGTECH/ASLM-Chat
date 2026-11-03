# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import html
import re


# Strip tags from a small HTML fragment and unescape entities.
def strip_html_fragment(text: str) -> str:
    clean = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    clean = re.sub(r"</p\s*>", "\n\n", clean, flags=re.IGNORECASE)
    clean = re.sub(r"<[^>]+>", "", clean)
    return html.unescape(clean).strip()


# Format a price string as grouped integer digits (space thousands separator).
def format_price_value(value: str) -> str:
    digits = re.sub(r"[^\d]", "", value or "")
    if not digits:
        return ""
    return f"{int(digits):,}".replace(",", " ")


# Normalize schema.org or free-text availability labels for display.
def normalize_availability(value: str) -> str:
    text = strip_html_fragment(value or "")
    text = re.sub(r"\s+", " ", text).strip(" \t\r\n:;,.-")
    if not text:
        return ""

    # schema.org enum values (e.g. ".../InStock"), compared with everything but letters removed.
    lower = text.lower()
    mapping = {
        "instock": "In stock",
        "outofstock": "Out of stock",
        "preorder": "Preorder",
        "limitedavailability": "Limited availability",
    }
    compact = re.sub(r"[^a-z]", "", lower)
    if compact in mapping:
        return mapping[compact]

    # Known human-readable labels, returned in their canonical spelling.
    allowed_exact = {
        "в наличии": "В наличии",
        "под заказ": "Под заказ",
        "нет в наличии": "Нет в наличии",
        "in stock": "In stock",
        "out of stock": "Out of stock",
    }
    if lower in allowed_exact:
        return allowed_exact[lower]

    # Anything else is accepted only when it looks like a short human label; long text, markup/code
    # characters and identifier-like words are page noise and are dropped.
    if len(text) > 40:
        return ""
    if any(ch in text for ch in ('"', "{", "}", "=", ";")):
        return ""
    if re.search(r"[A-Za-z_]{6,}", text) and "stock" not in lower:
        return ""

    return text


# Prepend structured retail fields (price, availability, specs) above markdown body.
def prepend_retail_metadata(markdown: str, meta: dict[str, str]) -> str:
    if not markdown or not meta:
        return markdown

    # Read the metadata fields; empty ones produce no line.
    lines: list[str] = []
    price = meta.get("price", "").strip()
    currency = meta.get("currency", "").strip().upper()
    availability = meta.get("availability", "").strip()
    title = meta.get("title", "").strip()
    summary = meta.get("hero_summary", "").strip()
    specs = meta.get("key_specs", "").strip()

    # Product, price (with currency), availability, summary and key specs, in that order.
    if title:
        lines.append(f"**Product:** {title}")

    if price:
        if currency == "RUB":
            lines.append(f"**Price:** {price} RUB")
        elif currency:
            lines.append(f"**Price:** {price} {currency}")
        else:
            lines.append(f"**Price:** {price}")
    if availability:
        lines.append(f"**Availability:** {availability}")
    if summary:
        lines.append(f"**Summary:** {summary}")
    if specs:
        lines.append(f"**Key specs:** {specs}")

    if not lines:
        return markdown

    return "\n".join(lines) + "\n\n" + markdown
