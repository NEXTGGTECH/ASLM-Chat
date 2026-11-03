# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import re

from custom_domains.retail_common import format_price_value, normalize_availability, strip_html_fragment


# Extract product metadata from Citilink product HTML (JSON-LD and inline fields).
def extract_citilink_metadata(raw_html: str) -> dict[str, str]:
    meta: dict[str, str] = {}

    # JSON-LD blocks: take the Product name/description and the offer price, currency and availability.
    for match in re.finditer(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        raw_html,
        re.IGNORECASE | re.DOTALL,
    ):
        payload = (match.group(1) or "").strip()
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except Exception:
            # A malformed JSON-LD block is skipped; other blocks may still be valid.
            continue
        if not isinstance(data, dict):
            continue
        if data.get("@type") == "Product":
            title = str(data.get("name") or "").strip()
            description = str(data.get("description") or "").strip()
            if title:
                meta["title"] = title[:180]
            if description:
                meta["hero_summary"] = strip_html_fragment(description)[:280]
        offers = data.get("offers") if isinstance(data.get("offers"), dict) else {}
        price = format_price_value(str(offers.get("price") or ""))
        currency = str(offers.get("priceCurrency") or "").strip().upper()
        availability = str(offers.get("availability") or "").strip()
        if price:
            meta["price"] = price
        if currency:
            meta["currency"] = currency
        if availability:
            tail = availability.rsplit("/", 1)[-1]
            normalized = normalize_availability(tail)
            if normalized:
                meta["availability"] = normalized

    # Fallback to the inline page state when JSON-LD gave no price (Citilink prices are in RUB).
    price_match = re.search(r'"price"\s*:\s*\{\s*"current"\s*:\s*"([^"]+)"', raw_html, re.IGNORECASE)
    if price_match and "price" not in meta:
        meta["price"] = format_price_value(price_match.group(1))
        meta.setdefault("currency", "RUB")

    # Same fallback for availability.
    availability_match = re.search(r'"availability"\s*:\s*"([^"]+)"', raw_html, re.IGNORECASE)
    if availability_match and "availability" not in meta:
        normalized = normalize_availability(availability_match.group(1).rsplit("/", 1)[-1])
        if normalized:
            meta["availability"] = normalized

    # Collect up to six distinct spec names from the inline specs/characteristics arrays.
    specs: list[str] = []
    for match in re.finditer(
        r'"(?:specs?|characteristics?)"\s*:\s*\[(.*?)\]',
        raw_html,
        re.IGNORECASE | re.DOTALL,
    ):
        chunk = match.group(1)
        specs.extend(re.findall(r'"name"\s*:\s*"([^"]+)"', chunk, re.IGNORECASE))
        if len(specs) >= 6:
            break
    if specs:
        meta["key_specs"] = ", ".join(dict.fromkeys(specs))[:300]

    return meta
