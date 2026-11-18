# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any
from urllib.parse import urlsplit

from core.search.nltk_languages import catalog_query, shopping_intent

logger = logging.getLogger("mcp.web_search.products")

# Product card: the ASLM-Search shopping preview in light form. When a web query reads as
# a product search (an explicit commerce word in any inflection, or a model-number
# shape), the shopping engine runs on the product terms alongside the web search,
# offers are filtered by query relevance, and the result is returned as JSON in the
# same shape as ASLM-Search's /api/shopping-preview/ plus a short block for the model.
# Fewer than two relevant priced offers -> not eligible, nothing is shown.

MIN_ELIGIBLE_PRODUCTS = 2


# Return the lowercase host of a URL without a leading "www.", or the fallback when the URL cannot be
# parsed.
def _domain(url: str, fallback: str = "") -> str:
    try:
        return (urlsplit(url).hostname or fallback or "").lower().removeprefix("www.")
    except ValueError:
        return fallback


# Convert one catalog product into the JSON shape of the ASLM-Search shopping preview (title, url, domain, prices,
# seller, rating, image).
def _product_json(product: Any) -> dict[str, Any]:
    domain = _domain(product.url, product.source_domain)
    return {
        "id": product.id,
        "title": " ".join(str(product.title or "").split()),
        "url": product.url,
        "domain": domain,
        "source": product.source,
        "price_text": product.price_text,
        "price_value": product.price_value,
        "currency": product.currency,
        "converted_prices": list(product.converted_prices or []),
        "seller": product.seller,
        "availability": product.availability,
        "rating": product.rating,
        "review_count": product.review_count,
        "image_url": str(getattr(product, "image_url", "") or product.meta.get("image_url") or ""),
        "favicon_url": product.favicon_url,
        "snippet": product.snippet,
        "relevance": product.meta.get("query_relevance"),
    }


# Build the shopping_preview payload with defaults for every field; `fields` override them and the elapsed
# time is measured from `started`.
def _preview(query: str, *, started: float, **fields: Any) -> dict[str, Any]:
    return {
        "query": query,
        "catalog_query": fields.pop("catalog_query", ""),
        "eligible": False,
        "products": [],
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
        "partial": False,
        "partial_reason": "",
        **fields,
    }


# Shopping preview for one query (never raises).
async def shopping_preview(
    query: str, *, language: str = "en", limit: int = 6, timeout_seconds: float = 5.0
) -> dict[str, Any]:
    started = time.perf_counter()
    normalized = " ".join(str(query or "").split()).strip()
    if not normalized or not shopping_intent(normalized, language=language):
        return _preview(normalized, started=started, partial_reason="not_shopping_intent")
    catalog = catalog_query(normalized, language=language)
    try:
        from core.fetch.shopping import search_shopping
        from core.fetch.shopping.relevance import dedupe_products, rank_products

        result = await asyncio.wait_for(
            search_shopping(catalog, effort="low", limit=limit + 4, language=language),
            timeout=timeout_seconds,
        )
        ranked = dedupe_products(rank_products(list(result.products), normalized, language=language))
    except (TimeoutError, asyncio.TimeoutError):
        return _preview(normalized, started=started, catalog_query=catalog, partial=True, partial_reason="timeout")
    except Exception as exc:  # noqa: BLE001 - the card is optional; the search stands
        logger.warning("shopping preview failed for %r: %s", normalized[:120], exc)
        return _preview(normalized, started=started, catalog_query=catalog, partial=True, partial_reason="error")

    products = [
        _product_json(product)
        for product in ranked
        if product.title and product.url and (product.price_text or product.price_value is not None)
    ][:limit]
    eligible = len(products) >= MIN_ELIGIBLE_PRODUCTS
    reason = ""
    if not eligible:
        reason = "no_relevant_products" if result.products else (result.partial_reason or "no_products")
    logger.info(
        "shopping preview query=%r catalog=%r raw=%d relevant=%d eligible=%s",
        normalized[:120], catalog[:120], len(result.products), len(products), eligible,
    )
    return _preview(
        normalized,
        started=started,
        catalog_query=catalog,
        eligible=eligible,
        products=products if eligible else [],
        partial=bool(result.partial) and not eligible,
        partial_reason=reason,
    )


# Format a product's price as one line: its own price text (or value and currency) plus converted prices.
def _price_line(product: dict[str, Any]) -> str:
    price = str(product.get("price_text") or "").strip()
    if not price and product.get("price_value") is not None:
        price = f"{product['price_value']:g} {product.get('currency') or ''}".strip()
    converted = " / ".join(
        str(item.get("price_text") or "").strip()
        for item in (product.get("converted_prices") or [])[:2]
        if isinstance(item, dict) and str(item.get("price_text") or "").strip()
    )
    return f"{price} (≈ {converted})" if price and converted else price


# Compact block for the model context (empty when the preview is not eligible).
def render_products(preview: dict[str, Any]) -> str:
    if not preview.get("eligible"):
        return ""
    lines = [
        "Product offers (live shopping preview: the query looks like a product search; "
        "prices and stock change - name the store and give the link when quoting an offer):"
    ]
    for number, product in enumerate(preview.get("products") or [], 1):
        details = [
            part
            for part in (
                _price_line(product),
                str(product.get("seller") or product.get("domain") or ""),
                (f"rating {product['rating']}" + (f" ({product['review_count']})" if product.get("review_count") else ""))
                if product.get("rating") else "",
                str(product.get("availability") or ""),
            )
            if part
        ]
        lines.append(f"  {number}. {product.get('title')} — " + " · ".join(details))
        lines.append(f"     {product.get('url')}")
    return "\n".join(lines)


# Answer-card entry for result["answer_cards"].
def products_card(preview: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "products",
        "provider": "Shopping",
        "title": preview.get("catalog_query") or preview.get("query"),
        "products": list(preview.get("products") or []),
    }


__all__ = ["products_card", "render_products", "shopping_preview"]
