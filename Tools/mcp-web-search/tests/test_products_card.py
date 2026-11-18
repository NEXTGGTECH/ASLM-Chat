# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

import pytest

from core.cards import products as products_mod
from core.cards.products import render_products, shopping_preview
from core.fetch.shopping.models import ShoppingProduct, ShoppingSearchResult
from core.fetch.shopping.relevance import is_product_page_url, product_relevance, rank_products
from core.search.nltk_languages import catalog_query, commerce_intent_score, shopping_intent

pytestmark = pytest.mark.unit


# Build a catalog product for the tests.
def _product(title: str, url: str, price: str = "1 000 ₽", value: float | None = 1000.0, **kw) -> ShoppingProduct:
    return ShoppingProduct(
        id=url, title=title, url=url, source=kw.pop("source", "test"), source_domain="shop.test",
        lane="primary", price_text=price, price_value=value, currency="RUB", **kw,
    )


# Build a shopping search result around the given products.
def _result(products: list[ShoppingProduct], **kw) -> ShoppingSearchResult:
    return ShoppingSearchResult(
        query="q", effort="low", primary_ratio=0.75, secondary_ratio=0.25,
        products=products, attempts=[], **kw,
    )



# Verify that product like queries have shopping intent.
@pytest.mark.parametrize("query", [
    "купить rtx 4070 super",
    "куплю ноутбук",            # inflected commerce word
    "iphone 15 цена",
    "наушники sony wh-1000xm5",
    "iphone 15 128gb price",
    "best price for a dishwasher",
    "rtx 4090",
    "4006381333931",            # EAN
])
def test_product_like_queries_have_shopping_intent(query):
    assert shopping_intent(query)


# Verify that ordinary queries have no shopping intent.
@pytest.mark.parametrize("query", [
    "высота эльбруса",
    "python release notes",
    "ключевая ставка цб",
    "how does photosynthesis work",
])
def test_ordinary_queries_have_no_shopping_intent(query):
    assert not shopping_intent(query)


# Verify that commerce words match in any inflection.
def test_commerce_words_match_in_any_inflection():
    assert commerce_intent_score("заказал велосипед", language="ru") == 1.0
    assert commerce_intent_score("buying a bike", language="en") == 1.0
    assert commerce_intent_score("велосипед для города", language="ru") == 0.0


# Verify that catalog query drops commerce words.
def test_catalog_query_drops_commerce_words():
    assert catalog_query("купить rtx 4070 super", language="ru") == "rtx 4070 super"
    assert catalog_query("цены на iphone 15", language="ru") == "на iphone 15"
    assert catalog_query("buy cheap usb-c hub", language="en") == "usb-c hub"
    assert catalog_query("купить", language="ru") == "купить"  # nothing left -> original



# Verify that model numbers must match exactly or with a listing suffix.
def test_model_numbers_must_match_exactly_or_with_a_listing_suffix():
    assert product_relevance(_product("AMD Ryzen 5 5600X", "https://s.test/p/1"), "ryzen 5600") is None
    assert product_relevance(_product("AMD Ryzen 5 5600", "https://s.test/p/1"), "ryzen 5600") is not None
    assert product_relevance(_product("Kit NYXL1046-3P", "https://s.test/p/2"), "nyxl1046") is not None


# Verify that word coverage uses stems.
def test_word_coverage_uses_stems():
    good = _product("Беспроводные наушники Sony WH-1000XM5", "https://s.test/p/1")
    bad = _product("Чехол для телефона", "https://s.test/p/2")
    assert product_relevance(good, "купить беспроводных наушников sony wh-1000xm5", language="ru") is not None
    assert product_relevance(bad, "беспроводные наушники", language="ru") is None


# Verify that product page urls.
@pytest.mark.parametrize("url,ok", [
    ("https://shop.test/product/123", True),
    ("https://www.amazon.com/dp/B0C1", True),
    ("https://shop.test/catalog/headphones-sony", True),
    ("https://shop.test/search?q=sony", False),
    ("https://shop.test/category/audio", False),
    ("https://news.shop.test/item/1", False),
    ("https://www.google.com/shopping/product/1", False),
])
def test_product_page_urls(url, ok):
    assert is_product_page_url(url) is ok


# Verify that rank drops irrelevant and non product pages.
def test_rank_drops_irrelevant_and_non_product_pages():
    products = [
        _product("RTX 4070 Super Founders", "https://a.test/product/1"),
        _product("RTX 4070 Super OC", "https://b.test/search?q=rtx"),
        _product("RTX 3060", "https://c.test/product/3"),
        _product("ASUS RTX 4070 SUPER Dual", "https://d.test/product/4", price="", value=None),
    ]
    ranked = rank_products(products, "rtx 4070 super", language="en")
    assert [p.url for p in ranked] == ["https://a.test/product/1", "https://d.test/product/4"]



# Replace the shopping search with a fake that returns the products (or raises the exception) and records
# the queries.
def _patch_engine(monkeypatch, products: list[ShoppingProduct] | Exception, calls: list[str] | None = None):
    async def fake_search_shopping(query, *, effort="medium", limit=12, language="en"):
        if calls is not None:
            calls.append(query)
        if isinstance(products, Exception):
            raise products
        return _result(list(products))

    import core.fetch.shopping as shopping_pkg

    monkeypatch.setattr(shopping_pkg, "search_shopping", fake_search_shopping)


# Verify that preview returns aslm search shaped json.
def test_preview_returns_aslm_search_shaped_json(monkeypatch):
    calls: list[str] = []
    _patch_engine(monkeypatch, [
        _product("Sony WH-1000XM5 Black", "https://a.test/product/1", rating="4.8", review_count=120),
        _product("Sony WH-1000XM5 Silver", "https://b.test/product/2", seller="B Store"),
        _product("Sony WH-1000XM4", "https://c.test/product/3"),
    ], calls)

    preview = asyncio.run(shopping_preview("купить наушники sony wh-1000xm5", language="ru"))

    assert calls == ["наушники sony wh-1000xm5"]
    assert preview["eligible"] is True
    assert set(preview) >= {"query", "eligible", "products", "elapsed_ms", "partial", "partial_reason"}
    assert [p["url"] for p in preview["products"]] == ["https://a.test/product/1", "https://b.test/product/2"]
    first = preview["products"][0]
    assert first["price_text"] == "1 000 ₽" and first["domain"] == "a.test" and first["relevance"]

    text = render_products(preview)
    assert text.startswith("Product offers")
    assert "1. Sony WH-1000XM5 Black — 1 000 ₽ · a.test · rating 4.8 (120)" in text
    assert "https://b.test/product/2" in text


# Verify that preview needs two relevant offers.
def test_preview_needs_two_relevant_offers(monkeypatch):
    _patch_engine(monkeypatch, [_product("Sony WH-1000XM5", "https://a.test/product/1")])
    preview = asyncio.run(shopping_preview("sony wh-1000xm5 price", language="en"))
    assert preview["eligible"] is False and preview["products"] == []
    assert preview["partial_reason"] == "no_relevant_products"
    assert render_products(preview) == ""


# Verify that preview skips the engine without intent.
def test_preview_skips_the_engine_without_intent(monkeypatch):
    calls: list[str] = []
    _patch_engine(monkeypatch, [], calls)
    preview = asyncio.run(shopping_preview("высота эльбруса", language="ru"))
    assert calls == [] and preview["partial_reason"] == "not_shopping_intent"


# Verify that preview fails open.
def test_preview_fails_open(monkeypatch):
    _patch_engine(monkeypatch, RuntimeError("boom"))
    preview = asyncio.run(shopping_preview("rtx 4090 price", language="en"))
    assert preview["eligible"] is False and preview["partial_reason"] == "error"

    async def slow(query, **kwargs):
        await asyncio.sleep(5)

    import core.fetch.shopping as shopping_pkg

    monkeypatch.setattr(shopping_pkg, "search_shopping", slow)
    preview = asyncio.run(shopping_preview("rtx 4090 price", language="en", timeout_seconds=0.05))
    assert preview["partial_reason"] == "timeout"



# Load mcp-server.py as a module for the tests.
def _server():
    spec = importlib.util.spec_from_file_location(
        "mcp_server_products_test", Path(__file__).resolve().parents[1] / "mcp-server.py"
    )
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    return server


# Replace the web search of the server with a fake returning a minimal result.
def _patch_search(monkeypatch, server):
    async def fake_search(*args, **kwargs):
        return {
            "query": "q",
            "sources": [],
            "model_context": "Search results for: q\n\nSources:\nCitation handle: [c1]",
        }

    for name in ("run_web_search", "run_web_search_plan", "run_web_search_batch"):
        monkeypatch.setattr(server, name, fake_search)
    monkeypatch.setattr(server, "write_search_io_event", lambda event: None)


# Verify that web search attaches product card for product queries.
def test_web_search_attaches_product_card_for_product_queries(monkeypatch):
    server = _server()
    _patch_search(monkeypatch, server)
    seen: list[str] = []

    async def fake_preview(query, **kwargs):
        seen.append(query)
        return {
            "query": query, "catalog_query": "rtx 4070", "eligible": True, "elapsed_ms": 1.0,
            "partial": False, "partial_reason": "",
            "products": [
                {"title": "RTX 4070 A", "url": "https://a.test/product/1", "price_text": "$599", "domain": "a.test"},
                {"title": "RTX 4070 B", "url": "https://b.test/product/2", "price_text": "$579", "domain": "b.test"},
            ],
        }

    monkeypatch.setattr(products_mod, "shopping_preview", fake_preview)
    result = asyncio.run(server._call_web_search({"query": "купить rtx 4070"}))

    assert seen == ["купить rtx 4070"]
    assert result["shopping_preview"]["eligible"] is True
    assert result["answer_cards"][-1]["kind"] == "products"
    context = result["model_context"]
    assert context.index("Product offers") < context.index("Sources:")
    assert "https://b.test/product/2" in context


# Verify that web search skips product card for ordinary queries.
def test_web_search_skips_product_card_for_ordinary_queries(monkeypatch):
    server = _server()
    _patch_search(monkeypatch, server)
    seen: list[str] = []

    async def fake_preview(query, **kwargs):
        seen.append(query)
        return {}

    monkeypatch.setattr(products_mod, "shopping_preview", fake_preview)
    result = asyncio.run(server._call_web_search({"query": "высота эльбруса"}))

    assert seen == []
    assert "shopping_preview" not in result and "Product offers" not in result["model_context"]
