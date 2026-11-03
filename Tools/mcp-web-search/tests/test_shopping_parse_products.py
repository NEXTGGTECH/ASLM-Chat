# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json

import pytest

from core.fetch.shopping.parse import parse_products


# Parse an HTML page as a pricerunner primary-lane page with GBP as the default currency.
def _parse(html: str):
    return parse_products(
        html,
        provider="pricerunner",
        lane="primary",
        method="httpx",
        base_url="https://www.pricerunner.com/results?q=test",
        default_currency="GBP",
    )


# Verify that a JSON-LD Product with a valid offer price is kept, with its price value and currency.
@pytest.mark.unit
def test_parse_products_keeps_jsonld_product_with_valid_price() -> None:
    payload = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": "Stable Product",
        "url": "https://example.com/p/1",
        "offers": {
            "@type": "Offer",
            "price": "123.45",
            "priceCurrency": "GBP",
        },
    }

    products = _parse(f'<script type="application/ld+json">{json.dumps(payload)}</script>')

    assert len(products) == 1
    assert products[0].price_value == pytest.approx(123.45)
    assert products[0].currency == "GBP"


# Verify that a JSON-LD Product whose offer has no price is dropped.
@pytest.mark.unit
def test_parse_products_drops_jsonld_product_without_valid_price() -> None:
    payload = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": "No Price Product",
        "url": "https://example.com/p/1",
        "offers": {"priceCurrency": "GBP"},
    }

    products = _parse(f'<script type="application/ld+json">{json.dumps(payload)}</script>')

    assert products == []


# Verify that when a JSON-LD Product lists several offers, the price of the first offer is used.
@pytest.mark.unit
def test_parse_products_uses_first_jsonld_offer_list_item() -> None:
    payload = {
        "@context": "https://schema.org",
        "@type": "Product",
        "name": "Offer List Product",
        "url": "https://example.com/p/1",
        "offers": [
            {"price": "199.99", "priceCurrency": "GBP"},
            {"price": "299.99", "priceCurrency": "GBP"},
        ],
    }

    products = _parse(f'<script type="application/ld+json">{json.dumps(payload)}</script>')

    assert len(products) == 1
    assert products[0].price_value == pytest.approx(199.99)


# Verify that a link card without a price is dropped for any provider (checked with ek_ua).
@pytest.mark.unit
def test_parse_products_drops_cards_without_price_for_all_providers() -> None:
    html = '<a href="https://ek.example/p/1">Product without a price</a>'

    products = parse_products(
        html,
        provider="ek_ua",
        lane="secondary",
        method="curl_cffi",
        base_url="https://ek.ua/ek-list.php?search_=test",
        default_currency="UAH",
    )

    assert products == []


# Verify that a card with a priced link is kept with its price and currency, and that no image URL is
# extracted.
@pytest.mark.unit
def test_parse_products_keeps_card_with_price() -> None:
    html = (
        '<div class="card">'
        '<a href="https://example.com/p/1">Card Product £123.45</a>'
        '<img src="https://example.com/p/1.jpg">'
        '</div>'
    )

    products = _parse(html)

    assert len(products) == 1
    assert products[0].price_value == pytest.approx(123.45)
    assert products[0].currency == "GBP"
    assert not hasattr(products[0], "image_url")


# Verify that a Yandex Market card takes the real price from the offer block ("51673 ₽") and not a number
# from the specification text (camera count and megapixels).
@pytest.mark.unit
def test_yandex_card_uses_real_price_outside_specification_branch() -> None:
    # Arrange: a product card with a title, a specification line and an offer price.
    html = (
        '<article class="product-card">'
        '<noframes>{"zoneData":{"price":{"currency":"RUR","value":"51673"}}}</noframes>'
        '<div class="content">'
        '<div class="details">'
        '<div><div><a href="/card/nothing-phone/4915325941">'
        'Смартфон Nothing Phone 2A 12/256 ГБ, Milk Белый, 2Sim'
        '</a></div></div>'
        '<span>Количество основных камер: 2 Разрешение основной камеры: 50 Мпикс</span>'
        '</div>'
        '<div class="offer"><span>Цена с картой Яндекс Пэй 51673 ₽</span></div>'
        '</div>'
        '</article>'
    )

    # Act.
    products = parse_products(
        html,
        provider="yandex_market",
        lane="primary",
        method="curl_cffi",
        base_url="https://market.yandex.ru/search?text=nothing",
        default_currency="RUB",
    )

    # Assert: the single product carries the offer price in rubles.
    assert len(products) == 1
    assert products[0].price_text == "51673 ₽"
    assert products[0].price_value == pytest.approx(51673.0)
    assert products[0].currency == "RUB"


# Verify that price filter facet links ("Unter 10 €", "10 € - 20 €") are not parsed as products.
@pytest.mark.unit
def test_parse_products_drops_price_filter_facets() -> None:
    html = (
        '<div class="filters">'
        '<a href="https://bing.example/filter/under-10">Unter 10 €</a>'
        '<a href="https://bing.example/filter/10-20">10 € - 20 €</a>'
        '<a href="https://bing.example/filter/over-30">Über 30 €</a>'
        '</div>'
    )

    products = parse_products(
        html,
        provider="bing_shopping",
        lane="secondary",
        method="curl_cffi",
        base_url="https://www.bing.com/shop?q=test",
        default_currency="EUR",
    )

    assert products == []


# Verify that a short category link is dropped when the price appears only in a nearby sibling text
# and not in the link itself.
@pytest.mark.unit
def test_parse_products_drops_short_anchor_when_price_is_only_nearby_context() -> None:
    html = (
        '<div class="card">'
        '<a href="https://example.com/category/accessories">Accessories</a>'
        '<span>Samsung Galaxy S26 Ultra glass £8.60</span>'
        '</div>'
    )

    products = _parse(html)

    assert products == []
