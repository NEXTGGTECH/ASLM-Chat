# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import time

import pytest

from core.fetch.shopping.assets import ShoppingAssetCache
from core.fetch.shopping.engine import ShoppingSearchEngine, result_to_jsonable
from core.fetch.shopping.models import ShoppingProviderAttempt
from core.fetch.shopping.providers import PROVIDERS, providers_for_lane


# Shopping now uses a process-wide ProviderState; reset it before each test so a cooldown
# tripped in one test cannot leak into another's provider selection.
@pytest.fixture(autouse=True)
def _reset_shopping_state():
    from core.fetch.shopping.engine import _GLOBAL_PROVIDER_STATE

    _GLOBAL_PROVIDER_STATE.failures.clear()
    _GLOBAL_PROVIDER_STATE.cooldown_until.clear()
    _GLOBAL_PROVIDER_STATE.last_status.clear()
    yield


# Build a fake provider results page with the given number of product cards, each a link carrying a
# price in the given currency symbol and an image.
def _html(provider: str, count: int, *, currency: str = "$") -> str:
    cards = []
    for idx in range(count):
        cards.append(
            f'<div class="card"><a href="https://{provider}.example/p/{idx}">'
            f"{provider} product {idx} {currency}{idx + 10}.99</a>"
            f'<img src="https://img.example/{provider}-{idx}.jpg"></div>'
        )
    return "<html><body>" + "".join(cards) + "</body></html>"


# Verify that a medium-effort search of 10 products mixes the primary and secondary lanes 60/40 (6 and 4
# products) and reports the ratios and non-negative timings.
@pytest.mark.unit
def test_shopping_engine_medium_effort_mixes_primary_secondary_60_40(monkeypatch) -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())

    # Answer every provider with a successful page of 20 products.
    async def fake_fetch(url, provider, method):
        html = _html(provider.name, 20)
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=True,
            elapsed_ms=1,
            status_code=200,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    result = asyncio_run(engine.search("iphone price", effort="medium", limit=10))

    assert len(result.products) == 10
    assert sum(1 for product in result.products if product.lane == "primary") == 6
    assert sum(1 for product in result.products if product.lane == "secondary") == 4
    assert result.primary_ratio == 0.60
    assert result.secondary_ratio == 0.40
    assert result.timings["total_elapsed_ms"] >= 0
    assert result.timings["fetch_elapsed_ms"] >= 0
    assert result.timings["parse_elapsed_ms"] >= 0
    assert all(attempt.parse_elapsed_ms >= 0 for attempt in result.attempts)


# Verify that when two secondary providers fail (bing_shopping with 202, geizhals with 429) the engine
# swaps in another secondary (ek_ua), still attempts the failed ones and puts geizhals into cooldown.
@pytest.mark.unit
def test_shopping_engine_hot_swaps_secondary_after_failures(monkeypatch) -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())

    # Fail bing_shopping (202, empty page) and geizhals (429), succeed for all other providers.
    async def fake_fetch(url, provider, method):
        if provider.name == "pricerunner":
            status = 200
            html = _html(provider.name, 20)
        elif provider.name == "bing_shopping":
            status = 202
            html = ""
        elif provider.name == "geizhals":
            status = 429
            html = ""
        else:
            status = 200
            html = _html(provider.name, 20, currency="₴")
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=bool(html) and 200 <= status < 300,
            elapsed_ms=1,
            status_code=status,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    result = asyncio_run(engine.search("rtx 5070 price", effort="medium", limit=10))

    secondary_sources = {product.source for product in result.products if product.lane == "secondary"}
    attempted = [attempt.provider for attempt in result.attempts]
    assert "bing_shopping" in attempted
    assert "geizhals" in attempted
    assert "ek_ua" in attempted
    assert secondary_sources == {"ek_ua"}
    assert "geizhals" in result.provider_state["cooldowns"]


# Verify that when the primary provider (pricerunner) times out, the shortfall of 6 primary products is
# backfilled from the secondary lane, so the result still has 10 products, all from bing_shopping.
@pytest.mark.unit
def test_shopping_engine_backfills_primary_shortfall_from_secondary(monkeypatch) -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())

    # Fail pricerunner with a timeout and answer all other providers with a page of 20 products.
    async def fake_fetch(url, provider, method):
        if provider.name == "pricerunner":
            html = ""
            return html, ShoppingProviderAttempt(
                provider=provider.name,
                lane=provider.lane,
                method=method,
                url=url,
                ok=False,
                elapsed_ms=5,
                error="timeout",
            )
        html = _html(provider.name, 20)
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=True,
            elapsed_ms=1,
            status_code=200,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    result = asyncio_run(engine.search("rtx 5070 price", effort="medium", limit=10))

    assert len(result.products) == 10
    assert {product.source for product in result.products} == {"bing_shopping"}
    assert result.timings["primary_shortfall"] == 6
    assert result.timings["secondary_fetch_limit"] == 10


# Verify that the primary and secondary lanes run concurrently: with every fetch taking 50 ms the whole
# search finishes in under 140 ms, and each lane takes at least 50 ms.
@pytest.mark.unit
def test_shopping_engine_runs_primary_and_secondary_concurrently(monkeypatch) -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())

    # Answer every provider after a 50 ms delay with a page of 20 products.
    async def fake_fetch(url, provider, method):
        await asyncio.sleep(0.05)
        html = _html(provider.name, 20)
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=True,
            elapsed_ms=50,
            status_code=200,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    started = time.perf_counter()
    result = asyncio_run(engine.search("iphone price", effort="medium", limit=10, hard_timeout_ms=1000))
    elapsed_ms = int((time.perf_counter() - started) * 1000)

    assert len(result.products) == 10
    assert elapsed_ms < 140
    assert result.timings["primary_lane_elapsed_ms"] >= 50
    assert result.timings["secondary_lane_elapsed_ms"] >= 50


# Verify that a slow primary lane (200 ms) does not block the result past the 80 ms hard timeout: the
# engine returns the partial buffer of 10 products from the fast secondary lane only.
@pytest.mark.unit
def test_shopping_engine_returns_partial_buffer_on_hard_timeout(monkeypatch) -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())

    # Delay primary providers by 200 ms and secondary ones by 10 ms, each answering with 20 products.
    async def fake_fetch(url, provider, method):
        if provider.lane == "primary":
            await asyncio.sleep(0.2)
            html = _html(provider.name, 20)
        else:
            await asyncio.sleep(0.01)
            html = _html(provider.name, 20)
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=True,
            elapsed_ms=10,
            status_code=200,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    result = asyncio_run(engine.search("iphone price", effort="medium", limit=10, hard_timeout_ms=80))

    assert result.partial is True
    assert result.partial_reason in {"buffer_full", "hard_timeout"}
    assert len(result.products) == 10
    assert {product.lane for product in result.products} == {"secondary"}


# Verify that when only the primary lane answers before the hard timeout, its partial buffer is not cut
# down to the primary quota: all 10 products come from the primary lane.
@pytest.mark.unit
def test_shopping_engine_partial_primary_buffer_is_not_cut_to_primary_quota(monkeypatch) -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())

    # Delay secondary providers by 200 ms and primary ones by 10 ms, each answering with 20 products.
    async def fake_fetch(url, provider, method):
        if provider.lane == "secondary":
            await asyncio.sleep(0.2)
        else:
            await asyncio.sleep(0.01)
        html = _html(provider.name, 20)
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=True,
            elapsed_ms=10,
            status_code=200,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    result = asyncio_run(engine.search("iphone price", effort="medium", limit=10, hard_timeout_ms=80))

    assert result.partial is True
    assert result.partial_reason in {"buffer_full", "hard_timeout"}
    assert len(result.products) == 10
    assert {product.lane for product in result.products} == {"primary"}


# Verify that a provider already in cooldown (bing_shopping) is skipped, so the secondary lane is served
# by geizhals instead.
@pytest.mark.unit
def test_shopping_engine_skips_provider_already_in_cooldown(monkeypatch) -> None:
    from core.fetch.shopping.engine import ProviderState

    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache(), state=ProviderState())
    engine.state.cooldown_until["bing_shopping"] = time.time() + 60

    # Answer every provider with a successful page of 20 products.
    async def fake_fetch(url, provider, method):
        html = _html(provider.name, 20)
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=True,
            elapsed_ms=1,
            status_code=200,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    result = asyncio_run(engine.search("camera price", effort="medium", limit=10))

    secondary_sources = {product.source for product in result.products if product.lane == "secondary"}
    assert "bing_shopping" not in secondary_sources
    assert secondary_sources == {"geizhals"}


# Verify that the provider router puts the Russian sources first for language "ru" in both lanes.
@pytest.mark.unit
def test_shopping_provider_router_prefers_russian_sources_for_ru() -> None:
    primary = [provider.name for provider in providers_for_lane("primary", language="ru")]
    secondary = [provider.name for provider in providers_for_lane("secondary", language="ru")]

    assert primary[:2] == ["yandex_market", "ek_ua"]
    assert secondary[:4] == ["ek_ua", "chinandex", "aliexpress", "kakaku"]


# Verify that the provider router puts the Chinese aggregator first for language "zh" in both lanes.
@pytest.mark.unit
def test_shopping_provider_router_prefers_chinese_aggregator_for_zh() -> None:
    primary = [provider.name for provider in providers_for_lane("primary", language="zh")]
    secondary = [provider.name for provider in providers_for_lane("secondary", language="zh")]

    assert primary[:2] == ["chinandex", "aliexpress"]
    assert secondary[:3] == ["aliexpress", "bing_shopping", "kakaku"]


# Verify that the provider router puts the Japanese aggregator first for language "ja" in the primary lane
# and the Chinese and Bing sources first in the secondary lane.
@pytest.mark.unit
def test_shopping_provider_router_prefers_japanese_aggregator_for_ja() -> None:
    primary = [provider.name for provider in providers_for_lane("primary", language="ja")]
    secondary = [provider.name for provider in providers_for_lane("secondary", language="ja")]

    assert primary[:2] == ["kakaku", "pricerunner"]
    assert secondary[:2] == ["chinandex", "bing_shopping"]


# Verify that the lane search passes its language to the provider routes: for "ru" the Russian
# yandex_market is tried before ek_ua.
@pytest.mark.unit
def test_shopping_engine_passes_language_to_routes(monkeypatch) -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())
    attempted: list[str] = []

    # Record the attempted provider; yandex_market returns an empty page, all others 20 products.
    async def fake_fetch(url, provider, method):
        attempted.append(provider.name)
        html = "" if provider.name == "yandex_market" else _html(provider.name, 20)
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=bool(html),
            elapsed_ms=1,
            status_code=200 if html else 204,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    products, _attempts = asyncio_run(engine._lane("купить iphone", "primary", limit=4, language="ru"))

    assert products
    assert "yandex_market" in attempted
    assert attempted.index("yandex_market") < attempted.index("ek_ua")


# Verify that for a Russian query the engine waits for the slower regional primary provider
# (yandex_market) instead of returning early with only the secondary results.
@pytest.mark.unit
def test_shopping_engine_waits_for_regional_primary_before_secondary_early_return(monkeypatch) -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())

    # Delay yandex_market by 50 ms; other primary providers return nothing, secondary ones succeed at once.
    async def fake_fetch(url, provider, method):
        if provider.name == "yandex_market":
            await asyncio.sleep(0.05)
            html = _html(provider.name, 20, currency="₽")
        elif provider.lane == "secondary":
            html = _html(provider.name, 20)
        else:
            html = ""
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=bool(html),
            elapsed_ms=1,
            status_code=200 if html else 204,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    result = asyncio_run(engine.search("защитное стекло дешево", effort="medium", limit=6, language="ru"))

    assert any(product.source == "yandex_market" for product in result.products)


# Verify that the hard timeout for medium effort is longer for the regional "ru" route than for "en".
@pytest.mark.unit
def test_shopping_engine_uses_longer_hard_timeout_for_regional_routes() -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())

    assert engine._hard_timeout_for_effort("medium", "ru") > engine._hard_timeout_for_effort("medium", "en")


# Verify that every product gets a favicon proxy URL without any favicon being fetched or cached.
@pytest.mark.unit
def test_shopping_engine_sets_favicon_proxy_without_fetching_favicon(monkeypatch) -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())

    # Answer every provider with a successful page of 20 products.
    async def fake_fetch(url, provider, method):
        html = _html(provider.name, 20)
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=True,
            elapsed_ms=1,
            status_code=200,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    result = asyncio_run(engine.search("iphone price", effort="low", limit=4))

    assert result.products
    assert all(product.favicon_url for product in result.products)
    assert all(product.favicon_url.startswith("/api/favicon/?domain=") for product in result.products)
    assert all(not hasattr(product, "favicon_cache_url") for product in result.products)


# Verify that every shopping provider is configured with exactly one transport, the one selected by
# probing (httpx for pricerunner, curl_cffi for the others).
@pytest.mark.unit
def test_shopping_providers_use_probe_selected_single_transport() -> None:
    methods = {provider.name: provider.methods for provider in PROVIDERS}

    assert methods == {
        "pricerunner": ("httpx",),
        "bing_shopping": ("curl_cffi",),
        "geizhals": ("curl_cffi",),
        "ek_ua": ("curl_cffi",),
        "yandex_market": ("curl_cffi",),
        "aliexpress": ("curl_cffi",),
        "chinandex": ("curl_cffi",),
        "kakaku": ("curl_cffi",),
    }


# Verify that the JSON form of a shopping result includes the total and fetch timings and the parse time
# of each attempt.
@pytest.mark.unit
def test_shopping_result_json_includes_timings(monkeypatch) -> None:
    engine = ShoppingSearchEngine(asset_cache=ShoppingAssetCache())

    # Answer every provider with a successful page of 20 products and a 7 ms fetch time.
    async def fake_fetch(url, provider, method):
        html = _html(provider.name, 20)
        return html, ShoppingProviderAttempt(
            provider=provider.name,
            lane=provider.lane,
            method=method,
            url=url,
            ok=True,
            elapsed_ms=7,
            status_code=200,
            bytes=len(html),
        )

    monkeypatch.setattr(engine, "_fetch", fake_fetch)

    payload = result_to_jsonable(asyncio_run(engine.search("iphone price", effort="low", limit=4)))

    assert payload["timings"]["total_elapsed_ms"] >= 0
    assert payload["timings"]["fetch_elapsed_ms"] >= 7
    assert "parse_elapsed_ms" in payload["attempts"][0]


# Run a coroutine to completion in a new event loop and return its result.
def asyncio_run(coro):
    import asyncio

    return asyncio.run(coro)
