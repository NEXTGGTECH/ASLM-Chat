# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any

import httpx

from .models import ShoppingProduct


logger = logging.getLogger("core.fetch.shopping.currency")

# Public ExchangeRate-API endpoint (USD base) and the attribution its terms require us to show.
EXCHANGE_RATE_API_URL = "https://open.er-api.com/v6/latest/USD"
EXCHANGE_RATE_API_ATTRIBUTION = "Rates By Exchange Rate API"
EXCHANGE_RATE_API_ATTRIBUTION_URL = "https://www.exchangerate-api.com"

# The project currently ships 20 UI locales. A shopping result exposes exactly the
# same number of currency positions: its original currency plus 19 conversions.
DEFAULT_DISPLAY_CURRENCIES = (
    "USD",
    "EUR",
    "RUB",
    "UAH",
    "GBP",
    "CNY",
    "JPY",
    "KRW",
    "INR",
    "IDR",
    "TRY",
    "PLN",
    "BRL",
    "CAD",
    "AUD",
    "CHF",
    "VND",
    "AED",
    "MXN",
    "TWD",
)
DISPLAY_CURRENCY_COUNT = len(DEFAULT_DISPLAY_CURRENCIES)

# How long a fetched rate table stays cached, and the timeout of one rate request.
RATE_CACHE_TTL_SECONDS = 60 * 60
RATE_REQUEST_TIMEOUT_SECONDS = 6.0


# One snapshot of exchange rates. `reference_per_unit` maps a currency code to the value of one
# unit of it in the reference currency, so any pair converts through the reference.
@dataclass(frozen=True, slots=True)
class ExchangeRateTable:
    provider: str
    date: str
    source_url: str
    reference_currency: str
    reference_per_unit: dict[str, Decimal]
    attribution: str = EXCHANGE_RATE_API_ATTRIBUTION
    attribution_url: str = EXCHANGE_RATE_API_ATTRIBUTION_URL
    documentation_url: str = ""
    terms_url: str = ""

    # Return how many units of `quote` one unit of `base` buys, or None when either currency is
    # unknown or the quote currency has no usable value.
    def rate(self, base: str, quote: str) -> Decimal | None:
        base_value = self.reference_per_unit.get(str(base or "").upper())
        quote_value = self.reference_per_unit.get(str(quote or "").upper())
        if base_value is None or quote_value in (None, Decimal("0")):
            return None
        return base_value / quote_value


# Cached rate tables as (monotonic expiry time, table), and the lock that serializes refreshes.
_rate_cache: dict[str, tuple[float, ExchangeRateTable]] = {}
_rate_cache_lock = asyncio.Lock()


# Parse a positive finite number (a comma is accepted as the decimal separator), or None.
def _decimal(value: Any) -> Decimal | None:
    try:
        parsed = Decimal(str(value or "").strip().replace(" ", "").replace(",", "."))
    except (InvalidOperation, ValueError):
        return None
    return parsed if parsed.is_finite() and parsed > 0 else None


# Return the last-update date of a rate payload as an ISO date (from the unix stamp, else the
# raw UTC text).
def _rate_date(payload: dict[str, Any]) -> str:
    try:
        stamp = int(payload.get("time_last_update_unix") or 0)
    except (TypeError, ValueError):
        # A missing or non-numeric stamp falls back to the textual date below.
        stamp = 0
    if stamp > 0:
        return datetime.fromtimestamp(stamp, tz=timezone.utc).date().isoformat()
    return str(payload.get("time_last_update_utc") or "").strip()


# Parse an ExchangeRate-API reply into a rate table. The API gives units of each currency per
# reference unit; the table stores the inverse. Raises ValueError for an unsuccessful reply or
# one without usable rates.
def parse_exchange_rate_api_table(content: bytes) -> ExchangeRateTable:
    # Reject an unsuccessful reply.
    payload = json.loads(content)
    if not isinstance(payload, dict) or payload.get("result") != "success":
        error = payload.get("error-type") if isinstance(payload, dict) else "invalid-response"
        raise ValueError(f"ExchangeRate-API returned {error or 'an unsuccessful response'}")

    # Read the reference currency and invert every rate.
    reference_currency = str(payload.get("base_code") or "").strip().upper()
    raw_rates = payload.get("rates")
    if not reference_currency or not isinstance(raw_rates, dict):
        raise ValueError("ExchangeRate-API response contained no currency rates")

    reference_per_unit: dict[str, Decimal] = {reference_currency: Decimal("1")}
    for raw_code, raw_rate in raw_rates.items():
        code = str(raw_code or "").strip().upper()
        units_per_reference = _decimal(raw_rate)
        if code and units_per_reference:
            reference_per_unit[code] = Decimal("1") / units_per_reference
    if len(reference_per_unit) <= 1:
        raise ValueError("ExchangeRate-API response contained no usable currency rates")

    return ExchangeRateTable(
        provider="ExchangeRate-API",
        date=_rate_date(payload),
        source_url=EXCHANGE_RATE_API_URL,
        reference_currency=reference_currency,
        reference_per_unit=reference_per_unit,
        documentation_url=str(payload.get("documentation") or "").strip(),
        terms_url=str(payload.get("terms_of_use") or "").strip(),
    )


# Return the rate table from the cache, fetching and caching a fresh one when it has expired.
async def _fetch_table(*, client: httpx.AsyncClient) -> ExchangeRateTable:
    cache_key = "exchange-rate-api-usd"
    cached = _rate_cache.get(cache_key)
    now = time.monotonic()
    if cached and cached[0] > now:
        return cached[1]

    # A double-checked lock prevents a burst of simultaneous shopping searches
    # from turning one expired cache entry into several external API requests.
    async with _rate_cache_lock:
        now = time.monotonic()
        cached = _rate_cache.get(cache_key)
        if cached and cached[0] > now:
            return cached[1]
        response = await client.get(EXCHANGE_RATE_API_URL)
        response.raise_for_status()
        table = parse_exchange_rate_api_table(response.content)
        _rate_cache[cache_key] = (time.monotonic() + RATE_CACHE_TTL_SECONDS, table)
        return table


# Convert a price into the target currencies using the first table that knows each pair. Each
# entry carries the converted value (rounded to cents), the rate and its provenance. Returns an
# empty list for a missing currency or a non-positive amount.
def converted_prices(
    amount: float,
    currency: str,
    tables: list[ExchangeRateTable],
    *,
    targets: tuple[str, ...] = DEFAULT_DISPLAY_CURRENCIES,
) -> list[dict[str, Any]]:
    base = str(currency or "").strip().upper()
    source_amount = _decimal(amount)
    if not base or source_amount is None:
        return []

    # `targets` describes the total display width, including the original price.
    # This keeps the result at 20 currencies even for a source currency outside
    # the standard list.
    conversion_slots = max(0, len(targets) - 1)
    normalized_targets: list[str] = []
    for raw_quote in targets:
        quote = str(raw_quote or "").strip().upper()
        if quote and quote != base and quote not in normalized_targets:
            normalized_targets.append(quote)
        if len(normalized_targets) >= conversion_slots:
            break

    # Convert into each target with the first table that has the pair.
    output: list[dict[str, Any]] = []
    for quote in normalized_targets:
        for table in tables:
            rate = table.rate(base, quote)
            if rate is None:
                continue
            value = (source_amount * rate).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
            output.append(
                {
                    "currency": quote,
                    "value": float(value),
                    "price_text": f"{value:f} {quote}",
                    "rate": float(rate),
                    "rate_date": table.date,
                    "rate_provider": table.provider,
                    "rate_url": table.source_url,
                    "attribution": table.attribution,
                    "attribution_url": table.attribution_url,
                }
            )
            break
    return output


# Fill `converted_prices` of every product that has a price and currency, and return a summary
# (providers used, converted product count, display currencies, cache TTL). A rate fetch failure
# is logged and leaves the original prices untouched. A client passed in is not closed here.
async def enrich_products_with_exchange_rates(
    products: list[ShoppingProduct],
    *,
    targets: tuple[str, ...] = DEFAULT_DISPLAY_CURRENCIES,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    # Only products with a price and a currency can be converted.
    priced = [
        product
        for product in products
        if product.price_value is not None and str(product.currency or "").strip()
    ]
    if not priced:
        return {
            "providers": [],
            "converted_products": 0,
            "display_currency_count": len(targets),
            "currencies": list(targets),
            "cache_ttl_seconds": RATE_CACHE_TTL_SECONDS,
        }

    # Fetch the rate table, closing the HTTP client only when this function created it.
    owns_client = client is None
    http = client or httpx.AsyncClient(
        timeout=RATE_REQUEST_TIMEOUT_SECONDS,
        follow_redirects=True,
        headers={"Accept": "application/json"},
    )
    tables: list[ExchangeRateTable] = []
    try:
        try:
            tables.append(await _fetch_table(client=http))
        except Exception as exc:  # noqa: BLE001 - preserve original prices on API failure
            logger.warning("ExchangeRate-API fetch failed: %s", exc)
    finally:
        if owns_client:
            await http.aclose()

    # Convert each product and report what was done.
    converted_count = 0
    for product in priced:
        product.converted_prices = converted_prices(
            product.price_value,
            product.currency,
            tables,
            targets=targets,
        )
        if product.converted_prices:
            converted_count += 1
    return {
        "providers": [
            {
                "name": table.provider,
                "date": table.date,
                "url": table.source_url,
                "attribution": table.attribution,
                "attribution_url": table.attribution_url,
                "documentation_url": table.documentation_url,
                "terms_url": table.terms_url,
            }
            for table in tables
        ],
        "converted_products": converted_count,
        "display_currency_count": len(targets),
        "converted_currency_count": max(0, len(targets) - 1),
        "currencies": list(targets),
        "cache_ttl_seconds": RATE_CACHE_TTL_SECONDS,
    }
