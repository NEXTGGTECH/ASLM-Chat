# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import hashlib
import html
import json
import re
import time
from urllib.parse import parse_qs, urljoin, urlparse

from bs4 import BeautifulSoup, Comment

from .models import ShoppingProduct


# Price patterns: the currency marker alternatives, then an amount that follows the marker
# ("£199") or precedes it ("199 £"). The amount groups are named differently (`amount`/`currency`
# versus `amount2`/`currency2`) so both patterns can be read from one match dict.
PRICE_MARKER_RE = r"[$€£₽₴¥]|zł(?!\w)|руб\.?(?!\w)|грн\.?(?!\w)|р\.?(?!\w)|円"
PRICE_CURRENCY_FIRST_RE = re.compile(
    rf"(?P<currency>{PRICE_MARKER_RE})\s*"
    r"(?P<amount>\d[\d\s.,]{1,14})",
    re.IGNORECASE,
)
PRICE_AMOUNT_FIRST_RE = re.compile(
    r"(?P<amount2>\d[\d\s.,]{1,14})\s*"
    rf"(?P<currency2>{PRICE_MARKER_RE})",
    re.IGNORECASE,
)

# Currency marker (lowercased, trailing dot removed) to ISO code.
CURRENCY_MAP = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₽": "RUB", "руб": "RUB", "руб.": "RUB", "р": "RUB", "р.": "RUB",
    "₴": "UAH", "грн": "UAH", "грн.": "UAH",
    "¥": "JPY", "円": "JPY",
    "zł": "PLN",
}


# Unescape HTML entities, collapse whitespace and cut the text to `limit` characters.
def compact(text: str, *, limit: int = 260) -> str:
    return re.sub(r"\s+", " ", html.unescape(text or "")).strip()[:limit].rstrip()


# Return the lowercase host of a URL without a leading "www.".
def source_domain(url: str) -> str:
    return (urlparse(url or "").netloc or "").lower().removeprefix("www.")


# Resolve a raw href into an absolute URL: unwrap Google "/url?q=" redirects, add the https scheme
# to protocol-relative links and join root-relative paths onto `base`.
def normalize_url(raw: str, base: str = "") -> str:
    value = html.unescape(raw or "").strip()
    if not value:
        return ""
    if value.startswith("/url?") or value.startswith("https://www.google.") and "/url?" in value:
        target = parse_qs(urlparse(value).query).get("q", [""])[0]
        if target:
            value = target
    if value.startswith("//"):
        value = "https:" + value
    if base and value.startswith("/"):
        value = urljoin(base, value)
    return value


# Find a price in free text and return (price text, amount, ISO currency); ("", None, "") when
# there is none. A currency-first match ("£199") wins over an amount-first one ("199 £"). With
# `allow_bare`, a bare number is accepted and gets `default_currency`. A currency marker without
# a known ISO mapping falls back to the upper-cased marker, then to `default_currency`.
def parse_price(text: str, *, default_currency: str = "", allow_bare: bool = False) -> tuple[str, float | None, str]:
    source = text or ""
    match = None

    # Prefer a currency-first match, skipping one glued to a preceding digit or separator.
    for candidate in PRICE_CURRENCY_FIRST_RE.finditer(source):
        if candidate.start() > 0 and source[candidate.start() - 1] in "0123456789,.":
            continue
        match = candidate
        break

    # Otherwise try amount-first, skipping a number cut off mid-number or a bare four-digit model
    # number followed by a space.
    if match is None:
        for candidate in PRICE_AMOUNT_FIRST_RE.finditer(source):
            if candidate.end() < len(source) and source[candidate.end() : candidate.end() + 1].isdigit():
                continue
            if _bare_integer_before_spaced_currency(candidate.group("amount2") or ""):
                continue
            match = candidate
            break

    # No marker at all: only a bare amount can still count, and only when the caller allows it.
    if not match:
        if not allow_bare:
            return "", None, ""
        bare_amount = parse_amount_value(source)
        if bare_amount is None:
            return "", None, ""
        return compact(source, limit=80), bare_amount, default_currency.upper()

    # Convert the matched amount and resolve the currency, rejecting rating-like "р" matches.
    groups = match.groupdict()
    currency_raw = (groups.get("currency") or groups.get("currency2") or "").lower().rstrip(".")
    amount_raw = groups.get("amount") or groups.get("amount2") or ""
    amount = parse_amount_value(amount_raw)
    if amount is None:
        return "", None, ""
    if _looks_like_rating_ruble_false_positive(match, source, amount):
        return "", None, ""
    currency = CURRENCY_MAP.get(currency_raw, currency_raw.upper()) or default_currency.upper()
    return compact(match.group(0), limit=80), amount, currency


# Parse a localized amount ("1 299,90", "1,299.90", "12.5") into a float, or None when the text
# has anything but digits, spaces, dots and commas, or its thousands grouping is invalid.
def parse_amount_value(raw: str) -> float | None:
    value = (raw or "").replace("\xa0", " ").strip()
    if not value or re.search(r"[^\d\s.,]", value):
        return None
    value = re.sub(r"\s*([,.])\s*", r"\1", value)
    value = re.sub(r"\s+", " ", value).strip()
    if not value or not re.search(r"\d", value):
        return None

    # Decide the decimal separator: the last of "," and "." when both occur, otherwise a single
    # separator followed by one or two digits.
    decimal_sep = ""
    if "," in value and "." in value:
        decimal_sep = "," if value.rfind(",") > value.rfind(".") else "."
    elif value.count(",") == 1:
        tail = value.rsplit(",", 1)[1]
        decimal_sep = "," if 1 <= len(tail) <= 2 else ""
    elif value.count(".") == 1:
        tail = value.rsplit(".", 1)[1]
        decimal_sep = "." if 1 <= len(tail) <= 2 else ""

    # Split off the decimal part (at most two digits).
    if decimal_sep:
        integer_part, decimal_part = value.rsplit(decimal_sep, 1)
        if not decimal_part.isdigit() or len(decimal_part) > 2:
            return None
    else:
        integer_part, decimal_part = value, ""

    # Find the thousands separator of the integer part.
    other_sep = "." if decimal_sep == "," else ","
    grouping_sep = ""
    if decimal_sep:
        grouping_sep = other_sep if other_sep in integer_part else ""
    elif "," in integer_part and "." not in integer_part:
        grouping_sep = ","
    elif "." in integer_part and "," not in integer_part:
        grouping_sep = "."

    # Validate the thousands groups (1-3 leading digits, then groups of exactly three) and join them.
    if grouping_sep:
        groups = integer_part.split(grouping_sep)
        if not _valid_grouped_integer(groups):
            return None
        integer_digits = "".join(groups)
    elif other_sep in integer_part:
        groups = integer_part.split(other_sep)
        if not _valid_grouped_integer(groups):
            return None
        integer_digits = "".join(groups)
    elif " " in integer_part:
        groups = integer_part.split(" ")
        if not _valid_grouped_integer(groups):
            return None
        integer_digits = "".join(groups)
    else:
        integer_digits = integer_part

    if not integer_digits.isdigit():
        return None

    normalized = integer_digits
    if decimal_part:
        normalized = f"{normalized}.{decimal_part}"
    try:
        return float(normalized)
    except ValueError:
        # Text that float() still rejects is treated as no price.
        return None


# Return True for a bare four-digit number followed by whitespace, which is more likely a model
# number than a price.
def _bare_integer_before_spaced_currency(raw: str) -> bool:
    value = raw or ""
    stripped = value.strip()

    # Four bare digits are commonly a model number (`RTX 5070 £...`). Five or
    # more digits are also a normal ungrouped marketplace price (`51673 ₽`).
    return len(stripped) == 4 and stripped.isdigit() and value[-1:].isspace()


# Return True when a match on the "р" ruble marker with a small amount (at most 5) sits next to
# the word "rating"/"рейтинг", i.e. it is a rating and not a price.
def _looks_like_rating_ruble_false_positive(match: re.Match[str], source: str, amount: float) -> bool:
    groups = match.groupdict()
    currency_raw = (groups.get("currency") or groups.get("currency2") or "").lower().rstrip(".")
    if currency_raw != "р" or amount > 5:
        return False
    before = source[max(0, match.start() - 18):match.start()].lower()
    after = source[match.end():match.end() + 18].lower()
    return "рейтинг" in before or "рейтинг" in after or "rating" in before or "rating" in after


# Return True when digit groups form a valid thousands-grouped integer: one to three leading
# digits followed by groups of exactly three (a single group only needs to be digits).
def _valid_grouped_integer(groups: list[str]) -> bool:
    if len(groups) <= 1:
        return bool(groups and groups[0].isdigit())
    if not groups[0].isdigit() or not (1 <= len(groups[0]) <= 3):
        return False
    return all(group.isdigit() and len(group) == 3 for group in groups[1:])


# Build a stable 16-character product id from the source, URL and title.
def product_id(url: str, title: str, source: str) -> str:
    payload = f"{source}\n{url}\n{title}".encode("utf-8", errors="replace")
    return hashlib.sha256(payload).hexdigest()[:16]


# Score a product by how complete it is: a base value plus bonuses for a price, an http(s) URL, a
# snippet and a seller or availability. The score never exceeds 1.0.
def score_product(product: ShoppingProduct) -> float:
    score = 0.2
    if product.price_text:
        score += 0.35
    if product.url.startswith("http"):
        score += 0.15
    if product.snippet:
        score += 0.1
    if product.seller or product.availability:
        score += 0.1
    return round(min(score, 1.0), 3)


# Extract priced products from a provider's HTML: JSON-LD products first, then product-like
# anchors; the combined list is deduped by URL and title.
def parse_products(
    html_text: str,
    *,
    provider: str,
    lane: str,
    method: str,
    base_url: str,
    default_currency: str = "",
) -> list[ShoppingProduct]:
    if not html_text:
        return []
    soup = BeautifulSoup(html_text, "html.parser")
    products: list[ShoppingProduct] = []

    # Structured JSON-LD products first, then anchor-based product cards.
    products.extend(_jsonld_products(
        soup,
        provider=provider,
        lane=lane,
        method=method,
        base_url=base_url,
        default_currency=default_currency,
    ))
    products.extend(_card_products(
        soup,
        provider=provider,
        lane=lane,
        method=method,
        base_url=base_url,
        default_currency=default_currency,
    ))
    return _dedupe(products)


# Build a ShoppingProduct with a generated id, compacted text and a completeness score. The
# price text, price value and currency are all required, otherwise ValueError is raised.
def _make_product(
    *,
    title: str,
    url: str,
    provider: str,
    lane: str,
    method: str,
    price_text: str = "",
    price_value: float | None = None,
    currency: str = "",
    snippet: str = "",
    seller: str = "",
    availability: str = "",
) -> ShoppingProduct:
    if not price_text or price_value is None or not currency:
        raise ValueError("shopping product requires a parsed price")

    # Fill the record, then score it.
    product = ShoppingProduct(
        id=product_id(url, title, provider),
        title=compact(title, limit=220),
        url=url,
        source=provider,
        source_domain=source_domain(url),
        lane=lane,
        price_text=price_text,
        price_value=price_value,
        currency=currency,
        seller=seller,
        availability=availability,
        snippet=compact(snippet, limit=700),
        fetched_at=time.time(),
        meta={"method": method},
    )
    product.confidence = score_product(product)
    return product


# Collect products from schema.org JSON-LD blocks. A "@graph" is flattened into the work queue,
# and only items typed "Product" with a title, URL, parsed price and currency are kept.
def _jsonld_products(
    soup: BeautifulSoup,
    *,
    provider: str,
    lane: str,
    method: str,
    base_url: str,
    default_currency: str,
) -> list[ShoppingProduct]:
    out: list[ShoppingProduct] = []
    for script in soup.find_all("script", attrs={"type": re.compile("ld\\+json", re.I)}):
        payload = script.get_text(" ", strip=True)
        if not payload:
            continue
        try:
            data = json.loads(payload)
        except Exception:
            # Invalid JSON in one block must not stop the other JSON-LD blocks.
            continue

        # Walk the items, expanding "@graph" containers and keeping only Product entries.
        queue = data if isinstance(data, list) else [data]
        for item in queue:
            if not isinstance(item, dict):
                continue
            graph = item.get("@graph")
            if isinstance(graph, list):
                queue.extend(x for x in graph if isinstance(x, dict))
                continue
            item_type = item.get("@type")
            item_types = item_type if isinstance(item_type, list) else [item_type]
            if "Product" not in item_types:
                continue

            # Read the offer's price and currency, falling back to the provider's currency.
            offers = _first_offer(item.get("offers"))
            title = compact(str(item.get("name") or ""))
            url = normalize_url(str(item.get("url") or offers.get("url") or base_url), base_url)
            price = str(offers.get("price") or "")
            currency = str(offers.get("priceCurrency") or default_currency).upper()
            price_value = parse_amount_value(price)
            price_text = compact(price, limit=80) if price_value is not None else ""
            if title and url and price_text and price_value is not None and currency:
                out.append(_make_product(
                    title=title,
                    url=url,
                    provider=provider,
                    lane=lane,
                    method=method,
                    price_text=price_text,
                    price_value=price_value,
                    currency=currency,
                    snippet=str(item.get("description") or ""),
                    availability=str(offers.get("availability") or "").rsplit("/", 1)[-1],
                ))
    return out


# Return the offer dict from a JSON-LD "offers" value, which may be a dict or a list (first dict).
def _first_offer(raw: object) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                return item
    return {}


# Collect products from anchors: the link text is the title, and the price comes from the title
# itself or from the nearest ancestor text that holds one. Anchors that look like listings, rating
# links, price-range filters or too-short titles are skipped.
def _card_products(
    soup: BeautifulSoup,
    *,
    provider: str,
    lane: str,
    method: str,
    base_url: str,
    default_currency: str,
) -> list[ShoppingProduct]:
    out: list[ShoppingProduct] = []
    for anchor in soup.find_all("a", href=True):
        title = compact(anchor.get_text(" ", strip=True), limit=220)
        url = normalize_url(anchor.get("href", ""), base_url)
        if len(title) < 8 or not url.startswith("http"):
            continue

        # Drop category/filter listings and rating sub-links — they masquerade as products
        # but only re-list one facet/score at the page-context price (the duplicate source).
        if _is_listing_or_facet_url(url) or _is_rating_anchor(title, url):
            continue

        # Find the price in the title, else in the nearest ancestor that contains one.
        price_text, price_value, currency = parse_price(title, default_currency=default_currency)
        price_from_title = bool(price_text)
        context = _visible_ancestor_text(anchor, depth=3)
        if not price_text:
            context, price_text, price_value, currency = _nearest_priced_context(
                anchor, default_currency=default_currency
            )

        # Skip price-range filter links, titles with too little product signal (when the price was
        # not in the title) and anchors without a complete price.
        if _looks_like_price_filter_title(title):
            continue
        if not price_from_title and not _has_enough_product_title_signal(title):
            continue
        if not price_text or price_value is None or not currency:
            continue
        out.append(_make_product(
            title=title,
            url=url,
            provider=provider,
            lane=lane,
            method=method,
            price_text=price_text,
            price_value=price_value,
            currency=currency,
            snippet=context,
        ))
    return out


# Tags whose text is never shown to the user and is excluded from visible text.
_NON_VISIBLE_TEXT_PARENTS = frozenset({"script", "style", "noscript", "noframes", "template"})


# Return the visible text under a node (no comments, scripts or styles), compacted to `limit`.
def _visible_text(node: object, *, limit: int = 1200) -> str:
    find_all = getattr(node, "find_all", None)
    if not callable(find_all):
        return ""
    strings: list[str] = []
    for item in find_all(string=True):
        if isinstance(item, Comment):
            continue
        if any(
            str(getattr(parent, "name", "") or "").lower() in _NON_VISIBLE_TEXT_PARENTS
            for parent in getattr(item, "parents", ())
            if parent is not node
        ):
            continue
        strings.append(str(item))
    return compact(" ".join(strings), limit=limit)


# Return the visible text of the ancestor `depth` levels above the anchor (or the topmost one).
def _visible_ancestor_text(anchor: object, *, depth: int) -> str:
    node = anchor
    for _ in range(max(0, depth)):
        parent = getattr(node, "parent", None)
        if parent is None:
            break
        node = parent
    return _visible_text(node)


# Walk up from the anchor and return (context text, price text, amount, currency) of the first
# ancestor whose visible text contains a price. When none does, return the last non-empty context
# with an empty price.
def _nearest_priced_context(
    anchor: object, *, default_currency: str
) -> tuple[str, str, float | None, str]:
    node = anchor
    last_context = ""

    # Yandex Market keeps specifications and the price in adjacent branches of
    # the product card. Climbing up to five ancestors reaches the common card
    # without reaching the surrounding result list; other providers usually
    # resolve earlier.
    for _ in range(5):
        parent = getattr(node, "parent", None)
        if parent is None:
            break
        node = parent
        context = _visible_text(node)
        if context:
            last_context = context
        price_text, price_value, currency = parse_price(
            context, default_currency=default_currency
        )
        if price_text and price_value is not None and currency:
            return context, price_text, price_value, currency
    return last_context, "", None, ""


# Listing/search/facet endpoints are not products: comparison sites expose category and
# filter links (…/results?q=…&af_CATEGORY=…) that share one page-context price, so each
# facet becomes a near-duplicate "product" at the same bogus price. Product pages live on
# detail paths (…/pl/…, …/product/…) instead. Identity-blind: keys off path/query shape.
_LISTING_PATH_SEGMENTS = frozenset({
    "results", "search", "browse", "catalog", "category", "categories",
    "suche", "filter", "filters",
})


# Return True when a URL looks like a listing or facet page: a listing word in the path, or a
# facet/category/filter query parameter.
def _is_listing_or_facet_url(url: str) -> bool:
    parsed = urlparse(url or "")
    segments = {seg.lower() for seg in (parsed.path or "").split("/") if seg}
    if segments & _LISTING_PATH_SEGMENTS:
        return True
    query = (parsed.query or "").lower()

    # Facet params (af_category=, category=, filter=) mark a filtered listing, not an item.
    return bool(re.search(r"(^|&)(af_[a-z]+|category|categories|filter|facet)=", query))


# Rating/review sub-links of a product card (…#ratings, "Bewertung: 4.6 von 5 Sternen …")
# parse into a separate row at the card's price — a duplicate of the real product. Match
# only rating-dominated titles so normal titles carrying an inline score ("… 4.5 Wireless")
# are untouched.
_RATING_FRAGMENTS = frozenset({"ratings", "reviews", "rating", "review"})
_RATING_TITLE_RE = re.compile(
    r"^\s*(bewertung\b|\d+([.,]\d+)?\s*(von\s*5\s*sternen|out\s*of\s*5\s*stars?|stars?\b))",
    re.IGNORECASE,
)


# Return True for an anchor that links to a ratings/reviews fragment or has a rating-dominated title.
def _is_rating_anchor(title: str, url: str) -> bool:
    if urlparse(url or "").fragment.lower() in _RATING_FRAGMENTS:
        return True
    return bool(_RATING_TITLE_RE.match(title or ""))


# Return True for a price-range filter label such as "under 50 £" or "10 - 20 $".
def _looks_like_price_filter_title(title: str) -> bool:
    value = compact(title, limit=120).lower()
    if not value:
        return False
    if re.match(r"^(under|unter|over|über|ueber|до|от)\s+[\d\s.,]+", value, flags=re.I):
        return True
    if re.match(r"^[\d\s.,]+\s*(?:[$€£₽₴¥]|руб\.?|грн\.?|р\.?|円)\s*[-–]\s*[\d\s.,]+", value, flags=re.I):
        return True
    if re.match(r"^[\d\s.,]+\s*[-–]\s*[\d\s.,]+\s*(?:[$€£₽₴¥]|руб\.?|грн\.?|р\.?|円)", value, flags=re.I):
        return True
    return False


# Return True when a title has at least four words, or a letters-and-digits token that looks like a
# model name (for example "rtx 5070"), so a bare category label is not taken for a product.
def _has_enough_product_title_signal(title: str) -> bool:
    value = compact(title, limit=120).lower()
    tokens = re.findall(r"[\w]+", value, flags=re.UNICODE)
    if len(tokens) >= 4:
        return True
    return bool(re.search(r"\b[a-zа-яё]{2,}\s*\d{2,}\b|\b\d{2,}\s*[a-zа-яё]{2,}\b", value, flags=re.I))


# Drop products whose URL (without query and fragment) or normalized title was already seen,
# visiting them by descending confidence so the best copy is kept.
def _dedupe(products: list[ShoppingProduct]) -> list[ShoppingProduct]:
    out: list[ShoppingProduct] = []
    seen_urls: set[str] = set()
    seen_titles: set[str] = set()
    for product in sorted(products, key=lambda item: item.confidence, reverse=True):
        title_key = re.sub(r"\W+", " ", product.title.lower()).strip()[:90]

        # Strip fragment AND query so facet/tracking variants of one page
        # (…/item?hloc=eu&nocookie=1#ratings) collapse to a single product.
        url_key = product.url.split("#", 1)[0].split("?", 1)[0]
        if url_key in seen_urls or title_key in seen_titles:
            continue
        seen_urls.add(url_key)
        seen_titles.add(title_key)
        out.append(product)
    return out
