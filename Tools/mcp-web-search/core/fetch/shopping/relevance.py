# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
import unicodedata
from urllib.parse import parse_qsl, urlsplit

from core.extract.stemming import stems
from core.search.nltk_languages import DEFAULT_LANGUAGE, TOKENIZER, catalog_query, script_of, stopwords_for

from .models import ShoppingProduct


_SEARCH_QUERY_PARAMETERS = frozenset({"k", "keyword", "keywords", "q", "query", "search", "searchtext", "text"})
_SEARCH_PATH_SEGMENTS = frozenset(
    {"categories", "category", "catalogsearch", "find", "s", "search", "searchresult", "searchresults", "tags"}
)
_NON_PRODUCT_SEGMENTS = frozenset({"article", "articles", "news", "question", "blog", "wiki"})
_PRODUCT_PATH_MARKERS = ("/card/", "/dp/", "/gp/product/", "/item/", "/p/", "/product/", "/products/")
_MACHINE_SNIPPET_MARKERS = ('"widgets"', '"collections"', "@marketfront/", "linkspeculation")
_ZERO_PRICE_RE = re.compile(
    r"(?<!\w)(?:[$€£¥₹₽₴₩]\s*0+(?:[.,]0+)?|0+(?:[.,]0+)?\s*(?:USD|EUR|GBP|JPY|CNY|RUB|UAH|KRW))\b",
    re.IGNORECASE,
)


# Query/title terms; CJK runs (no spaces) become character bigrams as in ASLM-Search.
def _terms(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", str(text or "")).casefold()
    terms: list[str] = []
    for token in TOKENIZER.tokenize(normalized):
        token = token.strip("-_.")
        if script_of(token) == "cjk" and not _is_identifier(token):
            terms.extend([token] if len(token) <= 2 else [token[i : i + 2] for i in range(len(token) - 1)])
        elif len(token) >= 2:
            terms.append(token)
    return list(dict.fromkeys(terms))


# Whether a term is a model number or code (letters with digits).
def _is_identifier(term: str) -> bool:
    return len(term) >= 2 and any(ch.isdigit() for ch in term) and any(ch.isalnum() for ch in term)


# Whether a model number occurs among the candidate tokens, as itself or as a prefix of a longer variant
# ("4070" in "4070-super").
def _identifier_present(term: str, candidates: set[str]) -> bool:
    if term in candidates:
        return True
    return any(
        candidate.startswith(term) and len(candidate) > len(term) and candidate[len(term)] in "-_."
        for candidate in candidates
    )


# Drop machine-readable snippets (JSON, markers) and zero prices from a product description.
def clean_product_snippet(value: str) -> str:
    snippet = " ".join(str(value or "").split()).strip()
    if snippet.startswith(("{", "[")) and any(marker in snippet.casefold() for marker in _MACHINE_SNIPPET_MARKERS):
        return ""
    return " ".join(_ZERO_PRICE_RE.sub("", snippet).split()).strip(" ·,;:-")


# True for a product detail page; search, category, article and engine pages are not.
def is_product_page_url(value: str) -> bool:
    try:
        parsed = urlsplit(str(value or "").strip())
    except ValueError:
        return False
    if parsed.scheme.casefold() not in {"http", "https"} or not parsed.hostname:
        return False
    host = parsed.hostname.casefold().removeprefix("www.")
    path = re.sub(r"/+", "/", parsed.path.casefold()) or "/"
    if host.startswith(("blog.", "news.", "zhidao.")):
        return False
    if host.endswith("bing.com"):
        return path.rstrip("/") == "/shop/productdetails"
    if host.endswith(("baidu.com", "brave.com", "duckduckgo.com", "google.com")):
        return False
    if any(marker in path for marker in _PRODUCT_PATH_MARKERS):
        return True
    segments = {segment for segment in path.split("/") if segment}
    if segments & (_SEARCH_PATH_SEGMENTS | _NON_PRODUCT_SEGMENTS):
        return False
    return not {name.casefold() for name, _ in parse_qsl(parsed.query)} & _SEARCH_QUERY_PARAMETERS


# 0..1 match of a product to the query, or None when it must be rejected (a model
# number from the query is missing, or too few of the query's words are present).
def product_relevance(product: ShoppingProduct, query: str, *, language: str = "") -> float | None:
    query_terms = _terms(catalog_query(query, language=language))
    identifiers = [term for term in query_terms if _is_identifier(term)]
    # NLTK stopwords of the query language plus English (mixed product queries).
    stopwords = stopwords_for(language) | stopwords_for(DEFAULT_LANGUAGE)
    words = [term for term in query_terms if not _is_identifier(term) and term not in stopwords]
    title_terms = set(_terms(product.title))
    text_terms = title_terms | set(_terms(product.snippet))

    if any(not _identifier_present(term, text_terms) for term in identifiers):
        return None
    if not words:
        return 1.0 if identifiers else 0.5

    # Whether a query word occurs among the candidate stems.
    def present(word: str, pool: set[str]) -> bool:
        if script_of(word) == "cjk":
            return word in pool
        word_stems = stems(word, language)
        return any(stems(candidate, language) & word_stems for candidate in pool if candidate.isalpha())

    in_title = sum(1 for word in words if present(word, title_terms))
    in_text = sum(1 for word in words if present(word, text_terms))
    coverage = (in_title + 0.5 * (in_text - in_title)) / len(words)
    if coverage < (0.5 if len(words) >= 2 else 1.0):
        return None
    return round(min(1.0, 0.4 + 0.6 * coverage), 3)


# Relevant products, best first; ties prefer a known price, then the engine's confidence.
def rank_products(products: list[ShoppingProduct], query: str, *, language: str = "") -> list[ShoppingProduct]:
    scored: list[tuple[float, ShoppingProduct]] = []
    for product in products:
        if not is_product_page_url(product.url):
            continue
        score = product_relevance(product, query, language=language)
        if score is None:
            continue
        product.snippet = clean_product_snippet(product.snippet)
        product.meta["query_relevance"] = score
        scored.append((score, product))
    scored.sort(key=lambda item: (item[0], item[1].price_value is not None, item[1].confidence), reverse=True)
    return [product for _score, product in scored]


# Key that identifies the same product across listings: host and path without query or fragment.
def canonical_product_key(product: ShoppingProduct) -> str:
    try:
        parsed = urlsplit(product.url)
    except ValueError:
        return product.url
    return f"{(parsed.hostname or '').casefold().removeprefix('www.')}{parsed.path.rstrip('/')}"


# Remove products whose canonical key was already seen, keeping the first.
def dedupe_products(products: list[ShoppingProduct]) -> list[ShoppingProduct]:
    seen: set[str] = set()
    out: list[ShoppingProduct] = []
    for product in products:
        key = canonical_product_key(product)
        if key in seen:
            continue
        seen.add(key)
        out.append(product)
    return out
