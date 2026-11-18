# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import regex
from nltk.corpus.reader.wordlist import NonbreakingPrefixesCorpusReader
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer

from core.search.query_locales import LOCALE_PROFILES, analyze_query, locale_profile

# NLTK language layer for the 20 locales the project supports (query_locales), ported
# from ASLM-Search core/search/nltk_languages.py. Per language it resolves what NLTK
# offers - Snowball stemmer, trained Punkt model, stopword list - by the language's NLTK
# name (nltk.langnames over the bcp47 data). Language scripts come from the project's own
# locale labels; sentence terminators come from Unicode (Sentence_Terminal), so there
# are no per-language tables except the commerce seed words below.

TOKENIZER = RegexpTokenizer(r"[^\W_]+(?:[-_.][^\W_]+)*")

PROJECT_LANGUAGES: tuple[str, ...] = tuple(dict.fromkeys(profile.language for profile in LOCALE_PROFILES))
DEFAULT_LANGUAGE = locale_profile("").language

# Commerce seed words for every project language. Snowball (or the prefix fallback) expands
# their inflections; NLTK has no multilingual commerce lexicon (WordNet/OMW lacks ru, de,
# uk, ko, hi, vi), so this one list is unavoidable. Seeds are concepts: buy, price, order,
# store. Russian also needs "куплю"/"покупка" because Snowball cannot join stem
# alternation ("купить" -> "куп", "куплю" -> "купл").
COMMERCE_SEEDS: dict[str, tuple[str, ...]] = {
    "ar": ("شراء", "اشتري", "سعر", "أسعار", "متجر"),
    "de": ("kaufen", "preis", "bestellen", "günstig"),
    "en": ("buy", "purchase", "price", "shop", "order", "sale", "deal", "cheap", "cost"),
    "es": ("comprar", "precio", "tienda", "oferta", "barato"),
    "fr": ("acheter", "prix", "boutique", "commander", "pas cher"),
    "hi": ("खरीदें", "खरीदना", "कीमत", "दाम", "ऑर्डर"),
    "id": ("beli", "harga", "toko", "jual", "murah"),
    "it": ("comprare", "prezzo", "negozio", "offerta"),
    "ja": ("購入", "価格", "値段", "買う", "通販", "最安値"),
    "ko": ("구매", "구입", "가격", "주문", "최저가"),
    "nl": ("kopen", "prijs", "bestellen", "goedkoop"),
    "pl": ("kupić", "kupic", "cena", "sklep", "zamówić", "tanio"),
    "pt": ("comprar", "preço", "loja", "barato"),
    "ru": ("купить", "куплю", "покупка", "цена", "стоимость", "заказать", "магазин", "продажа", "дешево"),
    "tr": ("satın", "almak", "fiyat", "sipariş", "ucuz"),
    "uk": ("купити", "ціна", "вартість", "замовити", "магазин"),
    "vi": ("mua", "giá", "đặt hàng", "cửa hàng"),
    "zh": ("购买", "購買", "价格", "價格", "多少钱", "多少錢", "买", "買"),
}

_NAME_CACHE: dict[str, str] = {}
_STEMMER_CACHE: dict[str, SnowballStemmer] = {}
_PUNKT_CACHE: dict[str, object] = {}
_STOPWORDS_CACHE: dict[str, frozenset[str]] = {}


# Language part of a locale code, lowercase ("pt-BR" gives "pt").
def base_code(code: str) -> str:
    return str(code or "").strip().lower().split("-", 1)[0]


# English NLTK name of an ISO code ("tr" -> "turkish"); "" until resolvable. NLTK's
# bundled prefix table covers the European codes; nltk.langnames (bcp47 data) the rest.
def nltk_language_name(code: str) -> str:
    code = base_code(code)
    if not code:
        return ""
    if code in _NAME_CACHE:
        return _NAME_CACHE[code]
    bundled = {c: n for n, c in NonbreakingPrefixesCorpusReader.available_langs.items() if len(n) > 2}
    # As in ASLM-Search: Arabic is implemented by NLTK Snowball but absent from its
    # prefix-code map, so it resolves without the bcp47 data too.
    if "arabic" in SnowballStemmer.languages:
        bundled.setdefault("ar", "arabic")
    name = bundled.get(code, "")
    if not name:
        from core.search.nltk_resources import ensure_resource

        if ensure_resource("bcp47"):
            try:
                from nltk.langnames import langname

                name = str(langname(code) or "").lower()
            except Exception:  # noqa: BLE001 - an unknown tag just means "no resources"
                name = ""
    if name:
        _NAME_CACHE[code] = name
    return name


# Script family of text: latin / cyrillic / arabic / devanagari / hangul / cjk / unknown.
def script_of(text: str) -> str:
    script = analyze_query(text).script
    return "cjk" if script == "kana" else script


# Script a project language is written in.
@lru_cache(maxsize=None)
def language_script(code: str) -> str:
    code = base_code(code)
    profile = next((p for p in LOCALE_PROFILES if p.language == code), None)
    return script_of(profile.label) if profile else "latin"


# Language whose rules apply to a token of `script` when the query language has a
# different script: the project default if it shares the script, else a language of
# that script with a Snowball stemmer, else the first one.
@lru_cache(maxsize=None)
def default_language_for_script(script: str) -> str:
    if language_script(DEFAULT_LANGUAGE) == script:
        return DEFAULT_LANGUAGE
    same = [code for code in PROJECT_LANGUAGES if language_script(code) == script]
    with_stemmer = [code for code in same if stemmer_for(code) is not None]
    return (with_stemmer or same or [DEFAULT_LANGUAGE])[0]


# NLTK Snowball stemmer for a language, or None when NLTK has none.
def stemmer_for(code: str) -> SnowballStemmer | None:
    name = nltk_language_name(code)
    if not name or name not in SnowballStemmer.languages:
        return None
    if name not in _STEMMER_CACHE:
        _STEMMER_CACHE[name] = SnowballStemmer(name)
    return _STEMMER_CACHE[name]



# Every Unicode Sentence_Terminal character, so Punkt splits sentences in any script.
@lru_cache(maxsize=1)
def sentence_terminators() -> tuple[str, ...]:
    text = "".join(chr(c) for c in range(0x30000) if not 0xD800 <= c <= 0xDFFF)
    return tuple(regex.findall(r"\p{Sentence_Terminal}", text))


# Full-width terminators (CJK "。！？") end a sentence without a following space.
@lru_cache(maxsize=1)
def _wide_terminator_re() -> re.Pattern[str]:
    wide = "".join(c for c in sentence_terminators() if unicodedata.east_asian_width(c) in "WF")
    return re.compile(f"[^{re.escape(wide)}]+[{re.escape(wide)}]*")


# Punkt tokenizer without trained data, using the Unicode sentence terminators.
def _untrained_punkt():
    from nltk.tokenize.punkt import PunktLanguageVars, PunktSentenceTokenizer

    # Punkt language variables that end sentences at any Unicode sentence terminator.
    class _UnicodeVars(PunktLanguageVars):
        sent_end_chars = sentence_terminators()

    return PunktSentenceTokenizer(lang_vars=_UnicodeVars())


# Punkt tokenizer trained for a language, loaded from the NLTK data.
def _trained_punkt(code: str):
    from nltk.tokenize.punkt import PunktTokenizer

    from core.search.nltk_resources import ensure_resource

    name = nltk_language_name(code)
    if not name or not ensure_resource("punkt_tab"):
        return None
    try:
        return PunktTokenizer(name)
    except (LookupError, OSError, ValueError):
        return None


# Punkt for a language: its trained model; else the trained model of the default language
# of its script (Ukrainian -> Russian, Vietnamese -> English); else untrained Punkt that
# ends sentences on every Unicode sentence terminator (Arabic "؟", Hindi "।", …).
def _punkt(code: str):
    code = base_code(code) or DEFAULT_LANGUAGE
    if code in _PUNKT_CACHE:
        return _PUNKT_CACHE[code]
    tokenizer = _trained_punkt(code)
    if tokenizer is None:
        fallback = default_language_for_script(language_script(code))
        tokenizer = _trained_punkt(fallback) if fallback != code else None
    if tokenizer is not None:
        _PUNKT_CACHE[code] = tokenizer
        return tokenizer
    return _PUNKT_CACHE.setdefault("_untrained", _untrained_punkt())


# (start, end) of every sentence in `text`.
def sentence_spans(text: str, language: str = DEFAULT_LANGUAGE) -> list[tuple[int, int]]:
    if not text:
        return []
    spans: list[tuple[int, int]] = []
    wide = _wide_terminator_re()
    for start, end in _punkt(language).span_tokenize(text):
        for match in wide.finditer(text, start, end):
            if text[match.start():match.end()].strip():
                spans.append((match.start(), match.end()))
    return spans



# NLTK stopword list of the language; empty when NLTK has none (hi, ja, ko, pl, uk, vi)
# or the corpus is not installed yet.
def stopwords_for(code: str) -> frozenset[str]:
    name = nltk_language_name(code)
    if not name:
        return frozenset()
    if name in _STOPWORDS_CACHE:
        return _STOPWORDS_CACHE[name]
    from core.search.nltk_resources import ensure_resource

    if not ensure_resource("stopwords"):
        return frozenset()
    from nltk.corpus import stopwords

    try:
        words = frozenset(word.casefold() for word in stopwords.words(name)) if name in stopwords.fileids() else frozenset()
    except (LookupError, OSError):
        return frozenset()
    _STOPWORDS_CACHE[name] = words
    return words



# Unique lowercase tokens of at least two characters from a query, split by the NLTK tokenizer.
def nltk_query_tokens(value: str) -> list[str]:
    return list(
        dict.fromkeys(token.casefold() for token in TOKENIZER.tokenize(value or "") if len(token) >= 2)
    )


# Whether the text is written in Chinese or Japanese characters.
def _is_cjk(text: str) -> bool:
    return script_of(text) == "cjk"


# Seed words as (language, words) with each word split into tokens; CJK seeds stay whole
# (they are matched as substrings: CJK queries have no spaces).
@lru_cache(maxsize=1)
def _seed_phrases() -> tuple[tuple[str, tuple[str, ...]], ...]:
    return tuple(
        (language, tuple(word.casefold().split()))
        for language, words in COMMERCE_SEEDS.items()
        for word in words
    )


# Whether a token is an inflected form of a seed word, by shared stem.
def _token_is_seed(token: str, seed: str, language: str) -> bool:
    from core.extract.stemming import stems

    return bool(stems(token, language) & stems(seed, language))


# Spans of commerce words in the query (any inflection, any project language).
def _commerce_spans(query: str) -> list[tuple[int, int]]:
    text = (query or "").casefold()
    tokens = [(m.group(0), m.start(), m.end()) for m in regex.finditer(r"[^\W_]+(?:[-_.][^\W_]+)*", text)]
    spans: list[tuple[int, int]] = []
    for language, seed in _seed_phrases():
        if len(seed) == 1 and _is_cjk(seed[0]):
            spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(seed[0]), text))
            continue
        n = len(seed)
        for i in range(len(tokens) - n + 1):
            if all(_token_is_seed(tokens[i + j][0], seed[j], language) for j in range(n)):
                spans.append((tokens[i][1], tokens[i + n - 1][2]))
    return spans


# 1.0 when the query carries an explicit commerce word in any inflection, else 0.0.
def commerce_intent_score(query: str, *, language: str = "", explicit_shopping: bool = False) -> float:
    if explicit_shopping:
        return 1.0
    return 1.0 if _commerce_spans(query) else 0.0


# Product-preview gate (ASLM-Search _preview_shopping_intent): an explicit commerce
# word, or a model-number shape - "rtx4090", "wh-1000xm5", "rtx 4090", an 8+ digit
# article/EAN - without any category lists.
def shopping_intent(query: str, *, language: str = "") -> bool:
    if commerce_intent_score(query, language=language) >= 1.0:
        return True
    tokens = TOKENIZER.tokenize((query or "").casefold())
    if any(any(ch.isalpha() for ch in token) and any(ch.isdigit() for ch in token) for token in tokens):
        return True
    if any(token.isdigit() and len(token) >= 8 for token in tokens):
        return True
    return any(
        token.isalpha() and 2 <= len(token) <= 6 and nxt.isdigit() and len(nxt) >= 3
        for token, nxt in zip(tokens, tokens[1:])
    )


# Product terms only: commerce words are cut out in any inflection and any project
# language (CJK seeds as substrings), so catalogs search for the product itself.
def catalog_query(query: str, *, language: str = "") -> str:
    text = query or ""
    folded = text.casefold()
    spans = sorted(_commerce_spans(text))
    if len(folded) != len(text):  # casefold changed lengths (rare): keep the query
        return text.strip()
    kept: list[str] = []
    position = 0
    for start, end in spans:
        if start >= position:
            kept.append(text[position:start])
            position = end
    kept.append(text[position:])
    cleaned = " ".join("".join(kept).split()).strip(" -,;:")
    return cleaned if len(regex.sub(r"\W", "", cleaned)) >= 2 else text.strip()
