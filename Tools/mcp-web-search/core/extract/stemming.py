# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import unicodedata
from functools import lru_cache

import regex

from core.search.nltk_languages import (
    base_code,
    default_language_for_script,
    language_script,
    script_of,
    stemmer_for,
)

# Word stems for read_page `find` and product relevance across the project's 20 locales,
# so the same word with different endings matches. Query and page go through the same
# function with the query language. Per token, by its script:
#   * the query language's rules if it is written in that script, else the default
#     language of the script (see nltk_languages.default_language_for_script);
#   * NLTK Snowball when the language has one (ar, de, en, es, fr, it, nl, pt, ru);
#   * otherwise truncation stemming - the first 5 letters (2 syllable blocks for Hangul),
#     the usual IR fallback for languages without a stemmer (hi, id, ko, pl, tr, uk, vi);
#   * Chinese / Japanese characters are single tokens, so a phrase is a substring match.

TOKEN_RE = regex.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}]|[^\W_\p{Han}\p{Hiragana}\p{Katakana}]+")

_TRUNCATE_LETTERS = 5
_TRUNCATE_HANGUL_SYLLABLES = 2


# NLTK's own Russian noun-suffix table and transliteration (RussianStemmer works on a
# romanized form internally). None if a future NLTK renames these internals.
@lru_cache(maxsize=1)
def _ru_noun_tables():
    stemmer = stemmer_for("ru")
    inner = getattr(stemmer, "stemmer", None)
    try:
        suffixes = sorted(inner._RussianStemmer__noun_suffixes, key=len, reverse=True)
        return inner._RussianStemmer__cyrillic_to_roman, inner._RussianStemmer__roman_to_cyrillic, suffixes
    except AttributeError:
        return None


# Russian Snowball also reads "-ен"/"-ан"/"-ал" as verb endings, so "токен" -> "ток" while
# "токенов" -> "токен". A second key with only Snowball's noun (case) ending removed
# reunites them.
def _ru_case_stem(word: str) -> str:
    tables = _ru_noun_tables()
    if tables is None:
        return word
    to_roman, to_cyrillic, suffixes = tables
    roman = to_roman(word)
    for suffix in suffixes:
        if roman.endswith(suffix):
            stem_ = to_cyrillic(roman[: -len(suffix)])
            if len(stem_) >= 3:
                return stem_
    return word


# Truncation keys: the first 5 letters, plus the word minus its last letter so short
# words whose ending changes still meet ("cena"/"ceny", "fiyat"/"fiyatı"). Combining vowel
# signs (Devanagari matras) do not count as letters; Hangul keeps 2 syllable blocks.
def _truncate(word: str, script: str) -> frozenset[str]:
    if script == "hangul":
        return frozenset({word[:_TRUNCATE_HANGUL_SYLLABLES]})
    letters = "".join(ch for ch in word if unicodedata.category(ch) not in {"Mn", "Mc"})
    keys = {letters[:_TRUNCATE_LETTERS]}
    if len(letters) >= 4:
        keys.add(letters[: min(_TRUNCATE_LETTERS, len(letters) - 1)])
    return frozenset(keys)


# Pick the language whose stemming rules apply to a token of the given script: the query language if it uses
# that script, else the script's default language.
def _rules_language(script: str, language: str) -> str:
    code = base_code(language)
    if code and language_script(code) == script:
        return code
    return default_language_for_script(script)


# Every stem a token may match on. Two tokens are the same word when their sets intersect.
@lru_cache(maxsize=131_072)
def stems(token: str, language: str = "") -> frozenset[str]:
    lowered = unicodedata.normalize("NFC", token).casefold().replace("ё", "е")
    # Identifiers with digits stay whole; vowel signs (Devanagari) are letters' marks.
    if any(ch.isdigit() for ch in lowered) or not any(ch.isalpha() for ch in lowered):
        return frozenset({lowered})
    script = script_of(lowered)
    if script in {"cjk", "unknown"}:
        return frozenset({lowered})
    code = _rules_language(script, language)
    stemmer = stemmer_for(code)
    if stemmer is None:
        return _truncate(lowered, script)
    keys = {stemmer.stem(lowered)}
    if code == "ru":
        keys.add(_ru_case_stem(lowered))
    return frozenset(keys)


# Return the shortest stem of a token.
def stem(token: str, language: str = "") -> str:
    return min(stems(token, language), key=len)


# Tokens of `text` as (stems, start, end); underscores split identifiers ("curl_cffi"),
# CJK characters are tokens of their own.
def stem_tokens(text: str, language: str = "") -> list[tuple[frozenset[str], int, int]]:
    return [(stems(m.group(0), language), m.start(), m.end()) for m in TOKEN_RE.finditer(text or "")]
