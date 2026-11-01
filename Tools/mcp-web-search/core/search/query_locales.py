# Copyright NEXTGGTECH. Elastic License 2.0.

"""Deterministic query-locale detection and result-language scoring.

Ported from ASLM-Search's query_analysis: stdlib-only (no nltk or model calls),
sub-millisecond per query. Two jobs:
- analyze_query() maps a raw query to one of the 20 supported locales using script
  ranges, distinctive diacritics and stop-word hints — this drives region routing
  so a French query searches fr-fr instead of us-en.
- detect_result_language() + language_mismatch_penalty() let triage gently prefer
  results written in the query's language without banning foreign-language pages.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True, slots=True)
class LocaleProfile:
    code: str
    language: str
    region: str
    label: str


LOCALE_PROFILES: tuple[LocaleProfile, ...] = (
    LocaleProfile("ar", "ar", "sa-ar", "العربية"),
    LocaleProfile("de", "de", "de-de", "Deutsch"),
    LocaleProfile("en", "en", "us-en", "English"),
    LocaleProfile("es", "es", "es-es", "Español"),
    LocaleProfile("fr", "fr", "fr-fr", "Français"),
    LocaleProfile("hi", "hi", "in-hi", "हिन्दी"),
    LocaleProfile("id", "id", "id-id", "Bahasa Indonesia"),
    LocaleProfile("it", "it", "it-it", "Italiano"),
    LocaleProfile("ja", "ja", "jp-ja", "日本語"),
    LocaleProfile("ko", "ko", "kr-ko", "한국어"),
    LocaleProfile("nl", "nl", "nl-nl", "Nederlands"),
    LocaleProfile("pl", "pl", "pl-pl", "Polski"),
    LocaleProfile("pt-BR", "pt", "br-pt", "Português (Brasil)"),
    LocaleProfile("pt", "pt", "pt-pt", "Português"),
    LocaleProfile("ru", "ru", "ru-ru", "Русский"),
    LocaleProfile("tr", "tr", "tr-tr", "Türkçe"),
    LocaleProfile("uk", "uk", "ua-uk", "Українська"),
    LocaleProfile("vi", "vi", "vn-vi", "Tiếng Việt"),
    LocaleProfile("zh-Hans", "zh", "cn-zh", "简体中文"),
    LocaleProfile("zh-Hant", "zh", "tw-zh", "繁體中文"),
)

_BY_CODE = {profile.code.lower(): profile for profile in LOCALE_PROFILES}
_DEFAULT_BY_LANGUAGE = {
    profile.language: profile for profile in reversed(LOCALE_PROFILES)
}
_DEFAULT_BY_LANGUAGE.update(
    {
        "en": _BY_CODE["en"],
        "pt": _BY_CODE["pt-br"],
        "zh": _BY_CODE["zh-hans"],
    }
)

_LATIN_HINTS: dict[str, frozenset[str]] = {
    "de": frozenset({"der", "die", "das", "ein", "eine", "und", "wie", "was", "für", "mit", "ohne", "kaufen"}),
    "es": frozenset({"el", "la", "los", "las", "una", "como", "cómo", "para", "con", "sin", "comprar", "mejor", "dónde", "qué", "más", "cuánto"}),
    "fr": frozenset({"le", "la", "les", "une", "des", "comment", "pour", "avec", "sans", "acheter", "meilleur"}),
    "id": frozenset({"yang", "dan", "untuk", "dengan", "tanpa", "cara", "beli", "terbaik", "harga"}),
    "it": frozenset({"il", "lo", "gli", "una", "come", "per", "con", "senza", "comprare", "migliore"}),
    "nl": frozenset({"de", "het", "een", "hoe", "voor", "met", "zonder", "kopen", "beste", "prijs"}),
    "pl": frozenset({"jak", "dla", "oraz", "bez", "kupić", "najlepszy", "cena", "jest", "nie"}),
    "pt": frozenset({"uma", "como", "para", "com", "sem", "comprar", "melhor", "onde", "não", "você", "preço", "quanto"}),
    "tr": frozenset({"bir", "ve", "için", "ile", "nasıl", "satın", "almak", "iyi", "fiyat"}),
    "vi": frozenset({"và", "của", "cho", "với", "không", "cách", "mua", "tốt", "nhất", "giá"}),
    "en": frozenset({"the", "a", "an", "how", "what", "for", "with", "without", "buy", "best", "price"}),
}

_DISTINCTIVE_LATIN: dict[str, frozenset[str]] = {
    "de": frozenset("äöüß"),
    "es": frozenset("ñ¿¡áíóú"),
    "fr": frozenset("æëîïœùûÿ"),
    "pl": frozenset("ąćęłńśźż"),
    "pt": frozenset("ãõáâê"),
    "tr": frozenset("ğış"),
    "vi": frozenset("ăđơưắằẳẵặớờởỡợứừửữự"),
}

_UKRAINIAN_CHARS = frozenset("іїєґ")
_UKRAINIAN_WORDS = frozenset({"як", "що", "для", "та", "або", "купити", "кращий", "ціна"})
_RUSSIAN_WORDS = frozenset({"как", "что", "для", "и", "или", "купить", "лучший", "цена"})
_TRADITIONAL_CHARS = frozenset("體國臺灣學術網頁搜尋購買與這為後發現點開關門書車電腦")

_TLD_LANGUAGES = {
    "de": "de", "at": "de", "es": "es", "fr": "fr", "id": "id", "it": "it",
    "jp": "ja", "kr": "ko", "nl": "nl", "pl": "pl", "br": "pt", "pt": "pt",
    "ru": "ru", "su": "ru", "tr": "tr", "ua": "uk", "vn": "vi", "cn": "zh", "tw": "zh",
}


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    language: str
    locale: str
    region: str
    confidence: float
    script: str


def locale_profile(code: str) -> LocaleProfile:
    return _BY_CODE.get(str(code or "").strip().lower(), _BY_CODE["en"])


def locale_for_language(language: str) -> LocaleProfile:
    return _DEFAULT_BY_LANGUAGE.get(str(language or "").lower(), _BY_CODE["en"])


def _letters(text: str) -> list[str]:
    return [character for character in text if character.isalpha()]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)


def _latin_language(text: str, tokens: list[str]) -> tuple[str, float]:
    scores = {language: 0.0 for language in _LATIN_HINTS}
    lowered = text.casefold()
    for language, characters in _DISTINCTIVE_LATIN.items():
        scores[language] += sum(character in characters for character in lowered) * 2.5
    for language, hints in _LATIN_HINTS.items():
        for token in tokens:
            # Plural tolerance: "mejores"/"precios" style suffixes must not hide
            # the base hint word from its language.
            if token in hints or (len(token) > 3 and token.rstrip("s") in hints):
                scores[language] += 1
    score = max(scores.values())
    if score <= 0:
        return "en", 0.35
    # Ties break toward English: short queries legitimately hit one hint in
    # several languages ("a" family collisions), and English is the web's
    # default — a wrong pt/es call routes the whole search to the wrong region.
    language = "en" if scores.get("en", 0.0) >= score else max(
        (key for key, value in scores.items() if value == score),
        # dict order is arbitrary; sort for determinism
        key=lambda key: (-scores[key], key),
    )
    runner_up = max((value for key, value in scores.items() if key != language), default=0.0)
    confidence = min(0.95, 0.5 + max(0.0, score - runner_up) / max(4.0, score + 1.0))
    return language, confidence


def analyze_query(text: str, *, locale_override: str = "") -> QueryAnalysis:
    normalized = unicodedata.normalize("NFKC", str(text or "")).strip()
    if locale_override:
        profile = locale_profile(locale_override)
        return QueryAnalysis(profile.language, profile.code, profile.region, 1.0, "configured")
    letters = _letters(normalized)
    if not letters:
        profile = _BY_CODE["en"]
        return QueryAnalysis(profile.language, profile.code, profile.region, 0.0, "unknown")
    tokens = _tokens(normalized)
    counts = {"arabic": 0, "cyrillic": 0, "devanagari": 0, "hangul": 0, "kana": 0, "cjk": 0}
    for character in letters:
        codepoint = ord(character)
        if 0x0600 <= codepoint <= 0x06FF or 0x0750 <= codepoint <= 0x077F or 0x08A0 <= codepoint <= 0x08FF:
            counts["arabic"] += 1
        elif 0x0400 <= codepoint <= 0x052F:
            counts["cyrillic"] += 1
        elif 0x0900 <= codepoint <= 0x097F:
            counts["devanagari"] += 1
        elif 0xAC00 <= codepoint <= 0xD7AF:
            counts["hangul"] += 1
        elif 0x3040 <= codepoint <= 0x30FF:
            counts["kana"] += 1
        elif 0x3400 <= codepoint <= 0x9FFF:
            counts["cjk"] += 1
    dominant_script, dominant_count = max(counts.items(), key=lambda item: item[1])
    ratio = dominant_count / max(1, len(letters))
    if counts["kana"] > 0 and (counts["kana"] + counts["cjk"]) / max(1, len(letters)) >= 0.15:
        profile = locale_profile("ja")
        confidence = min(0.99, 0.7 + (counts["kana"] + counts["cjk"]) / max(1, len(letters)) * 0.29)
        return QueryAnalysis("ja", profile.code, profile.region, confidence, "kana")
    if ratio >= 0.15:
        language, locale = "en", "en"
        if dominant_script == "arabic":
            language, locale = "ar", "ar"
        elif dominant_script == "devanagari":
            language, locale = "hi", "hi"
        elif dominant_script == "hangul":
            language, locale = "ko", "ko"
        elif dominant_script == "kana":
            language, locale = "ja", "ja"
        elif dominant_script == "cjk":
            traditional = sum(character in _TRADITIONAL_CHARS for character in normalized)
            language, locale = "zh", "zh-Hant" if traditional else "zh-Hans"
        elif dominant_script == "cyrillic":
            lowered = normalized.casefold()
            uk_score = sum(character in _UKRAINIAN_CHARS for character in lowered) * 2 + sum(token in _UKRAINIAN_WORDS for token in tokens)
            ru_score = sum(token in _RUSSIAN_WORDS for token in tokens)
            language, locale = ("uk", "uk") if uk_score > ru_score else ("ru", "ru")
        confidence = min(0.99, 0.65 + ratio * 0.34)
        profile = locale_profile(locale)
        return QueryAnalysis(language, profile.code, profile.region, confidence, dominant_script)
    language, confidence = _latin_language(normalized, tokens)
    profile = locale_for_language(language)
    return QueryAnalysis(language, profile.code, profile.region, confidence, "latin")


def detect_result_language(title: str, snippet: str = "", url: str = "") -> str:
    analysis = analyze_query(f"{title} {snippet}".strip())
    if analysis.script != "latin" or analysis.confidence >= 0.62:
        return analysis.language
    try:
        host = (urlsplit(url).hostname or "").lower()
    except ValueError:
        host = ""
    tld = host.rsplit(".", 1)[-1] if "." in host else ""
    return _TLD_LANGUAGES.get(tld, analysis.language)


def language_mismatch_penalty(query_language: str, result_language: str, weight: float) -> float:
    """Soft penalty for a result written in a different language than the query.

    English results never pay (it is the web's lingua franca and the fallback for
    every locale), and a confident same-language match never pays either.
    """
    query = str(query_language or "en").lower()
    result = str(result_language or "").lower()
    if not result or not weight or result == query or result == "en":
        return 0.0
    return max(0.0, min(0.3, float(weight)))
