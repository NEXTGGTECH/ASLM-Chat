# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from Settings.host_locale import get_language, normalize_host_language

logger = logging.getLogger(__name__)

LOCALES_DIR = Path(__file__).resolve().parent / "locales"
BASE_LOCALE = "en"
_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


# Return the on-disk path for one locale catalog file.
def _locale_file_path(locale: str) -> Path:
    return LOCALES_DIR / f"{locale}.json"


# Return locale codes that have a catalog file on disk.
def list_available_chat_locales() -> list[str]:
    if not LOCALES_DIR.is_dir():
        return [BASE_LOCALE]
    codes = sorted(
        path.stem
        for path in LOCALES_DIR.glob("*.json")
        if path.is_file() and not path.name.endswith(".tmp")
    )
    return codes or [BASE_LOCALE]


# Map host language to a Chat catalog file, falling back to English.
def resolve_effective_locale(host_language: str | None) -> str:
    normalized = normalize_host_language(host_language)
    if _locale_file_path(normalized).is_file():
        return normalized
    return BASE_LOCALE


# Resolve effective locale from snapshot.
def resolve_effective_locale_from_snapshot() -> str:
    return resolve_effective_locale(get_language())


# Load one locale JSON catalog from disk with caching.
@lru_cache(maxsize=32)
def _load_raw_catalog(locale: str) -> dict[str, Any]:
    path = _locale_file_path(locale)
    if not path.is_file():
        return {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read locale file %s: %s", path, exc)
        return {}
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("Could not parse locale file %s: %s", path, exc)
        return {}
    return raw if isinstance(raw, dict) else {}


# Return merged messages for ``locale`` with English as the base layer.
def load_catalog(locale: str) -> dict[str, Any]:
    base = _load_raw_catalog(BASE_LOCALE)
    if locale == BASE_LOCALE:
        return base
    overlay = _load_raw_catalog(locale)
    return _deep_merge(base, overlay)


# Deep merge.
def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


# Lookup nested.
def _lookup_nested(catalog: dict[str, Any], key: str) -> Any | None:
    current: Any = catalog
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


# Interpolate.
def _interpolate(template: str, params: dict[str, Any]) -> str:
    # Replace one `{name}` placeholder during interpolation.
    def repl(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in params:
            return match.group(0)
        return str(params[name])

    return _PLACEHOLDER_RE.sub(repl, template)


# Resolve a dot-path key with optional ``{name}`` placeholders.
def translate(key: str, *, locale: str | None = None, fallback: str | None = None, **params: Any) -> str:
    effective = resolve_effective_locale(locale) if locale else resolve_effective_locale_from_snapshot()
    catalog = load_catalog(effective)
    value = _lookup_nested(catalog, key)
    if value is None and effective != BASE_LOCALE:
        value = _lookup_nested(load_catalog(BASE_LOCALE), key)
    if value is None:
        if fallback is not None:
            return _interpolate(fallback, params) if params else fallback
        return key
    if not isinstance(value, str):
        return str(value)
    if params:
        return _interpolate(value, params)
    return value


# CLDR cardinal categories for whole-number counts in the host's supported languages.
# https://www.unicode.org/cldr/charts/latest/supplemental/language_plural_rules.html
def _plural_category(count: int, locale: str) -> str:
    n = abs(count)
    if locale in {"id", "ja", "ko", "vi", "zh-Hans", "zh-Hant"}:
        return "other"
    if locale == "ar":
        if n == 0:
            return "zero"
        if n == 1:
            return "one"
        if n == 2:
            return "two"
        if 3 <= n % 100 <= 10:
            return "few"
        if 11 <= n % 100 <= 99:
            return "many"
        return "other"
    if locale in {"ru", "uk", "pl"}:
        if (n == 1 if locale == "pl" else n % 10 == 1 and n % 100 != 11):
            return "one"
        if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
            return "few"
        return "many"
    if locale in {"es", "fr", "it", "pt", "pt-BR"} and n != 0 and n % 1_000_000 == 0:
        return "many"
    # The pt catalog uses European Portuguese; only pt-BR treats zero as singular.
    if locale in {"fr", "hi", "pt-BR"}:
        return "one" if n in {0, 1} else "other"
    return "one" if n == 1 else "other"


# Translate a count without forcing languages into English singular/plural grammar.
# display_count preserves provider abbreviations such as 42.9K independently of agreement.
def translate_plural(
    key: str,
    count: int,
    *,
    locale: str | None = None,
    display_count: str | None = None,
) -> str:
    effective = resolve_effective_locale(locale) if locale else resolve_effective_locale_from_snapshot()
    catalog = load_catalog(effective)
    category = _plural_category(count, effective)
    value = _lookup_nested(catalog, f"{key}.{category}")
    if not isinstance(value, str):
        value = _lookup_nested(catalog, f"{key}.other")
    if not isinstance(value, str):
        return key
    return _interpolate(value, {"count": count if display_count is None else display_count})


# Return the merged catalog tree embedded in pages for client-side ``t()``.
def catalog_for_js(locale: str | None = None) -> dict[str, Any]:
    effective = resolve_effective_locale(locale) if locale else resolve_effective_locale_from_snapshot()
    return load_catalog(effective)
