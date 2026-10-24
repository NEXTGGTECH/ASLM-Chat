# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest

from Apps.UI import locale_catalog


pytestmark = [pytest.mark.unit]


# Catalog discovery and locale resolution.


# Bundled locale catalogs are listed in sorted order.
def test_list_available_chat_locales_includes_bundled_catalogs():
    codes = locale_catalog.list_available_chat_locales()

    assert codes == sorted(codes)
    assert "en" in codes
    for expected in ("ru", "de", "ja", "zh-Hans"):
        assert expected in codes


# Supported host languages map onto their catalog files.
def test_resolve_effective_locale_maps_supported_languages():
    assert locale_catalog.resolve_effective_locale("ru") == "ru"
    assert locale_catalog.resolve_effective_locale("RU") == "ru"
    assert locale_catalog.resolve_effective_locale("de") == "de"


# Unsupported host languages fall back to English.
def test_resolve_effective_locale_falls_back_to_english():
    assert locale_catalog.resolve_effective_locale("klingon") == "en"
    assert locale_catalog.resolve_effective_locale(None) == "en"
    assert locale_catalog.resolve_effective_locale("") == "en"


# Catalog merging.


# Overlay values win while English-only sections survive the merge.
def test_load_catalog_merges_overlay_over_english_base():
    russian = locale_catalog.load_catalog("ru")

    # Overlay values win...
    assert russian["engines"]["openai"] == "Совместимый с OpenAI"
    # ...while English-only sections survive the merge.
    assert russian["engines"]["ollama-service"] == "Ollama"


# Deep merge keeps base branches and overlays nested values.
def test_deep_merge_prefers_overlay_and_keeps_base_branches():
    base = {"a": {"x": 1, "y": 2}, "b": "keep"}
    overlay = {"a": {"y": 3, "z": 4}}

    merged = locale_catalog._deep_merge(base, overlay)

    assert merged == {"a": {"x": 1, "y": 3, "z": 4}, "b": "keep"}


# The English catalog loads without an overlay layer.
def test_load_catalog_for_english_returns_base_catalog():
    base = locale_catalog.load_catalog("en")

    assert base["engines"]["openai"] == "OpenAI-Compatible"


# Translation.


# Nested keys resolve per locale.
def test_translate_resolves_nested_keys_per_locale():
    assert locale_catalog.translate("engines.openai", locale="en") == "OpenAI-Compatible"
    assert locale_catalog.translate("engines.openai", locale="ru") == "Совместимый с OpenAI"


# Missing overlay keys fall back to the English catalog.
def test_translate_falls_back_to_english_for_missing_overlay_keys(monkeypatch):
    catalogs = {
        "en": {"chat": {"title": "Untitled chat"}},
        "xx": {"chat": {}},
    }
    monkeypatch.setattr(locale_catalog, "resolve_effective_locale", lambda locale: locale or "en")
    monkeypatch.setattr(locale_catalog, "load_catalog", catalogs.get)

    # The overlay has no value; English provides it.
    assert locale_catalog.translate("chat.title", locale="xx") == "Untitled chat"


# Unknown paths use the fallback (with interpolation) and then the key itself.
def test_translate_uses_fallback_then_key_for_unknown_paths():
    assert locale_catalog.translate("missing.key", locale="en", fallback="Backup {place}", place=1) == "Backup 1"
    assert locale_catalog.translate("missing.key", locale="en", fallback="Backup") == "Backup"
    assert locale_catalog.translate("missing.key", locale="en") == "missing.key"


# Named placeholders interpolate into translated templates.
def test_translate_interpolates_named_params(monkeypatch):
    catalogs = {"en": {"common": {"greeting": "Hello {name}, you have {n} chats"}}}
    monkeypatch.setattr(locale_catalog, "resolve_effective_locale", lambda locale: "en")
    monkeypatch.setattr(locale_catalog, "load_catalog", catalogs.get)

    assert locale_catalog.translate("common.greeting", name="Sam", n=3) == "Hello Sam, you have 3 chats"


# Unknown placeholders stay intact instead of being dropped.
def test_translate_keeps_unknown_placeholders_intact(monkeypatch):
    catalogs = {"en": {"common": {"line": "Value {known} stays, {unknown} too"}}}
    monkeypatch.setattr(locale_catalog, "resolve_effective_locale", lambda locale: "en")
    monkeypatch.setattr(locale_catalog, "load_catalog", catalogs.get)

    assert locale_catalog.translate("common.line", known="A") == "Value A stays, {unknown} too"


# Plural forms.


# CLDR plural categories follow the language rules.
@pytest.mark.parametrize(
    ("count", "locale", "expected"),
    [
        (1, "ru", "one"),
        (2, "ru", "few"),
        (5, "ru", "many"),
        (11, "ru", "many"),
        (21, "ru", "one"),
        (25, "ru", "many"),
        (0, "ru", "many"),
        (1, "en", "one"),
        (2, "en", "other"),
        (0, "ja", "other"),
        (5, "ja", "other"),
        (0, "ar", "zero"),
        (1, "ar", "one"),
        (2, "ar", "two"),
        (5, "ar", "few"),
        (15, "ar", "many"),
        (100, "ar", "other"),
        (101, "ar", "other"),
    ],
)
def test_plural_categories_follow_cldr_rules(count, locale, expected):
    assert locale_catalog._plural_category(count, locale) == expected


# Russian plurals pick one/few/many forms by count.
def test_translate_plural_picks_russian_category_forms():
    assert locale_catalog.translate_plural("downloadsBridge.downloads", 1, locale="ru") == "1 загрузка"
    assert locale_catalog.translate_plural("downloadsBridge.downloads", 3, locale="ru") == "3 загрузки"
    assert locale_catalog.translate_plural("downloadsBridge.downloads", 7, locale="ru") == "7 загрузок"


# English plurals pick singular and plural forms.
def test_translate_plural_english_singular_and_plural():
    assert locale_catalog.translate_plural("downloadsBridge.downloads", 1, locale="en") == "1 download"
    assert locale_catalog.translate_plural("downloadsBridge.downloads", 9, locale="en") == "9 downloads"


# display_count overrides the number while keeping the grammar form.
def test_translate_plural_supports_display_count_override():
    rendered = locale_catalog.translate_plural(
        "downloadsBridge.downloads", 42900, locale="en", display_count="42.9K"
    )

    assert rendered == "42.9K downloads"


# A missing category form falls back to "other".
def test_translate_plural_falls_back_to_other_category(monkeypatch):
    catalogs = {"en": {"counts": {"staged": {"one": "One!", "other": "{count} items"}}}}
    monkeypatch.setattr(locale_catalog, "resolve_effective_locale", lambda locale: "en")
    monkeypatch.setattr(locale_catalog, "load_catalog", catalogs.get)
    # Simulate a language whose "few" form is missing from the catalog.
    monkeypatch.setattr(locale_catalog, "_plural_category", lambda count, locale: "few")

    rendered = locale_catalog.translate_plural("counts.staged", 3, locale="en")

    assert rendered == "3 items"


# With no forms at all the key itself is returned.
def test_translate_plural_returns_key_when_missing_everywhere(monkeypatch):
    catalogs = {"en": {}}
    monkeypatch.setattr(locale_catalog, "resolve_effective_locale", lambda locale: "en")
    monkeypatch.setattr(locale_catalog, "load_catalog", catalogs.get)

    assert locale_catalog.translate_plural("no.such.key", 1, locale="en") == "no.such.key"


# Client catalog embedding.


# The client catalog embeds the merged tree for the requested locale.
def test_catalog_for_js_returns_merged_catalog():
    payload = locale_catalog.catalog_for_js("ru")

    assert isinstance(payload, dict)
    assert "engines" in payload
    assert payload["engines"]["openai"] == "Совместимый с OpenAI"
