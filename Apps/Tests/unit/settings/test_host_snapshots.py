# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json

import pytest

from Settings import host_locale, host_theme


pytestmark = [pytest.mark.unit]


@pytest.fixture
def isolated_locale_file(tmp_path, monkeypatch):
    path = tmp_path / "host_locale.json"
    monkeypatch.setattr(host_locale, "HOST_LOCALE_FILE", path)
    return path


@pytest.fixture
def isolated_theme_file(tmp_path, monkeypatch):
    path = tmp_path / "host_theme.json"
    monkeypatch.setattr(host_theme, "HOST_THEME_FILE", path)
    return path


# Locale normalization.


# Language normalization maps host codes onto supported values.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("ru", "ru"),
        ("RU", "ru"),
        ("pt-br", "pt-BR"),
        ("zh-hans", "zh-Hans"),
        ("klingon", "en"),
        ("", "en"),
        (None, "en"),
        ("   ", "en"),
    ],
)
def test_normalize_host_language(raw, expected):
    assert host_locale.normalize_host_language(raw) == expected


# Locale snapshots.


# The locale snapshot round-trips through disk.
def test_host_locale_roundtrip(isolated_locale_file):
    payload = {"language": "de", "displayName": "Deutsch"}

    host_locale.save_host_locale_payload(payload)

    assert isolated_locale_file.is_file()
    assert host_locale.load_host_locale() == payload
    assert host_locale.get_language() == "de"
    assert host_locale.get_display_name() == "Deutsch"


# Atomic writes leave no temporary file behind.
def test_atomic_write_leaves_no_temp_file(isolated_locale_file):
    host_locale.save_host_locale_payload({"language": "en"})

    assert not isolated_locale_file.with_name("host_locale.json.tmp").exists()


# Rejecting non-dict payloads keeps the previous snapshot intact.
def test_save_host_locale_rejects_non_dict(isolated_locale_file):
    host_locale.save_host_locale_payload({"language": "ru"})

    with pytest.raises(TypeError):
        host_locale.save_host_locale_payload(["not", "a", "dict"])

    assert host_locale.get_language() == "ru"


# A missing snapshot degrades to English defaults.
def test_missing_locale_snapshot_defaults_to_english(isolated_locale_file):
    assert host_locale.load_host_locale() is None
    assert host_locale.get_language() == "en"
    assert host_locale.get_display_name() is None


# A corrupt or non-object snapshot file is ignored rather than raised.
@pytest.mark.parametrize("content", ["{broken", "[1, 2]", '"text"'])
def test_invalid_locale_snapshot_is_ignored(isolated_locale_file, content):
    isolated_locale_file.write_text(content, encoding="utf-8")

    assert host_locale.load_host_locale() is None
    assert host_locale.get_language() == "en"


# A UTF-8 BOM prefix (written by some host processes) is tolerated.
def test_locale_snapshot_with_bom_is_parsed(isolated_locale_file):
    isolated_locale_file.write_text(
        "﻿" + json.dumps({"language": "fr", "displayName": "Français"}),
        encoding="utf-8",
    )

    assert host_locale.get_language() == "fr"
    assert host_locale.get_display_name() == "Français"


# An empty displayName is reported as absent.
def test_blank_display_name_is_absent(isolated_locale_file):
    host_locale.save_host_locale_payload({"language": "en", "displayName": "   "})

    assert host_locale.get_display_name() is None


# Theme snapshots.


# The theme snapshot round-trips and degrades gracefully like the locale one.
def test_host_theme_roundtrip_and_missing_file(isolated_theme_file):
    assert host_theme.load_host_theme() is None

    payload = {"theme": "dark", "colors": {"accent": "#3b82f6"}}
    host_theme.save_host_theme_payload(payload)

    assert host_theme.load_host_theme() == payload


# Non-dict theme payloads are rejected outright.
def test_host_theme_rejects_non_dict(isolated_theme_file):
    with pytest.raises(TypeError):
        host_theme.save_host_theme_payload("dark")

    assert host_theme.load_host_theme() is None


# A corrupt theme snapshot is ignored rather than raised.
def test_invalid_theme_snapshot_is_ignored(isolated_theme_file):
    isolated_theme_file.write_text("{broken", encoding="utf-8")

    assert host_theme.load_host_theme() is None
