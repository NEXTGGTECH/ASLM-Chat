# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest

from Apps.UI import host_theme_bridge


pytestmark = [pytest.mark.unit]


# Provide a function that makes the host theme loader return a given snapshot payload.
@pytest.fixture
def host_theme_payload(monkeypatch):
    # Replace the host theme loader with one that returns the payload, and return the payload.
    def install(payload):
        monkeypatch.setattr(host_theme_bridge, "load_host_theme", lambda: payload)
        return payload

    return install


# MAUI color normalization

# Verify that host color formats (short and long hex, MAUI AARRGGBB with or without transparency, surrounding
# spaces) are converted to CSS colors. MAUI ToHex order is AARRGGBB, so #FF0000FF is opaque blue and #FFFF0000
# is opaque red.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("#fff", "#ffffff"),
        ("#00FF00", "#00ff00"),
        ("#FF0000FF", "#0000ff"),
        ("#FFFF0000", "#ff0000"),
        ("#80FF0000", "rgba(255, 0, 0, 0.502)"),
        ("  #abc  ", "#aabbcc"),
    ],
)
def test_normalize_color_to_css_converts_host_formats(raw, expected):
    assert host_theme_bridge.normalize_color_to_css(raw) == expected


# Verify that invalid color values (empty, wrong length, rgb() notation, text, non-hex digits) are rejected with
# None.
@pytest.mark.parametrize(
    "raw",
    [None, "", "#12345", "rgb(1,2,3)", "not-a-color", "#GGGGGG"],
)
def test_normalize_color_to_css_rejects_invalid_values(raw):
    assert host_theme_bridge.normalize_color_to_css(raw) is None


# Host palette resolution

# Verify that known palette keys map to CSS variables with normalized colors, teal is also exposed under the
# legacy cyan alias, and unknown or invalid colors are left out.
def test_resolve_host_css_variables_maps_and_aliases(host_theme_payload):
    host_theme_payload(
        {
            "theme": "dark",
            "colors": {
                "BackgroundPrimary": "#1c1c1e",
                "SystemTeal": "#30b0c7",
                "LabelPrimary": "#FFFFFF",
                "NotARealKey": "#123456",
                "Broken": "oops",
            },
        }
    )

    resolved, theme = host_theme_bridge.resolve_host_css_variables()

    assert theme == "dark"
    assert resolved["--c-bg"] == "#1c1c1e"
    assert resolved["--c-text"] == "#ffffff"

    # Teal is additionally exposed under the legacy cyan alias.
    assert resolved["--c-system-cyan"] == resolved["--c-system-teal"]

    # Unknown and invalid host colors never reach the CSS output.
    assert all("NotARealKey" not in var for var in resolved)


# Verify that a snapshot without colors gives an empty palette but keeps the theme name.
def test_resolve_host_css_variables_without_colors_is_empty(host_theme_payload):
    host_theme_payload({"theme": "light"})

    resolved, theme = host_theme_bridge.resolve_host_css_variables()

    assert resolved == {}
    assert theme == "light"


# Verify that without a snapshot the palette is empty and the theme defaults to dark.
def test_resolve_host_css_variables_defaults_to_dark_without_snapshot(monkeypatch):
    monkeypatch.setattr(host_theme_bridge, "load_host_theme", lambda: None)

    resolved, theme = host_theme_bridge.resolve_host_css_variables()

    assert resolved == {}
    assert theme == "dark"


# Luminance-driven surface inference

# Verify that hex and rgba CSS colors are parsed into sRGB channels between 0 and 1 (alpha ignored) and
# unparsable text gives None.
def test_css_color_to_srgb_channels_parses_css_colors():
    assert host_theme_bridge.css_color_to_srgb_channels("#ffffff") == (1.0, 1.0, 1.0)
    assert host_theme_bridge.css_color_to_srgb_channels("#000000") == (0.0, 0.0, 0.0)
    assert host_theme_bridge.css_color_to_srgb_channels("rgba(255, 0, 0, 0.5)") == (1.0, 0.0, 0.0)
    assert host_theme_bridge.css_color_to_srgb_channels("junk") is None


# Verify that light or dark activity surfaces follow the luminance of the canvas background rather than the
# theme name, and that the theme name decides only when no background color can be resolved.
def test_infer_prefer_light_activity_surfaces_uses_canvas_luminance():
    white = {"--c-bg": "#ffffff"}
    dark = {"--c-bg": "#1c1c1e"}

    assert host_theme_bridge.infer_prefer_light_activity_surfaces(white, "dark") is True
    assert host_theme_bridge.infer_prefer_light_activity_surfaces(dark, "dark") is False

    # Without a resolvable background the theme string decides.
    assert host_theme_bridge.infer_prefer_light_activity_surfaces({}, "light") is True
    assert host_theme_bridge.infer_prefer_light_activity_surfaces({}, "dark") is False


# Template context for the chat page

# Verify that the page context carries the availability flag, the effective theme, a :root block with the CSS
# variables and script-tag-safe JSON metadata.
def test_template_context_contains_root_block_and_metadata(host_theme_payload):
    # Arrange: a dark snapshot with a custom theme whose name contains angle brackets.
    host_theme_payload(
        {
            "theme": "dark",
            "appearance": "dark",
            "customThemeId": "custom-1",
            "customThemeName": "Night <City>",
            "colors": {
                "BackgroundPrimary": "#1c1c1e",
                "LabelPrimary": "#ffffff",
            },
        }
    )

    # Act: build the template context.
    context = host_theme_bridge.build_host_theme_template_context()

    # Assert: flags, CSS variables and the embedded JSON.
    assert context["host_theme_available"] is True
    assert context["host_theme_effective"] == "dark"

    css = context["host_theme_css_variables"]
    assert css.startswith(":root {")
    assert "--c-bg: #1c1c1e;" in css
    assert "--c-text: #ffffff;" in css

    # The embedded JSON is script-tag safe: angle brackets are escaped.
    assert "<City>" not in context["host_theme_json"]
    assert "custom-1" in context["host_theme_json"]


# Verify that without a snapshot the template context reports the theme as unavailable and falls back to the
# dark theme with no CSS variables and empty JSON.
def test_template_context_falls_back_when_snapshot_missing(monkeypatch):
    monkeypatch.setattr(host_theme_bridge, "load_host_theme", lambda: None)

    context = host_theme_bridge.build_host_theme_template_context()

    assert context == {
        "host_theme_available": False,
        "host_theme_effective": "dark",
        "host_theme_css_variables": "",
        "host_theme_json": "{}",
    }


# Preview document tokens for offline HTML shells

# Verify that without a host theme the preview tokens are the dark defaults and carry a 12-character fingerprint
# that is the same on every call.
def test_preview_document_theme_defaults_without_host(monkeypatch):
    monkeypatch.setattr(host_theme_bridge, "load_host_theme", lambda: None)

    tokens = host_theme_bridge.resolve_preview_document_theme()

    assert tokens["color_scheme"] == "dark"
    assert tokens["bg"] == "#1c1c1e"
    assert len(tokens["fingerprint"]) == 12

    # The fingerprint is deterministic for identical tokens.
    again = host_theme_bridge.resolve_preview_document_theme()
    assert again["fingerprint"] == tokens["fingerprint"]


# Verify that the host palette determines the color scheme and the background, surface, border, text and link
# preview tokens.
def test_preview_document_theme_follows_host_palette(host_theme_payload):
    host_theme_payload(
        {
            "theme": "light",
            "colors": {
                "BackgroundPrimary": "#f2f2f7",
                "BackgroundSecondary": "#ffffff",
                "BackgroundTertiary": "#e5e5ea",
                "SystemGray4": "#d1d1d6",
                "Separator": "#c6c6c8",
                "LabelPrimary": "#000000",
                "LinkColor": "#007aff",
            },
        }
    )

    tokens = host_theme_bridge.resolve_preview_document_theme()

    assert tokens["color_scheme"] == "light"
    assert tokens["bg"] == "#ffffff"
    assert tokens["surface"] == "#e5e5ea"
    assert tokens["surface_strong"] == "#d1d1d6"
    assert tokens["border"] == "#c6c6c8"
    assert tokens["text"] == "#000000"
    assert tokens["link"] == "#007aff"


# Verify that a changed palette gives a different preview fingerprint, so caches are invalidated.
def test_preview_fingerprint_changes_with_palette(host_theme_payload):
    host_theme_payload({"theme": "dark", "colors": {"BackgroundPrimary": "#1c1c1e"}})
    first = host_theme_bridge.resolve_preview_document_theme()["fingerprint"]

    host_theme_payload({"theme": "dark", "colors": {"BackgroundPrimary": "#2c2c2e"}})
    second = host_theme_bridge.resolve_preview_document_theme()["fingerprint"]

    assert first != second
