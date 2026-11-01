# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest

from Apps.UI import host_theme_bridge


pytestmark = [pytest.mark.unit]


@pytest.fixture
def host_theme_payload(monkeypatch):
    def install(payload):
        monkeypatch.setattr(host_theme_bridge, "load_host_theme", lambda: payload)
        return payload

    return install


# MAUI color normalization.


# Host color formats convert to CSS colors.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("#fff", "#ffffff"),
        ("#00FF00", "#00ff00"),
        # MAUI ToHex order is AARRGGBB: opaque blue, then opaque red.
        ("#FF0000FF", "#0000ff"),
        ("#FFFF0000", "#ff0000"),
        ("#80FF0000", "rgba(255, 0, 0, 0.502)"),
        ("  #abc  ", "#aabbcc"),
    ],
)
def test_normalize_color_to_css_converts_host_formats(raw, expected):
    assert host_theme_bridge.normalize_color_to_css(raw) == expected


# Invalid color values never reach the CSS output.
@pytest.mark.parametrize(
    "raw",
    [None, "", "#12345", "rgb(1,2,3)", "not-a-color", "#GGGGGG"],
)
def test_normalize_color_to_css_rejects_invalid_values(raw):
    assert host_theme_bridge.normalize_color_to_css(raw) is None


# Host palette resolution.


# Palette keys map to CSS variables and teal aliases to cyan.
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


# A snapshot without colors yields an empty palette but keeps the theme.
def test_resolve_host_css_variables_without_colors_is_empty(host_theme_payload):
    host_theme_payload({"theme": "light"})

    resolved, theme = host_theme_bridge.resolve_host_css_variables()

    assert resolved == {}
    assert theme == "light"


# Without a snapshot the palette is empty and the theme defaults to dark.
def test_resolve_host_css_variables_defaults_to_dark_without_snapshot(monkeypatch):
    monkeypatch.setattr(host_theme_bridge, "load_host_theme", lambda: None)

    resolved, theme = host_theme_bridge.resolve_host_css_variables()

    assert resolved == {}
    assert theme == "dark"


# Luminance-driven surface inference.


# CSS colors parse into sRGB channels.
def test_css_color_to_srgb_channels_parses_css_colors():
    assert host_theme_bridge.css_color_to_srgb_channels("#ffffff") == (1.0, 1.0, 1.0)
    assert host_theme_bridge.css_color_to_srgb_channels("#000000") == (0.0, 0.0, 0.0)
    assert host_theme_bridge.css_color_to_srgb_channels("rgba(255, 0, 0, 0.5)") == (1.0, 0.0, 0.0)
    assert host_theme_bridge.css_color_to_srgb_channels("junk") is None


# Light/dark activity surfaces follow the canvas luminance, not just the theme.
def test_infer_prefer_light_activity_surfaces_uses_canvas_luminance():
    white = {"--c-bg": "#ffffff"}
    dark = {"--c-bg": "#1c1c1e"}

    assert host_theme_bridge.infer_prefer_light_activity_surfaces(white, "dark") is True
    assert host_theme_bridge.infer_prefer_light_activity_surfaces(dark, "dark") is False
    # Without a resolvable background the theme string decides.
    assert host_theme_bridge.infer_prefer_light_activity_surfaces({}, "light") is True
    assert host_theme_bridge.infer_prefer_light_activity_surfaces({}, "dark") is False


# Template context for the chat page.


# The page context carries the :root block and script-safe metadata.
def test_template_context_contains_root_block_and_metadata(host_theme_payload):
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

    context = host_theme_bridge.build_host_theme_template_context()

    assert context["host_theme_available"] is True
    assert context["host_theme_effective"] == "dark"

    css = context["host_theme_css_variables"]
    assert css.startswith(":root {")
    assert "--c-bg: #1c1c1e;" in css
    assert "--c-text: #ffffff;" in css

    # The embedded JSON is script-tag safe: angle brackets are escaped.
    assert "<City>" not in context["host_theme_json"]
    assert "custom-1" in context["host_theme_json"]


# Without a snapshot the context falls back to dark defaults.
def test_template_context_falls_back_when_snapshot_missing(monkeypatch):
    monkeypatch.setattr(host_theme_bridge, "load_host_theme", lambda: None)

    context = host_theme_bridge.build_host_theme_template_context()

    assert context == {
        "host_theme_available": False,
        "host_theme_effective": "dark",
        "host_theme_css_variables": "",
        "host_theme_json": "{}",
    }


# Preview document tokens for offline HTML shells.


# Without a host theme the preview tokens are the documented dark defaults.
def test_preview_document_theme_defaults_without_host(monkeypatch):
    monkeypatch.setattr(host_theme_bridge, "load_host_theme", lambda: None)

    tokens = host_theme_bridge.resolve_preview_document_theme()

    assert tokens["color_scheme"] == "dark"
    assert tokens["bg"] == "#1c1c1e"
    assert len(tokens["fingerprint"]) == 12

    # The fingerprint is deterministic for identical tokens.
    again = host_theme_bridge.resolve_preview_document_theme()
    assert again["fingerprint"] == tokens["fingerprint"]


# A host palette drives every preview token.
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


# A changed palette produces a different cache fingerprint.
def test_preview_fingerprint_changes_with_palette(host_theme_payload):
    host_theme_payload({"theme": "dark", "colors": {"BackgroundPrimary": "#1c1c1e"}})
    first = host_theme_bridge.resolve_preview_document_theme()["fingerprint"]

    host_theme_payload({"theme": "dark", "colors": {"BackgroundPrimary": "#2c2c2e"}})
    second = host_theme_bridge.resolve_preview_document_theme()["fingerprint"]

    assert first != second
