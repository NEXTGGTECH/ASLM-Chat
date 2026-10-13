# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import re
from typing import Any, Final

from Settings.host_theme import load_host_theme

# Sync with ASLM ThemePaletteResolver / Colors.xaml when adding host palette keys.
_ASLM_COLOR_KEY_TO_CSS_VAR: Final[dict[str, str]] = {
    "SystemBlue": "--c-system-blue",
    "SystemGreen": "--c-system-green",
    "SystemIndigo": "--c-system-indigo",
    "SystemOrange": "--c-system-orange",
    "SystemPink": "--c-system-pink",
    "SystemPurple": "--c-system-purple",
    "SystemRed": "--c-system-red",
    "SystemTeal": "--c-system-teal",
    "SystemYellow": "--c-system-yellow",
    "SystemMint": "--c-system-mint",
    "SystemGray": "--c-gray-1",
    "SystemGray2": "--c-gray-2",
    "SystemGray3": "--c-gray-3",
    "SystemGray4": "--c-gray-4",
    "SystemGray5": "--c-gray-5",
    "SystemGray6": "--c-gray-6",
    "BackgroundPrimary": "--c-bg",
    "BackgroundSecondary": "--c-bg-surface",
    "BackgroundTertiary": "--c-bg-elevated",
    "LabelPrimary": "--c-text",
    "LabelSecondary": "--c-text-muted",
    "Separator": "--c-border",
    "LinkColor": "--c-link",
    "SystemBlueOverlay": "--c-overlay-blue",
    "White": "--c-white",
    "Black": "--c-black",
    "ActionRed": "--c-danger",
    "ActionBlue": "--c-primary",
    "ActionGreen": "--c-success",
    "OverlayBackground": "--c-overlay-scrim",
}

_HEX_RE = re.compile(r"^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$")
_RGBA_FN_RE = re.compile(
    r"^rgba\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*([01]|0?\.\d+)\s*\)$",
    re.IGNORECASE,
)

# sRGB relative luminance (WCAG). Used to pick activity/tool card contrast from the host canvas.
_LIGHT_SURFACE_LUMINANCE_THRESHOLD: Final[float] = 0.45


# Convert MAUI / ASLM hex strings to a CSS color (``#rrggbb`` or ``rgba(...)``).
def normalize_color_to_css(raw: str | None) -> str | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    if not _HEX_RE.match(s):
        return None
    body = s[1:]
    if len(body) == 3:
        r = int(body[0] + body[0], 16)
        g = int(body[1] + body[1], 16)
        b = int(body[2] + body[2], 16)
        return f"#{r:02x}{g:02x}{b:02x}"
    if len(body) == 6:
        return f"#{body.lower()}"
    # #AARRGGBB (MAUI ToHex)
    aa = int(body[0:2], 16)
    rr = int(body[2:4], 16)
    gg = int(body[4:6], 16)
    bb = int(body[6:8], 16)
    if aa == 255:
        return f"#{rr:02x}{gg:02x}{bb:02x}"
    alpha = round(aa / 255, 4)
    return f"rgba({rr}, {gg}, {bb}, {alpha})"


# Effective theme.
def _effective_theme(payload: dict[str, Any]) -> str:
    theme = str(payload.get("theme") or "").strip().lower()
    if theme in {"light", "dark"}:
        return theme
    appearance = str(payload.get("appearance") or "").strip().lower()
    if appearance == "light":
        return "light"
    return "dark"


# Srgb channel to linear.
def _srgb_channel_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# WCAG relative luminance for sRGB channels in 0..1.
def _relative_luminance_srgb(r: float, g: float, b: float) -> float:
    return (
        0.2126 * _srgb_channel_to_linear(r)
        + 0.7152 * _srgb_channel_to_linear(g)
        + 0.0722 * _srgb_channel_to_linear(b)
    )


# Parse ``#rrggbb`` (from :func:`normalize_color_to_css`) or ``rgba(r,g,b,a)`` into sRGB 0..1.
def css_color_to_srgb_channels(css: str) -> tuple[float, float, float] | None:
    s = str(css or "").strip()
    if not s:
        return None
    if s.startswith("#") and len(s) == 7:
        try:
            r = int(s[1:3], 16) / 255.0
            g = int(s[3:5], 16) / 255.0
            b = int(s[5:7], 16) / 255.0
        except ValueError:
            return None
        return (r, g, b)
    m = _RGBA_FN_RE.match(s)
    if not m:
        return None
    r_i, g_i, b_i = (int(m.group(i)) for i in range(1, 4))
    if not (0 <= r_i <= 255 and 0 <= g_i <= 255 and 0 <= b_i <= 255):
        return None
    return (r_i / 255.0, g_i / 255.0, b_i / 255.0)


# Choose light vs dark activity surfaces from the resolved host canvas (not only ``theme``).
def infer_prefer_light_activity_surfaces(resolved: dict[str, str], fallback_theme: str) -> bool:
    bg = resolved.get("--c-bg")
    ch = css_color_to_srgb_channels(bg) if bg else None
    if ch is not None:
        return _relative_luminance_srgb(*ch) >= _LIGHT_SURFACE_LUMINANCE_THRESHOLD
    return str(fallback_theme or "").strip().lower() == "light"


# Return keys for ``base.html``: CSS variable block, color-scheme, optional JSON.
def build_host_theme_template_context() -> dict[str, Any]:
    raw = load_host_theme()
    if not isinstance(raw, dict):
        return _empty_context()

    resolved, theme = resolve_host_css_variables()
    if not resolved:
        return _empty_context()

    prefer_light_activity = infer_prefer_light_activity_surfaces(resolved, theme)

    declarations = [f"  {var}: {value};" for var, value in resolved.items()]

    declarations.extend(_derived_theme_declarations(prefer_light_activity))

    inner = "\n".join(declarations)
    host_theme_css_variables = f":root {{\n{inner}\n}}"

    meta = {
        "theme": theme,
        "appearance": str(raw.get("appearance") or ""),
        "customThemeId": raw.get("customThemeId"),
        "customThemeName": raw.get("customThemeName"),
    }
    safe_json = _json_for_script_tag(meta)

    return {
        "host_theme_available": True,
        "host_theme_effective": theme,
        "host_theme_css_variables": host_theme_css_variables,
        "host_theme_json": safe_json,
    }


# Empty context.
def _empty_context() -> dict[str, Any]:
    return {
        "host_theme_available": False,
        "host_theme_effective": "dark",
        "host_theme_css_variables": "",
        "host_theme_json": "{}",
    }


# Return semantic UI variables that should follow the resolved host palette.
def _derived_theme_declarations(prefer_light_surfaces: bool) -> list[str]:
    if prefer_light_surfaces:
        activity_card_bg = "var(--surface-secondary)"
        activity_card_bg_hover = "color-mix(in srgb, var(--surface-secondary) 94%, var(--c-primary) 6%)"
        activity_card_subtle_bg = "var(--surface-secondary)"
        activity_card_hover_subtle_bg = activity_card_bg_hover
        activity_inner_bg = "color-mix(in srgb, var(--c-text) 4.5%, transparent)"
        activity_inner_bg_strong = "color-mix(in srgb, var(--c-text) 6.5%, transparent)"
        activity_card_border = "color-mix(in srgb, var(--surface-secondary) 94%, var(--c-text) 6%)"
        activity_card_border_hover = "color-mix(in srgb, var(--surface-secondary) 86%, var(--c-primary) 14%)"
        activity_monochrome_svg_filter = "brightness(0) saturate(100%)"
    else:
        activity_card_bg = "color-mix(in srgb, var(--c-text) 4.5%, transparent)"
        activity_card_bg_hover = "color-mix(in srgb, var(--c-text) 7%, transparent)"
        activity_card_subtle_bg = "color-mix(in srgb, var(--surface-secondary) 22%, transparent)"
        activity_card_hover_subtle_bg = "color-mix(in srgb, var(--surface-tertiary) 22%, transparent)"
        activity_inner_bg = "color-mix(in srgb, var(--c-black) 18%, transparent)"
        activity_inner_bg_strong = "color-mix(in srgb, var(--c-black) 20%, transparent)"
        activity_card_border = "color-mix(in srgb, var(--c-text) 10.5%, transparent)"
        activity_card_border_hover = "color-mix(in srgb, var(--c-text) 16%, transparent)"
        activity_monochrome_svg_filter = "none"

    return [
        "  --surface-hover: color-mix(in srgb, var(--c-text) 6%, transparent);",
        "  --surface-hover-strong: color-mix(in srgb, var(--c-text) 10%, transparent);",
        "  --surface-blue-soft: color-mix(in srgb, var(--c-primary) 10%, transparent);",
        "  --c-overlay-blue-strong: color-mix(in srgb, var(--c-primary) 16%, transparent);",
        "  --focus-ring: color-mix(in srgb, var(--c-primary) 18%, transparent);",
        "  --surface-purple-soft: color-mix(in srgb, var(--c-system-purple) 12%, transparent);",
        "  --surface-purple-strong: color-mix(in srgb, var(--c-system-purple) 15%, transparent);",
        "  --surface-green-soft: color-mix(in srgb, var(--c-success) 8%, transparent);",
        "  --border-success: color-mix(in srgb, var(--c-success) 40%, transparent);",
        "  --text-success: color-mix(in srgb, var(--c-success) 85%, transparent);",
        f"  --activity-card-bg: {activity_card_bg};",
        f"  --activity-card-bg-hover: {activity_card_bg_hover};",
        f"  --activity-card-border: {activity_card_border};",
        f"  --activity-card-border-hover: {activity_card_border_hover};",
        "  --activity-card-shadow: 0 10px 26px color-mix(in srgb, var(--c-black) 14%, transparent);",
        f"  --activity-card-inner-bg: {activity_inner_bg};",
        f"  --activity-card-inner-bg-strong: {activity_inner_bg_strong};",
        f"  --activity-card-subtle-bg: {activity_card_subtle_bg};",
        f"  --activity-card-hover-subtle-bg: {activity_card_hover_subtle_bg};",
        "  --activity-on-surface: var(--c-text);",
        "  --activity-on-surface-muted: color-mix(in srgb, var(--c-text) 72%, transparent);",
        "  --activity-on-surface-dim: color-mix(in srgb, var(--c-text) 54%, transparent);",
        "  --activity-status-bg: color-mix(in srgb, var(--c-text) 6%, transparent);",
        "  --activity-status-border: color-mix(in srgb, var(--c-text) 10%, transparent);",
        "  --activity-icon-bg: color-mix(in srgb, var(--surface-secondary) 94%, var(--c-text) 6%);",
        "  --activity-chip-bg: color-mix(in srgb, var(--c-text) 5.5%, transparent);",
        "  --activity-chip-bg-hover: color-mix(in srgb, var(--c-text) 8.5%, transparent);",
        f"  --activity-monochrome-svg-filter: {activity_monochrome_svg_filter};",
        "  --activity-preview-action-bg: color-mix(in srgb, var(--surface-secondary) 86%, var(--c-primary) 14%);",
        "  --activity-preview-action-bg-hover: color-mix(in srgb, var(--surface-secondary) 76%, var(--c-primary) 24%);",
        "  --activity-preview-action-border: color-mix(in srgb, var(--c-primary) 34%, var(--c-text) 10%);",
        "  --activity-image-frame-bg: color-mix(in srgb, var(--surface-secondary) 94%, var(--c-text) 6%);",
        "  --activity-image-grid-tile: color-mix(in srgb, var(--c-text) 3.5%, transparent);",
        "  --activity-image-loader-glow: color-mix(in srgb, var(--c-primary) 24%, transparent);",
        "  --activity-media-track-bg: color-mix(in srgb, var(--c-text) 10%, transparent);",
        "  --activity-media-track-buffered: color-mix(in srgb, var(--c-text) 22%, transparent);",
        "  --activity-media-control-bg: color-mix(in srgb, var(--surface-secondary) 88%, var(--c-text) 12%);",
        "  --activity-media-control-bg-hover: color-mix(in srgb, var(--surface-secondary) 82%, var(--c-text) 18%);",
        "  --activity-media-control-border: color-mix(in srgb, var(--c-text) 14%, transparent);",
    ]


# Return JSON that is safe to embed in an application/json script tag.
def _json_for_script_tag(value: dict[str, Any]) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


# Default dark preview tokens used when no host theme snapshot is available.
_PREVIEW_DOCUMENT_DEFAULTS: Final[dict[str, str]] = {
    "color_scheme": "dark",
    "bg": "#1c1c1e",
    "surface": "#232326",
    "surface_strong": "#2c2c2e",
    "border": "#38383a",
    "text": "#ffffff",
    "muted": "rgba(235, 235, 245, 0.72)",
    "link": "#0a84ff",
}


# Resolve host palette colors into CSS variables used by the chat UI.
def resolve_host_css_variables() -> tuple[dict[str, str], str]:
    """Return ``(css_var -> color, effective_theme)`` from the host theme snapshot."""

    raw = load_host_theme()
    if not isinstance(raw, dict):
        return {}, "dark"

    colors = raw.get("colors")
    if not isinstance(colors, dict):
        return {}, _effective_theme(raw)

    resolved: dict[str, str] = {}
    for aslm_key, css_var in _ASLM_COLOR_KEY_TO_CSS_VAR.items():
        val = colors.get(aslm_key)
        if val is None:
            continue
        css_color = normalize_color_to_css(str(val))
        if css_color is None:
            continue
        resolved[css_var] = css_color

    if "--c-system-teal" in resolved:
        resolved["--c-system-cyan"] = resolved["--c-system-teal"]

    return resolved, _effective_theme(raw)


# Build CSS tokens for standalone HTML documents (downloads bridge previews, etc.).
def resolve_preview_document_theme() -> dict[str, str]:
    """Return themed CSS tokens for offline HTML shells consumed by ASLM.

    Keys: ``color_scheme``, ``bg``, ``surface``, ``surface_strong``, ``border``,
    ``text``, ``muted``, ``link``, ``fingerprint``.
    Falls back to the dark ASLM-Chat defaults when no host theme is present.
    """

    resolved, theme = resolve_host_css_variables()
    tokens = dict(_PREVIEW_DOCUMENT_DEFAULTS)
    tokens["color_scheme"] = theme if theme in {"light", "dark"} else "dark"

    if resolved:
        tokens["bg"] = (
            resolved.get("--c-bg-surface")
            or resolved.get("--c-bg")
            or tokens["bg"]
        )
        tokens["surface"] = (
            resolved.get("--c-bg-elevated")
            or resolved.get("--c-bg-surface")
            or tokens["surface"]
        )
        tokens["surface_strong"] = (
            resolved.get("--c-gray-4")
            or resolved.get("--c-bg-elevated")
            or tokens["surface_strong"]
        )
        tokens["border"] = resolved.get("--c-border") or tokens["border"]
        tokens["text"] = resolved.get("--c-text") or tokens["text"]
        tokens["muted"] = (
            resolved.get("--c-text-muted")
            or resolved.get("--c-gray-2")
            or tokens["muted"]
        )
        tokens["link"] = (
            resolved.get("--c-link")
            or resolved.get("--c-system-blue")
            or resolved.get("--c-primary")
            or tokens["link"]
        )

    # Stable short fingerprint so HTML caches can key off the active theme.
    identity = json.dumps(
        {key: tokens[key] for key in sorted(tokens.keys())},
        ensure_ascii=True,
        sort_keys=True,
    )
    # Local import avoided: hashlib is stdlib and keeps this module free of Services deps.
    import hashlib

    tokens["fingerprint"] = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    return tokens
