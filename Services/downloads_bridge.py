# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from Apps.UI.locale_catalog import resolve_effective_locale_from_snapshot, translate, translate_plural
from Settings.host_theme import load_host_theme

# Bridge configuration
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = BASE_DIR / "Data" / "downloads_bridge_cache"
DETAIL_CACHE_DIR = CACHE_DIR / "ollama_details"
DETAIL_HTML_CACHE_DIR = CACHE_DIR / "ollama_detail_html"
SEARCH_CACHE_DIR = CACHE_DIR / "ollama_search"
REQUEST_TIMEOUT_SECONDS = 20
CACHE_SCHEMA_VERSION = 1
MAX_VARIANT_SIZE_BYTES = (1 << 63) - 1

OLLAMA_SEARCH_URL = "https://ollama.com/search"
OLLAMA_CATEGORY_ID = "ollama-library"
OLLAMA_CATEGORY_TITLE = "Ollama Models"
OLLAMA_CATEGORY_DESCRIPTION = "Models published in the official Ollama library."
OLLAMA_GROUP_KEY = "ollama-models"
OLLAMA_TARGET_REF = "ollama_models"
PROTOCOL_VERSION = 1

DEFAULT_SORT_KEY = "popular"
SORT_FILTER_PREFIX = "sort:"
CAPABILITY_FILTER_PREFIX = "capability:"
CAPABILITY_COLORS = {
    "vision": "SystemOrange",
    "tools": "SystemBlue",
    "thinking": "SystemGreen",
    "cloud": "SystemTeal",
}


# Search query model

# Normalized Ollama search request arguments.
@dataclass(frozen=True)
class OllamaSearchQuery:
    query_text: str
    sort_key: str
    capabilities: tuple[str, ...]


# Response helpers

# Build a standard bridge response payload.
def _response(
    *,
    success: bool = True,
    categories: list[dict[str, Any]] | None = None,
    items: list[dict[str, Any]] | None = None,
    filters: list[dict[str, Any]] | None = None,
    item_detail: dict[str, Any] | None = None,
    install_manifest: dict[str, Any] | None = None,
    uninstall_manifest: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "protocolVersion": PROTOCOL_VERSION,
        "success": success,
        "warnings": warnings or [],
    }
    if categories is not None:
        payload["categories"] = categories
    if items is not None:
        payload["items"] = items
    if filters is not None:
        payload["filters"] = filters
    if item_detail is not None:
        payload["itemDetail"] = item_detail
    if install_manifest is not None:
        payload["installManifest"] = install_manifest
    if uninstall_manifest is not None:
        payload["uninstallManifest"] = uninstall_manifest
    if error:
        payload["error"] = error
    owners = [*(items or []), *([item_detail] if item_detail else [])]
    resources = _build_resources(owners) if owners else None
    if resources:
        payload["resources"] = resources
    return _localize_response(payload) if owners or filters is not None else payload


# Resolve named assets and the current host palette when sending a response, not when caching items.
def _build_resources(owners: list[dict[str, Any]]) -> dict[str, Any]:
    theme = load_host_theme() or {}
    palette = theme.get("colors") or {}
    if not isinstance(palette, dict):
        palette = {}
    surface = _color_channels(palette.get("BackgroundSecondary"))
    colors: dict[str, str] = {}
    icons: dict[str, dict[str, str]] = {}
    for owner in owners:
        for field in [*owner.get("details", []), *owner.get("tags", [])]:
            icon = field.get("icon")
            if icon in CAPABILITY_COLORS:
                icons[icon] = {"path": f"Apps/UI/static/img/capabilities/{icon}.png"}
                color_key = CAPABILITY_COLORS[icon]
                foreground = _color_channels(palette.get(color_key))
                if foreground is not None:
                    colors[color_key] = "#" + "".join(f"{channel:02X}" for channel in foreground)
                    field["textColor"] = color_key
                    if surface is not None:
                        background_key = f"{color_key}Background"
                        colors[background_key] = "#" + "".join(
                            f"{(front + 4 * back + 2) // 5:02X}" for front, back in zip(foreground, surface)
                        )
                        field["backgroundColor"] = background_key
            elif icon in {"downloads", "update"}:
                icons[icon] = {"path": f"Apps/UI/static/img/info/{icon}.png"}
    return {key: value for key, value in {"colors": colors, "icons": icons}.items() if value}


# Host palette values use RGB or MAUI ARGB hex, with alpha first.
def _color_channels(value: Any) -> tuple[int, ...] | None:
    if not isinstance(value, str) or not re.fullmatch(r"#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})", value):
        return None
    body = value[1:]
    if len(body) == 6:
        body = "FF" + body
    return tuple(int(body[index:index + 2], 16) for index in range(0, 8, 2))


# Request helpers

# Read and validate the JSON request from stdin.
def _read_request() -> dict[str, Any]:
    raw_payload = sys.stdin.read().strip()
    if not raw_payload:
        return {}

    try:
        payload = json.loads(raw_payload)
    except json.JSONDecodeError:
        return {}

    return payload if isinstance(payload, dict) else {}


# Text normalization helpers

# Normalize text into a single readable line.
def _normalize_text(value: str | None) -> str:
    return " ".join(str(value or "").replace("\xa0", " ").split())


# Normalize bullet-style separators into the bridge format.
def _normalize_separator(value: str) -> str:
    normalized = str(value or "").replace("\u00b7", "|").replace("\u2022", "|")
    return " | ".join(part.strip() for part in normalized.split("|") if part.strip())


# Normalize a filter key for comparisons.
def _normalize_filter_key(value: str | None) -> str:
    return _normalize_text(value).lower()


# Remove duplicates while keeping the first occurrence.
def _deduplicate_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduplicated: list[str] = []

    for value in values:
        normalized = _normalize_text(value)
        if not normalized:
            continue

        identity = normalized.lower()
        if identity in seen:
            continue

        seen.add(identity)
        deduplicated.append(normalized)

    return deduplicated


# Parse an integer from mixed text.
def _parse_int(value: str) -> int:
    digits = re.sub(r"[^0-9]", "", _normalize_text(value))
    return int(digits) if digits else 0


# Cache helpers

# Ensure that all cache directories exist.
def _ensure_cache_dirs() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DETAIL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    DETAIL_HTML_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Read a JSON cache payload.
def _read_cache(path: Path) -> Any | None:
    if not path.exists():
        return None

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("_cacheSchema") != CACHE_SCHEMA_VERSION:
            return None
        payload.pop("_cacheSchema")
        # Rebuild incompatible cached payloads in place; never translate the old wire format.
        for item in payload.get("items", [payload]):
            if "detail" in item:
                return None
            for key in ("details", "tags"):
                fields = item.get(key)
                if fields is not None and (not isinstance(fields, list) or
                                           any(not isinstance(field, dict) for field in fields)):
                    return None
            for variant in item.get("variants", []):
                if any(key in variant for key in ("detail", "details", "tags", "version", "installedVersion")):
                    return None
                if type(variant.get("hasSize")) is not bool or type(variant.get("size")) is not int:
                    return None
                if not 0 <= variant["size"] <= MAX_VARIANT_SIZE_BYTES or (not variant["hasSize"] and variant["size"] != 0):
                    return None
        return payload
    except Exception:
        return None


# Write a JSON cache payload.
def _write_cache(path: Path, payload: Any) -> None:
    _ensure_cache_dirs()
    path.write_text(json.dumps({**payload, "_cacheSchema": CACHE_SCHEMA_VERSION}, ensure_ascii=False, indent=2), encoding="utf-8")


# Write a plain text cache file.
def _write_text_file(path: Path, content: str) -> None:
    _ensure_cache_dirs()
    path.write_text(content, encoding="utf-8")


# Resource key helpers

# Build a resource key for a model variant.
def _variant_resource_key(slug: str, tag: str) -> str:
    normalized_tag = _normalize_text(tag)
    return f"ollama:{slug}:{normalized_tag}" if normalized_tag else f"ollama:{slug}"


# Extract the slug portion from a resource key.
def _resource_key_to_slug(resource_key: str) -> str:
    normalized = str(resource_key or "").strip()
    if normalized.startswith("ollama:"):
        return normalized.partition(":")[2].strip()
    return normalized


# Platform / runtime format helpers
#
# Ollama publishes classic (CUDA/Vulkan/Metal-via-llama.cpp) tags and macOS MLX
# tags side-by-side (e.g. gemma4:12b vs gemma4:12b-mlx). The downloads bridge
# runs on the user machine via ASLM, so filter by local OS:
#   - Windows / Linux / other → classic tags only (exclude MLX)
#   - macOS → MLX tags only

# Resolve the host OS family for model-format filtering.
def _host_platform_family() -> str:
    system = platform.system().strip().lower()
    if system == "darwin" or sys.platform == "darwin":
        return "macos"
    if system == "windows" or sys.platform.startswith("win"):
        return "windows"
    if system == "linux" or sys.platform.startswith("linux"):
        return "linux"
    return "other"


# Report whether this host should only see MLX model builds.
def _host_wants_mlx_variants() -> bool:
    return _host_platform_family() == "macos"


# Return True when a model ref/tag denotes an MLX (Apple Silicon) build.
def _is_mlx_model_ref(value: str | None) -> bool:
    text = _normalize_text(value).lower()
    if not text:
        return False
    if text.startswith("ollama:"):
        text = text[len("ollama:") :].strip()

    # Prefer the tag segment when the ref is slug:tag.
    tag = text.split(":", 1)[1].strip() if ":" in text else text
    if not tag:
        return False

    # Common Ollama MLX tags: "12b-mlx", "e2b-mlx", rarely bare "mlx".
    if tag == "mlx" or tag.endswith("-mlx"):
        return True
    # Defensive: mlx as a discrete hyphen segment (foo-mlx-bar).
    if re.search(r"(^|-)mlx($|-)", tag):
        return True
    return False


# Return True when a parsed variant payload is an MLX build.
def _variant_is_mlx(variant: dict[str, Any] | None) -> bool:
    if not isinstance(variant, dict):
        return False
    if _is_mlx_model_ref(str(variant.get("resourceKey") or "")):
        return True
    if _is_mlx_model_ref(str(variant.get("title") or "")):
        return True
    return False


# Keep only variants that match the host platform (classic vs MLX).
def _filter_variants_for_host(variants: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    wants_mlx = _host_wants_mlx_variants()
    filtered: list[dict[str, Any]] = []
    for variant in variants or []:
        if not isinstance(variant, dict):
            continue
        is_mlx = _variant_is_mlx(variant)
        if wants_mlx and not is_mlx:
            continue
        if not wants_mlx and is_mlx:
            continue
        next_variant = dict(variant)
        next_variant["sortOrder"] = len(filtered)
        filtered.append(next_variant)
    return filtered


# Pick the default variant resource key after platform filtering.
def _select_default_variant_key(variants: list[dict[str, Any]]) -> str:
    if not variants:
        return ""
    for variant in variants:
        key = str(variant.get("resourceKey") or "")
        title = str(variant.get("title") or "").strip().lower()
        if key.endswith(":latest") or title == "latest":
            return key
    return str(variants[0].get("resourceKey") or "")


# Apply host platform variant filtering to a detail payload in place/copy.
def _apply_platform_variant_filter(detail: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(detail or {})
    variants = normalized.get("variants")
    if not isinstance(variants, list):
        return normalized

    filtered = _filter_variants_for_host(variants)
    normalized["variants"] = filtered
    normalized["defaultVariantResourceKey"] = _select_default_variant_key(filtered)
    if "variantCount" in normalized:
        normalized["variantCount"] = len(filtered)
    return normalized


# Reject install/uninstall of a format that cannot run on this host.
def _assert_resource_allowed_on_host(resource_key: str) -> None:
    normalized = _normalize_text(resource_key)
    if not normalized:
        return

    model_ref = normalized[len("ollama:") :].strip() if normalized.lower().startswith("ollama:") else normalized
    # Family-only keys (no tag) are allowed; the UI should prefer a variant key.
    if ":" not in model_ref:
        return

    is_mlx = _is_mlx_model_ref(normalized)
    wants_mlx = _host_wants_mlx_variants()
    family = _host_platform_family()

    if wants_mlx and not is_mlx:
        raise ValueError(
            f"Model '{model_ref}' is not an MLX build. "
            "On macOS the downloads catalog only installs MLX variants."
        )
    if not wants_mlx and is_mlx:
        raise ValueError(
            f"Model '{model_ref}' is an MLX (Apple Silicon) build. "
            f"On {family} the downloads catalog only installs classic CUDA/Vulkan variants."
        )


# Cache path helpers

# Resolve the JSON detail cache path for a slug.
def _detail_cache_path(slug: str) -> Path:
    safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", slug)
    return DETAIL_CACHE_DIR / f"{safe_slug}.json"


# Resolve the HTML detail cache path for a rendered block.
def _detail_html_cache_path(
    slug: str,
    block_id: str,
    content_digest: str,
    theme_fingerprint: str,
) -> Path:
    safe_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", slug)
    safe_block_id = re.sub(r"[^a-zA-Z0-9._-]+", "_", block_id)
    safe_theme = re.sub(r"[^a-zA-Z0-9._-]+", "_", theme_fingerprint or "default")[:16]
    safe_content = re.sub(r"[^a-zA-Z0-9._-]+", "_", content_digest or "content")[:16]
    return DETAIL_HTML_CACHE_DIR / f"{safe_slug}_{safe_block_id}_{safe_theme}_{safe_content}.html"


# Default dark preview shell (matches base ASLM-Chat palette when host theme is absent).
_DEFAULT_PREVIEW_THEME: dict[str, str] = {
    "color_scheme": "dark",
    "bg": "#1c1c1e",
    "surface": "#232326",
    "surface_strong": "#2c2c2e",
    "border": "#38383a",
    "text": "#ffffff",
    "muted": "rgba(235, 235, 245, 0.72)",
    "link": "#0a84ff",
    "fingerprint": "default-dark",
}


# Load host-theme tokens for standalone preview HTML documents.
def _resolve_preview_theme() -> dict[str, str]:
    tokens = dict(_DEFAULT_PREVIEW_THEME)
    try:
        # Reuse the same host palette mapping as the chat UI.
        from Apps.UI.host_theme_bridge import resolve_preview_document_theme

        resolved = resolve_preview_document_theme()
        if isinstance(resolved, dict) and resolved.get("bg") and resolved.get("text"):
            tokens = dict(resolved)
    except Exception:
        # Bridge must stay available even if the theme module is missing.
        pass
    # Include the rendered shell so CSS changes also refresh already-cached Readmes.
    shell = _build_html_document("", "", "", tokens)
    tokens["fingerprint"] = hashlib.sha256(shell.encode("utf-8")).hexdigest()[:16]
    return tokens


# Escape a CSS color/token value for safe interpolation into a style block.
def _css_token(value: str, fallback: str) -> str:
    text = str(value or "").strip() or fallback
    # Preview tokens are always resolved host colors or hardcoded defaults; still
    # strip characters that could break out of a CSS declaration.
    return re.sub(r"[^#(),.%a-zA-Z0-9\s_-]", "", text) or fallback


# URL helpers

# Resolve the Ollama page path for a model slug.
def _resolve_model_page_path(slug: str) -> str:
    normalized_slug = _normalize_text(slug).strip("/")
    if not normalized_slug:
        return "/library"

    return f"/{normalized_slug}" if "/" in normalized_slug else f"/library/{normalized_slug}"


# Resolve the absolute Ollama page URL for a model slug.
def _resolve_model_page_url(slug: str) -> str:
    return f"https://ollama.com{_resolve_model_page_path(slug)}"


# Build the cache path for a search query.
def _search_cache_path(search_query: OllamaSearchQuery) -> Path:
    identity = json.dumps(
        {
            "q": search_query.query_text,
            "sort": search_query.sort_key,
            "capabilities": list(search_query.capabilities),
        },
        ensure_ascii=True,
        sort_keys=True,
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
    return SEARCH_CACHE_DIR / f"{digest}.json"


# HTTP helpers

# Request text content from a remote page.
def _request_text(url: str, params: list[tuple[str, str]] | None = None) -> str:
    import requests

    response = requests.get(
        url,
        params=params or None,
        timeout=REQUEST_TIMEOUT_SECONDS,
        headers={"User-Agent": "ASLM-Chat Downloads Bridge"},
    )
    response.raise_for_status()
    return response.text


# Build a BeautifulSoup parser lazily so metadata-only bridge calls stay cheap.
def _make_soup(markup: str) -> Any:
    from bs4 import BeautifulSoup

    return BeautifulSoup(markup, "html.parser")


# Resolve a relative URL against the current page.
def _absolutize_url(candidate: str | None, base_url: str) -> str:
    normalized = _normalize_text(candidate)
    return urljoin(base_url, normalized) if normalized else ""


# Payload formatting helpers

# Localize only at the response boundary: disk caches retain Ollama's original labels.
def _localize_response(payload: dict[str, Any]) -> dict[str, Any]:
    localized = deepcopy(payload)
    locale = resolve_effective_locale_from_snapshot()
    for entry in localized.get("filters", []):
        key = entry.get("key")
        if key in {"sort:popular", "sort:newest"}:
            entry["title"] = translate(f"downloadsBridge.filters.{key.split(':', 1)[1]}", locale=locale)

    owners = [*localized.get("items", [])]
    if localized.get("itemDetail") is not None:
        owners.append(localized["itemDetail"])
    for owner in owners:
        for field in owner.get("details", []):
            text = _normalize_text(field.get("text"))
            if field.get("icon") == "downloads":
                match = re.fullmatch(r"(\d[\d,]*(?:\.\d+)?\s*[KMBT]?)\s+(?:downloads?|pulls?)", text, re.IGNORECASE)
                if match:
                    display_count = match.group(1)
                    number = display_count.replace(",", "").replace(" ", "").upper()
                    multiplier = 1000 ** ("KMBT".index(number[-1]) + 1) if number[-1] in "KMBT" else 1
                    count = Decimal(number[:-1] if multiplier != 1 else number) * multiplier
                    if count == count.to_integral_value():
                        field["text"] = translate_plural(
                            "downloadsBridge.downloads", int(count), locale=locale, display_count=display_count,
                        )
            elif field.get("icon") == "update":
                field["text"] = _localize_updated_text(text, locale)
        for variant in owner.get("variants", []):
            variant["summary"] = re.sub(
                r"\b(text|image)\b",
                lambda match: translate(f"downloadsBridge.summary.{match.group().lower()}", locale=locale),
                str(variant.get("summary") or ""),
                flags=re.IGNORECASE,
            )
    return localized


# Preserve unknown provider labels; never invent a timestamp for an unrecognized value.
def _localize_updated_text(value: str, locale: str) -> str:
    normalized = _strip_leading_label(value, ("updated",)).lower()
    aliases = {
        "now": "justNow",
        "just now": "justNow",
        "a few seconds ago": "fewSecondsAgo",
        "less than a minute ago": "lessThanMinuteAgo",
        "today": "today",
        "yesterday": "yesterday",
    }
    if normalized in aliases:
        return translate(f"downloadsBridge.updated.{aliases[normalized]}", locale=locale)
    match = re.fullmatch(r"(\d+|an?)\s+(year|month|week|day|hour|minute|second)s?\s+ago", normalized)
    if match:
        count, unit = match.groups()
        return translate_plural(
            f"downloadsBridge.updated.{unit}", 1 if count in {"a", "an"} else int(count), locale=locale,
        )
    return value


# Keep individual counters separate; the host decides how to lay them out.
def _build_details(pull_count: str, updated_text: str) -> list[dict[str, Any]]:
    return [
        {"text": text, "showInCatalog": True, "icon": icon}
        for text, icon in (
            (f"{pull_count} downloads" if pull_count else "", "downloads"),
            (updated_text, "update"),
        )
        if text
    ]


def _build_tags(values: list[str]) -> list[dict[str, Any]]:
    return [
        {"text": value, **({"icon": value.lower(), "showInCatalog": True} if value.lower() in CAPABILITY_COLORS else {})}
        for value in _deduplicate_preserving_order(values)
    ]


# Build the static Ollama category payload.
def _build_ollama_category() -> dict[str, Any]:
    return {
        "id": OLLAMA_CATEGORY_ID,
        "title": OLLAMA_CATEGORY_TITLE,
        "description": OLLAMA_CATEGORY_DESCRIPTION,
        "groupKey": OLLAMA_GROUP_KEY,
        "targetRef": OLLAMA_TARGET_REF,
        "sortOrder": 10,
    }


# Build default Ollama filter payloads when live filter data is not available.
def _build_default_filter_payloads(search_query: OllamaSearchQuery) -> list[dict[str, Any]]:
    active_capabilities = set(search_query.capabilities)
    active_sort = search_query.sort_key or DEFAULT_SORT_KEY
    filters: list[dict[str, Any]] = [
        {
            "key": f"{SORT_FILTER_PREFIX}popular",
            "title": "Popular",
            "kind": "sort",
            "selected": active_sort == "popular",
            "sortOrder": 10,
        },
        {
            "key": f"{SORT_FILTER_PREFIX}newest",
            "title": "Newest",
            "kind": "sort",
            "selected": active_sort == "newest",
            "sortOrder": 11,
        },
    ]

    for index, capability in enumerate(("cloud", "embedding", "vision", "tools", "thinking")):
        filters.append(
            {
                "key": f"{CAPABILITY_FILTER_PREFIX}{capability}",
                "title": capability.title(),
                "kind": "capability",
                "selected": capability in active_capabilities,
                "sortOrder": 100 + index,
            }
        )

    return filters


# HTML rendering helpers

# Build a standalone HTML document for preview blocks.
def _build_html_document(
    inner_html: str,
    page_url: str,
    title: str,
    theme: dict[str, str] | None = None,
) -> str:
    tokens = theme or _resolve_preview_theme()
    defaults = _DEFAULT_PREVIEW_THEME
    color_scheme = _css_token(tokens.get("color_scheme", ""), defaults["color_scheme"])
    if color_scheme not in {"light", "dark"}:
        color_scheme = defaults["color_scheme"]
    bg = _css_token(tokens.get("bg", ""), defaults["bg"])
    surface = _css_token(tokens.get("surface", ""), defaults["surface"])
    surface_strong = _css_token(tokens.get("surface_strong", ""), defaults["surface_strong"])
    border = _css_token(tokens.get("border", ""), defaults["border"])
    text = _css_token(tokens.get("text", ""), defaults["text"])
    link = _css_token(tokens.get("link", ""), defaults["link"])
    safe_title = (
        str(title or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="color-scheme" content="{color_scheme}">
  <base href="{page_url}">
  <title>{safe_title}</title>
  <style>
    :root {{
      color-scheme: {color_scheme};
      --bg: {bg};
      --surface: {surface};
      --surface-strong: {surface_strong};
      --border: {border};
      --text: {text};
      --link: {link};
    }}

    * {{
      box-sizing: border-box;
    }}

    html, body {{
      margin: 0;
      padding: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Segoe UI", "Open Sans", sans-serif;
      line-height: 1.58;
      font-size: 14px;
    }}

    body {{
      padding: 8px 12px 20px;
    }}

    .content {{
      max-width: none;
    }}

    h1, h2, h3, h4, h5, h6 {{
      color: var(--text);
      margin: 1.15em 0 0.55em;
      line-height: 1.22;
      font-weight: 700;
    }}

    h1:first-child, h2:first-child, h3:first-child {{
      margin-top: 0;
    }}

    p, ul, ol, blockquote, pre {{
      margin: 0 0 14px;
      color: var(--text);
    }}

    strong, b {{
      color: var(--text);
    }}

    a {{
      color: var(--link);
      text-decoration: none;
    }}

    a:hover {{
      text-decoration: underline;
    }}

    code {{
      font-family: "Cascadia Code", "Consolas", monospace;
      font-size: 0.92em;
      background: var(--surface);
      color: var(--text);
      padding: 2px 6px;
      border-radius: 7px;
    }}

    pre {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 12px 14px;
      overflow: auto;
      color: var(--text);
    }}

    pre code {{
      background: transparent;
      padding: 0;
      border-radius: 0;
    }}

    ul, ol {{
      padding-left: 22px;
    }}

    li {{
      margin: 6px 0;
      color: var(--text);
    }}

    img {{
      display: block;
      max-width: 100%;
      height: auto;
      margin: 16px auto;
      border-radius: 14px;
    }}

    .table-wrap {{
      width: 100%;
      max-width: 100%;
      overflow-x: auto;
      overflow-y: hidden;
      margin: 16px 0;
      padding-bottom: 2px;
    }}

    table {{
      width: max-content !important;
      min-width: 0 !important;
      max-width: none !important;
      border-collapse: collapse;
      margin: 0;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      display: table !important;
    }}

    thead, tbody {{
      width: auto;
    }}

    th, td {{
      border-bottom: 1px solid var(--border);
      padding: 8px 10px;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
      color: var(--text);
      font-size: 13px;
    }}

    th {{
      color: var(--text);
      background: var(--surface-strong);
      font-weight: 600;
    }}

    tr:last-child td {{
      border-bottom: 0;
    }}

    blockquote {{
      border-left: 3px solid var(--border);
      padding-left: 12px;
      color: var(--text);
    }}

    hr {{
      border: 0;
      border-top: 1px solid var(--border);
      margin: 18px 0;
    }}
  </style>
</head>
<body>
  <div class="content">
    {inner_html}
  </div>
</body>
</html>"""


# Pull the themed body content back out of a previously written preview HTML file.
def _extract_preview_inner_html(document: str) -> str:
    try:
        soup = _make_soup(document)
    except Exception:
        return ""
    content = soup.select_one("div.content")
    if content is None:
        return ""
    return content.decode_contents()


# Write a themed standalone HTML file for one preview block.
def _write_themed_preview_file(
    *,
    slug: str,
    block_id: str,
    title: str,
    inner_html: str,
    page_url: str,
    theme: dict[str, str] | None = None,
) -> str:
    tokens = theme or _resolve_preview_theme()
    theme_fingerprint = str(tokens.get("fingerprint") or "default")
    content_digest = hashlib.sha256(inner_html.encode("utf-8")).hexdigest()[:16]
    cache_path = _detail_html_cache_path(slug, block_id, content_digest, theme_fingerprint)

    if not cache_path.exists():
        html_document = _build_html_document(inner_html, page_url, title, tokens)
        _write_text_file(cache_path, html_document)

    return str(cache_path)


# Normalize and cache a rendered HTML block.
def _create_html_block_file(slug: str, block_id: str, title: str, node_html: str, page_url: str) -> str:
    fragment = _make_soup(node_html)

    # Normalize media links so cached files keep working offline
    for node in fragment.select("[src]"):
        node["src"] = _absolutize_url(node.get("src"), page_url)

    for node in fragment.select("[href]"):
        href = _normalize_text(node.get("href"))
        if href.startswith("#"):
            node["href"] = page_url + href
        else:
            node["href"] = _absolutize_url(href, page_url)

    # Wrap raw tables to preserve layout inside the embedded preview
    for table in list(fragment.select("table")):
        parent_classes = table.parent.get("class", []) if table.parent else []
        if "table-wrap" in parent_classes:
            continue

        wrapper = fragment.new_tag("div", attrs={"class": "table-wrap"})
        table.wrap(wrapper)

    return _write_themed_preview_file(
        slug=slug,
        block_id=block_id,
        title=title,
        inner_html=fragment.decode_contents(),
        page_url=page_url,
    )


# Rebuild html-file preview paths so cached details follow the current host theme.
def _retheme_detail_html_blocks(detail: dict[str, Any], slug: str) -> dict[str, Any]:
    blocks = detail.get("blocks")
    if not isinstance(blocks, list) or not blocks:
        return detail

    theme = _resolve_preview_theme()
    theme_fp = str(theme.get("fingerprint") or "default")
    page_url = _normalize_text(detail.get("homepageUrl")) or _resolve_model_page_url(slug)
    updated_blocks: list[Any] = []

    for block in blocks:
        if not isinstance(block, dict):
            updated_blocks.append(block)
            continue

        next_block = dict(block)
        if _normalize_filter_key(next_block.get("format")) != "html-file":
            updated_blocks.append(next_block)
            continue

        content_url = _normalize_text(next_block.get("contentUrl"))
        old_path = Path(content_url) if content_url else None
        # Already themed for the active host palette.
        if old_path is not None and theme_fp in old_path.name and old_path.is_file():
            updated_blocks.append(next_block)
            continue

        inner_html = ""
        if old_path is not None and old_path.is_file():
            try:
                inner_html = _extract_preview_inner_html(old_path.read_text(encoding="utf-8"))
            except OSError:
                inner_html = ""
        if not inner_html:
            # Fall back to inline markdown/html content when present.
            inner_html = str(next_block.get("content") or "").strip()

        if not inner_html:
            updated_blocks.append(next_block)
            continue

        block_id = _normalize_text(next_block.get("id")) or "readme"
        title = _normalize_text(next_block.get("title")) or f"{slug} {block_id}"
        next_block["contentUrl"] = _write_themed_preview_file(
            slug=slug,
            block_id=block_id,
            title=title,
            inner_html=inner_html,
            page_url=page_url,
            theme=theme,
        )
        updated_blocks.append(next_block)

    detail["blocks"] = updated_blocks
    return detail


# Search query helpers

# Normalize query text and selected filters.
def _normalize_search_query(query_text: str, filter_keys: list[str] | None) -> OllamaSearchQuery:
    normalized_query = _normalize_text(query_text)
    normalized_filters = [_normalize_filter_key(value) for value in filter_keys or [] if _normalize_filter_key(value)]

    sort_key = DEFAULT_SORT_KEY
    capabilities: list[str] = []

    # Split mixed filter keys into sort and capability groups
    for filter_key in normalized_filters:
        if filter_key.startswith(SORT_FILTER_PREFIX):
            candidate = filter_key.removeprefix(SORT_FILTER_PREFIX).strip()
            if candidate:
                sort_key = candidate
            continue

        if filter_key.startswith(CAPABILITY_FILTER_PREFIX):
            capability = filter_key.removeprefix(CAPABILITY_FILTER_PREFIX).strip()
            if capability and capability not in capabilities:
                capabilities.append(capability)

    capabilities.sort()
    return OllamaSearchQuery(
        query_text=normalized_query,
        sort_key=sort_key or DEFAULT_SORT_KEY,
        capabilities=tuple(capabilities),
    )


# Build Ollama search parameters from the normalized query.
def _build_search_params(search_query: OllamaSearchQuery) -> list[tuple[str, str]]:
    params: list[tuple[str, str]] = []

    if search_query.query_text:
        params.append(("q", search_query.query_text))

    for capability in search_query.capabilities:
        params.append(("c", capability))

    if search_query.sort_key and search_query.sort_key != DEFAULT_SORT_KEY:
        params.append(("o", search_query.sort_key))

    return params


# Search parsing helpers

# Strip a trailing label from a meta group such as "263.3K Pulls".
def _strip_trailing_label(text: str, labels: tuple[str, ...]) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    lower = normalized.lower()
    for label in labels:
        suffix = f" {label.lower()}"
        if lower.endswith(suffix):
            return normalized[: -len(suffix)].strip()
        if lower == label.lower():
            return ""
    return normalized


# Strip a leading label from a meta group such as "Updated 3 weeks ago".
def _strip_leading_label(text: str, labels: tuple[str, ...]) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    lower = normalized.lower()
    for label in labels:
        prefix = f"{label.lower()} "
        if lower.startswith(prefix):
            return normalized[len(prefix) :].strip()
        if lower == label.lower():
            return ""
    return normalized


# Extract pull/tag/updated meta from one Ollama search card (current + legacy markup).
def _extract_search_card_meta(list_item: Any) -> tuple[str, str, str]:
    pull_count_text = ""
    tag_count_text = ""
    updated_text = ""

    # Current Ollama cards expose labeled icon+text groups without x-test attributes.
    for group in list_item.select("span.flex.items-center"):
        group_text = _normalize_text(group.get_text(" ", strip=True))
        if not group_text:
            continue
        lower = group_text.lower()

        if "pull" in lower or "download" in lower:
            pull_count_text = _strip_trailing_label(group_text, ("pulls", "pull", "downloads", "download"))
            continue
        if re.search(r"\btags?\b", lower):
            tag_count_text = _strip_trailing_label(group_text, ("tags", "tag"))
            continue
        if "updated" in lower or re.search(r"\bago\b", lower) or lower in {"today", "yesterday", "just now", "now"}:
            updated_text = _strip_leading_label(group_text, ("updated",))
            if not updated_text:
                updated_text = group_text
            continue

    # Legacy x-test selectors kept as a fallback for older cached HTML.
    if not pull_count_text:
        node = list_item.select_one("[x-test-pull-count]")
        pull_count_text = _normalize_text(node.get_text(" ", strip=True) if node else "")
    if not tag_count_text:
        node = list_item.select_one("[x-test-tag-count]")
        tag_count_text = _normalize_text(node.get_text(" ", strip=True) if node else "")
    if not updated_text:
        node = list_item.select_one("[x-test-updated]")
        updated_text = _normalize_text(node.get_text(" ", strip=True) if node else "")

    return pull_count_text, tag_count_text, updated_text


# Parse filter payloads from the Ollama search page.
def _parse_filter_payloads(soup: Any, search_query: OllamaSearchQuery) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    active_capabilities = set(search_query.capabilities)
    active_sort = search_query.sort_key or DEFAULT_SORT_KEY

    # Parse sort options from the primary desktop select first
    sort_options = soup.select("#desktop-sort-select option")
    if not sort_options:
        # Keep a generic fallback because Ollama has changed this control before
        sort_options = soup.select("select option")

    if sort_options:
        for index, option in enumerate(sort_options):
            value = _normalize_filter_key(option.get("value"))
            if not value:
                continue

            payloads.append(
                {
                    "key": f"{SORT_FILTER_PREFIX}{value}",
                    "title": _normalize_text(option.get_text(" ", strip=True)) or value.title(),
                    "kind": "sort",
                    "selected": value == active_sort,
                    "sortOrder": 10 + index,
                }
            )

    # Parse capability checkboxes exposed by the search UI
    for index, input_node in enumerate(soup.select("input[type='checkbox'][name='c'][value]")):
        value = _normalize_filter_key(input_node.get("value"))
        if not value:
            continue

        label = soup.select_one(f"label[for='{input_node.get('id', '')}']")
        title = _normalize_text(label.get_text(" ", strip=True) if label else value.title()) or value.title()

        payloads.append(
            {
                "key": f"{CAPABILITY_FILTER_PREFIX}{value}",
                "title": title,
                "kind": "capability",
                # Respect both the requested query and the page's own checked state
                "selected": value in active_capabilities or input_node.has_attr("checked"),
                "sortOrder": 100 + index,
            }
        )

    return payloads


# Parse model cards from the Ollama search page.
def _parse_search_items(soup: Any) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_slugs: set[str] = set()

    # Current library search uses a plain role=list grid. Keep legacy x-test cards
    # as a fallback so older cached HTML still parses if present.
    list_items = soup.select("ul[role='list'] > li")
    if not list_items:
        list_items = soup.select("li[x-test-model]")
    if not list_items:
        list_items = soup.select("ul.grid > li")

    for index, list_item in enumerate(list_items):
        # Resolve the card identity from its main library link
        anchor = list_item.select_one("a[href^='/library/'], a[href*='/library/']")
        if anchor is None:
            anchor = list_item.select_one("a[href]")
        if anchor is None:
            continue

        href = _normalize_text(anchor.get("href"))
        if not href.startswith("/"):
            continue

        if href.startswith("/library/"):
            slug = href.removeprefix("/library/").strip("/")
        else:
            # Ignore non-library links that can appear in the same list shell.
            continue

        # Cards are family-level only; skip accidental tag deep-links.
        if not slug or ":" in slug or slug in seen_slugs:
            continue

        # Read the visible card content (new markup first, legacy selectors second).
        title_node = (
            list_item.select_one("h2")
            or list_item.select_one("[x-test-search-response-title]")
        )
        title = _normalize_text(title_node.get_text(" ", strip=True) if title_node else "")

        header_container = list_item.select_one("div[title]") or anchor.find("div", attrs={"title": True})
        if not title and header_container is not None:
            title = _normalize_text(header_container.get("title"))

        summary_node = None
        if header_container is not None:
            summary_node = header_container.find("p")
        if summary_node is None:
            summary_node = list_item.select_one("p.max-w-lg")
        summary = _normalize_text(summary_node.get_text(" ", strip=True) if summary_node else "")

        tag_container = (
            list_item.select_one("div.flex.flex-wrap")
            or list_item.select_one("div.flex.flex-col > div.flex.flex-wrap")
        )
        if tag_container is not None:
            # Prefer direct badge chips; fall back to all spans inside the wrap.
            badge_nodes = [
                child for child in tag_container.children
                if getattr(child, "name", None) == "span"
            ]
            if not badge_nodes:
                badge_nodes = tag_container.select("span")
            tag_values = _deduplicate_preserving_order(
                [_normalize_text(node.get_text(" ", strip=True)) for node in badge_nodes]
            )
        else:
            tag_values = []

        # Drop labels that are really meta counters, not capability/size chips.
        tag_values = [
            value for value in tag_values
            if value and value.lower() not in {"pulls", "pull", "tags", "tag", "updated", "downloads", "download"}
        ]

        pull_count_text, tag_count_text, updated_text = _extract_search_card_meta(list_item)
        variant_count = _parse_int(tag_count_text)

        # Build the compact bridge card payload used by the ASLM catalog list
        items.append(
            {
                "resourceKey": f"ollama:{slug}",
                "categoryId": OLLAMA_CATEGORY_ID,
                "groupKey": OLLAMA_GROUP_KEY,
                "title": title or slug,
                "summary": summary,
                "provider": "Ollama",
                "version": "",
                "homepageUrl": f"https://ollama.com{href.split('?', 1)[0]}",
                "details": _build_details(
                    pull_count_text,
                    updated_text,
                ),
                "tags": _build_tags(tag_values),
                "variantCount": variant_count,
                "defaultVariantResourceKey": _variant_resource_key(slug, "latest"),
                "sortOrder": index,
            }
        )
        seen_slugs.add(slug)

    return items


# Load search results with cache fallback.
def _load_search_payload(
    query_text: str,
    filter_keys: list[str] | None,
    prefer_cached: bool,
    force_refresh: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    # Resolve the cache entry for the current search request
    search_query = _normalize_search_query(query_text, filter_keys)
    cache_path = _search_cache_path(search_query)
    cached = _read_cache(cache_path)

    # Serve the cached payload immediately when the caller requests it
    if cached and prefer_cached and not force_refresh:
        return list(cached.get("items", [])), list(cached.get("filters", [])), []

    # Keep cache-first catalog opens fast. If no real cache exists yet, ASLM's
    # background force-refresh pass will populate it from Ollama.
    if prefer_cached and not force_refresh:
        return [], _build_default_filter_payloads(search_query), []

    try:
        # Refresh the cache from the live Ollama search page
        html = _request_text(OLLAMA_SEARCH_URL, _build_search_params(search_query))
        soup = _make_soup(html)
        filters = _parse_filter_payloads(soup, search_query) or _build_default_filter_payloads(search_query)
        items = _parse_search_items(soup)
        warnings: list[str] = []
        if not items and html:
            warnings.append(
                "Ollama search returned no parseable model cards. "
                "The library page markup may have changed."
            )
        # Avoid overwriting a good cache with an empty parse result.
        if items or not cached:
            _write_cache(cache_path, {"items": items, "filters": filters})
        elif cached and not items:
            return (
                list(cached.get("items", [])),
                list(cached.get("filters", [])) or filters,
                warnings + ["Kept previous cached Ollama search data after empty live parse."],
            )
        return items, filters, warnings
    except Exception as exc:
        # Fall back to the previous cache when live refresh fails
        if cached:
            return (
                list(cached.get("items", [])),
                list(cached.get("filters", [])),
                [f"Using cached Ollama search data: {exc}"],
            )
        raise


# Detail parsing helpers

# Convert the provider's decimal (GB) or binary (GiB) size label to whole bytes.
# A missing/cloud size remains unknown; an explicit zero is a known empty download.
# Sizes outside the bridge's int64 byte range cannot be represented and remain unknown.
def _parse_size_bytes(value: str) -> int | None:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(B|[KMGTPE]i?B)", _normalize_text(value), re.IGNORECASE)
    if match is None:
        return None
    number, unit = match.groups()
    unit = unit.upper()
    exponent = "BKMGTPE".index(unit[0])
    multiplier = (1024 if "I" in unit else 1000) ** exponent
    size = Decimal(number) * multiplier
    return int(size) if size <= MAX_VARIANT_SIZE_BYTES else None


# Parse available model variants from the model page.
def _parse_variant_payloads(soup: BeautifulSoup, slug: str) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    seen_models: set[str] = set()
    page_path = _resolve_model_page_path(slug)

    # Variant links reuse the model page path and append the tag after a colon
    prefix = f"{page_path}:"

    # Capture the mobile variant rows first because they include fallback metadata
    mobile_rows: dict[str, dict[str, str]] = {}
    for mobile_anchor in soup.find_all("a", href=True):
        href = str(mobile_anchor.get("href", "")).strip()
        if not href.startswith(prefix):
            continue

        css_classes = " ".join(str(css_class) for css_class in mobile_anchor.get("class", []))
        if "sm:hidden" not in css_classes:
            continue

        secondary_text = mobile_anchor.find("p", class_=lambda value: value and "text-neutral-500" in value)
        # The mobile row often keeps metadata that is omitted from the desktop grid
        detail_text = _normalize_separator(_normalize_text(secondary_text.get_text(" ", strip=True) if secondary_text else ""))
        segments = [segment.strip() for segment in detail_text.split("|") if segment.strip()]
        mobile_rows[href] = {
            "size": segments[0] if len(segments) > 0 else "",
            "input": segments[2] if len(segments) > 2 else "",
        }

    # Parse the main grid and backfill missing fields from the mobile rows
    for row in soup.select("div.group.sm\\:grid"):
        anchor = row.find("a", href=lambda value: isinstance(value, str) and value.startswith(prefix))
        if anchor is None:
            continue

        href = str(anchor.get("href", "")).strip()
        model_name = href.removeprefix("/library/").lstrip("/") if href.startswith("/library/") else href.lstrip("/")
        # Skip duplicates when the same tag appears in multiple responsive containers
        if model_name in seen_models:
            continue

        tag = model_name.partition(":")[2].strip()
        columns = row.select("p.col-span-2")
        size_text = _normalize_text(columns[0].get_text(" ", strip=True) if len(columns) > 0 else "")
        input_text = _normalize_text(columns[2].get_text(" ", strip=True) if len(columns) > 2 else "")

        mobile_row = mobile_rows.get(href, {})
        if not size_text:
            size_text = mobile_row.get("size", "")
        if not input_text:
            input_text = mobile_row.get("input", "")
        size_bytes = _parse_size_bytes(size_text)
        summary = input_text if input_text else ""

        variants.append(
            {
                "resourceKey": _variant_resource_key(slug, tag),
                "title": tag or "latest",
                "summary": summary,
                "size": size_bytes if size_bytes is not None else 0,
                "hasSize": size_bytes is not None,
                "homepageUrl": f"https://ollama.com{prefix}{tag}" if tag else f"https://ollama.com{page_path}",
                "sortOrder": len(variants),
            }
        )
        seen_models.add(model_name)

    return variants


# Extract the raw markdown readme from the page.
def _extract_readme_markdown(soup: BeautifulSoup) -> str:
    editor = soup.select_one("textarea#editor")
    if editor is None:
        return ""

    markdown = unescape(editor.get_text("\n", strip=False))
    lines = [line.rstrip() for line in markdown.splitlines()]
    return "\n".join(lines).strip()


# Extract and cache the rendered HTML readme block.
def _extract_readme_html_file(slug: str, soup: BeautifulSoup, page_url: str) -> str:
    readme_node = soup.select_one("#readme #display") or soup.select_one("#display")
    if readme_node is None:
        return ""

    return _create_html_block_file(
        slug=slug,
        block_id="readme",
        title=f"{slug} Readme",
        node_html=readme_node.decode_contents(),
        page_url=page_url,
    )


# Extract pull/download and updated labels from a model detail page.
def _extract_detail_page_meta(soup: Any) -> tuple[str, str]:
    pull_count_text = ""
    updated_text = ""

    # Legacy x-test markers (older library pages).
    pull_count_node = soup.select_one("[x-test-pull-count]")
    updated_node = soup.select_one("[x-test-updated]")
    if pull_count_node is not None:
        pull_count_text = _normalize_text(pull_count_node.get_text(" ", strip=True))
    if updated_node is not None:
        updated_text = _normalize_text(updated_node.get_text(" ", strip=True))

    # Current library pages use plain "18.7M Downloads" / "Updated 2 weeks ago" groups.
    # Prefer leaf span groups; parent p.flex often concatenates both labels.
    if not pull_count_text or not updated_text:
        for group in soup.select("span.flex.items-center"):
            group_text = _normalize_text(group.get_text(" ", strip=True))
            if not group_text or len(group_text) > 64:
                continue
            lower = group_text.lower()

            if not pull_count_text and re.search(r"\b(downloads?|pulls?)\b", lower):
                pull_count_text = _strip_trailing_label(
                    group_text,
                    ("downloads", "download", "pulls", "pull"),
                )
                continue
            if not updated_text and ("updated" in lower or re.search(r"\bago\b", lower) or
                                     lower in {"today", "yesterday", "just now", "now"}):
                updated_text = _strip_leading_label(group_text, ("updated",))
                if not updated_text:
                    updated_text = group_text

    return pull_count_text, updated_text


# Parse full item details from the model page.
def _parse_item_detail(slug: str, html: str) -> dict[str, Any]:
    soup = _make_soup(html)
    page_url = _resolve_model_page_url(slug)

    # Read the primary page metadata
    title = _normalize_text(soup.title.get_text(" ", strip=True) if soup.title else slug)
    summary_node = soup.select_one("textarea#summary-textarea")
    summary = _normalize_text(summary_node.get_text(" ", strip=True) if summary_node else "")

    pull_count_text, updated_text = _extract_detail_page_meta(soup)
    # Resolve variants; host platform filtering is applied later so cached details
    # stay reusable and are filtered at serve time.
    variants = _parse_variant_payloads(soup, slug)
    details = _build_details(
        pull_count_text,
        updated_text,
    )
    default_variant = _select_default_variant_key(variants)

    # Prefer the rendered readme block because it matches the source page styling
    # Keep markdown as a fallback for pages that do not expose the rendered section
    readme_html_file = _extract_readme_html_file(slug, soup, page_url)
    readme = _extract_readme_markdown(soup)
    blocks: list[dict[str, Any]] = []

    if readme_html_file:
        blocks.append(
            {
                "id": "readme",
                "title": "Readme",
                "format": "html-file",
                "content": "",
                "contentUrl": readme_html_file,
                "sourceUrl": page_url,
                "sortOrder": 10,
            }
        )
    elif readme:
        blocks.append(
            {
                "id": "readme",
                "title": "Readme",
                "format": "markdown",
                "content": readme,
                "contentUrl": "",
                "sourceUrl": page_url,
                "sortOrder": 10,
            }
        )

    return {
        "resourceKey": f"ollama:{slug}",
        "categoryId": OLLAMA_CATEGORY_ID,
        "groupKey": OLLAMA_GROUP_KEY,
        "title": title or slug,
        "summary": summary,
        "provider": "Ollama",
        "version": "",
        "homepageUrl": page_url,
        "details": details,
        # Capability tags remain inherited from list_items; this page does not always expose them.
        "defaultVariantResourceKey": default_variant,
        "variants": variants,
        "blocks": blocks,
    }


# Sanitize detail payloads loaded from cache or network.
def _sanitize_detail_payload(detail: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(detail or {})
    blocks = normalized.get("blocks")
    if isinstance(blocks, list):
        # Older cached payloads may still contain blocks removed from the new UI
        normalized["blocks"] = [
            block for block in blocks
            if _normalize_filter_key((block or {}).get("id")) != "quick-start"
        ]

    # Drop MLX or classic variants that cannot run on this host OS.
    return _apply_platform_variant_filter(normalized)


# Load detail payloads with cache fallback.
def _load_item_detail(
    slug: str,
    prefer_cached: bool,
    force_refresh: bool,
) -> tuple[dict[str, Any], list[str]]:
    # Resolve the cache entry for the requested model
    cache_path = _detail_cache_path(slug)
    cached = _read_cache(cache_path) if prefer_cached or not force_refresh else None

    # Serve sanitized cached details immediately when requested
    if cached and prefer_cached and not force_refresh:
        detail = _retheme_detail_html_blocks(_sanitize_detail_payload(dict(cached)), slug)
        return detail, []

    try:
        # Refresh the cache from the live model page
        detail = _retheme_detail_html_blocks(
            _sanitize_detail_payload(_parse_item_detail(slug, _request_text(_resolve_model_page_url(slug)))),
            slug,
        )

        _write_cache(cache_path, detail)
        return detail, []
    except Exception as exc:
        # Fall back to cached details when refresh fails
        if cached and prefer_cached:
            detail = _retheme_detail_html_blocks(_sanitize_detail_payload(dict(cached)), slug)
            return detail, [f"Using cached details for {slug}: {exc}"]
        raise


# Manifest builders

# Build an install manifest for a selected resource.
def _build_install_manifest(resource_key: str) -> dict[str, Any]:
    model_name = _resource_key_to_slug(resource_key)
    if not model_name:
        raise ValueError("Missing resourceKey for resolve_install.")
    _assert_resource_allowed_on_host(resource_key)

    return {
        "resourceKey": f"ollama:{model_name}",
        "categoryId": OLLAMA_CATEGORY_ID,
        "title": model_name,
        "version": "",
        "targetRef": OLLAMA_TARGET_REF,
        "actions": [
            {
                "type": "ollama_pull",
                "title": f"Pull {model_name} from the Ollama library",
                "engineId": "ollama-service",
                "targetRef": OLLAMA_TARGET_REF,
                "model": model_name,
            }
        ],
    }


# Build an uninstall manifest for a selected resource.
def _build_uninstall_manifest(resource_key: str) -> dict[str, Any]:
    model_name = _resource_key_to_slug(resource_key)
    if not model_name:
        raise ValueError("Missing resourceKey for resolve_uninstall.")
    # Uninstall is allowed for any previously installed ref, including formats
    # that are no longer offered for this host (e.g. leftover MLX on Windows).
    # Only block clearly malformed keys; platform install guard stays on install.

    return {
        "resourceKey": f"ollama:{model_name}",
        "categoryId": OLLAMA_CATEGORY_ID,
        "title": model_name,
        "version": "",
        "targetRef": OLLAMA_TARGET_REF,
        "actions": [
            {
                "type": "ollama_remove",
                "title": f"Remove {model_name} from the local Ollama library",
                "engineId": "ollama-service",
                "targetRef": OLLAMA_TARGET_REF,
                "model": model_name,
            }
        ],
    }


# Bridge operation handlers

# Handle the list_categories operation.
def _handle_list_categories() -> dict[str, Any]:
    return _response(categories=[_build_ollama_category()])


# Handle the list_items operation.
def _handle_list_items(
    category_id: str,
    query_text: str,
    filter_keys: list[str] | None,
    prefer_cached: bool,
    force_refresh: bool,
) -> dict[str, Any]:
    if category_id != OLLAMA_CATEGORY_ID:
        return _response(success=False, error=f"Unsupported categoryId: {category_id}")

    items, filters, warnings = _load_search_payload(query_text, filter_keys, prefer_cached, force_refresh)
    return _response(items=items, filters=filters, warnings=warnings)


# Handle the describe_item operation.
def _handle_describe_item(
    category_id: str,
    resource_key: str,
    prefer_cached: bool,
    force_refresh: bool,
) -> dict[str, Any]:
    if category_id != OLLAMA_CATEGORY_ID:
        return _response(success=False, error=f"Unsupported categoryId: {category_id}")

    slug = _resource_key_to_slug(resource_key).split(":", 1)[0].strip()
    if not slug:
        return _response(success=False, error="Missing resourceKey for describe_item.")

    item_detail, warnings = _load_item_detail(slug, prefer_cached, force_refresh)
    return _response(item_detail=item_detail, warnings=warnings)


# Handle the resolve_install operation.
def _handle_resolve_install(category_id: str, resource_key: str) -> dict[str, Any]:
    if category_id != OLLAMA_CATEGORY_ID:
        return _response(success=False, error=f"Unsupported categoryId: {category_id}")

    try:
        return _response(install_manifest=_build_install_manifest(resource_key))
    except ValueError as exc:
        return _response(success=False, error=str(exc))


# Handle the resolve_uninstall operation.
def _handle_resolve_uninstall(category_id: str, resource_key: str) -> dict[str, Any]:
    if category_id != OLLAMA_CATEGORY_ID:
        return _response(success=False, error=f"Unsupported categoryId: {category_id}")

    try:
        return _response(uninstall_manifest=_build_uninstall_manifest(resource_key))
    except ValueError as exc:
        return _response(success=False, error=str(exc))


# Dispatch helpers

# Route a bridge request to the matching handler.
def dispatch(request: dict[str, Any]) -> dict[str, Any]:
    # Normalize the incoming request fields first
    operation = _normalize_text(request.get("operation")).lower()
    category_id = _normalize_text(request.get("categoryId"))
    resource_key = _normalize_text(request.get("resourceKey"))
    query_text = _normalize_text(request.get("queryText"))
    raw_filters = request.get("filters") if isinstance(request.get("filters"), list) else []
    filter_keys = [_normalize_text(value) for value in raw_filters if _normalize_text(value)]
    prefer_cached = bool(request.get("preferCached"))
    force_refresh = bool(request.get("forceRefresh"))

    # Route the request explicitly so unsupported operations fail clearly
    if operation == "list_categories":
        return _handle_list_categories()
    if operation == "list_items":
        return _handle_list_items(category_id, query_text, filter_keys, prefer_cached, force_refresh)
    if operation == "describe_item":
        return _handle_describe_item(category_id, resource_key, prefer_cached, force_refresh)
    if operation == "resolve_install":
        return _handle_resolve_install(category_id, resource_key)
    if operation == "resolve_uninstall":
        return _handle_resolve_uninstall(category_id, resource_key)

    return _response(success=False, error=f"Unsupported downloads bridge operation: {operation or '<empty>'}")


# CLI entry point

# Execute the bridge in stdin/stdout mode.
def run_cli() -> int:
    try:
        response = dispatch(_read_request())
    except Exception as exc:
        response = _response(success=False, error=str(exc))

    print(json.dumps(response, ensure_ascii=True))
    return 0
