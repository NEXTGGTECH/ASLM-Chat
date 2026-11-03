# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from django import template
from django.templatetags.static import static as django_static
from django.utils.safestring import mark_safe

from Apps.UI import STATIC_CACHE_VERSION
from Apps.UI.locale_catalog import translate

# Template tag library of this module.
register = template.Library()
# Root folder of the local ES modules that the import map covers.
_STATIC_JS_ROOT = Path(__file__).resolve().parents[1] / "static" / "js"


# Append the per-process cache-bust query (``v=<STATIC_CACHE_VERSION>``) to a URL.
def append_static_cache_version(url: str) -> str:
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={STATIC_CACHE_VERSION}"


# Resolve a static asset URL with the per-process cache-bust query. It shadows Django's ``static`` tag
# of the same name in templates that load this library.
@register.simple_tag
def static(path: str) -> str:
    return append_static_cache_version(django_static(path))


# Build the JSON of an import map that maps every local ES module URL to the same URL carrying the
# current backend cache version. The result is cached for the process lifetime, and ``<`` is escaped so
# the JSON can be embedded in a script tag.
@lru_cache(maxsize=1)
def _static_import_map_json() -> str:
    imports: dict[str, str] = {}
    for module_path in sorted(_STATIC_JS_ROOT.rglob("*.js")):
        relative_path = module_path.relative_to(_STATIC_JS_ROOT.parent).as_posix()
        source_url = django_static(relative_path)
        imports[source_url] = append_static_cache_version(source_url)
    return json.dumps(
        {"imports": imports},
        ensure_ascii=False,
        separators=(",", ":"),
    ).replace("<", "\\u003c")


# Return a process-versioned import map for the complete local JS graph, marked safe for direct
# output in a template.
@register.simple_tag
def static_import_map() -> str:
    return mark_safe(_static_import_map_json())


# Translate ``key`` using the effective locale from the template context.
@register.simple_tag(takes_context=True)
def t(context: template.Context, key: str) -> str:
    locale = context.get("host_language_effective")
    return translate(key, locale=locale)


# Translate ``key`` with ``{name}`` placeholders filled from the tag's keyword arguments.
@register.simple_tag(takes_context=True)
def t_param(context: template.Context, key: str, **kwargs: str) -> str:
    locale = context.get("host_language_effective")
    return translate(key, locale=locale, **kwargs)
