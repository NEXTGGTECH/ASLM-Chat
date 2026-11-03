# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Any


# Provider names grouped by the directives they can execute. Google (and SerpApi, which proxies it)
# take exact after:/before: dates; the rolling-date engines only have coarse time controls; the
# remaining sets list who understands filetype:, intitle: and inurl:.
_GOOGLE_DIALECTS = frozenset({"google", "serpapi"})
_ROLLING_DATE_DIALECTS = frozenset({"duckduckgo", "startpage", "brave"})
_FILETYPE_DIALECTS = frozenset({
    "google", "serpapi", "startpage", "duckduckgo", "brave",
})
_TITLE_DIALECTS = frozenset({
    "google", "serpapi", "startpage", "duckduckgo", "brave",
})
_URL_DIALECTS = frozenset({"google", "serpapi", "startpage", "duckduckgo"})


# One provider-ready query plus diagnostics about lossy translations: the optional rolling time limit
# and the names of the operators the provider could not express (and so were left out).
@dataclass(frozen=True, slots=True)
class ProviderQuery:
    query: str
    timelimit: str | None = None
    omitted_operators: tuple[str, ...] = ()


# Collapse all whitespace (including newlines) to single spaces and trim the text.
def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").replace("\r", " ").replace("\n", " ")).strip()


# Return the cleaned text wrapped in double quotes, with embedded quotes removed.
def _quoted(value: Any) -> str:
    return f'"{_clean(value).replace(chr(34), "")}"'


# Return the cleaned text, quoted when it contains a space so it stays one search term.
def _value(value: Any) -> str:
    clean = _clean(value).replace('"', "")
    return _quoted(clean) if " " in clean else clean


# Join the non-empty rendered values with the OR token inside parentheses (a single value is
# returned bare, no values give an empty string).
def _or_group(values: Any, *, token: str = "OR", render=_value) -> str:
    rendered = [render(value) for value in list(values or []) if _clean(value)]
    if not rendered:
        return ""
    return rendered[0] if len(rendered) == 1 else f"({' {} '.format(token).join(rendered)})"


# Approximate an open-ended lower date bound with the providers' rolling date controls: return
# d/w/m/y (day, week, month, year) for the smallest window that covers it, or None when there is
# no usable lower-only bound or it is older than a year.
def _rolling_timelimit(after: str, before: str, *, today: dt.date) -> str | None:
    if not after or before:
        return None
    try:
        lower = dt.date.fromisoformat(after)
    except ValueError:
        return None
    days = (today - lower).days
    if days < 0:
        return None
    if days <= 1:
        return "d"
    if days <= 7:
        return "w"
    if days <= 31:
        return "m"
    if days <= 366:
        return "y"
    return None


# Compile one normalized search intent into the query syntax of one engine/provider.
# The advanced MCP schema keeps operators structured, but the canonical query shown in the UI uses
# Google's operator dialect; sending that string unchanged to every engine would turn unsupported
# directives (especially after:/before:) into accidental search terms. This is the boundary where
# the intent becomes the dialect a provider can actually execute.
# Unknown providers receive the conservative common subset. Exact date bounds are emitted only for
# Google syntax; rolling-date engines receive their native coarse time control, while other
# engines omit the directive instead of searching for its literal text. The canonical fallback is
# used unchanged for legacy calls that do not carry a structured operator map.
def compile_provider_query(
    text: str,
    operators: dict[str, Any] | None,
    provider: str,
    *,
    fallback_query: str = "",
    timelimit: str | None = None,
    today: dt.date | None = None,
) -> ProviderQuery:
    # Normalize the provider name; without structured operators use the fallback query as is.
    provider = _clean(provider).lower().removeprefix("hosted:")
    ops = dict(operators or {})
    if not ops:
        return ProviderQuery(_clean(fallback_query or text), timelimit)

    parts: list[str] = [_clean(text)]
    omitted: list[str] = []
    or_token = "|" if provider == "yandex" else "OR"

    # Phrase, OR and exclusion operators are common to every provider.
    parts.extend(_quoted(value) for value in ops.get("exact_phrases", []) if _clean(value))

    if group := _or_group(ops.get("or_terms"), token=or_token):
        parts.append(group)
    for values in ops.get("or_groups", []) or []:
        if group := _or_group(values, token=or_token):
            parts.append(group)

    parts.extend(f"-{_value(value)}" for value in ops.get("exclude_terms", []) if _clean(value))

    # Site restrictions.
    if group := _or_group(
        ops.get("site_include"), token=or_token,
        render=lambda value: f"site:{_clean(value)}",
    ):
        parts.append(group)
    parts.extend(f"-site:{_clean(value)}" for value in ops.get("site_exclude", []) if _clean(value))

    # File types: Yandex uses mime:, supporting providers use filetype:, others get the bare extension.
    file_prefix = "mime:" if provider == "yandex" else "filetype:"
    file_render = (
        (lambda value: f"{file_prefix}{_clean(value).lstrip('.')}")
        if provider == "yandex" or provider in _FILETYPE_DIALECTS
        else (lambda value: _clean(value).lstrip("."))
    )
    if group := _or_group(
        ops.get("file_types"), token=or_token,
        render=file_render,
    ):
        parts.append(group)

    # Title and URL terms keep their directive only where the provider supports it; otherwise they
    # degrade to plain search terms.
    title_prefix = "intitle:"
    for value in ops.get("title_terms", []) or []:
        if _clean(value):
            parts.append(
                f"{title_prefix}{_value(value)}"
                if provider in _TITLE_DIALECTS
                else _value(value)
            )
    url_prefix = "url:" if provider == "yandex" else "inurl:"
    for value in ops.get("url_terms", []) or []:
        if _clean(value):
            parts.append(
                f"{url_prefix}{_value(value)}"
                if provider == "yandex" or provider in _URL_DIALECTS
                else _value(value)
            )

    # Date bounds: exact for Google and Yandex syntax, otherwise a rolling time limit when the
    # provider has one, and the unexpressible bounds are reported as omitted.
    after = _clean(ops.get("after"))
    before = _clean(ops.get("before"))
    if provider in _GOOGLE_DIALECTS:
        if after:
            parts.append(f"after:{after}")
        if before:
            parts.append(f"before:{before}")
    elif provider == "yandex":
        if after:
            parts.append(f"date:>{after.replace('-', '')}")
        if before:
            parts.append(f"date:<{before.replace('-', '')}")
    elif after or before:
        native = (
            _rolling_timelimit(after, before, today=today or dt.date.today())
            if provider in _ROLLING_DATE_DIALECTS
            else None
        )
        timelimit = native or timelimit
        if after and native is None:
            omitted.append("after")
        if before:
            omitted.append("before")

    # Assemble the final query, falling back to the canonical query when nothing was produced.
    query = " ".join(part for part in parts if part).strip()
    return ProviderQuery(
        query=query or _clean(fallback_query),
        timelimit=timelimit,
        omitted_operators=tuple(omitted),
    )
