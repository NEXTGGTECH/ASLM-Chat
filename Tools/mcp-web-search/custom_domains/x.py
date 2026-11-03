# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import json as _json
import re
from urllib.parse import quote as _quote, urlparse

from core.fetch.constants import DEFAULT_UA as _UA
from core.fetch.thread_pool import io_pool as _io_pool
from custom_domains.retail_common import strip_html_fragment


# Matches the status id in an X/Twitter post path, including the /i/web/status/ and /user/web/status/ forms.
_X_POST_PATTERN = re.compile(r"/(?:(?:i|[^/]+)/web/)?status/(\d+)|/[^/]+/status/(\d+)")


# URL helpers

# Normalize URL host (strip www./m. prefixes).
def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.").removeprefix("m.")


# Extract numeric status id from an X/Twitter post URL path.
def x_status_id(url: str) -> str | None:
    match = _X_POST_PATTERN.search(urlparse(url).path)
    if not match:
        return None
    return match.group(1) or match.group(2)


# Return True when the URL is an x.com/twitter.com status permalink.
def is_x_post(url: str) -> bool:
    return _host(url) in {"twitter.com", "x.com"} and x_status_id(url) is not None


# Payload parsing

# Format syndication API JSON into readable markdown.
def parse_x_syndication_payload(data: dict, url: str) -> str:
    # Post text and author/date fields.
    text = str(data.get("text") or "").strip()
    user = data.get("user") if isinstance(data.get("user"), dict) else {}
    author_name = str(user.get("name") or "").strip()
    author_handle = str(user.get("screen_name") or "").strip()
    created_at = str(data.get("created_at") or "").strip()

    # Header: author, creation time, URL and the post text.
    lines: list[str] = ["# X Post"]
    meta: list[str] = []
    if author_name or author_handle:
        if author_name and author_handle:
            meta.append(f"Author: {author_name} (@{author_handle})")
        elif author_handle:
            meta.append(f"Author: @{author_handle}")
        else:
            meta.append(f"Author: {author_name}")
    if created_at:
        meta.append(f"Created: {created_at}")
    if meta:
        lines.extend(meta)
    lines.append(f"URL: {url}")
    lines.append("")
    if text:
        lines.append(text)

    # Expanded links from the post entities (falling back to the short URL).
    entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
    urls = entities.get("urls") if isinstance(entities.get("urls"), list) else []
    expanded = [str(item.get("expanded_url") or item.get("url") or "").strip() for item in urls if isinstance(item, dict)]
    expanded = [item for item in expanded if item]
    if expanded:
        lines.append("")
        lines.append("Links:")
        lines.extend(f"- {item}" for item in expanded)

    # Media URLs (images and video thumbnails).
    media = data.get("mediaDetails") if isinstance(data.get("mediaDetails"), list) else []
    media_urls: list[str] = []
    for item in media:
        if not isinstance(item, dict):
            continue
        candidate = str(item.get("media_url_https") or item.get("media_url") or "").strip()
        if candidate:
            media_urls.append(candidate)
    if media_urls:
        lines.append("")
        lines.append("Media:")
        lines.extend(f"- {item}" for item in media_urls)

    return "\n".join(lines).strip()


# Format oEmbed HTML payload into readable markdown.
def parse_x_oembed_payload(data: dict, url: str) -> str:
    author_name = str(data.get("author_name") or "").strip()
    author_url = str(data.get("author_url") or "").strip()
    html_block = str(data.get("html") or "")
    text = strip_html_fragment(html_block)

    lines = ["# X Post"]
    if author_name:
        lines.append(f"Author: {author_name}")
    if author_url:
        lines.append(f"Author URL: {author_url}")
    lines.append(f"URL: {url}")
    lines.append("")
    if text:
        lines.append(text)
    return "\n".join(lines).strip()


# Fetching

# GET JSON from syndication or oEmbed endpoint via curl_cffi (blocking, TLS impersonation).
def _x_fetch_json(endpoint: str, timeout: int) -> dict | None:
    from curl_cffi import requests as _r

    resp = _r.get(
        endpoint,
        impersonate="chrome124",
        timeout=timeout,
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": _UA,
        },
    )
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        # Fall back to parsing the raw body when the response's own JSON decoding fails.
        return _json.loads(resp.text)


# Fetch post via syndication API, then oEmbed fallback; return markdown or error string.
async def fetch_x_post(url: str, timeout: float) -> str:
    status_id = x_status_id(url)
    if not status_id:
        return f"Error: Could not extract X/Twitter status ID from: {url}"

    loop = asyncio.get_running_loop()

    # Try syndication JSON first, then publish.twitter.com oEmbed.
    def _sync() -> str | None:
        syndication_url = f"https://cdn.syndication.twimg.com/tweet-result?id={status_id}&token=x"
        try:
            data = _x_fetch_json(syndication_url, max(10, int(timeout)))
            if isinstance(data, dict) and data.get("text"):
                return parse_x_syndication_payload(data, url)
        except Exception:
            # Any failure of the syndication call gives up on this post (no oEmbed attempt).
            return None

        oembed_url = "https://publish.twitter.com/oembed?omit_script=1&url=" + _quote(url, safe="")
        try:
            data = _x_fetch_json(oembed_url, max(10, int(timeout)))
            if isinstance(data, dict) and data.get("html"):
                return parse_x_oembed_payload(data, url)
        except Exception:
            # oEmbed was the last option; report that nothing could be fetched.
            return None
        return None

    result = await loop.run_in_executor(_io_pool, _sync)
    return result or f"Error: Could not fetch X/Twitter post content from: {url}"


# Handler

from custom_domains.base import FetchContext, PageResult


# Unified handler: fetch an X/Twitter post via syndication/oEmbed APIs.
class XHandler:
    name = "x"
    fallback_to_generic = False

    # read_page only; web_search keeps X posts as snippets instead of parsing them inline.
    scope = "read_page"

    # Return True for x.com/twitter.com status permalinks.
    def matches(self, url: str) -> bool:
        return is_x_post(url)

    # Fetch and format the post as markdown.
    async def read(self, url: str, ctx: FetchContext) -> PageResult:
        markdown = await fetch_x_post(url, ctx.timeout)
        ok = bool(markdown) and not markdown.startswith("Error:")
        return PageResult(
            markdown=markdown or f"Error: Could not fetch X/Twitter post content from: {url}",
            ok=ok,
            method="x_syndication",
            error="" if ok else (markdown or "x fetch failed"),
        )


# Handler instance registered in custom_domains/__init__.py.
HANDLER = XHandler()
