# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import html
import re
from typing import Any
from urllib.parse import parse_qs, urlparse

from core.fetch.thread_pool import io_pool as _io_pool

# Algolia HN API endpoint template for one item; it returns the whole item tree (story plus
# nested comments) in a single keyless call.
_ALGOLIA_ITEM_URL = "https://hn.algolia.com/api/v1/items/{item_id}"

# Rendering caps: enough thread to be useful, small enough to survive compaction.
_MAX_COMMENTS = 40
_MAX_DEPTH = 3
_MAX_COMMENT_CHARS = 1_200


# Return True when url is a Hacker News item page (news.ycombinator.com/item?id=NNN).
def is_hackernews_item_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.netloc.lower().removeprefix("www.") != "news.ycombinator.com":
        return False
    if parsed.path.rstrip("/") != "/item":
        return False
    ids = parse_qs(parsed.query).get("id") or []
    return bool(ids and ids[0].isdigit())


# Return the numeric item id from an item URL, or None.
def hackernews_item_id(url: str) -> str | None:
    ids = parse_qs(urlparse(url).query).get("id") or []
    return ids[0] if ids and ids[0].isdigit() else None


# Convert an Algolia HTML fragment (comment/story text) to plain text.
def _strip_html_fragment(fragment: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", fragment or "", flags=re.IGNORECASE)
    text = re.sub(r"</?p\s*>", "\n\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<pre>(.*?)</pre>", lambda m: "\n```text\n" + m.group(1) + "\n```\n",
                  text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", "", text)
    return html.unescape(text).strip()


# Fetch one Algolia item tree as a dict (blocking; runs in the shared I/O pool).
def _algolia_get_sync(item_id: str, timeout: int) -> dict[str, Any]:
    from curl_cffi import requests as _r

    resp = _r.get(
        _ALGOLIA_ITEM_URL.format(item_id=item_id),
        timeout=timeout,
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, dict) else {}


# Flatten the nested children tree depth-first into (depth, node) pairs.
def _walk_comments(node: dict[str, Any], depth: int, out: list[tuple[int, dict[str, Any]]]) -> None:
    for child in node.get("children") or []:
        if not isinstance(child, dict) or len(out) >= _MAX_COMMENTS:
            return
        if child.get("text"):
            out.append((depth, child))
        if depth < _MAX_DEPTH:
            _walk_comments(child, depth + 1, out)


# Render an Algolia item tree as read_page markdown: a metadata header, the item text and the
# flattened comment thread. Returns an "Error: ..." string when the tree has no item data.
def render_hackernews_markdown(data: dict[str, Any], url: str) -> str:
    if not data or not data.get("id"):
        return f"Error: Hacker News API returned no item data for: {url}"

    # Header: a story has a title, while a bare comment link is shown as a comment thread.
    title = str(data.get("title") or "").strip()
    is_story = bool(title)
    lines = [f"# {title or 'Hacker News comment thread'}", ""]
    lines.append("**Site:** news.ycombinator.com")
    lines.append(f"**URL:** {url}")
    if data.get("author"):
        lines.append(f"**Author:** {data['author']}")
    if data.get("created_at"):
        lines.append(f"**Date:** {data['created_at']}")
    if data.get("points") is not None:
        lines.append(f"**Points:** {data['points']}")
    if is_story and data.get("url"):
        lines.append(f"**Story Link:** {data['url']}")

    # Flatten the comment tree (capped by count and depth) and note how many are shown.
    comments: list[tuple[int, dict[str, Any]]] = []
    _walk_comments(data, 0, comments)
    lines.append(f"**Comments shown:** {len(comments)}")
    lines.extend(["", "---", ""])

    # The item's own text (story body), if any.
    if data.get("text"):
        lines.extend([_strip_html_fragment(str(data["text"])), ""])

    # One bullet per comment, indented by its depth in the thread.
    if comments:
        lines.extend(["## Comments", ""])
        for depth, node in comments:
            body = _strip_html_fragment(str(node.get("text") or ""))[:_MAX_COMMENT_CHARS]
            indent = "  " * depth
            meta = str(node.get("author") or "?")
            if node.get("created_at"):
                meta += f" | {str(node['created_at'])[:10]}"
            lines.append(f"{indent}- **{meta}**: {body}")
            lines.append("")

    return "\n".join(lines).strip()


# Fetch a Hacker News item through the Algolia mirror and return it as markdown for read_page and
# the web_search inline parse. news.ycombinator.com itself rate-limits by IP (429) so aggressively
# that neither plain HTTP nor a real browser survives a search-time fetch, while the Algolia
# mirror returns the full item tree in one keyless call; this is therefore the terminal fetch
# method for HN, with the same contract as the Stack Exchange API handler: failures come back as
# an "Error: ..." string and never raise.
async def fetch_hackernews_item(url: str, timeout: float = 20.0) -> str:
    item_id = hackernews_item_id(url)
    if not item_id:
        return f"Error: Unsupported Hacker News URL: {url}"
    loop = asyncio.get_running_loop()
    try:
        data = await loop.run_in_executor(
            _io_pool, lambda: _algolia_get_sync(item_id, max(5, int(timeout)))
        )
    except Exception as exc:  # noqa: BLE001 — terminal handler: report, never raise
        return f"Error: Hacker News API fetch failed for {url}: {exc}"
    return render_hackernews_markdown(data, url)
