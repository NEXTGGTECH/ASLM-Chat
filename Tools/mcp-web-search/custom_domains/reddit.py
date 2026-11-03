# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any
from urllib.parse import urlparse, urlunparse

from core.fetch.thread_pool import io_pool as _io_pool
from custom_domains.reddit_parse import try_extract_markdown

logger = logging.getLogger("custom_domains.reddit")

# Matches a Reddit thread comments page URL (any host variant of reddit.com).
_REDDIT_PATTERN = re.compile(r"reddit\.com/r/[^/]+/comments/")


# URL helpers

# Return True when the URL looks like a Reddit thread comments page.
def is_reddit(url: str) -> bool:
    return bool(_REDDIT_PATTERN.search(url))


# Build the thread .json endpoint (limit/depth query for comments). host overrides the
# netloc so the same thread can be retried on old.reddit.com as a fallback.
def reddit_json_url(url: str, host: str | None = None) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith(".json"):
        path = path[: -len(".json")]
    path += ".json"
    return urlunparse((parsed.scheme, host or parsed.netloc, path, "", "limit=50&depth=3", ""))


# Thread page URL without the .json suffix (used as Referer / page-render fallback).
def reddit_thread_url(url: str, host: str | None = None) -> str:
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path.endswith(".json"):
        path = path[: -len(".json")]
    return urlunparse((parsed.scheme, host or parsed.netloc, path, "", "", ""))


# Content extraction

# Format Reddit thread JSON as markdown (post + nested comments), capped at 15,000 characters.
def reddit_data_to_markdown(data: list[Any], url: str) -> str:
    lines: list[str] = []

    # Post header: subreddit, author, score, title and self text.
    try:
        post = data[0]["data"]["children"][0]["data"]
    except (IndexError, KeyError, TypeError):
        return f"Error: Unexpected Reddit response structure for {url}"
    lines.append(f"r/{post.get('subreddit','')} | u/{post.get('author','')} | score: {post.get('score',0)}")
    lines.append(f"# {post.get('title','')}")
    if post.get("selftext"):
        lines.append(post["selftext"])
    lines.append("")

    # Append the comments (kind "t1") indented by depth and recurse into their replies; deleted
    # and empty bodies are skipped.
    def _comments(children: list, depth: int = 0) -> None:
        for child in children:
            if child.get("kind") != "t1":
                continue
            child_data = child["data"]
            body = child_data.get("body", "").strip()
            if body and body != "[deleted]":
                lines.append(
                    "  " * depth
                    + f"[{child_data.get('author','?')} | +{child_data.get('score',0)}] {body}"
                )
            replies = child_data.get("replies")
            if isinstance(replies, dict):
                _comments(replies["data"]["children"], depth + 1)

    # The second listing of the thread JSON holds the top-level comments.
    if len(data) > 1:
        _comments(data[1]["data"]["children"])

    return "\n".join(lines)[:15_000]


# Nav-cruft prefix pattern: lines before the real post body (www SPA header).
_NAV_END_RE = re.compile(
    r"(?:^|\n)(?:Go to \w|r/\w+)\n",
    re.MULTILINE,
)

# old.reddit "submitted … by author" line — the post title sits on the line right above it.
_OLD_SUBMITTED_RE = re.compile(r"^submitted .+? by ", re.MULTILINE)
# old.reddit footer start (site links + cookie banner) — everything from here is cruft.
_OLD_FOOTER_RE = re.compile(r"\nabout\nblog\n")


# Strip nav/footer cruft from thread inner_text. old.reddit markers first (title line found
# via the "submitted … by" line, footer via the about/blog link block), then the www SPA
# header pattern as fallback.
def _strip_reddit_nav(text: str) -> str:
    text = text.strip()
    foot = _OLD_FOOTER_RE.search(text)
    if foot:
        text = text[: foot.start()].rstrip()
    sub = _OLD_SUBMITTED_RE.search(text)
    if sub:
        # The title is the nearest non-empty line above "submitted … by" (blank line between).
        head_lines = text[: sub.start()].splitlines()
        title = next((ln.strip() for ln in reversed(head_lines) if ln.strip()), "")
        body = text[sub.start() :].strip()
        return f"{title}\n{body}" if title else body
    m = _NAV_END_RE.search(text)
    return text[m.start() :].strip() if m else text


# Fetching

# Fetch thread JSON via curl_cffi TLS impersonation (blocking). Return None when the response is
# not a listing array.
def _fetch_reddit_json_curl(json_url: str, thread_url: str, timeout: float) -> list[Any] | None:
    from curl_cffi import requests as _r

    resp = _r.get(
        json_url,
        impersonate="chrome124",
        timeout=max(5.0, timeout),
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": thread_url,
        },
    )
    resp.raise_for_status()
    data = resp.json()
    return data if isinstance(data, list) else None


# Fetch the rendered thread page via the warm cloakbrowser and format it as markdown.
# Two extraction paths on the SAME rendered page (no second fetch):
#   1. structural parse of result.html — post + nested comments with per-comment
#      score/depth/author/OP preserved (the high-value path, ported from webclaw);
#   2. inner_text + nav strip — the legacy fallback when the structure isn't there
#      (a layout change, a comment-permalink page the selectors miss, an error wall).
async def _fetch_reddit_browser_page(thread_url: str, timeout: float) -> str | None:
    from core.fetch.browser.client import browser_fetch

    # Render the page; a failed render ends this path.
    result = await browser_fetch(thread_url, nav_timeout=max(float(timeout), 30.0), wait_sec=5.0)
    if not result.ok:
        logger.debug(
            "reddit browser page fetch failed for %s: %s", thread_url, result.error or result.status
        )
        return None

    # Path 1: structural parse of the HTML, accepted only when it yields substantial content.
    if result.html:
        try:
            markdown = try_extract_markdown(result.html, thread_url)
        except Exception as exc:  # noqa: BLE001 — parse must never break the fetch
            logger.debug("reddit structural parse failed for %s: %s", thread_url, exc)
            markdown = None
        if markdown and len(markdown) >= 200:
            return markdown

    # Path 2: the visible text with navigation and footer cruft removed.
    text = _strip_reddit_nav(result.text or "")
    if len(text.strip()) < 200:
        logger.debug("reddit browser inner_text too short (%d chars) for %s", len(text), thread_url)
        return None
    return text


# Fetch a thread and return its markdown or an "Error: ..." string. Two methods only, ordered to
# look like a human before touching the API:
#   1. warm-browser page (old.reddit) — a normal user browsing the light host; works even
#      when the IP is rate-limited, and doesn't feed the antibot extra .json suspicion
#   2. curl_cffi .json               — the public API, structured fallback (post score +
#      nested comments); host is irrelevant — www and old .json behave identically, and
#      a rate-limited IP loses both, which is exactly why the page render goes first.
async def fetch_reddit_json(url: str, timeout: float = 15.0) -> str:
    # 1. Render the old.reddit thread page and strip its nav header.
    try:
        text = await _fetch_reddit_browser_page(reddit_thread_url(url, host="old.reddit.com"), timeout)
        if text:
            return text
    except Exception as exc:
        logger.info("reddit old.reddit page fetch failed for %s: %s — falling back to .json", url, exc)

    # 2. Fallback: the thread .json API.
    www_json = reddit_json_url(url)
    try:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(
            _io_pool,
            lambda: _fetch_reddit_json_curl(www_json, reddit_thread_url(url), timeout),
        )
        if data is not None:
            return reddit_data_to_markdown(data, url)
    except Exception as exc:
        logger.warning("reddit .json api failed for %s: %s", www_json, exc)

    return f"Error: Reddit fetch failed for {url}"


# Handler

from custom_domains.base import FetchContext, PageResult

# Reddit needs room for a warm-browser render of the thread page + the .json fallback.
_REDDIT_READ_TIMEOUT_SEC = 60.0


# Unified handler: fetch a Reddit thread (old.reddit page render → .json API).
class RedditHandler:
    name = "reddit"
    fallback_to_generic = False
    # The browser/JSON path is too slow for web_search inline parsing.
    scope = "read_page"

    # Return True for Reddit thread comment URLs.
    def matches(self, url: str) -> bool:
        return is_reddit(url)

    # Fetch and format the thread; Reddit gets its own generous timeout floor.
    async def read(self, url: str, ctx: FetchContext) -> PageResult:
        timeout = max(float(ctx.timeout), _REDDIT_READ_TIMEOUT_SEC)
        markdown = await fetch_reddit_json(url, timeout=timeout)
        ok = bool(markdown) and not markdown.startswith("Error:")
        return PageResult(
            markdown=markdown or f"Error: Reddit fetch failed for {url}",
            ok=ok,
            method="reddit",
            apply_budget=ok,
            error="" if ok else (markdown or "reddit fetch failed"),
        )


# Handler instance registered in custom_domains/__init__.py.
HANDLER = RedditHandler()
