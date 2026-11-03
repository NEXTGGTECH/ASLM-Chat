# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
from urllib.parse import unquote, urljoin, urlparse

from core.extract.content_processor import _truncate_markdown_to_budget

from custom_domains.base import FetchContext, PageResult

# Download size cap, and the ceiling and floor of the markdown kept before the shared read_page
# budget pass (the floor applies below four times the requested size).
_MAX_RESPONSE_BYTES = 1 * 1024 * 1024
_MAX_PRE_BUDGET_CHARS = 32_000
_MIN_PRE_BUDGET_CHARS = 8_000

# User-Agent sent to Telegram.
_USER_AGENT = "ASLM-Chat/1.0 (public Telegram post reader)"

# A valid public username (letter first, up to 64 characters) and the first path segments that are
# Telegram features rather than usernames.
_USERNAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_RESERVED_PATHS = {
    "addlist",
    "addstickers",
    "c",
    "confirmphone",
    "contact",
    "invoice",
    "joinchat",
    "login",
    "proxy",
    "setlanguage",
    "share",
    "socks",
}


# URL parsing

# Return True for http(s) URLs on t.me / telegram.me (with or without www.).
def is_telegram_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    return (
        parsed.scheme.lower() in {"http", "https"}
        and host in {"t.me", "telegram.me", "www.t.me", "www.telegram.me"}
    )


# Convert a public post or forum-topic link to Telegram's server-rendered widget URL.
# Forum links have /username/topic_id/message_id; the widget addresses the same post
# by /username/message_id, so the final numeric path component is authoritative.
def telegram_embed_url(url: str) -> str | None:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if host not in {"t.me", "telegram.me", "www.t.me", "www.telegram.me"}:
        return None
    if parsed.scheme.lower() not in {"http", "https"}:
        return None

    # Path segments: [s/]username[/topic_id]/message_id.
    parts = [unquote(part).strip() for part in parsed.path.split("/") if part.strip()]
    if parts and parts[0].lower() == "s":
        parts = parts[1:]
    if len(parts) not in {2, 3}:
        return None

    # Accept only a real username, a numeric topic (if any) and a positive numeric message id.
    username = parts[0]
    message_id = parts[-1]
    if (
        username.lower() in _RESERVED_PATHS
        or not _USERNAME_RE.fullmatch(username)
        or (len(parts) == 3 and not parts[1].isdigit())
        or not message_id.isdigit()
        or int(message_id) <= 0
    ):
        return None
    return f"https://t.me/{username}/{message_id}?embed=1&mode=tme"


# Public broadcast channels expose a lightweight, server-rendered history at /s/name.
# Public groups do not: Telegram redirects /s/name back to their join card.  We still
# return the candidate feed URL here; the parser detects the redirect/card by the
# absence of message nodes and fails cleanly instead of ingesting Telegram's CTA.
def telegram_feed_url(url: str) -> str | None:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if host not in {"t.me", "telegram.me", "www.t.me", "www.telegram.me"}:
        return None
    if parsed.scheme.lower() not in {"http", "https"}:
        return None

    # Path segments: [s/]username, nothing more.
    parts = [unquote(part).strip() for part in parsed.path.split("/") if part.strip()]
    if len(parts) == 2 and parts[0].lower() == "s":
        parts = parts[1:]
    if len(parts) != 1:
        return None

    username = parts[0]
    if username.lower() in _RESERVED_PATHS or not _USERNAME_RE.fullmatch(username):
        return None
    return f"https://t.me/s/{username}"


# Fetching

# Fetch a bounded widget document. Telegram posts are small; the 1 MiB ceiling leaves
# room for markup and reactions while preventing a malformed response from consuming
# an eager web_search parse slot.
async def _fetch_embed_html(embed_url: str, timeout: float) -> str:
    import httpx

    chunks: list[bytes] = []
    total = 0
    headers = {
        "Accept": "text/html,application/xhtml+xml",
        "User-Agent": _USER_AGENT,
    }
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
        async with client.stream("GET", embed_url, headers=headers) as response:
            response.raise_for_status()
            declared = int(response.headers.get("content-length") or 0)
            if declared > _MAX_RESPONSE_BYTES:
                raise ValueError(f"Telegram embed response exceeds {_MAX_RESPONSE_BYTES} bytes")
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > _MAX_RESPONSE_BYTES:
                    raise ValueError(f"Telegram embed response exceeds {_MAX_RESPONSE_BYTES} bytes")
                chunks.append(chunk)
    return b"".join(chunks).decode("utf-8", errors="replace")


# Markdown conversion

# Keep enough source text for query-aware BM25 selection, but cap the intermediate
# document before it reaches the shared read_page budget pass.
def _pre_budget_limit(markdown: str, max_chars: int) -> str:
    requested = max(0, int(max_chars or 0))
    limit = min(
        _MAX_PRE_BUDGET_CHARS,
        max(_MIN_PRE_BUDGET_CHARS, requested * 4),
    )
    if len(markdown) <= limit:
        return markdown
    return _truncate_markdown_to_budget(markdown, limit, preserve_fenced_code=True)


# Convert Telegram's message body element (BeautifulSoup node, modified in place) to markdown text.
def _body_to_markdown(body, source_url: str) -> str:
    from bs4 import NavigableString

    # Replace the formatting elements with their markdown equivalents: emoji by their text, code
    # blocks and inline code, bold, italics, links (made absolute), quotes and line breaks.
    for emoji in list(body.select("tg-emoji")):
        emoji.replace_with(NavigableString(emoji.get_text("", strip=True)))
    for pre in list(body.select("pre")):
        value = pre.get_text("\n", strip=False).strip("\n")
        pre.replace_with(NavigableString(f"\n```\n{value}\n```\n"))
    for code in list(body.select("code")):
        value = code.get_text("", strip=False).replace("`", "\\`")
        code.replace_with(NavigableString(f"`{value}`"))
    for strong in list(body.select("b, strong")):
        value = strong.get_text("", strip=False).strip()
        strong.replace_with(NavigableString(f"**{value}**" if value else ""))
    for emphasis in list(body.select("i:not(.emoji), em")):
        value = emphasis.get_text("", strip=False).strip()
        emphasis.replace_with(NavigableString(f"*{value}*" if value else ""))
    for link in list(body.select("a[href]")):
        label = link.get_text(" ", strip=True).replace("]", "\\]")
        href = urljoin(source_url, str(link.get("href") or "").strip())
        link.replace_with(NavigableString(f"[{label}]({href})" if label and href else label))
    for quote in list(body.select("blockquote")):
        lines = quote.get_text("\n", strip=True).splitlines()
        quote.replace_with(NavigableString("\n" + "\n".join(f"> {line}" for line in lines) + "\n"))
    for line_break in list(body.select("br")):
        line_break.replace_with(NavigableString("\n"))

    # Flatten to text and tidy the whitespace: trailing and leading blanks per line, runs of blank lines.
    text = body.get_text("", strip=False).replace("\xa0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# Isolate Telegram's message body and retain its small, known formatting vocabulary. Return an empty
# string when the HTML parser is unavailable or the page has no message text.
def _embed_to_markdown(source_url: str, raw_html: str, *, max_chars: int = 20_000) -> str:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        # The HTML parser is optional; without it nothing can be extracted.
        return ""

    # Locate the message text and drop embedded scripts and styles.
    soup = BeautifulSoup(raw_html or "", "lxml")
    message = soup.select_one(".tgme_widget_message")
    body = message.select_one(".tgme_widget_message_text") if message else None
    if message is None or body is None or not body.get_text(" ", strip=True):
        return ""

    for unwanted in body.select("script, style"):
        unwanted.decompose()

    # Title ("author in channel"), publication time and the converted body.
    author_node = message.select_one(".tgme_widget_message_author_name")
    owner_node = message.select_one(".tgme_widget_message_owner_name")
    time_node = message.select_one(".tgme_widget_message_date time[datetime]")
    author = author_node.get_text(" ", strip=True) if author_node else ""
    owner = owner_node.get_text(" ", strip=True) if owner_node else ""
    published = str(time_node.get("datetime") or "").strip() if time_node else ""
    title = " in ".join(part for part in (author, owner) if part) or "Telegram post"
    content = _body_to_markdown(body, source_url)
    if not content:
        return ""

    # Header lines followed by the content, capped before the shared budget pass.
    parts = [f"# {title}", "", "**Site:** t.me", f"**URL:** {source_url}"]
    if published:
        parts.append(f"**Date:** {published[:10]}")
    if author:
        parts.append(f"**Author:** {author}")
    parts.extend(("", "---", "", content))
    markdown = "\n".join(parts)
    return _pre_budget_limit(markdown, max_chars) if markdown else ""


# Convert the recent messages exposed by a public broadcast channel's /s/ page into
# one bounded document.  Newest-first ordering makes the pre-budget cap useful even
# before the shared query-aware BM25 pass selects the most relevant chunks.
def _feed_to_markdown(source_url: str, raw_html: str, *, max_chars: int = 20_000) -> str:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        # The HTML parser is optional; without it nothing can be extracted.
        return ""

    # No message nodes means the redirect/join card was served instead of a feed.
    soup = BeautifulSoup(raw_html or "", "lxml")
    messages = soup.select(".tgme_widget_message[data-post]")
    if not messages:
        return ""

    # Channel title from the header, falling back to og:title.
    title_node = soup.select_one(".tgme_channel_info_header_title")
    title = title_node.get_text(" ", strip=True) if title_node else ""
    if not title:
        meta_title = soup.select_one('meta[property="og:title"]')
        title = str(meta_title.get("content") or "").strip() if meta_title else ""
    title = title or "Telegram channel"

    # Convert every message that has text (the page lists them oldest first, so iterate reversed to
    # get newest first); each becomes a heading linking to the post plus its body.
    posts: list[tuple[str, str]] = []
    latest_date = ""
    for message in reversed(messages):
        body = message.select_one(".tgme_widget_message_text")
        if body is None or not body.get_text(" ", strip=True):
            continue
        for unwanted in body.select("script, style"):
            unwanted.decompose()
        content = _body_to_markdown(body, source_url)
        if not content:
            continue

        post_ref = str(message.get("data-post") or "").strip().strip("/")
        post_url = f"https://t.me/{post_ref}" if post_ref else source_url
        time_node = message.select_one(".tgme_widget_message_date time[datetime]")
        published = str(time_node.get("datetime") or "").strip() if time_node else ""
        if published and not latest_date:
            latest_date = published[:10]
        heading = f"## [{post_ref or 'Post'}]({post_url})"
        if published:
            heading += f" — {published[:10]}"
        posts.append((heading, content))

    # Assemble the document: channel header, then the posts.
    if not posts:
        return ""

    parts = [f"# {title}", "", "**Site:** t.me", f"**URL:** {source_url}"]
    if latest_date:
        parts.append(f"**Date:** {latest_date}")
    parts.extend(("", "---", ""))
    for heading, content in posts:
        parts.extend((heading, "", content, ""))
    return _pre_budget_limit("\n".join(parts).strip(), max_chars)


# Page reading and handler

# Read a public Telegram post (or a public channel's recent messages) through Telegram's lightweight
# embed HTML and return markdown, or an "Error: ..." string on failure.
async def fetch_telegram_post(
    url: str,
    *,
    timeout: float = 10.0,
    max_chars: int = 20_000,
) -> str:
    # Pick the embed widget for a post link, else the channel feed.
    embed_url = telegram_embed_url(url)
    feed_url = telegram_feed_url(url) if embed_url is None else None
    fetch_url = embed_url or feed_url
    if fetch_url is None:
        return f"Error: Unsupported public Telegram URL: {url}"
    try:
        raw_html = await _fetch_embed_html(fetch_url, timeout)
    except Exception as exc:
        return f"Error: Telegram fetch failed for {url}: {exc}"

    # Convert the document; an empty result is reported with an error that fits the kind of URL.
    if embed_url is not None:
        markdown = _embed_to_markdown(url, raw_html, max_chars=max_chars)
    else:
        markdown = _feed_to_markdown(url, raw_html, max_chars=max_chars)
    if markdown:
        return markdown
    if feed_url is not None:
        return f"Error: Telegram does not expose a public message feed for: {url}"
    return f"Error: Telegram post has no public extractable text: {url}"


# Handler for public Telegram posts and channel feeds (no generic fallback).
class TelegramHandler:
    name = "telegram"
    fallback_to_generic = False

    # Claim the whole host so unsupported/root/invite links cannot fall through to the generic
    # parser and ingest Telegram's join CTA as useful page content.
    def matches(self, url: str) -> bool:
        return is_telegram_url(url)

    # Fetch the post or feed as markdown; an "Error:" result is reported as a failed PageResult.
    async def read(self, url: str, ctx: FetchContext) -> PageResult:
        markdown = await fetch_telegram_post(
            url,
            timeout=min(float(ctx.timeout), 10.0),
            max_chars=ctx.max_chars,
        )

        ok = bool(markdown) and not markdown.lstrip().lower().startswith("error:")
        return PageResult(
            markdown=markdown,
            ok=ok,
            method="telegram_embed",
            apply_budget=ok,
            error="" if ok else markdown,
        )


# Handler instance registered in custom_domains/__init__.py.
HANDLER = TelegramHandler()

__all__ = [
    "HANDLER",
    "TelegramHandler",
    "fetch_telegram_post",
    "is_telegram_url",
    "telegram_embed_url",
    "telegram_feed_url",
]
