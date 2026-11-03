# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import custom_domains
import custom_domains.telegram as telegram


# Embed document of one public post: author, channel, text with emoji and links, and a date link.
_POST_HTML = """
<!doctype html>
<html><body>
  <div class="tgme_widget_message js-widget_message">
    <div class="tgme_widget_message_author">
      <a class="tgme_widget_message_author_name" href="https://t.me/alice"><span>Alice</span></a>
      <a class="tgme_widget_message_owner_name" href="https://t.me/example_channel"><span>Example Channel</span></a>
    </div>
    <div class="tgme_widget_message_text js-message_text" dir="auto">
      <tg-emoji><i class="emoji"><b>✅</b></i></tg-emoji>
      <b>Important update</b><br/><br/>
      The complete public post is available through the embed document.
      <a href="/example_channel/41">Related post</a>
    </div>
    <a class="tgme_widget_message_date" href="https://t.me/example_channel/42">
      <time datetime="2026-04-16T09:03:24+00:00">Apr 16 at 09:03</time>
    </a>
  </div>
</body></html>
"""

# Server-rendered channel feed with two posts, listed oldest first.
_FEED_HTML = """
<!doctype html>
<html><head><meta property="og:title" content="Example Channel"/></head><body>
  <div class="tgme_widget_message" data-post="example_channel/41">
    <div class="tgme_widget_message_text">Older post about radios.</div>
    <a class="tgme_widget_message_date"><time datetime="2026-04-15T09:00:00+00:00"></time></a>
  </div>
  <div class="tgme_widget_message" data-post="example_channel/42">
    <div class="tgme_widget_message_text"><b>Newest settings</b><br/>Use MEDIUM_FAST.</div>
    <a class="tgme_widget_message_date"><time datetime="2026-04-16T09:00:00+00:00"></time></a>
  </div>
</body></html>
"""


# Verify that public post URLs (plain, /s/ preview and forum topic) map to the embed document URL, and a
# forum topic maps to the message id.
def test_public_post_and_forum_topic_urls_map_to_embed_document():
    assert telegram.telegram_embed_url("https://t.me/example_channel/42") == (
        "https://t.me/example_channel/42?embed=1&mode=tme"
    )
    assert telegram.telegram_embed_url("https://t.me/s/example_channel/42?single=1") == (
        "https://t.me/example_channel/42?embed=1&mode=tme"
    )
    assert telegram.telegram_embed_url("https://t.me/example_channel/82432/169148") == (
        "https://t.me/example_channel/169148?embed=1&mode=tme"
    )


# Verify that private invites, /c/ links, bare channels, nested topic paths and lookalike hosts have no
# embed URL, and that is_telegram_url accepts the invite but rejects the lookalike host.
def test_private_invite_channel_only_and_lookalike_urls_do_not_match():
    assert telegram.telegram_embed_url("https://t.me/+private-invite") is None
    assert telegram.telegram_embed_url("https://t.me/c/123456/42") is None
    assert telegram.telegram_embed_url("https://t.me/example_channel") is None
    assert telegram.telegram_embed_url("https://t.me/example_channel/topic/42") is None
    assert telegram.telegram_embed_url("https://t.me.evil.test/example_channel/42") is None
    assert telegram.is_telegram_url("https://t.me/+private-invite")
    assert not telegram.is_telegram_url("https://t.me.evil.test/example_channel/42")


# Verify that a public channel root (plain or /s/) maps to the server-rendered feed URL, while a post
# URL or a private invite has no feed URL.
def test_public_channel_root_maps_to_server_rendered_feed():
    assert telegram.telegram_feed_url("https://t.me/example_channel") == (
        "https://t.me/s/example_channel"
    )
    assert telegram.telegram_feed_url("https://t.me/s/example_channel") == (
        "https://t.me/s/example_channel"
    )
    assert telegram.telegram_feed_url("https://t.me/example_channel/42") is None
    assert telegram.telegram_feed_url("https://t.me/+private-invite") is None


# Verify that the embed document is converted to markdown with a title from author and channel, the
# date, the author, the emoji, the body text and absolute links, without the human-readable date text.
def test_embed_html_normalizes_body_author_date_emoji_and_links():
    markdown = telegram._embed_to_markdown(
        "https://t.me/example_channel/82432/42",
        _POST_HTML,
        max_chars=4_000,
    )

    assert markdown.startswith("# Alice in Example Channel")
    assert "**Date:** 2026-04-16" in markdown
    assert "**Author:** Alice" in markdown
    assert "✅" in markdown
    assert "Important update" in markdown
    assert "[Related post](https://t.me/example_channel/41)" in markdown
    assert "Apr 16 at 09:03" not in markdown


# Verify that the pre-budget guard bounds a very long markdown to the minimum pre-budget size and keeps
# its heading, so the BM25 input stays small.
def test_pre_budget_guard_keeps_bm25_input_bounded():
    markdown = "# Telegram post\n\n" + ("Useful sentence for ranking. " * 2_000)
    bounded = telegram._pre_budget_limit(markdown, max_chars=2_000)

    assert len(bounded) <= telegram._MIN_PRE_BUDGET_CHARS
    assert bounded.startswith("# Telegram post")


# Verify that the channel feed markdown lists the newest post first, has the channel title and date, and
# does not contain the "view and join" card text.
def test_channel_feed_is_newest_first_and_excludes_join_card_chrome():
    markdown = telegram._feed_to_markdown(
        "https://t.me/example_channel",
        _FEED_HTML,
        max_chars=4_000,
    )

    assert markdown.startswith("# Example Channel")
    assert "**Date:** 2026-04-16" in markdown
    assert markdown.index("example_channel/42") < markdown.index("example_channel/41")
    assert "Use MEDIUM_FAST" in markdown
    assert "view and join" not in markdown.lower()


# Verify that the handler reads a post through the light HTTP fetch with the request's timeout and
# character limit, and returns an ok result with method "telegram_embed" that asks for the shared budget.
def test_handler_uses_light_http_result_and_requests_shared_budget(monkeypatch):
    # Check the arguments the handler passes and return a canned markdown post.
    async def fake_fetch(url: str, *, timeout: float, max_chars: int) -> str:
        assert url == "https://t.me/example_channel/42"
        assert timeout == 6.0
        assert max_chars == 4_000
        return "# Telegram post\n\nFull public content"

    monkeypatch.setattr(telegram, "fetch_telegram_post", fake_fetch)
    result = asyncio.run(
        telegram.HANDLER.read(
            "https://t.me/example_channel/42",
            SimpleNamespace(
                timeout=6.0,
                max_chars=4_000,
            ),
        )
    )

    assert result.ok
    assert result.method == "telegram_embed"
    assert result.apply_budget


# Verify that a group without a public HTTP feed gives a failed result (ok is false) that carries the
# fetch error message.
def test_group_without_public_http_feed_is_parsed_false(monkeypatch):
    # Return the error text for a group that has no public message feed.
    async def no_public_feed(*args, **kwargs):
        return "Error: Telegram does not expose a public message feed for: https://t.me/group"

    monkeypatch.setattr(telegram, "fetch_telegram_post", no_public_feed)
    result = asyncio.run(
        telegram.HANDLER.read(
            "https://t.me/example_group",
            SimpleNamespace(
                timeout=6.0,
                max_chars=4_000,
            ),
        )
    )

    assert not result.ok
    assert result.method == "telegram_embed"
    assert "does not expose a public message feed" in result.error


# Verify that the Telegram handler is registered for post, channel and private invite URLs, and that
# it is not read_page-only (it can be used for inline prefetch).
def test_telegram_handler_is_registered_for_inline_prefetch():
    handler = custom_domains.match("https://t.me/example_channel/42")

    assert handler is telegram.HANDLER
    assert custom_domains.is_read_page_only("https://t.me/example_channel/42") is False
    assert custom_domains.match("https://t.me/example_channel") is telegram.HANDLER
    assert custom_domains.match("https://t.me/+private-invite") is telegram.HANDLER
