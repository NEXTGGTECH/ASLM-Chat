# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio

import custom_domains.reddit as reddit

# Reddit thread URL used by all tests.
_THREAD = "https://www.reddit.com/r/codex/comments/1uaqzvk/again_we_are_here/"

# Minimal but realistic listing payload: one post + one comment with a nested reply.
_LISTING = [
    {"data": {"children": [{"data": {
        "subreddit": "codex", "author": "op", "score": 251,
        "title": "Again! We are here", "selftext": "body text",
    }}]}},
    {"data": {"children": [{"kind": "t1", "data": {
        "author": "alice", "score": 30, "body": "top comment",
        "replies": {"data": {"children": [{"kind": "t1", "data": {
            "author": "bob", "score": 3, "body": "nested reply", "replies": "",
        }}]}},
    }}]}},
]


# Verify that the .json and thread page URLs honor a host override, so the old.reddit page path hits the
# right host.
def test_url_host_override():
    assert reddit.reddit_json_url(_THREAD).startswith("https://www.reddit.com/")
    assert reddit.reddit_json_url(_THREAD).endswith(".json?limit=50&depth=3")
    assert reddit.reddit_json_url(_THREAD, host="old.reddit.com").startswith("https://old.reddit.com/")
    assert reddit.reddit_thread_url(_THREAD, host="old.reddit.com") == (
        "https://old.reddit.com/r/codex/comments/1uaqzvk/again_we_are_here"
    )


# Verify that the markdown carries the post header plus the comments with author and score, and that the
# nested reply is indented one level deeper.
def test_markdown_shape():
    md = reddit.reddit_data_to_markdown(_LISTING, _THREAD)
    assert "u/op" in md and "score: 251" in md
    assert "# Again! We are here" in md
    assert "[alice | +30] top comment" in md
    assert "  [bob | +3] nested reply" in md


# Verify that for the old.reddit inner text the navigation head is cut at the post title and the footer
# at the about/blog block.
def test_strip_old_reddit_nav_and_footer():
    raw = (
        "jump to content\nMY SUBREDDITS\nPOPULAR-ALL-USERS\nMORE »\n"
        "this post was submitted on 09 Apr 2026\n18 points (100% upvoted)\n"
        "Welcome to Reddit.\n\n18\n\n"
        "Anyone here tried Hermes Agent?Discussion (self.Rag)\n\n"
        "submitted 2 months ago by marwan_rashad5\n"
        "post body\n\n[–]alice 1 point 2 months ago\ntop comment\n"
        "permalinkembedsavereportreply\n"
        "about\nblog\nabout\nadvertising\ncareers\n"
        "Use of this site constitutes acceptance of our User Agreement.\n"
        "Reddit uses cookies and similar technologies to:\nCONTINUE"
    )
    text = reddit._strip_reddit_nav(raw)
    assert text.startswith("Anyone here tried Hermes Agent?")
    assert "top comment" in text
    assert "jump to content" not in text and "MY SUBREDDITS" not in text
    assert "advertising" not in text and "CONTINUE" not in text


# Verify that the old.reddit page render goes first and, when it succeeds, the .json API is never
# touched.
def test_page_first_json_untouched(monkeypatch):
    page_urls: list[str] = []

    # Record the requested page URL and return a rendered page.
    async def _page_ok(thread_url, _timeout):
        page_urls.append(thread_url)
        return "r/codex\npost body rendered from old.reddit"

    # Fail the test if the .json API is called.
    def _curl_must_not_run(*_a, **_k):
        raise AssertionError(".json API must not be called when the page render succeeds")

    monkeypatch.setattr(reddit, "_fetch_reddit_browser_page", _page_ok)
    monkeypatch.setattr(reddit, "_fetch_reddit_json_curl", _curl_must_not_run)

    md = asyncio.run(reddit.fetch_reddit_json(_THREAD, timeout=5.0))
    assert "post body rendered from old.reddit" in md
    assert page_urls and "old.reddit.com" in page_urls[0]


# Verify that when the page render fails (browser down or content too short) the .json API rung wins
# and its listing is turned into markdown.
def test_fallback_json_api(monkeypatch):
    # Return no page, as if the browser were down.
    async def _page_dead(*_a, **_k):
        return None

    json_urls: list[str] = []

    # Record the requested .json URL and return the canned listing.
    def _curl_ok(json_url, _thread_url, _timeout):
        json_urls.append(json_url)
        return _LISTING

    monkeypatch.setattr(reddit, "_fetch_reddit_browser_page", _page_dead)
    monkeypatch.setattr(reddit, "_fetch_reddit_json_curl", _curl_ok)

    md = asyncio.run(reddit.fetch_reddit_json(_THREAD, timeout=5.0))
    assert "# Again! We are here" in md
    assert json_urls and json_urls[0].endswith(".json?limit=50&depth=3")


# Verify that when both rungs fail the result is an explicit "Error:" marker, on which the handler sets
# ok=False.
def test_all_rungs_blocked(monkeypatch):
    # Fail as if the browser daemon were unavailable.
    async def _page_dead(*_a, **_k):
        raise RuntimeError("browser daemon unavailable")

    # Fail as if Reddit rejected the request with HTTP 429.
    def _curl_blocked(*_a, **_k):
        raise RuntimeError("HTTP Error 429")

    monkeypatch.setattr(reddit, "_fetch_reddit_browser_page", _page_dead)
    monkeypatch.setattr(reddit, "_fetch_reddit_json_curl", _curl_blocked)

    md = asyncio.run(reddit.fetch_reddit_json(_THREAD, timeout=5.0))
    assert md.startswith("Error:")
