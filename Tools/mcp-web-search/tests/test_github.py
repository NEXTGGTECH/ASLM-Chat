# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from custom_domains.base import FetchContext, PageResult
from custom_domains import github


# Build a fetch context for the GitHub handler whose generic page reader is the given coroutine.
def _context(generic_read):
    return FetchContext(
        timeout=10,
        max_chars=10000,
        focus="",
        cfg=SimpleNamespace(),
        cache=None,
        generic_read=generic_read,
    )


# Verify that a rate-limit error from the GitHub API makes the handler read the page as HTML and
# report the github_html_fallback method.
def test_github_handler_falls_back_to_html_on_api_rate_limit(monkeypatch):
    # Fail like the GitHub API does when the rate limit is exceeded.
    async def api_failure(_url, timeout=20.0):
        return "Error: GitHub API repo fetch failed: 403 rate limit exceeded for api.github.com"

    # Return an HTML page read for the repository URL.
    async def html_success(request):
        assert request.url == "https://github.com/openai/codex"
        return PageResult(markdown="# openai/codex\nHTML content", ok=True, method="basic")

    monkeypatch.setattr(github, "fetch_github_page", api_failure)

    result = asyncio.run(github.HANDLER.read(
        "https://github.com/openai/codex",
        _context(html_success),
    ))

    assert result.ok is True
    assert result.method == "github_html_fallback"
    assert "HTML content" in result.markdown


# Verify that an API error other than a rate limit (404) is returned as a failed result and the
# HTML fallback is not used.
def test_github_handler_keeps_non_rate_limit_api_error(monkeypatch):
    fallback_called = False

    # Fail like the GitHub API does for a missing repository.
    async def api_failure(_url, timeout=20.0):
        return "Error: GitHub API repo fetch failed: 404 Not Found"

    # Record that the HTML fallback was used.
    async def html_fallback(_request):
        nonlocal fallback_called
        fallback_called = True
        return PageResult(markdown="unexpected", ok=True)

    monkeypatch.setattr(github, "fetch_github_page", api_failure)

    result = asyncio.run(github.HANDLER.read(
        "https://github.com/openai/missing",
        _context(html_fallback),
    ))

    assert result.ok is False
    assert fallback_called is False
