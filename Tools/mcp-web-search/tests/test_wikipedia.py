# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import custom_domains
import custom_domains.wikipedia as wikipedia


# Verify that an article URL maps to the API endpoint of its language wiki, with the page name (without
# the fragment) and redirects enabled.
def test_wikipedia_article_urls_map_to_their_language_api():
    endpoint, params = wikipedia._api_request(
        "https://ru.wikipedia.org/wiki/Python_(язык_программирования)#История"
    )

    assert endpoint == "https://ru.wikipedia.org/w/api.php"
    assert params["page"] == "Python_(язык_программирования)"
    assert params["redirects"] == "1"


# Verify that a mobile host with a revision URL maps to the desktop API endpoint and passes the oldid
# instead of a page name.
def test_mobile_and_revision_urls_are_supported():
    endpoint, params = wikipedia._api_request(
        "https://en.m.wikipedia.org/w/index.php?title=Python&oldid=12345"
    )

    assert endpoint == "https://en.wikipedia.org/w/api.php"
    assert params["oldid"] == "12345"
    assert "page" not in params


# Verify that the portal root, a wiki main page without an article and a lookalike host have no API
# request.
def test_non_article_and_lookalike_urls_do_not_match():
    assert wikipedia._api_request("https://www.wikipedia.org/") is None
    assert wikipedia._api_request("https://en.wikipedia.org/") is None
    assert wikipedia._api_request("https://en.wikipedia.org.evil.test/wiki/Python") is None


# Verify that the API payload is converted to markdown with the title heading, the lead paragraph, the
# section heading and an absolute link.
def test_payload_normalization_keeps_lead_headings_and_links():
    payload = {
        "parse": {
            "title": "Example article",
            "text": (
                '<div class="mw-parser-output">'
                '<p>Lead paragraph with <a href="/wiki/Useful">a useful link</a>. '
                + "Substantial article prose remains in the API output. " * 8
                + "</p>"
                '<h2>History</h2><p>Historical details remain available.</p>'
                "</div>"
            ),
        }
    }

    markdown = wikipedia._payload_to_markdown(
        "https://en.wikipedia.org/wiki/Example_article", payload
    )

    assert markdown.startswith("# Example article")
    assert "Lead paragraph" in markdown
    assert "## History" in markdown
    assert "[a useful link](https://en.wikipedia.org/wiki/Useful)" in markdown


# Verify that the title from the API is used as the document title even when the HTML starts with a
# section heading.
def test_api_title_overrides_an_html_section_selected_as_document_title():
    payload = {
        "parse": {
            "title": "Canonical API title",
            "text": "<h2>Misleading section</h2><p>" + "Article body. " * 30 + "</p>",
        }
    }

    markdown = wikipedia._payload_to_markdown(
        "https://en.wikipedia.org/wiki/Canonical_API_title", payload
    )

    assert markdown.startswith("# Canonical API title\n")


# Verify that the handler reads an article through the API fetch with the request's timeout and returns
# an ok result with method "wikipedia_api" that asks for the shared budget.
def test_handler_uses_api_result_and_requests_budget(monkeypatch):
    # Check the timeout the handler passes and return canned markdown.
    async def fake_fetch(url: str, timeout: float = 20.0) -> str:
        assert timeout == 7.0
        return "# Article\n\nAPI content"

    monkeypatch.setattr(wikipedia, "fetch_wikipedia_page", fake_fetch)
    ctx = SimpleNamespace(timeout=7.0)
    result = asyncio.run(
        wikipedia.HANDLER.read("https://en.wikipedia.org/wiki/Article", ctx)
    )

    assert result.ok
    assert result.method == "wikipedia_api"
    assert result.apply_budget


# Verify that the Wikipedia handler is registered for article URLs of any language wiki.
def test_wikipedia_handler_is_registered():
    handler = custom_domains.match("https://de.wikipedia.org/wiki/Python")

    assert handler is wikipedia.HANDLER
