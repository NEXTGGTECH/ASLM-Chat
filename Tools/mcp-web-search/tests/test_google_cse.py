# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import json

from core.engines.google_cse import (
    GoogleParser,
    _decode_jsonp,
    _extract_bootstrap_token,
    _token_cache,
)
from core.engines.models import ParseStatus
from core.fetch._base import TransportResponse


# Build a fake CSE bootstrap script: a JSONP-style callback wrapped in JavaScript that carries the
# token, library version and experiments.
def _bootstrap() -> str:
    payload = {
        "cse_token": "temporary-token",
        "cselibVersion": "abc123",
        "exp": ["feature-a", "feature-b"],
    }
    return "window.__gcse = window.__gcse || {}; callback(" + json.dumps(payload) + ");"


# Transport double that records every request and always answers with the bootstrap script and
# the configured status.
class _FakeTransport:
    # Store the status to answer with and start with no recorded requests.
    def __init__(self, *, status: int = 200) -> None:
        self.status = status
        self.requests = []

    # Record the request and answer with the bootstrap script.
    async def fetch(self, request):
        self.requests.append(request)
        return TransportResponse(
            status=self.status,
            body=_bootstrap().encode(),
            transport="fake",
        )

    # Do nothing, as the fake holds no connection.
    async def close(self) -> None:
        return None


# Verify that the token, library version, experiments and an expiry in the future are extracted
# from the bootstrap script wrapped in JavaScript.
def test_bootstrap_token_is_extracted_from_wrapped_javascript():
    token = _extract_bootstrap_token(_bootstrap(), now=100.0)

    assert token.value == "temporary-token"
    assert token.library_version == "abc123"
    assert token.experiments == "feature-a,feature-b"
    assert token.expires_at > 100.0


# Verify that the JSONP decoder strips the callback wrapper and returns the JSON payload.
def test_jsonp_decoder_ignores_callback_wrapper():
    payload = _decode_jsonp('__aslm_cse({"results":[{"x":1}]});')

    assert payload == {"results": [{"x": 1}]}


# Verify that the async request builder fetches the bootstrap token once and builds a CSE search
# request with the engine id, query, safe search, language, country, date sort and referer.
def test_async_builder_fetches_token_and_builds_search_request(monkeypatch):
    monkeypatch.setenv("GOOGLE_CSE_ID", "test-engine-id")
    _token_cache.clear()
    transport = _FakeTransport()
    parser = GoogleParser()

    # Act: build a request for a German, safe-search-off, last-week query.
    request = asyncio.run(
        parser.build_request_async(
            transport,
            "asyncio TaskGroup",
            region="de-de",
            safesearch="off",
            timelimit="w",
        )
    )

    # Assert: one bootstrap request was made and the built request carries the search parameters.
    assert len(transport.requests) == 1
    assert transport.requests[0].params["cx"] == "test-engine-id"
    assert request.url == "https://cse.google.com/cse/element/v1"
    assert request.params["cx"] == "test-engine-id"
    assert request.params["q"] == "asyncio TaskGroup"
    assert request.params["safe"] == "off"
    assert request.params["hl"] == "de"
    assert request.params["gl"] == "DE"
    assert request.params["sort"].startswith("date:r:")
    assert request.headers["Referer"] == "https://cse.google.com/"


# Verify that two requests built in a row share the cached token, so the bootstrap is fetched
# only once.
def test_token_is_reused_until_expiry(monkeypatch):
    monkeypatch.setenv("GOOGLE_CSE_ID", "cache-test-id")
    _token_cache.clear()
    transport = _FakeTransport()

    # Build two search requests with separate parsers.
    async def build_twice():
        await GoogleParser().build_request_async(transport, "first")
        await GoogleParser().build_request_async(transport, "second")

    asyncio.run(build_twice())

    assert len(transport.requests) == 1


# Verify that the parser decodes the JSONP results (unescaping HTML entities in titles), counts the
# card with an invalid URL as malformed and reports a partial parse.
def test_parser_returns_structured_results_and_skips_malformed_cards():
    document = "callback(" + json.dumps(
        {
            "results": [
                {
                    "unescapedUrl": "https://docs.python.org/3/library/asyncio-task.html",
                    "titleNoFormatting": "Coroutines &amp; Tasks",
                    "contentNoFormatting": "TaskGroup documentation.",
                },
                {"unescapedUrl": "not-a-url", "titleNoFormatting": "Broken"},
            ]
        }
    ) + ");"

    result = GoogleParser().parse(document)

    assert result.status == ParseStatus.PARTIAL
    assert result.parser_variant == "cse_jsonp"
    assert result.cards_seen == 2
    assert result.malformed_cards == 1
    assert result.results[0].title == "Coroutines & Tasks"


# Verify that a failed bootstrap (HTTP 503) makes the request builder fall back to the regular
# Google HTML search, whose enable-JavaScript page the parser reports as blocked.
def test_bootstrap_failure_uses_html_reserve(monkeypatch):
    monkeypatch.setenv("GOOGLE_CSE_ID", "failed-test-id")
    _token_cache.clear()
    transport = _FakeTransport(status=503)
    parser = GoogleParser()

    request = asyncio.run(parser.build_request_async(transport, "fallback query"))
    result = parser.parse(
        '<noscript><a href="/httpservice/retry/enablejs">enable JavaScript</a></noscript>'
    )

    assert request.url == "https://www.google.com/search"
    assert result.status == ParseStatus.BLOCKED
    assert any("HTML reserve" in item for item in result.diagnostics)
