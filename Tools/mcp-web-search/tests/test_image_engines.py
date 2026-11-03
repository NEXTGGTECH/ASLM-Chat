# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json

import core.engines.images as images
from core.config.settings import ImageEnginesSection
from core.engines.images import BingImagesParser, DuckDuckGoImagesParser
from core.engines.models import ParseStatus


# Build a DuckDuckGo image search request for "red panda" with a pre-stored vqd token, and clear
# the vqd cache afterwards so other tests are not affected.
def _ddg_request():
    images._store_vqd("red panda", images.for_engine("duckduckgo_images").user_agent, "4-123")
    try:
        return DuckDuckGoImagesParser.build_request("red panda", region="us-en")
    finally:
        images._VQD_CACHE.clear()


# Verify that the DuckDuckGo image request is an XHR (cors mode) without the navigation-only
# headers. The anti-bot behavior checked in this file was live-verified on 2026-10-28.
def test_ddg_xhr_request_has_no_navigation_only_headers() -> None:
    # DDG answers i.js with a 202 JSON challenge whenever Sec-Fetch-User is sent.
    headers = _ddg_request().headers
    for name in ("Sec-Fetch-User", "Upgrade-Insecure-Requests", "Cache-Control"):
        assert name not in headers
    assert headers["Sec-Fetch-Mode"] == "cors"


# Verify that a DuckDuckGo JSON challenge response is reported as blocked, not as an empty result.
def test_ddg_json_challenge_is_blocked_not_empty() -> None:
    document = json.dumps({"challenge": {"cc": "sre", "e": 0, "gk": "abc"}})
    result = DuckDuckGoImagesParser().parse(document)
    assert result.status == ParseStatus.BLOCKED
    assert result.results == []


# Verify that a DuckDuckGo JSON response with one image result parses successfully.
def test_ddg_results_parse() -> None:
    document = json.dumps({"results": [{
        "title": "Red panda",
        "url": "https://example.com/panda",
        "image": "https://example.com/panda.jpg",
        "thumbnail": "https://example.com/panda_t.jpg",
        "width": 800,
        "height": 600,
    }]})
    result = DuckDuckGoImagesParser().parse(document)
    assert result.status == ParseStatus.SUCCESS
    assert len(result.results) == 1


# Verify that the empty Bing page shell (only deferred scripts) is reported as blocked.
def test_bing_empty_shell_is_soft_block() -> None:
    shell = '<script nonce="x">_w.rms.js({});</script><div id="aRmsDefer"></div>'
    result = BingImagesParser().parse(shell)
    assert result.status == ParseStatus.BLOCKED


# Verify that a large Bing page without any image cards is reported as changed markup.
def test_bing_large_page_without_cards_is_changed_markup() -> None:
    page = "<html><body>" + ("<div>layout</div>" * 2000) + "</body></html>"
    result = BingImagesParser().parse(page)
    assert result.status == ParseStatus.CHANGED


# Verify that a Bing image card with HTML-escaped metadata parses successfully and yields the
# full-size image URL.
def test_bing_metadata_cards_parse() -> None:
    metadata = json.dumps({
        "t": "Red panda",
        "purl": "https://example.com/panda",
        "murl": "https://example.com/panda.jpg",
        "turl": "https://tse.example.com/th.jpg",
    }).replace('"', "&quot;")
    result = BingImagesParser().parse(f'<a class="iusc" m="{metadata}"></a>')
    assert result.status == ParseStatus.SUCCESS
    assert result.results[0].image_url == "https://example.com/panda.jpg"


# Verify that by default Bing, Brave and DuckDuckGo are enabled for images and Openverse is not.
def test_default_image_engines() -> None:
    engines = ImageEnginesSection().as_map()
    assert engines == {"bing": True, "brave": True, "duckduckgo": True, "openverse": False}
