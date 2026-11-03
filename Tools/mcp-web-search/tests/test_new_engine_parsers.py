# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import json

from core.engines import QwantParser, StartpageParser, YandexParser
from core.engines.models import ParseStatus


# Qwant

# Build a Qwant API response with one ads block and one web block holding a valid and a malformed card.
def _qwant_success() -> str:
    # The parser must skip the ads block; the second web card has no title and an invalid URL.
    return json.dumps(
        {
            "status": "success",
            "data": {
                "result": {
                    "items": {
                        "mainline": [
                            {
                                "type": "ads",
                                "items": [{"title": "Ad", "url": "https://ad.example/x"}],
                            },
                            {
                                "type": "web",
                                "items": [
                                    {
                                        "title": "Real Python",
                                        "url": "https://realpython.com/a",
                                        "desc": "asyncio walkthrough",
                                    },
                                    {"title": "", "url": "not-a-url", "desc": "broken"},
                                ],
                            },
                        ]
                    }
                }
            },
        }
    )


# Verify that the Qwant parser keeps only the valid web card, skips the ads block, counts the malformed
# card and reports a partial parse.
def test_qwant_parses_web_block_and_skips_ads_and_malformed():
    result = QwantParser().parse(_qwant_success())

    # One malformed card is present, hence PARTIAL rather than SUCCESS.
    assert result.status == ParseStatus.PARTIAL
    assert [r.url for r in result.results] == ["https://realpython.com/a"]
    assert result.malformed_cards == 1


# Verify that a Qwant error response with error code 24 (rate limit) is reported as blocked.
def test_qwant_rate_limit_is_blocked():
    body = json.dumps({"status": "error", "data": {"error_code": 24}})
    assert QwantParser().parse(body).status == ParseStatus.BLOCKED


# Verify that a DataDome captcha interstitial reply is reported as blocked.
def test_qwant_datadome_interstitial_is_blocked():
    body = json.dumps({"url": "https://geo.captcha-delivery.com/interstitial/?cid=x"})
    assert QwantParser().parse(body).status == ParseStatus.BLOCKED


# Verify that a non-JSON body is reported as changed (the API shape drifted).
def test_qwant_invalid_json_is_changed():
    assert QwantParser().parse("<html>nope</html>").status == ParseStatus.CHANGED


# Yandex

# Verify that the Yandex parser extracts url and snippet from a serp-item result and reports success.
def test_yandex_parses_serp_items():
    html = (
        "<html><body>"
        '<li class="serp-item">'
        '<h3 class="b-serp-item__title">'
        '<a class="b-serp-item__title-link" href="https://realpython.com/async-io-python/">'
        "Hands-On asyncio</a></h3>"
        '<div class="b-serp-item__content"><div class="b-serp-item__text">A walkthrough.</div></div>'
        "</li></body></html>"
    )
    result = YandexParser().parse(html)
    assert result.status == ParseStatus.SUCCESS
    assert result.results[0].url == "https://realpython.com/async-io-python/"
    assert result.results[0].snippet == "A walkthrough."


# Verify that a page mentioning the Yandex captcha is reported as blocked.
def test_yandex_captcha_is_blocked():
    assert YandexParser().parse("<html>showcaptcha please</html>").status == ParseStatus.BLOCKED


# Startpage

# Verify that the Startpage parser reads the web-google block embedded in the React props script and
# strips HTML from the description.
def test_startpage_parses_embedded_web_google_block():
    # Arrange: React props with one web-google result, wrapped in the script call the parser looks for.
    props = {
        "render": {
            "presenter": {
                "regions": {
                    "mainline": [
                        {
                            "display_type": "web-google",
                            "results": [
                                {
                                    "clickUrl": "https://realpython.com/async-io-python/",
                                    "title": "Python's asyncio",
                                    "description": "<b>asyncio</b> walkthrough",
                                }
                            ],
                        }
                    ]
                }
            }
        }
    }
    document = f"<html><script>React.createElement(UIStartpage.AppSerpWeb, {json.dumps(props)})</script></html>"

    # Act and assert: the URL is taken from clickUrl and the snippet has its markup removed.
    result = StartpageParser().parse(document)
    assert result.status == ParseStatus.SUCCESS
    assert result.results[0].url == "https://realpython.com/async-io-python/"
    assert result.results[0].snippet == "asyncio walkthrough"


# Verify that a page with the Startpage captcha redirect is reported as blocked.
def test_startpage_captcha_is_blocked():
    assert StartpageParser().parse("<html>/sp/captcha redirect</html>").status == ParseStatus.BLOCKED


# Verify that a page without the embedded React props blob is reported as changed.
def test_startpage_missing_blob_is_changed():
    assert StartpageParser().parse("<html>no react here</html>").status == ParseStatus.CHANGED


# Verify that after a failed sc-token homepage scrape, the next lookup within the cooldown does not
# scrape again. A failing scrape must not be retried on every search, otherwise one blocked homepage
# serializes a fresh fetch behind the global lock for all callers.
def test_startpage_sc_token_backs_off_after_failure(monkeypatch):
    from core.engines import startpage as sp

    # Reset the module-level token cache and failure timestamps.
    monkeypatch.setattr(sp, "_sc_code", "", raising=False)
    monkeypatch.setattr(sp, "_sc_fetched_at", 0.0, raising=False)
    monkeypatch.setattr(sp, "_sc_failed_at", 0.0, raising=False)

    calls = 0

    # Count the scrape attempts and fail with an empty token.
    async def _fail(_transport) -> str:
        nonlocal calls
        calls += 1
        return ""

    monkeypatch.setattr(sp, "_fetch_sc_code", _fail)

    # Request the token twice: the first call scrapes, the second one falls inside the cooldown.
    async def _run() -> None:
        assert await sp._get_sc_code(object()) == ""
        assert await sp._get_sc_code(object()) == ""

    asyncio.run(_run())
    assert calls == 1
