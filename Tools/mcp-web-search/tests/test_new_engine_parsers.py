# Copyright NEXTGGTECH. Elastic License 2.0.

"""Offline parser coverage for the Qwant, Yandex and Startpage engines.

These exercise parse() against synthetic fixtures only (no network), guarding
against silent regressions when a SERP's markup or API shape drifts.
"""

from __future__ import annotations

import asyncio
import json

from core.engines import QwantParser, StartpageParser, YandexParser
from core.engines.models import ParseStatus


# --- Qwant -----------------------------------------------------------------

def _qwant_success() -> str:
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


def test_qwant_parses_web_block_and_skips_ads_and_malformed():
    result = QwantParser().parse(_qwant_success())
    assert result.status == ParseStatus.PARTIAL  # one malformed card present
    assert [r.url for r in result.results] == ["https://realpython.com/a"]
    assert result.malformed_cards == 1


def test_qwant_rate_limit_is_blocked():
    body = json.dumps({"status": "error", "data": {"error_code": 24}})
    assert QwantParser().parse(body).status == ParseStatus.BLOCKED


def test_qwant_datadome_interstitial_is_blocked():
    body = json.dumps({"url": "https://geo.captcha-delivery.com/interstitial/?cid=x"})
    assert QwantParser().parse(body).status == ParseStatus.BLOCKED


def test_qwant_invalid_json_is_changed():
    assert QwantParser().parse("<html>nope</html>").status == ParseStatus.CHANGED


# --- Yandex ----------------------------------------------------------------

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


def test_yandex_captcha_is_blocked():
    assert YandexParser().parse("<html>showcaptcha please</html>").status == ParseStatus.BLOCKED


# --- Startpage -------------------------------------------------------------

def test_startpage_parses_embedded_web_google_block():
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
    result = StartpageParser().parse(document)
    assert result.status == ParseStatus.SUCCESS
    assert result.results[0].url == "https://realpython.com/async-io-python/"
    assert result.results[0].snippet == "asyncio walkthrough"


def test_startpage_captcha_is_blocked():
    assert StartpageParser().parse("<html>/sp/captcha redirect</html>").status == ParseStatus.BLOCKED


def test_startpage_missing_blob_is_changed():
    assert StartpageParser().parse("<html>no react here</html>").status == ParseStatus.CHANGED


def test_startpage_sc_token_backs_off_after_failure(monkeypatch):
    # A failing homepage scrape must not be retried on every search — otherwise one
    # blocked homepage serializes a fresh fetch behind the global lock for all callers.
    from core.engines import startpage as sp

    monkeypatch.setattr(sp, "_sc_code", "", raising=False)
    monkeypatch.setattr(sp, "_sc_fetched_at", 0.0, raising=False)
    monkeypatch.setattr(sp, "_sc_failed_at", 0.0, raising=False)

    calls = 0

    async def _fail(_transport) -> str:
        nonlocal calls
        calls += 1
        return ""

    monkeypatch.setattr(sp, "_fetch_sc_code", _fail)

    async def _run() -> None:
        assert await sp._get_sc_code(object()) == ""  # first attempt scrapes
        assert await sp._get_sc_code(object()) == ""  # within cooldown → no scrape

    asyncio.run(_run())
    assert calls == 1
