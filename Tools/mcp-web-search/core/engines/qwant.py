# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
from urllib.parse import urlparse

from ..fetch.profiles import accept_language_for, build_nav_headers, for_engine
from .models import EngineParseResult, EngineRequest, ParseStatus, SearchResult
from .parsing import split_region

# Qwant safesearch level expected by the API: 0=off, 1=moderate, 2=strict.
_SAFESEARCH = {"off": "0", "moderate": "1", "on": "2"}


# Qwant web-search parser backed by Qwant's undocumented JSON API.
#
# The API mirrors what https://www.qwant.com/ requests in its network log. The
# web vertical returns a 'mainline' list whose blocks each carry a 'type'; only
# 'web' blocks hold organic results (ads and side modules are skipped).
class QwantParser:
    name = "qwant"
    provider_family = "qwant"
    search_url = "https://api.qwant.com/v3/search/web"

    # Build a Qwant API request for the requested locale and safe-search level.
    @staticmethod
    def build_request(
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        page: int = 1,
    ) -> EngineRequest:
        country, language = split_region(region)
        profile = for_engine("qwant")

        # Query parameters of the web vertical: ten results per page with the offset following the page. The
        # time limit is not supported and ignored.
        params = {
            "q": query,
            "count": "10",
            "locale": f"{language}_{country.upper()}",
            "offset": str(max(0, page - 1) * 10),
            "device": "desktop",
            "safesearch": _SAFESEARCH.get(safesearch, "1"),
            "tgp": "1",
            "display": "true",
            "llm": "false",
        }

        # XHR-style headers: the API is called by the Qwant web app with a same-site Origin and CORS fetch mode.
        extra = {
            "Accept": "application/json",
            "Origin": "https://www.qwant.com",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
        }
        accept_language = accept_language_for(language, country)
        if accept_language:
            extra["Accept-Language"] = accept_language
        headers = build_nav_headers(
            profile,
            referer="https://www.qwant.com/",
            sec_fetch_site="same-site",
            extra=extra,
        )
        return EngineRequest(
            method="GET",
            url=QwantParser.search_url,
            params=params,
            headers=headers,
            primp_target=profile.primp_target,
            primp_os=profile.primp_os,
            identity_key="qwant",
        )

    # Parse Qwant's JSON response, excluding ads and non-web modules.
    def parse(self, document: str) -> EngineParseResult:
        # Decode the JSON body; anything else means the API changed.
        try:
            payload = json.loads(document)
        except json.JSONDecodeError as exc:
            return EngineParseResult(
                engine=self.name,
                status=ParseStatus.CHANGED,
                diagnostics=[f"Invalid JSON response: {exc}"],
            )

        # A bare {"url": ".../captcha..."} payload is an upstream anti-bot (DataDome)
        # interstitial served before Qwant's own API even runs.
        if isinstance(payload, dict) and "captcha" in str(payload.get("url") or "").lower():
            return EngineParseResult(
                engine=self.name,
                status=ParseStatus.BLOCKED,
                diagnostics=["Anti-bot captcha interstitial"],
            )

        # A non-success status is reported as BLOCKED for rate limits and challenges, as ERROR otherwise.
        data = payload.get("data") or {}
        if payload.get("status") != "success":
            error_data = data.get("error_data") or {}

            # error_code 24 is Qwant's rate-limit signal; a captchaUrl means a challenge.
            blocked = data.get("error_code") == 24 or bool(error_data.get("captchaUrl"))
            message = data.get("message") or "Qwant API error"
            if isinstance(message, list):
                message = ", ".join(str(part) for part in message) or "Qwant API error"
            return EngineParseResult(
                engine=self.name,
                status=ParseStatus.BLOCKED if blocked else ParseStatus.ERROR,
                diagnostics=[str(message)],
            )

        # Collect results from the "web" blocks of the mainline only; items without a title or a valid http(s)
        # URL are counted as malformed.
        mainline = (((data.get("result") or {}).get("items") or {}).get("mainline") or [])
        results: list[SearchResult] = []
        cards_seen = 0
        malformed = 0
        for block in mainline:
            if block.get("type") != "web":
                continue
            for item in block.get("items") or []:
                cards_seen += 1
                title = str(item.get("title") or "").strip()
                href = str(item.get("url") or "").strip()
                snippet = str(item.get("desc") or "").strip()
                parsed = urlparse(href)
                if not title or parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    malformed += 1
                    continue
                results.append(SearchResult(title=title, url=href, snippet=snippet))

        # No results is an empty result, and any malformed item downgrades a success to PARTIAL.
        status = ParseStatus.SUCCESS if results else ParseStatus.EMPTY
        if results and malformed:
            status = ParseStatus.PARTIAL
        return EngineParseResult(
            engine=self.name,
            status=status,
            results=results,
            parser_variant="json_mainline",
            cards_seen=cards_seen,
            malformed_cards=malformed,
        )
