# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import html
import json
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from .google import GoogleHtmlParser
from .models import EngineParseResult, EngineRequest, ParseStatus, SearchResult
from .parsing import split_region, valid_http_url

# Programmable Search endpoints: the bootstrap script that issues the token, and the JSONP search element
# API. The default engine id is the public partner id used by the Google search element; the token stays
# valid for 55 minutes before it is requested again.
_BOOTSTRAP_URL = "https://www.google.com/cse/cse.js"
_SEARCH_URL = "https://cse.google.com/cse/element/v1"
_DEFAULT_ENGINE_ID = "partner-pub-8993703457585266:4862972284"
_TOKEN_TTL_SECONDS = 55 * 60


# Structural type of an HTTP response returned by the transport (status, raw body and decoded text).
class _TransportResponse(Protocol):
    status: int
    body: bytes

    # Return the response body decoded as text.
    def text(self) -> str: ...


# Structural type of the HTTP transport backend that executes an EngineRequest.
class _Transport(Protocol):
    # Execute the request and return the response.
    async def fetch(self, request: EngineRequest) -> _TransportResponse: ...


# One CSE bootstrap token with the library version and experiment flags it was issued with. The expiry
# is a time.monotonic() timestamp.
@dataclass(frozen=True, slots=True)
class _CseToken:
    value: str
    library_version: str
    experiments: str
    expires_at: float


# Bootstrap tokens by engine id, so concurrent searches reuse one token until it expires.
_token_cache: dict[str, _CseToken] = {}


# Return the Programmable Search engine id, taken from GOOGLE_CSE_ID or the built-in default.
def _engine_id() -> str:
    return (os.environ.get("GOOGLE_CSE_ID") or _DEFAULT_ENGINE_ID).strip()


# Parse the CSE bootstrap script and return the token it embeds. Raise ValueError when the script has no
# token, the embedded object is malformed or the token is empty. The optional now (a monotonic timestamp)
# makes the expiry deterministic.
def _extract_bootstrap_token(document: str, *, now: float | None = None) -> _CseToken:
    # Locate the JSON object that contains "cse_token": it is the argument of a call ending in "});".
    marker = '"cse_token"'
    marker_at = document.find(marker)
    if marker_at < 0:
        raise ValueError("CSE bootstrap did not contain a token")
    object_start = document.rfind("({", 0, marker_at)
    object_end = document.find("});", marker_at)
    if object_start < 0 or object_end < 0:
        raise ValueError("CSE bootstrap payload was malformed")

    # Decode the object and read the token.
    payload = json.loads(document[object_start + 1 : object_end + 1])
    if not isinstance(payload, dict):
        raise ValueError("CSE bootstrap payload was not an object")
    value = str(payload.get("cse_token") or "").strip()
    if not value:
        raise ValueError("CSE bootstrap token was empty")

    # Experiment flags come as a list or a single value; normalize them to a comma-separated string.
    experiments = payload.get("exp")
    if isinstance(experiments, list):
        experiment_text = ",".join(str(item) for item in experiments if item is not None)
    else:
        experiment_text = str(experiments or "")

    # Stamp the token with its expiry.
    clock = time.monotonic() if now is None else now
    return _CseToken(
        value=value,
        library_version=str(payload.get("cselibVersion") or ""),
        experiments=experiment_text,
        expires_at=clock + _TOKEN_TTL_SECONDS,
    )


# Return a valid CSE token for the configured engine id, using the cache while the token is fresh and
# otherwise downloading the bootstrap script through the transport. Raise RuntimeError on an HTTP error.
async def _get_token(transport: _Transport) -> _CseToken:
    # Reuse the cached token until it expires.
    engine_id = _engine_id()
    now = time.monotonic()
    cached = _token_cache.get(engine_id)
    if cached is not None and cached.expires_at > now:
        return cached

    # Download the bootstrap script and cache the token extracted from it.
    response = await transport.fetch(
        EngineRequest(
            method="GET",
            url=_BOOTSTRAP_URL,
            params={"cx": engine_id},
            headers={"Accept": "*/*"},
            cookies={"CONSENT": "YES+"},
            identity_key="google",
        )
    )
    if response.status >= 400:
        raise RuntimeError(f"CSE bootstrap returned HTTP {response.status}")
    token = _extract_bootstrap_token(response.text(), now=now)
    _token_cache[engine_id] = token
    return token


# Convert a time limit (d, w, m or y) into a CSE date range sort expression, or an empty string for no limit.
def _date_sort(timelimit: str | None) -> str:
    days = {"d": 1, "w": 7, "m": 30, "y": 365}.get(str(timelimit or "").lower())
    if days is None:
        return ""
    end = datetime.now(UTC).date()
    start = end - timedelta(days=days)
    return f"date:r:{start:%Y%m%d}:{end:%Y%m%d}"


# Decode a JSONP response (callback wrapper around a JSON object) into a dict. Raise ValueError when no
# JSON object is found or the payload is not an object.
def _decode_jsonp(document: str) -> dict[str, Any]:
    start = document.find("{")
    end = document.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("CSE response did not contain JSON")
    payload = json.loads(document[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("CSE response was not an object")
    return payload


# Google web results through the public Programmable Search element backend. When the bootstrap token
# cannot be fetched it falls back to the plain Google HTML parser for that request.
class GoogleParser:
    name = "google"
    provider_family = "google"

    # Start with the CSE backend selected; build_request_async switches to the HTML reserve on failure.
    def __init__(self) -> None:
        self._html_fallback = False

    # Build the CSE search request, fetching (or reusing) the bootstrap token through the transport. If the
    # token cannot be obtained, build a Google HTML request instead and remember that parse must use the
    # HTML parser.
    async def build_request_async(
        self,
        transport: _Transport,
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        page: int = 1,
    ) -> EngineRequest:
        # Obtain the token; any failure switches this parser to the HTML reserve.
        try:
            token = await _get_token(transport)
        except Exception:
            self._html_fallback = True
            return GoogleHtmlParser.build_request(
                query,
                region=region,
                safesearch=safesearch,
                timelimit=timelimit,
                page=page,
            )

        # Map the safe search level to the CSE value.
        country, language = split_region(region)
        safe = {"on": "high", "moderate": "medium", "off": "off"}.get(
            safesearch, "medium"
        )

        # Query parameters of the JSONP search element call: the token, language and country restrictions,
        # 20 results per page and the callback name that wraps the response.
        params = {
            "rsz": "filtered_cse",
            "num": "20",
            "hl": language,
            "cselibv": token.library_version,
            "cx": _engine_id(),
            "q": query,
            "safe": safe,
            "cse_tok": token.value,
            "callback": "__aslm_cse",
            "rurl": "",
            "searchtype": "",
            "lr": f"lang_{language}",
            "cr": f"country{country.upper()}",
            "gl": country.upper(),
        }

        # Optional parameters: experiment flags, result offset for later pages and a date range sort.
        if token.experiments:
            params["exp"] = token.experiments
        if page > 1:
            params["start"] = str((page - 1) * 20)
        if date_sort := _date_sort(timelimit):
            params["sort"] = date_sort
        return EngineRequest(
            method="GET",
            url=_SEARCH_URL,
            params=params,
            headers={"Accept": "*/*", "Referer": "https://cse.google.com/"},
            cookies={"CONSENT": "YES+"},
            identity_key="google",
        )

    # Parse the response of the request built by build_request_async: a JSONP document from the CSE backend,
    # or an HTML page when the HTML reserve was selected.
    def parse(self, document: str) -> EngineParseResult:
        # The request used the HTML reserve, so delegate to the HTML parser and note it in the diagnostics.
        if self._html_fallback:
            result = GoogleHtmlParser().parse(document)
            result.diagnostics.append("CSE bootstrap unavailable; HTML reserve used.")
            return result

        # Decode the JSONP wrapper; a body that is not JSON means the backend changed.
        try:
            payload = _decode_jsonp(document)
        except Exception as exc:
            return EngineParseResult(
                engine=self.name,
                status=ParseStatus.CHANGED,
                diagnostics=[f"Invalid CSE response: {exc}"],
            )

        # An error object is reported as BLOCKED for HTTP 429 and as ERROR otherwise.
        error = payload.get("error")
        if isinstance(error, dict):
            code = int(error.get("code") or 0)
            message = str(error.get("message") or "CSE request failed")
            return EngineParseResult(
                engine=self.name,
                status=ParseStatus.BLOCKED if code == 429 else ParseStatus.ERROR,
                diagnostics=[f"CSE error {code}: {message}"],
            )

        # Read the result items; entries that are not objects, or lack a title or a valid http(s) URL, are
        # counted as malformed.
        raw_results = payload.get("results")
        cards = raw_results if isinstance(raw_results, list) else []
        results: list[SearchResult] = []
        malformed = 0
        for item in cards:
            if not isinstance(item, dict):
                malformed += 1
                continue
            url = str(item.get("unescapedUrl") or item.get("url") or "").strip()
            title = html.unescape(str(item.get("titleNoFormatting") or "")).strip()
            snippet = html.unescape(str(item.get("contentNoFormatting") or "")).strip()
            if not title or not valid_http_url(url):
                malformed += 1
                continue
            results.append(SearchResult(title=title, url=url, snippet=snippet))

        # Without results, an empty result list is EMPTY, while a missing result set means the format changed.
        if results:
            status = ParseStatus.PARTIAL if malformed else ParseStatus.SUCCESS
        else:
            status = ParseStatus.EMPTY if isinstance(raw_results, list) else ParseStatus.CHANGED
        diagnostics = [] if status != ParseStatus.CHANGED else ["CSE response had no result set."]
        return EngineParseResult(
            engine=self.name,
            status=status,
            results=results,
            parser_variant="cse_jsonp",
            cards_seen=len(cards),
            malformed_cards=malformed,
            diagnostics=diagnostics,
        )


__all__ = ["GoogleParser"]
