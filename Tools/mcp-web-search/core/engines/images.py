# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import html
import json
import re
import threading
import time
from dataclasses import dataclass
from urllib.parse import parse_qs, unquote, urlsplit

from ..fetch.profiles import accept_language_for, build_nav_headers, for_engine
from .models import EngineParseResult, EngineRequest, ParseStatus
from .parsing import clean_text, node_text, parse_html, split_region, valid_http_url


# Result model and shared helpers

# One image plus the page that published it; the shared normalized result shape of all image engines.
# url is the publishing page, image_url the full-size image and thumbnail_url its preview; width, height
# and filesize are None when unknown.
@dataclass(frozen=True, slots=True)
class ImageResult:
    title: str
    url: str
    snippet: str = ""
    image_url: str = ""
    thumbnail_url: str = ""
    source: str = ""
    source_domain: str = ""
    width: int | None = None
    height: int | None = None
    image_format: str = ""
    filesize: int | None = None
    author: str = ""
    license: str = ""


# Return the lowercase host of a URL without a leading "www.", or an empty string if it cannot be parsed.
def _domain(value: str) -> str:
    try:
        return (urlsplit(value).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


# Convert a value to a positive integer, or None when it is not a number or not greater than zero.
def _integer(value: object) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


# Build a normalized ImageResult from raw engine fields, or None when the page URL or the image URL is not a
# valid http(s) URL. Text fields are whitespace-cleaned; a missing title falls back to the page domain, a
# missing or invalid thumbnail to the full image, and a missing source to the domain.
def _image_result(
    *,
    title: object,
    url: object,
    image_url: object,
    thumbnail_url: object = "",
    snippet: object = "",
    source: object = "",
    width: object = None,
    height: object = None,
    image_format: object = "",
    filesize: object = None,
    author: object = "",
    license_name: object = "",
) -> ImageResult | None:
    # Validate the page and image URLs; a bad thumbnail is dropped instead of rejecting the result.
    page = str(url or "").strip()
    original = str(image_url or "").strip()
    thumbnail = str(thumbnail_url or "").strip()
    if not valid_http_url(page) or not valid_http_url(original):
        return None
    if thumbnail and not valid_http_url(thumbnail):
        thumbnail = ""

    # Normalize the remaining fields and apply the domain and image fallbacks.
    domain = _domain(page)
    clean_title = clean_text([str(title or "")]) or domain
    return ImageResult(
        title=clean_title,
        url=page,
        snippet=clean_text([str(snippet or "")]),
        image_url=original,
        thumbnail_url=thumbnail or original,
        source=clean_text([str(source or "")]) or domain,
        source_domain=domain,
        width=_integer(width),
        height=_integer(height),
        image_format=str(image_format or "").strip().upper(),
        filesize=_integer(filesize),
        author=clean_text([str(author or "")]),
        license=str(license_name or "").strip(),
    )


# Build the EngineParseResult of an image engine. Images win over every other flag (PARTIAL when any card
# was malformed), then blocked, then changed; with none of them the page is a genuine empty result.
def _parse_result(
    engine: str,
    images: list[ImageResult],
    *,
    cards_seen: int,
    malformed: int,
    variant: str,
    blocked: bool = False,
    changed: bool = False,
    diagnostics: list[str] | None = None,
) -> EngineParseResult:
    # Pick the status in priority order.
    if images:
        status = ParseStatus.PARTIAL if malformed else ParseStatus.SUCCESS
    elif blocked:
        status = ParseStatus.BLOCKED
    elif changed:
        status = ParseStatus.CHANGED
    else:
        status = ParseStatus.EMPTY

    # ImageResult stands in for SearchResult in the shared result container, hence the type ignore.
    return EngineParseResult(
        engine=engine,
        status=status,
        results=images,  # type: ignore[arg-type]
        parser_variant=variant,
        cards_seen=cards_seen,
        malformed_cards=malformed,
        diagnostics=diagnostics or [],
    )


# Headers a browser sends only on top-level navigations, never on XHR/fetch.
_NAVIGATION_ONLY_HEADERS = frozenset(
    {"Sec-Fetch-User", "Upgrade-Insecure-Requests", "Cache-Control"}
)


# Bing images

# A real Bing async results page is hundreds of KB; the soft-block shell is ~1 KB.
_BING_SHELL_MAX_BYTES = 8_000


# Bing image search parser for the async results endpoint; each result card carries its metadata as JSON.
class BingImagesParser:
    name = "bing_images"
    provider_family = "bing"
    search_url = "https://www.bing.com/images/async"
    page_size = 35

    # Build the async Bing images request: the result offset follows the page number, the time limit maps to
    # an age filter in minutes and safe search maps to the adlt level.
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
        profile = for_engine("bing_images")

        # Query parameters and the optional age filter.
        params = {
            "q": query,
            "async": "1",
            "first": str((max(1, page) - 1) * BingImagesParser.page_size + 1),
            "count": str(BingImagesParser.page_size),
            "cc": country.upper(),
            "setlang": language,
            "adlt": {"off": "off", "moderate": "moderate", "on": "strict"}.get(
                safesearch, "moderate"
            ),
        }
        if timelimit:
            minutes = {"d": 1440, "w": 10080, "m": 44640, "y": 525600}.get(timelimit)
            if minutes:
                params["qft"] = f"filterui:age-lt{minutes}"

        # Navigation headers as if searching from within bing.com, with the region's Accept-Language.
        extra: dict[str, str] = {}
        accept_language = accept_language_for(language, country)
        if accept_language:
            extra["Accept-Language"] = accept_language
        return EngineRequest(
            method="GET",
            url=BingImagesParser.search_url,
            params=params,
            headers=build_nav_headers(
                profile,
                referer="https://www.bing.com/images",
                sec_fetch_site="same-origin",
                extra=extra or None,
            ),
            primp_target=profile.primp_target,
            primp_os=profile.primp_os,
            identity_key="bing_images",
        )

    # Parse a Bing async images document: each a.iusc card holds an HTML-escaped JSON blob in its "m"
    # attribute, and a card whose blob is invalid or lacks usable URLs counts as malformed.
    def parse(self, document: str) -> EngineParseResult:
        # Decode the metadata of every card into an ImageResult.
        tree = parse_html(document)
        cards = tree.css("a.iusc")
        images: list[ImageResult] = []
        malformed = 0
        for card in cards:
            raw = html.unescape(str(card.attributes.get("m") or ""))
            try:
                metadata = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                malformed += 1
                continue
            image = _image_result(
                title=metadata.get("t"),
                url=metadata.get("purl"),
                image_url=metadata.get("murl"),
                thumbnail_url=metadata.get("turl"),
                snippet=metadata.get("desc"),
                source=_domain(str(metadata.get("purl") or "")),
            )
            if image is None:
                malformed += 1
            else:
                images.append(image)
        lowered = document.casefold()

        # Bing soft-blocks a flagged session silently: HTTP 200 with a script-only
        # shell and no result cards. A fresh cookie jar recovers it (verified live),
        # so report it as BLOCKED to trigger the cookieless retry.
        soft_blocked = not cards and 0 < len(document) < _BING_SHELL_MAX_BYTES
        return _parse_result(
            self.name,
            images,
            cards_seen=len(cards),
            malformed=malformed,
            variant="metadata_cards",
            blocked=soft_blocked or any(
                marker in lowered
                for marker in ("challenge-platform", "verify you are human", "/captcha/")
            ),
            changed=not cards and bool(document.strip()),
            diagnostics=["empty result shell (soft block)"] if soft_blocked else None,
        )


# DuckDuckGo images

# DuckDuckGo vqd tokens: i.js requires a vqd token scraped from the search page. Tokens are cached for an
# hour per (normalized query, user agent) and guarded by a lock because searches run on several threads.
# The patterns extract the token from plain and JSON-escaped page text.
_VQD_TTL_SECONDS = 3600
_VQD_CACHE: dict[tuple[str, str], tuple[str, float]] = {}
_VQD_LOCK = threading.Lock()
_VQD_PATTERNS = (
    re.compile(r'vqd=["\']([^"\']+)'),
    re.compile(r'vqd\\?"\s*:\s*\\?"([^"\\]+)'),
)


# Build the vqd cache key from the whitespace-normalized query and the user agent.
def _vqd_key(query: str, user_agent: str) -> tuple[str, str]:
    return (" ".join(str(query or "").split()), str(user_agent or ""))


# Return the cached vqd token for the query and user agent, or an empty string when absent or expired
# (an expired entry is evicted).
def _cached_vqd(query: str, user_agent: str) -> str:
    key = _vqd_key(query, user_agent)
    with _VQD_LOCK:
        value, expires = _VQD_CACHE.get(key, ("", 0.0))
        if expires <= time.time():
            _VQD_CACHE.pop(key, None)
            return ""
        return value


# Cache a vqd token for the query and user agent; empty inputs are ignored.
def _store_vqd(query: str, user_agent: str, value: str) -> None:
    clean = str(value or "").strip()
    if not query or not user_agent or not clean:
        return
    with _VQD_LOCK:
        _VQD_CACHE[_vqd_key(query, user_agent)] = (
            clean,
            time.time() + _VQD_TTL_SECONDS,
        )


# DuckDuckGo image search parser for the i.js JSON endpoint. It needs a two-step flow: a warm-up request to
# the search page captures the vqd token, which the i.js request then sends.
class DuckDuckGoImagesParser:
    name = "duckduckgo_images"
    provider_family = "duckduckgo"
    home_url = "https://duckduckgo.com/"
    search_url = "https://duckduckgo.com/i.js"
    page_size = 100

    # Return True when no vqd token is cached for the query, so the warm-up request must run first.
    @staticmethod
    def needs_warmup(query: str) -> bool:
        profile = for_engine("duckduckgo_images")
        return not _cached_vqd(query, profile.user_agent)

    # Build the search page request whose response carries the vqd token. Safe search, time limit and page do
    # not matter for it and are ignored.
    @staticmethod
    def build_warmup_request(
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        page: int = 1,
    ) -> EngineRequest:
        del safesearch, timelimit, page
        country, language = split_region(region)
        profile = for_engine("duckduckgo_images")
        extra: dict[str, str] = {}
        accept_language = accept_language_for(language, country)
        if accept_language:
            extra["Accept-Language"] = accept_language
        return EngineRequest(
            method="GET",
            url=DuckDuckGoImagesParser.home_url,
            params={"q": query, "iar": "images", "iax": "images", "ia": "images", "t": "h_"},
            headers=build_nav_headers(profile, extra=extra or None),
            primp_target=profile.primp_target,
            primp_os=profile.primp_os,
            identity_key="duckduckgo_images",
        )

    # Extract the vqd token from the warm-up response and cache it under the request's query and user agent.
    @staticmethod
    def capture_response(request: EngineRequest, document: str) -> None:
        query = str(request.params.get("q") or "")
        user_agent = str(request.headers.get("User-Agent") or "")
        for pattern in _VQD_PATTERNS:
            match = pattern.search(document)
            if match:
                _store_vqd(query, user_agent, match.group(1))
                return

    # Build the i.js JSON request using the cached vqd token. Raise RuntimeError when the token is missing,
    # because the warm-up request must have run first.
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
        profile = for_engine("duckduckgo_images")
        vqd = _cached_vqd(query, profile.user_agent)
        if not vqd:
            raise RuntimeError("DuckDuckGo image token is not initialized")

        # Query parameters; the offset follows the page and the time limit is passed through.
        params = {
            "o": "json",
            "q": query,
            "u": "bing",
            "l": region,
            "bpia": "1",
            "vqd": vqd,
            "a": "h_",
            "ct": country.upper(),
        }
        if page > 1:
            params["s"] = str((page - 1) * DuckDuckGoImagesParser.page_size)
        if timelimit:
            params["f"] = f"time:{timelimit}"

        # The kl cookie carries the region; safe search (off or strict) goes into both parameter and cookie.
        cookies = {"kl": region}
        safe_value = {"off": "-2", "on": "1"}.get(safesearch)
        if safe_value:
            params["p"] = safe_value
            cookies["p"] = safe_value

        # XHR-style headers: i.js is fetched by script, so Accept and Sec-Fetch-Dest/Mode describe a CORS fetch.
        extra = {
            "Accept": "application/json, text/javascript, */*; q=0.01",
            "Sec-Fetch-Dest": "empty",
            "Sec-Fetch-Mode": "cors",
        }
        accept_language = accept_language_for(language, country)
        if accept_language:
            extra["Accept-Language"] = accept_language

        # i.js is an XHR, not a navigation. A real browser never sends the
        # navigation-only headers on it, and DDG's anti-bot answers any request
        # carrying Sec-Fetch-User with a 202 JSON challenge (verified live).
        headers = {
            name: value
            for name, value in build_nav_headers(
                profile,
                referer=DuckDuckGoImagesParser.home_url,
                sec_fetch_site="same-origin",
                extra=extra,
            ).items()
            if name not in _NAVIGATION_ONLY_HEADERS
        }
        return EngineRequest(
            method="GET",
            url=DuckDuckGoImagesParser.search_url,
            params=params,
            headers=headers,
            cookies=cookies,
            primp_target=profile.primp_target,
            primp_os=profile.primp_os,
            identity_key="duckduckgo_images",
        )

    # Parse an i.js JSON response. A body that is not JSON is CHANGED, or BLOCKED when it contains a bot
    # challenge marker; a JSON "challenge" object is BLOCKED; otherwise every row of "results" becomes an image.
    def parse(self, document: str) -> EngineParseResult:
        # Decode the JSON body; challenge pages are served as HTML instead of JSON.
        try:
            payload = json.loads(document)
        except json.JSONDecodeError as exc:
            lowered = document.casefold()
            return _parse_result(
                self.name,
                [],
                cards_seen=0,
                malformed=0,
                variant="json",
                blocked=any(
                    marker in lowered
                    for marker in (
                        "anomaly-modal",
                        "challenge-form",
                        "verify you are human",
                        "botnet_challenge",
                        "response_type",
                    )
                ),
                changed=True,
                diagnostics=[f"Invalid JSON response: {exc}"],
            )
        if isinstance(payload, dict) and isinstance(payload.get("challenge"), dict):
            # HTTP 202 + {"challenge": {...}}: DDG's JSON anti-bot answer.
            return _parse_result(
                self.name,
                [],
                cards_seen=0,
                malformed=0,
                variant="json",
                blocked=True,
                diagnostics=["DuckDuckGo JSON challenge"],
            )

        # Convert each result row; rows that are not objects or lack usable URLs count as malformed.
        rows = payload.get("results") if isinstance(payload, dict) else []
        rows = rows if isinstance(rows, list) else []
        images: list[ImageResult] = []
        malformed = 0
        for row in rows:
            if not isinstance(row, dict):
                malformed += 1
                continue
            image = _image_result(
                title=row.get("title"),
                url=row.get("url"),
                image_url=row.get("image"),
                thumbnail_url=row.get("thumbnail"),
                source=row.get("source"),
                width=row.get("width"),
                height=row.get("height"),
            )
            if image is None:
                malformed += 1
            else:
                images.append(image)
        return _parse_result(
            self.name,
            images,
            cards_seen=len(rows),
            malformed=malformed,
            variant="json",
        )


# Brave images

# Regex template (filled with the escaped key name) that captures the double-quoted string value of a
# JavaScript object key such as title:"..." inside Brave's embedded page data.
_JS_STRING_TEMPLATE = r'(?<![\w]){name}:"((?:\\.|[^"\\])*)"'


# Return the string value of the key in a JavaScript object literal, decoding its escapes, or an empty
# string when the key is missing. Undecodable escapes leave the raw text.
def _js_string(document: str, name: str) -> str:
    match = re.search(_JS_STRING_TEMPLATE.format(name=re.escape(name)), document)
    if not match:
        return ""
    try:
        return str(json.loads(f'"{match.group(1)}"'))
    except json.JSONDecodeError:
        return match.group(1)


# Return the positive integer value of the key in a JavaScript object literal, or None when it is missing.
def _js_integer(document: str, name: str) -> int | None:
    match = re.search(rf"(?<![\w]){re.escape(name)}:(\d+)", document)
    return _integer(match.group(1)) if match else None


# Return the brace-balanced block that starts at the opening brace index, skipping braces inside quoted
# strings, or an empty string when the block never closes.
def _brace_block(document: str, opening: int) -> str:
    # Track brace depth and whether the scan is inside a quoted string (with a pending escape).
    depth = 0
    quoted = False
    escaped = False
    for index in range(opening, len(document)):
        character = document[index]
        if quoted:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return document[opening : index + 1]
    return ""


# Brave image search parser. The images page embeds its results as JavaScript object data instead of
# markup, so the parser scans that text directly.
class BraveImagesParser:
    name = "brave_images"
    provider_family = "brave"
    search_url = "https://search.brave.com/images"

    # Build the Brave images page request. Time limit and page are not supported and ignored.
    @staticmethod
    def build_request(
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        page: int = 1,
    ) -> EngineRequest:
        del timelimit, page
        country, language = split_region(region)
        profile = for_engine("brave_images")

        # Cookies carry the region, UI language and safe search level; location and summarizer are switched off.
        cookies = {
            "country": country,
            "useLocation": "0",
            "summarizer": "0",
            "ui_lang": f"{language}-{country}",
            "safesearch": {"off": "off", "moderate": "moderate", "on": "strict"}.get(
                safesearch, "moderate"
            ),
        }

        # Navigation headers as if searching from within brave.com; Accept-Encoding is limited to gzip and
        # deflate.
        extra = {"Accept-Encoding": "gzip, deflate"}
        accept_language = accept_language_for(language, country)
        if accept_language:
            extra["Accept-Language"] = accept_language
        return EngineRequest(
            method="GET",
            url=BraveImagesParser.search_url,
            params={"q": query, "source": "web"},
            headers=build_nav_headers(
                profile,
                referer="https://search.brave.com/",
                sec_fetch_site="same-origin",
                extra=extra,
            ),
            cookies=cookies,
            primp_target=profile.primp_target,
            primp_os=profile.primp_os,
            identity_key="brave_images",
        )

    # Parse the embedded page data: every "properties:{url:" marks one image. The item starts at the nearest
    # preceding "{title:" and holds the page fields, an optional thumbnail object and then the properties
    # object with the full-size image fields. An image without such a start counts as malformed.
    def parse(self, document: str) -> EngineParseResult:
        # One marker per image; previous stops the backwards search from reaching into the previous image.
        markers = [match.start() for match in re.finditer(r"properties:\{url:", document)]
        images: list[ImageResult] = []
        malformed = 0
        previous = 0
        for marker in markers:
            item_start = document.rfind("{title:", previous, marker)
            previous = marker + 1
            if item_start < 0:
                malformed += 1
                continue

            # Cut out the properties object, the optional thumbnail object and the header text that holds the
            # title, page URL, description and source.
            properties_open = document.find("{", marker)
            properties = _brace_block(document, properties_open)
            thumbnail_marker = document.find("thumbnail:{", item_start, marker)
            thumbnail = ""
            if thumbnail_marker >= 0:
                thumbnail = _brace_block(document, document.find("{", thumbnail_marker))
            header = document[item_start : thumbnail_marker if thumbnail_marker >= 0 else marker]

            # Read the fields out of the JavaScript text and normalize them.
            image = _image_result(
                title=_js_string(header, "title"),
                url=_js_string(header, "url"),
                image_url=_js_string(properties, "url"),
                thumbnail_url=_js_string(thumbnail, "src"),
                snippet=_js_string(header, "description"),
                source=_js_string(header, "source"),
                width=_js_integer(properties, "width"),
                height=_js_integer(properties, "height"),
                image_format=_js_string(properties, "format"),
            )
            if image is None:
                malformed += 1
            else:
                images.append(image)

        # Bot challenge markers mark the page as blocked; a non-empty page without any image marker means the
        # embedded data format changed.
        lowered = document.casefold()
        return _parse_result(
            self.name,
            images,
            cards_seen=len(markers),
            malformed=malformed,
            variant="embedded_data",
            blocked=any(
                marker in lowered
                for marker in ("challenge-platform", "verify you are human", "cf-error-code")
            ),
            changed=not markers and bool(document.strip()),
        )


# Openverse images

# Openverse image search parser for its public JSON API.
class OpenverseImagesParser:
    name = "openverse_images"
    provider_family = "openverse"
    search_url = "https://api.openverse.org/v1/images/"
    # Anonymous access is capped at page_size 20 — anything larger returns 401.
    page_size = 20

    # Build the Openverse API request. The time limit is not supported and ignored; mature content is only
    # requested when safe search is off.
    @staticmethod
    def build_request(
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        page: int = 1,
    ) -> EngineRequest:
        del timelimit
        country, language = split_region(region)
        profile = for_engine("openverse_images")

        # Plain JSON API headers: the profile's user agent and the region's Accept-Language.
        headers = {
            "User-Agent": profile.user_agent,
            "Accept": "application/json",
        }
        accept_language = accept_language_for(language, country)
        if accept_language:
            headers["Accept-Language"] = accept_language
        return EngineRequest(
            method="GET",
            url=OpenverseImagesParser.search_url,
            params={
                "q": query,
                "page": str(max(1, page)),
                "page_size": str(OpenverseImagesParser.page_size),
                "mature": "true" if safesearch == "off" else "false",
            },
            headers=headers,
            identity_key="openverse_images",
        )

    # Parse an Openverse JSON response: invalid JSON is CHANGED, otherwise every row of "results" becomes an
    # image with its license, creator and file details.
    def parse(self, document: str) -> EngineParseResult:
        # Decode the JSON body.
        try:
            payload = json.loads(document)
        except json.JSONDecodeError as exc:
            return _parse_result(
                self.name,
                [],
                cards_seen=0,
                malformed=0,
                variant="json",
                changed=True,
                diagnostics=[f"Invalid JSON response: {exc}"],
            )

        # Convert each result row; rows that are not objects or lack usable URLs count as malformed.
        rows = payload.get("results") if isinstance(payload, dict) else []
        rows = rows if isinstance(rows, list) else []
        images: list[ImageResult] = []
        malformed = 0
        for row in rows:
            if not isinstance(row, dict):
                malformed += 1
                continue

            # Join license and version into one label such as "CC-BY 4.0".
            license_name = " ".join(
                part
                for part in (
                    str(row.get("license") or "").upper(),
                    str(row.get("license_version") or ""),
                )
                if part
            )
            image = _image_result(
                title=row.get("title"),
                url=row.get("foreign_landing_url") or row.get("detail_url"),
                image_url=row.get("url"),
                thumbnail_url=row.get("thumbnail"),
                source=row.get("source"),
                width=row.get("width"),
                height=row.get("height"),
                image_format=row.get("filetype"),
                filesize=row.get("filesize"),
                author=row.get("creator"),
                license_name=license_name,
            )
            if image is None:
                malformed += 1
            else:
                images.append(image)
        return _parse_result(
            self.name,
            images,
            cards_seen=len(rows),
            malformed=malformed,
            variant="json",
        )


# Engine registry

# Image parser classes by engine type name. Ported from ASLM-Search (four of its seven engines): qwant,
# dogpile and google-wml stay unported because all three are hard-blocked by DataDome/CloudFront/403 on most
# residential networks and would only rot here. Live-checked defaults (2026-10-28) are bing, brave and
# duckduckgo; openverse ships config-off because it timed out on every request.
IMAGE_ENGINE_TYPES = {
    "bing": BingImagesParser,
    "brave": BraveImagesParser,
    "duckduckgo": DuckDuckGoImagesParser,
    "openverse": OpenverseImagesParser,
}


__all__ = [
    "BingImagesParser",
    "BraveImagesParser",
    "DuckDuckGoImagesParser",
    "IMAGE_ENGINE_TYPES",
    "ImageResult",
    "OpenverseImagesParser",
]
