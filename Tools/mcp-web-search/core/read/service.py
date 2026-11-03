# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import custom_domains
from custom_domains import FetchContext, GenericRequest, PageResult
from custom_domains.retail import extract_retail_metadata, prepend_retail_metadata

from core.cache import get_page_cache
from core.config import load_search_config
from core.extract.content_processor import compress_read_page_markdown
from core.extract.nextjs_rsc import extract_nextjs_rsc_text
from core.extract.page_normalizer import has_extractable_content, normalize_page
from core.fetch.antibot import is_antibot
from core.fetch.browser.client import browser_available, browser_fetch
from core.fetch.browser.models import STATUS_TIMEOUT, BrowserFetch
from core.fetch.constants import DEFAULT_UA as _UA
from core.fetch.download_types import get_download_info
from core.fetch.thread_pool import io_pool as _io_pool
from core.fetch.url_utils import (
    UnsafeFetchUrl,
    has_non_text_extension,
    is_non_text_content_type,
    max_safe_redirects,
    validate_public_fetch_url,
    validate_redirect_target,
)
from core.profiles import (
    METHOD_BROWSER,
    METHOD_CURL_CFFI,
    METHOD_HTTPX,
    FetchAttempt,
    domain_of,
    get_override,
    get_runtime_profiles,
)

logger = logging.getLogger("services.read_page")
trace_logger = logging.getLogger("trace.read_page")

# Tag mixed into every read cache key; changing it invalidates all cached reads of the old strategy.
# Bumped 2026-11 when the normalizer stopped cutting pages at 50k chars: older cache entries hold
# truncated text that `find` and the adaptive curve must not reuse.
_READ_PAGE_STRATEGY_VERSION = "2026-11-core-v2"

# Minimum global read deadlines in seconds for Reddit (curl plus browser render) and .onion pages (slow Tor).
_REDDIT_READ_TIMEOUT_SEC = 60.0
_ONION_READ_TIMEOUT_SEC = 90.0

# Video hosts (and their subdomains) that are not fetched as text pages.
_SKIP_HOSTS = ("vimeo.com", "tiktok.com")


# Fetch result and URL helpers

# Outcome of one low-level HTTP fetch with the metadata needed for runtime profiling.
# tls_failed marks a Firefox-grade transport verdict (bad certificate, plain HTTP on
# 443, protocol failure): the host is dead for every honest client, so callers must not
# burn fallback transports or browser slots on it.
@dataclass(slots=True)
class RawFetch:
    html: str | None
    method: str
    user_agent: str = ""
    status: int = 0
    fetch_ms: float = 0.0
    blocked: bool = False
    timed_out: bool = False
    tls_failed: bool = False

    # Build a FetchAttempt for the runtime profile store, folding in parse-time stats.
    def attempt(self, *, parse_ms: float = 0.0, quality: int = 0, success: bool = False) -> FetchAttempt:
        empty = not self.html
        return FetchAttempt(
            method=self.method,
            user_agent=self.user_agent,
            status=self.status,
            fetch_ms=self.fetch_ms,
            parse_ms=parse_ms,
            quality=quality,
            success=success,
            blocked=self.blocked,
            timed_out=self.timed_out,
            empty=empty or (quality == 0 and not success),
        )


# Return the lowercase host of a URL without a leading "www." or "m." prefix.
def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.").removeprefix("m.")


# Return True for HTTP redirect status codes.
def _is_redirect_status(status_code: int) -> bool:
    return status_code in {301, 302, 303, 307, 308}


# Return True when read_page should not fetch HTML (non-text hosts or extensions). PDF URLs are never skipped.
def _is_skippable(url: str) -> bool:
    from core.extract.pdf_extractor import looks_like_pdf_url

    if looks_like_pdf_url(url):
        return False
    if has_non_text_extension(url):
        return True
    return any(_host(url) == s or _host(url).endswith("." + s) for s in _SKIP_HOSTS)


# Return the cache key for a read: the URL with a "__rpv" query parameter carrying the strategy tag and the
# variant label.
def _cache_key_for_read(url: str, *, variant: str = "") -> str:
    parsed = urlparse(url)
    query_items = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k != "__rpv"]
    suffix = f"{_READ_PAGE_STRATEGY_VERSION}:{variant}" if variant else _READ_PAGE_STRATEGY_VERSION
    query_items.append(("__rpv", suffix))
    return urlunparse(parsed._replace(query=urlencode(query_items, doseq=True)))


# Label DNS-shop variant URLs for cache/trace.
def _variant_label(url: str) -> str:
    if url.endswith("/.xaml"):
        return "dns_xaml"
    if "/product/characteristics/" in url:
        return "dns_characteristics"
    return "default"


# Return True when extracted markdown is empty, shorter than min_length, has at most four non-empty lines, or
# consists mostly of the fallback header boilerplate.
def _is_weak_extraction(markdown: str, *, min_length: int) -> bool:
    # Empty or too short text is weak.
    if not markdown:
        return True
    stripped = markdown.strip()
    if not stripped or len(stripped) < min_length:
        return True

    # Very few lines are weak, as is a short text made of the Site/URL/separator header of the fallback.
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) <= 4:
        return True
    boilerplate_hits = sum(1 for marker in ("**Site:**", "**URL:**", "---") if marker in stripped)
    if boilerplate_hits >= 3 and len(lines) <= 6:
        return True
    return False


# Markdown fallbacks

# Wrap already-extracted fallback text in minimal markdown headers.
def _fallback_text_to_markdown(url: str, text: str) -> str:
    return f"**Site:** {_host(url)}\n**URL:** {url}\n\n---\n\n{text.strip()}"


# Convert raw DOM innerText to minimal markdown (SPA last resort).
def _inner_text_to_markdown(url: str, inner_text: str) -> str:
    lines = [line for line in inner_text.splitlines() if line.strip()]
    return _fallback_text_to_markdown(url, "\n\n".join(lines))


# Extract Next.js RSC text and wrap as minimal markdown.
def _nextjs_rsc_to_markdown(url: str, raw_html: str) -> str:
    text = extract_nextjs_rsc_text(raw_html)
    return _fallback_text_to_markdown(url, text) if text else ""


# Recover supplemental prose from JSON data islands when the DOM markdown parsed sparse.
# Word count drives the gate (matches webclaw's SPARSE_THRESHOLD); returns "" when the DOM
# already had enough, nothing parsed, or the recovered text duplicated the DOM.
def _data_island_supplement(raw_html: str, dom_markdown: str) -> str:
    try:
        from core.extract.data_island import try_extract_data_islands

        dom_words = len((dom_markdown or "").split())
        return try_extract_data_islands(raw_html, dom_words, dom_markdown) or ""
    except Exception as exc:  # noqa: BLE001 — a fallback must never break extraction
        logger.debug("data-island extraction skipped: %s", exc)
        return ""


# Return True when an exception chain bottoms out in a TLS failure. httpx wraps ssl.SSLError
# inside ConnectError (walk __cause__/__context__); curl_cffi flattens curl codes
# 35/51/58/60 into a message string, so a text probe is the only uniform test there.
def _is_tls_error(exc: BaseException | None) -> bool:
    import ssl

    # Walk the cause chain, at most eight links deep, testing the type and the message at each link.
    for _ in range(8):
        if exc is None:
            return False
        if isinstance(exc, ssl.SSLError):
            return True
        text = str(exc)
        if "SSL" in text or "certificate verify" in text.lower():
            return True
        exc = exc.__cause__ or exc.__context__
    return False


# Network fetchers

# Fetch HTML via httpx with per-redirect SSRF checks and return an instrumented RawFetch. A non-text content
# type, an anti-bot page, an HTTP error or too many redirects yield a RawFetch without HTML.
async def _fetch_httpx(url: str, timeout: float, tls_verify: bool = True) -> RawFetch:
    t0 = time.perf_counter()
    try:
        import httpx

        # Redirects are followed by hand so every hop is validated as a public URL (SSRF protection).
        current_url = validate_public_fetch_url(url)
        async with httpx.AsyncClient(
            headers={"User-Agent": _UA, "Accept": "text/html,*/*;q=0.8"},
            timeout=timeout,
            follow_redirects=False,
            verify=tls_verify,
        ) as client:
            r = None
            for _ in range(max_safe_redirects() + 1):
                r = await client.get(current_url)
                if _is_redirect_status(r.status_code):
                    current_url = validate_redirect_target(current_url, r.headers.get("location", ""))
                    continue
                break
            else:
                # The redirect limit was exhausted without a final response.
                return RawFetch(None, METHOD_HTTPX, _UA, 0, (time.perf_counter() - t0) * 1000)

            # Accept HTML from a successful response; an anti-bot page is flagged as blocked and dropped.
            fetch_ms = (time.perf_counter() - t0) * 1000
            if r is not None and 200 <= r.status_code < 400:
                if is_non_text_content_type(r.headers.get("content-type", "")):
                    return RawFetch(None, METHOD_HTTPX, _UA, r.status_code, fetch_ms)
                text = r.text
                blocked = bool(text) and is_antibot(text)
                html = text if text and not blocked else None
                return RawFetch(html, METHOD_HTTPX, _UA, r.status_code, fetch_ms, blocked=blocked)
            return RawFetch(None, METHOD_HTTPX, _UA, r.status_code if r else 0, fetch_ms)
    except UnsafeFetchUrl as exc:
        # An unsafe URL is logged and reported as an empty fetch below.
        logger.warning("blocked unsafe read_page fetch url=%r reason=%s", url, exc)
    except Exception as exc:
        # A TLS failure is flagged so callers skip the other transports; other errors give an empty fetch.
        if _is_tls_error(exc):
            logger.info("TLS failure for %s: %s", url, exc)
            return RawFetch(
                None, METHOD_HTTPX, _UA, 0, (time.perf_counter() - t0) * 1000, tls_failed=True
            )
        logger.debug("httpx fetch failed for %s", url, exc_info=True)
    return RawFetch(None, METHOD_HTTPX, _UA, 0, (time.perf_counter() - t0) * 1000)


# Fetch HTML via curl_cffi (Chrome impersonation) with the same redirect-by-redirect SSRF checks as the httpx
# fetch. The blocking client runs in the I/O thread pool.
async def _fetch_curl_cffi(url: str, timeout: int) -> RawFetch:
    loop = asyncio.get_running_loop()

    # Blocking fetch executed in the thread pool; returns an empty RawFetch on any failure.
    def _sync() -> RawFetch:
        t0 = time.perf_counter()
        try:
            from curl_cffi import requests as cffi_req

            # Redirects are followed by hand so every hop is validated as a public URL.
            current_url = validate_public_fetch_url(url)
            r = None
            for _ in range(max_safe_redirects() + 1):
                r = cffi_req.get(
                    current_url,
                    impersonate="chrome124",
                    timeout=timeout,
                    headers={"User-Agent": _UA},
                    allow_redirects=False,
                )
                if _is_redirect_status(int(r.status_code)):
                    current_url = validate_redirect_target(current_url, r.headers.get("location", ""))
                    continue
                break
            else:
                # The redirect limit was exhausted without a final response.
                return RawFetch(None, METHOD_CURL_CFFI, _UA, 0, (time.perf_counter() - t0) * 1000)

            # Accept HTML from a successful response; an anti-bot page is flagged as blocked and dropped.
            fetch_ms = (time.perf_counter() - t0) * 1000
            if is_non_text_content_type(r.headers.get("content-type", "")):
                return RawFetch(None, METHOD_CURL_CFFI, _UA, int(r.status_code), fetch_ms)
            text = r.text
            blocked = bool(text) and is_antibot(text)
            html = text if text and 200 <= int(r.status_code) < 400 and not blocked else None
            return RawFetch(html, METHOD_CURL_CFFI, _UA, int(r.status_code), fetch_ms, blocked=blocked)
        except UnsafeFetchUrl as exc:
            # An unsafe URL is logged and reported as an empty fetch below.
            logger.warning("blocked unsafe read_page curl url=%r reason=%s", url, exc)
        except Exception as exc:
            # A TLS failure is flagged so callers skip the other transports; other errors give an empty fetch.
            if _is_tls_error(exc):
                logger.info("TLS failure for %s: %s", url, exc)
                return RawFetch(
                    None, METHOD_CURL_CFFI, _UA, 0, (time.perf_counter() - t0) * 1000,
                    tls_failed=True,
                )
            logger.debug("curl_cffi fetch failed for %s", url, exc_info=True)
        return RawFetch(None, METHOD_CURL_CFFI, _UA, 0, (time.perf_counter() - t0) * 1000)

    return await loop.run_in_executor(_io_pool, _sync)


# Race httpx vs curl_cffi; the first response that carries HTML wins and the other fetch is cancelled. When
# neither does, return the last result, with tls_failed set if either transport reported a TLS failure.
async def _fetch_race(url: str, timeout: float, tls_verify: bool = True) -> RawFetch:
    # Start both transports; curl_cffi gets three extra seconds of timeout.
    t_httpx = asyncio.create_task(_fetch_httpx(url, timeout, tls_verify=tls_verify))
    t_curl = asyncio.create_task(_fetch_curl_cffi(url, int(timeout) + 3))
    pending: set = {t_httpx, t_curl}
    last: RawFetch | None = None
    tls_failed = False

    # Wait for the transports one by one; a failed task is skipped and the race continues.
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                candidate = task.result()
            except Exception:
                logger.debug("Fetch race task failed for %s", url, exc_info=True)
                continue
            last = candidate
            tls_failed = tls_failed or candidate.tls_failed
            if candidate.html:
                for p in pending:
                    p.cancel()
                return candidate

    # Nobody produced HTML: report the last result, carrying any TLS verdict.
    if last is not None and tls_failed:
        last.tls_failed = True
    return last or RawFetch(None, METHOD_HTTPX, _UA, 0, 0.0)


# Run the warm browser within read_page's deadline and adapt the result into a RawFetch.
async def _fetch_browser(url: str, timeout: float) -> tuple[RawFetch, BrowserFetch | None]:
    result = await browser_fetch(url, wait_sec=4.0, nav_timeout=min(float(timeout), 20.0))

    # A failed or empty render is logged and reported without a browser result.
    if not result.ok or not result.html:
        detail = result.error or f"status={result.status}"
        logger.warning("warm browser read_page fetch failed for %s: %s", url, detail)
        timed_out = result.status == STATUS_TIMEOUT or "timeout" in detail.lower()
        engine = result.engine or "browser"
        return RawFetch(None, METHOD_BROWSER, engine, 0, result.ms, blocked=result.blocked, timed_out=timed_out), None

    # A successful render counts as HTTP 200; the browser engine name takes the place of the user agent.
    return RawFetch(
        result.html, METHOD_BROWSER, result.engine or "browser", 200, result.ms
    ), result


# Fetch PDF bytes with SSRF checks and the MAX_PDF_BYTES ceiling (httpx, then curl_cffi). Return empty bytes
# when the body is too large, is not a PDF, or neither transport succeeds.
async def _fetch_pdf_bytes(url: str, timeout: float, tls_verify: bool = True) -> bytes:
    from core.extract.pdf_extractor import MAX_PDF_BYTES, looks_like_pdf_bytes

    # First attempt: stream the body with httpx, following redirects by hand with SSRF validation.
    try:
        import httpx

        current_url = validate_public_fetch_url(url)
        headers = {"User-Agent": _UA, "Accept": "application/pdf,*/*;q=0.8"}
        async with httpx.AsyncClient(
            headers=headers, timeout=timeout, follow_redirects=False, verify=tls_verify
        ) as client:
            for _ in range(max_safe_redirects() + 1):
                async with client.stream("GET", current_url) as r:
                    if _is_redirect_status(r.status_code):
                        current_url = validate_redirect_target(current_url, r.headers.get("location", ""))
                        continue

                    # Reject oversized bodies early by Content-Length, then again while streaming.
                    try:
                        content_length = int(r.headers.get("content-length", "0") or "0")
                    except ValueError:
                        # An unparsable Content-Length is treated as unknown; the streaming cap still applies.
                        content_length = 0
                    if content_length > MAX_PDF_BYTES:
                        return b""
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in r.aiter_bytes():
                        total += len(chunk)
                        if total > MAX_PDF_BYTES:
                            return b""
                        chunks.append(chunk)

                    # Only a successful response whose bytes really look like a PDF is returned.
                    data = b"".join(chunks)
                    if 200 <= r.status_code < 400 and looks_like_pdf_bytes(data):
                        return data
                    return b""
    except UnsafeFetchUrl as exc:
        logger.warning("blocked unsafe read_page PDF url=%r reason=%s", url, exc)
    except Exception:
        logger.debug("httpx PDF fetch failed for %s", url, exc_info=True)

    # Second attempt: curl_cffi with Chrome impersonation, run in the I/O thread pool.
    loop = asyncio.get_running_loop()

    # Blocking curl_cffi download with the same redirect validation and size ceiling; returns empty bytes on
    # any failure.
    def _sync() -> bytes:
        try:
            from curl_cffi import requests as cffi_req

            # Follow redirects by hand, validating every hop as a public URL.
            current_url = validate_public_fetch_url(url)
            for _ in range(max_safe_redirects() + 1):
                r = cffi_req.get(
                    current_url,
                    impersonate="chrome124",
                    timeout=int(timeout) + 3,
                    headers={"User-Agent": _UA, "Accept": "application/pdf,*/*;q=0.8"},
                    allow_redirects=False,
                )
                if _is_redirect_status(int(r.status_code)):
                    current_url = validate_redirect_target(current_url, r.headers.get("location", ""))
                    continue
                break
            else:
                # The redirect limit was exhausted without a final response.
                return b""

            # Return the body only for a successful response within the size ceiling that looks like a PDF.
            data = bytes(r.content or b"")
            if 200 <= int(r.status_code) < 400 and len(data) <= MAX_PDF_BYTES and looks_like_pdf_bytes(data):
                return data
        except UnsafeFetchUrl as exc:
            logger.warning("blocked unsafe read_page PDF curl url=%r reason=%s", url, exc)
        except Exception:
            logger.debug("curl_cffi PDF fetch failed for %s", url, exc_info=True)
        return b""

    return await loop.run_in_executor(_io_pool, _sync)


# Read page service

# Options of one read_page call: the per-fetch timeout in seconds, the character budget of the returned
# markdown and an optional focus phrase that steers the compression.
@dataclass
class ReadPageOptions:
    timeout: float = 20.0
    max_chars: int = 20_000
    focus: str = ""
    # When False, the warm browser path is disabled entirely. web_search sets this so
    # the browser stays exclusive to the read_page tool: a wide search must be cheap/
    # HTTP-only and skip browser-only sources rather than pay seconds a page.
    allow_browser: bool = True
    # Size-adaptive compression (the read_page tool). False keeps the fixed char budget
    # with focus-query chunk selection that web_search previews rely on.
    adaptive: bool = False
    # grep-style lookup: when set, return only the matching passages of the full page.
    find: str = ""


# Return the global asyncio deadline in seconds for one read: at least 30 seconds, raised to the Reddit floor
# (curl plus browser render) or the .onion floor (Tor circuits and descriptors are slow) for those hosts.
def _read_page_deadline(url: str, opts: ReadPageOptions) -> float:
    base = float(opts.timeout)
    if _host(url) == "reddit.com" or _host(url).endswith(".reddit.com"):
        return max(base, _REDDIT_READ_TIMEOUT_SEC)
    if _host(url).endswith(".onion"):
        return max(base, _ONION_READ_TIMEOUT_SEC)
    return max(base, 30.0)


# Fetch a single URL and return clean markdown text via the custom-domain dispatch
# layer and a profile-driven generic pipeline.
class ReadPageService:
    # Load the search config, page cache and runtime fetch profiles; options default to the configured
    # extraction timeout and character limit.
    def __init__(self, options: Optional[ReadPageOptions] = None) -> None:
        cfg = load_search_config()
        self._cfg = cfg
        self._cache = get_page_cache()
        self._profiles = get_runtime_profiles()
        self._opts = options or ReadPageOptions(
            timeout=cfg.extraction.timeout_seconds,
            max_chars=cfg.extraction.max_page_chars,
        )
        self._focus = (self._opts.focus or "").strip()

    # Return whether the warm browser path may run. Disabled by web_search so the browser
    # stays exclusive to the read_page tool; otherwise gated on backend availability.
    async def _browser_ok(self) -> bool:
        return self._opts.allow_browser and await browser_available()

    # Apply the read_page compression from config: the size-adaptive curve for the
    # read_page tool, or the fixed BM25 budget for web_search previews.
    def _apply_budget(self, markdown: str, url: str) -> str:
        ext = self._cfg.extraction
        if self._opts.adaptive:
            if not ext.enable_read_page_compress:
                return markdown
            from core.extract.adaptive_compression import (
                CompressionCurve,
                compress_adaptive,
                compression_note,
                insert_note,
            )

            curve = CompressionCurve(
                soft_limit=ext.read_page_soft_limit_chars,
                ratio_at_double=ext.read_page_ratio_at_double,
                max_ratio=ext.read_page_max_ratio,
                hard_cap=ext.read_page_max_output_chars,
            )
            compressed, report = compress_adaptive(markdown, curve, url=url, focus=self._focus)
            if report.compressed:
                logger.info(
                    "read_page.adaptive url=%r original=%d target=%d output=%d dropped=%d "
                    "trimmed=%d protected=%d kinds=%s hard_capped=%s",
                    url, report.original_chars, report.target_chars, report.output_chars,
                    report.dropped_blocks, report.trimmed_blocks, report.protected_chars,
                    report.kinds, report.hard_capped,
                )
                compressed = insert_note(compressed, compression_note(report))
            return compressed
        return compress_read_page_markdown(
            markdown,
            url=url,
            focus=self._focus,
            max_chars=self._opts.max_chars,
            compress_threshold=ext.read_page_compress_threshold_chars,
            compress_target=ext.read_page_compress_target_chars,
            enable_compress=ext.enable_read_page_compress,
        )

    # Build the FetchContext handed to custom-domain handlers.
    def _context(self) -> FetchContext:
        return FetchContext(
            timeout=self._opts.timeout,
            max_chars=self._opts.max_chars,
            focus=self._focus,
            cfg=self._cfg,
            cache=self._cache,
            generic_read=self._generic_read,
        )

    # Pick the fetch strategy for a domain from hard overrides then runtime profiles. Return
    # (browser_first, http_method, prefer_rsc): whether to start with the warm browser, which single HTTP
    # transport to use instead of racing (None races both), and whether to prefer Next.js RSC extraction.
    def _resolve_strategy(self, url: str, req: GenericRequest) -> tuple[bool, str | None, bool]:
        domain = domain_of(url)
        override = get_override(domain)
        hint = self._profiles.best_method(domain)

        # The request and hard overrides decide first.
        prefer_rsc = req.prefer_rsc or bool(override and override.parsing_mode == "nextjs_rsc")
        browser_first = req.browser_first or bool(override and override.required_method == METHOD_BROWSER)
        http_method: str | None = None

        # A confident learned profile may switch to the browser or pin one HTTP transport.
        if hint and not hint.avoid and hint.confidence >= 0.5:
            if hint.method == METHOD_BROWSER:
                browser_first = True
            elif hint.method in (METHOD_HTTPX, METHOD_CURL_CFFI):
                http_method = hint.method
        return browser_first, http_method, prefer_rsc

    # Fetch one candidate URL with the chosen method (warm browser / single http / race).
    async def _fetch_candidate(
        self, url: str, *, browser_first: bool, http_method: str | None
    ) -> tuple[RawFetch, BrowserFetch | None]:
        if browser_first and await self._browser_ok():
            return await _fetch_browser(url, self._opts.timeout)
        if http_method == METHOD_HTTPX:
            return await _fetch_httpx(url, self._opts.timeout, tls_verify=self._cfg.search.tls_verify), None
        if http_method == METHOD_CURL_CFFI:
            return await _fetch_curl_cffi(url, int(self._opts.timeout) + 3), None
        return await _fetch_race(url, self._opts.timeout, tls_verify=self._cfg.search.tls_verify), None

    # Generic fetch+normalise pipeline shared by the default path and strategy handlers.
    # Records every attempt into the runtime profile store so later reads skip dead ends.
    # Candidate URL variants are tried in order until one yields non-weak markdown (the last one is accepted
    # as is); each is served from the cache when fresh, otherwise fetched, escalated to the warm browser on
    # empty, anti-bot or weak results, extracted, and cached.
    async def _generic_read(self, req: GenericRequest) -> PageResult:
        # Resolve the strategy and prepare the state shared by all variants.
        url = req.url
        min_len = self._cfg.extraction.min_content_length
        browser_first, http_method, prefer_rsc = self._resolve_strategy(url, req)
        variants = req.url_variants or [url]

        markdown = ""
        raw_html: str | None = None
        winning_method = ""
        last_http_status = 0

        for idx, cand in enumerate(variants):
            # Look the variant up in the cache first.
            variant_label = _variant_label(cand)
            cache_key = _cache_key_for_read(cand, variant=variant_label)
            cached = self._cache.get_cached(cache_key)
            cache_fresh = cached is not None and self._cache.is_fresh(cache_key)

            # Reuse previously extracted markdown directly — no network, no re-extraction.
            # The cache stores clean text (not raw HTML), so a repeat read is instant and the
            # entry stays FTS-searchable. raw_html-only entries (legacy/warm) fall through.
            if cache_fresh and cached.clean_text:
                md_cached = cached.clean_text
                if not _is_weak_extraction(md_cached, min_length=min_len) or idx == len(variants) - 1:
                    raw_html, markdown, winning_method = "cache", md_cached, "cache"
                    trace_logger.info(
                        "read_page.variant url=%r variant=%s method=cache markdown_len=%d (clean_text reuse)",
                        cand, variant_label, len(md_cached),
                    )
                    break
                continue

            # Get HTML from a legacy raw-HTML cache entry, otherwise fetch it with the chosen strategy.
            raw: RawFetch | None = None
            cam: BrowserFetch | None = None
            if cache_fresh and cached.raw_html:
                html: str | None = cached.raw_html
                method = "cache"
            else:
                raw, cam = await self._fetch_candidate(
                    cand, browser_first=browser_first, http_method=http_method
                )
                html = raw.html
                method = raw.method

            # Empty fetch: record the failed attempt, then try the warm browser once before giving up on
            # this variant.
            if not html:
                if raw is not None:
                    if raw.method != METHOD_BROWSER and raw.status:
                        last_http_status = raw.status
                    self._profiles.record(cand, raw.attempt(success=False))

                    # Firefox-grade verdict: the host does not speak valid TLS. Feed the
                    # reputation store and move on — no transport (or browser) will do
                    # better, they all validate the same chain.
                    if raw.tls_failed:
                        self._profiles.record_reputation(cand, tls_failed=True)
                        continue

                # Do this at most once; browser-first failures must not recurse back here.
                if method != METHOD_BROWSER and await self._browser_ok():
                    logger.info(
                        "empty HTTP fetch for %s (status=%s) — retrying via warm browser",
                        cand, raw.status if raw is not None else 0,
                    )
                    browser_raw, browser_result = await _fetch_browser(cand, self._opts.timeout)
                    if browser_raw.html:
                        raw, cam = browser_raw, browser_result
                        html, method = browser_raw.html, METHOD_BROWSER
                    else:
                        self._profiles.record(cand, browser_raw.attempt(success=False))
                        continue
                else:
                    continue

            # Anti-bot wall — escalate to a real browser once if not already there.
            if is_antibot(html):
                if raw is not None:
                    self._profiles.record(cand, raw.attempt(success=False))
                if method != METHOD_BROWSER and await self._browser_ok():
                    raw, cam = await _fetch_browser(cand, self._opts.timeout)
                    if raw.html and not is_antibot(raw.html):
                        html, method = raw.html, METHOD_BROWSER
                    else:
                        if raw is not None:
                            self._profiles.record(cand, raw.attempt(success=False))
                        continue
                else:
                    continue

            # A served PDF text dump should go through the PDF extractor instead.
            from core.extract.pdf_extractor import looks_like_pdf_text_dump

            if looks_like_pdf_text_dump(html):
                return await self._read_pdf(url)

            # Extract markdown from the HTML; parse_ms feeds the runtime profile of the host.
            parse0 = time.perf_counter()
            html_text: str = html

            # Extraction (trafilatura/lxml) is CPU-bound; run it off the event loop so a
            # pathological page can't block the loop — that would freeze sibling parses
            # AND make the asyncio deadline unenforceable (timeouts fire only at awaits).
            # Build the markdown of the current variant: Next.js RSC text when preferred and available,
            # otherwise the normalized DOM plus data-island prose, with retail metadata prepended.
            def _extract() -> str:
                retail_meta = extract_retail_metadata(cand, html_text)
                if prefer_rsc:
                    rsc = _nextjs_rsc_to_markdown(cand, html_text)
                    if rsc:
                        return prepend_retail_metadata(rsc, retail_meta)
                page_md = normalize_page(cand, html_text)

                # SPA fallback: when the DOM parsed sparse, recover prose from JSON data
                # islands (React/Next/Contentful ship content as script JSON and hydrate
                # client-side, so a static fetch's DOM is near-empty). Appended, never a
                # replacement — genuine DOM content stays first and duplicates are dropped.
                island_md = _data_island_supplement(html_text, page_md)
                if island_md:
                    page_md = f"{page_md}\n\n{island_md}" if page_md.strip() else island_md
                return prepend_retail_metadata(page_md, retail_meta)

            md = await asyncio.get_running_loop().run_in_executor(_io_pool, _extract)
            parse_ms = (time.perf_counter() - parse0) * 1000
            weak = _is_weak_extraction(md, min_length=min_len)

            if raw is not None:
                self._profiles.record(
                    cand, raw.attempt(parse_ms=parse_ms, quality=len(md), success=not weak)
                )

            # Weak HTML extraction — retry through the warm browser (normalize → innerText → RSC).
            if weak and method not in (METHOD_BROWSER, "cache") and await self._browser_ok():
                logger.info("weak extraction for %s — retrying via warm browser SPA fallback", cand)
                craw, cres = await _fetch_browser(cand, self._opts.timeout)
                if craw.html and cres is not None:
                    # SPA recover also runs CPU-bound extraction — keep it off the loop.
                    md, weak, method = await asyncio.get_running_loop().run_in_executor(
                        _io_pool, self._spa_recover, cand, cres, md, min_len
                    )
                    html = cres.html
                    self._profiles.record(cand, craw.attempt(quality=len(md), success=not weak))
                else:
                    self._profiles.record(cand, craw.attempt(success=False))

            # Trust observation: what the page physically parsed into, after all retries.
            # Cache reuse records nothing — no new evidence about the live host.
            if method != "cache":
                self._profiles.record_reputation(cand, parse_ok=not weak, parse_empty=weak)

            # Cache the EXTRACTED markdown (not raw HTML): far smaller, FTS-searchable, and
            # reused verbatim by the next read with no re-fetch or re-extraction. Only a
            # genuine, non-weak, non-antibot success is stored (after any SPA retry above).
            if method != "cache" and not weak and html and not is_antibot(html):
                title = md.splitlines()[0].lstrip("# ").strip()[:200] if md.strip() else ""
                self._cache.cache_page(cache_key, title, clean_text=md, raw_html="")

            raw_html, markdown, winning_method = html, md, method

            trace_logger.info(
                "read_page.variant url=%r variant=%s method=%s markdown_len=%d weak=%s",
                cand, variant_label, method, len(md), weak,
            )
            if not weak or idx == len(variants) - 1:
                break

        # Turn the outcome of the variant loop into a PageResult: fetch failure, anti-bot wall, too little
        # content (still returned, with a warning) or a normal success.
        if not raw_html:
            status_detail = f" (HTTP {last_http_status})" if last_http_status else ""
            return PageResult(
                markdown=f"Error: Could not fetch content from: {url}{status_detail}", ok=False
            )
        if is_antibot(raw_html):
            return PageResult(
                markdown=(
                    f"Error: Anti-bot protection detected on {url}. "
                    "Cannot extract content without a browser."
                ),
                ok=False,
                blocked=True,
            )
        if not markdown or len(markdown.strip()) < min_len:
            return PageResult(
                markdown=f"Warning: Very little content extracted from: {url}\n\n{markdown}",
                ok=True,
                method=winning_method,
            )
        return PageResult(markdown=markdown, ok=True, method=winning_method, apply_budget=True)

    # Recover the best markdown from a warm-browser SPA render: normalize → innerText → RSC.
    # Return (markdown, weak, method), taking the first source that is not weak or at least longer than the
    # previous markdown; when none helps, keep the previous markdown marked weak.
    def _spa_recover(
        self, url: str, cres: BrowserFetch, prev_md: str, min_len: int
    ) -> tuple[str, bool, str]:
        # First choice: the normalized rendered DOM.
        spa_md = normalize_page(url, cres.html)
        spa_weak = _is_weak_extraction(spa_md, min_length=min_len)
        if not spa_weak or len(spa_md) > len(prev_md or ""):
            return spa_md, spa_weak, METHOD_BROWSER

        # Second choice: the page's visible innerText.
        if cres.text and len(cres.text.strip()) > min_len:
            it_md = _inner_text_to_markdown(url, cres.text)
            if not _is_weak_extraction(it_md, min_length=min_len) or len(it_md) > len(prev_md or ""):
                logger.info("SPA innerText fallback used for %s", url)
                return it_md, _is_weak_extraction(it_md, min_length=min_len), "browser_innertext"

        # Last choice: Next.js RSC payload text.
        rsc_md = _nextjs_rsc_to_markdown(url, cres.html)
        if rsc_md and (not _is_weak_extraction(rsc_md, min_length=min_len) or len(rsc_md) > len(prev_md or "")):
            logger.info("SPA Next.js RSC fallback used for %s", url)
            return rsc_md, _is_weak_extraction(rsc_md, min_length=min_len), "browser_rsc"

        return prev_md, True, METHOD_BROWSER

    # Download a PDF and return extracted markdown.
    async def _read_pdf(self, url: str) -> PageResult:
        from core.extract.pdf_extractor import pdf_bytes_to_markdown

        # Download the bytes, then convert them to markdown; a failed conversion is treated as no text.
        data = await _fetch_pdf_bytes(url, timeout=self._opts.timeout, tls_verify=self._cfg.search.tls_verify)
        if not data:
            return PageResult(markdown=f"Error: Could not fetch PDF content from: {url}", ok=False)
        try:
            markdown = pdf_bytes_to_markdown(url=url, data=data, max_chars=self._opts.max_chars)
        except Exception as exc:
            logger.warning("PDF extraction failed for %s: %s", url, exc)
            markdown = ""
        if not markdown:
            return PageResult(markdown=f"Error: Could not extract text from PDF: {url}", ok=False)

        # Cache the extracted text under the first line as the title.
        self._cache.cache_page(url, markdown.splitlines()[0].lstrip("# ").strip()[:500], markdown, "")
        return PageResult(markdown=markdown, ok=True, method="pdf")

    # Fetch a .onion page over Tor and extract it through the same normalizer as the generic
    # path. Gated on tor.enabled; soft, clear errors when tor is off/unreachable or the page
    # is walled. Uses the tor fetch_timeout (Tor is slow) rather than the short read timeout.
    async def _read_onion(self, url: str) -> PageResult:
        # The onion layer is imported lazily; a missing module or tor-disabled config gives a clear error.
        try:
            from core.config import load_search_config
            from core.fetch.onion import onion_fetch
        except Exception:  # noqa: BLE001
            return PageResult(markdown=f"Error: onion layer unavailable: {url}", ok=False)
        if not load_search_config().tor.enabled:
            return PageResult(
                markdown=f"Error: this is an onion URL; enable the tor path in config to read it: {url}",
                ok=False,
            )

        # Fetch over Tor; with no explicit timeout, onion_fetch uses tor.fetch_timeout.
        result = await onion_fetch(url)
        if not result.ok or not result.text:
            logger.warning(
                "onion fetch failed url=%r http_status=%d error=%s",
                url,
                result.http_status,
                result.error or result.status,
            )
            return PageResult(
                markdown=f"Error: could not fetch onion page ({result.error or result.status}): {url}",
                ok=False,
            )

        # Reject challenge walls, then extract off the event loop (favoring recall for sparse onion pages).
        html = result.text
        if is_antibot(html):
            return PageResult(markdown=f"Error: onion page returned an antibot/challenge wall: {url}", ok=False)
        md = await asyncio.get_running_loop().run_in_executor(
            _io_pool, lambda: normalize_page(url, html, favor_recall=True)
        )
        if not has_extractable_content(md):
            logger.warning(
                "onion extraction returned no page content url=%r html_len=%d", url, len(html)
            )
            return PageResult(markdown=f"Error: no extractable content from onion page: {url}", ok=False)

        # A short Onion page is still valid content; the generic reader follows the same rule.
        weak = _is_weak_extraction(md, min_length=self._cfg.extraction.min_content_length)
        if not weak:
            self._cache.cache_page(url, md.splitlines()[0].lstrip("# ").strip()[:500], md, "")
        output = f"Warning: Very little content extracted from: {url}\n\n{md}" if weak else md
        return PageResult(markdown=output, ok=True, method="onion", apply_budget=True)

    # Core read pipeline: SSRF, custom-domain dispatch, then the generic pipeline. Downloadable files and
    # unsupported URL types get an explanatory message instead of a fetch; PDFs use the PDF reader.
    async def _read(self, url: str) -> str:
        from core.extract.pdf_extractor import looks_like_pdf_url

        logger.info("read_page url=%r", url)

        # .onion is fetched through Tor (not a direct connect), and can't pass the public-host
        # SSRF check — route it to the onion path before SSRF. Gated on tor.enabled inside.
        if _host(url).endswith(".onion"):
            return self._finalize(url, await self._read_onion(url))

        # Refuse URLs that fail the public-host SSRF check.
        try:
            validate_public_fetch_url(url)
        except UnsafeFetchUrl as exc:
            logger.warning("blocked unsafe read_page url=%r reason=%s", url, exc)
            return f"Error: URL blocked by SSRF protection: {url}"

        # A matching custom-domain handler goes first; its failure only continues when it opts into the
        # generic fallback.
        handler = custom_domains.match(url)
        if handler is not None:
            result = await handler.read(url, self._context())
            if result.ok or not getattr(handler, "fallback_to_generic", False):
                return self._finalize(url, result)

        # Non-text URLs are not fetched: describe a downloadable file, or report the type as unsupported.
        if _is_skippable(url):
            if has_non_text_extension(url):
                info = get_download_info(url)
                if info:
                    ext, category = info
                    return (
                        f"This URL points to a downloadable file, not a web page.\n"
                        f"  URL      : {url}\n"
                        f"  Extension: {ext}\n"
                        f"  Category : {category}"
                    )
            return f"Error: URL type not supported for text extraction: {url}"

        # PDFs have their own reader; everything else goes through the generic pipeline.
        if looks_like_pdf_url(url):
            result = await self._read_pdf(url)
            return self._finalize(url, result)

        result = await self._generic_read(GenericRequest(url=url))
        return self._finalize(url, result)

    # Apply `find` or the compression budget when the result asks for it, then return markdown.
    def _finalize(self, url: str, result: PageResult) -> str:
        markdown = result.markdown or ""
        lowered = markdown.lstrip().lower()
        if self._opts.find.strip() and markdown and not lowered.startswith("error:"):
            # find always runs on the full extracted text, never on compressed output.
            from core.extract.page_find import find_in_page

            ext = self._cfg.extraction
            return find_in_page(
                markdown,
                self._opts.find,
                url=url,
                max_hits=ext.read_page_find_max_hits,
                max_chars=ext.read_page_find_max_chars,
            )
        # Handlers and PDFs that skip the fixed budget used to self-truncate at a small
        # max_chars; in adaptive mode they get the hard ceiling instead, so the curve covers
        # every result (it is a no-op under the soft limit).
        wants_budget = result.apply_budget or self._opts.adaptive
        if wants_budget and markdown and not lowered.startswith(("error:", "warning:")):
            markdown = self._apply_budget(markdown, url)
        return markdown

    # Public entry: fetch the URL as markdown, returning an error message when the global deadline passes.
    async def read(self, url: str) -> str:
        url = url.strip()
        deadline = _read_page_deadline(url, self._opts)
        try:
            return await asyncio.wait_for(self._read(url), timeout=deadline)
        except asyncio.TimeoutError:
            logger.warning("read_page global deadline exceeded (%.0fs) for %s", deadline, url)
            return f"Error: Page read timed out after {int(deadline)}s: {url}"


# Convenience entry point for the central API and debug CLI: read one URL with the given limits. Reddit
# reads get at least the Reddit timeout floor. With max_chars given (web_search previews) the budget is
# fixed and `focus` selects chunks; with max_chars omitted (the read_page tool) the size-adaptive curve
# applies, with the hard ceiling from config as the extraction limit, and `find` greps the full page.
async def run_read_page(
    url: str,
    timeout: float = 20.0,
    max_chars: int | None = None,
    focus: str = "",
    allow_browser: bool = True,
    find: str = "",
) -> str:
    host = urlparse(url.strip()).netloc.lower().removeprefix("www.").removeprefix("m.")
    effective_timeout = max(timeout, _REDDIT_READ_TIMEOUT_SEC) if host == "reddit.com" else timeout
    adaptive = max_chars is None
    if max_chars is None:
        ext = load_search_config().extraction
        # Handlers and the PDF extractor cut at max_chars before compression or find see
        # the text: give find the whole page, and the adaptive curve its hard ceiling.
        max_chars = ext.read_page_find_scan_chars if find.strip() else ext.read_page_max_output_chars
    service = ReadPageService(
        options=ReadPageOptions(
            timeout=effective_timeout, max_chars=max_chars, focus=focus,
            allow_browser=allow_browser, adaptive=adaptive, find=find,
        )
    )
    return await service.read(url)
