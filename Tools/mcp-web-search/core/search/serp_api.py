# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, replace
from typing import Any, Protocol
from urllib.parse import urlparse

import orjson

from ..engines import (
    BraveParser,
    DuckDuckGoParser,
    GoogleParser,
    ParseStatus,
    QwantParser,
    StartpageParser,
    YandexParser,
)
from ..engines.models import EngineParseResult, EngineRequest
from ..fetch.transport import AdaptiveTransport, TransportResponse

DEFAULT_ENGINES = (
    GoogleParser,
    BraveParser,
    DuckDuckGoParser,
    YandexParser,
    QwantParser,
    StartpageParser,
)


# Protocol for transport backends accepted by SerpApi.
class SerpTransport(Protocol):
    async def fetch(self, request: EngineRequest) -> TransportResponse: ...

    async def close(self) -> None: ...


# Build one engine's request, using its optional async builder when present.
#
# Most engines build a stateless request synchronously. Engines that need a
# preflight call (e.g. Startpage prefetching its sc token) expose an async
# build_request_async(transport, ...) instead; both produce a plain EngineRequest,
# so the rest of the pipeline stays uniform with no per-engine branches.
async def _build_engine_request(
    parser: Any,
    transport: SerpTransport,
    query: str,
    *,
    region: str,
    safesearch: str,
    timelimit: str | None,
) -> EngineRequest:
    builder = getattr(parser, "build_request_async", None)
    if builder is not None:
        return await builder(
            transport, query, region=region, safesearch=safesearch, timelimit=timelimit
        )
    return parser.build_request(query, region=region, safesearch=safesearch, timelimit=timelimit)


# Serialize a payload to indented JSON bytes using orjson.
def encode_json(payload: Any) -> bytes:
    return orjson.dumps(payload, option=orjson.OPT_INDENT_2)


# Return the lowercased host of a URL, or an empty string.
def _host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ""


# Build an error EngineParseResult with a single diagnostic message.
def _error_parse_result(engine: str, message: str) -> EngineParseResult:
    return EngineParseResult(engine=engine, status=ParseStatus.ERROR, diagnostics=[message])


# Drop the engine's stored HTTP cookies for the request host (captcha-burn recovery).
# Best-effort: a store failure never blocks the retry itself.
def _drop_identity_cookies(request: EngineRequest) -> int:
    if not request.identity_key:
        return 0
    host = _host_of(request.url)
    if not host:
        return 0
    try:
        from ..fetch.browser.identity_store import get_identity_store

        return get_identity_store().clear_http_cookies(request.identity_key, host)
    except Exception:  # noqa: BLE001 — recovery must never break the fetch
        return 0


# Serialize one engine parse result into the final payload dict.
def _parse_result_payload(
    result: EngineParseResult,
    *,
    provider_family: str,
    limit: int,
    http_status: int | None,
    fetch_ms: float,
    parse_ms: float,
    response_bytes: int,
    transport: str,
) -> dict[str, Any]:
    return {
        "engine": result.engine,
        "provider_family": provider_family,
        "status": result.status.value,
        "http_status": http_status,
        "fetch_ms": round(fetch_ms, 2),
        "parse_ms": round(parse_ms, 2),
        "response_bytes": response_bytes,
        "transport": transport,
        "parser_variant": result.parser_variant,
        "cards_seen": result.cards_seen,
        "malformed_cards": result.malformed_cards,
        "coverage": round(result.coverage, 3),
        "sources": [asdict(source) for source in result.results[:limit]],
        "diagnostics": result.diagnostics,
    }


# Run general-purpose engines concurrently through one pooled transport.
class SerpApi:

    # Initialize the API with an optional pre-built transport and search limits.
    # engines selects which parsers run; callers (tiering/health) pass a subset.
    def __init__(
        self,
        transport: SerpTransport | None = None,
        *,
        timeout_seconds: float = 8.0,
        source_limit: int = 3,
        engines: tuple[type, ...] = DEFAULT_ENGINES,
    ) -> None:
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.source_limit = max(1, int(source_limit))
        self.engines = tuple(engines) or DEFAULT_ENGINES
        self._owns_transport = transport is None
        self._transport = transport or AdaptiveTransport(timeout_seconds=self.timeout_seconds)

    # Close the owned transport if this instance created it.
    async def close(self) -> None:
        if self._owns_transport:
            await self._transport.close()

    # Support async context manager entry.
    async def __aenter__(self) -> "SerpApi":
        return self

    # Close resources on async context manager exit.
    async def __aexit__(self, *_args: object) -> None:
        await self.close()

    # Fetch one engine, parse its response, and return the serialized result dict.
    # blocked_retries re-fetches on a BLOCKED outcome after dropping the engine's
    # stored identity cookies: a captcha branch is usually pinned to the session
    # cookie (sticky per identity), so a cookieless re-roll recovers it far more
    # often than replaying the same request would.
    async def _run_engine(
        self,
        parser_type,
        query: str,
        *,
        region: str,
        safesearch: str,
        timelimit: str | None,
        blocked_retries: int = 0,
    ) -> dict[str, Any]:
        parser = parser_type()
        fetch_started = time.perf_counter()
        http_status: int | None = None
        response_bytes = 0
        transport = ""
        fetch_ms = 0.0
        parse_ms = 0.0
        request: EngineRequest | None = None
        try:
            async with asyncio.timeout(self.timeout_seconds):
                # Warmup protocol (image engines): a token-hungry parser gets one
                # cheap priming request first (e.g. DuckDuckGo's vqd), whose response
                # is captured into the parser's own token cache before the real fetch.
                needs_warmup = getattr(parser, "needs_warmup", None)
                capture_response = getattr(parser, "capture_response", None)
                if callable(needs_warmup) and needs_warmup(query):
                    warmup_builder = getattr(parser, "build_warmup_request", None)
                    if callable(warmup_builder):
                        warmup_request = warmup_builder(
                            query, region=region, safesearch=safesearch, timelimit=timelimit
                        )
                        warmup_response = await self._transport.fetch(warmup_request)
                        if warmup_response.status < 400 and callable(capture_response):
                            capture_response(warmup_request, warmup_response.text())
                request = await _build_engine_request(
                    parser,
                    self._transport,
                    query,
                    region=region,
                    safesearch=safesearch,
                    timelimit=timelimit,
                )
                response = await self._transport.fetch(request)
                if response.status < 400 and callable(capture_response):
                    capture_response(request, response.text())
            fetch_ms = (time.perf_counter() - fetch_started) * 1000
            http_status = response.status
            response_bytes = len(response.body)
            transport = response.transport
            if response.status >= 400:
                result = _error_parse_result(parser.name, f"HTTP {response.status}")
                if response.status in {403, 429}:
                    result.status = ParseStatus.BLOCKED
            else:
                parse_started = time.perf_counter()
                result = parser.parse(response.text())
                parse_ms = (time.perf_counter() - parse_started) * 1000
        except TimeoutError:
            fetch_ms = (time.perf_counter() - fetch_started) * 1000
            result = _error_parse_result(parser.name, f"Timed out after {self.timeout_seconds:.1f}s")
            result.status = ParseStatus.TIMEOUT
        except Exception as exc:  # noqa: BLE001
            fetch_ms = (time.perf_counter() - fetch_started) * 1000
            result = _error_parse_result(parser.name, f"{type(exc).__name__}: {exc}")

        # Captcha-burn recovery loop. A blocked branch sticks to TWO cookie jars the
        # first attempt rode: the identity store (replayed into the request) and the
        # transport's pooled client (primp keeps its own jar). Each retry drops both
        # and re-fetches with identity_key cleared — a truly cookieless re-roll. The
        # final outcome is what the health tracker records, so exhausted retries still
        # cool the engine down instead of hiding the failure.
        attempts = 0
        parser_identity = request.identity_key if request is not None else ""
        while (
            blocked_retries > 0
            and result.status == ParseStatus.BLOCKED
            and attempts < blocked_retries
            and request is not None
        ):
            attempts += 1
            host = _host_of(request.url)
            _drop_identity_cookies(request)
            drop_clients = getattr(self._transport, "drop_identity_clients", None)
            if callable(drop_clients) and host:
                drop_clients(host)
            try:
                async with asyncio.timeout(self.timeout_seconds):
                    request = await _build_engine_request(
                        parser,
                        self._transport,
                        query,
                        region=region,
                        safesearch=safesearch,
                        timelimit=timelimit,
                    )
                    # Naked fetch: no identity replay and no Set-Cookie capture, so the
                    # retry cannot inherit or persist the burned branch.
                    request = replace(request, identity_key="")
                    response = await self._transport.fetch(request)
                    request = replace(request, identity_key=parser_identity)
            except TimeoutError:
                result = _error_parse_result(parser.name, f"Timed out after {self.timeout_seconds:.1f}s")
                result.status = ParseStatus.TIMEOUT
                break
            except Exception as exc:  # noqa: BLE001
                result = _error_parse_result(parser.name, f"{type(exc).__name__}: {exc}")
                break
            fetch_ms = (time.perf_counter() - fetch_started) * 1000
            http_status = response.status
            response_bytes = len(response.body)
            transport = response.transport
            if response.status >= 400:
                result = _error_parse_result(parser.name, f"HTTP {response.status}")
                if response.status in {403, 429}:
                    result.status = ParseStatus.BLOCKED
            else:
                parse_started = time.perf_counter()
                result = parser.parse(response.text())
                parse_ms = (time.perf_counter() - parse_started) * 1000
            result.diagnostics.append(f"captcha retry {attempts}/{blocked_retries}")

        return _parse_result_payload(
            result,
            provider_family=str(getattr(parser, "provider_family", parser.name)),
            limit=self.source_limit,
            http_status=http_status,
            fetch_ms=fetch_ms,
            parse_ms=parse_ms,
            response_bytes=response_bytes,
            transport=transport,
        )

    # Stream sources and per-engine status events through a short-lived in-process
    # buffer as each engine completes, yielding them in real time. The deadline is a
    # hard cutoff: producers still running when it fires are cancelled, and whatever
    # already reached the buffer is the result (timeouts are normal operation, not a
    # fallback). Each item is {"type": "source" | "vote" | "engine", ...}; "vote" marks
    # a URL re-surfaced by another engine (consensus signal for triage).
    async def search_stream(
        self,
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        engine_queries: dict[str, str] | None = None,
        engine_timelimits: dict[str, str | None] | None = None,
        engine_omitted_operators: dict[str, tuple[str, ...]] | None = None,
        engine_blocked_retries: dict[str, int] | None = None,
        deadline_seconds: float | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        query = " ".join(str(query or "").split()).strip()
        if not query:
            raise ValueError("query must not be empty")

        deadline = self.timeout_seconds if deadline_seconds is None else max(0.1, float(deadline_seconds))
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        seen: set[str] = set()

        # Run one engine, emit each new source, then emit its full status payload.
        async def run(parser_type) -> None:
            engine_query = (engine_queries or {}).get(parser_type.name, query)
            engine_timelimit = (engine_timelimits or {}).get(parser_type.name, timelimit)
            payload = await self._run_engine(
                parser_type,
                engine_query,
                region=region,
                safesearch=safesearch,
                timelimit=engine_timelimit,
                blocked_retries=int((engine_blocked_retries or {}).get(parser_type.name, 0)),
            )
            payload["query"] = engine_query
            payload["timelimit"] = engine_timelimit
            payload["omitted_operators"] = list(
                (engine_omitted_operators or {}).get(parser_type.name, ())
            )
            for rank, source in enumerate(payload["sources"], 1):
                url = str(source.get("url") or "")
                if not url:
                    continue
                if url in seen:
                    # Duplicate across engines is not noise — it is a consensus vote
                    # the triage layer uses to upgrade the already-emitted source.
                    await queue.put(
                        {
                            "type": "vote",
                            "engine": payload["engine"],
                            "provider_family": payload["provider_family"],
                            "rank": rank,
                            "url": {"url": url, "host": _host_of(url)},
                        }
                    )
                    continue
                seen.add(url)
                await queue.put(
                    {
                        "type": "source",
                        "engine": payload["engine"],
                        "provider_family": payload["provider_family"],
                        "rank": rank,
                        "url": {"url": url, "host": _host_of(url)},
                        "serp": {
                            "title": source.get("title", ""),
                            "snippet": source.get("snippet", ""),
                            "fetch_ms": payload["fetch_ms"],
                            "parse_ms": payload["parse_ms"],
                        },
                    }
                )
            await queue.put({"type": "engine", "engine": payload["engine"], "payload": payload})

        # Drive all engines concurrently, then signal completion with a sentinel.
        async def produce() -> None:
            try:
                async with asyncio.TaskGroup() as group:
                    for parser_type in self.engines:
                        group.create_task(run(parser_type), name=f"serp:{parser_type.name}")
            finally:
                await queue.put(None)

        producer = asyncio.create_task(produce())
        try:
            async with asyncio.timeout(deadline):
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    yield item
        except TimeoutError:
            # Budget exhausted — stop pulling; whatever was buffered is the result.
            pass
        finally:
            producer.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await producer

    # Run all engines and return the combined result dict by draining the stream.
    async def search(
        self,
        query: str,
        *,
        region: str = "us-en",
        safesearch: str = "moderate",
        timelimit: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        engine_results: dict[str, dict[str, Any]] = {}
        async for item in self.search_stream(
            query,
            region=region,
            safesearch=safesearch,
            timelimit=timelimit,
        ):
            if item["type"] == "engine":
                payload = item["payload"]
                engine_results[payload["engine"]] = payload

        return {
            "query": " ".join(str(query or "").split()).strip(),
            "region": region,
            "source_limit_per_engine": self.source_limit,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "engines": engine_results,
        }


# Module-level transport singleton — kept alive between calls so TCP/TLS connections
# are reused within the keepalive window instead of renegotiated on every search.
_shared_transport: AdaptiveTransport | None = None


def _get_transport(timeout_seconds: float) -> AdaptiveTransport:
    global _shared_transport
    if _shared_transport is None:
        _shared_transport = AdaptiveTransport(timeout_seconds=timeout_seconds)
    return _shared_transport


# Convenience wrapper that reuses the shared transport for a single search.
async def run_serp_search(
    query: str,
    *,
    region: str = "us-en",
    safesearch: str = "moderate",
    timelimit: str | None = None,
    timeout_seconds: float = 8.0,
    source_limit: int = 3,
) -> dict[str, Any]:
    transport = _get_transport(timeout_seconds)
    api = SerpApi(transport=transport, timeout_seconds=timeout_seconds, source_limit=source_limit)
    return await api.search(query, region=region, safesearch=safesearch, timelimit=timelimit)
