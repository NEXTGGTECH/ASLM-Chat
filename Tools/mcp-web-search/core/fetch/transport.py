# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import aiohttp
import primp

from ..engines.models import EngineRequest
from ._base import TransportResponse
from .httpx_transport import HttpxTransport

logger = logging.getLogger("core.fetch.transport")

# Chromium families whose HTTP engines also read the warm browser daemon's earned cookies
# (stored under the "chromium" identity) — the browser-feeds-HTTP half of Stage B.
_CHROMIUM_FAMILIES = frozenset({"chrome", "edge"})
_WARM_FAMILY = "chromium"


# Imported-cookie jars, one per browser family. Chrome/Edge/Brave land in "chromium"; Firefox
# in "firefox". Replay prefers the engine's own family then falls back across the rest.
_IMPORTED_FAMILIES = ("chromium", "firefox")

# Engines that must NOT receive imported real-browser cookies. Google's parser relies on the
# GSA (Google Search App) User-Agent trick to get the clean, parseable mobile SERP; that identity
# is deliberately cookie-light. Injecting the user's full desktop session (verified live) flips
# Google onto its JS-gated desktop layout (the `enablejs` page) which the GSA parser can't read —
# so the "logged-in" cookies actively hurt this one engine. Other engines (Startpage/DDG/Brave/
# Qwant/Bing/Yandex) and read_page still benefit from the imported session.
_IMPORTED_COOKIE_DENYLIST = frozenset({"google"})


# Map an engine's primp_target to its imported-cookie family (opt-in profile import). Chrome/Edge
# share the chromium jar (Chrome/Edge/Brave imports); Firefox has its own. Unknown → no primary.
def _imported_family(primp_target: str) -> str:
    if primp_target in _CHROMIUM_FAMILIES:
        return "chromium"
    if primp_target == "firefox":
        return "firefox"
    return ""


# Merge an engine's persistent cookies into the outgoing request (read half of Stage B).
# Layered: stored HTTP-earned cookies, plus the warm browser's cookies for chromium engines,
# then the engine's own per-request seed cookies on top (fresh intent wins on a name clash).
def _replay_identity_cookies(request: EngineRequest, host: str) -> EngineRequest:
    owner = request.identity_key
    if not owner:
        return request
    try:
        from core.fetch.browser.identity_store import get_identity_store

        store = get_identity_store()
        stored = store.http_cookies_map(owner, host)
        if request.primp_target in _CHROMIUM_FAMILIES:
            for cookie in store.cookies_for(_WARM_FAMILY, host=host):
                name = cookie.get("name")
                if name:
                    stored[str(name)] = str(cookie.get("value", ""))
        # Imported real-browser cookies (opt-in) are the BASE layer: a session the engine can't
        # earn itself (a logged-in SID) fills in, but any cookie the engine has since earned
        # (consent/region) overrides the imported one. The engine's own family jar is preferred,
        # then OTHER families fill in for the same host: a valid session cookie is not bound to a
        # TLS fingerprint, so a Google SID living in the user's Firefox still helps a Chrome-
        # impersonating Google engine (its chromium jar is empty when Chrome uses App-Bound enc).
        if owner not in _IMPORTED_COOKIE_DENYLIST:
            primary = _imported_family(request.primp_target)
            imported: dict[str, str] = {}
            for family in ([primary] if primary else []) + [f for f in _IMPORTED_FAMILIES if f != primary]:
                for name, value in store.imported_cookies_map(family, host).items():
                    imported.setdefault(name, value)
            if imported:
                stored = {**imported, **stored}
    except Exception as exc:  # noqa: BLE001 — cookie replay must never break a fetch
        logger.debug("identity cookie replay skipped for %s: %s", owner, exc)
        return request
    if not stored:
        return request
    return replace(request, cookies={**stored, **request.cookies})


# Persist any Set-Cookie the response carried back into the engine's cookie history
# (write half of Stage B). Best-effort: a store failure never affects the returned response.
def _capture_identity_cookies(request: EngineRequest, host: str, response: TransportResponse) -> None:
    owner = request.identity_key
    if not owner or not response.set_cookie:
        return
    try:
        from core.fetch.browser.identity_store import get_identity_store

        get_identity_store().merge_set_cookie(owner, host, response.set_cookie)
    except Exception as exc:  # noqa: BLE001 — capture is opportunistic
        logger.debug("identity cookie capture skipped for %s: %s", owner, exc)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# One pooled aiohttp session shared by all engine requests.
class AiohttpTransport:

    # Initialize timeout and connection pool settings without opening a session yet.
    def __init__(self, *, timeout_seconds: float = 8.0, connection_limit: int = 12) -> None:
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self.connection_limit = max(1, int(connection_limit))
        self._session: aiohttp.ClientSession | None = None

    # Open a new aiohttp session if one is not already active.
    async def start(self) -> None:
        if self._session is not None and not self._session.closed:
            return
        connector = aiohttp.TCPConnector(
            limit=self.connection_limit,
            limit_per_host=4,
            ttl_dns_cache=300,
            enable_cleanup_closed=True,
            keepalive_timeout=30.0,
        )
        timeout = aiohttp.ClientTimeout(
            total=self.timeout_seconds,
            connect=min(3.0, self.timeout_seconds),
            sock_connect=min(3.0, self.timeout_seconds),
            sock_read=self.timeout_seconds,
        )
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers=DEFAULT_HEADERS,
            auto_decompress=True,
            raise_for_status=False,
        )

    # Close and discard the active aiohttp session.
    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    # Send one HTTP request and return the raw response.
    async def fetch(self, request: EngineRequest) -> TransportResponse:
        await self.start()
        assert self._session is not None
        headers = dict(request.headers)
        if request.cookies:
            headers["Cookie"] = "; ".join(f"{name}={value}" for name, value in request.cookies.items())
        async with self._session.request(
            request.method,
            request.url,
            params=request.params or None,
            data=request.data or None,
            headers=headers or None,
            allow_redirects=True,
        ) as response:
            set_cookie = list(response.headers.getall("Set-Cookie", []))
            return TransportResponse(
                status=response.status, body=await response.read(),
                transport="aiohttp", set_cookie=set_cookie,
            )


# Best-effort Set-Cookie extraction from a primp response. primp exposes parsed cookies as
# a {name: value} dict (attributes already consumed), so we synthesise host-scoped
# "name=value" lines; the identity store defaults their domain to the request host.
def _primp_set_cookie(response: object) -> list[str]:
    cookies = getattr(response, "cookies", None)
    if isinstance(cookies, dict) and cookies:
        return [f"{name}={value}" for name, value in cookies.items() if name]
    return []


# Bounded browser-impersonating transport for engines that reject generic TLS clients.
class PrimpTransport:

    # Initialize timeout and a thread-pool executor for blocking primp calls.
    def __init__(self, *, timeout_seconds: float = 8.0, max_workers: int = 4) -> None:
        self.timeout_seconds = max(0.1, float(timeout_seconds))
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="serp-primp",
        )
        self._clients: dict[str, primp.Client] = {}
        # Populate the imported-browser cookie layer once per process (config-gated; a no-op
        # unless profile_import.enabled). Best-effort — never let it break transport setup.
        try:
            from core.fetch.browser.profile_import import ensure_imported

            ensure_imported()
        except Exception as exc:  # noqa: BLE001
            logger.debug("profile_import trigger skipped: %s", exc)

    # Return or create a primp client keyed by host+impersonation identity.
    def _client(self, host: str, primp_target: str, primp_os: str) -> primp.Client:
        key = f"{host}:{primp_target}:{primp_os}"
        client = self._clients.get(key)
        if client is None:
            client = primp.Client(
                timeout=self.timeout_seconds,
                impersonate=primp_target,
                impersonate_os=primp_os,
                verify=True,
            )
            self._clients[key] = client
        return client

    # Execute one HTTP request synchronously using the matching primp client.
    def _fetch_sync(self, request: EngineRequest) -> TransportResponse:
        host = request.url.split("/", 3)[2]
        headers = dict(request.headers)
        if request.cookies:
            headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in request.cookies.items())
        response = self._client(host, request.primp_target, request.primp_os).request(
            request.method,
            request.url,
            params=request.params or None,
            data=request.data or None,
            headers=headers or None,
        )
        return TransportResponse(
            status=response.status_code, body=response.content,
            transport="primp", set_cookie=_primp_set_cookie(response),
        )

    # Run the blocking primp fetch on the thread-pool executor.
    async def fetch(self, request: EngineRequest) -> TransportResponse:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._fetch_sync, request)

    # Release all cached clients and shut down the executor.
    async def close(self) -> None:
        self._clients.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)


# Route each engine to one known transport without retry chains.
class AdaptiveTransport:

    # Initialize all transports: specialized httpx for Google, primp for hosts that
    # reject generic TLS (Brave, Yandex, Startpage SERPs and the Qwant API),
    # fast aiohttp for DuckDuckGo.
    def __init__(self, *, timeout_seconds: float = 8.0) -> None:
        self._fast = AiohttpTransport(timeout_seconds=timeout_seconds)
        # One impersonated worker per impersonated host so all engines can run
        # concurrently without starving each other on the thread pool.
        self._impersonated = PrimpTransport(
            timeout_seconds=timeout_seconds,
            max_workers=len(self._IMPERSONATED_HOSTS) + 1,
        )
        self._httpx = HttpxTransport(timeout_seconds=timeout_seconds)

    # Hosts whose anti-bot TLS fingerprinting requires browser impersonation.
    _IMPERSONATED_HOSTS = frozenset(
        {
            "www.bing.com",
            "search.brave.com",
            "yandex.com",
            "www.startpage.com",
            "api.qwant.com",
            "ya.ru",
        }
    )

    # Forward the request to the appropriate transport based on the target host.
    # Around the dispatch, the persistent identity cookies (Stage B) are replayed into the
    # request and any Set-Cookie the response carried is written back, so an engine builds
    # one coherent cookie history (consent/region/session) across searches and restarts.
    async def fetch(self, request: EngineRequest) -> TransportResponse:
        host = request.url.split("/", 3)[2]
        request = _replay_identity_cookies(request, host)
        if host == "www.google.com":
            response = await self._httpx.fetch(request)
        elif host in self._IMPERSONATED_HOSTS:
            response = await self._impersonated.fetch(request)
        else:
            response = await self._fast.fetch(request)
        _capture_identity_cookies(request, host, response)
        return response

    # Close all underlying transports.
    async def close(self) -> None:
        await self._fast.close()
        await self._impersonated.close()
        await self._httpx.close()
