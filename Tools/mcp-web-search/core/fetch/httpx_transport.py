# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import random
import ssl

import httpx

from ..engines.models import EngineRequest
from ._base import TransportResponse

# Old Android device strings paired with a legacy Chrome 39 engine. The trick: an
# ancient browser cannot run Google's modern result-rendering JavaScript, so Google
# falls back to serving a plain server-rendered HTML SERP instead of the JS shell it
# gives current browsers. The NSTNWV token is part of Google's own Search App
# identifier and marks a native WebView context, which is treated more leniently.
# Build numbers are randomised per request so no two look identical.
_GSA_DEVICES: tuple[str, ...] = (
    "Linux; Android 5.0; SM-G900P Build/LRX21T",
    "Linux; Android 5.1.1; SM-G920F Build/LMY47X",
    "Linux; Android 6.0.1; Nexus 5X Build/MMB29P",
    "Linux; Android 5.0.2; HTC One Build/LRX22G",
    "Linux; Android 5.1; LG-D855 Build/LMY47D",
)


# Build a randomised GSA-style User-Agent with a legacy Chrome 39 engine.
def _gsa_user_agent() -> str:
    device = random.choice(_GSA_DEVICES)
    build = f"39.0.{random.randint(1000, 3600)}.{random.randint(1000, 1999)}"
    return (
        f"Mozilla/5.0 ({device}) AppleWebKit/537.36 (KHTML, like Gecko) "
        f"Chrome/{build} Mobile Safari/537.36 NSTNWV"
    )


# Headers that only make sense on desktop and should not appear in mobile requests.
_DESKTOP_ONLY_HEADERS = frozenset(
    {
        "Sec-CH-UA",
        "Sec-CH-UA-Mobile",
        "Sec-CH-UA-Platform",
        "Upgrade-Insecure-Requests",
        "Sec-Fetch-Dest",
        "Sec-Fetch-Mode",
        "Sec-Fetch-Site",
        "Sec-Fetch-User",
    }
)


# Build an SSL context whose cipher list is randomised per-instance.
# The first two ciphers (mandatory TLS 1.3 suites) stay in place;
# everything else is shuffled so the TLS ClientHello fingerprint changes
# with every new connection, making per-fingerprint blocking ineffective.
def _make_ssl_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    try:
        ciphers = [c["name"] for c in ctx.get_ciphers()]
        head, tail = ciphers[:2], ciphers[2:]
        random.shuffle(tail)
        ctx.set_ciphers(":".join(head + tail))
    except ssl.SSLError:
        # A build that rejects the shuffled cipher list keeps the default context.
        pass
    return ctx


# Async httpx transport with per-request TLS fingerprint randomisation.
# A fresh httpx.AsyncClient is created for every Google request so each connection
# gets a different cipher order in its TLS ClientHello. This also avoids the
# ConnectError that occurs when Google closes a keepalive connection and httpx
# tries to reuse it for the next request.
class HttpxTransport:
    # Store the request timeout in seconds, floored at 0.1.
    def __init__(self, *, timeout_seconds: float = 8.0) -> None:
        self.timeout_seconds = max(0.1, float(timeout_seconds))

    # Send the request as a legacy GSA mobile client: random old UA, plain Accept,
    # no desktop-only client hints. A new SSL context (new cipher order) is created
    # per call, so every TLS handshake looks different to fingerprint-based systems.
    async def fetch(self, request: EngineRequest) -> TransportResponse:
        # Adapt the engine headers to a legacy mobile client.
        headers = {k: v for k, v in request.headers.items() if k not in _DESKTOP_ONLY_HEADERS}
        headers["User-Agent"] = _gsa_user_agent()
        headers["Accept"] = "*/*"
        headers["Accept-Encoding"] = "gzip, deflate"

        # One short-lived client per request, so each connection has its own TLS fingerprint.
        async with httpx.AsyncClient(
            verify=_make_ssl_context(),
            timeout=httpx.Timeout(self.timeout_seconds, connect=min(4.0, self.timeout_seconds)),
            follow_redirects=True,
        ) as client:
            response = await client.request(
                request.method,
                request.url,
                params=request.params or None,
                content=request.data and "&".join(f"{k}={v}" for k, v in request.data.items()) or None,
                headers=headers,
                cookies=request.cookies or None,
            )

        # Surface the raw Set-Cookie lines so the identity store can accumulate them.
        return TransportResponse(
            status=response.status_code, body=response.content,
            transport="httpx", set_cookie=list(response.headers.get_list("set-cookie")),
        )

    # Do nothing: every request opens and closes its own client, so there is nothing to release.
    async def close(self) -> None:
        pass
