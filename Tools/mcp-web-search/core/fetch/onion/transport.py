# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from core.fetch.thread_pool import io_pool as _io_pool

from .tor_proxy import resolve_socks

logger = logging.getLogger("core.fetch.onion.transport")


# Envelope for one fetch over Tor. The status is one of ok, unavailable (no tor SOCKS) or error;
# http_status is the HTTP code of the response, or 0 when no response arrived.
@dataclass(slots=True)
class OnionFetch:
    url: str
    status: str = "error"
    ok: bool = False
    http_status: int = 0
    text: str = ""
    error: str = ""
    ms: float = 0.0


# Return True when the onion layer can serve a fetch right now (tor enabled and a SOCKS resolved).
def onion_available() -> bool:
    return resolve_socks() is not None


# Do a blocking curl_cffi GET through the tor SOCKS and return (status code, body text);
# it runs in the io_pool.
def _get(url: str, socks_url: str, timeout: float, impersonate: str) -> tuple[int, str]:
    from curl_cffi import requests as _r

    resp = _r.get(url, impersonate=impersonate, timeout=max(5.0, timeout),
                  proxies={"http": socks_url, "https": socks_url})
    return resp.status_code, (resp.text or "")


# Fetch one URL over Tor via curl_cffi + socks5h. Never raises — returns an OnionFetch envelope.
# curl_cffi (not raw httpx) is used so the request carries a real browser TLS fingerprint, which
# is needed against fingerprint-gating servers and harmless for plain onion services. Every
# request goes through the SOCKS url resolved by tor_proxy; when none is available the fetch is
# a no-op returning status="unavailable".
async def onion_fetch(url: str, *, timeout: float | None = None,
                      impersonate: str = "chrome124") -> OnionFetch:
    import time

    socks_url = resolve_socks()
    if socks_url is None:
        return OnionFetch(url=url, status="unavailable", error="tor unavailable/disabled")

    # Default the timeout from the tor config section.
    if timeout is None:
        from core.config import load_search_config
        timeout = load_search_config().tor.fetch_timeout

    # Run the blocking request in the io_pool; a failure becomes an error envelope.
    t0 = time.perf_counter()
    loop = asyncio.get_running_loop()
    try:
        http_status, text = await loop.run_in_executor(
            _io_pool, lambda: _get(url, socks_url, timeout, impersonate)
        )
    except Exception as exc:  # noqa: BLE001
        return OnionFetch(url=url, status="error", error=f"{type(exc).__name__}: {exc}",
                          ms=round((time.perf_counter() - t0) * 1000, 1))

    # Only 2xx counts as success; any other status is reported as an error with the HTTP code.
    ok = 200 <= http_status < 300
    return OnionFetch(
        url=url, status="ok" if ok else "error", ok=ok, http_status=http_status,
        text=text, error="" if ok else f"HTTP {http_status}",
        ms=round((time.perf_counter() - t0) * 1000, 1),
    )
