# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from core.config import load_search_config

from .models import (
    STATUS_ERROR,
    STATUS_UNAVAILABLE,
    BrowserFetch,
)

logger = logging.getLogger("core.fetch.browser.client")

# Health probes are cached briefly so per-fetch availability checks stay free.
_HEALTH_TTL = 5.0
# How long to wait for an autostarted daemon to answer /health, and the minimum gap
# between spawn attempts so a daemon that fails to come up is not respawned in a loop.
_SPAWN_WAIT = 25.0
_SPAWN_THROTTLE = 30.0


# Routes page fetches to the persistent warm cloakbrowser daemon (chromium) over HTTP, and
# autostarts that daemon on the first call. The search pipeline depends only on this module's
# browser_fetch, browser_available and shutdown_browser, never on cloakbrowser or the daemon
# directly: callers hand over a URL and get a BrowserFetch back.
class BrowserClient:
    # Resolve the browser config (the search config by default) and set up the daemon state.
    def __init__(self, cfg=None) -> None:
        self._cfg = cfg or load_search_config().browser

        # Shared httpx.AsyncClient, built lazily on the first daemon call.
        self._http: Any | None = None

        # Cached result of the last /health probe and when it was taken.
        self._health_ok = False
        self._health_checked_at = 0.0

        # Autostart bookkeeping: the lock serializes spawn attempts, the timestamp throttles
        # them, and _spawned records whether this client launched the daemon itself.
        self._spawn_lock = asyncio.Lock()
        self._spawn_attempted_at = 0.0
        self._spawned = False

    # The browser layer is usable when not disabled and the warm daemon is reachable.
    async def available(self) -> bool:
        if self._cfg.browser_fallback == "off":
            return False
        return await self._daemon_ready()

    # Cached readiness: a raw /health probe, plus a lazy autostart on the first miss.
    async def _daemon_ready(self) -> bool:
        now = time.monotonic()
        if self._health_checked_at and now - self._health_checked_at < _HEALTH_TTL:
            return self._health_ok
        self._health_checked_at = now
        ok = await self._probe()
        if not ok and self._cfg.autostart_daemon:
            ok = await self._autostart()
        self._health_ok = ok
        return ok

    # A single uncached GET /health; True when the daemon answers 200.
    async def _probe(self) -> bool:
        try:
            resp = await self._client().get(f"{self._cfg.daemon_url}/health")
            return resp.status_code == 200
        except Exception as exc:  # noqa: BLE001
            logger.debug("browser daemon health probe failed: %s", exc)
            return False

    # Spawn the daemon (once, throttled) and poll until it answers or times out.
    async def _autostart(self) -> bool:
        async with self._spawn_lock:
            # Another caller may have won the race and already started the daemon.
            if await self._probe():
                return True

            # Throttle spawn attempts. Zero means no previous attempt, including just after
            # an OS/WSL startup.
            now = time.monotonic()
            if self._spawn_attempted_at and now - self._spawn_attempted_at < _SPAWN_THROTTLE:
                return False
            self._spawn_attempted_at = now
            self._spawn_process()

            # Poll /health until the daemon answers or the wait runs out.
            deadline = time.monotonic() + _SPAWN_WAIT
            while time.monotonic() < deadline:
                await asyncio.sleep(0.5)
                if await self._probe():
                    logger.info("warm browser daemon autostarted on first tool call")
                    return True
            logger.warning("warm browser daemon did not come up within %.0fs", _SPAWN_WAIT)
            return False

    # Launch the daemon as a windowless background process (it reads its own config:
    # idle-shutdown etc.). On Windows that means pythonw.exe (no console subsystem) plus
    # CREATE_NO_WINDOW, so the long-lived daemon never pops a console window; its own
    # process group keeps a parent-console Ctrl-C from reaching it.
    def _spawn_process(self) -> None:
        port = urlparse(self._cfg.daemon_url).port or 20010
        root = Path(__file__).resolve().parents[3]
        executable = sys.executable
        kwargs: dict[str, Any] = {
            "cwd": str(root), "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL,
        }

        # Detach the daemon from this process: no console window on Windows, a new session elsewhere.
        if sys.platform == "win32":
            pythonw = Path(executable).with_name("pythonw.exe")
            if pythonw.exists():
                executable = str(pythonw)
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW
            )
        else:
            kwargs["start_new_session"] = True

        # A failed spawn only logs; the caller notices through the /health polling that follows.
        try:
            subprocess.Popen(
                [executable, "-m", "core.fetch.browser.daemon", "--port", str(port)], **kwargs
            )
            self._spawned = True
            logger.info("spawned warm browser daemon on port %d", port)
        except Exception as exc:  # noqa: BLE001
            logger.warning("failed to spawn warm browser daemon: %s", exc)

    # Build (once) the shared httpx client for daemon calls.
    def _client(self):
        if self._http is None:
            import httpx

            # The daemon is an ASLM-owned loopback endpoint and must never use
            # a desktop/system proxy, even when HTTPX cannot read its bypass list.
            self._http = httpx.AsyncClient(
                timeout=self._cfg.fetch_timeout + 5.0,
                trust_env=False,
            )
        return self._http

    # Fetch one URL through the configured backend; never raises — returns a BrowserFetch.
    async def fetch(
        self,
        url: str,
        *,
        wait_sec: float | None = None,
        nav_timeout: float | None = None,
        family: str = "",
    ) -> BrowserFetch:
        if self._cfg.browser_fallback == "off":
            return BrowserFetch(url=url, status=STATUS_UNAVAILABLE, error="browser disabled")
        return await self._fetch_warm(url, wait_sec=wait_sec, nav_timeout=nav_timeout, family=family)

    # Warm path: POST /fetch to the daemon and adapt the JSON body to a BrowserFetch.
    # Ensures the daemon is up first (lazy autostart on the first tool call).
    async def _fetch_warm(
        self, url: str, *, wait_sec: float | None, nav_timeout: float | None, family: str
    ) -> BrowserFetch:
        if not await self._daemon_ready():
            return BrowserFetch(url=url, status=STATUS_UNAVAILABLE, backend="warm",
                                error="daemon unreachable")

        # Build the request body; the optional wait, timeout and family are sent only when set.
        payload: dict[str, Any] = {"url": url, "engine": self._cfg.engine}
        if wait_sec is not None:
            payload["wait_ms"] = int(max(0.0, wait_sec) * 1000)
        if nav_timeout is not None:
            payload["timeout_ms"] = int(max(0.1, nav_timeout) * 1000)
        if family:
            payload["family"] = family

        # A transport failure marks the cached health as down so the next call re-probes.
        try:
            client = self._client()
            resp = await client.post(f"{self._cfg.daemon_url}/fetch", json=payload)
        except Exception as exc:  # noqa: BLE001
            logger.debug("browser daemon fetch failed for %s: %s", url, exc)
            self._health_ok = False
            return BrowserFetch(url=url, status=STATUS_UNAVAILABLE, backend="warm", error=str(exc))

        # The daemon answers with JSON even for failed scrapes; anything else is a bad response.
        try:
            body = resp.json()
        except Exception:  # noqa: BLE001
            return BrowserFetch(url=url, status=STATUS_ERROR, backend="warm",
                                error=f"bad daemon response (HTTP {resp.status_code})")
        return BrowserFetch.from_daemon(body, backend="warm")

    # Close the shared HTTP client only. The daemon is deliberately LEFT RUNNING so it stays
    # warm across tool calls — a per-call worker that killed it on exit would force a cold
    # Chromium relaunch every single call, defeating the whole point of a warm daemon. The
    # daemon bounds its own life via daemon_idle_shutdown_sec (self-terminates once idle), so
    # an autostarted one lingers at most that long after the last fetch instead of forever.
    async def aclose(self) -> None:
        if self._http is not None:
            try:
                await self._http.aclose()
            except Exception:  # noqa: BLE001
                # Closing is best-effort during shutdown; the reference is dropped regardless.
                pass
            self._http = None


# Process-wide BrowserClient singleton, created on first use by get_browser_client.
_client: Optional[BrowserClient] = None


# Return the lazily-initialised process-wide BrowserClient singleton.
def get_browser_client() -> BrowserClient:
    global _client
    if _client is None:
        _client = BrowserClient()
    return _client


# True when the configured browser backend may be used right now.
async def browser_available() -> bool:
    return await get_browser_client().available()


# Fetch one URL through the configured browser backend (never raises).
async def browser_fetch(
    url: str, *, wait_sec: float | None = None, nav_timeout: float | None = None, family: str = ""
) -> BrowserFetch:
    return await get_browser_client().fetch(
        url, wait_sec=wait_sec, nav_timeout=nav_timeout, family=family
    )


# Release the client's HTTP resources (wire into the MCP server shutdown hook).
async def shutdown_browser() -> None:
    if _client is not None:
        await _client.aclose()
