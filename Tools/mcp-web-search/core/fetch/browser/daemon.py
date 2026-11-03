# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional
from urllib.parse import unquote, urlparse

# Make the web search root importable when the daemon runs as a standalone module.
_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Force UTF-8 on stdout so non-ASCII page text in console output cannot raise encoding errors.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from aiohttp import web

from core.fetch.antibot import is_antibot
from core.fetch.browser.identity_store import IdentityStore, get_identity_store
from core.fetch.browser.tempjanitor import sweep_stale_browser_temp
from core.logging_setup import setup_logging

logger = logging.getLogger("core.fetch.browser.daemon")

# Minimum visible text the page wait expects, and minimum HTML size of a usable scrape.
_MIN_TEXT_CHARS = 200
_MIN_HTML_CHARS = 200

# Hard ceilings for the scrape steps that carry NO native timeout. page.goto and
# wait_for_function are already bounded by their own timeouts; content()/evaluate()/
# title() and page.close() are not, and a pathological page (infinite script, beforeunload
# dialog, wedged renderer) can otherwise hang them indefinitely while holding the fetch
# lock — stalling the whole daemon. These caps guarantee the lock is released and the
# browser recycled instead of sleeping forever on a dead crawl. The extract cap covers
# content + evaluate + title combined, the close cap covers closing one page, and the
# teardown cap covers closing the context/browser during recycle/shutdown.
_EXTRACT_TIMEOUT = 15.0
_CLOSE_TIMEOUT = 5.0
_TEARDOWN_TIMEOUT = 10.0

# Consecutive blocked fetches that flip a recycle from "restore" to "rotate identity".
_BURN_STREAK = 3


# Fetch results

# Why the warm browser was torn down — drives whether identity is restored or rotated.
class RecycleReason:
    REQUESTS = "requests"
    AGE = "age"
    RSS = "rss"
    BURN = "burn"


# Outcome of a single page fetch through the warm browser (mirrors client BrowserFetch).
@dataclass
class ScrapeResult:
    url: str
    # One of: ok | blocked | timeout | error.
    status: str = "error"
    ok: bool = False
    title: str = ""
    html: str = ""
    text: str = ""
    engine: str = "chromium"
    ms: float = 0.0
    error: str = ""


# Process helpers

# Validate a proxy URL for cloakbrowser: return it unchanged when it has a scheme and a host,
# otherwise None.
def _parse_proxy(proxy_url: str | None) -> Optional[str]:
    if not proxy_url:
        return None
    parsed = urlparse(proxy_url)
    if not parsed.scheme or not parsed.hostname:
        return None
    return proxy_url


# Return the resident memory of this process plus its children (the chromium tree), in MB.
# Returns 0.0 when psutil is missing or the measurement fails, which disables the RSS recycle.
def _process_tree_rss_mb() -> float:
    try:
        import psutil
    except Exception:  # noqa: BLE001 — psutil optional; without it RSS recycle is skipped
        return 0.0
    try:
        proc = psutil.Process()
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except Exception:  # noqa: BLE001
                # A child can exit between listing and measuring; skip it.
                continue
        return total / (1024 * 1024)
    except Exception as exc:  # noqa: BLE001
        logger.debug("rss measurement failed: %s", exc)
        return 0.0


# Warm browser

# One warm Chromium with its identity context, recycle policy and checkpoint discipline.
# Supervision:
# - one identity context seeded from the IdentityStore on launch (earned cookies /
#   storageState carry across restarts);
# - idle checkpoints: every checkpoint_interval, if no fetch is in flight and the state is
#   dirty, storageState is exported back to the store;
# - recycle on request count, age or RSS of the browser process tree, each preceded by a
#   forced checkpoint; a burn (blocked streak) rotates identity instead of restoring it.
class WarmChromium:
    name = "chromium"

    # Store the launch, recycle and checkpoint settings; the browser itself starts lazily.
    def __init__(
        self,
        *,
        identity: IdentityStore,
        family: str = "chromium",
        headless: bool = True,
        humanize: bool = False,
        locale: str = "en-US",
        proxy: str | None = None,
        max_requests: int = 40,
        max_age_sec: float = 900.0,
        max_rss_mb: int = 2048,
        checkpoint_interval: float = 30.0,
        nav_timeout: float = 30.0,
        wait: float = 3.0,
    ) -> None:
        # Launch and recycle settings, clamped to sane minimums.
        self._identity = identity
        self._family = family
        self.headless = headless
        self.humanize = humanize
        self.locale = locale
        self.proxy = proxy
        self.max_requests = max(1, int(max_requests))
        self.max_age = max(1.0, float(max_age_sec))
        self.max_rss_mb = max(256, int(max_rss_mb))
        self.checkpoint_interval = max(5.0, float(checkpoint_interval))
        self.nav_timeout = max(1.0, float(nav_timeout))
        self.wait = max(0.0, float(wait))

        # Runtime state of the current browser generation. The lock serializes fetches with
        # recycles and checkpoints; _dirty marks state changed since the last checkpoint and
        # _wedged marks a hung page/close that forces a clean respawn.
        self._lock = asyncio.Lock()
        self._browser: Any | None = None
        self._context: Any | None = None
        self._teardown: Callable[[], Awaitable[None]] | None = None
        self._started_at = 0.0
        self._last_used = 0.0
        self._requests = 0
        self._total = 0
        self._recycles = 0
        self._blocked_streak = 0
        self._dirty = False
        self._inflight = 0
        self._wedged = False
        self._checkpoint_task: asyncio.Task | None = None

    # Lifecycle

    # Launch chromium and open a context seeded with the family's latest good identity.
    async def _open(self) -> None:
        import cloakbrowser

        browser = await cloakbrowser.launch_async(
            headless=self.headless,
            stealth_args=True,
            humanize=self.humanize,
            locale=self.locale,
            proxy=self.proxy,
        )
        seed = self._identity.latest_good(self._family)
        try:
            context = await browser.new_context(storage_state=seed) if seed else await browser.new_context()
        except Exception:  # noqa: BLE001 — fall back to the default context if seeding fails
            context = await browser.new_context()

        # Close the context and then the browser of this generation.
        async def _close() -> None:
            # Bound both closes: a wedged browser must not make teardown hang forever (that
            # would defeat the recycle path it is supposed to enable).
            try:
                await asyncio.wait_for(context.close(), timeout=_TEARDOWN_TIMEOUT)
            except Exception:  # noqa: BLE001
                # Teardown is best-effort; the browser close below still runs.
                pass
            try:
                await asyncio.wait_for(browser.close(), timeout=_TEARDOWN_TIMEOUT)
            except Exception:  # noqa: BLE001
                # A browser that cannot close in time is abandoned; the next launch starts fresh.
                pass

        self._browser, self._context, self._teardown = browser, context, _close
        self._started_at = time.monotonic()
        self._requests = 0
        self._dirty = False

    # Launch the browser if it is not running.
    async def _ensure(self) -> None:
        if self._browser is None:
            await self._open()

    # Export storageState into the store (best-effort; needs a live, quiet browser).
    async def _checkpoint(self, *, good: bool = True) -> None:
        if self._context is None:
            return
        try:
            state = await asyncio.wait_for(self._context.storage_state(), timeout=10.0)
        except Exception as exc:  # noqa: BLE001
            logger.debug("checkpoint export failed: %s", exc)
            return
        self._identity.checkpoint(self._family, state, good=good)
        self._dirty = False

    # Tear the browser down (optionally checkpointing first).
    async def _shutdown_browser(self, *, checkpoint: bool, good: bool = True) -> None:
        if checkpoint:
            await self._checkpoint(good=good)
        if self._teardown is not None:
            try:
                await self._teardown()
            except Exception:  # noqa: BLE001
                # A failed teardown still drops the references so the next fetch relaunches.
                pass
        self._browser = self._context = self._teardown = None

    # Decide whether to recycle, returning the reason or None. Pure given current counters.
    def _recycle_reason(self) -> str | None:
        if self._browser is None:
            return None
        if self._blocked_streak >= _BURN_STREAK:
            return RecycleReason.BURN
        if self._requests >= self.max_requests:
            return RecycleReason.REQUESTS
        if (time.monotonic() - self._started_at) >= self.max_age:
            return RecycleReason.AGE
        if self.max_rss_mb and _process_tree_rss_mb() >= self.max_rss_mb:
            return RecycleReason.RSS
        return None

    # Recycle the browser: checkpoint (or rotate on burn), respawn, reseed.
    async def _recycle(self, reason: str) -> None:
        logger.info("recycling warm chromium reason=%s requests=%d", reason, self._requests)
        if reason == RecycleReason.BURN:
            # Poisoned identity: mark it burned and drop to an older good generation / seed.
            await self._shutdown_browser(checkpoint=True, good=False)
            self._identity.rotate(self._family)
        else:
            await self._shutdown_browser(checkpoint=True, good=True)
        self._recycles += 1
        self._blocked_streak = 0
        await self._open()

    # Fetching

    # Open a page, navigate, wait for content to settle, and scrape it.
    async def _scrape(self, url: str, *, wait: float, nav_timeout: float) -> ScrapeResult:
        page = await self._context.new_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=int(nav_timeout * 1000))
            if wait > 0:
                try:
                    await page.wait_for_function(
                        f"document.body && document.body.innerText.trim().length > {_MIN_TEXT_CHARS}",
                        timeout=int(wait * 1000),
                    )
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(min(wait, 1.5))

            # Read the page HTML, visible text and title. These calls have no native timeout,
            # so the caller caps them to keep a wedged renderer from holding the fetch lock.
            async def _extract() -> tuple[str, str, str]:
                html = await page.content()
                text = await page.evaluate("document.body ? document.body.innerText : ''") or ""
                title = await page.title()
                return html, text, title

            html, text, title = await asyncio.wait_for(_extract(), timeout=_EXTRACT_TIMEOUT)
        finally:
            # A wedged page (beforeunload/dialog/dead renderer) can hang close() forever;
            # bound it and, on failure, abandon the page and flag a respawn so the daemon
            # recovers instead of deadlocking on a dead crawl.
            try:
                await asyncio.wait_for(page.close(), timeout=_CLOSE_TIMEOUT)
            except Exception:  # noqa: BLE001
                self._wedged = True

        # Classify the scrape: too little HTML, an anti-bot page, or a usable result.
        if len(html) < _MIN_HTML_CHARS:
            return ScrapeResult(url=url, status="error", error="insufficient HTML")
        if is_antibot(html):
            return ScrapeResult(url=url, status="blocked", title=title, html=html, text=text)
        return ScrapeResult(url=url, status="ok", ok=True, title=title, html=html, text=text)

    # Fetch one URL through the warm browser, recycling first if a threshold was crossed.
    async def fetch(self, url: str, *, wait: float | None = None,
                    nav_timeout: float | None = None) -> ScrapeResult:
        t0 = time.perf_counter()
        async with self._lock:
            # Recycle before the fetch when a threshold was crossed, then make sure a browser runs.
            reason = self._recycle_reason()
            if reason is not None:
                await self._recycle(reason)
            await self._ensure()
            self._inflight += 1

            # Scrape the page; a timeout or crash becomes an error result instead of an exception.
            try:
                result = await self._scrape(
                    url,
                    wait=self.wait if wait is None else wait,
                    nav_timeout=self.nav_timeout if nav_timeout is None else nav_timeout,
                )
            except (TimeoutError, asyncio.TimeoutError):
                result = ScrapeResult(url=url, status="timeout", error="navigation timeout")
            except Exception as exc:  # noqa: BLE001 — a crash usually kills the browser
                await self._shutdown_browser(checkpoint=False)
                result = ScrapeResult(url=url, status="error", error=f"{type(exc).__name__}: {exc}")
            finally:
                self._inflight -= 1

                # A hung page/close left the browser in an unknown state — tear it down now
                # (bounded) so the next fetch respawns clean instead of inheriting the wedge.
                # A wedge is a page fault, NOT a poisoned identity: the cookies/storage are
                # still valid, so checkpoint them best-effort (good=True) before teardown.
                # _checkpoint bounds storage_state at 10s, so a fully-dead browser just times
                # out and proceeds — we never trade the lock for an unbounded save, but we
                # also never throw away a good identity we could still have grabbed.
                if self._wedged:
                    self._wedged = False
                    await self._shutdown_browser(checkpoint=True, good=True)

            # Update the recycle counters; only consecutive blocked results build a burn streak.
            self._requests += 1
            self._total += 1
            self._last_used = time.monotonic()
            self._dirty = True
            self._blocked_streak = self._blocked_streak + 1 if result.status == "blocked" else 0
        result.ms = round((time.perf_counter() - t0) * 1000, 1)
        return result

    # Supervision

    # Eagerly warm the browser and start the idle-checkpoint loop.
    async def start(self) -> None:
        async with self._lock:
            await self._ensure()
        self._checkpoint_task = asyncio.create_task(self._idle_checkpoint_loop(), name="bg:checkpoint")

    # Periodically checkpoint identity while the browser is idle and state is dirty.
    async def _idle_checkpoint_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self.checkpoint_interval)
                if self._inflight == 0 and self._dirty and self._browser is not None:
                    async with self._lock:
                        if self._inflight == 0 and self._dirty:
                            await self._checkpoint(good=True)
        except asyncio.CancelledError:
            # stop() cancels the loop; cancellation is the normal way out.
            pass

    # Stop the checkpoint loop, then take a final checkpoint and tear the browser down.
    async def stop(self) -> None:
        if self._checkpoint_task is not None:
            self._checkpoint_task.cancel()
            try:
                await self._checkpoint_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                # The loop is being shut down; its cancellation or last error is irrelevant now.
                pass
        async with self._lock:
            await self._shutdown_browser(checkpoint=True, good=True)

    # Return the runtime snapshot served by /health.
    def health(self) -> dict[str, Any]:
        now = time.monotonic()
        return {
            "engine": self.name,
            "alive": self._browser is not None,
            "requests_since_recycle": self._requests,
            "total_served": self._total,
            "recycles": self._recycles,
            "blocked_streak": self._blocked_streak,
            "rss_mb": round(_process_tree_rss_mb(), 1),
            "age_sec": round(now - self._started_at, 1) if self._browser is not None else None,
            "idle_sec": round(now - self._last_used, 1) if self._last_used else None,
            "max_requests": self.max_requests,
            "max_age_sec": self.max_age,
            "max_rss_mb": self.max_rss_mb,
        }


# HTTP API

# Small HTTP server around one warm Chromium, so callers pay the browser cold-start cost
# once. Chromium-only by design: it is the sole browser backend in web search.
class BrowserDaemon:
    # Create the warm browser from the command line settings; nothing launches until start().
    def __init__(self, args: argparse.Namespace) -> None:
        self.browser = WarmChromium(
            identity=get_identity_store(),
            headless=args.headless,
            humanize=args.humanize,
            locale=args.locale,
            proxy=_parse_proxy(args.proxy),
            max_requests=args.max_requests,
            max_age_sec=args.max_age,
            max_rss_mb=args.max_rss_mb,
            checkpoint_interval=args.checkpoint_interval,
            nav_timeout=args.nav_timeout,
            wait=args.wait,
        )
        self.idle_shutdown_sec = max(0.0, float(getattr(args, "idle_shutdown_sec", 0.0)))
        self._stop_event = asyncio.Event()
        self._started_at = time.monotonic()
        self._idle_task: asyncio.Task | None = None

    # Warm the browser and, when an idle limit is set, start the idle shutdown monitor.
    async def start(self) -> None:
        logger.info("daemon starting chromium ...")
        await self.browser.start()
        logger.info("daemon chromium ready")
        if self.idle_shutdown_sec > 0:
            self._idle_task = asyncio.create_task(self._idle_monitor(), name="bg:idle-shutdown")

    # Self-terminate after idle_shutdown_sec without a served fetch (0 = eternal).
    async def _idle_monitor(self) -> None:
        try:
            while not self._stop_event.is_set():
                await asyncio.sleep(min(30.0, self.idle_shutdown_sec))
                last = self.browser._last_used or self._started_at
                idle = time.monotonic() - last
                if self.browser._inflight == 0 and idle >= self.idle_shutdown_sec:
                    logger.info("idle %.0fs >= %.0fs — shutting down", idle, self.idle_shutdown_sec)
                    self._stop_event.set()
                    return
        except asyncio.CancelledError:
            # stop() cancels the monitor; cancellation is the normal way out.
            pass

    # Stop the idle monitor and shut the warm browser down.
    async def stop(self) -> None:
        if self._idle_task is not None:
            self._idle_task.cancel()
        await self.browser.stop()

    # POST /fetch {url, wait_ms?, timeout_ms?, html?} -> ScrapeResult json.
    async def handle_fetch(self, request: web.Request) -> web.Response:
        try:
            data = await request.json()
        except Exception:  # noqa: BLE001
            return web.json_response({"error": "invalid JSON body"}, status=400)
        url = str(data.get("url") or "").strip()
        if not url:
            return web.json_response({"error": "missing 'url'"}, status=400)
        wait = float(data["wait_ms"]) / 1000 if "wait_ms" in data else None
        nav_timeout = float(data["timeout_ms"]) / 1000 if "timeout_ms" in data else None
        result = await self.browser.fetch(url, wait=wait, nav_timeout=nav_timeout)
        logger.info(
            "fetch status=%s ms=%.0f html=%d recycles=%d req=%d url=%r",
            result.status, result.ms, len(result.html), self.browser._recycles,
            self.browser._requests, url[:200],
        )
        payload = asdict(result)
        if data.get("html") is False:
            payload.pop("html", None)
        return web.json_response(payload, status=200 if result.ok else 502)

    # GET /health -> runtime snapshot of the warm browser.
    async def handle_health(self, _request: web.Request) -> web.Response:
        return web.json_response({"engines": [self.browser.health()]})

    # POST /shutdown -> ask the serve loop to stop.
    async def handle_shutdown(self, _request: web.Request) -> web.Response:
        self._stop_event.set()
        return web.json_response({"stopping": True})

    # Build the aiohttp application with the fetch, health and shutdown routes.
    def make_app(self) -> web.Application:
        app = web.Application(client_max_size=16 * 1024 * 1024)
        app.add_routes([
            web.post("/fetch", self.handle_fetch),
            web.get("/health", self.handle_health),
            web.post("/shutdown", self.handle_shutdown),
        ])
        return app


# Entry point

# Run the daemon: warm the browser, serve the HTTP API until a shutdown request, idle timeout
# or Ctrl+C, then stop cleanly.
async def _serve(args: argparse.Namespace) -> None:
    # The daemon is windowless when autostarted — logs must go to a file.
    setup_logging()

    # Reap browser temp profiles orphaned by earlier killed/crashed daemons before launching
    # our own (off the event loop — rmtree over many dirs can stall).
    try:
        await asyncio.get_running_loop().run_in_executor(None, sweep_stale_browser_temp)
    except Exception as exc:  # noqa: BLE001 — janitor must never block startup
        logger.debug("temp janitor at startup failed: %s", exc)

    # Warm the browser before the port opens, so the first fetch does not pay the launch cost.
    daemon = BrowserDaemon(args)
    idle = f"idle-shutdown={args.idle_shutdown_sec:.0f}s" if args.idle_shutdown_sec > 0 else "eternal"
    logger.info("warm browser daemon: chromium headless=%s proxy=%s %s",
                args.headless, "yes" if args.proxy else "no", idle)
    await daemon.start()

    # Serve until the stop event fires, then stop the browser and the server.
    runner = web.AppRunner(daemon.make_app())
    await runner.setup()
    site = web.TCPSite(runner, args.host, args.port)
    await site.start()
    logger.info("serving on http://%s:%d (POST /shutdown or Ctrl+C to stop)", args.host, args.port)
    try:
        await daemon._stop_event.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        # Ctrl+C and task cancellation are normal stop signals; cleanup runs below.
        pass
    finally:
        logger.info("daemon stopping ...")
        await daemon.stop()
        await runner.cleanup()


# Parse the command line; defaults come from the browser section of the search config.
def _parse_args() -> argparse.Namespace:
    from core.config import load_search_config

    cfg = load_search_config().browser
    p = argparse.ArgumentParser(description="Persistent warm stealth-browser daemon (chromium).")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=urlparse(cfg.daemon_url).port or 20010)
    p.add_argument("--no-headless", action="store_false", dest="headless")
    p.add_argument("--humanize", action="store_true", default=cfg.humanize)
    p.add_argument("--proxy", default=cfg.proxy or None)
    p.add_argument("--locale", default="en-US")
    p.add_argument("--max-requests", type=int, default=cfg.max_requests, dest="max_requests")
    p.add_argument("--max-age", type=float, default=cfg.max_age_sec, dest="max_age")
    p.add_argument("--max-rss-mb", type=int, default=cfg.max_rss_mb, dest="max_rss_mb")
    p.add_argument("--checkpoint-interval", type=float, default=cfg.checkpoint_interval,
                   dest="checkpoint_interval")
    p.add_argument("--nav-timeout", type=float, default=cfg.nav_timeout, dest="nav_timeout")
    p.add_argument("--wait", type=float, default=cfg.wait)
    p.add_argument("--idle-shutdown-sec", type=float, default=cfg.daemon_idle_shutdown_sec,
                   dest="idle_shutdown_sec", help="self-terminate after this idle time; 0 = eternal")
    p.set_defaults(headless=cfg.headless)
    return p.parse_args()


# Run: python -m core.fetch.browser.daemon --port 20010
if __name__ == "__main__":
    from core.runtime import run_fast

    run_fast(_serve(_parse_args()))
