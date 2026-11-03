# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import logging
import time

logger = logging.getLogger("services.web_search")
trace_logger = logging.getLogger("trace.web_search")


# Tracked background warm-up tasks for result URLs (crawl-ahead): after a search, the top result
# URLs the model did NOT already get parsed content for are warmed into the page cache, so a
# follow-up read_page on them is instant. Pages are cached under the SAME key read_page looks up,
# so the warm content is actually reused.
# This is NOT fire-and-forget. Every warm-up is one asyncio.Task held in a registry, bounded by a
# hard timeout and a concurrency semaphore, self-removing on completion, and cancellable via
# shutdown().
class PrefetchManager:
    # Set the concurrency limit and the per-URL and per-batch timeouts (clamped to sane minimums).
    def __init__(self, *, max_concurrency: int = 3, per_url_timeout: float = 8.0,
                 task_timeout: float = 30.0) -> None:
        self._tasks: set[asyncio.Task] = set()
        self._sem = asyncio.Semaphore(max(1, max_concurrency))
        self._per_url_timeout = max(1.0, float(per_url_timeout))
        self._task_timeout = max(2.0, float(task_timeout))

    # Warm one URL into the page cache under read_page's cache key. Return True only when a new
    # entry was cached; a fresh entry, a blocked or failed fetch and an unusable page return False.
    async def _warm_one(self, url: str) -> bool:
        from core.cache import get_page_cache
        from core.fetch.antibot import is_antibot
        from core.fetch.url_utils import UnsafeFetchUrl, validate_public_fetch_url
        from core.profiles import get_runtime_profiles
        from core.read.service import _cache_key_for_read, _is_tls_error, _variant_label

        # Skip URLs whose cache entry is still fresh.
        cache = get_page_cache()
        cache_key = _cache_key_for_read(url, variant=_variant_label(url))
        if cache.is_fresh(cache_key):
            return False
        try:
            import httpx

            # Fetch the page directly (public URLs only), limited by the concurrency semaphore.
            safe_url = validate_public_fetch_url(url)
            async with self._sem:
                async with httpx.AsyncClient(
                    headers={"User-Agent": "Mozilla/5.0", "Accept": "text/html,*/*;q=0.8"},
                    timeout=self._per_url_timeout,
                    follow_redirects=True,
                ) as client:
                    resp = await client.get(safe_url)

            # Reject error responses, empty bodies and anti-bot challenge pages.
            if resp.status_code >= 400:
                return False
            html = resp.text
            if not html or is_antibot(html):
                return False

            # Extract now (off the event loop) and warm the clean markdown, not raw HTML:
            # far smaller, FTS-searchable, and reused directly by a later read_page. A page
            # that doesn't extract to usable text is not worth warming (and not hoarded).
            from core.extract.page_normalizer import normalize_page

            md = await asyncio.get_running_loop().run_in_executor(
                None, normalize_page, safe_url, html
            )
            if not md or len(md.strip()) < 200:
                get_runtime_profiles().record_reputation(safe_url, parse_empty=True)
                return False
            title = md.splitlines()[0].lstrip("# ").strip()[:200] if md.strip() else ""
            cache.cache_page(cache_key, title, clean_text=md, raw_html="")
            get_runtime_profiles().record_reputation(safe_url, parse_ok=True)
            return True
        except UnsafeFetchUrl as exc:
            # A non-public URL is skipped; the block is only traced.
            trace_logger.info("prefetch.blocked url=%r reason=%s", url, exc)
        except Exception as exc:  # noqa: BLE001
            # Prefetch verifies TLS, so it observes broken hosts for free — record the
            # verdict so the next search's triage already knows.
            if _is_tls_error(exc):
                get_runtime_profiles().record_reputation(url, tls_failed=True)
                trace_logger.info("prefetch.tls_fail url=%r", url)
            else:
                logger.debug("prefetch fetch failed for %s: %s", url, exc)
        return False

    # Warm a batch of URLs under one hard-timeout-bounded task and trace how many were cached.
    async def _warm_batch(self, urls: list[str]) -> None:
        t0 = time.perf_counter()
        warmed = 0
        try:
            async with asyncio.timeout(self._task_timeout):
                results = await asyncio.gather(
                    *(self._warm_one(u) for u in urls), return_exceptions=True
                )
            warmed = sum(1 for r in results if r is True)
        except (TimeoutError, asyncio.CancelledError):
            # A timed-out or cancelled batch is dropped silently: warming is best-effort and the
            # pages are simply fetched on demand later.
            pass
        except Exception as exc:  # noqa: BLE001
            logger.debug("prefetch batch error: %s", exc)
        trace_logger.info(
            "prefetch.done urls=%d warmed=%d ms=%.0f", len(urls), warmed,
            (time.perf_counter() - t0) * 1000,
        )

    # Schedule a tracked warm-up task for the given URLs. Return the task, or None when no URL is given.
    def schedule(self, urls: list[str]) -> asyncio.Task | None:
        # Drop empty URLs and duplicates, keeping the original order.
        targets = list(dict.fromkeys(u for u in urls if u))
        if not targets:
            return None
        task = asyncio.create_task(self._warm_batch(targets), name="prefetch")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    # Cancel and await all outstanding warm-up tasks (call at server shutdown).
    async def shutdown(self) -> None:
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()


# Process-wide manager, created on first use by get_prefetch_manager().
_manager: PrefetchManager | None = None


# Return the lazily-initialised global PrefetchManager, configured from search_config.
def get_prefetch_manager() -> PrefetchManager:
    global _manager
    if _manager is None:
        from core.config import load_search_config

        cfg = load_search_config().search
        _manager = PrefetchManager(per_url_timeout=cfg.prefetch_fetch_timeout)
    return _manager


# Cancel all outstanding prefetch tasks (exposed for the MCP server's shutdown hook).
async def shutdown_prefetch() -> None:
    if _manager is not None:
        await _manager.shutdown()
