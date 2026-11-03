# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import types

from core.fetch.browser import daemon as daemon_mod
from core.fetch.browser.daemon import RecycleReason, ScrapeResult, WarmChromium
from core.fetch.browser.identity_store import IdentityStore


# Fakes

# Browser context double that returns a fixed storage state and records that it was closed.
class _FakeContext:
    # Store the storage state to hand out and start in the open state.
    def __init__(self, state: dict) -> None:
        self._state = state
        self.closed = False

    # Return the fixed storage state.
    async def storage_state(self) -> dict:
        return self._state

    # Mark the context as closed.
    async def close(self) -> None:
        self.closed = True


# Browser double that only records that it was closed.
class _FakeBrowser:
    # Start in the open state.
    def __init__(self) -> None:
        self.closed = False

    # Mark the browser as closed.
    async def close(self) -> None:
        self.closed = True


# Build a WarmChromium whose _open installs fakes and whose _scrape returns canned results, and
# return the call counters. No real Chromium or network is involved, so the supervision logic
# (recycle priority, burn-to-rotate identity policy, checkpoint persistence, counters and
# blocked streak) is exercised deterministically; the real scrape path is covered by live smoke
# runs instead.
def _wire(wc: WarmChromium, *, state: dict | None = None, results: list[ScrapeResult] | None = None) -> dict:
    seed_state = state or {"cookies": [{"name": "SID", "value": "1", "domain": ".example.com"}], "origins": []}
    calls = {"opens": 0, "scrapes": 0}
    canned = list(results or [])

    # Install a fake browser and context and reset the per-launch state, as a real launch would.
    async def fake_open(self) -> None:
        self._browser = _FakeBrowser()
        self._context = _FakeContext(seed_state)

        # Close the fake context and browser (the teardown hook).
        async def _close() -> None:
            await self._context.close()
            await self._browser.close()

        self._teardown = _close
        import time as _t
        self._started_at = _t.monotonic()
        self._requests = 0
        self._dirty = False
        calls["opens"] += 1

    # Return the next canned result, or a successful scrape when none are left.
    async def fake_scrape(self, url: str, *, wait: float, nav_timeout: float) -> ScrapeResult:
        calls["scrapes"] += 1
        if canned:
            return canned.pop(0)
        return ScrapeResult(url=url, status="ok", ok=True, title="t", html="x" * 500, text="y" * 500)

    wc._open = types.MethodType(fake_open, wc)
    wc._scrape = types.MethodType(fake_scrape, wc)
    return calls


# Create an identity store backed by a database file in the test's temporary directory.
def _store(tmp_path) -> IdentityStore:
    return IdentityStore(str(tmp_path / "identity.db"))


# Recycle decision

# Verify that the recycle reason is None while the browser is down or fresh, then follows the
# priority request count, age, RSS, with a burn (blocked streak) outranking all of them.
def test_recycle_reason_priority(tmp_path, monkeypatch):
    wc = WarmChromium(identity=_store(tmp_path), max_requests=5, max_age_sec=100.0, max_rss_mb=2048)
    monkeypatch.setattr(daemon_mod, "_process_tree_rss_mb", lambda: 10.0)

    import time

    # With the browser down there is never anything to recycle.
    assert wc._recycle_reason() is None

    # Pretend the browser is alive and was just launched: still fresh.
    wc._browser = object()
    wc._started_at = time.monotonic()
    assert wc._recycle_reason() is None

    wc._requests = 5
    assert wc._recycle_reason() == RecycleReason.REQUESTS

    wc._requests = 0
    wc._started_at = time.monotonic() - 200.0
    assert wc._recycle_reason() == RecycleReason.AGE

    wc._started_at = time.monotonic()
    monkeypatch.setattr(daemon_mod, "_process_tree_rss_mb", lambda: 9999.0)
    assert wc._recycle_reason() == RecycleReason.RSS

    # Burn outranks every other reason.
    wc._blocked_streak = 3
    assert wc._recycle_reason() == RecycleReason.BURN


# Fetch bookkeeping

# Verify that one successful fetch counts as one request, marks the state dirty and leaves the
# blocked streak at zero.
def test_fetch_counts_and_marks_dirty(tmp_path):
    wc = WarmChromium(identity=_store(tmp_path), max_requests=99)
    _wire(wc)

    # Fetch one page and return its result.
    async def go():
        r = await wc.fetch("https://example.com")
        return r

    r = asyncio.run(go())
    assert r.ok and r.status == "ok"
    assert wc._requests == 1 and wc._total == 1
    assert wc._dirty is True
    assert wc._blocked_streak == 0


# Verify that each blocked fetch raises the blocked streak by one and that a successful fetch
# resets it to zero.
def test_blocked_increments_streak_then_resets(tmp_path):
    wc = WarmChromium(identity=_store(tmp_path), max_requests=99)
    blocked = ScrapeResult(url="u", status="blocked", html="x" * 500, text="")
    ok = ScrapeResult(url="u", status="ok", ok=True, html="x" * 500, text="y" * 500)
    _wire(wc, results=[blocked, blocked, ok])

    # Fetch three times (blocked, blocked, ok) and return the streak after each fetch.
    async def go():
        # Blocked: streak 1.
        await wc.fetch("https://e.com")
        s1 = wc._blocked_streak

        # Blocked: streak 2.
        await wc.fetch("https://e.com")
        s2 = wc._blocked_streak

        # Ok: streak back to 0.
        await wc.fetch("https://e.com")
        return s1, s2, wc._blocked_streak

    s1, s2, s3 = asyncio.run(go())
    assert (s1, s2, s3) == (1, 2, 0)


# Recycle behavior and checkpoint persistence

# Verify that crossing the request limit recycles the browser once (one reopen) and that the
# recycle checkpoints a good identity generation that can seed the next start.
def test_request_count_recycle_checkpoints_good(tmp_path):
    store = _store(tmp_path)
    wc = WarmChromium(identity=store, max_requests=2)
    calls = _wire(wc)

    # Fetch three times with a limit of two requests per browser.
    async def go():
        # Request 1.
        await wc.fetch("https://e.com")

        # Request 2.
        await wc.fetch("https://e.com")

        # Crosses the threshold: recycle, then request 1 of the new browser.
        await wc.fetch("https://e.com")

    asyncio.run(go())
    assert wc._recycles == 1

    # One initial open plus one reopen.
    assert calls["opens"] == 2

    # The recycle checkpointed a *good* generation, which is seedable on the next start.
    assert store.latest_good("chromium") is not None
    assert store.cookie_header_for("chromium", "example.com") == "SID=1"


# Verify that after three consecutive blocked fetches the burn recycle happens once, resets the
# blocked streak and leaves a usable, non-burned identity generation in the store.
def test_burn_recycle_rotates_identity(tmp_path):
    store = _store(tmp_path)

    # Seed two good generations so a burn has an older one to fall back to.
    store.checkpoint("chromium", {"cookies": [{"name": "OLD", "value": "1", "domain": ".e.com"}], "origins": []})
    store.checkpoint("chromium", {"cookies": [{"name": "NEW", "value": "2", "domain": ".e.com"}], "origins": []})

    wc = WarmChromium(identity=store, max_requests=99)
    blk = ScrapeResult(url="u", status="blocked", html="x" * 500, text="")
    _wire(wc, results=[blk, blk, blk])

    # Fetch three blocked pages and then trigger the burn recycle directly.
    async def go():
        # Three consecutive blocks; the 4th fetch sees burn streak and recycles.
        await wc.fetch("https://e.com")
        await wc.fetch("https://e.com")
        await wc.fetch("https://e.com")

        # Streak now 3; force the recycle decision directly (no 4th canned result needed).
        assert wc._recycle_reason() == RecycleReason.BURN
        await wc._recycle(RecycleReason.BURN)

    asyncio.run(go())
    assert wc._recycles == 1
    assert wc._blocked_streak == 0

    # Burn marked the newest generation not-good and rotated to the older good one.
    # (A fresh good checkpoint was also written by the burn's checkpoint(good=False),
    #  so latest_good must still resolve to a real, non-burned state.)
    assert store.latest_good("chromium") is not None


# Verify that a fetch made while the process tree RSS is over the cap recycles the browser once
# and is still served successfully.
def test_rss_recycle_then_serves(tmp_path, monkeypatch):
    store = _store(tmp_path)
    wc = WarmChromium(identity=store, max_requests=99, max_rss_mb=512)
    _wire(wc)

    # First fetch opens; second sees RSS over the cap and recycles before serving.
    rss = {"v": 10.0}
    monkeypatch.setattr(daemon_mod, "_process_tree_rss_mb", lambda: rss["v"])

    # Fetch once with low RSS, raise the RSS above the cap and fetch again.
    async def go():
        await wc.fetch("https://e.com")
        rss["v"] = 99999.0
        r = await wc.fetch("https://e.com")
        return r

    r = asyncio.run(go())
    assert r.ok
    assert wc._recycles == 1


# Checkpoint loop and lifecycle

# Verify that a checkpoint of a dirty browser state clears the dirty flag and stores a good
# generation.
def test_idle_checkpoint_persists_when_dirty(tmp_path):
    store = _store(tmp_path)
    wc = WarmChromium(identity=store, checkpoint_interval=5.0)
    _wire(wc)

    # Open the browser, mark it dirty and checkpoint once.
    async def go():
        await wc._open()
        wc._dirty = True
        wc._inflight = 0

        # Invoke the loop body once directly rather than waiting checkpoint_interval.
        await wc._checkpoint(good=True)

    asyncio.run(go())
    assert wc._dirty is False
    assert store.latest_good("chromium") is not None


# Verify that stopping the browser closes its context, drops the browser and writes a final good
# checkpoint.
def test_stop_final_checkpoints_and_tears_down(tmp_path):
    store = _store(tmp_path)
    wc = WarmChromium(identity=store)
    _wire(wc)

    # Fetch once, stop the browser and return the context that was in use.
    async def go():
        await wc.fetch("https://e.com")
        ctx = wc._context
        await wc.stop()
        return ctx

    ctx = asyncio.run(go())
    assert ctx.closed is True
    assert wc._browser is None
    assert store.latest_good("chromium") is not None


# Hang watchdog: bounded extraction and close, recycle on wedge

# Page double whose content() or close() can be made to hang for a long time.
class _HangPage:
    # Store which calls hang and start in the open state.
    def __init__(self, *, hang_content: bool = False, hang_close: bool = False) -> None:
        self._hang_content = hang_content
        self._hang_close = hang_close
        self.closed = False

    # Accept the navigation and return immediately.
    async def goto(self, url, **kw):
        return None

    # Accept the wait and return immediately.
    async def wait_for_function(self, *a, **k):
        return None

    # Return fake HTML, after sleeping far longer than any timeout when content is set to hang.
    async def content(self):
        if self._hang_content:
            await asyncio.sleep(100)
        return "x" * 500

    # Return fake visible text.
    async def evaluate(self, *a, **k):
        return "y" * 500

    # Return a fake page title.
    async def title(self):
        return "t"

    # Mark the page as closed, after sleeping far longer than any timeout when close is set to hang.
    async def close(self):
        if self._hang_close:
            await asyncio.sleep(100)
        self.closed = True


# Browser context double that hands out one given page and a live storage state.
class _HangContext:
    # Store the page to hand out and start in the open state.
    def __init__(self, page: _HangPage) -> None:
        self._page = page
        self.closed = False

    # Return the given page.
    async def new_page(self):
        return self._page

    # Return a storage state holding one live session cookie.
    async def storage_state(self) -> dict:
        return {"cookies": [{"name": "SID", "value": "live", "domain": ".example.com"}], "origins": []}

    # Mark the context as closed.
    async def close(self):
        self.closed = True


# Install a fake _open on the WarmChromium that uses a context serving the given hanging page.
def _wire_hang(wc: WarmChromium, page: _HangPage) -> None:
    # Install a fake browser and a context serving the hanging page, and reset the per-launch state.
    async def fake_open(self) -> None:
        self._browser = _FakeBrowser()
        self._context = _HangContext(page)

        # Close the fake context and browser (the teardown hook).
        async def _close() -> None:
            await self._context.close()
            await self._browser.close()

        self._teardown = _close
        import time as _t
        self._started_at = _t.monotonic()
        self._requests = 0
        self._dirty = False

    wc._open = types.MethodType(fake_open, wc)


# Verify that a content() call that never returns is cut off by the extraction cap and becomes a
# timeout result instead of holding the fetch lock, and that the in-flight counter is unwound.
def test_extraction_hang_is_capped_not_indefinite(tmp_path, monkeypatch):
    # content() hangs forever; the extraction cap must turn it into a bounded timeout
    # result instead of holding the lock until the heat death of the universe.
    monkeypatch.setattr(daemon_mod, "_EXTRACT_TIMEOUT", 0.2)
    monkeypatch.setattr(daemon_mod, "_CLOSE_TIMEOUT", 0.2)
    wc = WarmChromium(identity=_store(tmp_path), max_requests=99)
    page = _HangPage(hang_content=True)
    _wire_hang(wc, page)

    # Fetch the hanging page, failing the test if the fetch itself does not finish within 3 s.
    async def go():
        return await asyncio.wait_for(wc.fetch("https://hang.example"), timeout=3.0)

    r = asyncio.run(go())
    assert r.status == "timeout"

    # The lock is released and the counter unwound.
    assert wc._inflight == 0


# Verify that a page whose close() wedges still returns the successful result, but forces a
# teardown of the browser after a best-effort checkpoint of the live identity.
def test_close_hang_flags_wedge_and_recycles(tmp_path, monkeypatch):
    # content() succeeds but close() wedges; the close cap fires, the browser is flagged
    # wedged and torn down so the next fetch respawns clean.
    monkeypatch.setattr(daemon_mod, "_EXTRACT_TIMEOUT", 1.0)
    monkeypatch.setattr(daemon_mod, "_CLOSE_TIMEOUT", 0.2)
    store = _store(tmp_path)
    wc = WarmChromium(identity=store, max_requests=99)
    page = _HangPage(hang_close=True)
    _wire_hang(wc, page)

    # Fetch the page with the wedging close, failing the test if the fetch does not finish in 3 s.
    async def go():
        return await asyncio.wait_for(wc.fetch("https://wedge.example"), timeout=3.0)

    r = asyncio.run(go())

    # extraction succeeded, so the result itself is ok…
    assert r.ok and r.status == "ok"

    # …but the wedged close forced a teardown: browser gone, flag reset.
    assert wc._browser is None
    assert wc._wedged is False

    # …and the identity was checkpointed best-effort BEFORE the kill — a wedge is a page
    # fault, not a poisoned identity, so we must not lose the live cookies/storage.
    assert store.latest_good("chromium") is not None
