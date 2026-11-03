# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import json

import httpx

from core.config.settings import _CONFIG_PATH, BrowserSection, load_search_config
from core.fetch.browser.client import BrowserClient
from core.fetch.browser.identity_store import IdentityStore
from core.fetch.browser.models import STATUS_OK, STATUS_UNAVAILABLE, BrowserFetch


# Identity store

# Build a browser storage state that holds the given cookies and no origins.
def _state(*cookies: dict) -> dict:
    return {"cookies": list(cookies), "origins": []}


# Verify that the first checkpoint of a family gets generation 1 and becomes the latest good state,
# which is None before any checkpoint.
def test_identity_checkpoint_and_latest_good(tmp_path):
    store = IdentityStore(str(tmp_path / "id.db"))
    assert store.latest_good("google") is None
    gen = store.checkpoint("google", _state({"name": "NID", "value": "1", "domain": ".google.com"}))
    assert gen == 1
    state = store.latest_good("google")
    assert state and state["cookies"][0]["name"] == "NID"


# Verify that with max_generations=2 only the two newest of five checkpoint generations survive.
def test_identity_prune_keeps_newest_good(tmp_path):
    store = IdentityStore(str(tmp_path / "id.db"), max_generations=2)
    for i in range(5):
        store.checkpoint("yandex", _state({"name": f"c{i}", "value": str(i), "domain": "yandex.ru"}))

    # Only the 2 newest generations survive pruning.
    rows = store._get_conn().execute(
        "SELECT generation FROM identity_generations WHERE family = 'yandex' ORDER BY generation"
    ).fetchall()
    assert [r["generation"] for r in rows] == [4, 5]


# Verify that rotate burns the newest generation and returns the older good one, which then
# becomes the latest good state.
def test_identity_rotate_burns_latest_and_falls_back(tmp_path):
    store = IdentityStore(str(tmp_path / "id.db"))
    store.checkpoint("brave", _state({"name": "old", "value": "1", "domain": "search.brave.com"}))
    store.checkpoint("brave", _state({"name": "new", "value": "2", "domain": "search.brave.com"}))

    # Rotate burns the newest (gen 2) and returns the older good generation (gen 1).
    fallback = store.rotate("brave")
    assert fallback and fallback["cookies"][0]["name"] == "old"

    # The burned generation is no longer the "latest good".
    assert store.latest_good("brave")["cookies"][0]["name"] == "old"


# Verify that rotate returns None when no older good generation exists.
def test_identity_rotate_with_no_fallback_returns_none(tmp_path):
    store = IdentityStore(str(tmp_path / "id.db"))
    store.checkpoint("ddg", _state({"name": "only", "value": "1", "domain": "duckduckgo.com"}))

    # Nothing good is left, so the caller falls back to the seed.
    assert store.rotate("ddg") is None


# Verify that the cookie header for a host contains only the cookies whose domain matches it.
def test_identity_cookie_header_is_host_scoped(tmp_path):
    store = IdentityStore(str(tmp_path / "id.db"))
    store.checkpoint(
        "google",
        _state(
            {"name": "NID", "value": "abc", "domain": ".google.com"},
            {"name": "other", "value": "x", "domain": "example.com"},
        ),
    )
    header = store.cookie_header_for("google", "www.google.com")

    # The example.com cookie is excluded by host scoping.
    assert header == "NID=abc"


# Config axes

# Verify the default browser settings: page fallback, 2048 MB RSS cap, daemon autostart on the first
# tool call and a 15 minute idle shutdown (0 would keep it warm forever).
def test_browser_section_defaults():
    sec = BrowserSection()
    assert sec.browser_fallback == "page"
    assert sec.max_rss_mb == 2048
    assert sec.autostart_daemon is True
    assert sec.daemon_idle_shutdown_sec == 900.0


# Verify that the shipped search config enables the full browser fallback for search.
def test_shipped_config_allows_search_browser_fallback():
    cfg = load_search_config(path=_CONFIG_PATH)
    assert cfg.browser.browser_fallback == "full"


# Verify that an invalid browser_fallback value falls back to the safe default while a valid
# engine value is preserved.
def test_browser_config_validates_enums(tmp_path):
    cfg_path = tmp_path / "search_config.json"
    cfg_path.write_text(
        json.dumps({"browser": {"browser_fallback": "bogus", "engine": "chromium"}}),
        encoding="utf-8",
    )
    cfg = load_search_config(path=cfg_path)

    # The invalid value falls back to the safe default.
    assert cfg.browser.browser_fallback == "page"

    # The valid value is preserved.
    assert cfg.browser.engine == "chromium"


# Client dispatch (mocked daemon)

# HTTP response double with a JSON body and a status code.
class _FakeResp:
    # Store the JSON body and the status code.
    def __init__(self, body: dict, status: int = 200) -> None:
        self._body = body
        self.status_code = status

    # Return the stored JSON body.
    def json(self) -> dict:
        return self._body


# Stand-in for the HTTP client of the daemon: records the last posted payload and answers with a
# canned body, so the dispatch runs without a live browser or daemon.
class _FakeHttp:
    # Store the canned response body.
    def __init__(self, body: dict) -> None:
        self._body = body
        self.posted: dict | None = None

    # Answer a GET with an empty engine list.
    async def get(self, url: str) -> _FakeResp:
        return _FakeResp({"engines": []})

    # Record the posted JSON payload and answer with the canned body.
    async def post(self, url: str, json: dict) -> _FakeResp:  # noqa: A002 — mirror httpx kwarg
        self.posted = json
        return _FakeResp(self._body)


# Verify that the client returns the unavailable status when the browser fallback is off.
def test_client_off_returns_unavailable():
    client = BrowserClient(cfg=BrowserSection(browser_fallback="off"))
    result = asyncio.run(client.fetch("https://example.com"))
    assert result.status == STATUS_UNAVAILABLE


# Verify that the client posts the fetch to the warm daemon with the wait converted to
# milliseconds and the chromium engine, and returns an ok result from the warm backend.
def test_client_warm_dispatches_to_daemon():
    client = BrowserClient(cfg=BrowserSection(browser_fallback="page"))
    fake = _FakeHttp({"url": "https://example.com", "status": "ok", "html": "<html>hi</html>", "ms": 12.0})
    client._http = fake
    result = asyncio.run(client.fetch("https://example.com", wait_sec=2.0))
    assert isinstance(result, BrowserFetch)
    assert result.status == STATUS_OK and result.ok and result.backend == "warm"
    assert fake.posted["wait_ms"] == 2000 and fake.posted["engine"] == "chromium"


# Verify that the HTTP client used to reach the local daemon is created with trust_env disabled, so
# environment proxy settings are ignored.
def test_daemon_http_client_ignores_environment_proxy(monkeypatch):
    captured: dict = {}

    # Replacement for httpx.AsyncClient that records its keyword arguments.
    class _FakeAsyncClient:
        # Record the constructor keyword arguments.
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _FakeAsyncClient)
    client = BrowserClient(cfg=BrowserSection(browser_fallback="page"))

    assert isinstance(client._client(), _FakeAsyncClient)
    assert captured["trust_env"] is False


# Verify that an unreachable daemon gives an unavailable result and available() == False instead of
# raising.
def test_client_warm_unreachable_is_unavailable_not_crash():
    client = BrowserClient(cfg=BrowserSection(autostart_daemon=False))

    # HTTP client double whose every call fails like a daemon that is down.
    class _Boom:
        # Fail the POST with a connection error.
        async def post(self, *a, **k):
            raise ConnectionError("daemon down")

        # Fail the GET with a connection error.
        async def get(self, *a, **k):
            raise ConnectionError("daemon down")

    client._http = _Boom()
    result = asyncio.run(client.fetch("https://example.com"))
    assert result.status == STATUS_UNAVAILABLE
    assert asyncio.run(client.available()) is False


# Verify that with autostart enabled an unreachable daemon is spawned exactly once and the client
# then reports it ready.
def test_client_autostart_spawns_daemon_when_unreachable(monkeypatch):
    client = BrowserClient(cfg=BrowserSection(autostart_daemon=True))
    state = {"up": False, "spawns": 0}

    # Report whether the fake daemon is up.
    async def fake_probe() -> bool:
        return state["up"]

    # Count the spawn and bring the fake daemon up.
    def fake_spawn() -> None:
        state["spawns"] += 1
        state["up"] = True

    monkeypatch.setattr(client, "_probe", fake_probe)
    monkeypatch.setattr(client, "_spawn_process", fake_spawn)
    assert asyncio.run(client._daemon_ready()) is True
    assert state["spawns"] == 1


# Verify that with autostart disabled an unreachable daemon is reported as not ready and is never
# spawned.
def test_client_no_autostart_when_disabled(monkeypatch):
    client = BrowserClient(cfg=BrowserSection(autostart_daemon=False))
    spawns: list[int] = []

    # Report the fake daemon as down.
    async def fake_probe() -> bool:
        return False

    monkeypatch.setattr(client, "_probe", fake_probe)
    monkeypatch.setattr(client, "_spawn_process", lambda: spawns.append(1))
    assert asyncio.run(client._daemon_ready()) is False
    assert spawns == []
