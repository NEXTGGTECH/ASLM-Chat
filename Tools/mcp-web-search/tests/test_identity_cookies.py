# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import time

from core.engines.models import EngineRequest
from core.fetch.browser.identity_store import IdentityStore
import core.fetch.transport as transport_mod


# Create an identity store backed by a database file in the test's temporary directory.
def _store(tmp_path) -> IdentityStore:
    return IdentityStore(str(tmp_path / "ident.db"))


# Verify that Set-Cookie values merged for an engine come back in its cookie header, and that
# another engine does not see them.
def test_merge_set_cookie_and_header_roundtrip(tmp_path):
    store = _store(tmp_path)
    store.merge_set_cookie("startpage", "www.startpage.com", [
        "consent=1; Domain=.startpage.com; Path=/; Max-Age=3600",
        # A session cookie without a domain is host-scoped.
        "session=abc; Path=/",
    ])
    header = store.http_cookie_header("startpage", "www.startpage.com")
    assert "consent=1" in header
    assert "session=abc" in header

    # Different owner sees nothing — cookies are per-engine.
    assert store.http_cookie_header("yandex", "www.startpage.com") == ""


# Verify that a cookie is sent to the host its domain covers and not to an unrelated host.
def test_cookie_domain_scoping(tmp_path):
    store = _store(tmp_path)
    store.merge_set_cookie("google", "www.google.com",
                           ["NID=xyz; Domain=.google.com; Max-Age=1000"])
    assert "NID=xyz" in store.http_cookie_header("google", "www.google.com")

    # Unrelated host must not receive the cookie.
    assert store.http_cookie_header("google", "yandex.com") == ""


# Verify that a cookie deleted with Max-Age=0 and a cookie whose stored expiry has passed are no
# longer returned, while the other cookie is kept.
def test_expired_and_deleted_cookies_are_dropped(tmp_path):
    store = _store(tmp_path)
    store.merge_set_cookie("qwant", "api.qwant.com", ["a=1; Max-Age=1000", "b=2; Max-Age=1000"])

    # Max-Age=0 deletes an existing cookie.
    store.merge_set_cookie("qwant", "api.qwant.com", ["a=1; Max-Age=0"])
    header = store.http_cookie_header("qwant", "api.qwant.com")
    assert "a=1" not in header and "b=2" in header

    # An already-expired cookie is never returned.
    store.merge_set_cookie("qwant", "api.qwant.com", ["c=3; Max-Age=1000"])
    store._get_conn().execute(
        "UPDATE http_cookies SET expires = ? WHERE owner='qwant' AND name='c'", (time.time() - 5,)
    )
    store._get_conn().commit()
    assert "c=3" not in store.http_cookie_header("qwant", "api.qwant.com")


# Verify that a session cookie (no Max-Age) stops being replayed once it is older than the session
# TTL, while a persistent cookie of the same age is still sent.
def test_session_cookies_age_out(tmp_path):
    store = _store(tmp_path)

    # A cookie without Max-Age is a session cookie.
    store.merge_set_cookie("startpage", "www.startpage.com", ["sess=tok"])
    assert "sess=tok" in store.http_cookie_header("startpage", "www.startpage.com")

    # Backdate its update time past the session TTL → it must no longer be replayed.
    from core.fetch.browser.identity_store import _SESSION_COOKIE_TTL

    store._get_conn().execute(
        "UPDATE http_cookies SET updated = ? WHERE owner='startpage' AND name='sess'",
        (time.time() - _SESSION_COOKIE_TTL - 10,),
    )
    store._get_conn().commit()
    assert store.http_cookie_header("startpage", "www.startpage.com") == ""

    # A persistent (Max-Age) cookie of the same age is unaffected.
    store.merge_set_cookie("startpage", "www.startpage.com", ["keep=1; Max-Age=99999"])
    store._get_conn().execute(
        "UPDATE http_cookies SET updated = ? WHERE owner='startpage' AND name='keep'",
        (time.time() - _SESSION_COOKIE_TTL - 10,),
    )
    store._get_conn().commit()
    assert "keep=1" in store.http_cookie_header("startpage", "www.startpage.com")


# Verify that the transport replays the cookies stored for the engine into the request while
# keeping the cookies the engine already set.
def test_transport_replay_merges_stored_cookies(tmp_path, monkeypatch):
    store = _store(tmp_path)
    store.merge_set_cookie("brave", "search.brave.com", ["pref=dark; Max-Age=9999"])
    monkeypatch.setattr(transport_mod, "get_identity_store", lambda: store, raising=False)
    monkeypatch.setattr(
        "core.fetch.browser.identity_store.get_identity_store", lambda: store
    )
    req = EngineRequest(method="GET", url="https://search.brave.com/search",
                        cookies={"fresh": "1"}, primp_target="firefox", identity_key="brave")
    merged = transport_mod._replay_identity_cookies(req, "search.brave.com")

    # The cookie is replayed from the store.
    assert merged.cookies["pref"] == "dark"

    # The engine's own seed cookie is kept.
    assert merged.cookies["fresh"] == "1"


# Verify that the transport stores the Set-Cookie values of a response for the request's engine.
def test_transport_capture_writes_back(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(
        "core.fetch.browser.identity_store.get_identity_store", lambda: store
    )
    req = EngineRequest(method="GET", url="https://yandex.com/search", identity_key="yandex")
    resp = transport_mod.TransportResponse(
        status=200, body=b"", set_cookie=["yp=earned; Domain=.yandex.com; Max-Age=5000"]
    )
    transport_mod._capture_identity_cookies(req, "yandex.com", resp)
    assert "yp=earned" in store.http_cookie_header("yandex", "yandex.com")


# Verify that a request without an identity key is returned unchanged by the replay and that the
# capture accepts a response for it without raising.
def test_no_identity_key_is_a_noop(tmp_path, monkeypatch):
    store = _store(tmp_path)
    monkeypatch.setattr(
        "core.fetch.browser.identity_store.get_identity_store", lambda: store
    )
    req = EngineRequest(method="GET", url="https://x.com/", cookies={"a": "1"})

    # No identity_key → request returned unchanged, nothing captured.
    assert transport_mod._replay_identity_cookies(req, "x.com") is req
    resp = transport_mod.TransportResponse(status=200, body=b"", set_cookie=["z=9"])

    # The capture must not raise.
    transport_mod._capture_identity_cookies(req, "x.com", resp)
