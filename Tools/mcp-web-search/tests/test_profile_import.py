# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import sqlite3

import pytest

from core.engines.models import EngineRequest
from core.fetch.browser import profile_import as PI
from core.fetch.browser.identity_store import IdentityStore
from core.fetch.browser.profile_import import DomainFilter, ImportedCookie


# Config gating

# Verify that profile import is disabled by default while the cross-browser set is already present,
# so enabling it needs no extra config. The tests of this module are hermetic: they never read the
# machine's real browser profiles.
def test_profile_import_disabled_by_default():
    from core.config.settings import load_search_config

    cfg = load_search_config()
    assert cfg.profile_import.enabled is False
    assert set(cfg.profile_import.browsers) == {"chrome", "edge", "brave", "firefox"}


# Verify that the browser list from a config file drops the unknown browser "netscape".
def test_config_string_list_drops_unknown_browser(tmp_path):
    import json
    from core.config.settings import load_search_config

    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"profile_import": {"browsers": ["chrome", "netscape", "firefox"]}}))
    cfg = load_search_config(path=path)
    assert cfg.profile_import.browsers == ["chrome", "firefox"]


# Chromium value decryption (AES-GCM v10)

# Encrypt a plaintext into a synthetic Chromium v10 cookie blob (prefix, nonce, AES-GCM ciphertext) under
# a known key. The DPAPI key-unwrap step is Windows-only and is covered by discovery, not by these tests.
def _aesgcm_v10(key: bytes, plaintext: bytes) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    nonce = os.urandom(12)
    return b"v10" + nonce + AESGCM(key).encrypt(nonce, plaintext, None)


# Verify that a v10 blob decrypts back to the original plaintext with the same key.
def test_decrypt_chromium_v10_roundtrip():
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = AESGCM.generate_key(bit_length=256)
    blob = _aesgcm_v10(key, b"session-token-123")
    plain, ok = PI._decrypt_chromium_value(blob, key)
    assert ok is True
    assert plain == "session-token-123"


# Verify that decrypting a v10 blob with the wrong key is skipped (bad tag) and returns an empty value,
# never a garbage one.
def test_decrypt_chromium_v10_wrong_key_is_skipped_not_fabricated():
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    blob = _aesgcm_v10(AESGCM.generate_key(bit_length=256), b"secret")
    plain, ok = PI._decrypt_chromium_value(blob, AESGCM.generate_key(bit_length=256))
    assert ok is False and plain == ""


# Verify that a v20 App-Bound blob is reported as not decryptable. App-Bound Encryption is sealed to the
# browser process, so it must not be pretended to be readable.
def test_decrypt_chromium_v20_app_bound_is_not_decryptable():
    plain, ok = PI._decrypt_chromium_value(b"v20" + os.urandom(40), os.urandom(32))
    assert ok is False and plain == ""


# Synthetic cookie DBs

# Create a Chromium-style cookies table at the path and fill it with the given rows.
def _make_chromium_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE cookies (host_key TEXT, name TEXT, encrypted_value BLOB, value TEXT, "
        "path TEXT, expires_utc INTEGER, is_secure INTEGER, is_httponly INTEGER)"
    )
    conn.executemany(
        "INSERT INTO cookies (host_key, name, encrypted_value, value, path, expires_utc, "
        "is_secure, is_httponly) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


# Create a Firefox-style moz_cookies table (plaintext values) at the path and fill it with the given rows.
def _make_firefox_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE moz_cookies (host TEXT, name TEXT, value TEXT, path TEXT, "
        "expiry INTEGER, isSecure INTEGER, isHttpOnly INTEGER)"
    )
    conn.executemany(
        "INSERT INTO moz_cookies (host, name, value, path, expiry, isSecure, isHttpOnly) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.commit()
    conn.close()


# Verify that reading a Chromium cookie DB decrypts v10 values, drops cookies outside the domain
# allowlist and counts the App-Bound (v20) cookie as locked instead of skipped.
def test_read_chromium_cookies_decrypts_and_filters(tmp_path):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    key = AESGCM.generate_key(bit_length=256)
    db = tmp_path / "Cookies"
    _make_chromium_db(db, [
        (".google.com", "SID", _aesgcm_v10(key, b"real"), "", "/", 0, 1, 1),
        # Filtered out by the allowlist.
        (".other.com", "X", _aesgcm_v10(key, b"nope"), "", "/", 0, 0, 0),
        # Counted as abe_locked.
        (".google.com", "ABE", b"v20" + os.urandom(30), "", "/", 0, 0, 0),
    ])
    cookies, skipped, abe = PI._read_chromium_cookies(db, key, DomainFilter(["google.com"]))
    names = {c.name: c.value for c in cookies}
    assert names == {"SID": "real"}
    assert skipped == 0 and abe == 1


# Verify that reading a Firefox cookie DB returns the plaintext cookies of the allowed domains and
# drops the others.
def test_read_firefox_cookies_plaintext(tmp_path):
    db = tmp_path / "cookies.sqlite"
    _make_firefox_db(db, [
        (".google.com", "PREF", "abc", "/", 9999999999, 0, 0),
        (".yandex.ru", "yp", "def", "/", 0, 1, 1),
        # Filtered out by the allowlist.
        (".spam.com", "junk", "zzz", "/", 0, 0, 0),
    ])
    cookies = PI._read_firefox_cookies(db, DomainFilter(["google.com", "yandex.ru"]))
    got = {c.name: c.value for c in cookies}
    assert got == {"PREF": "abc", "yp": "def"}


# Domain allowlist

# Verify that the domain filter matches by domain suffix only (not by lookalike hosts) and that an
# empty allowlist allows every domain.
def test_domain_filter_suffix_match_and_empty_allows_all():
    f = DomainFilter(["google.com"])
    assert f.matches(".google.com") and f.matches("www.google.com")
    assert not f.matches("notgoogle.com") and not f.matches("google.com.evil.net")
    assert DomainFilter([]).matches("anything.example")


# Default-browser-first ordering

# Verify that the detected default browser is moved to the front, and that the order stays unchanged
# when the default browser is not in the wanted set.
def test_default_first_ordering(monkeypatch):
    monkeypatch.setattr(PI, "detect_default_browser", lambda: "firefox")
    assert PI._default_first(["chrome", "edge", "firefox"]) == ["firefox", "chrome", "edge"]

    # Default not in the wanted set: order unchanged.
    monkeypatch.setattr(PI, "detect_default_browser", lambda: "safari")
    assert PI._default_first(["chrome", "firefox"]) == ["chrome", "firefox"]


# IdentityStore imported layer

# Create an IdentityStore backed by a database file inside the test's temp directory.
def _store(tmp_path) -> IdentityStore:
    return IdentityStore(str(tmp_path / "ident.db"))


# Verify that when the same cookie is imported twice the first one wins, the duplicate is not counted,
# and the stored cookies can be read back as a name-value map and as storage-state entries.
def test_import_first_wins_dedup_and_read(tmp_path):
    store = _store(tmp_path)
    n = store.import_cookies("chromium", [
        ImportedCookie(domain=".google.com", name="SID", value="win", secure=True),
        # Duplicate: dropped.
        ImportedCookie(domain=".google.com", name="SID", value="lose"),
        ImportedCookie(domain=".yandex.ru", name="yp", value="y", expires=9999999999),
    ], source="chrome")
    assert n == 2
    assert store.imported_cookies_map("chromium", "www.google.com") == {"SID": "win"}
    ss = store.imported_cookies_for("chromium", host="google.com")
    assert ss == [{"name": "SID", "value": "win", "domain": ".google.com", "path": "/", "secure": True}]


# Verify that an expired imported cookie is dropped on read, while live and session cookies are kept.
def test_expired_imported_cookie_dropped_on_read(tmp_path):
    store = _store(tmp_path)
    store.import_cookies("chromium", [
        ImportedCookie(domain=".google.com", name="live", value="1", expires=9999999999),
        # Expired in the past: dropped on read.
        ImportedCookie(domain=".google.com", name="dead", value="0", expires=1),
        # Session cookie: kept.
        ImportedCookie(domain=".google.com", name="sess", value="s", expires=0),
    ])
    got = store.imported_cookies_map("chromium", "google.com")
    assert "live" in got and "sess" in got and "dead" not in got


# Verify that a second import into the same family replaces the first set (replace is the default)
# and that clear_imported removes what is left.
def test_import_replace_and_clear(tmp_path):
    store = _store(tmp_path)
    store.import_cookies("firefox", [ImportedCookie(domain=".a.com", name="k", value="1")])
    store.import_cookies("firefox", [ImportedCookie(domain=".b.com", name="k2", value="2")])
    assert store.imported_cookies_map("firefox", "a.com") == {}
    assert store.imported_cookies_map("firefox", "b.com") == {"k2": "2"}
    store.clear_imported("firefox")
    assert store.imported_cookies_map("firefox", "b.com") == {}


# Verify that cookies imported for the chromium and firefox families do not leak into each other.
def test_families_are_isolated(tmp_path):
    store = _store(tmp_path)
    store.import_cookies("chromium", [ImportedCookie(domain=".google.com", name="c", value="1")])
    store.import_cookies("firefox", [ImportedCookie(domain=".google.com", name="f", value="2")])
    assert store.imported_cookies_map("chromium", "google.com") == {"c": "1"}
    assert store.imported_cookies_map("firefox", "google.com") == {"f": "2"}


# Transport merge: imported is the base layer, earned and seed cookies win

# Verify that imported cookies form the base layer of the replayed cookies: they fill a session the
# engine cannot earn itself, while a cookie the engine earned overrides the imported one.
def test_transport_merges_imported_as_base_layer(tmp_path, monkeypatch):
    import core.fetch.transport as T

    store = _store(tmp_path)
    store.import_cookies("chromium", [
        ImportedCookie(domain=".qwant.com", name="SID", value="imported"),
        ImportedCookie(domain=".qwant.com", name="CONSENT", value="imported-consent"),
    ])

    # Engine has since earned its own CONSENT (should override the imported one).
    store.merge_set_cookie("qwant", "www.qwant.com", ["CONSENT=earned; Domain=.qwant.com; Max-Age=1000"])
    monkeypatch.setattr("core.fetch.browser.identity_store.get_identity_store", lambda: store)

    req = EngineRequest(
        method="GET", url="https://www.qwant.com/", params={}, headers={},
        cookies={}, primp_target="chrome", primp_os="windows", identity_key="qwant",
    )
    merged = T._replay_identity_cookies(req, "www.qwant.com")
    assert merged.cookies["SID"] == "imported"
    assert merged.cookies["CONSENT"] == "earned"


# Verify that a firefox impersonation target replays cookies imported from Firefox, and that the target
# to cookie-family mapping gives firefox, chromium and an empty family for an unknown target.
def test_transport_firefox_family_mapping(tmp_path, monkeypatch):
    import core.fetch.transport as T

    store = _store(tmp_path)
    store.import_cookies("firefox", [ImportedCookie(domain=".duckduckgo.com", name="ff", value="1")])
    monkeypatch.setattr("core.fetch.browser.identity_store.get_identity_store", lambda: store)
    req = EngineRequest(
        method="GET", url="https://duckduckgo.com/", params={}, headers={},
        cookies={}, primp_target="firefox", primp_os="windows", identity_key="duckduckgo",
    )
    merged = T._replay_identity_cookies(req, "duckduckgo.com")
    assert merged.cookies.get("ff") == "1"
    assert T._imported_family("firefox") == "firefox"
    assert T._imported_family("chrome") == "chromium"

    # An unknown target has no primary family.
    assert T._imported_family("qwant") == ""


# Verify that a Firefox-imported session cookie is still replayed for a Chrome-impersonating engine. A
# real session for a host may live in the user's Firefox while the chromium jar is empty (App-Bound
# encryption), and a session cookie is not bound to the TLS fingerprint.
def test_transport_cross_family_fallback(tmp_path, monkeypatch):
    import core.fetch.transport as T

    store = _store(tmp_path)
    store.import_cookies("firefox", [ImportedCookie(domain=".startpage.com", name="pref", value="ff-session")])
    monkeypatch.setattr("core.fetch.browser.identity_store.get_identity_store", lambda: store)
    req = EngineRequest(
        method="GET", url="https://www.startpage.com/sp/search", params={}, headers={},
        cookies={}, primp_target="chrome", primp_os="windows", identity_key="startpage",
    )
    merged = T._replay_identity_cookies(req, "www.startpage.com")
    assert merged.cookies.get("pref") == "ff-session"


# Verify that the google engine (identity_key="google") receives no imported cookies. Google's GSA-UA
# trick needs a cookie-light identity; imported desktop cookies flip it to the JS layout the parser
# cannot read (verified live).
def test_google_gsa_engine_is_denied_imported_cookies(tmp_path, monkeypatch):
    import core.fetch.transport as T

    store = _store(tmp_path)
    store.import_cookies("firefox", [ImportedCookie(domain=".google.com", name="SID", value="ff")])
    monkeypatch.setattr("core.fetch.browser.identity_store.get_identity_store", lambda: store)
    req = EngineRequest(
        method="GET", url="https://www.google.com/search", params={}, headers={},
        cookies={}, primp_target="chrome", primp_os="windows", identity_key="google",
    )
    merged = T._replay_identity_cookies(req, "www.google.com")
    assert "SID" not in merged.cookies


# Run-once trigger honours the config switch

# Verify that running the import with the feature disabled and purge_on_disable set removes the
# previously imported cookies and reports that nothing ran.
def test_run_import_disabled_purges(tmp_path, monkeypatch):
    import core.config.settings as S
    from core.config.settings import SearchConfig, ProfileImportSection

    store = _store(tmp_path)
    store.import_cookies("chromium", [ImportedCookie(domain=".x.com", name="k", value="1")])
    monkeypatch.setattr("core.fetch.browser.identity_store.get_identity_store", lambda: store)
    cfg = SearchConfig(profile_import=ProfileImportSection(enabled=False, purge_on_disable=True))
    monkeypatch.setattr(S, "load_search_config", lambda *a, **k: cfg)

    out = PI.run_profile_import(force=True)
    assert out.enabled is False and out.ran is False
    assert store.imported_cookies_map("chromium", "x.com") == {}
