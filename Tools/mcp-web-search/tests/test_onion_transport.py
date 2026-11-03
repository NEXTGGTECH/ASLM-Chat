# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
from urllib.parse import urlparse

import curl_cffi
import curl_cffi.requests  # noqa: F401  - bind the submodule so monkeypatch.setattr finds it

import core.config as cfgmod
import core.fetch.onion.tor_proxy as tp
from core.config.settings import SearchConfig, TorSection
from core.fetch.onion import transport


# SOCKS resolution

# Patch the config loader to return a search config with the given Tor settings and drop the cached
# SOCKS resolution. These tests are offline: sockets and curl are faked, no Tor and no network are used.
def _use_cfg(monkeypatch, **tor_kw):
    cfg = SearchConfig(tor=TorSection(**tor_kw))
    monkeypatch.setattr(cfgmod, "load_search_config", lambda *a, **k: cfg)
    tp.reset()


# Verify that SOCKS resolution is a no-op returning None when the Tor feature is disabled.
def test_disabled_is_noop(monkeypatch):
    _use_cfg(monkeypatch, enabled=False)
    assert tp.resolve_socks() is None


# Verify that a running Tor daemon answering on port 9050 is reused.
def test_reuses_running_tor(monkeypatch):
    _use_cfg(monkeypatch, enabled=True)
    monkeypatch.setattr(
        tp,
        "_socks5_open",
        lambda url, timeout=1.0: urlparse(url).port == 9050,
    )
    assert tp.resolve_socks() == "socks5h://127.0.0.1:9050"


# Verify that the Tor Browser port 9150 is used when it is the only port answering.
def test_prefers_tor_browser_when_only_9150(monkeypatch):
    _use_cfg(monkeypatch, enabled=True)
    monkeypatch.setattr(
        tp,
        "_socks5_open",
        lambda url, timeout=1.0: urlparse(url).port == 9150,
    )
    assert tp.resolve_socks() == "socks5h://127.0.0.1:9150"


# Verify that an explicitly configured socks_url is used when it answers.
def test_explicit_socks_url_used_when_answering(monkeypatch):
    _use_cfg(monkeypatch, enabled=True, socks_url="socks5h://127.0.0.1:9999")
    monkeypatch.setattr(
        tp,
        "_socks5_open",
        lambda url, timeout=1.0: urlparse(url).port == 9999,
    )
    assert tp.resolve_socks() == "socks5h://127.0.0.1:9999"


# Verify that no running Tor leaves the feature disabled (None): Tor is never spawned.
def test_no_running_tor_keeps_feature_disabled(monkeypatch):
    _use_cfg(monkeypatch, enabled=True)
    monkeypatch.setattr(tp, "_socks5_open", lambda url, timeout=1.0: False)
    assert tp.resolve_socks() is None


# Verify that a resolved SOCKS URL is served from the cache without probing again, and that force=True
# probes again.
def test_resolution_is_cached_until_reset(monkeypatch):
    _use_cfg(monkeypatch, enabled=True)
    calls = {"n": 0}

    # Count the probes and accept only port 9050.
    def probe(url, timeout=1.0):
        calls["n"] += 1
        return urlparse(url).port == 9050

    monkeypatch.setattr(tp, "_socks5_open", probe)
    assert tp.resolve_socks() == "socks5h://127.0.0.1:9050"
    n_after_first = calls["n"]

    # The second call is served from the cache without a new probe.
    assert tp.resolve_socks() == "socks5h://127.0.0.1:9050"
    assert calls["n"] == n_after_first

    # force=True probes again.
    assert tp.resolve_socks(force=True) == "socks5h://127.0.0.1:9050"
    assert calls["n"] > n_after_first


# Verify that the probe accepts a listener only after it completes SOCKS5 method negotiation: the
# greeting is sent and the no-auth reply is accepted.
def test_socks5_probe_negotiates_protocol(monkeypatch):
    # Fake socket that answers the SOCKS5 greeting with a no-auth response.
    class FakeSocket:
        # Prepare a valid SOCKS5 no-auth response.
        def __init__(self):
            self.response = bytearray(b"\x05\x00")
            self.sent: list[bytes] = []

        # Enter the fake socket context.
        def __enter__(self):
            return self

        # Leave the fake socket context.
        def __exit__(self, exc_type, exc, traceback):
            return False

        # Accept the timeout selected for the local probe.
        def settimeout(self, timeout):
            self.timeout = timeout

        # Record the SOCKS5 greeting.
        def sendall(self, payload):
            self.sent.append(payload)

        # Return the requested response fragment.
        def recv(self, size):
            result = bytes(self.response[:size])
            del self.response[:size]
            return result

    connection = FakeSocket()
    monkeypatch.setattr(tp.socket, "create_connection", lambda endpoint, timeout: connection)

    assert tp._socks5_open("socks5h://127.0.0.1:9050") is True
    assert connection.sent == [b"\x05\x01\x00"]


# Verify that the probe rejects a TCP listener that does not negotiate SOCKS5.
def test_socks5_probe_rejects_plain_listener(monkeypatch):
    # Fake socket that answers the greeting with a non-SOCKS reply.
    class FakeSocket:
        # Enter the fake socket context.
        def __enter__(self):
            return self

        # Leave the fake socket context.
        def __exit__(self, exc_type, exc, traceback):
            return False

        # Accept the timeout selected for the local probe.
        def settimeout(self, timeout):
            self.timeout = timeout

        # Ignore the greeting written to the plain listener.
        def sendall(self, payload):
            return None

        # Return a non-SOCKS protocol response.
        def recv(self, size):
            return b"NO"[:size]

    monkeypatch.setattr(tp.socket, "create_connection", lambda endpoint, timeout: FakeSocket())

    assert tp._socks5_open("socks5h://127.0.0.1:9050") is False


# Transport

# Verify that fetching degrades to an "unavailable" result when no SOCKS proxy can be resolved.
def test_fetch_unavailable_when_no_socks(monkeypatch):
    monkeypatch.setattr(transport, "resolve_socks", lambda force=False: None)
    r = asyncio.run(transport.onion_fetch("http://x.onion/", timeout=5))
    assert r.status == "unavailable" and r.ok is False


# Verify that a fetch over a resolved SOCKS proxy returns the body and HTTP status of the faked response.
def test_fetch_ok_over_fake_tor(monkeypatch):
    monkeypatch.setattr(transport, "resolve_socks",
                        lambda force=False: "socks5h://127.0.0.1:9150")

    # Fake HTTP response with status 200 and a small body.
    class _Resp:
        status_code = 200
        text = "<html>onion body</html>"

    # Fake curl_cffi.requests module that always returns the canned response.
    class _Fake:
        # Return the canned response for any URL.
        def get(self, url, **kw):
            return _Resp()

    monkeypatch.setattr(curl_cffi, "requests", _Fake())
    r = asyncio.run(transport.onion_fetch("http://x.onion/", timeout=5))
    assert r.ok and r.status == "ok" and r.http_status == 200
    assert "onion body" in r.text


# Verify that a freshly reloaded tor_proxy module initializes _resolved to the _UNSET sentinel. If it
# were None, the very first resolve_socks() would return that cached None and never probe, and the
# onion layer would always report "unavailable" (regression guard).
def test_resolved_default_is_unset_not_none():
    import importlib
    importlib.reload(tp)
    assert tp._resolved is tp._UNSET
