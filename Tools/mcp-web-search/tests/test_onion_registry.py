# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import curl_cffi
import curl_cffi.requests  # noqa: F401  - bind the submodule so monkeypatch.setattr finds it

from core.fetch.onion import registry, resolver
from core.fetch.onion.models import OnionService


# Verify that the seed loads into vetted records: at least 11 services, each with a TLS clearnet trust
# anchor, an onion address and a category.
def test_seed_loads_vetted_services():
    svcs = registry.load_services()
    assert len(svcs) >= 11
    for s in svcs:
        # The trust anchor must be TLS.
        assert s.clearnet_anchor.startswith("https://")
        assert ".onion" in s.onion
        assert s.category


# Verify that service lookup by name works (unknown names give None) and that category lookup finds the
# media services.
def test_lookup_helpers():
    assert registry.service_for("torproject") is not None
    assert registry.service_for("NOPE") is None
    media = registry.services_in("media")
    assert {s.name for s in media} >= {"guardian", "rferl", "dw"}


# Demo service with a seeded onion address, used as the resolver input.
_SVC = OnionService(
    name="demo", category="media",
    clearnet_anchor="https://demo.example/",
    onion="http://seedaddr0000000000000000000000000000000000000000000000.onion/",
)


# Fake HTTP response that carries only headers.
class _Resp:
    # Store the response headers.
    def __init__(self, headers):
        self.headers = headers


# Fake curl_cffi.requests module that counts GET calls and answers with an optional Onion-Location header.
class _Fake:
    # Remember the Onion-Location value to advertise (None means the anchor advertises nothing).
    def __init__(self, header_value):
        self.calls = 0
        self._hv = header_value

    # Count the call and return a response with the configured Onion-Location header.
    def get(self, url, **kw):
        self.calls += 1
        return _Resp({"onion-location": self._hv} if self._hv else {})


# Verify that the resolver takes the fresh Onion-Location from the clearnet anchor and then serves it
# from the cache, so only one network request is made for two lookups.
def test_resolver_prefers_fresh_and_caches(monkeypatch):
    resolver.reset_cache()
    fresh = "http://freshaddr1111111111111111111111111111111111111111111.onion/"
    fake = _Fake(fresh)
    monkeypatch.setattr(curl_cffi, "requests", fake)

    # The first lookup refreshes from the anchor, the second one is served from the cache.
    assert resolver.resolve_onion(_SVC) == fresh
    assert resolver.resolve_onion(_SVC) == fresh
    assert fake.calls == 1


# Verify that the resolver falls back to the seeded address when the anchor advertises no Onion-Location.
def test_resolver_falls_back_to_seed(monkeypatch):
    resolver.reset_cache()

    # The anchor advertises nothing.
    fake = _Fake(None)
    monkeypatch.setattr(curl_cffi, "requests", fake)
    assert resolver.resolve_onion(_SVC) == _SVC.onion


# Verify that force=True bypasses the cache and queries the anchor again.
def test_resolver_force_refetches(monkeypatch):
    resolver.reset_cache()
    fresh = "http://addr222.onion/"
    fake = _Fake(fresh)
    monkeypatch.setattr(curl_cffi, "requests", fake)
    resolver.resolve_onion(_SVC)
    resolver.resolve_onion(_SVC, force=True)
    assert fake.calls == 2
