# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import time
import hashlib

from core.cache import hosted_cache as hosted_cache_module
from core.cache.hosted_cache import HostedSearchCache
from core.search.recent_tracker import RecentSearchTracker
from core.search.web_search import _infer_pdf_url


# Hosted cache: query-results cache

# Verify that a payload stored in the hosted cache can be read back with the same query and effort.
def test_hosted_cache_roundtrip(tmp_path):
    cache = HostedSearchCache(str(tmp_path / "h.db"), default_ttl=100, negative_ttl=5)
    payload = {"query": "q", "sources": [{"url": "https://a.com"}]}
    cache.set("python asyncio", payload, effort="low")
    got = cache.get("python asyncio", effort="low")
    assert got is not None and got["sources"][0]["url"] == "https://a.com"


# Verify that the cache key depends on the effort and region parameters: a different effort or region
# misses, the same parameters hit.
def test_hosted_cache_key_is_param_sensitive(tmp_path):
    cache = HostedSearchCache(str(tmp_path / "h.db"))
    cache.set("q", {"x": 1}, effort="low")
    assert cache.get("q", effort="medium") is None
    assert cache.get("q", region="ru-ru", effort="low") is None
    assert cache.get("q", effort="low") == {"x": 1}


# Verify that harmless case and whitespace differences in the query still share one cache key.
def test_hosted_cache_normalizes_query(tmp_path):
    cache = HostedSearchCache(str(tmp_path / "h.db"))
    cache.set("  Python   asyncio  ", {"hit": True}, effort="low")
    assert cache.get("python asyncio", effort="low") == {"hit": True}


# Verify that queries differing by a semantic particle (not, OR/NOT/AND) or by word order do not share a
# cache entry.
def test_hosted_cache_preserves_semantic_particles_and_word_order(tmp_path):
    cache = HostedSearchCache(str(tmp_path / "meaning.db"))
    cache.set("allow telemetry", {"query": "positive"}, effort="low")
    cache.set("alpha OR beta", {"query": "or"}, effort="low")
    cache.set("dog bites man", {"query": "first-order"}, effort="low")

    assert cache.get("not allow telemetry", effort="low") is None
    assert cache.get("alpha NOT beta", effort="low") is None
    assert cache.get("alpha AND beta", effort="low") is None
    assert cache.get("man bites dog", effort="low") is None


# Verify that the current cache key differs from the key a legacy (unversioned) layout would produce, so
# legacy entries are never reused.
def test_hosted_cache_key_version_does_not_reuse_legacy_entries():
    legacy_raw = "alpha beta||moderate||low|0|0"
    legacy_key = hashlib.sha256(legacy_raw.encode()).hexdigest()

    assert HostedSearchCache.make_key("alpha beta", effort="low") != legacy_key


# Verify that an entry is served before its default TTL (1 s) elapses and is gone afterwards.
def test_hosted_cache_expiry(tmp_path):
    cache = HostedSearchCache(str(tmp_path / "h.db"), default_ttl=1)
    cache.set("q", {"x": 1}, effort="low")
    assert cache.get("q", effort="low") is not None
    time.sleep(1.2)
    assert cache.get("q", effort="low") is None


# Verify that an empty result is cached with the short negative TTL (1 s): it is served at first and
# expires long before the default TTL.
def test_hosted_cache_negative_ttl_for_empty(tmp_path):
    cache = HostedSearchCache(str(tmp_path / "h.db"), default_ttl=1000, negative_ttl=1)
    cache.set("q", {"sources": []}, effort="low", is_empty=True)
    assert cache.get("q", effort="low") is not None
    time.sleep(1.2)
    assert cache.get("q", effort="low") is None


# Verify that a shopping payload expires one hour after it was stored, even with a default TTL of six
# hours: still served at 3599 s, gone at 3601 s (time is faked).
def test_shopping_payload_cache_expires_after_one_hour(tmp_path, monkeypatch):
    now = 1000.0
    monkeypatch.setattr(hosted_cache_module.time, "time", lambda: now)
    cache = HostedSearchCache(str(tmp_path / "shopping.db"), default_ttl=21_600)
    cache.set("headphones", {"sources": [{"price": 100}]}, shopping=True)

    now = 4599.0
    assert cache.get("headphones", shopping=True) is not None
    now = 4601.0
    assert cache.get("headphones", shopping=True) is None


# Verify that evict_expired removes the single expired entry and reports a count of one.
def test_hosted_cache_evict_expired(tmp_path):
    cache = HostedSearchCache(str(tmp_path / "h.db"), default_ttl=1)
    cache.set("q", {"x": 1}, effort="low")
    time.sleep(1.2)
    assert cache.evict_expired() == 1


# Recent tracker: repeat block and source suppression

# Verify that a recorded query is reported as a repeat with a small age inside the window, and as
# not seen before anything was recorded.
def test_repeat_block_within_window():
    t = RecentSearchTracker()
    key = t.query_key("python asyncio", effort="low")

    # Nothing is recorded yet.
    assert t.repeat_age(key, window=30) is None
    t.record(key, ["https://a.com"])
    age = t.repeat_age(key, window=30)
    assert age is not None and age < 1.0


# Verify that a zero-length window never reports a repeat, even for a query that was just recorded.
def test_repeat_block_outside_window():
    t = RecentSearchTracker()
    key = t.query_key("q", effort="low")
    t.record(key, [])
    assert t.repeat_age(key, window=0) is None


# Verify that recently seen source URLs are reported while URLs never recorded are not.
def test_source_suppression():
    t = RecentSearchTracker()
    key = t.query_key("q1", effort="low")
    t.record(key, ["https://a.com/x", "https://b.com/y"])
    seen = t.recently_seen(["https://a.com/x", "https://c.com/z"], window=30)
    assert "https://a.com/x" in seen and "https://c.com/z" not in seen


# Verify that the same URL with www, scheme and trailing slash variations is still recognised as seen.
def test_source_suppression_canonicalizes():
    t = RecentSearchTracker()
    t.record(t.query_key("q", effort="low"), ["https://www.a.com/x/"])
    seen = t.recently_seen(["http://a.com/x"], window=30)
    assert "http://a.com/x" in seen


# PDF inference

# Verify that an arXiv abstract URL maps to its PDF URL, a .pdf URL is returned as is, and other pages
# or an empty URL give an empty string.
def test_infer_pdf_url():
    assert _infer_pdf_url("https://arxiv.org/abs/1706.03762") == "https://arxiv.org/pdf/1706.03762"
    assert _infer_pdf_url("https://example.com/paper.pdf") == "https://example.com/paper.pdf"
    assert _infer_pdf_url("https://example.com/page") == ""
    assert _infer_pdf_url("") == ""


# Operator queries

# Verify that queries with different search operators (-site:, quotes) or a different word order never
# share a cache entry, while the exact same operator query still hits (#2).
def test_operator_queries_get_distinct_cache_keys(tmp_path):
    cache = HostedSearchCache(str(tmp_path / "ops.db"), default_ttl=100, negative_ttl=5)
    cache.set("python site:github.com", {"sources": [{"url": "https://a"}]}, effort="low")

    # A differently-meaning operator query must not hit the first one's entry.
    assert cache.get("python -site:github.com", effort="low") is None
    assert cache.get('python "exact"', effort="low") is None

    # The exact same operator query still hits.
    assert cache.get("python site:github.com", effort="low") is not None

    # Word order is semantic even without explicit operators.
    cache.set("python asyncio guide", {"sources": [{"url": "https://b"}]}, effort="low")
    assert cache.get("guide asyncio python", effort="low") is None
