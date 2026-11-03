# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from pathlib import Path

from core.config import load_search_config

from .source_cache import CachedPage, SourceCache, canonicalize_url, content_hash, url_hash

# Server root is two levels up from this file (core/cache/__init__.py).
_CACHE_PATH = Path(__file__).resolve().parents[2] / "_cache" / "source_cache.db"

# Lazily created page cache shared by every caller of get_page_cache().
_page_cache: SourceCache | None = None


# Return the shared SourceCache singleton for read_page (decoupled from the search pipeline).
# The TTL comes from the page_ttl_seconds cache setting.
def get_page_cache() -> SourceCache:
    global _page_cache
    if _page_cache is None:
        ttl = int(load_search_config().cache.page_ttl_seconds)
        _page_cache = SourceCache(str(_CACHE_PATH), default_ttl=ttl)
    return _page_cache


__all__ = [
    "CachedPage",
    "SourceCache",
    "canonicalize_url",
    "content_hash",
    "get_page_cache",
    "url_hash",
]
