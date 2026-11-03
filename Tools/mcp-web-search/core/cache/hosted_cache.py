# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

from core.cache.query_normalizer import normalize_query_key

logger = logging.getLogger("core.cache.hosted_cache")

# Default database file under the server root, and the version prefix of the cache key;
# bumping the version invalidates every previously stored entry. Shopping results are
# capped at a one hour lifetime because prices go stale quickly.
_CACHE_PATH = Path(__file__).resolve().parents[2] / "_cache" / "hosted_cache.db"
_CACHE_KEY_VERSION = "v4"
_SHOPPING_TTL_SECONDS = 60 * 60


# SQLite-backed cache for whole web_search result payloads, keyed by the normalized query plus
# the parameters that change the result (region, safesearch, timelimit, effort, shopping,
# academic). It uses one flat TTL for every query; empty or failed result sets get a short
# negative TTL so a broken query is not hammered at every retry.
class HostedSearchCache:
    # Store the TTL settings (clamped to at least one second), create the database directory
    # and the table. Connections are opened lazily, one per thread.
    def __init__(self, db_path: str, *, default_ttl: int = 21_600, negative_ttl: int = 300) -> None:
        self._db_path = db_path
        self._default_ttl = max(1, int(default_ttl))
        self._negative_ttl = max(1, int(negative_ttl))

        # Serializes writes; reads go through the per-thread connection in _local.
        self._write_lock = threading.Lock()
        self._local = threading.local()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # Per-thread persistent SQLite connection (WAL for concurrent reads).
    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10)
            conn.execute("PRAGMA journal_mode=WAL")

            # Cap the WAL file at 8 MiB: it is truncated back to that size after a checkpoint.
            conn.execute("PRAGMA journal_size_limit=8388608")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    # Create the hosted_cache table (key, JSON payload, store timestamp, per-row TTL) if missing.
    def _init_db(self) -> None:
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS hosted_cache (
                    key  TEXT    PRIMARY KEY,
                    data TEXT    NOT NULL,
                    ts   REAL    NOT NULL,
                    ttl  INTEGER NOT NULL
                )
                """
            )
            conn.commit()

    # Deterministic key over the normalized query and result-affecting parameters.
    #
    # Preserve query semantics. Only case, whitespace, and unambiguous technical spellings
    # are normalized; terms, operators, punctuation, repetition, and order remain intact.
    @staticmethod
    def make_key(
        query: str,
        *,
        region: str = "",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        effort: str = "low",
        shopping: bool = False,
        academic: bool = False,
    ) -> str:
        normalized = normalize_query_key(query)
        raw = (
            f"{_CACHE_KEY_VERSION}|{normalized}|{region}|{safesearch}|{timelimit or ''}|{effort}"
            f"|{int(bool(shopping))}|{int(bool(academic))}"
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    # Return a cached payload, or None when missing/expired.
    def get(
        self,
        query: str,
        *,
        region: str = "",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        effort: str = "low",
        shopping: bool = False,
        academic: bool = False,
    ) -> Optional[dict[str, Any]]:
        key = self.make_key(
            query, region=region, safesearch=safesearch, timelimit=timelimit,
            effort=effort, shopping=shopping, academic=academic,
        )

        # Look the entry up; a database error is a cache miss, never a failed search.
        try:
            row = self._get_conn().execute(
                "SELECT data, ts, ttl FROM hosted_cache WHERE key = ?", (key,)
            ).fetchone()
        except Exception as exc:  # noqa: BLE001
            logger.warning("[hosted_cache] read error: %s", exc)
            return None

        # Shopping lookups shorten the stored TTL to the shopping cap, then expiry is checked.
        ttl = min(int(row["ttl"]), _SHOPPING_TTL_SECONDS) if row is not None and shopping else (
            int(row["ttl"]) if row is not None else 0
        )
        if row is None or (time.time() - row["ts"]) > ttl:
            return None
        try:
            return json.loads(row["data"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("[hosted_cache] deserialize error for key %s: %s", key[:12], exc)
            return None

    # Store a payload. An empty result set gets the short negative TTL.
    def set(
        self,
        query: str,
        payload: dict[str, Any],
        *,
        region: str = "",
        safesearch: str = "moderate",
        timelimit: str | None = None,
        effort: str = "low",
        shopping: bool = False,
        academic: bool = False,
        is_empty: bool = False,
    ) -> None:
        # Pick the lifetime: negative for empty results, flat otherwise, capped for shopping.
        ttl = self._negative_ttl if is_empty else self._default_ttl
        if shopping and not is_empty:
            ttl = min(ttl, _SHOPPING_TTL_SECONDS)
        key = self.make_key(
            query, region=region, safesearch=safesearch, timelimit=timelimit,
            effort=effort, shopping=shopping, academic=academic,
        )

        # Serialize first; a payload that cannot be stored is skipped, not raised.
        try:
            data = json.dumps(payload, ensure_ascii=False, default=str)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[hosted_cache] serialize error: %s", exc)
            return

        # Write under the lock; a database error only loses this cache entry.
        with self._write_lock:
            try:
                conn = self._get_conn()
                conn.execute(
                    "INSERT OR REPLACE INTO hosted_cache (key, data, ts, ttl) VALUES (?, ?, ?, ?)",
                    (key, data, time.time(), ttl),
                )
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[hosted_cache] write error: %s", exc)

    # Delete expired rows and return the number removed (0 when the delete fails).
    def evict_expired(self) -> int:
        with self._write_lock:
            try:
                conn = self._get_conn()
                deleted = conn.execute(
                    "DELETE FROM hosted_cache WHERE (? - ts) > ttl", (time.time(),)
                ).rowcount
                conn.commit()
            except Exception as exc:  # noqa: BLE001
                logger.warning("[hosted_cache] evict error: %s", exc)
                return 0
        if deleted:
            logger.info("hosted_cache: evicted %d expired entries", deleted)
        return deleted


# Process-wide cache instance, created on first use under a lock (double-checked locking).
_singleton: Optional[HostedSearchCache] = None
_singleton_lock = threading.Lock()


# Return the lazily-initialised global HostedSearchCache, configured from search_config.
def get_hosted_cache() -> HostedSearchCache:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                from core.config import load_search_config

                cfg = load_search_config().cache
                _singleton = HostedSearchCache(
                    str(_CACHE_PATH),
                    default_ttl=cfg.search_ttl_seconds,
                    negative_ttl=cfg.search_negative_ttl_seconds,
                )
    return _singleton
