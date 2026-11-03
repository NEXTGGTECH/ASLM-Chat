# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import hashlib
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

logger = logging.getLogger("core.cache.source_cache")

# Lowercase fragments of sqlite3 error messages that mean the on-disk database file is corrupt.
_SQLITE_CORRUPTION_MARKERS = (
    "database disk image is malformed",
    "file is not a database",
    "malformed database schema",
)


# Return True when the exception is a sqlite3 database error that indicates a corrupt on-disk file.
def _is_sqlite_corruption(exc: BaseException) -> bool:
    if not isinstance(exc, sqlite3.DatabaseError):
        return False
    message = str(exc).lower()
    return any(marker in message for marker in _SQLITE_CORRUPTION_MARKERS)


from core.fetch.url_utils import _TRACKING_PARAMS, UnsafeFetchUrl, validate_public_fetch_url


# Normalize a URL for dedup: https scheme, no www prefix, no trailing slash, no fragment, and the
# query parameters filtered of tracking keys and sorted. An unparsable URL is returned stripped.
def canonicalize_url(url: str) -> str:
    try:
        p = urlparse(url.strip())
        scheme = (p.scheme or "https").lower()
        if scheme == "http":
            scheme = "https"
        netloc = p.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        path = p.path or "/"
        if path != "/" and path.endswith("/"):
            path = path.rstrip("/")

        # Drop tracking parameters and sort the rest so equivalent URLs hash identically.
        qp = parse_qs(p.query, keep_blank_values=False)
        filtered = {k: v for k, v in qp.items() if k.lower() not in _TRACKING_PARAMS}
        sorted_q = urlencode(sorted(filtered.items()), doseq=True)
        return urlunparse((scheme, netloc, path, "", sorted_q, ""))
    except Exception:
        # Not a parsable URL; fall back to the raw text so it still gets a stable key.
        return url.strip()


# Return the SHA-256 hex digest of the canonical form of the URL (the pages primary key).
def url_hash(url: str) -> str:
    return hashlib.sha256(canonicalize_url(url).encode()).hexdigest()


# Return the SHA-256 hex digest of the page text, used to detect changed content.
def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


# One cached page row. status is "ok", "error", "antibot" or "timeout"; score is the FTS5 BM25
# score and is only set by search_local.
@dataclass
class CachedPage:
    url: str
    domain: str
    title: str
    clean_text: str
    raw_html: str
    fetched_at: float
    content_hash: str
    status: str
    char_count: int
    score: float = 0.0


# DDL run on every open: the pages table with its indexes, the query to source link tables, and
# the FTS5 index (porter stemming over title and text) used by search_local.
_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS pages (
    url_hash     TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    domain       TEXT NOT NULL,
    title        TEXT DEFAULT '',
    clean_text   TEXT DEFAULT '',
    raw_html     TEXT DEFAULT '',
    fetched_at   REAL NOT NULL,
    content_hash TEXT DEFAULT '',
    status       TEXT DEFAULT 'ok',
    char_count   INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_pages_domain     ON pages(domain);
CREATE INDEX IF NOT EXISTS idx_pages_fetched_at ON pages(fetched_at);
CREATE INDEX IF NOT EXISTS idx_pages_status     ON pages(status);

CREATE TABLE IF NOT EXISTS query_sources (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    query    TEXT NOT NULL,
    url_hash TEXT NOT NULL,
    rank     INTEGER DEFAULT 0,
    found_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qs_query    ON query_sources(query);
CREATE INDEX IF NOT EXISTS idx_qs_url_hash ON query_sources(url_hash);

CREATE TABLE IF NOT EXISTS query_source_classes (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    query                TEXT NOT NULL,
    url_hash             TEXT NOT NULL,
    class_mix_json       TEXT DEFAULT '',
    content_classes_json TEXT DEFAULT '',
    snippet_score        REAL DEFAULT 0.0,
    parsed_score         REAL DEFAULT 0.0,
    recorded_at          REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_qsc_query    ON query_source_classes(query);
CREATE INDEX IF NOT EXISTS idx_qsc_url_hash ON query_source_classes(url_hash);

CREATE VIRTUAL TABLE IF NOT EXISTS pages_fts USING fts5(
    url_hash UNINDEXED,
    title,
    clean_text,
    tokenize='porter unicode61'
);
"""


# SQLite + FTS5 local page cache for cache-first retrieval. Read and write methods that hit a
# database error signalling a corrupt file quarantine it and start an empty cache, so a damaged
# cache degrades to cache misses instead of failing the search.
class SourceCache:
    # Store the database path and default freshness window (seconds), create the directory and
    # apply the schema. Connections are opened lazily, one per thread.
    def __init__(self, db_path: str, default_ttl: int = 86_400) -> None:
        self._db_path = db_path
        self._default_ttl = default_ttl

        # Per-thread connections, a lock that serializes writes, and a flag that stops
        # corruption recovery from re-entering itself while it recreates the database.
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._recovering_corrupt_db = False
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # Connection and recovery

    # Return a per-thread persistent SQLite connection.
    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = None
            try:
                conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL")

                # Cap the -wal sidecar at 32 MiB: after each checkpoint SQLite truncates it back
                # to this size instead of leaving it at its high-water mark (default -1 =
                # unbounded, which let _cache/*.db-wal grow without ever shrinking).
                conn.execute("PRAGMA journal_size_limit=33554432")
                conn.execute("PRAGMA busy_timeout=5000")
                self._local.conn = conn
            except sqlite3.DatabaseError:
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        # The connection is already unusable; the original error is re-raised below.
                        pass
                self._local.conn = None
                raise
        return conn

    # Close the current thread's SQLite connection.
    def _close_thread_conn(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                # Closing is best-effort; the reference is dropped below either way.
                pass
            self._local.conn = None

    # Quarantine corrupt DB files and recreate an empty cache. Return True when a fresh database
    # was created, False when the error is not corruption, recovery is already running, or the
    # recreation failed.
    def _recover_corrupt_db(self, exc: BaseException) -> bool:
        if self._recovering_corrupt_db or not _is_sqlite_corruption(exc):
            return False

        logger.warning("source_cache: recovering corrupt sqlite cache %s: %s", self._db_path, exc)
        self._recovering_corrupt_db = True
        try:
            self._close_thread_conn()

            # Rename the database and its WAL/SHM sidecars aside with a timestamp suffix; if a
            # rename fails, delete the file so the new database does not inherit the corruption.
            stamp = time.strftime("%Y%m%d%H%M%S")
            for suffix in ("", "-wal", "-shm"):
                path = f"{self._db_path}{suffix}"
                if not os.path.exists(path):
                    continue
                dst = f"{path}.corrupt-{stamp}"
                try:
                    os.replace(path, dst)
                    logger.warning("source_cache: quarantined corrupt cache file %s -> %s", path, dst)
                except OSError as move_exc:
                    logger.warning("source_cache: could not quarantine corrupt cache file %s: %s", path, move_exc)
                    try:
                        os.remove(path)
                        logger.warning("source_cache: removed corrupt cache file %s", path)
                    except OSError as remove_exc:
                        logger.warning("source_cache: could not remove corrupt cache file %s: %s", path, remove_exc)
            self._init_db()
        except sqlite3.DatabaseError as recreate_exc:
            logger.warning("source_cache: failed to recreate cache after corruption: %s", recreate_exc)
            return False
        finally:
            self._recovering_corrupt_db = False
        return True

    # Apply schema DDL on first open (or after corruption recovery).
    def _init_db(self) -> None:
        try:
            conn = self._get_conn()
            for stmt in _SCHEMA_SQL.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.commit()
        except sqlite3.DatabaseError as exc:
            if self._recover_corrupt_db(exc):
                return
            raise

    # Page storage and lookup

    # FTS5 + BM25 search over cached pages (status=ok, within freshness window). The best
    # matches (lowest BM25 score) come first. Return [] on an empty query or a database error.
    def search_local(
        self,
        query: str,
        limit: int = 10,
        min_freshness_sec: int | None = None,
    ) -> list[CachedPage]:
        if not query or not query.strip():
            return []

        max_age = min_freshness_sec or self._default_ttl
        cutoff = time.time() - max_age

        # Run the full-text query. A corrupt database is quarantined and treated as no results.
        try:
            rows = self._get_conn().execute(
                """
                SELECT p.url, p.domain, p.title, p.clean_text, p.raw_html,
                       p.fetched_at, p.content_hash, p.status, p.char_count,
                       bm25(pages_fts, 5.0, 1.0) AS score
                FROM pages_fts
                JOIN pages p ON pages_fts.url_hash = p.url_hash
                WHERE pages_fts MATCH ?
                  AND p.status = 'ok'
                  AND p.fetched_at > ?
                ORDER BY score
                LIMIT ?
                """,
                (query, cutoff, limit),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            if self._recover_corrupt_db(exc):
                return []
            logger.warning("FTS5 search failed for %r: %s", query, exc)
            return []

        # Map the result rows to CachedPage objects carrying their BM25 score.
        return [
            CachedPage(
                url=r["url"],
                domain=r["domain"],
                title=r["title"],
                clean_text=r["clean_text"],
                raw_html=r["raw_html"],
                fetched_at=r["fetched_at"],
                content_hash=r["content_hash"],
                status=r["status"],
                char_count=r["char_count"],
                score=r["score"],
            )
            for r in rows
        ]

    # Insert or update a page (pages + FTS5 in one transaction). Unsafe (non-public) URLs are
    # refused and never stored. The domain is derived from the URL when not given.
    # char_count is always the length of clean_text.
    def cache_page(
        self,
        url: str,
        title: str,
        clean_text: str,
        raw_html: str = "",
        domain: str = "",
        status: str = "ok",
    ) -> None:
        # Refuse URLs that fail the public-URL safety check (private hosts and similar).
        try:
            validate_public_fetch_url(url)
        except UnsafeFetchUrl as exc:
            logger.warning("source_cache: blocked unsafe cache write url=%r reason=%s", url, exc)
            return

        if not domain:
            try:
                domain = urlparse(url).netloc.lower()
            except Exception:
                # An unparsable URL is stored without a domain.
                domain = ""

        uhash = url_hash(url)
        chash = content_hash(clean_text) if clean_text else ""
        now = time.time()

        # Upsert the page row and replace its FTS5 row in one transaction under the write lock.
        try:
            with self._write_lock:
                conn = self._get_conn()
                with conn:
                    conn.execute(
                        """
                        INSERT INTO pages (url_hash, url, domain, title, clean_text, raw_html,
                                           fetched_at, content_hash, status, char_count)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(url_hash) DO UPDATE SET
                            title        = excluded.title,
                            clean_text   = excluded.clean_text,
                            raw_html     = excluded.raw_html,
                            fetched_at   = excluded.fetched_at,
                            content_hash = excluded.content_hash,
                            status       = excluded.status,
                            char_count   = excluded.char_count
                        """,
                        (uhash, canonicalize_url(url), domain, title, clean_text, raw_html,
                         now, chash, status, len(clean_text)),
                    )
                    conn.execute("DELETE FROM pages_fts WHERE url_hash = ?", (uhash,))
                    conn.execute(
                        "INSERT INTO pages_fts (url_hash, title, clean_text) VALUES (?, ?, ?)",
                        (uhash, title, clean_text),
                    )
        except sqlite3.DatabaseError as exc:
            if self._recover_corrupt_db(exc):
                return
            logger.warning("source_cache: write failed url=%r: %s", url, exc)

    # Return a cached page by URL, or None for an unsafe URL, a miss or a database error. The
    # page is returned whatever its status and age; callers check freshness themselves.
    def get_cached(self, url: str) -> Optional[CachedPage]:
        # Refuse URLs that fail the public-URL safety check, the same as on write.
        try:
            validate_public_fetch_url(url)
        except UnsafeFetchUrl as exc:
            logger.warning("source_cache: blocked unsafe cache read url=%r reason=%s", url, exc)
            return None

        # Look the page up by the hash of its canonical URL.
        uhash = url_hash(url)
        try:
            r = self._get_conn().execute(
                "SELECT * FROM pages WHERE url_hash = ?", (uhash,)
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            if self._recover_corrupt_db(exc):
                return None
            logger.warning("source_cache: read failed url=%r: %s", url, exc)
            return None

        if not r:
            return None
        return CachedPage(
            url=r["url"],
            domain=r["domain"],
            title=r["title"],
            clean_text=r["clean_text"],
            raw_html=r["raw_html"],
            fetched_at=r["fetched_at"],
            content_hash=r["content_hash"],
            status=r["status"],
            char_count=r["char_count"],
        )

    # Return True when a cached ok page exists and is younger than max_age_sec (default TTL if unset).
    def is_fresh(self, url: str, max_age_sec: int | None = None) -> bool:
        try:
            validate_public_fetch_url(url)
        except UnsafeFetchUrl:
            return False

        uhash = url_hash(url)
        ttl = max_age_sec or self._default_ttl
        cutoff = time.time() - ttl

        try:
            r = self._get_conn().execute(
                "SELECT fetched_at FROM pages WHERE url_hash = ? AND status = 'ok'",
                (uhash,),
            ).fetchone()
        except sqlite3.DatabaseError as exc:
            if self._recover_corrupt_db(exc):
                return False
            logger.warning("source_cache: freshness check failed url=%r: %s", url, exc)
            return False
        return r is not None and r["fetched_at"] > cutoff

    # Query records and maintenance

    # Associate a query with a source URL for stats and future ranking.
    def record_query_source(self, query: str, url: str, rank: int = 0) -> None:
        uhash = url_hash(url)
        try:
            with self._write_lock:
                conn = self._get_conn()
                with conn:
                    conn.execute(
                        "INSERT INTO query_sources (query, url_hash, rank, found_at) VALUES (?, ?, ?, ?)",
                        (query, uhash, rank, time.time()),
                    )
        except sqlite3.DatabaseError as exc:
            if self._recover_corrupt_db(exc):
                return
            logger.warning("source_cache: query source write failed query=%r url=%r: %s", query, url, exc)

    # Attach class/relevance metadata (class mix and content classes as JSON text, plus snippet
    # and parsed relevance scores) to a query-source observation by inserting a new row.
    def record_query_source_classes(
        self,
        query: str,
        url: str,
        *,
        class_mix_json: str = "",
        content_classes_json: str = "",
        snippet_score: float = 0.0,
        parsed_score: float = 0.0,
    ) -> None:
        uhash = url_hash(url)

        # Insert the row under the write lock; scores are coerced to float with None as 0.0.
        try:
            with self._write_lock:
                conn = self._get_conn()
                with conn:
                    conn.execute(
                        """
                        INSERT INTO query_source_classes (
                            query, url_hash, class_mix_json, content_classes_json,
                            snippet_score, parsed_score, recorded_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            query,
                            uhash,
                            class_mix_json,
                            content_classes_json,
                            float(snippet_score or 0.0),
                            float(parsed_score or 0.0),
                            time.time(),
                        ),
                    )
        except sqlite3.DatabaseError as exc:
            if self._recover_corrupt_db(exc):
                return
            logger.warning("source_cache: query source class write failed query=%r url=%r: %s", query, url, exc)

    # Return the total number of cached pages, whatever their status.
    def page_count(self) -> int:
        r = self._get_conn().execute("SELECT COUNT(*) FROM pages").fetchone()
        return int(r[0]) if r else 0

    # Basic cache statistics (totals and distinct queries).
    def cache_stats(self) -> dict:
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) FROM pages").fetchone()[0] or 0
        ok = conn.execute("SELECT COUNT(*) FROM pages WHERE status = 'ok'").fetchone()[0] or 0
        queries = conn.execute("SELECT COUNT(DISTINCT query) FROM query_sources").fetchone()[0] or 0
        return {"total_pages": total, "ok_pages": ok, "unique_queries": queries}

    # Delete pages older than max_age_sec (default TTL if unset) together with their FTS5 rows
    # and query_sources links; return the number of deleted pages.
    def evict_stale(self, max_age_sec: int | None = None) -> int:
        ttl = max_age_sec or self._default_ttl
        cutoff = time.time() - ttl

        # Remove dependent rows first, then the pages, in a single transaction.
        with self._write_lock:
            conn = self._get_conn()
            with conn:
                stale_hashes = [
                    r["url_hash"] for r in conn.execute(
                        "SELECT url_hash FROM pages WHERE fetched_at < ?", (cutoff,)
                    ).fetchall()
                ]
                for uhash in stale_hashes:
                    conn.execute("DELETE FROM pages_fts WHERE url_hash = ?", (uhash,))
                    conn.execute("DELETE FROM query_sources WHERE url_hash = ?", (uhash,))
                result = conn.execute(
                    "DELETE FROM pages WHERE fetched_at < ?", (cutoff,)
                )
                deleted = result.rowcount
        if deleted:
            logger.info("source_cache: evicted %d stale pages (ttl=%ds)", deleted, ttl)
        return deleted
