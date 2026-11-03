# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time

from .known_domains import domain_of, get_override
from .models import FetchAttempt, ProfileHint, ReputationSnapshot

logger = logging.getLogger("core.profiles.runtime")

# Observations older than this are ignored when recommending a method, so a site that
# changed its protection no longer pins read_page to a stale strategy.
_TTL_SECONDS = 14 * 86_400

# Recency half-life: a week-old success counts half as much as a fresh one.
_HALFLIFE_SECONDS = 7 * 86_400

# Below this many observations a recommendation is low-confidence (still usable as a hint).
_MIN_OBS_FOR_CONFIDENCE = 2

# A working method whose mean wall-clock fetch exceeds this is flagged avoid for fast paths.
_AVOID_FETCH_MS = 12_000.0

# A method that succeeds less than half the time is treated as unreliable.
_MIN_SUCCESS_RATE = 0.5

# Domain reputation (negative trust, earned at runtime).
# Anti-recursion contract (do not weaken):
#   * the table stores RAW event counters only — the penalty is computed at read time
#     and never stored, so a penalty can never compound on top of itself;
#   * events come exclusively from real fetch/parse outcomes — a source triage skipped
#     writes nothing, so a penalised domain is never re-penalised for being skipped;
#   * counters from a stale epoch (silent for _TTL_SECONDS) are reset on the next
#     write, and the snapshot ignores domains without FRESH failures — a domain nobody
#     re-observes simply ages out of the penalty set instead of rotting in it.
# Penalty tuning. TLS: failures count from the second one on (one broken handshake may be a
# transient network blip), each failure beyond the first adds _REP_TLS_STEP up to _REP_TLS_CAP.
# Empty parses: the ratio needs _REP_EMPTY_MIN_ATTEMPTS samples to mean anything and adds at most
# _REP_EMPTY_CAP. _REP_TOTAL_CAP is the hard ceiling for the combined penalty magnitude, a penalty
# below _REP_MIN_PENALTY leaves the domain out of the snapshot, and _REP_PROVEN_PARSES recent
# successful parses mark a domain proven.
_REP_TLS_MIN_FAILS = 2
_REP_TLS_STEP = 0.10
_REP_TLS_CAP = 0.20
_REP_EMPTY_MIN_ATTEMPTS = 3
_REP_EMPTY_CAP = 0.15
_REP_TOTAL_CAP = 0.25
_REP_MIN_PENALTY = 0.02
_REP_PROVEN_PARSES = 2

# SQLite schema: `domain_reputation` holds the raw trust counters per domain, `domain_methods`
# the rolling fetch statistics per (domain, method).
_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS domain_reputation (
    domain      TEXT PRIMARY KEY,
    tls_fail    INTEGER NOT NULL DEFAULT 0,
    parse_ok    INTEGER NOT NULL DEFAULT 0,
    parse_empty INTEGER NOT NULL DEFAULT 0,
    last_fail   REAL NOT NULL DEFAULT 0,
    last_ok     REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS domain_methods (
    domain            TEXT NOT NULL,
    method            TEXT NOT NULL,
    obs               INTEGER NOT NULL DEFAULT 0,
    success           INTEGER NOT NULL DEFAULT 0,
    blocked           INTEGER NOT NULL DEFAULT 0,
    timeout           INTEGER NOT NULL DEFAULT 0,
    empty             INTEGER NOT NULL DEFAULT 0,
    sum_fetch_ms      REAL NOT NULL DEFAULT 0,
    sum_parse_ms      REAL NOT NULL DEFAULT 0,
    sum_quality       REAL NOT NULL DEFAULT 0,
    last_user_agent   TEXT NOT NULL DEFAULT '',
    last_seen         REAL NOT NULL DEFAULT 0,
    PRIMARY KEY (domain, method)
);

CREATE INDEX IF NOT EXISTS idx_dm_domain ON domain_methods(domain);
"""


# Learns, per (domain, fetch-method), how reliable and how fast each approach is, so
# read_page can pick a known-good method up front instead of probing a fallback chain.
class RuntimeDomainProfiles:
    # Open the SQLite store and apply the schema.
    def __init__(self, db_path: str) -> None:
        self._db_path = db_path
        self._local = threading.local()
        self._write_lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    # Return a per-thread persistent SQLite connection.
    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10)
            conn.row_factory = sqlite3.Row

            # WAL mode; keep the -wal file at most 8 MiB (truncated after a checkpoint); wait up
            # to 5 s on a locked database.
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA journal_size_limit=8388608")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    # Apply schema DDL on first open.
    def _init_db(self) -> None:
        conn = self._get_conn()
        for stmt in _SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        conn.commit()

    # Record one fetch attempt against a domain, updating rolling per-method statistics.
    def record(self, url_or_domain: str, attempt: FetchAttempt) -> None:
        domain = domain_of(url_or_domain)
        if not domain or not attempt.method:
            return
        now = time.time()

        # Upsert the (domain, method) row: the first observation inserts it, later ones add
        # their counters and timings to the running sums. A database error is only logged.
        try:
            with self._write_lock:
                conn = self._get_conn()
                with conn:
                    conn.execute(
                        """
                        INSERT INTO domain_methods (
                            domain, method, obs, success, blocked, timeout, empty,
                            sum_fetch_ms, sum_parse_ms, sum_quality, last_user_agent, last_seen
                        )
                        VALUES (?, ?, 1, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(domain, method) DO UPDATE SET
                            obs             = obs + 1,
                            success         = success + excluded.success,
                            blocked         = blocked + excluded.blocked,
                            timeout         = timeout + excluded.timeout,
                            empty           = empty + excluded.empty,
                            sum_fetch_ms    = sum_fetch_ms + excluded.sum_fetch_ms,
                            sum_parse_ms    = sum_parse_ms + excluded.sum_parse_ms,
                            sum_quality     = sum_quality + excluded.sum_quality,
                            last_user_agent = excluded.last_user_agent,
                            last_seen       = excluded.last_seen
                        """,
                        (
                            domain,
                            attempt.method,
                            1 if attempt.success else 0,
                            1 if attempt.blocked else 0,
                            1 if attempt.timed_out else 0,
                            1 if attempt.empty else 0,
                            float(attempt.fetch_ms),
                            float(attempt.parse_ms),
                            float(attempt.quality),
                            attempt.user_agent or "",
                            now,
                        ),
                    )
        except sqlite3.DatabaseError as exc:
            logger.warning("runtime_profiles: record failed domain=%r: %s", domain, exc)

    # Record one trust observation for a domain. Raw counters only — the caller reports
    # what physically happened (TLS handshake failed / page parsed to content / parsed to
    # nothing); any penalty in force at evaluation time is deliberately not consulted.
    def record_reputation(
        self,
        url_or_domain: str,
        *,
        tls_failed: bool = False,
        parse_ok: bool = False,
        parse_empty: bool = False,
    ) -> None:
        domain = domain_of(url_or_domain)
        if not domain or not (tls_failed or parse_ok or parse_empty):
            return
        now = time.time()
        stale_before = now - _TTL_SECONDS
        failed = tls_failed or parse_empty

        # Upsert the domain row. Counters of a stale epoch (no event within the TTL) restart
        # from zero before the new event is added. A database error is only logged.
        try:
            with self._write_lock:
                conn = self._get_conn()
                with conn:
                    conn.execute(
                        """
                        INSERT INTO domain_reputation (
                            domain, tls_fail, parse_ok, parse_empty, last_fail, last_ok
                        )
                        VALUES (:domain, :tls, :ok, :empty, :last_fail, :last_ok)
                        ON CONFLICT(domain) DO UPDATE SET
                            tls_fail    = CASE WHEN MAX(last_fail, last_ok) < :stale
                                               THEN 0 ELSE tls_fail END + excluded.tls_fail,
                            parse_ok    = CASE WHEN MAX(last_fail, last_ok) < :stale
                                               THEN 0 ELSE parse_ok END + excluded.parse_ok,
                            parse_empty = CASE WHEN MAX(last_fail, last_ok) < :stale
                                               THEN 0 ELSE parse_empty END + excluded.parse_empty,
                            last_fail   = MAX(last_fail, excluded.last_fail),
                            last_ok     = MAX(last_ok, excluded.last_ok)
                        """,
                        {
                            "domain": domain,
                            "tls": 1 if tls_failed else 0,
                            "ok": 1 if parse_ok else 0,
                            "empty": 1 if parse_empty else 0,
                            "last_fail": now if failed else 0.0,
                            "last_ok": now if parse_ok else 0.0,
                            "stale": stale_before,
                        },
                    )
        except sqlite3.DatabaseError as exc:
            logger.warning("runtime_profiles: reputation record failed domain=%r: %s", domain, exc)

    # Compute the point-in-time trust snapshot for one search session. Penalties are
    # derived from raw counters on every call (never persisted), decayed by failure age,
    # and capped at _REP_TOTAL_CAP so a bad domain is nudged down, not executed.
    def reputation_snapshot(self) -> ReputationSnapshot:
        now = time.time()
        cutoff = now - _TTL_SECONDS

        # Load the domains with any event inside the TTL; on a database error return an empty
        # snapshot.
        try:
            rows = self._get_conn().execute(
                "SELECT * FROM domain_reputation WHERE MAX(last_fail, last_ok) > ?",
                (cutoff,),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.warning("runtime_profiles: reputation snapshot failed: %s", exc)
            return ReputationSnapshot(penalties={}, proven=frozenset())

        # Per domain: mark proven domains, then compute the decayed penalty from fresh failures.
        penalties: dict[str, float] = {}
        proven: set[str] = set()
        for row in rows:
            domain = str(row["domain"])
            tls_fail = int(row["tls_fail"] or 0)
            parse_ok = int(row["parse_ok"] or 0)
            parse_empty = int(row["parse_empty"] or 0)
            last_fail = float(row["last_fail"] or 0.0)
            last_ok = float(row["last_ok"] or 0.0)

            if parse_ok >= _REP_PROVEN_PARSES and last_ok > cutoff:
                proven.add(domain)

            # No fresh failures means no penalty, whatever the history says.
            if last_fail <= cutoff:
                continue

            penalty = 0.0

            # TLS: needs repeat failures AND no success since — a host that recovered
            # (or renewed its certificate) sheds the penalty on the first good fetch.
            if tls_fail >= _REP_TLS_MIN_FAILS and last_fail > last_ok:
                penalty += min(_REP_TLS_CAP, _REP_TLS_STEP * (tls_fail - 1))

            # Systematic empty parses: ratio-based so successes offset failures; scales
            # 0 → cap as the empty share climbs from ½ to 1.
            attempts = parse_ok + parse_empty
            if attempts >= _REP_EMPTY_MIN_ATTEMPTS:
                empty_share = parse_empty / attempts
                if empty_share > 0.5:
                    penalty += _REP_EMPTY_CAP * (empty_share - 0.5) * 2.0

            # Cap the total, fade it with the age of the last failure, and drop negligible ones.
            decay = 0.5 ** ((now - last_fail) / _HALFLIFE_SECONDS)
            penalty = min(_REP_TOTAL_CAP, penalty) * decay
            if penalty >= _REP_MIN_PENALTY:
                penalties[domain] = round(penalty, 4)
        return ReputationSnapshot(penalties=penalties, proven=frozenset(proven))

    # Recommend the best learned fetch method for a domain, or None when unknown.
    # A hard override from known_domains wins while runtime confidence is still low.
    def best_method(self, url_or_domain: str) -> ProfileHint | None:
        domain = domain_of(url_or_domain)
        if not domain:
            return None

        override = get_override(domain)

        # Load the learned statistics of the domain; a database error counts as no data.
        try:
            rows = self._get_conn().execute(
                "SELECT * FROM domain_methods WHERE domain = ?", (domain,)
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.warning("runtime_profiles: lookup failed domain=%r: %s", domain, exc)
            rows = []

        # Pick the best method among those that succeeded at least once within the TTL.
        now = time.time()
        best: ProfileHint | None = None
        best_rank: tuple[float, float] = (-1.0, 0.0)
        for row in rows:
            obs = int(row["obs"] or 0)
            success = int(row["success"] or 0)
            last_seen = float(row["last_seen"] or 0.0)
            if obs <= 0 or success <= 0:
                continue
            age = now - last_seen
            if age > _TTL_SECONDS:
                continue
            decay = 0.5 ** (age / _HALFLIFE_SECONDS)
            success_rate = success / obs
            avg_fetch_ms = float(row["sum_fetch_ms"] or 0.0) / obs
            avg_quality = float(row["sum_quality"] or 0.0) / obs
            confidence = success_rate * decay * min(1.0, obs / _MIN_OBS_FOR_CONFIDENCE)

            # Rank by reliability first, then prefer the cheaper method.
            rank = (round(success_rate * decay, 4), -avg_fetch_ms)
            if rank > best_rank:
                best_rank = rank
                best = ProfileHint(
                    method=str(row["method"]),
                    user_agent=str(row["last_user_agent"] or ""),
                    expected_fetch_ms=avg_fetch_ms,
                    expected_quality=avg_quality,
                    confidence=round(confidence, 4),
                    avoid=success_rate < _MIN_SUCCESS_RATE or avg_fetch_ms > _AVOID_FETCH_MS,
                )

        # Hard override takes precedence until runtime data is both present and confident.
        if override and override.required_method:
            if best is None or best.confidence < 0.5 or best.method != override.required_method:
                return ProfileHint(
                    method=override.required_method,
                    confidence=1.0,
                    avoid=False,
                )
        return best


# The shared store, created on first use by get_runtime_profiles().
_store: RuntimeDomainProfiles | None = None

from pathlib import Path  # noqa: E402

# Runtime DB lives beside the page cache under the server root's _cache/ directory.
_DB_PATH = Path(__file__).resolve().parents[2] / "_cache" / "domain_runtime.db"


# Return the shared RuntimeDomainProfiles singleton, creating it on first use.
def get_runtime_profiles() -> RuntimeDomainProfiles:
    global _store
    if _store is None:
        _store = RuntimeDomainProfiles(str(_DB_PATH))
    return _store
