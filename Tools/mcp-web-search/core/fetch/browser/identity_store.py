# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("core.fetch.browser.identity")

# How many generations to retain per family. Older ones are pruned, but the most
# recent good generation is always preserved as a rotation fallback.
_MAX_GENERATIONS = 5

# Sliding TTL for session cookies (those with no Max-Age/Expires). A real browser keeps
# them only for the session; we keep them across searches for continuity but let them age
# out if not refreshed, so a stale session token isn't replayed forever. The clock is each
# cookie's `updated` time, so any re-capture refreshes it.
_SESSION_COOKIE_TTL = 6 * 3600.0

# SQLite schema, executed statement by statement on startup: browser storageState generations
# per family, cookies earned over HTTP per owner engine, and cookies imported from the user's
# real browser per family.
_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS identity_generations (
    family      TEXT    NOT NULL,
    generation  INTEGER NOT NULL,
    state       TEXT    NOT NULL,
    good        INTEGER NOT NULL DEFAULT 1,
    created     REAL    NOT NULL,
    PRIMARY KEY (family, generation)
);

CREATE INDEX IF NOT EXISTS idx_ident_family ON identity_generations(family);

CREATE TABLE IF NOT EXISTS http_cookies (
    owner    TEXT    NOT NULL,
    domain   TEXT    NOT NULL,
    name     TEXT    NOT NULL,
    value    TEXT    NOT NULL,
    expires  REAL    NOT NULL DEFAULT 0,
    updated  REAL    NOT NULL,
    PRIMARY KEY (owner, domain, name)
);

CREATE INDEX IF NOT EXISTS idx_http_cookies_owner ON http_cookies(owner);

CREATE TABLE IF NOT EXISTS imported_cookies (
    family   TEXT    NOT NULL,
    domain   TEXT    NOT NULL,
    name     TEXT    NOT NULL,
    value    TEXT    NOT NULL,
    path     TEXT    NOT NULL DEFAULT '/',
    expires  REAL    NOT NULL DEFAULT 0,
    secure   INTEGER NOT NULL DEFAULT 0,
    source   TEXT    NOT NULL DEFAULT '',
    imported REAL    NOT NULL,
    PRIMARY KEY (family, domain, name)
);

CREATE INDEX IF NOT EXISTS idx_imported_family ON imported_cookies(family);
"""


# True when a stored cookie domain applies to the given host (lenient suffix match).
def _domain_matches(cookie_domain: str, host: str) -> bool:
    cd = (cookie_domain or "").lstrip(".").lower()
    h = (host or "").lower()
    if not cd or not h:
        return False
    return h == cd or h.endswith("." + cd) or cd.endswith("." + h)


# Persistent, family-keyed browser identity store: the earned browser state (Playwright
# storageState with cookies, localStorage and IndexedDB snapshot) kept in SQLite, layered over
# the generated, non-stored static personality (UA, platform, locale).
# - The warm daemon is the primary writer (idle and recycle checkpoints); engine request builders
#   read locally on the hot path with no IPC; MCP-side Set-Cookie capture writes under the same lock.
# - The N most recent good generations are kept per family. A memory or age recycle restores the
#   latest; a captcha burn rotates to an older good generation (a poisoned state is never the only
#   backup), falling back to None (the seed) when none remain.
# - Cookies are family-bound: callers ask for a family's cookies and replay them only through that
#   family's transport, because mixing families is an anti-bot signal.
class IdentityStore:
    # Open (creating if needed) the SQLite database at db_path and ensure the schema exists.
    # At least one generation is always retained per family.
    def __init__(self, db_path: str, *, max_generations: int = _MAX_GENERATIONS) -> None:
        self._db_path = db_path
        self._max_generations = max(1, int(max_generations))

        # Each thread gets its own connection; the lock serializes all writers.
        self._local = threading.local()
        self._write_lock = threading.Lock()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # Return the per-thread persistent SQLite connection (WAL; readers never block the writer).
    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")

            # Cap the WAL file at 8 MiB so it is truncated after a checkpoint.
            conn.execute("PRAGMA journal_size_limit=8388608")
            conn.execute("PRAGMA busy_timeout=5000")
            self._local.conn = conn
        return conn

    # Create the tables and indexes by running each statement of the schema script.
    def _init_db(self) -> None:
        with self._write_lock:
            conn = self._get_conn()
            for stmt in _SCHEMA_SQL.strip().split(";"):
                if stmt.strip():
                    conn.execute(stmt)
            conn.commit()

    # Persist a new generation of a family's storageState; returns its generation number.
    # good=False marks a checkpoint that should not be restored as-is (e.g. burned).
    def checkpoint(self, family: str, state: dict[str, Any], *, good: bool = True) -> int:
        family = (family or "").strip() or "default"

        # Serialize the state; a state that cannot be encoded is skipped (-1) instead of raising.
        try:
            blob = json.dumps(state, ensure_ascii=False, default=str)
        except (TypeError, ValueError) as exc:
            logger.warning("identity_store: cannot serialise state for %r: %s", family, exc)
            return -1
        now = time.time()

        # Insert the next generation number and prune old ones in a single transaction.
        with self._write_lock:
            conn = self._get_conn()
            try:
                with conn:
                    row = conn.execute(
                        "SELECT COALESCE(MAX(generation), 0) AS g FROM identity_generations WHERE family = ?",
                        (family,),
                    ).fetchone()
                    generation = int(row["g"]) + 1
                    conn.execute(
                        "INSERT INTO identity_generations (family, generation, state, good, created) "
                        "VALUES (?, ?, ?, ?, ?)",
                        (family, generation, blob, 1 if good else 0, now),
                    )
                    self._prune(conn, family)
                return generation
            except sqlite3.DatabaseError as exc:
                logger.warning("identity_store: checkpoint failed for %r: %s", family, exc)
                return -1

    # Keep the newest N generations, but never drop the most-recent good one.
    def _prune(self, conn: sqlite3.Connection, family: str) -> None:
        rows = conn.execute(
            "SELECT generation, good FROM identity_generations WHERE family = ? ORDER BY generation DESC",
            (family,),
        ).fetchall()
        if len(rows) <= self._max_generations:
            return
        keep = {int(r["generation"]) for r in rows[: self._max_generations]}
        newest_good = next((int(r["generation"]) for r in rows if int(r["good"]) == 1), None)
        if newest_good is not None:
            keep.add(newest_good)
        doomed = [int(r["generation"]) for r in rows if int(r["generation"]) not in keep]
        if doomed:
            conn.executemany(
                "DELETE FROM identity_generations WHERE family = ? AND generation = ?",
                [(family, g) for g in doomed],
            )

    # Return the most recent generation of a family as a parsed storageState dict; with
    # good_only, burned generations are skipped. None when there is none or it is unreadable.
    def _fetch_state(self, family: str, *, good_only: bool) -> Optional[dict[str, Any]]:
        family = (family or "").strip() or "default"
        sql = "SELECT state FROM identity_generations WHERE family = ?"
        if good_only:
            sql += " AND good = 1"
        sql += " ORDER BY generation DESC LIMIT 1"
        try:
            row = self._get_conn().execute(sql, (family,)).fetchone()
        except sqlite3.DatabaseError as exc:
            logger.warning("identity_store: read failed for %r: %s", family, exc)
            return None
        if row is None:
            return None
        try:
            return json.loads(row["state"])
        except (TypeError, ValueError):
            return None

    # Latest good storageState to seed a fresh browser on start / memory recycle.
    def latest_good(self, family: str) -> Optional[dict[str, Any]]:
        return self._fetch_state(family, good_only=True)

    # Latest storageState regardless of good flag (diagnostics / forced restore).
    def latest(self, family: str) -> Optional[dict[str, Any]]:
        return self._fetch_state(family, good_only=False)

    # Burn the newest generation and return an older good one to rotate to (or None → seed).
    # Used on a captcha/burn recycle so a poisoned identity is not restored.
    def rotate(self, family: str) -> Optional[dict[str, Any]]:
        family = (family or "").strip() or "default"
        with self._write_lock:
            conn = self._get_conn()
            try:
                with conn:
                    top = conn.execute(
                        "SELECT generation FROM identity_generations WHERE family = ? "
                        "ORDER BY generation DESC LIMIT 1",
                        (family,),
                    ).fetchone()
                    if top is not None:
                        conn.execute(
                            "UPDATE identity_generations SET good = 0 WHERE family = ? AND generation = ?",
                            (family, int(top["generation"])),
                        )
            except sqlite3.DatabaseError as exc:
                logger.warning("identity_store: rotate failed for %r: %s", family, exc)
                return None
        return self.latest_good(family)

    # Return the cookies from the family's latest good state, optionally narrowed to a host.
    def cookies_for(self, family: str, *, host: str = "") -> list[dict[str, Any]]:
        state = self.latest_good(family)
        if not state:
            return []
        cookies = state.get("cookies") or []
        if not host:
            return list(cookies)
        return [c for c in cookies if _domain_matches(str(c.get("domain", "")), host)]

    # Return a ready "k=v; k2=v2" Cookie header for a host from the family's identity, or "".
    def cookie_header_for(self, family: str, host: str) -> str:
        pairs = [
            f"{c.get('name')}={c.get('value')}"
            for c in self.cookies_for(family, host=host)
            if c.get("name")
        ]
        return "; ".join(pairs)

    # HTTP-side cookie accumulation

    # Merge captured Set-Cookie header lines for a host into the owner's cookie history.
    # A cookie with Max-Age<=0 (a deletion) removes the stored entry instead of adding it.
    # This layer is separate from the browser storageState generations above: these are cookies
    # an HTTP SERP engine earns itself (consent, region, session), keyed by the engine that owns
    # the identity (e.g. "startpage"). The transport replays them on later requests and writes
    # back captured Set-Cookie, so a logical engine accumulates one coherent cookie history
    # across searches and restarts, the same continuity the warm browser gives, over HTTP.
    def merge_set_cookie(self, owner: str, host: str, set_cookie_headers: list[str]) -> None:
        owner = (owner or "").strip()
        if not owner or not set_cookie_headers:
            return
        now = time.time()
        upserts: list[tuple[str, str, str, str, float, float]] = []
        deletes: list[tuple[str, str, str]] = []

        # Parse every header into upserts (with an absolute expiry) or deletes (Max-Age<=0).
        for raw in set_cookie_headers:
            try:
                jar: SimpleCookie = SimpleCookie()
                jar.load(raw)
            except Exception:  # noqa: BLE001 — a malformed Set-Cookie must not break capture
                continue
            for name, morsel in jar.items():
                domain = (morsel["domain"] or host).lstrip(".").lower() or host.lower()
                max_age = str(morsel["max-age"] or "").strip()
                if max_age:
                    try:
                        age = float(max_age)
                    except ValueError:
                        age = None
                    if age is not None and age <= 0:
                        deletes.append((owner, domain, name))
                        continue
                    expires = now + age if age is not None else 0.0
                else:
                    # Session cookie: expires 0 means it ages out by the sliding session TTL.
                    expires = 0.0
                upserts.append((owner, domain, name, morsel.value, expires, now))
        if not upserts and not deletes:
            return

        # Apply the deletes and upserts in one transaction.
        with self._write_lock:
            conn = self._get_conn()
            try:
                with conn:
                    if deletes:
                        conn.executemany(
                            "DELETE FROM http_cookies WHERE owner = ? AND domain = ? AND name = ?",
                            deletes,
                        )
                    if upserts:
                        conn.executemany(
                            "INSERT OR REPLACE INTO http_cookies "
                            "(owner, domain, name, value, expires, updated) VALUES (?, ?, ?, ?, ?, ?)",
                            upserts,
                        )
            except sqlite3.DatabaseError as exc:
                logger.warning("identity_store: cookie merge failed for %r: %s", owner, exc)

    # Return the owner's non-expired cookies for a host as {name: value} (newest write wins).
    def http_cookies_map(self, owner: str, host: str) -> dict[str, str]:
        owner = (owner or "").strip()
        if not owner or not host:
            return {}
        now = time.time()
        try:
            rows = self._get_conn().execute(
                "SELECT domain, name, value, expires, updated FROM http_cookies WHERE owner = ? "
                "ORDER BY updated ASC",
                (owner,),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.warning("identity_store: cookie read failed for %r: %s", owner, exc)
            return {}
        out: dict[str, str] = {}
        for row in rows:
            expires = row["expires"]
            if expires:
                # An explicit expiry that has passed.
                if expires < now:
                    continue
            elif (now - float(row["updated"])) > _SESSION_COOKIE_TTL:
                # A session cookie that was not refreshed within the sliding TTL.
                continue
            if _domain_matches(str(row["domain"]), host):
                out[str(row["name"])] = str(row["value"])
        return out

    # Return a ready "k=v; ..." Cookie header for a host from the owner's HTTP cookie history.
    def http_cookie_header(self, owner: str, host: str) -> str:
        return "; ".join(f"{n}={v}" for n, v in self.http_cookies_map(owner, host).items())

    # Drop the owner's HTTP-earned cookies whose domain matches the host. Captcha-burn
    # recovery: a BLOCKED SERP is usually pinned to the engine's sticky session cookie,
    # so a retry must clear the branch instead of replaying it. Returns rows removed.
    def clear_http_cookies(self, owner: str, host: str) -> int:
        owner = (owner or "").strip()
        if not owner or not host:
            return 0

        # Read the owner's cookies and pick those whose domain matches the host.
        try:
            rows = self._get_conn().execute(
                "SELECT domain, name FROM http_cookies WHERE owner = ?", (owner,)
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.warning("identity_store: cookie clear-read failed for %r: %s", owner, exc)
            return 0
        doomed = [
            (owner, str(row["domain"]), str(row["name"]))
            for row in rows
            if _domain_matches(str(row["domain"]), host)
        ]
        if not doomed:
            return 0

        # Delete them in one transaction and report how many were removed.
        with self._write_lock:
            conn = self._get_conn()
            try:
                with conn:
                    conn.executemany(
                        "DELETE FROM http_cookies WHERE owner = ? AND domain = ? AND name = ?",
                        doomed,
                    )
            except sqlite3.DatabaseError as exc:
                logger.warning("identity_store: cookie clear failed for %r: %s", owner, exc)
                return 0
        return len(doomed)

    # Imported browser cookies

    # Replace (or, with replace=False, merge) a family's imported cookies and return how many
    # were stored. `cookies` is a list of dicts/objects exposing domain/name/value(/path/expires/
    # secure). Within one batch the FIRST occurrence of a (domain,name) wins, so callers pass
    # default-browser cookies first. This is a third, family-keyed layer, distinct from the
    # rotated warm-browser generations and the self-earned per-engine http_cookies: the cookies
    # are imported wholesale from the user's REAL browser (chromium from Chrome/Edge/Brave,
    # firefox from Firefox), replayed by the transport for that family and used to seed the warm
    # browser. A re-import replaces the family's set so stale sessions do not accumulate, and
    # disabling the feature clears it.
    def import_cookies(
        self, family: str, cookies: list[Any], *, source: str = "", replace: bool = True
    ) -> int:
        family = (family or "").strip() or "default"
        now = time.time()
        seen: set[tuple[str, str]] = set()
        rows: list[tuple[str, str, str, str, str, float, int, str, float]] = []

        # Normalize the cookies into rows, skipping nameless ones and duplicates.
        for c in cookies:
            domain = str(_attr(c, "domain") or "").strip()
            name = str(_attr(c, "name") or "").strip()
            if not domain or not name:
                continue
            dedup_key = (domain.lstrip(".").lower(), name)
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            rows.append(
                (
                    family,
                    domain,
                    name,
                    str(_attr(c, "value") or ""),
                    str(_attr(c, "path") or "/") or "/",
                    float(_attr(c, "expires") or 0.0),
                    1 if _attr(c, "secure") else 0,
                    source,
                    now,
                )
            )

        # Write the batch in one transaction, clearing the family's old set first when replacing.
        with self._write_lock:
            conn = self._get_conn()
            try:
                with conn:
                    if replace:
                        conn.execute("DELETE FROM imported_cookies WHERE family = ?", (family,))
                    if rows:
                        conn.executemany(
                            "INSERT OR REPLACE INTO imported_cookies "
                            "(family, domain, name, value, path, expires, secure, source, imported) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            rows,
                        )
            except sqlite3.DatabaseError as exc:
                logger.warning("identity_store: import_cookies failed for %r: %s", family, exc)
                return 0
        return len(rows)

    # Return the family's non-expired imported cookies, host-filtered, in storageState cookie
    # shape (name/value/domain/path/secure) for seeding a Playwright context.
    def imported_cookies_for(self, family: str, *, host: str = "") -> list[dict[str, Any]]:
        rows = self._imported_rows(family)
        out: list[dict[str, Any]] = []
        for row in rows:
            if host and not _domain_matches(str(row["domain"]), host):
                continue
            out.append(
                {
                    "name": str(row["name"]),
                    "value": str(row["value"]),
                    "domain": str(row["domain"]),
                    "path": str(row["path"] or "/"),
                    "secure": bool(row["secure"]),
                }
            )
        return out

    # Return the family's non-expired imported cookies for a host as {name: value} (transport replay).
    def imported_cookies_map(self, family: str, host: str) -> dict[str, str]:
        if not host:
            return {}
        out: dict[str, str] = {}
        for row in self._imported_rows(family):
            if _domain_matches(str(row["domain"]), host):
                out[str(row["name"])] = str(row["value"])
        return out

    # Return the non-expired imported rows for a family (expires>0 in the past is dropped;
    # 0 means a session cookie and is kept).
    def _imported_rows(self, family: str) -> list[sqlite3.Row]:
        family = (family or "").strip() or "default"
        now = time.time()
        try:
            rows = self._get_conn().execute(
                "SELECT domain, name, value, path, expires, secure FROM imported_cookies "
                "WHERE family = ?",
                (family,),
            ).fetchall()
        except sqlite3.DatabaseError as exc:
            logger.warning("identity_store: imported read failed for %r: %s", family, exc)
            return []
        return [r for r in rows if not (r["expires"] and float(r["expires"]) < now)]

    # Drop imported cookies for one family, or all families when family is None.
    def clear_imported(self, family: str | None = None) -> None:
        with self._write_lock:
            conn = self._get_conn()
            try:
                with conn:
                    if family is None:
                        conn.execute("DELETE FROM imported_cookies")
                    else:
                        conn.execute(
                            "DELETE FROM imported_cookies WHERE family = ?",
                            ((family or "").strip() or "default",),
                        )
            except sqlite3.DatabaseError as exc:
                logger.warning("identity_store: clear_imported failed: %s", exc)

    # Return the seconds since the most recent import across all families (infinity when nothing
    # is imported or the read fails); callers use it as the refresh clock.
    def imported_age_seconds(self) -> float:
        try:
            row = self._get_conn().execute(
                "SELECT MAX(imported) AS m FROM imported_cookies"
            ).fetchone()
        except sqlite3.DatabaseError:
            return float("inf")
        if not row or row["m"] is None:
            return float("inf")
        return max(0.0, time.time() - float(row["m"]))


# Read an attribute from a dict or an object (ImportedCookie) uniformly.
def _attr(obj: Any, key: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


# Process-wide IdentityStore singleton and the lock guarding its lazy creation.
_store: Optional[IdentityStore] = None
_store_lock = threading.Lock()

# Location of the identity database inside the web search _cache directory.
_DB_PATH = Path(__file__).resolve().parents[3] / "_cache" / "browser_identity.db"


# Return the lazily-initialised process-wide IdentityStore singleton.
def get_identity_store() -> IdentityStore:
    global _store
    if _store is None:
        # Double-checked locking: re-test under the lock so only one thread creates the store.
        with _store_lock:
            if _store is None:
                _store = IdentityStore(str(_DB_PATH))
    return _store
