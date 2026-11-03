# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import configparser
import json
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger("core.fetch.browser.profile_import")

# Chromium family label shared with the warm daemon / transport (`_WARM_FAMILY`). Chrome, Edge
# and Brave all serve the same engine, so their cookies land under one family identity.
CHROMIUM_FAMILY = "chromium"
FIREFOX_FAMILY = "firefox"

# Browser names that share the chromium engine and cookie format; Brave, Edge and Chrome differ
# only in their on-disk location.
_CHROMIUM_BROWSERS = ("chrome", "edge", "brave")


# One decrypted cookie, normalised to the storageState cookie shape. The expires field holds
# unix seconds, with 0 meaning a session cookie.
@dataclass(slots=True)
class ImportedCookie:
    domain: str
    name: str
    value: str
    path: str = "/"
    expires: float = 0.0
    secure: bool = False
    http_only: bool = False


# Result of harvesting one browser profile: its cookies plus the diagnostics of the read.
# skipped_encrypted counts cookies that could not be decrypted (bad key or another OS's scheme),
# abe_locked counts cookies sealed with App-Bound Encryption (v20) that need the CDP path, and
# is_default marks a browser that is the OS default (harvested first).
@dataclass(slots=True)
class ProfileHarvest:
    browser: str
    family: str
    profile: str
    cookies: list[ImportedCookie] = field(default_factory=list)
    accept_language: str = ""
    skipped_encrypted: int = 0
    abe_locked: int = 0
    is_default: bool = False
    error: str = ""

    # Return True when the profile yielded at least one usable cookie.
    @property
    def ok(self) -> bool:
        return bool(self.cookies)


# Profile discovery

# Per-OS base directory for each chromium browser's "User Data" root.
def _chromium_user_data_roots() -> dict[str, list[Path]]:
    home = Path.home()
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", home / "AppData/Local"))
        return {
            "chrome": [local / "Google/Chrome/User Data"],
            "edge": [local / "Microsoft/Edge/User Data"],
            "brave": [local / "BraveSoftware/Brave-Browser/User Data"],
        }
    if sys.platform == "darwin":
        app = home / "Library/Application Support"
        return {
            "chrome": [app / "Google/Chrome"],
            "edge": [app / "Microsoft Edge"],
            "brave": [app / "BraveSoftware/Brave-Browser"],
        }

    # Linux and other POSIX systems.
    cfg = Path(os.environ.get("XDG_CONFIG_HOME", home / ".config"))
    return {
        "chrome": [cfg / "google-chrome"],
        "edge": [cfg / "microsoft-edge"],
        "brave": [cfg / "BraveSoftware/Brave-Browser"],
    }


# Per-OS directory that holds Firefox's profiles.ini and Profiles folder.
def _firefox_profile_roots() -> list[Path]:
    home = Path.home()
    if sys.platform == "win32":
        return [Path(os.environ.get("APPDATA", home / "AppData/Roaming")) / "Mozilla/Firefox"]
    if sys.platform == "darwin":
        return [home / "Library/Application Support/Firefox"]
    return [home / ".mozilla/firefox"]


# The cookie DB file inside a chromium profile dir (newer builds nest it under Network/).
def _chromium_cookie_db(profile_dir: Path) -> Optional[Path]:
    for candidate in (profile_dir / "Network" / "Cookies", profile_dir / "Cookies"):
        if candidate.is_file():
            return candidate
    return None


# Return the profile subdirectories to harvest inside a chromium "User Data" root: only
# Default, or Default plus every "Profile N" directory with all_profiles.
def _chromium_profiles(user_data: Path, all_profiles: bool) -> list[Path]:
    if not user_data.is_dir():
        return []
    default = user_data / "Default"
    if not all_profiles:
        return [default] if default.is_dir() else []
    out: list[Path] = []
    if default.is_dir():
        out.append(default)
    for child in sorted(user_data.glob("Profile *")):
        if child.is_dir():
            out.append(child)
    return out


# Return the Firefox profile dirs that have a cookie DB: the default (from profiles.ini) or
# all under the root.
def _firefox_profiles(root: Path, all_profiles: bool) -> list[Path]:
    if not root.is_dir():
        return []
    profiles_dir = root / "Profiles"
    if all_profiles:
        return [d for d in sorted(profiles_dir.glob("*")) if (d / "cookies.sqlite").is_file()]

    # Resolve the default profile from profiles.ini.
    default = _firefox_default_profile(root)
    if default and (default / "cookies.sqlite").is_file():
        return [default]

    # Fallback: the *.default-release / *.default dir with a cookie DB.
    for pattern in ("*.default-release", "*.default*", "*"):
        for d in sorted(profiles_dir.glob(pattern)):
            if (d / "cookies.sqlite").is_file():
                return [d]
    return []


# Resolve the default Firefox profile directory from profiles.ini, or None when it cannot be
# determined.
def _firefox_default_profile(root: Path) -> Optional[Path]:
    ini = root / "profiles.ini"
    if not ini.is_file():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read(ini, encoding="utf-8")
    except (OSError, configparser.Error):
        return None

    # Prefer an [Install*] Default= (the profile Firefox actually launches), else a
    # [Profile*] with Default=1.
    candidates: list[str] = []
    for section in parser.sections():
        if section.lower().startswith("install"):
            if val := parser.get(section, "Default", fallback=""):
                candidates.append(val)
    for section in parser.sections():
        if section.lower().startswith("profile") and parser.get(section, "Default", fallback="") == "1":
            if val := parser.get(section, "Path", fallback=""):
                candidates.append(val)

    # Return the first candidate that exists; relative paths are relative to the Firefox root.
    for rel in candidates:
        p = (root / rel) if not Path(rel).is_absolute() else Path(rel)
        if p.is_dir():
            return p
    return None


# Chromium key and value decryption

# DPAPI-unwrap the AES-256 key stored in a chromium "Local State" file. Windows only; returns
# None elsewhere or when the file/key is missing or ABE-only. Decryption is implemented for
# Windows (DPAPI): on macOS and Linux the key lives behind a different KDF (Keychain, kwallet),
# so this is a no-op there and encrypted values are skipped rather than returned as garbage.
def _chromium_aes_key(user_data: Path) -> Optional[bytes]:
    local_state = user_data / "Local State"
    if not local_state.is_file():
        return None

    # Read the base64 encrypted key from the os_crypt section.
    try:
        raw = json.loads(local_state.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("profile_import: cannot read Local State %s: %s", local_state, exc)
        return None
    b64_key = (raw.get("os_crypt") or {}).get("encrypted_key")
    if not b64_key:
        return None
    import base64

    # Only DPAPI-wrapped keys (the "DPAPI" prefix) on Windows can be unwrapped here.
    try:
        blob = base64.b64decode(b64_key)
    except (ValueError, TypeError):
        return None
    if not blob.startswith(b"DPAPI"):
        return None
    if sys.platform != "win32":
        return None
    try:
        import win32crypt  # type: ignore

        # CryptUnprotectData returns (description, data).
        _, key = win32crypt.CryptUnprotectData(blob[5:], None, None, None, 0)
        return bytes(key)
    except Exception as exc:  # noqa: BLE001 — any DPAPI failure → no key, skip encrypted values
        logger.debug("profile_import: DPAPI key unwrap failed: %s", exc)
        return None


# Decrypt one chromium cookie value. Returns (plaintext, ok). ok=False means it was encrypted
# with a scheme we can't handle (App-Bound v20, or non-Windows key) — the caller counts it as
# skipped rather than treating "" as a real empty cookie. Chromium (v80+) values are `v10`/`v11`
# followed by a 12-byte nonce, the ciphertext and a 16-byte GCM tag (AES-256-GCM under the key
# from Local State); legacy values are DPAPI-encrypted directly. App-Bound Encryption (Chrome
# 127+, the `v20` prefix) is not decryptable outside the browser process, so those cookies are
# skipped, not guessed.
def _decrypt_chromium_value(encrypted: bytes, key: Optional[bytes]) -> tuple[str, bool]:
    if not encrypted:
        return "", True
    prefix = encrypted[:3]

    # Current scheme: AES-256-GCM with the Local State key.
    if prefix in (b"v10", b"v11"):
        if key is None:
            return "", False
        try:
            from cryptography.hazmat.primitives.ciphers.aead import AESGCM

            nonce = encrypted[3:15]
            payload = encrypted[15:]
            plain = AESGCM(key).decrypt(nonce, payload, None)

            # Newer Chrome builds prepend a 32-byte SHA256(domain) hash to the plaintext that is
            # not part of the value. It is not stripped here, so on such profiles the decoded
            # value keeps that prefix; profiles without it decode correctly.
            return plain.decode("utf-8", "replace"), True
        except Exception as exc:  # noqa: BLE001 — bad tag/key → skip, don't fabricate a value
            logger.debug("profile_import: AES-GCM cookie decrypt failed: %s", exc)
            return "", False

    # App-Bound Encryption: the key is sealed to the browser process and not recoverable here.
    if prefix == b"v20":
        return "", False

    # Legacy: the whole value is DPAPI-encrypted (pre-v80 Chrome).
    if sys.platform == "win32":
        try:
            import win32crypt  # type: ignore

            _, plain = win32crypt.CryptUnprotectData(encrypted, None, None, None, 0)
            return bytes(plain).decode("utf-8", "replace"), True
        except Exception:  # noqa: BLE001
            return "", False

    # Other platforms have no usable key scheme here.
    return "", False


# Cookie DB readers

# Copy a (possibly browser-locked) SQLite DB to a temp file and open a connection to the copy.
# Chrome holds an exclusive-ish lock while running; a copy sidesteps it and never mutates the
# user's live file. Returns (connection, temp path) on success or (None, error message).
def _open_readonly_copy(db_path: Path) -> tuple[Optional[sqlite3.Connection], Optional[str]]:
    try:
        fd, tmp = tempfile.mkstemp(suffix=".sqlite", prefix="idimport_")
        os.close(fd)
        shutil.copy2(db_path, tmp)

        # Copy the WAL/SHM sidecars too, so recent (unflushed) writes are visible.
        for suffix in ("-wal", "-shm"):
            side = Path(str(db_path) + suffix)
            if side.is_file():
                shutil.copy2(side, tmp + suffix)
        conn = sqlite3.connect(tmp)
        conn.row_factory = sqlite3.Row
        return conn, tmp
    except (OSError, sqlite3.Error) as exc:
        return None, f"open failed: {exc}"


# Delete the temp DB copy and its WAL/SHM sidecars.
def _cleanup_copy(tmp: Optional[str]) -> None:
    if not tmp:
        return
    for path in (tmp, tmp + "-wal", tmp + "-shm"):
        try:
            os.unlink(path)
        except OSError:
            # Cleanup is best-effort: the sidecars may not exist or the file may be locked.
            pass


# Offset between the Chrome epoch (microseconds since 1601-01-01) and the unix epoch, in seconds.
_CHROME_EPOCH_OFFSET = 11644473600


# Convert a Chrome timestamp (microseconds since 1601-01-01) to unix seconds; 0 stays 0 (session).
def _chrome_time_to_unix(value: int) -> float:
    if not value:
        return 0.0
    return max(0.0, value / 1_000_000 - _CHROME_EPOCH_OFFSET)


# Read the allowlisted cookies of a chromium profile and return (cookies, skipped_encrypted,
# abe_locked). Raises RuntimeError when the DB copy cannot be opened or queried.
def _read_chromium_cookies(
    db_path: Path, key: Optional[bytes], allow: "DomainFilter"
) -> tuple[list[ImportedCookie], int, int]:
    conn, tmp = _open_readonly_copy(db_path)
    if conn is None:
        raise RuntimeError(tmp or "cookie db open failed")
    cookies: list[ImportedCookie] = []
    skipped = 0
    abe_locked = 0

    # Load every cookie row from the copied DB; the domain filter is applied below.
    try:
        rows = conn.execute(
            "SELECT host_key, name, encrypted_value, value, path, expires_utc, "
            "is_secure, is_httponly FROM cookies"
        ).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        _cleanup_copy(tmp)
        raise RuntimeError(f"cookie query failed: {exc}") from exc

    # Keep allowlisted cookies; count the undecryptable ones instead of importing garbage.
    for row in rows:
        host = str(row["host_key"] or "")
        if not allow.matches(host):
            continue
        enc = row["encrypted_value"]
        plain, ok = ("", True)
        if enc:
            enc_bytes = bytes(enc)

            # App-Bound Encryption cookies are not file-decryptable.
            if enc_bytes[:3] == b"v20":
                abe_locked += 1
                continue
            plain, ok = _decrypt_chromium_value(enc_bytes, key)
        elif row["value"]:
            # A plaintext value stored in the non-encrypted column.
            plain = str(row["value"])
        if not ok:
            skipped += 1
            continue
        if not plain:
            continue
        cookies.append(
            ImportedCookie(
                domain=host,
                name=str(row["name"] or ""),
                value=plain,
                path=str(row["path"] or "/"),
                expires=_chrome_time_to_unix(int(row["expires_utc"] or 0)),
                secure=bool(row["is_secure"]),
                http_only=bool(row["is_httponly"]),
            )
        )
    conn.close()
    _cleanup_copy(tmp)
    return cookies, skipped, abe_locked


# Read the allowlisted cookies of a Firefox profile. Firefox stores cookie values in plaintext
# in cookies.sqlite, so no key or DPAPI is needed and this works on every platform. Raises
# RuntimeError when the DB copy cannot be opened or queried.
def _read_firefox_cookies(db_path: Path, allow: "DomainFilter") -> list[ImportedCookie]:
    conn, tmp = _open_readonly_copy(db_path)
    if conn is None:
        raise RuntimeError(tmp or "cookie db open failed")
    cookies: list[ImportedCookie] = []

    # Load every cookie row from the copied DB; the domain filter is applied below.
    try:
        rows = conn.execute(
            "SELECT host, name, value, path, expiry, isSecure, isHttpOnly FROM moz_cookies"
        ).fetchall()
    except sqlite3.Error as exc:
        conn.close()
        _cleanup_copy(tmp)
        raise RuntimeError(f"cookie query failed: {exc}") from exc

    # Keep allowlisted, non-empty cookies.
    for row in rows:
        host = str(row["host"] or "")
        if not allow.matches(host):
            continue
        value = str(row["value"] or "")
        if not value:
            continue
        cookies.append(
            ImportedCookie(
                domain=host,
                name=str(row["name"] or ""),
                value=value,
                path=str(row["path"] or "/"),
                # Firefox already stores unix seconds.
                expires=float(row["expiry"] or 0),
                secure=bool(row["isSecure"]),
                http_only=bool(row["isHttpOnly"]),
            )
        )
    conn.close()
    _cleanup_copy(tmp)
    return cookies


# Domain allowlist

# Matches a cookie host against a configured domain allowlist (suffix match). An empty
# allowlist matches every host (the config default narrows it to search engines). The importer
# keeps only cookies whose domain passes this filter.
class DomainFilter:
    # Normalize the allowlist entries to lowercase domains without a leading dot.
    def __init__(self, domains: Iterable[str]) -> None:
        self._domains = [d.strip().lstrip(".").lower() for d in domains if d and d.strip()]

    # Return True when the host equals an allowed domain or is a subdomain of one.
    def matches(self, host: str) -> bool:
        if not self._domains:
            return True
        h = (host or "").lstrip(".").lower()
        if not h:
            return False
        return any(h == d or h.endswith("." + d) for d in self._domains)


# Harvesting

# Harvest every profile of one browser. Returns [] when the browser isn't installed.
def _harvest_browser(browser: str, allow: "DomainFilter", all_profiles: bool) -> list[ProfileHarvest]:
    harvests: list[ProfileHarvest] = []

    # Chromium family: one AES key per User Data root, one cookie DB per profile.
    if browser in _CHROMIUM_BROWSERS:
        for user_data in _chromium_user_data_roots().get(browser, []):
            key = _chromium_aes_key(user_data)
            for profile_dir in _chromium_profiles(user_data, all_profiles):
                db = _chromium_cookie_db(profile_dir)
                if not db:
                    continue
                harvest = ProfileHarvest(browser=browser, family=CHROMIUM_FAMILY, profile=profile_dir.name)
                try:
                    harvest.cookies, harvest.skipped_encrypted, harvest.abe_locked = _read_chromium_cookies(
                        db, key, allow
                    )
                    harvest.accept_language = _chromium_accept_language(profile_dir)
                    if harvest.abe_locked and not harvest.cookies:
                        harvest.error = (
                            f"{harvest.abe_locked} cookies App-Bound-Encrypted (v20) - "
                            "not importable by file read; needs the CDP export path"
                        )
                except Exception as exc:  # noqa: BLE001 — one bad profile must not abort the sweep
                    harvest.error = str(exc)
                    logger.info("profile_import: %s/%s skipped: %s", browser, profile_dir.name, exc)
                harvests.append(harvest)
    elif browser == "firefox":
        # Firefox: plaintext cookies, no key needed.
        for root in _firefox_profile_roots():
            for profile_dir in _firefox_profiles(root, all_profiles):
                harvest = ProfileHarvest(browser="firefox", family=FIREFOX_FAMILY, profile=profile_dir.name)
                try:
                    harvest.cookies = _read_firefox_cookies(profile_dir / "cookies.sqlite", allow)
                    harvest.accept_language = _firefox_accept_language(profile_dir)
                except Exception as exc:  # noqa: BLE001
                    harvest.error = str(exc)
                    logger.info("profile_import: firefox/%s skipped: %s", profile_dir.name, exc)
                harvests.append(harvest)
    return harvests


# Harvest cookies from the configured, installed browsers, DEFAULT BROWSER FIRST then the rest
# as fallback (per the identity contract: prefer the browser the user actually lives in; fall
# back to others only if it yields nothing — e.g. an App-Bound-locked Chrome falls through to a
# working Firefox). Never raises: a broken/locked profile yields a ProfileHarvest with `.error`.
# The importer only ever reads profiles (a COPY of the cookie DB, never the live file) and only
# keeps cookies whose domain matches the allowlist built from `domains`.
def harvest_profiles(
    *,
    browsers: Iterable[str],
    domains: Iterable[str],
    all_profiles: bool = False,
) -> list[ProfileHarvest]:
    allow = DomainFilter(domains)
    wanted = [b.strip().lower() for b in browsers if b and b.strip()]
    ordered = _default_first(wanted)
    default = ordered[0] if ordered else ""

    # Harvest in order and flag the results of the first browser as the default one (it is the
    # OS default whenever that browser is among the requested ones).
    harvests: list[ProfileHarvest] = []
    for browser in ordered:
        browser_harvests = _harvest_browser(browser, allow, all_profiles)
        for h in browser_harvests:
            h.is_default = h.browser == default
        harvests.extend(browser_harvests)
    return harvests


# Reorder the wanted browsers so the OS default (if present and requested) is harvested first.
def _default_first(wanted: list[str]) -> list[str]:
    default = detect_default_browser()
    if default and default in wanted:
        return [default] + [b for b in wanted if b != default]
    return list(wanted)


# Default-browser detection

# Map a Windows UserChoice ProgId prefix to a browser we support.
_PROGID_BROWSER = {
    "ChromeHTML": "chrome",
    "MSEdgeHTM": "edge",
    "BraveHTML": "brave",
    "FirefoxURL": "firefox",
}


# Return the user's OS default browser as one of our supported names, or "" when undetectable.
def detect_default_browser() -> str:
    if sys.platform == "win32":
        return _default_browser_windows()
    if sys.platform == "darwin":
        return _default_browser_macos()
    return _default_browser_linux()


# Read the https UserChoice ProgId from the registry and map it to a supported browser name.
def _default_browser_windows() -> str:
    try:
        import winreg  # type: ignore

        key_path = r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path) as key:
            prog_id, _ = winreg.QueryValueEx(key, "ProgId")
    except OSError:
        return ""
    for progid_prefix, browser in _PROGID_BROWSER.items():
        if str(prog_id).startswith(progid_prefix):
            return browser
    return ""


# Detect the default browser on macOS. Not implemented: LaunchServices holds the default handler
# and parsing its plist is brittle, so this always reports "undetectable".
def _default_browser_macos() -> str:
    return ""


# Ask xdg-settings for the default web browser and map it to a supported browser name.
def _default_browser_linux() -> str:
    try:
        import subprocess

        out = subprocess.run(
            ["xdg-settings", "get", "default-web-browser"],
            capture_output=True, text=True, timeout=3,
        ).stdout.strip().lower()
    except (OSError, ValueError):
        return ""
    for name in ("chrome", "edge", "brave", "firefox"):
        if name in out:
            return name
    return ""


# Metadata extraction

# Chromium stores the negotiated Accept-Language set in Preferences → intl.accept_languages.
def _chromium_accept_language(profile_dir: Path) -> str:
    prefs = profile_dir / "Preferences"
    if not prefs.is_file():
        return ""
    try:
        raw = json.loads(prefs.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return ""
    langs = (raw.get("intl") or {}).get("accept_languages") or ""
    return str(langs).strip()


# Firefox keeps intl.accept_languages in prefs.js as a user_pref line.
def _firefox_accept_language(profile_dir: Path) -> str:
    prefs = profile_dir / "prefs.js"
    if not prefs.is_file():
        return ""
    try:
        text = prefs.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    marker = 'user_pref("intl.accept_languages", "'
    idx = text.find(marker)
    if idx == -1:
        return ""
    start = idx + len(marker)
    end = text.find('"', start)
    return text[start:end].strip() if end != -1 else ""


# Run-once import trigger

# Guard for the process-wide run-once import: the lock serializes callers and the flag records
# that the import already ran.
_trigger_lock = threading.Lock()
_trigger_done = False


# Summary of one import call. ran says whether a harvest actually ran this call, imported counts
# the cookies written to the store, families maps each family to its count and notes collects
# human-readable diagnostics.
@dataclass(slots=True)
class ImportOutcome:
    ran: bool = False
    enabled: bool = False
    imported: int = 0
    families: dict[str, int] = field(default_factory=dict)
    default_browser: str = ""
    notes: list[str] = field(default_factory=list)


# Run the profile import at most once per process (guarded), honoring config. Safe to call from
# the hot path: after the first call it is a bool check. Disabled config → optional purge, no read.
def ensure_imported() -> ImportOutcome:
    global _trigger_done
    if _trigger_done:
        return ImportOutcome(ran=False)
    with _trigger_lock:
        # Re-check under the lock: another thread may have finished the import meanwhile.
        if _trigger_done:
            return ImportOutcome(ran=False)
        try:
            return _run_import()
        except Exception as exc:  # noqa: BLE001 — import must never break a search
            logger.warning("profile_import: import run failed: %s", exc)
            return ImportOutcome(ran=True, notes=[f"error: {exc}"])
        finally:
            _trigger_done = True


# Force a re-run on the next ensure_imported() (config reload / tests).
def reset_import_trigger() -> None:
    global _trigger_done
    with _trigger_lock:
        _trigger_done = False


# Explicit entry point for a manual or forced import: run the core import now and return its
# outcome. With force=True the refresh guard is skipped.
def run_profile_import(*, force: bool = False) -> ImportOutcome:
    return _run_import(force=force)


# The core import: read the config and the store, harvest default-first, write per family. The
# import is STRICTLY OPT-IN (nothing is read unless `profile_import.enabled` is set), because
# reading a cookie jar exposes the user's live sessions.
def _run_import(*, force: bool = False) -> ImportOutcome:
    from core.config.settings import load_search_config
    from core.fetch.browser.identity_store import get_identity_store

    cfg = load_search_config().profile_import
    store = get_identity_store()
    outcome = ImportOutcome(enabled=cfg.enabled, default_browser=detect_default_browser())

    # Opt-in gate: when disabled, optionally purge earlier imports and read nothing.
    if not cfg.enabled:
        if cfg.purge_on_disable:
            store.clear_imported()
            outcome.notes.append("disabled: imported cookies purged")
        return outcome

    # Refresh guard: skip a re-harvest while the last import is still fresh.
    if not force and cfg.refresh_hours > 0:
        age = store.imported_age_seconds()
        if age < cfg.refresh_hours * 3600.0:
            outcome.notes.append(f"fresh ({age/3600:.1f}h < {cfg.refresh_hours}h) — reused")
            return outcome

    outcome.ran = True
    harvests = harvest_profiles(
        browsers=cfg.browsers, domains=cfg.domains, all_profiles=cfg.all_profiles
    )

    # Merge per family in harvest order (default browser first → its cookies win on conflict).
    by_family: dict[str, list[ImportedCookie]] = {}
    for h in harvests:
        if h.error:
            outcome.notes.append(f"{h.browser}/{h.profile}: {h.error}")
        by_family.setdefault(h.family, []).extend(h.cookies)

    # Replace each family's stored set with the freshly harvested one.
    for family, cookies in by_family.items():
        n = store.import_cookies(family, cookies, source="profile_import", replace=True)
        outcome.families[family] = n
        outcome.imported += n

    logger.info(
        "profile_import: default=%s imported=%d families=%s",
        outcome.default_browser or "?", outcome.imported, outcome.families,
    )
    return outcome
