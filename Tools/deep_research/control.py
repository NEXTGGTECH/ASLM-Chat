# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import os
import platform
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping


# Repository root and the old repository-local store that is migrated to the runtime directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_CONTROL_ROOT = PROJECT_ROOT / "Data" / "deep_research"


# Return the per-user runtime directory that holds deep research state: the explicit environment
# override when set, otherwise the platform's local state location.
def _default_control_root() -> Path:
    # An explicit runtime directory from the environment wins over every platform default.
    override = str(os.environ.get("ASLM_DEEP_RESEARCH_RUNTIME_DIR") or "").strip()
    if override:
        return Path(os.path.expandvars(override)).expanduser()

    # LocalAppData on Windows and Application Support on macOS.
    system = platform.system().lower()
    if system == "windows":
        local_app_data = str(os.environ.get("LOCALAPPDATA") or "").strip()
        base = Path(local_app_data) if local_app_data else Path.home() / "AppData" / "Local"
        return base / "ASLM-Chat" / "runtime" / "deep-research"
    if system == "darwin":
        return (
            Path.home()
            / "Library"
            / "Application Support"
            / "ASLM-Chat"
            / "runtime"
            / "deep-research"
        )

    # Linux and other systems use the XDG state home, falling back to ~/.local/state.
    xdg_state_home = str(os.environ.get("XDG_STATE_HOME") or "").strip()
    base = Path(xdg_state_home).expanduser() if xdg_state_home else Path.home() / ".local" / "state"
    return base / "aslm-chat" / "deep-research"


# Runtime store: session snapshots live in STATE_ROOT, per-session UI commands in COMMAND_ROOT.
CONTROL_ROOT = _default_control_root()
STATE_ROOT = CONTROL_ROOT / "state"
COMMAND_ROOT = CONTROL_ROOT / "commands"

# Names of the marker and lock files of the one-time legacy migration and of the retention sweep
# (all created in the storage root), the interval between retention sweeps, the age after which a
# lock file counts as abandoned, and the time of the last sweep in this process.
_MIGRATION_MARKER = ".legacy-data-migrated-v1.json"
_MIGRATION_LOCK = ".legacy-data-migration-v1.lock"
_RETENTION_LOCK = ".terminal-retention-v1.lock"
_MAINTENANCE_INTERVAL_SECONDS = 300.0
_LOCK_STALE_SECONDS = 120.0
_last_retention_check = 0.0

# Public session id format (deep-research:<32 hex chars>) and the statuses after which a run is final.
_SESSION_RE = re.compile(r"^deep-research:([a-f0-9]{32})$", flags=re.I)
_TERMINAL_STATUSES = {"completed", "partial", "cancelled", "failed", "expired"}


# Error raised when a caller supplies an unsafe or malformed session id.
class InvalidResearchSession(ValueError):
    pass


# Session ids

# Return a new random session id in the public form deep-research:<32 hex chars>.
def new_session_id() -> str:
    return f"deep-research:{uuid.uuid4().hex}"


# Validate a public session id and return its lowercase 32-character hex key, which is safe to use
# in file names. Raise InvalidResearchSession for anything else.
def session_key(session_id: Any) -> str:
    raw = str(session_id or "").strip()
    match = _SESSION_RE.fullmatch(raw)
    if not match:
        raise InvalidResearchSession("Invalid deep research session id.")
    return match.group(1).lower()


# Return True when the status is final, so the run can no longer change.
def is_terminal_status(status: Any) -> bool:
    return str(status or "").strip().lower() in _TERMINAL_STATUSES


# Storage helpers

# Return the common parent of the state and command roots, or CONTROL_ROOT when the two roots do not
# share a parent (tests replace the leaf roots).
def _storage_root() -> Path:
    if STATE_ROOT.parent == COMMAND_ROOT.parent:
        return STATE_ROOT.parent
    return CONTROL_ROOT


# Read an integer from the environment, clamped to the minimum; return the default when the value
# is not a valid integer.
def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(str(os.environ.get(name) or default).strip()))
    except (TypeError, ValueError):
        return default


# Acquire a short cross-process lease without blocking request polling. Create the lock file
# exclusively and return its descriptor, or None when another process holds the lease. A lock
# file older than _LOCK_STALE_SECONDS counts as abandoned: it is removed and the acquisition is
# retried once.
def _try_acquire_lease(path: Path) -> int | None:
    for attempt in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(descriptor, f"{os.getpid()} {time.time()}\n".encode("ascii"))
            return descriptor
        except FileExistsError:
            # The lock exists: give up on the second attempt or while it is still fresh,
            # otherwise remove the stale lock and try again.
            if attempt:
                return None
            try:
                if time.time() - path.stat().st_mtime <= _LOCK_STALE_SECONDS:
                    return None
                path.unlink(missing_ok=True)
            except OSError:
                return None
        except OSError:
            return None
    return None


# Close the lease descriptor and remove its lock file. Do nothing when no lease was acquired.
def _release_lease(path: Path, descriptor: int | None) -> None:
    if descriptor is None:
        return
    try:
        os.close(descriptor)
    finally:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # An undeletable lock file is not fatal: it becomes stale and is reclaimed later.
            pass


# Atomically replace the destination with the temporary file, retrying briefly on PermissionError
# because Windows can transiently lock files that another process is reading.
def _replace_with_retry(temporary_path: Path, destination: Path) -> None:
    for attempt in range(5):
        try:
            os.replace(temporary_path, destination)
            return
        except PermissionError:
            if attempt >= 4:
                raise
            time.sleep(0.01 * (attempt + 1))


# Legacy migration

# Copy one legacy artifact through an atomic destination-side swap. Return False when the source
# cannot be read.
def _atomic_copy(source: Path, destination: Path) -> bool:
    try:
        content = source.read_bytes()
    except OSError:
        # Antivirus/indexing can transiently lock files on Windows. Migration is
        # best-effort: leave this artifact in legacy storage and retry next time.
        return False

    # Write a temporary file next to the destination, flush it to disk and swap it in.
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as target_handle:
            target_handle.write(content)
            target_handle.flush()
            os.fsync(target_handle.fileno())
        _replace_with_retry(temporary_path, destination)
        return True
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            # The temporary file is already gone after a successful swap; a leftover is harmless.
            pass


# Return the payload's updated_at as a float, or 0.0 when it is missing or not numeric.
def _json_timestamp(payload: Mapping[str, Any]) -> float:
    try:
        return float(payload.get("updated_at") or 0)
    except (TypeError, ValueError):
        return 0.0


# Return True when both files hold the same bytes or the same non-empty parsed JSON object, so
# formatting differences between a migrated copy and its source do not count as a mismatch.
def _same_file_payload(source: Path, destination: Path) -> bool:
    try:
        if source.read_bytes() == destination.read_bytes():
            return True
    except OSError:
        return False
    source_payload = _read_json(source)
    destination_payload = _read_json(destination)
    return bool(source_payload) and source_payload == destination_payload


# Remove the empty legacy per-session command folders and then the emptied legacy directories.
def _remove_empty_legacy_directories() -> None:
    # Per-session command folders first.
    command_root = LEGACY_CONTROL_ROOT / "commands"
    if command_root.is_dir():
        for directory in command_root.iterdir():
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    # A folder that still holds files is kept.
                    pass

    # Then the state, command and root directories, deepest first.
    for directory in (
        LEGACY_CONTROL_ROOT / "state",
        command_root,
        LEGACY_CONTROL_ROOT,
    ):
        try:
            directory.rmdir()
        except OSError:
            # A directory that is not empty is kept.
            pass


# Return the sorted files under root that match the glob pattern, or an empty list when the root
# cannot be listed.
def _legacy_files(root: Path, pattern: str) -> list[Path]:
    try:
        return sorted(root.glob(pattern))
    except OSError:
        return []


# Return True when files matching the pattern remain under root. Assume they do when the root
# cannot be listed, so the migration is not marked as finished.
def _legacy_files_remain(root: Path, pattern: str) -> bool:
    try:
        return any(root.glob(pattern))
    except OSError:
        return True


# Move the old repository-local store to the runtime directory once, preserving the newest
# snapshots. Under a cross-process lease, copy or merge each legacy state snapshot and command
# file, delete the source once the destination is verified and write a marker when nothing is
# left. Files that cannot be migrated safely stay in place and are retried on the next start.
def _migrate_legacy_storage() -> None:
    root = _storage_root()

    # Replaced roots in tests and embedding applications must not unexpectedly
    # import the real repository's legacy data.
    if root != CONTROL_ROOT or LEGACY_CONTROL_ROOT == root:
        return

    # Skip when an earlier start already finished the migration.
    marker = root / _MIGRATION_MARKER
    if marker.exists():
        return

    # Take the migration lease, waiting up to five seconds for another process that holds it
    # and giving up early when that process writes the marker.
    lock = root / _MIGRATION_LOCK
    descriptor = _try_acquire_lease(lock)
    deadline = time.monotonic() + 5.0
    while descriptor is None and not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.02)
        descriptor = _try_acquire_lease(lock)
    if descriptor is None:
        return
    states_copied = 0
    commands_copied = 0
    try:
        # Re-check under the lease: another process may have finished the migration meanwhile.
        if marker.exists():
            return

        # State snapshots: copy missing ones, merge into existing ones (the newer snapshot wins
        # on conflicting keys), verify the destination and only then delete the source.
        legacy_state_root = LEGACY_CONTROL_ROOT / "state"
        if legacy_state_root.is_dir():
            for source in _legacy_files(legacy_state_root, "*.json"):
                destination = STATE_ROOT / source.name
                legacy_payload = _read_json(source)
                expected_payload: dict[str, Any] | None = None
                if destination.exists():
                    current_payload = _read_json(destination)
                    if legacy_payload and current_payload:
                        if _json_timestamp(current_payload) >= _json_timestamp(legacy_payload):
                            expected_payload = {**legacy_payload, **current_payload}
                        else:
                            expected_payload = {**current_payload, **legacy_payload}
                        _atomic_write_json(destination, expected_payload)
                    elif not _same_file_payload(source, destination):
                        # A malformed conflict cannot be merged safely. Leave the
                        # legacy artifact in place and retry on a later startup.
                        continue
                else:
                    # No destination snapshot yet: copy the file as it is.
                    if not _atomic_copy(source, destination):
                        continue
                    expected_payload = legacy_payload or None

                # Delete the source only after the destination is verified to hold the
                # migrated content.
                if legacy_payload:
                    if _read_json(destination) != (expected_payload or legacy_payload):
                        continue
                elif not _same_file_payload(source, destination):
                    continue
                try:
                    source.unlink(missing_ok=True)
                except OSError:
                    # The source cannot be deleted now; it is migrated again on the next start.
                    continue
                states_copied += 1

        # Command files live in one folder per session; folders whose names are not session
        # keys are ignored.
        legacy_command_root = LEGACY_CONTROL_ROOT / "commands"
        if legacy_command_root.is_dir():
            for source in _legacy_files(legacy_command_root, "*/*.json"):
                session_directory = source.parent.name
                if not re.fullmatch(r"[a-f0-9]{32}", session_directory, flags=re.I):
                    continue
                destination = COMMAND_ROOT / session_directory.lower() / source.name
                if not destination.exists():
                    if not _atomic_copy(source, destination):
                        continue
                if not _same_file_payload(source, destination):
                    continue
                try:
                    source.unlink(missing_ok=True)
                except OSError:
                    # The source cannot be deleted now; it is migrated again on the next start.
                    continue
                commands_copied += 1

        # Write the marker and clean up only when no legacy file is left; otherwise retry on
        # the next start.
        remaining_states = legacy_state_root.is_dir() and _legacy_files_remain(
            legacy_state_root,
            "*.json",
        )
        remaining_commands = legacy_command_root.is_dir() and _legacy_files_remain(
            legacy_command_root,
            "*/*.json",
        )
        if remaining_states or remaining_commands:
            return
        _atomic_write_json(
            marker,
            {
                "schema_version": 1,
                "migrated_at": time.time(),
                "legacy_root": str(LEGACY_CONTROL_ROOT),
                "states_moved": states_copied,
                "commands_moved": commands_copied,
            },
        )
        _remove_empty_legacy_directories()
    finally:
        _release_lease(lock, descriptor)


# Retention

# Delete the JSON command files of one session and then its command folder.
def _remove_command_directory(session_key_value: str) -> None:
    directory = COMMAND_ROOT / session_key_value
    if not directory.is_dir():
        return
    for path in directory.glob("*.json"):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            # A locked file stays; a later sweep removes it.
            pass
    try:
        directory.rmdir()
    except OSError:
        # The folder still holds files that could not be removed.
        pass


# Drop commands durably acknowledged by a worker, including after a restart: remove the whole
# command folder of finished sessions and the files listed in processed_commands of running ones.
def _cleanup_persisted_commands() -> None:
    try:
        directories = list(COMMAND_ROOT.iterdir())
    except OSError:
        return
    for directory in directories:
        # Only session command folders (named by a session key) are considered.
        if not directory.is_dir() or not re.fullmatch(
            r"[a-f0-9]{32}",
            directory.name,
            flags=re.I,
        ):
            continue

        # Without a snapshot there is nothing to judge the folder by; leave it alone.
        snapshot = _read_json(STATE_ROOT / f"{directory.name.lower()}.json")
        if not snapshot:
            continue

        # A finished run no longer needs any of its commands.
        if is_terminal_status(snapshot.get("status")):
            _remove_command_directory(directory.name.lower())
            continue

        # Remove only the plain *.json file names the worker reports as processed.
        processed_values = snapshot.get("processed_commands")
        processed = {
            str(filename)
            for filename in processed_values
            if Path(str(filename or "")).name == str(filename or "")
            and str(filename or "").endswith(".json")
        } if isinstance(processed_values, list) else set()
        for filename in processed:
            try:
                (directory / filename).unlink(missing_ok=True)
            except OSError:
                # A locked file is removed by a later sweep.
                pass

        # Drop the folder once it is empty.
        try:
            directory.rmdir()
        except OSError:
            # The folder still holds unprocessed commands, which is the normal case.
            pass


# Bound terminal snapshots by age and count while retaining every active run. Unless forced, run
# at most once per maintenance interval, under a cross-process lease. The limits come from
# ASLM_DEEP_RESEARCH_TERMINAL_RETENTION_DAYS (default 30) and
# ASLM_DEEP_RESEARCH_TERMINAL_RETENTION_MAX (default 256).
def _apply_terminal_retention(*, force: bool = False) -> None:
    # Throttle the sweep to one per maintenance interval in this process.
    global _last_retention_check
    now = time.time()
    if not force and now - _last_retention_check < _MAINTENANCE_INTERVAL_SECONDS:
        return
    _last_retention_check = now

    # Only one process sweeps at a time; the others skip this round.
    root = _storage_root()
    lock = root / _RETENTION_LOCK
    descriptor = _try_acquire_lease(lock)
    if descriptor is None:
        return
    try:
        _cleanup_persisted_commands()

        # Retention limits: maximum age in days and maximum number of terminal snapshots.
        retention_days = _env_int("ASLM_DEEP_RESEARCH_TERMINAL_RETENTION_DAYS", 30)
        retention_max = _env_int("ASLM_DEEP_RESEARCH_TERMINAL_RETENTION_MAX", 256)
        cutoff = now - (retention_days * 86400)

        # Collect the terminal snapshots with their last update time; active runs are never touched.
        terminal: list[tuple[float, Path, str]] = []
        for path in STATE_ROOT.glob("*.json"):
            if not re.fullmatch(r"[a-f0-9]{32}", path.stem, flags=re.I):
                continue
            payload = _read_json(path)
            if not payload or not is_terminal_status(payload.get("status")):
                continue
            try:
                updated_at = float(payload.get("updated_at") or path.stat().st_mtime)
            except (OSError, TypeError, ValueError):
                # Without a usable timestamp treat the snapshot as just updated, so age alone
                # never expires it.
                updated_at = now
            terminal.append((updated_at, path, path.stem.lower()))

        # Newest first: a snapshot expires when it is older than the cutoff or ranks beyond the
        # maximum count.
        terminal.sort(key=lambda item: item[0], reverse=True)
        expired = {
            path
            for index, (updated_at, path, _key) in enumerate(terminal)
            if updated_at < cutoff or index >= retention_max
        }

        # Delete the expired snapshots together with their command folders.
        for _updated_at, path, key in terminal:
            if path not in expired:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                # Keep its command folder too; the next sweep retries the whole snapshot.
                continue
            _remove_command_directory(key)
    finally:
        _release_lease(lock, descriptor)


# Snapshot storage

# Create the state and command folders, then run the legacy migration and the throttled retention
# sweep.
def _ensure_roots() -> None:
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    COMMAND_ROOT.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_storage()
    _apply_terminal_retention()


# Return the snapshot file path of a session.
def _state_path(session_id: Any) -> Path:
    return STATE_ROOT / f"{session_key(session_id)}.json"


# Return the command folder path of a session.
def _command_dir(session_id: Any) -> Path:
    return COMMAND_ROOT / session_key(session_id)


# Read a JSON object from a file, retrying briefly on transient errors. Return an empty dict when
# the file is missing, malformed or not a JSON object.
def _read_json(path: Path) -> dict[str, Any]:
    for attempt in range(5):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except (OSError, json.JSONDecodeError):
            if attempt >= 4:
                return {}

            # A worker may be atomically swapping this snapshot while Django
            # polls it. Windows can expose a very short sharing violation.
            time.sleep(0.005 * (attempt + 1))
    return {}


# Write the payload as JSON through a temporary file in the same folder, flush it to disk and
# atomically replace the target, so readers never see a partial file.
def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}-",
        suffix=".tmp",
        dir=str(path.parent),
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(dict(payload), handle, ensure_ascii=False, default=str)
            handle.flush()
            os.fsync(handle.fileno())

        # Windows can briefly lock the destination while another process polls
        # it; retry the atomic swap without losing the previous snapshot.
        _replace_with_retry(temporary_path, path)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            # The temporary file is already gone after a successful swap; a leftover is harmless.
            pass


# Session snapshots

# Create the first snapshot of a session and return it. When the session already has a snapshot,
# return that one untouched. Entries of extra are merged into the new snapshot.
def create_session(
    session_id: Any,
    *,
    topic: str,
    status: str = "planning",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_roots()
    normalized_id = f"deep-research:{session_key(session_id)}"
    path = _state_path(normalized_id)

    # An existing snapshot is never overwritten.
    existing = _read_json(path)
    if existing:
        return existing

    # Build the initial snapshot; the caller's extra entries override these defaults.
    now = time.time()
    snapshot: dict[str, Any] = {
        "schema_version": 1,
        "session_id": normalized_id,
        "status": str(status or "planning"),
        "phase": str(status or "planning"),
        "topic": str(topic or "").strip(),
        "plan": "",
        "plan_version": 0,
        "checklist": [],
        "latest_action": "Preparing the research plan",
        "source_count": 0,
        "queries_used": 0,
        "query_budget": 0,
        "iteration": 0,
        "last_sequence": 0,
        "created_at": now,
        "updated_at": now,
    }
    if isinstance(extra, Mapping):
        snapshot.update(dict(extra))
    _atomic_write_json(path, snapshot)
    return snapshot


# Return the current snapshot of a session, or an empty dict when it has none.
def read_state(session_id: Any) -> dict[str, Any]:
    _ensure_roots()
    return _read_json(_state_path(session_id))


# Return a whitespace-normalized, case-folded checklist title used to match items across snapshots.
def _checklist_title_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


# Merge the incoming checklist with the stored one so a stale worker checkpoint cannot undo accepted
# progress. Incoming items are matched to stored ones by id (when the title is unchanged) or by a
# unique title; an item that was done or skipped keeps that status, and its note, when the incoming
# item regresses. An empty incoming list keeps the stored items.
def _merge_checklist_snapshot(previous: Any, current: Any) -> list[dict[str, Any]]:
    # Normalize both lists to plain dicts; anything else is ignored.
    previous_items = (
        [dict(item) for item in previous if isinstance(item, dict)]
        if isinstance(previous, list)
        else []
    )
    current_items = (
        [dict(item) for item in current if isinstance(item, dict)]
        if isinstance(current, list)
        else []
    )
    if not current_items:
        return previous_items

    # Index the stored items by id and by title; a title that occurs more than once is ambiguous
    # and is not used for matching.
    previous_by_id = {
        str(item.get("id") or ""): item
        for item in previous_items
        if str(item.get("id") or "")
    }
    previous_by_title: dict[str, dict[str, Any]] = {}
    duplicate_titles: set[str] = set()
    for item in previous_items:
        title_key = _checklist_title_key(item.get("title"))
        if not title_key:
            continue
        if title_key in previous_by_title:
            duplicate_titles.add(title_key)
        else:
            previous_by_title[title_key] = item
    for title_key in duplicate_titles:
        previous_by_title.pop(title_key, None)

    # Find the stored version of each incoming item: by id when the title is unchanged, else by
    # unique title.
    output: list[dict[str, Any]] = []
    for item in current_items:
        item_id = str(item.get("id") or "")
        title_key = _checklist_title_key(item.get("title"))
        prior = previous_by_id.get(item_id)
        if prior is not None and _checklist_title_key(prior.get("title")) != title_key:
            prior = None
        if prior is None:
            prior = previous_by_title.get(title_key)

        # Never turn a finished item back into an open one; keep its note when the new one is empty.
        prior_status = str((prior or {}).get("status") or "pending").strip().lower()
        current_status = str(item.get("status") or "pending").strip().lower()
        if prior_status in {"done", "skipped"} and current_status not in {"done", "skipped"}:
            item["status"] = prior_status
            if (
                not str(item.get("note") or "").strip()
                and str((prior or {}).get("note") or "").strip()
            ):
                item["note"] = str(prior.get("note"))[:300]
        output.append(item)
    return output


# Merge a worker checkpoint into the current durable snapshot and return the result. Checklist
# progress is merged so a stale checkpoint cannot undo it, the stop state and its controls are
# protected, and a terminal snapshot removes the session's command inbox.
def update_state(session_id: Any, **changes: Any) -> dict[str, Any]:
    _ensure_roots()
    normalized_id = f"deep-research:{session_key(session_id)}"
    path = _state_path(normalized_id)

    # Load the snapshot, creating it when the session has none yet.
    snapshot = _read_json(path)
    if not snapshot:
        snapshot = create_session(normalized_id, topic=str(changes.get("topic") or ""))

    # A stale checkpoint must not undo accepted checklist progress.
    if "checklist" in changes:
        changes["checklist"] = _merge_checklist_snapshot(
            snapshot.get("checklist"),
            changes.get("checklist"),
        )

    # While the UI is stopping the run, a non-terminal checkpoint may not change status or controls.
    current_status = str(snapshot.get("status") or "").strip().lower()
    incoming_status = str(changes.get("status") or "").strip().lower()
    if current_status == "stopping" and not is_terminal_status(incoming_status):
        # The UI persists ``stopping`` before aborting the worker. A checkpoint
        # already in flight must not resurrect its stale controls on poll/reload.
        for protected_key in (
            "status",
            "phase",
            "can_stop",
            "can_edit",
            "can_approve",
        ):
            changes.pop(protected_key, None)

    # Apply the changes and persist the snapshot atomically.
    snapshot.update(changes)
    snapshot["schema_version"] = 1
    snapshot["session_id"] = normalized_id
    snapshot["updated_at"] = time.time()
    _atomic_write_json(path, snapshot)
    if is_terminal_status(snapshot.get("status")):
        # Once the terminal snapshot is durable, commands cannot affect this run
        # anymore and keeping their immutable inbox only leaks runtime files.
        _remove_command_directory(session_key(normalized_id))
    return snapshot


# Commands

# Append one immutable UI command (approve, revise or cancel) to the session's command inbox for
# the worker to consume. Return the accepted and terminal flags with the state, plus the command
# when it was queued; a finished session queues nothing and reports only cancel as accepted.
# Raise ValueError for an unknown action, FileNotFoundError for an unknown session and
# RuntimeError (ACTION_NOT_AVAILABLE or STALE_PLAN_VERSION) when the action does not apply.
def submit_command(
    session_id: Any,
    action: str,
    *,
    payload: Mapping[str, Any] | None = None,
    expected_plan_version: int | None = None,
) -> dict[str, Any]:
    # Validate the session id and the action.
    normalized_id = f"deep-research:{session_key(session_id)}"
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in {"approve", "revise", "cancel"}:
        raise ValueError("Unsupported deep research action.")

    # Reject unknown sessions, answer for finished ones without queueing, and refuse actions
    # that the session's controls currently disable.
    snapshot = read_state(normalized_id)
    if not snapshot:
        raise FileNotFoundError("Deep research session was not found.")
    if is_terminal_status(snapshot.get("status")):
        return {
            "accepted": normalized_action == "cancel",
            "terminal": True,
            "state": snapshot,
        }
    if normalized_action == "approve" and snapshot.get("can_approve") is False:
        raise RuntimeError("ACTION_NOT_AVAILABLE")
    if normalized_action == "revise" and snapshot.get("can_edit") is False:
        raise RuntimeError("ACTION_NOT_AVAILABLE")

    # Reject a command issued against an outdated plan version. Stop must remain idempotent and
    # immediately acceptable even if a newer plan snapshot reached the worker just before the
    # user's click.
    current_version = int(snapshot.get("plan_version") or 0)
    if (
        normalized_action != "cancel"
        and expected_plan_version is not None
        and int(expected_plan_version) != current_version
    ):
        raise RuntimeError("STALE_PLAN_VERSION")

    # Store the command under a name that sorts by creation time, so the worker replays commands
    # in the order they were submitted.
    command = {
        "schema_version": 1,
        "command_id": uuid.uuid4().hex,
        "session_id": normalized_id,
        "action": normalized_action,
        "expected_plan_version": current_version,
        "payload": dict(payload or {}),
        "created_at": time.time(),
        "created_at_ns": time.time_ns(),
    }
    directory = _command_dir(normalized_id)
    directory.mkdir(parents=True, exist_ok=True)
    filename = f"{command['created_at_ns']:020d}-{command['command_id']}.json"
    _atomic_write_json(directory / filename, command)
    return {"accepted": True, "terminal": False, "command": command, "state": snapshot}


# Read the unprocessed commands of a session in creation order, as (filename, command) pairs.
# A command stays durable until the worker includes its filename in processed. A later poll
# removes those acknowledged files, retaining crash-safe replay without turning the command
# inbox into a permanent log archive.
def read_commands(
    session_id: Any,
    *,
    processed: set[str] | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    directory = _command_dir(session_id)
    if not directory.is_dir():
        return []

    # Remove the acknowledged files; only plain *.json file names are accepted.
    seen = processed if isinstance(processed, set) else set()
    for filename in seen:
        safe_name = Path(str(filename or "")).name
        if safe_name != str(filename or "") or not safe_name.endswith(".json"):
            continue
        try:
            (directory / safe_name).unlink(missing_ok=True)
        except OSError:
            # A locked file is removed by a later poll.
            pass

    # Collect the remaining readable commands in file name (creation) order.
    output: list[tuple[str, dict[str, Any]]] = []
    for path in sorted(directory.glob("*.json"), key=lambda item: item.name):
        if path.name in seen:
            continue
        command = _read_json(path)
        if command:
            output.append((path.name, command))

    # Drop the folder once the inbox is empty.
    if not output:
        try:
            directory.rmdir()
        except OSError:
            # The folder still holds files that could not be read or removed.
            pass
    return output


# Return True when a cancel command is waiting in the session's command inbox.
def latest_cancel_requested(session_id: Any) -> bool:
    return any(
        str(command.get("action") or "").strip().lower() == "cancel"
        for _name, command in read_commands(session_id)
    )
