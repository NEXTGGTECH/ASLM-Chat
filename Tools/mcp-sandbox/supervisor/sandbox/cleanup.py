# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import contextlib
import ctypes
import json
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

from sandbox.config import (
    WORKSPACE_CLEANUP_ENABLED,
    WORKSPACE_CLEANUP_IDLE_SECONDS,
    WORKSPACE_CLEANUP_INTERVAL_SECONDS,
    WORKSPACE_CLEANUP_RECYCLE_SECONDS,
)
from sandbox.workspace import task_root

# Staging layout inside the task root: the tmp directory that holds staged batches, the batch directory
# name prefix, the per-batch metadata file and the file that records the latest cleanup event.
TMP_DIR_NAME = "tmp"
BATCH_PREFIX = "idle-"
BATCH_METADATA = ".sandbox_cleanup_batch.json"
STATE_FILENAME = ".sandbox_cleanup_state.json"

# Task-root entries that are never staged (the staging area itself, the state file and Skills).
_RESERVED_ROOT_NAMES = {TMP_DIR_NAME, STATE_FILENAME, "Skills"}

# Monitor state shared between tool calls and the monitor thread: the lock guarding it, whether the
# monitor thread was started, the number of tool calls in flight and the time of the last tool activity.
_LOCK = threading.RLock()
_MONITOR_STARTED = False
_ACTIVE_CALLS = 0
_LAST_ACTIVITY_MONOTONIC = time.monotonic()


# Time helpers

# Return the current time as an aware UTC datetime.
def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# Return the current instant as an ISO-8601 string.
def _iso_now() -> str:
    return _utc_now().isoformat()


# Parse an ISO timestamp string into an aware datetime (naive values are taken as UTC), or return None
# when the value is not a valid timestamp.
def _parse_iso_timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


# Build a batch directory name from the batch prefix and the current UTC time, to the second
# (collisions are resolved by _unique_child).
def _safe_batch_name() -> str:
    return BATCH_PREFIX + _utc_now().strftime("%Y%m%d-%H%M%S")


# Filesystem and metadata helpers

# Pick a non-colliding child path under parent: the name itself, then name-2 up to name-999, then a
# nanosecond-timestamp suffix.
def _unique_child(parent: Path, name: str) -> Path:
    candidate = parent / name
    if not candidate.exists():
        return candidate
    for index in range(2, 1000):
        candidate = parent / f"{name}-{index}"
        if not candidate.exists():
            return candidate
    return parent / f"{name}-{time.time_ns()}"


# Atomically write JSON metadata to path.
def _write_json(path: Path, payload: dict[str, object]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


# Read JSON metadata from path (empty dict on failure).
def _read_json(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


# Persist the latest cleanup event for observability.
def _remember_cleanup_event(root: Path, event: str, **details: object) -> None:
    try:
        _write_json(
            root / STATE_FILENAME,
            {
                "event": event,
                "updated_at": _iso_now(),
                **details,
            },
        )
    except OSError:
        # The event record is informational only; failing to write it must not break cleanup.
        pass


# List task-root entries eligible for idle staging (excluding reserved names).
def _iter_stageable_root_entries(root: Path) -> list[Path]:
    try:
        entries = list(root.iterdir())
    except OSError:
        return []
    return [
        entry
        for entry in entries
        if entry.name not in _RESERVED_ROOT_NAMES
    ]


# Return True when any registered background job is still running.
def _has_running_background_jobs() -> bool:
    try:
        from sandbox.jobs import JOB_REGISTRY
    except Exception:
        return False

    try:
        return any(job.get("status") == "running" for job in JOB_REGISTRY.list_jobs())
    except Exception:
        return False


# Ensure tmp/ exists under task root and cannot escape via symlinks.
def _safe_tmp_root(root: Path) -> Path:
    tmp_root = root / TMP_DIR_NAME
    if tmp_root.exists() and tmp_root.is_symlink():
        tmp_root.unlink()
    tmp_root.mkdir(parents=True, exist_ok=True)
    try:
        tmp_root.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Cleanup tmp path escaped sandbox root: {tmp_root}") from exc
    return tmp_root


# Move current root entries into one tmp batch inside the sandbox. Return the batch directory, or None
# when a background job is running, there is nothing to stage or nothing could be moved.
def stage_workspace_to_tmp(staged_at: datetime | None = None) -> Path | None:
    # Never move files out from under a running background job.
    if _has_running_background_jobs():
        return None

    root = task_root()
    root.mkdir(parents=True, exist_ok=True)
    entries = _iter_stageable_root_entries(root)
    if not entries:
        return None

    # Create a fresh batch directory inside the guarded tmp root.
    tmp_root = _safe_tmp_root(root)
    batch_dir = _unique_child(tmp_root, _safe_batch_name())
    batch_dir.mkdir(parents=True, exist_ok=False)

    # Move every entry into the batch; entries that cannot be moved stay where they are.
    moved: list[str] = []
    for entry in entries:
        if not entry.exists():
            continue
        destination = batch_dir / entry.name
        try:
            shutil.move(str(entry), str(destination))
            moved.append(entry.name)
        except (OSError, shutil.Error):
            # Locked or vanished entry: leave it in place and continue with the rest.
            continue

    # Drop the empty batch directory when nothing was moved.
    if not moved:
        with contextlib.suppress(OSError):
            batch_dir.rmdir()
        return None

    # Record when and what was staged so the batch can be recycled later.
    staged_at = staged_at or _utc_now()
    metadata = {
        "staged_at": staged_at.isoformat(),
        "source": str(root),
        "moved_entries": moved,
    }
    with contextlib.suppress(OSError):
        _write_json(batch_dir / BATCH_METADATA, metadata)
    _remember_cleanup_event(
        root,
        "staged",
        batch=str(batch_dir),
        moved_count=len(moved),
    )
    return batch_dir


# Resolve when a tmp batch was staged: from its metadata file, falling back to the directory mtime.
# Return None when neither is available.
def _batch_staged_at(batch_dir: Path) -> datetime | None:
    metadata = _read_json(batch_dir / BATCH_METADATA)
    staged_at = _parse_iso_timestamp(metadata.get("staged_at"))
    if staged_at is not None:
        return staged_at
    try:
        return datetime.fromtimestamp(batch_dir.stat().st_mtime, timezone.utc)
    except OSError:
        return None


# Send a path to the Windows recycle bin via SHFileOperationW.
def _send_to_windows_recycle_bin(path: Path) -> None:
    # ctypes mirror of the Win32 SHFILEOPSTRUCTW structure passed to SHFileOperationW.
    class SHFILEOPSTRUCTW(ctypes.Structure):
        _fields_ = [
            ("hwnd", ctypes.c_void_p),
            ("wFunc", ctypes.c_uint),
            ("pFrom", ctypes.c_wchar_p),
            ("pTo", ctypes.c_wchar_p),
            ("fFlags", ctypes.c_uint),
            ("fAnyOperationsAborted", ctypes.c_bool),
            ("hNameMappings", ctypes.c_void_p),
            ("lpszProgressTitle", ctypes.c_wchar_p),
        ]

    # Request FO_DELETE (3) with the flags undo (move to the recycle bin), no confirmation, silent and
    # no error UI. pFrom must be double-null terminated.
    operation = SHFILEOPSTRUCTW()
    operation.wFunc = 3
    operation.pFrom = str(path) + "\0\0"
    operation.fFlags = 0x40 | 0x10 | 0x04 | 0x400

    result = ctypes.windll.shell32.SHFileOperationW(ctypes.byref(operation))
    if result != 0 or operation.fAnyOperationsAborted:
        raise OSError(f"SHFileOperationW failed with code {result}")


# Send a path to the platform trash (Windows, send2trash, or gio).
def _send_to_platform_trash(path: Path) -> None:
    if os.name == "nt":
        _send_to_windows_recycle_bin(path)
        return

    # Prefer the optional send2trash package; without it, fall back to the gio command line tool.
    try:
        from send2trash import send2trash  # type: ignore
    except Exception:
        # The package is optional; a missing or broken install just selects the gio fallback.
        send2trash = None

    if send2trash is not None:
        send2trash(str(path))
        return

    gio = shutil.which("gio")
    if gio:
        subprocess.run([gio, "trash", str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return

    raise OSError("No platform trash backend is available.")


# Move tmp batches older than the recycle period to the OS trash. Batches are recognized by the idle-
# name prefix; the paths of the batches that were recycled are returned.
def recycle_due_tmp_batches(now: datetime | None = None) -> list[Path]:
    root = task_root()
    tmp_root = _safe_tmp_root(root)
    if not tmp_root.exists() or not tmp_root.is_dir():
        return []

    now = now or _utc_now()
    recycled: list[Path] = []
    for batch_dir in sorted(tmp_root.iterdir(), key=lambda path: path.name):
        # Skip foreign entries, batches of unknown age and batches that are not old enough yet.
        if not batch_dir.is_dir() or not batch_dir.name.startswith(BATCH_PREFIX):
            continue
        staged_at = _batch_staged_at(batch_dir)
        if staged_at is None:
            continue
        if (now - staged_at).total_seconds() < WORKSPACE_CLEANUP_RECYCLE_SECONDS:
            continue

        # Trash the expired batch; a failure is recorded and the batch is retried on a later pass.
        try:
            _send_to_platform_trash(batch_dir)
        except OSError as exc:
            _remember_cleanup_event(
                root,
                "recycle_failed",
                batch=str(batch_dir),
                error=str(exc),
            )
            continue
        recycled.append(batch_dir)
        _remember_cleanup_event(root, "recycled", batch=str(batch_dir))

    return recycled


# Monitor

# Perform one cleanup pass: recycle expired batches, then stage the workspace when no tool call or
# background job is active and the sandbox has been idle long enough.
def run_cleanup_once() -> None:
    if not WORKSPACE_CLEANUP_ENABLED:
        return

    with _LOCK:
        now = _utc_now()
        recycle_due_tmp_batches(now=now)
        if _ACTIVE_CALLS > 0:
            return
        if _has_running_background_jobs():
            return
        idle_for = time.monotonic() - _LAST_ACTIVITY_MONOTONIC
        if idle_for >= WORKSPACE_CLEANUP_IDLE_SECONDS:
            stage_workspace_to_tmp(staged_at=now)


# Background loop that periodically runs idle workspace cleanup.
def _monitor_loop() -> None:
    interval = max(1, WORKSPACE_CLEANUP_INTERVAL_SECONDS)
    while True:
        time.sleep(interval)

        # A failed pass must not kill the monitor thread; the next tick tries again.
        with contextlib.suppress(Exception):
            run_cleanup_once()


# Start the daemon cleanup monitor thread once per process.
def ensure_cleanup_monitor_started() -> None:
    global _MONITOR_STARTED
    if not WORKSPACE_CLEANUP_ENABLED:
        return
    with _LOCK:
        if _MONITOR_STARTED:
            return
        thread = threading.Thread(
            target=_monitor_loop,
            name="sandbox-workspace-cleanup",
            daemon=True,
        )
        thread.start()
        _MONITOR_STARTED = True


# Track active tool calls so idle cleanup does not race tool execution.
@contextlib.contextmanager
def sandbox_tool_activity() -> Iterator[None]:
    global _ACTIVE_CALLS, _LAST_ACTIVITY_MONOTONIC
    ensure_cleanup_monitor_started()
    with _LOCK:
        _ACTIVE_CALLS += 1
        _LAST_ACTIVITY_MONOTONIC = time.monotonic()
    try:
        yield
    finally:
        with _LOCK:
            _ACTIVE_CALLS = max(0, _ACTIVE_CALLS - 1)
            _LAST_ACTIVITY_MONOTONIC = time.monotonic()
