# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


# Project root, the persisted probe result file under Data, and the time limit of one docker probe.
BASE_DIR = Path(__file__).resolve().parent.parent
STATE_FILE = BASE_DIR / "Data" / "docker_status.json"
_PROBE_TIMEOUT_SECONDS = 6

# Serializes reads and writes of the state file between the probe thread and request threads.
_state_lock = threading.Lock()


# Hide the transient console window a docker probe would flash on Windows.
def _no_window_flags() -> int:
    return getattr(subprocess, "CREATE_NO_WINDOW", 0)


# Return True when Docker is installed on this host, regardless of whether the
# daemon is currently running. The sandbox tool starts a stopped daemon on demand,
# so we probe the client binary ("docker --version"), which answers even when
# Docker Desktop is not running - "installed" is what gates tool availability.
def probe() -> bool:
    try:
        result = subprocess.run(
            ["docker", "--version"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_SECONDS,
            creationflags=_no_window_flags(),
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return result.returncode == 0


# Read the persisted probe result, tolerating a missing or corrupt file.
def get_status() -> dict[str, Any]:
    with _state_lock:
        try:
            payload = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            # No usable state yet: report Docker as unavailable.
            return {"available": False, "checked_at": None}
    if not isinstance(payload, dict):
        return {"available": False, "checked_at": None}
    return {
        "available": bool(payload.get("available", False)),
        "checked_at": payload.get("checked_at"),
    }


# Return only the persisted availability flag.
def is_available() -> bool:
    return get_status()["available"]


# Persist one probe result to Data/docker_status.json.
def _write_status(available: bool) -> dict[str, Any]:
    status = {"available": bool(available), "checked_at": time.time()}
    with _state_lock:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(
            json.dumps(status, indent=4, ensure_ascii=False), encoding="utf-8"
        )
    return status


# Probe Docker now and persist the fresh result.
def refresh() -> dict[str, Any]:
    return _write_status(probe())


# Probe Docker on a background thread so callers never block on it.
def refresh_async() -> None:
    thread = threading.Thread(target=refresh, name="aslm-chat-docker-probe", daemon=True)
    thread.start()
