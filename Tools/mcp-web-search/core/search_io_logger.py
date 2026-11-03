# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Lock that serializes the read-modify-write of the log file, and the path of the JSON array file.
_LOG_LOCK = threading.Lock()
_LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "model_search_io.json"


# Coerce a value to something JSON-serializable for the IO log.
def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except Exception:
        # Not serializable even with default=str, so log its repr instead.
        return repr(value)


# Drop redundant preview fields when they duplicate the snippet text.
def _without_duplicate_preview(value: Any) -> Any:
    if isinstance(value, list):
        return [_without_duplicate_preview(item) for item in value]
    if not isinstance(value, dict):
        return value

    cleaned = {key: _without_duplicate_preview(item) for key, item in value.items()}
    snippet = cleaned.get("snippet")
    preview = cleaned.get("preview")
    if isinstance(snippet, str) and isinstance(preview, str) and snippet == preview:
        cleaned.pop("preview", None)
    return cleaned


# Append one full search / read_page tool IO event (with a UTC timestamp) to a single readable JSON
# array, so the exact request and response the model saw can be inspected after the fact. This is
# diagnostics only: any failure is swallowed so it can never break the tool call itself.
def write_search_io_event(event: dict[str, Any]) -> None:
    # Stamp the event and drop previews that duplicate the snippet.
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        **dict(event or {}),
    }
    record = _without_duplicate_preview(record)
    try:
        _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _LOG_LOCK:
            # Load the existing array; a missing, empty or corrupt file starts a fresh one.
            entries: list[Any] = []
            if _LOG_PATH.exists():
                try:
                    loaded = json.loads(_LOG_PATH.read_text(encoding="utf-8") or "[]")
                    if isinstance(loaded, list):
                        entries = loaded
                except Exception:
                    # A corrupt log file is replaced rather than blocking new events.
                    entries = []

            # Append the new event and rewrite the whole file.
            entries.append(_jsonable(record))
            _LOG_PATH.write_text(
                json.dumps(entries, ensure_ascii=False, default=str, indent=2),
                encoding="utf-8",
            )
    except Exception:
        # Diagnostics must never break search itself.
        return
