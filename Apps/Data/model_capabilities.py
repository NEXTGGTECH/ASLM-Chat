# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import time
from typing import Any

from Settings import settings

logger = logging.getLogger(__name__)
CACHE_DIR = settings.BASE_DIR / "Data" / "model_capabilities"
CACHE_TTL_SECONDS = 24 * 60 * 60
_cache_lock = threading.RLock()
_FLAG_KEYS = (
    "supports_vision", "supports_tool_calling", "supports_thinking",
    "supports_think_toggle", "supports_think_level", "supports_audio_input",
    "supports_video_input", "supports_files", "is_cloud",
)


def get_endpoint(engine: str) -> str:
    """Ollama is managed locally; other engines are scoped to their effective URL."""
    return "" if settings.is_ollama_engine(engine) else settings.get_engine_url(engine).rstrip("/")


def _read_records(engine: str) -> list[dict[str, Any]]:
    try:
        records = json.loads((CACHE_DIR / f"{engine}.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, ValueError):
        logger.warning("Could not read model capabilities for %s", engine, exc_info=True)
        return []
    if not isinstance(records, list):
        return []
    return [
        record for record in records
        if isinstance(record, dict)
        and isinstance(record.get("model"), str) and record["model"]
        and isinstance(record.get("endpoint", ""), str)
        and isinstance(record.get("updated_at"), (int, float))
        and math.isfinite(record["updated_at"])
        and isinstance(record.get("capabilities"), dict)
        and all(isinstance(record["capabilities"].get(key), bool) for key in _FLAG_KEYS)
    ]


def list_cached(engine: str, endpoint: str) -> list[dict[str, Any]]:
    """Read this endpoint's records without probing models or loading presets."""
    engine = settings.normalize_engine_name(engine)
    if engine not in settings.ENGINE_IDS:
        return []
    with _cache_lock:
        records = [record for record in _read_records(engine) if record.get("endpoint", "") == endpoint]
    # Earlier LM Studio snapshots could mark unloaded models as unsupported.
    # Recheck those records without discarding their last-known data on failure.
    for record in records:
        if engine == "lms" and not record.get("source"):
            record["stale"] = True
    return records


def is_fresh(record: dict[str, Any]) -> bool:
    age = time.time() - record["updated_at"]
    return not record.get("stale", False) and 0 <= age < CACHE_TTL_SECONDS


def store(engine: str, model: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any] | None:
    """Atomically replace one successful capability snapshot, preserving other endpoints."""
    engine = settings.normalize_engine_name(engine)
    if engine not in settings.ENGINE_IDS or not model or payload.get("metadata_fallback"):
        return None
    capabilities = {key: bool(payload.get(key, False)) for key in _FLAG_KEYS}
    capabilities["is_cloud"] = settings.is_ollama_engine(engine) and capabilities["is_cloud"]
    for key in ("think_param_name", "think_level_param_name"):
        if isinstance(payload.get(key), str):
            capabilities[key] = payload[key]
    if isinstance(payload.get("think_level_options"), list):
        capabilities["think_level_options"] = [str(value) for value in payload["think_level_options"]]
    record = {"model": model, "updated_at": time.time(), "capabilities": capabilities}
    if payload.get("capabilities_source"):
        record["source"] = str(payload["capabilities_source"])
    if not settings.is_ollama_engine(engine):
        record["endpoint"] = endpoint
    else:
        endpoint = ""

    with _cache_lock:
        temporary_path = None
        try:
            records = _read_records(engine)
            records = [
                row for row in records
                if (row["model"], row.get("endpoint", "")) != (model, endpoint)
            ]
            records.append(record)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=CACHE_DIR, suffix=".tmp", delete=False) as stream:
                temporary_path = stream.name
                json.dump(records, stream, ensure_ascii=False, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_path, CACHE_DIR / f"{engine}.json")
        except OSError:
            # A read-only cache must not prevent model selection or generation.
            logger.warning("Could not save model capabilities for %s", engine, exc_info=True)
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except OSError:
                    pass
    return record
