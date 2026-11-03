# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Make the project root (Settings, API packages) and the Tools directory importable when the script runs directly.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
TOOLS = ROOT / "Tools"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))


# Point Settings at the compression engine port before importing API modules.
def _configure_runtime() -> tuple[str, int]:
    # Read the engine and port from the environment, with local defaults.
    port = int(os.environ.get("COMPRESSION_PORT", os.environ.get("OLLAMA_PORT", "20004")))
    engine = os.environ.get("COMPRESSION_ENGINE", "ollama-service").strip() or "ollama-service"

    # Import lazily, after sys.path is prepared, and override the engine endpoint in the loaded settings.
    from Settings import settings as settings_module

    snapshot = settings_module.load_settings()
    if engine == "ollama-service":
        snapshot["ollama-service_port"] = port
    elif engine == "lms":
        snapshot["lms_url"] = os.environ.get("COMPRESSION_LMS_URL", f"127.0.0.1:{port}")
    elif engine in ("openai", "openai-api"):
        # Any OpenAI-compatible local server (LM Studio serves this on its port).
        snapshot["openai_url"] = os.environ.get("COMPRESSION_OPENAI_URL", f"127.0.0.1:{port}/v1")

    # Store the patched snapshot in the settings cache so API modules imported next use this endpoint.
    settings_module._store_settings_cache(snapshot, settings_module._get_settings_mtime_ns())
    return engine, port


# Configure the engine endpoint at import time, before the API modules below are imported.
ENGINE, _RUNTIME_PORT = _configure_runtime()

from context_compression.history_compressor import (
    _is_assistant_navigation,
    _looks_like_valid_path,
    build_structured_history_summary,
)
from context_compression.tests.cache_chat_utils import (
    REPORTS_DIR,
    collect_chat_entries,
    connect_cache_db,
    load_fattest_chat,
    resolve_cache_db_path,
)

from API import llm_api

# Lowercased substrings that mark a summary field value as raw tool or search output.
RAW_NOISE_MARKERS = (
    "citation rules",
    "cite search evidence",
    "search results for:",
    "**site:**",
    "**url:**",
    "traceback",
    "exit_code",
)

# Lowercased assistant openers that count as navigation filler in source memory.
NAV_PREFIXES = (
    "assistant: now let me",
    "assistant: let me ",
)

# Substrings that reveal a source-code fragment mistaken for a file path.
FALSE_PATH_MARKERS = (r"\n ", "=", "readlines", "open(", ".strip", ".match", ".Draw", ".Add")


# Count list values that still contain raw tool dumps or noisy URLs.
def _count_noise(values: list[str]) -> int:
    return sum(
        1
        for value in values
        if any(marker in str(value).lower() for marker in RAW_NOISE_MARKERS)
        or str(value).lower().startswith("tool:")
        or str(value).lower().count("http://") + str(value).lower().count("https://") > 1
    )


# Flag artifact file paths that fail validation or look like code fragments.
def _bad_files(files: list[str]) -> list[str]:
    bad: list[str] = []
    for name in files:
        lowered = str(name).lower()
        if not _looks_like_valid_path(name):
            bad.append(f"invalid_path:{name!r}")
            continue
        if any(marker in lowered for marker in FALSE_PATH_MARKERS):
            bad.append(f"code_fragment:{name!r}")
    return bad


# Flag source_memory lines that are assistant navigation filler.
def _bad_source_memory(items: list[str]) -> list[str]:
    bad: list[str] = []
    for item in items:
        text = str(item)
        if _is_assistant_navigation(text):
            bad.append(f"navigation:{text[:80]!r}")
            continue
        lowered = text.lower()
        if any(lowered.startswith(prefix) for prefix in NAV_PREFIXES):
            bad.append(f"nav_prefix:{text[:80]!r}")
    return bad


# Choose a local model name from the active engine, preferring smaller local tags.
def _pick_model(engine: str) -> str:
    # Collect model names from dict-shaped and object-shaped list entries.
    models = llm_api.get_models(engine)
    names: list[str] = []
    for item in models:
        if isinstance(item, dict):
            name = str(item.get("name") or item.get("model") or item.get("id") or "").strip()
        else:
            name = str(getattr(item, "model", None) or getattr(item, "name", None) or item or "").strip()
        if name:
            names.append(name)
    if not names:
        raise RuntimeError(f"No models from engine {engine!r}")

    # Pick the first name matching a preferred family in order, then any non-cloud model, then the first one.
    preferred = [
        "gpt-oss:20b",
        "gpt-oss",
        "qwen2.5",
        "qwen2",
        "llama3.2",
        "gemma",
        "mistral",
        "qwen3.6",
    ]
    for hint in preferred:
        for name in names:
            if hint in name.lower() and ":cloud" not in name.lower():
                return name
    for name in names:
        if ":cloud" not in name.lower():
            return name
    return names[0]


# Extract visible assistant text from a streamed or dict-shaped LLM chunk.
def _chunk_visible_text(chunk: object) -> str:
    if isinstance(chunk, tuple) and len(chunk) == 2 and chunk[0] == "message":
        message = chunk[1]
    elif isinstance(chunk, dict):
        message = chunk.get("message")
    else:
        message = getattr(chunk, "message", None)

    if isinstance(message, dict):
        return str(message.get("content") or "")
    if message is not None:
        return str(getattr(message, "content", "") or "")
    if isinstance(chunk, str):
        return chunk
    return ""


# Build a non-streaming summarize callback for the compression prompt.
def _summarize_with_model(engine: str, model_name: str):
    # Send the prompt to the model with temperature 0 and return the concatenated visible reply text;
    # an empty reply raises RuntimeError.
    def _call(prompt_messages: list[dict[str, str]]) -> str:
        chunks = llm_api.generate(
            engine=engine,
            model_name=model_name,
            messages=prompt_messages,
            stream=False,
            options={"temperature": 0.0, "num_predict": 8192},
            think=False,
        )
        parts: list[str] = []
        for chunk in chunks:
            text = _chunk_visible_text(chunk)
            if text:
                parts.append(text)
        text = "".join(parts).strip()
        if not text:
            raise RuntimeError("Model returned empty compression output.")
        return text

    return _call


# Summarize sanitization metrics for one compression payload.
def _report(label: str, payload: dict) -> dict[str, int | list[str]]:
    # Read the list fields defensively, as a model payload may contain wrong types.
    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    files = artifacts.get("files") if isinstance(artifacts.get("files"), list) else []
    key_facts = payload.get("key_facts") if isinstance(payload.get("key_facts"), list) else []
    source_memory = payload.get("source_memory") if isinstance(payload.get("source_memory"), list) else []
    short_facts = [fact for fact in key_facts if len(str(fact)) < 15]

    # Counts and sanitization problems; key facts shorter than 15 chars are listed as suspicious.
    report = {
        "label": label,
        "work_summary_chars": len(str(payload.get("work_summary") or "")),
        "key_facts": len(key_facts),
        "source_memory": len(source_memory),
        "files": len(files),
        "noise_key_facts": _count_noise([str(v) for v in key_facts]),
        "noise_source_memory": _count_noise([str(v) for v in source_memory]),
        "bad_files": _bad_files([str(v) for v in files]),
        "bad_source_memory": _bad_source_memory([str(v) for v in source_memory]),
        "short_key_facts": [str(v) for v in short_facts[:5]],
    }
    return report


# Run raw-only and model-backed compression against the bundled fat chat and write a report.
def main() -> None:
    port = _RUNTIME_PORT
    engine = ENGINE

    # Load the largest chat from the cache database and choose the model (COMPRESSION_MODEL or auto-picked).
    db_path = resolve_cache_db_path()
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")
    conn = connect_cache_db(db_path)
    fat = load_fattest_chat(conn)
    if fat is None:
        raise SystemExit(f"No chats found in cache database: {db_path}")
    entries, recent = collect_chat_entries(conn, fat["id"])

    model_name = os.environ.get("COMPRESSION_MODEL", "").strip() or _pick_model(engine)
    print("db:", db_path)
    print("chat:", fat["id"], fat["title"], "chars=", fat["chars"], "entries=", len(entries))
    print("engine:", engine, "port:", port, "model:", model_name)

    # Baseline run without a model: only the raw harvest.
    _summary_raw, raw_payload = build_structured_history_summary(
        overflow_entries=entries,
        recent_user_messages=recent,
        direct_user_directives=[],
        summarize_with_model=None,
        max_overflow_entries=120,
    )
    raw_report = _report("raw_only", raw_payload)
    print("\n=== raw_only ===")
    print(json.dumps({k: v for k, v in raw_report.items() if k != "label"}, ensure_ascii=False, indent=2))

    # Live run through the real model; model_parsed only checks the work summary for the fallback warning text.
    summarize = _summarize_with_model(engine, model_name)
    summary_text, payload = build_structured_history_summary(
        overflow_entries=entries,
        recent_user_messages=recent,
        direct_user_directives=[],
        summarize_with_model=summarize,
        max_overflow_entries=120,
    )
    parsed = "could not be parsed" not in str(payload.get("work_summary") or "").lower()
    model_report = _report("with_model", payload)
    print("model_parsed:", parsed)
    print("\n=== with_model ===")
    print(json.dumps({k: v for k, v in model_report.items() if k != "label"}, ensure_ascii=False, indent=2))
    print("summary_chars:", len(summary_text))
    print("sample_files:", (payload.get("artifacts") or {}).get("files", [])[:8])
    print("sample_key_facts:", (payload.get("key_facts") or [])[:3])
    print("sample_source_memory:", (payload.get("source_memory") or [])[:3])

    # Save both reports and the summary payload as JSON.
    out_path = REPORTS_DIR / "live_fat_compression_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "chat": {"id": fat["id"], "title": fat["title"], "chars": fat["chars"], "entries": len(entries)},
                "engine": engine,
                "port": port,
                "model": model_name,
                "raw_report": raw_report,
                "model_report": model_report,
                "summary_payload": payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print("\nWrote:", out_path)

    # Exit with an error when either run produced invalid or code-fragment file paths,
    # or navigation filler in source memory.
    failures: list[str] = []
    for key in ("bad_files", "bad_source_memory"):
        for rep in (raw_report, model_report):
            items = rep.get(key) or []
            if items:
                failures.append(f"{rep['label']}:{key}={items[:5]}")
    if failures:
        raise SystemExit("Sanitization regressions:\n" + "\n".join(failures))


# Run directly against a live local model: python run_live_fat_compression.py
if __name__ == "__main__":
    main()
