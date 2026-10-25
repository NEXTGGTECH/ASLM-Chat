# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from io import BytesIO
from pathlib import Path

MARKITDOWN_TIMEOUT_SECONDS = 8
MARKITDOWN_MAX_INPUT_BYTES = 32 * 1024 * 1024

_MARKITDOWN_INSTANCE = None
_MARKITDOWN_IMPORT_ERROR = False


# Return a shared MarkItDown converter instance when available.
def _get_markitdown_instance():
    global _MARKITDOWN_INSTANCE, _MARKITDOWN_IMPORT_ERROR
    if _MARKITDOWN_INSTANCE is not None:
        return _MARKITDOWN_INSTANCE
    if _MARKITDOWN_IMPORT_ERROR:
        return None
    try:
        from markitdown import MarkItDown
    except Exception:
        _MARKITDOWN_IMPORT_ERROR = True
        return None
    _MARKITDOWN_INSTANCE = MarkItDown(enable_plugins=False)
    return _MARKITDOWN_INSTANCE


# Run one MarkItDown conversion for the given bytes.
def _extract_once(file_bytes: bytes, *, name: str, mime: str) -> str:
    converter = _get_markitdown_instance()
    if converter is None:
        return ""

    suffix = Path(str(name or "")).suffix
    file_extension = suffix or str(name or "")
    result = converter.convert_stream(
        BytesIO(file_bytes),
        file_extension=file_extension,
        mime_type=str(mime or ""),
    )
    return str(getattr(result, "text_content", "") or "")


# Return Markdown extracted from document bytes, or an empty string on failure.
def extract_markdown(file_bytes: bytes, *, name: str, mime: str) -> str:
    payload = bytes(file_bytes or b"")
    if not payload or len(payload) > MARKITDOWN_MAX_INPUT_BYTES:
        return ""

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(_extract_once, payload, name=str(name or ""), mime=str(mime or ""))
        try:
            return future.result(timeout=MARKITDOWN_TIMEOUT_SECONDS).strip()
        except FuturesTimeoutError:
            future.cancel()
            return ""
        except Exception:
            return ""
