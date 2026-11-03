# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

# This server's folder and the default location of the model runtime metadata file next to it.
SERVER_ROOT = Path(__file__).resolve().parent
MODEL_RUNTIME_METADATA_PATH = SERVER_ROOT.parent / "model_runtime_metadata.json"

# Where screenshots are stored per sandbox mode: the path parts below the module's Tools folder and the
# prefix of the path reported to the model.
SANDBOX_SCREEN_TARGETS = {
    "linux_sandbox": (("mcp-sandbox", "_sandbox", "screens"), "screens"),
}


# Load model runtime metadata from the module dir or the default tools path.
def _load_model_runtime_metadata(module_dir: str | None = None) -> dict[str, Any]:
    candidates = []
    if module_dir:
        candidates.append(Path(module_dir) / "Tools" / "model_runtime_metadata.json")
    candidates.append(MODEL_RUNTIME_METADATA_PATH)

    # Use the first candidate that holds a JSON object.
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A missing or unreadable file just means this candidate is not usable; try the next one.
            continue
        if isinstance(payload, dict):
            return payload
    return {}


# Return whether the active model supports vision according to runtime metadata. The result is a tuple of
# the vision flag, the model's metadata record and a reason string ("matched" or why the lookup failed:
# "missing_metadata", "missing_models", "missing_model_record").
def _model_supports_vision(context: dict[str, Any] | None) -> tuple[bool, dict[str, Any], str]:
    safe_context = context or {}
    payload = _load_model_runtime_metadata(str(safe_context.get("module_dir") or ""))
    if not payload:
        return False, {}, "missing_metadata"

    # The engine and model come from the call context, falling back to the active model in the metadata.
    active = payload.get("active", {})
    if not isinstance(active, dict):
        active = {}

    engine = str(safe_context.get("engine") or active.get("engine") or "").strip()
    model_name = str(safe_context.get("model_name") or active.get("model") or "").strip()
    models = payload.get("models", {})
    if not isinstance(models, dict):
        return False, {}, "missing_models"

    # Look up the record under "engine:model"; if absent, retry with the active model of the metadata.
    record = models.get(f"{engine}:{model_name}") if engine and model_name else None
    if not isinstance(record, dict):
        active_engine = str(active.get("engine") or "").strip()
        active_model = str(active.get("model") or "").strip()
        record = models.get(f"{active_engine}:{active_model}") if active_engine and active_model else None

    if not isinstance(record, dict):
        return False, {}, "missing_model_record"

    # Vision is a capability flag of the record.
    capabilities = record.get("capabilities", {})
    if not isinstance(capabilities, dict):
        capabilities = {}
    return bool(capabilities.get("vision", False)), record, "matched"


# Parse width and height from the IHDR chunk of a PNG file header; return None for data that is not a PNG.
def _png_dimensions(data: bytes) -> dict[str, int] | None:
    if len(data) < 24 or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return None
    return {
        "width": int.from_bytes(data[16:20], "big"),
        "height": int.from_bytes(data[20:24], "big"),
    }


# Resolve the sandbox screenshots directory and the path prefix reported to the model when sandbox mode is
# active; return (None, "") otherwise.
def _sandbox_screens_dir(context: dict[str, Any] | None) -> tuple[Path | None, str]:
    safe_context = context or {}

    # Sandbox mode is on when the context says so or the sandbox tool server is selected.
    sandbox_enabled = bool(safe_context.get("sandbox_enabled"))
    selected = safe_context.get("selected_tool_server_ids")
    if isinstance(selected, list):
        sandbox_enabled = sandbox_enabled or any(str(item) == "sandbox" for item in selected)

    module_dir = str(safe_context.get("module_dir") or safe_context.get("project_dir") or "").strip()
    if not module_dir:
        return None, ""

    mode = "linux_sandbox" if sandbox_enabled else ""
    target = SANDBOX_SCREEN_TARGETS.get(mode)
    if not target:
        return None, ""
    path_parts, model_prefix = target
    return Path(module_dir).joinpath("Tools", *path_parts), model_prefix


# Build the image payload dict with optional inline preview for vision models.
def _image_result_from_png(
    *,
    data: bytes,
    path: str,
    host_path: str,
    supports_vision: bool,
    model_record: dict[str, Any],
    metadata_source: str,
) -> dict[str, Any]:
    image: dict[str, Any] = {
        "kind": "image",
        "path": path,
        "host_path": host_path,
        "mime": "image/png",
        "size_bytes": len(data),
        "encoding": None,
    }
    dimensions = _png_dimensions(data)
    if dimensions:
        image.update(dimensions)

    # Vision models get the PNG inline as base64.
    if supports_vision:
        image["preview"] = {
            "type": "inline_base64",
            "mime_type": "image/png",
            "data_base64": base64.b64encode(data).decode("utf-8"),
        }
        return image

    # Other models get a text placeholder and the reason the preview was withheld.
    image["preview"] = {
        "type": "text_placeholder",
        "message": (
            "Visual preview was withheld because the active model metadata says "
            "this model does not support vision."
        ),
    }
    image["vision_gate"] = {
        "allowed": False,
        "metadata_source": metadata_source,
        "engine": model_record.get("engine", ""),
        "model": model_record.get("model", ""),
    }
    return image


# Wrap a captured image into the tool result shape expected by MCP callers.
def _structured_image_result(image: dict[str, Any], *, supports_vision: bool) -> dict[str, Any]:
    # With an inline preview the image is returned as the tool result.
    if supports_vision and image.get("preview", {}).get("type") == "inline_base64":
        return {
            "ok": True,
            "tool": "browser_screenshot",
            "result": image,
            "error": None,
            "warnings": [],
            "truncated": False,
        }

    # Without vision, return the image metadata plus a text note telling the model where the file is.
    path = str(image.get("path") or image.get("host_path") or "screenshot.png")
    result = dict(image)
    result["model_context"] = (
        f"Screenshot saved: {path}. "
        "The active model does not support vision, so only image metadata is available. "
        "If sandbox is enabled, use view_image on this path with a vision-capable model or edit the file from the workspace."
    )
    result["ui"] = {
        "kind": "browser_screenshot",
        "status": "done",
        "image": dict(image),
    }
    return result


# Capture a PNG screenshot and return structured image metadata for the model.
async def capture_browser_screenshot(full_page: bool, context: dict[str, Any] | None = None) -> dict[str, Any]:
    from browser import DOWNLOADS_DIR, log, state

    # Save the screenshot into the sandbox screens folder, or the downloads folder outside sandbox mode.
    sandbox_dir, sandbox_prefix = _sandbox_screens_dir(context)
    output_dir = sandbox_dir or (DOWNLOADS_DIR / "screens")
    output_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"screenshot_{int(time.time())}.png"
    file_path = output_dir / file_name
    await state.page.screenshot(path=str(file_path), full_page=full_page)
    log.info("Screenshot saved: %s", file_path)

    # Report the file to the model, attaching an inline preview only when the active model supports vision.
    data = file_path.read_bytes()
    model_path = f"{sandbox_prefix}/{file_name}" if sandbox_prefix else str(file_path)
    supports_vision, model_record, metadata_source = _model_supports_vision(context)
    image = _image_result_from_png(
        data=data,
        path=model_path,
        host_path=str(file_path),
        supports_vision=supports_vision,
        model_record=model_record,
        metadata_source=metadata_source,
    )
    return _structured_image_result(image, supports_vision=supports_vision)
