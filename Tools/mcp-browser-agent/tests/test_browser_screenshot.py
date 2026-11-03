# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from browser_screenshot import _load_model_runtime_metadata, _model_supports_vision, _png_dimensions


# Build a minimal valid PNG with the given IHDR dimensions (test helper).
def _minimal_png(width: int, height: int) -> bytes:
    # PNG signature, then the IHDR chunk (length, type, dimensions and fixed format bytes, CRC) and IEND.
    signature = b"\x89PNG\r\n\x1a\n"
    ihdr_data = (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + b"\x08\x02\x00\x00\x00"
    )
    ihdr_chunk = (
        len(ihdr_data).to_bytes(4, "big")
        + b"IHDR"
        + ihdr_data
        + b"\x00\x00\x00\x00"
    )
    iend_chunk = b"\x00\x00\x00\x00IEND\xaeB`\x82"
    return signature + ihdr_chunk + iend_chunk


# Verify that PNG dimensions are read from the IHDR chunk and that non-PNG bytes give None.
@pytest.mark.unit
def test_png_dimensions_reads_width_and_height() -> None:
    data = _minimal_png(640, 480)
    dims = _png_dimensions(data)
    assert dims == {"width": 640, "height": 480}
    assert _png_dimensions(b"not-a-png") is None


# Verify that the vision flag is read from a runtime metadata file found under the module directory, and
# that the metadata loader returns the models mapping from the same file.
@pytest.mark.unit
def test_model_supports_vision_from_runtime_metadata(tmp_path: Path) -> None:
    # Write metadata for one active vision-capable model into <module_dir>/Tools.
    meta = {
        "active": {"engine": "ollama-service", "model": "vision-model"},
        "models": {
            "ollama-service:vision-model": {
                "capabilities": {"vision": True},
            }
        },
    }
    path = tmp_path / "Tools" / "model_runtime_metadata.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(meta), encoding="utf-8")

    # The vision lookup finds the record by engine and model name.
    supported, record, reason = _model_supports_vision(
        {"module_dir": str(tmp_path), "engine": "ollama-service", "model_name": "vision-model"}
    )
    assert supported
    assert reason == "matched"
    assert isinstance(record, dict)

    # The loader reads the same file.
    payload = _load_model_runtime_metadata(str(tmp_path))
    assert isinstance(payload.get("models"), dict)
