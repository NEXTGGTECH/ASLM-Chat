# Copyright NEXTGGTECH. Elastic License 2.0.

import os
import sys
from pathlib import Path
from unittest.mock import patch

# Package root of mcp-sandbox; its source directories are put on sys.path and the host workspace is set at
# import time so the sandbox package can be imported.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tmp"))
os.environ["SANDBOX_HOST_WORKSPACE"] = str(ROOT)

from sandbox.config import RG_TYPE_TO_GLOB, _load_rg_type_map, _RG_TYPE_FALLBACK
from sandbox.presenters import _CODE_EXTENSIONS, _TEXT_EXTENSIONS


# Loading the ripgrep type map

# Verify that the type map is a non-empty dict.
def test_returns_nonempty_dict():
    result = _load_rg_type_map()
    assert isinstance(result, dict)
    assert len(result) > 0, "Expected non-empty type map"


# Verify that the type map contains the common language types (py, js, ts, go, rs, java, sh).
def test_common_types_present():
    result = _load_rg_type_map()
    required = ["py", "js", "ts", "go", "rs", "java", "sh"]
    missing = [t for t in required if t not in result]
    assert not missing, f"Missing common types: {missing}"


# Verify that every glob in the type map starts with "*.".
def test_glob_format():
    result = _load_rg_type_map()
    for key, glob in result.items():
        assert glob.startswith("*."), f"Bad glob for '{key}': {glob!r}"


# Verify that the short "py" alias maps to "*.py" even though rg uses "python" as the canonical name.
def test_short_alias_derived():
    result = _load_rg_type_map()
    assert "py" in result, "'py' alias missing"
    assert result["py"] == "*.py", f"Expected '*.py', got {result['py']!r}"


# Verify that the built-in fallback map is returned when rg cannot be run.
def test_fallback_on_rg_unavailable():
    with patch("subprocess.run", side_effect=FileNotFoundError("rg not found")):
        result = _load_rg_type_map()
    assert result == _RG_TYPE_FALLBACK


# Verify that a non-zero rg exit code also returns the built-in fallback map.
def test_fallback_on_rg_nonzero_exit():
    import subprocess as sp
    mock_proc = sp.CompletedProcess(args=[], returncode=1, stdout="", stderr="err")
    with patch("subprocess.run", return_value=mock_proc):
        result = _load_rg_type_map()
    assert result == _RG_TYPE_FALLBACK


# Module-level type map

# Verify that the module-level RG_TYPE_TO_GLOB is a non-empty dict.
def test_module_level_map_is_dict():
    assert isinstance(RG_TYPE_TO_GLOB, dict)
    assert len(RG_TYPE_TO_GLOB) > 0


# Code extensions

# Verify that the set of code extensions is not empty.
def test_code_extensions_nonempty():
    assert len(_CODE_EXTENSIONS) > 0


# Verify that no extension is both a code extension and a text extension.
def test_code_extensions_no_overlap_with_text():
    overlap = _CODE_EXTENSIONS & _TEXT_EXTENSIONS
    assert not overlap, f"Overlap between code and text extensions: {overlap}"


# Verify that the code extensions include the core languages (py, js, ts, go, rs, java).
def test_code_extensions_contain_core_languages():
    required = {".py", ".js", ".ts", ".go", ".rs", ".java"}
    missing = required - _CODE_EXTENSIONS
    assert not missing, f"Missing core language extensions: {missing}"


# Verify that every code extension starts with a dot.
def test_code_extensions_all_start_with_dot():
    bad = [e for e in _CODE_EXTENSIONS if not e.startswith(".")]
    assert not bad, f"Extensions without leading dot: {bad}"
