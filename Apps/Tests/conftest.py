# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from Settings import settings as project_settings


# Settings file constants redirected into a temporary directory.
ISOLATED_SETTINGS_FILE_ATTRS = (
    "SETTINGS_FILE",
    "WEB_SEARCH_API_KEYS_FILE",
    "WEB_SEARCH_CONFIG_FILE",
    "BROWSER_AGENT_CONFIG_FILE",
    "SANDBOX_CONFIG_FILE",
)


# Drop every ASLM_* environment variable so settings written by one test
# cannot leak into later tests through the environment-override layer.
@pytest.fixture(autouse=True)
def clean_aslm_environment():
    saved = {key: value for key, value in os.environ.items() if key.startswith("ASLM_")}
    for key in saved:
        del os.environ[key]

    yield

    for key in [key for key in os.environ if key.startswith("ASLM_") and key not in saved]:
        del os.environ[key]
    os.environ.update(saved)


# Clear the tool registry caches around every test so discovery results,
# cooldowns, quotas, and worker sessions never leak between tests.
@pytest.fixture(autouse=True)
def reset_tool_registry():
    from API import mcp as tool_registry

    tool_registry.reset_cache()
    yield
    tool_registry.reset_cache()


# Redirect every persisted settings file into a temporary directory; the
# manifest sync target is redirected as well.
@pytest.fixture
def isolated_settings(tmp_path, monkeypatch) -> Path:
    settings_dir = tmp_path / "settings"
    settings_dir.mkdir()

    for attr in ISOLATED_SETTINGS_FILE_ATTRS:
        monkeypatch.setattr(project_settings, attr, settings_dir / f"{attr.lower()}.json")

    manifest_path = settings_dir / "ASLM_Module.json"
    monkeypatch.setattr(project_settings, "_get_module_manifest_path", lambda: manifest_path)

    project_settings._invalidate_settings_cache()
    yield settings_dir
    project_settings._invalidate_settings_cache()


# Isolated Tools directory with helpers for writing fake mcp-server.py modules.
class ToolsDir:
    def __init__(self, root: Path):
        self.root = root

    # Write one temporary MCP server module and refresh discovery.
    def write_server(self, folder: str, body: str) -> Path:
        from API import mcp as tool_registry

        server_dir = self.root / folder
        server_dir.mkdir(parents=True, exist_ok=True)
        server_file = server_dir / tool_registry.SERVER_FILENAME
        server_file.write_text(textwrap.dedent(body).strip() + "\n", encoding="utf-8")
        tool_registry.reset_cache()
        return server_file

    # Return the path of a previously written server module.
    def server_file(self, folder: str) -> Path:
        from API import mcp as tool_registry

        return self.root / folder / tool_registry.SERVER_FILENAME


# Replace the bundled Tools directory with an isolated temporary one.
@pytest.fixture
def isolated_tools_dir(tmp_path, monkeypatch) -> ToolsDir:
    from API import mcp as tool_registry

    tools_dir = ToolsDir(tmp_path / "Tools")
    tools_dir.root.mkdir()

    monkeypatch.setattr(tool_registry, "TOOLS_DIR", tools_dir.root)
    tool_registry.reset_cache()
    yield tools_dir
    tool_registry.reset_cache()


# Manifest helpers.


# Write a minimal module manifest for manifest mirroring tests.
def write_test_manifest(path: Path, setting_keys: tuple[str, ...]) -> Path:
    manifest = {
        "fileVersion": 2,
        "id": "aslm-chat",
        "settings": [
            {"key": key, "value": None, "getExec": "main.py get_setting " + key}
            for key in setting_keys
        ],
    }
    path.write_text(json.dumps(manifest, indent=4), encoding="utf-8")
    return path


# Prepared temporary module manifest next to the isolated settings files.
@pytest.fixture
def isolated_manifest(isolated_settings) -> SimpleNamespace:
    manifest_path = write_test_manifest(
        isolated_settings / "ASLM_Module.json",
        ("ui-port", "llm-engine"),
    )
    return SimpleNamespace(path=manifest_path, dir=isolated_settings)
