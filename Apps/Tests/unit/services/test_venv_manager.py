# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
from pathlib import Path

import pytest

from Services import venv_manager


pytestmark = [pytest.mark.unit]


@pytest.fixture
def fake_requirements(tmp_path, monkeypatch):
    requirements = tmp_path / "venv_requirements.json"
    monkeypatch.setattr(venv_manager, "REQUIREMENTS_FILE", requirements)
    return requirements


@pytest.fixture
def isolated_base(tmp_path, monkeypatch):
    base = tmp_path / "module"
    base.mkdir()
    monkeypatch.setattr(venv_manager, "BASE_DIR", base)
    return base


# Write one requirements document with the given venv definitions.
def _write_requirements(path: Path, venvs: list[dict]) -> None:
    path.write_text(json.dumps({"fileVersion": 1, "venvs": venvs}), encoding="utf-8")


# Configuration loading.


# A missing requirements file yields an empty payload.
def test_load_config_returns_empty_payload_for_missing_file(fake_requirements):
    assert venv_manager.load_config() == {"fileVersion": 1, "venvs": []}


# A corrupt requirements file degrades to an empty payload.
def test_load_config_degrades_on_corrupt_file(fake_requirements):
    fake_requirements.write_text("{broken", encoding="utf-8")

    assert venv_manager.load_config() == {"fileVersion": 1, "venvs": []}


# Definitions normalize: blank ids drop, package lists strip and keep values.
def test_iter_venv_configs_normalizes_definitions(fake_requirements):
    _write_requirements(
        fake_requirements,
        [
            {
                "id": "server",
                "path": "Data/venvs/server",
                "packages": ["django==6.0.2", 42, ""],
                "packagesNoDeps": ["torch==9.9"],
                "tool": "",
                "tools": ["helper", ""],
            },
            {"id": "", "path": "x"},
            {"path": "missing-id"},
            "not-a-dict",
        ],
    )

    configs = venv_manager.iter_venv_configs()

    assert len(configs) == 1
    config = configs[0]
    assert config["id"] == "server"
    assert config["packages"] == ["django==6.0.2", "42"]
    assert config["packages_no_deps"] == ["torch==9.9"]
    assert config["tools"] == ["helper"]
    assert config["tool"] == ""


# Configs resolve by their id.
def test_get_venv_config_resolves_by_id(fake_requirements):
    _write_requirements(fake_requirements, [{"id": "server", "path": "Data/venvs/server"}])

    assert venv_manager.get_venv_config("server")["path"] == "Data/venvs/server"
    assert venv_manager.get_venv_config("missing") is None


# Tool directories map to their owning venv id.
def test_get_tool_venv_id_maps_tool_directories(fake_requirements):
    _write_requirements(
        fake_requirements,
        [
            {"id": "web-search", "path": "x", "tool": "mcp-web-search"},
            {"id": "multi", "path": "y", "tools": ["mcp-sandbox", "other"]},
        ],
    )

    assert venv_manager.get_tool_venv_id("mcp-web-search") == "web-search"
    assert venv_manager.get_tool_venv_id("mcp-sandbox") == "multi"
    assert venv_manager.get_tool_venv_id("unknown-tool") == ""


# Path resolution.


# Relative paths resolve against the module root.
def test_get_venv_path_joins_module_root(isolated_base, fake_requirements):
    _write_requirements(fake_requirements, [{"id": "server", "path": "Data/venvs/server"}])

    assert venv_manager.get_venv_path("server") == isolated_base / "Data" / "venvs" / "server"


# Absolute configured paths stay absolute.
def test_get_venv_path_accepts_absolute_paths(isolated_base, fake_requirements):
    absolute = isolated_base.parent / "elsewhere"
    _write_requirements(fake_requirements, [{"id": "abs", "path": str(absolute)}])

    assert venv_manager.get_venv_path("abs") == absolute


# Unknown venv ids fail with a clear error.
def test_get_venv_path_raises_for_unknown_venv(fake_requirements):
    with pytest.raises(KeyError, match="Unknown ASLM-Chat venv"):
        venv_manager.get_venv_path("missing")


# The venv Python executable points inside the venv scripts directory.
def test_get_venv_python_points_inside_venv(isolated_base, fake_requirements):
    _write_requirements(fake_requirements, [{"id": "server", "path": "Data/venvs/server"}])

    python_path = venv_manager.get_venv_python("server")

    assert python_path.parent.name == ("Scripts" if venv_manager.os.name == "nt" else "bin")


# A mapped tool without an existing venv reports no Python.
def test_get_tool_python_returns_none_when_venv_missing(isolated_base, fake_requirements):
    _write_requirements(fake_requirements, [{"id": "ghost", "path": "Data/venvs/ghost"}])

    assert venv_manager.get_tool_python("ghost") is None


# The subprocess environment is a copy of the host environment.
def test_subprocess_environment_copies_host_environment(isolated_base):
    environment = venv_manager.get_subprocess_environment(Path("python"))

    assert environment == venv_manager.os.environ
    assert environment is not venv_manager.os.environ


# Package signatures.


# The signature is order-sensitive and covers the no-deps list.
def test_packages_signature_is_stable_and_sensitive():
    base = venv_manager._packages_signature(["a==1", "b==2"])

    assert base == venv_manager._packages_signature(["a==1", "b==2"])
    assert base != venv_manager._packages_signature(["b==2", "a==1"])
    assert base != venv_manager._packages_signature(["a==1", "b==2"], ["c==3"])


# Install state.


# The install state round-trips and tolerates a corrupt file.
def test_state_roundtrip_and_corruption(isolated_base, fake_requirements):
    _write_requirements(fake_requirements, [{"id": "server", "path": "Data/venvs/server"}])
    venv_path = venv_manager.get_venv_path("server")
    venv_path.mkdir(parents=True)

    assert venv_manager._read_state(venv_path) == {}

    venv_manager._write_state(venv_path, ["a==1"], ["b==2"])
    state = venv_manager._read_state(venv_path)
    assert state["packageCount"] == 2
    assert state["packagesHash"] == venv_manager._packages_signature(["a==1"], ["b==2"])

    (venv_path / venv_manager.STATE_FILE_NAME).write_text("{broken", encoding="utf-8")
    assert venv_manager._read_state(venv_path) == {}


# ensure_venv flow.


# Fake subprocess boundary: record commands and produce venv interpreters.
class _FakeRunner:
    def __init__(self):
        self.commands: list[list[str]] = []
        self.fail_commands: set[str] = set()

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        if command and command[-1] in self.fail_commands:
            return False

        # A successful `python -m venv <path>` produces the venv interpreter.
        if command[1:3] == ["-m", "venv"]:
            venv_path = Path(command[-1])
            scripts = "Scripts" if venv_manager.os.name == "nt" else "bin"
            executable = "python.exe" if venv_manager.os.name == "nt" else "python"
            python_file = venv_path / scripts / executable
            python_file.parent.mkdir(parents=True, exist_ok=True)
            python_file.touch()
        return True

    # Return the recorded pip install commands.
    def pip_install_commands(self) -> list[list[str]]:
        return [cmd for cmd in self.commands if "pip" in cmd and "install" in cmd]

    # Return the recorded venv creation commands.
    def venv_create_commands(self) -> list[list[str]]:
        return [cmd for cmd in self.commands if cmd[1:3] == ["-m", "venv"]]


@pytest.fixture
def fake_runner(monkeypatch):
    runner = _FakeRunner()
    monkeypatch.setattr(venv_manager, "_run", runner)
    return runner


# Configure one venv on disk, optionally creating its interpreter.
def _configure_venv(fake_requirements, isolated_base, venv_id="server", packages=("a==1",), create_python=True):
    venv_path = isolated_base / "Data" / "venvs" / venv_id
    python_dir = venv_path / ("Scripts" if venv_manager.os.name == "nt" else "bin")
    python_dir.mkdir(parents=True, exist_ok=True)
    python_file = python_dir / ("python.exe" if venv_manager.os.name == "nt" else "python")
    if create_python:
        python_file.touch()
    _write_requirements(
        fake_requirements,
        [{"id": venv_id, "path": str(venv_path.relative_to(isolated_base)), "packages": list(packages)}],
    )
    return venv_path, python_file


# A missing venv is created, packages installed, and state persisted.
def test_ensure_venv_creates_installs_and_persists_state(isolated_base, fake_requirements, fake_runner):
    _configure_venv(fake_requirements, isolated_base, create_python=False)

    assert venv_manager.ensure_venv("server", log=False) is True

    assert fake_runner.venv_create_commands(), "venv creation command was not invoked"
    assert fake_runner.pip_install_commands(), "pip install was not invoked"
    assert fake_runner.pip_install_commands()[0][-1] == "a==1"
    state = venv_manager._read_state(isolated_base / "Data" / "venvs" / "server")
    assert state["packagesHash"] == venv_manager._packages_signature(["a==1"])


# A current install state skips both creation and installation.
def test_ensure_venv_skips_reinstall_when_state_matches(isolated_base, fake_requirements, fake_runner):
    venv_path, _python = _configure_venv(fake_requirements, isolated_base)
    venv_manager._write_state(venv_path, ["a==1"])

    assert venv_manager.ensure_venv("server", log=False) is True

    assert fake_runner.pip_install_commands() == []
    assert fake_runner.venv_create_commands() == []


# Changed packages trigger a reinstall with the new list.
def test_ensure_venv_reinstalls_when_packages_change(isolated_base, fake_requirements, fake_runner):
    venv_path, _python = _configure_venv(fake_requirements, isolated_base, packages=("a==1",))
    venv_manager._write_state(venv_path, ["a==1"])

    _write_requirements(
        fake_requirements,
        [{"id": "server", "path": "Data/venvs/server", "packages": ["a==1", "b==2"]}],
    )

    assert venv_manager.ensure_venv("server", log=False) is True

    pip_commands = fake_runner.pip_install_commands()
    assert pip_commands and pip_commands[0][-2:] == ["a==1", "b==2"]


# The no-deps list installs as a separate pip invocation.
def test_ensure_venv_installs_no_deps_list_separately(isolated_base, fake_requirements, fake_runner):
    venv_path, _python = _configure_venv(fake_requirements, isolated_base)
    _write_requirements(
        fake_requirements,
        [
            {
                "id": "server",
                "path": "Data/venvs/server",
                "packages": ["a==1"],
                "packagesNoDeps": ["torch==9.9"],
            }
        ],
    )

    assert venv_manager.ensure_venv("server", log=False) is True

    pip_commands = fake_runner.pip_install_commands()
    assert len(pip_commands) == 2
    assert "--no-deps" not in pip_commands[0]
    assert "--no-deps" in pip_commands[1]


# A failed install leaves no state behind.
def test_ensure_venv_fails_when_pip_install_fails(isolated_base, fake_requirements, fake_runner):
    _configure_venv(fake_requirements, isolated_base)
    fake_runner.fail_commands.add("a==1")

    assert venv_manager.ensure_venv("server", log=False) is False
    assert venv_manager._read_state(isolated_base / "Data" / "venvs" / "server") == {}


# Unknown venv ids fail without running anything.
def test_ensure_venv_fails_for_unknown_venv(fake_runner, fake_requirements):
    _write_requirements(fake_requirements, [])

    assert venv_manager.ensure_venv("ghost", log=False) is False
    assert fake_runner.commands == []


# run_venv_code ensures the venv and executes the snippet.
def test_run_venv_code_ensures_then_executes(isolated_base, fake_requirements, fake_runner):
    _configure_venv(fake_requirements, isolated_base)
    venv_manager._write_state(isolated_base / "Data" / "venvs" / "server", ["a==1"])

    assert venv_manager.run_venv_code("server", "print('hi')", log=False) is True
    executed = [cmd for cmd in fake_runner.commands if cmd[-2:] == ["-c", "print('hi')"]]
    assert executed
