# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


BASE_DIR = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILE = BASE_DIR / "Settings" / "venv_requirements.json"
STATE_FILE_NAME = ".aslm_venv_state.json"
ACTIVE_VENV_ENV = "ASLM_CHAT_ACTIVE_VENV"


# Format a log line for a specific venv/process.
def _log_message(message: str, *, log_prefix: str | None = None) -> str:
    prefix = "[ASLM-Chat]"
    if log_prefix:
        prefix = f"{prefix}[{log_prefix}]"
    return f"{prefix} {message}"


# Print one log line for a specific venv/process.
def _print_log(message: str, *, log_prefix: str | None = None) -> None:
    print(_log_message(message, log_prefix=log_prefix), flush=True)


# Configuration loading

# Read venv requirements from Settings/venv_requirements.json.
def load_config() -> dict[str, Any]:
    if not REQUIREMENTS_FILE.exists():
        return {"fileVersion": 1, "venvs": []}

    try:
        payload = json.loads(REQUIREMENTS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"fileVersion": 1, "venvs": []}

    return payload if isinstance(payload, dict) else {"fileVersion": 1, "venvs": []}


# Return normalized venv definitions from the requirements file.
def iter_venv_configs() -> list[dict[str, Any]]:
    raw_venvs = load_config().get("venvs", [])
    if not isinstance(raw_venvs, list):
        return []

    venvs: list[dict[str, Any]] = []
    for raw_item in raw_venvs:
        if not isinstance(raw_item, dict):
            continue

        venv_id = str(raw_item.get("id", "") or "").strip()
        relative_path = str(raw_item.get("path", "") or "").strip()
        if not venv_id or not relative_path:
            continue

        packages = raw_item.get("packages", [])
        if not isinstance(packages, list):
            packages = []
        packages_no_deps = raw_item.get("packagesNoDeps", [])
        if not isinstance(packages_no_deps, list):
            packages_no_deps = []
        tools = raw_item.get("tools", [])
        if not isinstance(tools, list):
            tools = []

        venvs.append(
            {
                "id": venv_id,
                "path": relative_path,
                "tool": str(raw_item.get("tool", "") or "").strip(),
                "tools": [str(tool).strip() for tool in tools if str(tool).strip()],
                "packages": [str(package).strip() for package in packages if str(package).strip()],
                "packages_no_deps": [
                    str(package).strip() for package in packages_no_deps if str(package).strip()
                ],
            }
        )

    return venvs


# Resolve one venv config by id.
def get_venv_config(venv_id: str) -> dict[str, Any] | None:
    normalized_id = str(venv_id or "").strip()
    for config in iter_venv_configs():
        if config["id"] == normalized_id:
            return config
    return None


# Resolve the venv id assigned to a tool directory name.
def get_tool_venv_id(tool_dir_name: str) -> str:
    normalized_name = str(tool_dir_name or "").strip()
    for config in iter_venv_configs():
        if config.get("tool") == normalized_name or normalized_name in config.get("tools", []):
            return str(config["id"])
    return ""


# Path resolution

# Return the absolute filesystem path for a configured venv.
def get_venv_path(venv_id: str) -> Path:
    config = get_venv_config(venv_id)
    if config is None:
        raise KeyError(f"Unknown ASLM-Chat venv: {venv_id}")

    path = Path(config["path"])
    return path if path.is_absolute() else BASE_DIR / path


# Return the Python executable inside a configured venv.
def get_venv_python(venv_id: str) -> Path:
    venv_path = get_venv_path(venv_id)
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable_name = "python.exe" if os.name == "nt" else "python"
    return venv_path / scripts_dir / executable_name


# Return the tool venv Python executable when the tool is mapped in config.
def get_tool_python(tool_dir_name: str) -> Path | None:
    venv_id = get_tool_venv_id(tool_dir_name)
    if not venv_id:
        return None

    python_path = get_venv_python(venv_id)
    return python_path if python_path.exists() else None


# Subprocess environments

# Prepare child-process storage without changing the host process or adding module data roots.
def get_subprocess_environment(python_path: Path) -> dict[str, str]:
    """Keep Linux browser downloads in their venv and scratch files in the ASLM profile."""

    environment = os.environ.copy()
    if not sys.platform.startswith("linux"):
        return environment

    # Browser binaries belong to the existing preserved environment, not a separate update payload.
    browsers = python_path.parent.parent / "browsers"
    environment.update({
        "XDG_CACHE_HOME": str(browsers),
        "CLOAKBROWSER_CACHE_DIR": str(browsers / "cloakbrowser"),
        "CLOAKBROWSER_AUTO_UPDATE": "false",
        "PLAYWRIGHT_BROWSERS_PATH": str(browsers / "playwright"),
        "PIP_NO_CACHE_DIR": "1",
    })

    # Use the host's short scratch path: Chromium's Unix socket names cannot fit a deep venv path.
    root = BASE_DIR.parent.parent / "Data" / "App" if BASE_DIR.parent.name == "Modules" else BASE_DIR / "Data"
    root.mkdir(parents=True, exist_ok=True)
    for key in ("TMPDIR", "TEMP", "TMP"):
        environment[key] = str(root)
    return environment


# Internal helpers

# Run one subprocess command and optionally stream output to the console.
def _run(
    command: list[str],
    *,
    log: bool,
    cwd: Path | None = None,
    log_prefix: str | None = None,
) -> bool:
    working_dir = str(cwd or BASE_DIR)
    environment = get_subprocess_environment(Path(command[0]))
    if log:
        try:
            process = subprocess.Popen(
                command,
                cwd=working_dir,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
            )
        except Exception as exc:
            _print_log(f"Command could not be started: {exc}", log_prefix=log_prefix)
            return False

        assert process.stdout is not None
        for line in process.stdout:
            text = line.rstrip()
            if text:
                _print_log(text, log_prefix=log_prefix)

        return_code = process.wait()
        if return_code != 0:
            _print_log(
                f"Command failed with exit code {return_code}: {' '.join(command)}",
                log_prefix=log_prefix,
            )
        return return_code == 0

    try:
        result = subprocess.run(
            command,
            cwd=working_dir,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    except Exception:
        return False

    return result.returncode == 0


# Compute a stable hash for one venv package manifest.
def _packages_signature(packages: list[str], packages_no_deps: list[str] | None = None) -> str:
    payload = json.dumps(
        {"packages": packages, "packagesNoDeps": packages_no_deps or []},
        ensure_ascii=True,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# Read persisted install state for one venv directory.
def _read_state(venv_path: Path) -> dict[str, Any]:
    state_path = venv_path / STATE_FILE_NAME
    if not state_path.exists():
        return {}

    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    return payload if isinstance(payload, dict) else {}


# Persist the package signature after a successful install.
def _write_state(venv_path: Path, packages: list[str], packages_no_deps: list[str] | None = None) -> None:
    state_path = venv_path / STATE_FILE_NAME
    no_deps = packages_no_deps or []
    payload = {
        "packagesHash": _packages_signature(packages, no_deps),
        "packageCount": len(packages) + len(no_deps),
    }
    state_path.write_text(json.dumps(payload, indent=4, ensure_ascii=False), encoding="utf-8")


# Create a Python virtual environment for one configured venv id.
def _create_venv(venv_id: str, log: bool, log_prefix: str | None = None) -> bool:
    venv_path = get_venv_path(venv_id)
    venv_path.parent.mkdir(parents=True, exist_ok=True)

    if log:
        _print_log(f"Creating venv '{venv_id}' at {venv_path}", log_prefix=log_prefix)

    if importlib.util.find_spec("venv") is not None and _run(
        [sys.executable, "-m", "venv", str(venv_path)],
        log=log,
        log_prefix=log_prefix,
    ):
        return True

    # Some embeddable Python runtimes do not ship the stdlib venv module.
    _run(
        [sys.executable, "-m", "pip", "install", "--no-warn-script-location", "virtualenv"],
        log=log,
        log_prefix=log_prefix,
    )
    return _run(
        [sys.executable, "-m", "virtualenv", str(venv_path)],
        log=log,
        log_prefix=log_prefix,
    )


# Install one package list into a venv via pip.
def _pip_install(
    python_path: Path,
    packages: list[str],
    *,
    no_deps: bool,
    log: bool,
    label: str,
    log_prefix: str | None = None,
) -> bool:
    if not packages:
        return True

    command = [
        str(python_path),
        "-m",
        "pip",
        "install",
        "--no-warn-script-location",
        "--prefer-binary",
    ]
    if no_deps:
        command.append("--no-deps")
    command.extend(packages)

    if log:
        suffix = " (no deps)" if no_deps else ""
        _print_log(
            f"Installing {len(packages)} package(s){suffix} into venv '{label}'",
            log_prefix=log_prefix,
        )

    return _run(command, log=log, log_prefix=log_prefix)


# Install regular and no-deps package lists for one venv id.
def _install_packages(
    venv_id: str,
    packages: list[str],
    packages_no_deps: list[str],
    log: bool,
    log_prefix: str | None = None,
) -> bool:
    if not packages and not packages_no_deps:
        return True

    python_path = get_venv_python(venv_id)
    if not python_path.exists():
        return False

    if packages and not _pip_install(
        python_path, packages, no_deps=False, log=log, label=venv_id, log_prefix=log_prefix
    ):
        return False

    if packages_no_deps and not _pip_install(
        python_path,
        packages_no_deps,
        no_deps=True,
        log=log,
        label=venv_id,
        log_prefix=log_prefix,
    ):
        return False

    return True


# Public venv lifecycle

# Create or update one configured ASLM-Chat venv when packages changed.
def ensure_venv(venv_id: str, *, log: bool = True, log_prefix: str | None = None) -> bool:
    config = get_venv_config(venv_id)
    if config is None:
        if log:
            _print_log(f"Unknown venv '{venv_id}'.", log_prefix=log_prefix)
        return False

    venv_path = get_venv_path(venv_id)
    python_path = get_venv_python(venv_id)
    packages = list(config.get("packages", []))
    packages_no_deps = list(config.get("packages_no_deps", []))

    if not python_path.exists() and not _create_venv(venv_id, log, log_prefix):
        return False

    state = _read_state(venv_path)
    if state.get("packagesHash") == _packages_signature(packages, packages_no_deps):
        if not python_path.exists():
            if log:
                _print_log(
                    f"Venv '{venv_id}' state is current but Python is missing; reinstalling.",
                    log_prefix=log_prefix,
                )
        else:
            if log:
                _print_log(f"Venv '{venv_id}' is up to date.", log_prefix=log_prefix)
            return True

    if not _install_packages(venv_id, packages, packages_no_deps, log, log_prefix):
        return False

    _write_state(venv_path, packages, packages_no_deps)
    return True


# Create or update every venv listed in the requirements file.
def ensure_all(*, log: bool = True) -> bool:
    configs = iter_venv_configs()
    if not configs:
        return True

    if log:
        _print_log(f"Preparing {len(configs)} venv(s) in parallel.")

    ok = True
    with ThreadPoolExecutor(max_workers=len(configs)) as executor:
        futures = {
            executor.submit(
                ensure_venv,
                str(config["id"]),
                log=log,
                log_prefix=str(config["id"]),
            ): str(config["id"])
            for config in configs
        }
        for future in as_completed(futures):
            venv_id = futures[future]
            try:
                venv_ok = future.result()
            except Exception as exc:
                venv_ok = False
                if log:
                    _print_log(
                        f"Venv '{venv_id}' failed with unexpected error: {exc}",
                        log_prefix=venv_id,
                    )

            ok = venv_ok and ok
            if log:
                status = "ready" if venv_ok else "failed"
                _print_log(f"Venv '{venv_id}' {status}.", log_prefix=venv_id)

    return ok


# Run a Python command inside one configured venv.
def run_venv_python(venv_id: str, args: list[str], *, log: bool = True) -> bool:
    if not ensure_venv(venv_id, log=log):
        return False

    return _run([str(get_venv_python(venv_id)), *args], log=log)


# Run a Python code snippet inside one configured venv.
def run_venv_code(venv_id: str, code: str, *, log: bool = True) -> bool:
    return run_venv_python(venv_id, ["-c", code], log=log)
