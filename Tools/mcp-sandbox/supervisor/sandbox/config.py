# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


# Core sandbox paths.
#
# IMPORTANT: sandbox.env is the single bridge between host and container config.
# It must be loaded BEFORE any os.getenv() call that reads sandbox-controlled
# variables, otherwise edits to sandbox.env are silently ignored.
# Only SANDBOX_IN_CONTAINER and SANDBOX_CONFIG_FILE are read pre-load —
# the first decides whether we load at all (container env wins), the second
# tells us where to load from.

IN_CONTAINER = os.getenv("SANDBOX_IN_CONTAINER", "").strip().lower() in {
    "1",
    "true",
    "yes",
}

CONFIG_FILE_PATH = os.getenv(
    "SANDBOX_CONFIG_FILE",
    os.path.join(str(Path(__file__).resolve().parents[2]), "sandbox.env"),
)


# Load sandbox.env into os.environ (does not overwrite vars already set).
def _load_sandbox_env(path: str) -> None:
    try:
        with open(path, encoding="utf-8") as fh:
            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if key and key not in os.environ:
                    os.environ[key] = value.strip()
    except OSError:
        pass


if not IN_CONTAINER:
    # Regenerate the ENV bridge before any sandbox-controlled variable is read.
    from sandbox.config_bridge import sync_sandbox_env

    sync_sandbox_env(json_path=Path(CONFIG_FILE_PATH).with_suffix(".json"), env_path=Path(CONFIG_FILE_PATH))
    _load_sandbox_env(CONFIG_FILE_PATH)

CONTAINER_NAME = os.getenv("SANDBOX_CONTAINER_NAME", "aslm-chat-sandbox")
SANDBOX_IMAGE = os.getenv("SANDBOX_IMAGE", "nggtlightkeeper/aslm-chat-sandbox:latest")
SANDBOX_IMAGE_SOURCE = os.getenv("SANDBOX_IMAGE_SOURCE", "registry").strip().lower()
SNAPSHOT_IMAGE_PREFIX = os.getenv(
    "SANDBOX_SNAPSHOT_PREFIX",
    f"{CONTAINER_NAME}-snapshot",
)
CONTAINER_WORKSPACE = "/workspace"
DEFAULT_TASK_DIR = os.getenv("SANDBOX_DEFAULT_TASK_DIR", "_sandbox")
MODEL_WORKSPACE_CONTAINER = (
    f"{CONTAINER_WORKSPACE.rstrip('/')}/{DEFAULT_TASK_DIR}"
    if DEFAULT_TASK_DIR not in ("", ".")
    else CONTAINER_WORKSPACE
)

COMMAND_USER = os.getenv("SANDBOX_COMMAND_USER", "sandbox_user")
SUPERVISOR_SRC = os.getenv("SANDBOX_SUPERVISOR_SRC", "/opt/sandbox-src")
SUPERVISOR_VENV = os.getenv("SANDBOX_SUPERVISOR_VENV", "/opt/sandbox-venv")
SUPERVISOR_VENV_HOST = os.getenv("SANDBOX_SUPERVISOR_VENV_HOST", "").strip()

_PROJECT_ROOT = (
    Path(__file__).resolve().parents[2]  # mcp-sandbox/
    if not IN_CONTAINER
    else Path(SUPERVISOR_SRC)
)
_DEFAULT_SANDBOX_DIR = str(_PROJECT_ROOT)
HOST_WORKSPACE = os.getenv(
    "SANDBOX_HOST_WORKSPACE",
    CONTAINER_WORKSPACE if IN_CONTAINER else str(_PROJECT_ROOT),
)
# In production the supervisor source is baked into the image
# (COPY supervisor /opt/sandbox-src). Bind-mounting the host source tree on top is
# a dev-only convenience; it must be explicitly opted-in via SANDBOX_DEV_BIND=1 so
# production containers run baked code.
DEV_BIND = os.getenv("SANDBOX_DEV_BIND", "").strip().lower() in {"1", "true", "yes"}
SUPERVISOR_SRC_HOST = os.getenv(
    "SANDBOX_SUPERVISOR_SRC_HOST",
    str(_PROJECT_ROOT / "supervisor") if DEV_BIND else "",
).strip()


# Import and sharing settings.

IMPORT_ROOTS_ENV = os.getenv("SANDBOX_IMPORT_ROOTS", "")

# Execution limits.

MAX_OUTPUT_BYTES = int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", "60000"))
OUTPUT_HEAD_RATIO = float(os.getenv("SANDBOX_OUTPUT_HEAD_RATIO", "0.5"))
JOB_ROOT = os.getenv(
    "SANDBOX_JOB_ROOT",
    "/tmp/mcp-sandbox-jobs"
    if IN_CONTAINER
    else os.path.join(HOST_WORKSPACE, ".sandbox_jobs"),
)
DEFAULT_TIMEOUT = int(os.getenv("SANDBOX_DEFAULT_TIMEOUT", "60"))
OCR_TIMEOUT = int(os.getenv("SANDBOX_OCR_TIMEOUT", "60"))
MAX_READ_BYTES = int(os.getenv("SANDBOX_MAX_READ_BYTES", "200000"))
MAX_CAT_FILE_BYTES = int(os.getenv("SANDBOX_MAX_CAT_FILE_BYTES", "30720"))
MAX_CAT_LINE_THRESHOLD = int(os.getenv("SANDBOX_MAX_CAT_LINE_THRESHOLD", "300"))
MAX_IMAGE_PREVIEW_BYTES = int(os.getenv("SANDBOX_MAX_IMAGE_PREVIEW_BYTES", "2000000"))
MAX_LS_ENTRIES = int(os.getenv("SANDBOX_MAX_LS_ENTRIES", "500"))
MAX_FIND_RESULTS = int(os.getenv("SANDBOX_MAX_FIND_RESULTS", "200"))
MAX_GREP_RESULTS = int(os.getenv("SANDBOX_MAX_GREP_RESULTS", "200"))
LOOP_BREAK_THRESHOLD = int(os.getenv("SANDBOX_LOOP_BREAK_THRESHOLD", "3"))

# Presentation layer tuning.

MAX_FILE_MAP_SYMBOLS = int(os.getenv("SANDBOX_MAX_FILE_MAP_SYMBOLS", "50"))
CPU_LIMIT = os.getenv("SANDBOX_CPU_LIMIT", "8")
THREAD_LIMIT = int(os.getenv("SANDBOX_THREAD_LIMIT", "8"))
MEMORY_LIMIT = os.getenv("SANDBOX_MEMORY_LIMIT", "6g")
MEMORY_SWAP_LIMIT = os.getenv("SANDBOX_MEMORY_SWAP_LIMIT", "8g")
PIDS_LIMIT = os.getenv("SANDBOX_PIDS_LIMIT", "1024")
STORAGE_LIMIT = os.getenv("SANDBOX_STORAGE_LIMIT", "12G")
NETWORK_ENABLED = os.getenv("SANDBOX_NETWORK_ENABLED", "1").strip().lower() not in {
    "0", "false", "no", "off",
}
NETWORK_LIMIT_MBIT = int(os.getenv("SANDBOX_NETWORK_LIMIT_MBIT", "0"))
DOCKER_START_TIMEOUT_SECONDS = int(
    os.getenv("SANDBOX_DOCKER_START_TIMEOUT_SECONDS", "60")
)
BACKGROUND_TIMEOUT_THRESHOLD = int(
    os.getenv("SANDBOX_BACKGROUND_TIMEOUT_THRESHOLD", "10")
)
WORKSPACE_CLEANUP_ENABLED = os.getenv(
    "SANDBOX_WORKSPACE_CLEANUP_ENABLED",
    "1",
).strip().lower() not in {"0", "false", "no", "off"}
WORKSPACE_CLEANUP_IDLE_SECONDS = int(
    os.getenv("SANDBOX_WORKSPACE_CLEANUP_IDLE_SECONDS", "5400")
)
WORKSPACE_CLEANUP_RECYCLE_SECONDS = int(
    os.getenv("SANDBOX_WORKSPACE_CLEANUP_RECYCLE_SECONDS", "10800")
)
WORKSPACE_CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("SANDBOX_WORKSPACE_CLEANUP_INTERVAL_SECONDS", "5")
)


# Docker Desktop discovery.

WINDOWS_DOCKER_DESKTOP_PATHS = [
    os.path.expandvars(r"%ProgramFiles%\Docker\Docker\Docker Desktop.exe"),
    os.path.expandvars(r"%LocalAppData%\Docker\Docker Desktop.exe"),
]

MACOS_DOCKER_DESKTOP_PATHS = [
    "/Applications/Docker.app",
    os.path.expanduser("~/Applications/Docker.app"),
]

# Linux Docker Desktop binaries (best-effort; the system daemon usually needs a service).
LINUX_DOCKER_DESKTOP_PATHS = [
    "/opt/docker-desktop/bin/docker-desktop",
    "/usr/bin/docker-desktop",
    "/usr/local/bin/docker-desktop",
]


IGNORED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".idea",
    ".vscode",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "dist",
    "build",
    "target",
    ".next",
    ".nuxt",
}


_RG_TYPE_FALLBACK = {
    "c": "*.c",
    "cc": "*.cc",
    "cpp": "*.cpp",
    "cs": "*.cs",
    "go": "*.go",
    "java": "*.java",
    "js": "*.js",
    "jsx": "*.jsx",
    "kt": "*.kt",
    "mjs": "*.mjs",
    "php": "*.php",
    "py": "*.py",
    "python": "*.py",
    "rs": "*.rs",
    "rust": "*.rs",
    "scala": "*.scala",
    "sh": "*.sh",
    "swift": "*.swift",
    "ts": "*.ts",
    "tsx": "*.tsx",
}


# Return ripgrep type aliases mapped to representative glob patterns.
def _load_rg_type_map() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["rg", "--type-list"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return dict(_RG_TYPE_FALLBACK)

    if result.returncode != 0:
        return dict(_RG_TYPE_FALLBACK)

    type_map: dict[str, str] = {}
    for raw_line in result.stdout.splitlines():
        name, sep, patterns = raw_line.partition(":")
        if not sep:
            continue
        first_pattern = patterns.split(",", 1)[0].strip()
        if not first_pattern.startswith("*."):
            continue
        type_name = name.strip()
        if not type_name:
            continue
        type_map[type_name] = first_pattern
        ext_alias = first_pattern[2:]
        if ext_alias:
            type_map.setdefault(ext_alias, first_pattern)

    for key, value in _RG_TYPE_FALLBACK.items():
        type_map.setdefault(key, value)
    return type_map or dict(_RG_TYPE_FALLBACK)


RG_TYPE_TO_GLOB = _load_rg_type_map()


# Return True when the host workspace path looks safe.
def _validate_workspace_path(path: str) -> bool:
    normalized = os.path.normpath(os.path.abspath(path)).lower()
    path_name = os.path.basename(normalized.rstrip("\\/"))
    dangerous_patterns = [
        "windows",
        "system32",
        "program files",
        "program files (x86)",
        "programdata",
        "/bin",
        "/sbin",
        "/usr",
        "/etc",
        "/var",
        "/root",
    ]

    if len(normalized) <= 3 and normalized.endswith((":\\", ":/")):
        return False

    if normalized in {"/", "\\"}:
        return False

    for pattern in dangerous_patterns:
        if pattern in normalized:
            return False

    path_parts = normalized.replace("\\", "/").strip("/").split("/")
    if DEFAULT_TASK_DIR in {"", "."} and "sandbox" not in path_name:
        return False

    return len(path_parts) >= 2


if not IN_CONTAINER and not _validate_workspace_path(HOST_WORKSPACE):
    print(
        f"CRITICAL SECURITY ERROR: HOST_WORKSPACE '{HOST_WORKSPACE}' is not safe!",
        file=sys.stderr,
    )
    print("Using fallback isolated directory instead.", file=sys.stderr)
    HOST_WORKSPACE = _DEFAULT_SANDBOX_DIR

HOST_WORKSPACE = os.path.abspath(HOST_WORKSPACE)


# Return normalized host roots allowed for imports.
def get_allowed_import_roots() -> list[str]:
    raw_roots: list[str] = []

    if IMPORT_ROOTS_ENV.strip():
        raw_roots.extend(
            item for item in IMPORT_ROOTS_ENV.split(os.pathsep) if item.strip()
        )

    normalized_roots: list[str] = []
    seen: set[str] = set()

    for raw_root in raw_roots:
        normalized = os.path.normpath(os.path.abspath(os.path.expanduser(raw_root)))
        if normalized in seen:
            continue

        seen.add(normalized)
        normalized_roots.append(normalized)

    return normalized_roots


ALLOWED_IMPORT_ROOTS = get_allowed_import_roots()

os.makedirs(HOST_WORKSPACE, exist_ok=True)
os.makedirs(os.path.join(HOST_WORKSPACE, DEFAULT_TASK_DIR), exist_ok=True)
