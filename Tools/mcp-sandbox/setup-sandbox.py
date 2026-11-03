# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

# Locations of this script, its sandbox.env and the supervisor package (importable below for the env sync).
_SCRIPT_DIR = Path(__file__).resolve().parent
_ENV_FILE = _SCRIPT_DIR / "sandbox.env"
_SUPERVISOR_DIR = _SCRIPT_DIR / "supervisor"
if str(_SUPERVISOR_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPERVISOR_DIR))

from sandbox.config_bridge import sync_sandbox_env

# Default image name, and the image label and value that mark an image with the current supervisor runtime.
_DEFAULT_IMAGE = "nggtlightkeeper/aslm-chat-sandbox:latest"
_REQUIRED_LABEL = "org.aslm.sandbox.supervisor-runtime"
_REQUIRED_LABEL_VALUE = "container-v2"


# Parse key=value lines from a sandbox.env file; a missing or unreadable file yields an empty dict.
def _load_env_file(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            result[key.strip()] = value.strip()
    except OSError:
        # A missing or unreadable sandbox.env is fine: every setting then falls back to its default.
        pass
    return result


# Regenerate sandbox.env from sandbox.json at import time so the overrides read below are current.
sync_sandbox_env(env_path=_ENV_FILE)
_env_overrides = _load_env_file(_ENV_FILE)


# Resolve a config key from env, sandbox.env, or default.
def _cfg(key: str, default: str) -> str:
    return os.environ.get(key) or _env_overrides.get(key) or default


# Image to pull or build: SANDBOX_IMAGE from the environment or sandbox.env, else the default.
SANDBOX_IMAGE = _cfg("SANDBOX_IMAGE", _DEFAULT_IMAGE)


# Helpers

# Run a subprocess and capture stdout/stderr as text.
def _run(args: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
    )


# Run a command with real-time stdout/stderr visible to the user.
def _stream(args: list[str], timeout: int = 1800) -> int:
    proc = subprocess.run(args, timeout=timeout)
    return proc.returncode


# Print a success line.
def _ok(msg: str) -> None:
    print(f"  [ok] {msg}", flush=True)


# Print a progress line.
def _info(msg: str) -> None:
    print(f"  ... {msg}", flush=True)


# Print an error line to stderr.
def _fail(msg: str) -> None:
    print(f"  [!!] {msg}", file=sys.stderr, flush=True)


# Docker checks

# Verify the Docker CLI is installed and responsive.
def _check_docker() -> bool:
    try:
        r = _run(["docker", "--version"], timeout=5)
        if r.returncode != 0:
            _fail("Docker CLI found but returned an error.")
            return False
        _ok(r.stdout.strip())
        return True
    except FileNotFoundError:
        _fail("Docker CLI not found. Install Docker Desktop and try again.")
        return False
    except Exception as exc:
        _fail(f"Unexpected error checking docker: {exc}")
        return False


# Verify the Docker daemon is reachable.
def _check_daemon() -> bool:
    try:
        r = _run(["docker", "info", "--format", "{{.ServerVersion}}"], timeout=10)
        if r.returncode == 0:
            _ok(f"Docker daemon running (server {r.stdout.strip()})")
            return True
        _fail("Docker daemon is not running. Start Docker Desktop and try again.")
        return False
    except Exception as exc:
        _fail(f"Cannot reach Docker daemon: {exc}")
        return False


# Image checks

# Return True when the local image has the required runtime label.
def _image_exists_and_valid() -> bool:
    r = _run(["docker", "image", "inspect", SANDBOX_IMAGE], timeout=10)
    if r.returncode != 0:
        return False
    import json
    try:
        data = json.loads(r.stdout)
        labels = (data[0].get("Config", {}).get("Labels") or {}) if data else {}
        return labels.get(_REQUIRED_LABEL) == _REQUIRED_LABEL_VALUE
    except Exception:
        # Unparseable inspect output means the image cannot be trusted; treat it as invalid.
        return False


# Pull and build

# Pull the sandbox image from the registry.
def _pull() -> bool:
    _info(f"Pulling {SANDBOX_IMAGE} from Docker Hub...")
    code = _stream(["docker", "pull", SANDBOX_IMAGE], timeout=600)
    if code == 0:
        if _image_exists_and_valid():
            _ok("Image pulled and validated.")
            return True
        _fail("Image pulled but missing required runtime label — needs rebuild.")
        return False
    _fail("Pull failed.")
    return False


# Build the sandbox image from the local Dockerfile.
def _build() -> bool:
    dockerfile_path = _SCRIPT_DIR / "Dockerfile"
    if not dockerfile_path.exists():
        _fail(f"Dockerfile not found at {dockerfile_path}")
        return False
    _info(f"Building {SANDBOX_IMAGE} from local Dockerfile (this may take 10–20 min)...")
    code = _stream(
        ["docker", "build", "-t", SANDBOX_IMAGE, str(_SCRIPT_DIR)],
        timeout=1800,
    )
    if code == 0:
        _ok("Image built successfully.")
        return True
    _fail("Local build failed. Check the output above for details.")
    return False


# Entry point: pull or build the sandbox image according to --source.
# Standalone setup, run before first use or to refresh the image; it needs only Docker (independent of ASLM
# and MCP). Usage: python setup-sandbox.py [--source local|registry|auto] [--force]. Return 0 when the
# image is ready and 1 on failure (details on stderr).
def main() -> int:
    parser = argparse.ArgumentParser(description="Set up the mcp-sandbox Docker image.")
    parser.add_argument(
        "--source",
        choices=["local", "registry", "auto"],
        default=None,
        help="Image source: local=build first, registry=pull first, auto=pull first (default: from sandbox.env or 'registry')",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-pull / rebuild even if a valid image already exists locally.",
    )
    args = parser.parse_args()

    # Resolve the image source: command line, then SANDBOX_IMAGE_SOURCE, falling back to the registry.
    source = args.source or _cfg("SANDBOX_IMAGE_SOURCE", "registry")
    if source not in {"local", "registry", "auto"}:
        source = "registry"

    print("\nmcp-sandbox image setup", flush=True)
    print(f"  image  : {SANDBOX_IMAGE}", flush=True)
    print(f"  source : {source}", flush=True)
    print(flush=True)

    # 1. Docker availability
    print("Checking Docker...", flush=True)
    if not _check_docker():
        return 1
    if not _check_daemon():
        return 1
    print(flush=True)

    # 2. Already have a valid image?
    if not args.force and _image_exists_and_valid():
        _ok(f"Image '{SANDBOX_IMAGE}' already present and valid. Nothing to do.")
        print(flush=True)
        return 0

    if args.force:
        _info("--force: skipping existing image check.")

    # 3. Pull / build according to source
    print("Fetching image...", flush=True)
    if source == "local":
        success = _build() or _pull()
    elif source == "registry":
        success = _pull() or _build()
    else:
        # Source "auto" pulls first, like "registry".
        success = _pull() or _build()

    print(flush=True)
    if success:
        print("Setup complete. You can now start the sandbox.", flush=True)
        return 0
    else:
        print("Setup failed. See errors above.", file=sys.stderr, flush=True)
        return 1


# Run the setup and use its result as the process exit code.
if __name__ == "__main__":
    raise SystemExit(main())
