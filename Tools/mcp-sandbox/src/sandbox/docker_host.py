# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Empty, Queue
from typing import Callable

from sandbox.config import (
    BACKGROUND_TIMEOUT_THRESHOLD,
    COMMAND_USER,
    CONFIG_FILE_PATH,
    CONTAINER_NAME,
    CONTAINER_WORKSPACE,
    CPU_LIMIT,
    DEFAULT_TASK_DIR,
    DEFAULT_TIMEOUT,
    DEV_BIND,
    DOCKER_START_TIMEOUT_SECONDS,
    HOST_WORKSPACE,
    MAX_CAT_FILE_BYTES,
    MAX_CAT_LINE_THRESHOLD,
    MAX_FIND_RESULTS,
    MAX_FILE_MAP_SYMBOLS,
    MAX_GREP_RESULTS,
    MAX_IMAGE_PREVIEW_BYTES,
    MAX_LS_ENTRIES,
    MAX_OUTPUT_BYTES,
    OUTPUT_HEAD_RATIO,
    MAX_READ_BYTES,
    MEMORY_LIMIT,
    MEMORY_SWAP_LIMIT,
    MODEL_WORKSPACE_CONTAINER,
    NETWORK_ENABLED,
    NETWORK_LIMIT_MBIT,
    PIDS_LIMIT,
    SANDBOX_IMAGE,
    SNAPSHOT_IMAGE_PREFIX,
    STORAGE_LIMIT,
    SUPERVISOR_SRC,
    SUPERVISOR_SRC_HOST,
    SUPERVISOR_VENV,
    SUPERVISOR_VENV_HOST,
    THREAD_LIMIT,
    WORKSPACE_CLEANUP_ENABLED,
    WORKSPACE_CLEANUP_IDLE_SECONDS,
    WORKSPACE_CLEANUP_INTERVAL_SECONDS,
    WORKSPACE_CLEANUP_RECYCLE_SECONDS,
    LINUX_DOCKER_DESKTOP_PATHS,
    MACOS_DOCKER_DESKTOP_PATHS,
    WINDOWS_DOCKER_DESKTOP_PATHS,
)
from sandbox.config_bridge import sync_sandbox_env
from sandbox.exec import (
    BoundedOutputCollector,
    _background_error_result,
    _new_background_job_id,
    _read_stream_chunks,
    _truncate,
)
from sandbox.jobs import BackgroundJob, JOB_REGISTRY
from sandbox.workspace import (
    get_secure_task_path,
    normalize_model_relative_path,
    task_root,
)

logger = logging.getLogger(__name__)

REQUIRED_IMAGE_LABEL = "org.aslm.sandbox.supervisor-runtime"
REQUIRED_IMAGE_LABEL_VALUE = "container-v2"
SUPERVISOR_PONG = "sandbox-supervisor-pong-v2"
LEGACY_UPLOAD_MODEL_ROOT = "/mnt/data/User"
UPLOAD_MODEL_ROOT = f"{MODEL_WORKSPACE_CONTAINER.rstrip('/')}/User"
DEFAULT_DOCKER_JOB_ROOT = "/workspace/.sandbox_jobs"


# Keep old model-facing upload paths usable in plain bash commands.
def _rewrite_legacy_upload_paths(command: str) -> str:
    return str(command or "").replace(LEGACY_UPLOAD_MODEL_ROOT, UPLOAD_MODEL_ROOT)


# Pick a container-writable root for background spool files.
def _resolve_docker_job_root() -> str:
    configured = (
        os.getenv("SANDBOX_CONTAINER_JOB_ROOT", "").strip()
        or os.getenv("SANDBOX_JOB_ROOT", "").strip()
    )
    if configured.startswith("/"):
        return configured.rstrip("/") or "/"
    return DEFAULT_DOCKER_JOB_ROOT


# Return ordered job-root candidates for docker background jobs.
def _docker_job_root_candidates() -> list[str]:
    configured = _resolve_docker_job_root()
    candidates = [
        configured,
        "/workspace/.sandbox_jobs",
        "/tmp/mcp-sandbox-jobs",
        "/dev/shm/mcp-sandbox-jobs",
        "$HOME/.sandbox_jobs",
    ]
    unique: list[str] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


# Subprocess helpers

# Run a subprocess command with UTF-8 text handling.
def _run_command(
    args: list[str],
    timeout: int = 30,
    check: bool = False,
    input_text: str | None = None,
):
    return subprocess.run(
        args,
        input=input_text,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=check,
    )


# Docker availability

# Return True when the docker CLI is installed.
def _docker_cli_available() -> bool:
    try:
        result = _run_command(["docker", "--version"], timeout=5)
        return result.returncode == 0
    except Exception:
        return False


# Query docker daemon info.
def _docker_info(timeout: int = 5):
    return _run_command(["docker", "info", "--format", "{{json .}}"], timeout=timeout)


# Return whether host tools may launch Docker Desktop automatically.
def _auto_start_docker_enabled() -> bool:
    value = os.environ.get("SANDBOX_AUTO_START_DOCKER", "1")
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Attempt to launch Docker for the current platform.
# Returns (launched, message); launched=True means a start was triggered and the
# caller should poll the daemon, False means nothing could be started here.
def _launch_docker_desktop() -> tuple[bool, str]:
    if sys.platform == "win32":
        for path in WINDOWS_DOCKER_DESKTOP_PATHS:
            if not os.path.isfile(path):
                continue
            subprocess.Popen(
                [path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=0x00000008,  # DETACHED_PROCESS
            )
            return True, "Docker Desktop is starting."
        return False, "Docker Desktop not found. Install or start it manually."

    if sys.platform == "darwin":
        app_path = next((p for p in MACOS_DOCKER_DESKTOP_PATHS if os.path.isdir(p)), None)
        command = ["open", "-a", "Docker"] if app_path is None else ["open", app_path]
        try:
            subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True, "Docker Desktop is starting."
        except Exception as exc:
            return False, f"Could not launch Docker Desktop: {exc}"

    if sys.platform.startswith("linux"):
        # Docker Desktop for Linux runs as a per-user systemd service.
        try:
            result = _run_command(["systemctl", "--user", "start", "docker-desktop"], timeout=15)
            if result.returncode == 0:
                return True, "Docker Desktop is starting."
        except Exception:
            pass
        for path in LINUX_DOCKER_DESKTOP_PATHS:
            if not os.path.isfile(path):
                continue
            try:
                subprocess.Popen(
                    [path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,
                )
                return True, "Docker Desktop is starting."
            except Exception:
                continue
        # The system docker daemon cannot be started without privileges from here.
        return (
            False,
            "Docker daemon is not running. Start it manually "
            "(e.g. 'sudo systemctl start docker') and try again.",
        )

    return False, "Docker daemon is not running. Start Docker and try again."


# Ensure the Docker daemon is available.
def _ensure_docker_running() -> tuple[bool, str]:
    if not _docker_cli_available():
        return False, "Docker CLI not found. Install Docker Desktop first."

    try:
        info_result = _docker_info(timeout=5)
        if info_result.returncode == 0:
            return True, "Docker daemon is running."
    except Exception:
        pass

    if not _auto_start_docker_enabled():
        return (
            False,
            "Docker daemon is not running. Start Docker to use sandbox tools, "
            "or set SANDBOX_AUTO_START_DOCKER=1 to allow automatic launch.",
        )

    launched, message = _launch_docker_desktop()
    if not launched:
        return False, message

    deadline = time.time() + DOCKER_START_TIMEOUT_SECONDS
    while time.time() < deadline:
        time.sleep(2)
        try:
            check_result = _docker_info(timeout=5)
            if check_result.returncode == 0:
                return True, "Docker started successfully."
        except Exception:
            continue

    return (
        False,
        f"Docker was launched but the daemon did not respond in {DOCKER_START_TIMEOUT_SECONDS}s.",
    )


# Container existence checks

def _container_exists() -> bool:
    result = _run_command(
        ["docker", "ps", "-aq", "-f", f"name=^{CONTAINER_NAME}$"], timeout=10
    )
    return bool(result.stdout.strip())


# Image preparation

def _image_has_required_runtime(inspect_stdout: str) -> bool:
    try:
        image_info = json.loads(inspect_stdout)
    except json.JSONDecodeError:
        return False
    if not image_info:
        return False
    labels = image_info[0].get("Config", {}).get("Labels") or {}
    return labels.get(REQUIRED_IMAGE_LABEL) == REQUIRED_IMAGE_LABEL_VALUE


# Ensure the sandbox image exists locally by delegating to setup-sandbox.py.
def _ensure_image(force_rebuild: bool = False) -> tuple[bool, str]:
    inspect_result = _run_command(
        ["docker", "image", "inspect", SANDBOX_IMAGE], timeout=20
    )
    if inspect_result.returncode == 0 and not force_rebuild:
        if _image_has_required_runtime(inspect_result.stdout):
            return True, "Image already exists locally."

    setup_script = Path(__file__).resolve().parents[2] / "setup-sandbox.py"
    cmd = [sys.executable, str(setup_script)]
    if force_rebuild:
        cmd.append("--force")
    result = subprocess.run(cmd, timeout=1900)
    if result.returncode == 0:
        return True, f"Image '{SANDBOX_IMAGE}' is ready."
    return False, (
        f"Image setup failed (exit {result.returncode}). "
        "Run setup-sandbox.py manually for details."
    )


# Container run command

def _linux_venv_bind_source() -> str | None:
    if not SUPERVISOR_VENV_HOST:
        return None
    venv_path = Path(SUPERVISOR_VENV_HOST).expanduser()
    bin_python = venv_path / "bin" / "python"
    windows_python = venv_path / "Scripts" / "python.exe"
    pyvenv_cfg = venv_path / "pyvenv.cfg"
    if windows_python.exists() or not bin_python.exists() or not pyvenv_cfg.exists():
        raise ValueError(
            "SANDBOX_SUPERVISOR_VENV_HOST must point to a Linux venv "
            "with bin/python and pyvenv.cfg, not a Windows venv."
        )
    return str(venv_path)


# Regenerate sandbox.env from sandbox.json before each container lifecycle check.
def _ensure_sandbox_config(config_path: str = CONFIG_FILE_PATH) -> None:
    try:
        sync_sandbox_env(json_path=Path(config_path).with_suffix(".json"), env_path=Path(config_path))
    except OSError as exc:
        logger.warning("Could not synchronize sandbox config %s: %s", config_path, exc)


# Build the docker run command for the sandbox container.
def _build_run_command(
    image_name: str,
    include_storage_limit: bool = True,
    *,
    container_name: str = CONTAINER_NAME,
    task_host_path: str | None = None,
    restart_policy: str | None = "unless-stopped",
    auto_remove: bool = False,
) -> list[str]:
    command = [
        "docker", "run", "-d",
        "--name", container_name,
        "--cpus", CPU_LIMIT,
        "--memory", MEMORY_LIMIT,
        "--memory-swap", MEMORY_SWAP_LIMIT,
        "--pids-limit", PIDS_LIMIT,
        "--no-healthcheck",
    ]
    if restart_policy:
        command.extend(["--restart", restart_policy])
    if auto_remove:
        command.append("--rm")
    # Do not add --security-opt no-new-privileges here unless the sudo model
    # is redesigned. The sandbox_user intentionally has passwordless sudo for
    # package-manager compatibility, and no-new-privileges would break sudo.

    if not NETWORK_ENABLED:
        command.extend(["--network", "none"])
    elif NETWORK_LIMIT_MBIT > 0:
        command.extend(["--cap-add", "NET_ADMIN"])

    if include_storage_limit and STORAGE_LIMIT:
        command.extend(["--storage-opt", f"size={STORAGE_LIMIT}"])

    task_host_path = task_host_path or os.path.join(HOST_WORKSPACE, DEFAULT_TASK_DIR)
    os.makedirs(task_host_path, exist_ok=True)
    supervisor_src_host = SUPERVISOR_SRC_HOST
    supervisor_venv_host = _linux_venv_bind_source()

    # Internal container constants — always set explicitly.
    command.extend([
        "-v", f"{task_host_path}:{MODEL_WORKSPACE_CONTAINER}",
        "-w", MODEL_WORKSPACE_CONTAINER,
        # Runtime identity — container-side constants.
        "-e", "SANDBOX_IN_CONTAINER=1",
        "-e", f"SANDBOX_COMMAND_USER={COMMAND_USER}",
        "-e", f"SANDBOX_HOST_WORKSPACE={CONTAINER_WORKSPACE}",
        "-e", f"SANDBOX_DEFAULT_TASK_DIR={DEFAULT_TASK_DIR}",
        "-e", f"SANDBOX_SUPERVISOR_SRC={SUPERVISOR_SRC}",
        "-e", f"SANDBOX_SUPERVISOR_VENV={SUPERVISOR_VENV}",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        # Execution limits — user-configurable via sandbox.env, passed explicitly
        # so that host-only vars (image name, container name, Windows paths, etc.)
        # never leak into the container environment.
        "-e", f"SANDBOX_DEFAULT_TIMEOUT={DEFAULT_TIMEOUT}",
        "-e", f"SANDBOX_MAX_OUTPUT_BYTES={MAX_OUTPUT_BYTES}",
        "-e", f"SANDBOX_OUTPUT_HEAD_RATIO={OUTPUT_HEAD_RATIO}",
        "-e", f"SANDBOX_MAX_READ_BYTES={MAX_READ_BYTES}",
        "-e", f"SANDBOX_MAX_CAT_FILE_BYTES={MAX_CAT_FILE_BYTES}",
        "-e", f"SANDBOX_MAX_CAT_LINE_THRESHOLD={MAX_CAT_LINE_THRESHOLD}",
        "-e", f"SANDBOX_MAX_IMAGE_PREVIEW_BYTES={MAX_IMAGE_PREVIEW_BYTES}",
        "-e", f"SANDBOX_MAX_LS_ENTRIES={MAX_LS_ENTRIES}",
        "-e", f"SANDBOX_MAX_FIND_RESULTS={MAX_FIND_RESULTS}",
        "-e", f"SANDBOX_MAX_GREP_RESULTS={MAX_GREP_RESULTS}",
        "-e", f"SANDBOX_MAX_FILE_MAP_SYMBOLS={MAX_FILE_MAP_SYMBOLS}",
        "-e", f"SANDBOX_BACKGROUND_TIMEOUT_THRESHOLD={BACKGROUND_TIMEOUT_THRESHOLD}",
        "-e", f"SANDBOX_WORKSPACE_CLEANUP_ENABLED={int(WORKSPACE_CLEANUP_ENABLED)}",
        "-e", f"SANDBOX_WORKSPACE_CLEANUP_IDLE_SECONDS={WORKSPACE_CLEANUP_IDLE_SECONDS}",
        "-e", f"SANDBOX_WORKSPACE_CLEANUP_RECYCLE_SECONDS={WORKSPACE_CLEANUP_RECYCLE_SECONDS}",
        "-e", f"SANDBOX_WORKSPACE_CLEANUP_INTERVAL_SECONDS={WORKSPACE_CLEANUP_INTERVAL_SECONDS}",
        "-e", f"SANDBOX_NETWORK_LIMIT_MBIT={NETWORK_LIMIT_MBIT}",
        "-e", f"SANDBOX_NETWORK_ENABLED={int(NETWORK_ENABLED)}",
        # Thread limits — propagated to ML libraries too.
        "-e", f"SANDBOX_THREAD_LIMIT={THREAD_LIMIT}",
        "-e", f"OMP_NUM_THREADS={THREAD_LIMIT}",
        "-e", f"OPENBLAS_NUM_THREADS={THREAD_LIMIT}",
        "-e", f"MKL_NUM_THREADS={THREAD_LIMIT}",
        "-e", f"NUMEXPR_NUM_THREADS={THREAD_LIMIT}",
        "-e", f"VECLIB_MAXIMUM_THREADS={THREAD_LIMIT}",
    ])

    if DEV_BIND and supervisor_src_host and os.path.isdir(supervisor_src_host):
        command.extend(["-v", f"{supervisor_src_host}:{SUPERVISOR_SRC}:ro"])

    if DEV_BIND and supervisor_venv_host is not None:
        command.extend(["-v", f"{supervisor_venv_host}:{SUPERVISOR_VENV}:ro"])

    command.append(image_name)
    return command


# Container lifecycle

_container_lock = threading.Lock()


# Ensure the sandbox container is running (idempotent with retry).
def _ensure_container_running(
    image_name: str = SANDBOX_IMAGE,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> tuple[bool, str]:
    _ensure_sandbox_config()
    docker_ok, msg = _ensure_docker_running()
    if not docker_ok:
        return False, msg
    with _container_lock:
        return _ensure_container_locked(image_name, max_retries, retry_delay)


def _ensure_container_locked(
    image_name: str,
    max_retries: int,
    retry_delay: float,
) -> tuple[bool, str]:
    last_error = ""
    for attempt in range(max_retries):
        if attempt > 0:
            time.sleep(retry_delay)
            logger.info("Retry %d/%d for container setup", attempt + 1, max_retries)

        state = _inspect_container()

        if state is None:
            ok, msg = _create_container(image_name)
            if not ok:
                if "already in use" in msg.lower():
                    logger.debug("Name conflict on create, will retry")
                last_error = msg
                continue
            running, run_msg = _wait_for_container_running()
            if running:
                return True, msg
            _force_remove()
            last_error = run_msg
            continue

        if state["running"] and state["volume_ok"]:
            return True, "Container is running."

        if not state["volume_ok"]:
            ok, msg = _force_remove()
            if not ok:
                last_error = f"Failed to remove mismatched container: {msg}"
                continue
            ok, msg = _create_container(image_name)
            if not ok:
                last_error = msg
                continue
            running, run_msg = _wait_for_container_running()
            if running:
                return True, msg
            _force_remove()
            last_error = run_msg
            continue

        if state["exists"] and not state["running"]:
            ok, msg = _start_existing()
            if ok:
                return True, msg
            logger.debug("Start failed (%s), removing and recreating", msg)
            _force_remove()
            ok, msg = _create_container(image_name)
            if not ok:
                last_error = msg
                continue
            running, run_msg = _wait_for_container_running()
            if running:
                return True, msg
            _force_remove()
            last_error = run_msg
            continue

    return False, f"Failed after {max_retries} attempts. Last error: {last_error}"


def _inspect_container() -> dict | None:
    result = _run_command(["docker", "inspect", CONTAINER_NAME], timeout=10)
    if result.returncode != 0:
        return None
    try:
        data = json.loads(result.stdout)
        if not data:
            return None
        info = data[0]
        state_dict = info.get("State", {})
        running = state_dict.get("Running", False)
        status = state_dict.get("Status", "unknown")
        mounts = " ".join(m.get("Source", "") for m in info.get("Mounts", []))
        expected_mount = os.path.normcase(
            os.path.normpath(os.path.join(HOST_WORKSPACE, DEFAULT_TASK_DIR))
        )
        actual_mount = os.path.normcase(os.path.normpath(mounts))
        return {
            "exists": True,
            "running": running,
            "status": status,
            "volume_ok": expected_mount in actual_mount.split(),
        }
    except Exception:
        return None


def _force_remove() -> tuple[bool, str]:
    result = _run_command(["docker", "rm", "-f", CONTAINER_NAME], timeout=30)
    if result.returncode == 0:
        return True, "Removed."
    if "no such container" in result.stderr.lower():
        return True, "Already gone."
    return False, result.stderr.strip()


def _start_existing() -> tuple[bool, str]:
    result = _run_command(["docker", "start", CONTAINER_NAME], timeout=30)
    if result.returncode == 0:
        time.sleep(0.5)
        state = _inspect_container()
        if state and state["running"]:
            return True, "Started."
        return False, "Started but not running."
    return False, result.stderr.strip()


def _storage_limit_unsupported(stderr_text: str) -> bool:
    lowered = stderr_text.lower()
    return any(
        indicator in lowered
        for indicator in ["storage-opt", "unknown flag", "quota", "overlay2", "xfs",
                          "not supported", "invalid argument"]
    )


def _create_container(image_name: str) -> tuple[bool, str]:
    build_ok, build_msg = _ensure_image()
    if not build_ok:
        return False, build_msg

    if STORAGE_LIMIT:
        try:
            cmd = _build_run_command(image_name, include_storage_limit=True)
        except ValueError as exc:
            return False, str(exc)
        result = _run_command(cmd, timeout=60)
        if result.returncode == 0:
            return True, f"Container '{CONTAINER_NAME}' created."
        if _storage_limit_unsupported(result.stderr):
            try:
                cmd = _build_run_command(image_name, include_storage_limit=False)
            except ValueError as exc:
                return False, str(exc)
            result = _run_command(cmd, timeout=60)
            if result.returncode == 0:
                return True, f"Container '{CONTAINER_NAME}' created (no storage quota)."
        return False, result.stderr.strip()

    try:
        cmd = _build_run_command(image_name, include_storage_limit=False)
    except ValueError as exc:
        return False, str(exc)
    result = _run_command(cmd, timeout=60)
    if result.returncode == 0:
        return True, f"Container '{CONTAINER_NAME}' created."
    return False, result.stderr.strip()


# Poll until the container reaches 'running' state or the timeout expires.
#
# Returns False if the container enters a restart loop (crashes immediately)
# so the caller can report a meaningful error rather than getting a confusing
# 'Container is restarting' error from docker exec.
def _wait_for_container_running(timeout_s: float = 10.0) -> tuple[bool, str]:
    deadline = time.time() + timeout_s
    consecutive_restarting = 0
    while time.time() < deadline:
        state = _inspect_container()
        if state is None:
            return False, "Container disappeared after creation."
        status = state.get("status", "").lower()
        if state.get("running"):
            return True, "running"
        if status == "restarting":
            consecutive_restarting += 1
            if consecutive_restarting >= 3:
                # Seen restarting 3 times in a row → crash-loop
                result = _run_command(
                    ["docker", "logs", "--tail", "20", CONTAINER_NAME], timeout=10
                )
                tail = (result.stdout or result.stderr or "").strip()
                hint = f"\nLast container logs:\n{tail}" if tail else ""
                return False, f"Container is crash-looping (status=restarting).{hint}"
        else:
            consecutive_restarting = 0
        time.sleep(0.5)
    return False, f"Container did not reach 'running' within {timeout_s}s."


# Restart the sandbox container.
def restart_container() -> tuple[bool, str]:
    _ensure_sandbox_config()
    docker_ok, docker_message = _ensure_docker_running()
    if not docker_ok:
        return False, docker_message
    if not _container_exists():
        return _ensure_container_running()

    result = _run_command(["docker", "restart", CONTAINER_NAME], timeout=60)
    if result.returncode == 0:
        return True, f"Container '{CONTAINER_NAME}' restarted."
    return False, result.stderr.strip() or "Failed to restart container."


def snapshot_image_name(name: str) -> str:
    safe_name = "".join(
        char if char.isalnum() or char in ("-", "_", ".") else "-" for char in name
    ).strip("-")
    safe_name = safe_name or "stable"
    return f"{SNAPSHOT_IMAGE_PREFIX}:{safe_name}"


def _snapshot_check_result(
    name: str,
    ok: bool,
    *,
    message: str = "",
    details: dict | None = None,
) -> dict:
    return {
        "name": name,
        "ok": ok,
        "message": message,
        "details": details or {},
    }


# Run safety/stability checks before mutating Docker snapshot state.
def _run_snapshot_preflight() -> dict:
    checks: list[dict] = []

    def add(name: str, ok: bool, message: str = "", details: dict | None = None) -> None:
        checks.append(_snapshot_check_result(name, ok, message=message, details=details))

    health_ok, health_message = healthcheck_container_supervisor(timeout_s=5)
    add("supervisor_healthcheck", health_ok, health_message)

    if not health_ok:
        return {"ok": False, "checks": checks}

    workspace_jobs = _docker_exec_shell(
        f"test ! -e {shlex.quote(MODEL_WORKSPACE_CONTAINER)}/.sandbox_jobs",
        timeout=10,
    )
    add(
        "job_state_outside_workspace",
        workspace_jobs.returncode == 0,
        (
            "No model-writable .sandbox_jobs under workspace."
            if workspace_jobs.returncode == 0
            else "Workspace contains .sandbox_jobs; refusing to snapshot runtime state."
        ),
    )

    job_root_probe = _docker_exec_shell(
        (
            f"{shlex.quote(SUPERVISOR_VENV)}/bin/python - <<'PY'\n"
            "import os, sys\n"
            "os.environ.setdefault('SANDBOX_IN_CONTAINER', '1')\n"
            f"os.environ.setdefault('SANDBOX_SUPERVISOR_SRC', {SUPERVISOR_SRC!r})\n"
            f"sys.path.insert(0, {SUPERVISOR_SRC!r})\n"
            "from sandbox.exec import job_root\n"
            "root = job_root()\n"
            "print(root)\n"
            "raise SystemExit(0 if str(root).startswith('/tmp/') and not root.is_symlink() else 1)\n"
            "PY"
        ),
        timeout=15,
    )
    add(
        "job_root_private",
        job_root_probe.returncode == 0,
        job_root_probe.stdout.strip() or job_root_probe.stderr.strip(),
    )

    cap_probe = _exec_bash_docker(
        "python3 - <<'PY'\nprint('A' * 120000)\nPY",
        timeout_s=15,
        background="never",
    )
    cap_stdout = str(cap_probe.get("stdout") or "")
    add(
        "stdout_cap_head_tail",
        bool(cap_probe.get("truncated")) and "[output truncated:" in cap_stdout,
        "Large stdout is bounded and marked.",
        {
            "exit_code": cap_probe.get("exit_code"),
            "stdout_bytes": len(cap_stdout.encode("utf-8", errors="replace")),
            "truncated": cap_probe.get("truncated"),
        },
    )

    invalid_utf8_probe = _exec_bash_docker(
        "python3 - <<'PY'\nimport sys\nsys.stdout.buffer.write(bytes([0xff, 0xfe, 0xfd]) + b' ok\\n')\nPY",
        timeout_s=15,
        background="never",
    )
    add(
        "invalid_utf8_replace",
        invalid_utf8_probe.get("exit_code") == 0 and "ok" in str(invalid_utf8_probe.get("stdout") or ""),
        "Invalid UTF-8 does not crash output readers.",
        {"exit_code": invalid_utf8_probe.get("exit_code")},
    )

    bg_probe = _exec_bash_docker(
        "sleep 2; printf background-ok",
        timeout_s=1,
        background="always",
    )
    job_id = bg_probe.get("job_id")
    bg_ok = bool(job_id) and bg_probe.get("error_type") == "backgrounded"
    fg_ok = False
    fg_details: dict = {"job_id": job_id}
    if job_id:
        time.sleep(2.3)
        try:
            fg_result = foreground_background_job(str(job_id))
            fg_stdout = str(fg_result.get("new_stdout") or "")
            fg_ok = fg_result.get("status") == "done" and "background-ok" in fg_stdout
            fg_details.update(
                {
                    "status": fg_result.get("status"),
                    "exit_code": fg_result.get("exit_code"),
                    "stdout": fg_stdout,
                }
            )
            JOB_REGISTRY.remove(str(job_id), cleanup=False)
        except Exception as exc:
            fg_details["error"] = str(exc)
    add(
        "background_job_lifecycle",
        bg_ok and fg_ok,
        "Background job can be tracked and foregrounded.",
        fg_details,
    )

    ok = all(check["ok"] for check in checks)
    return {"ok": ok, "checks": checks}


def snapshot_container(name: str = "stable", *, preflight: bool = True) -> dict:
    container_ok, message = _ensure_container_running()
    if not container_ok:
        return {"ok": False, "error": message}
    preflight_result: dict | None = None
    if preflight:
        preflight_result = _run_snapshot_preflight()
        if not preflight_result.get("ok"):
            failed = [
                check["name"]
                for check in preflight_result.get("checks", [])
                if not check.get("ok")
            ]
            return {
                "ok": False,
                "error": (
                    "Snapshot preflight failed; docker commit was not executed. "
                    f"Failed checks: {', '.join(failed) or 'unknown'}."
                ),
                "preflight": preflight_result,
            }
    image_name = snapshot_image_name(name)
    result = _run_command(["docker", "commit", CONTAINER_NAME, image_name], timeout=600)
    if result.returncode != 0:
        return {
            "ok": False,
            "error": result.stderr.strip() or "Failed to snapshot container.",
            "preflight": preflight_result,
        }
    return {
        "ok": True,
        "snapshot_image": image_name,
        "name": name,
        "preflight": preflight_result,
    }


# Supervisor stdio proxy

def _supervisor_exec_command(
    *,
    interactive: bool = True,
    supervisor_args: list[str] | None = None,
) -> list[str]:
    command = ["docker", "exec"]
    if interactive:
        command.append("-i")
    command.extend([
        "-u", "root",
        "-w", MODEL_WORKSPACE_CONTAINER,
        "-e", "SANDBOX_IN_CONTAINER=1",
        "-e", f"SANDBOX_COMMAND_USER={COMMAND_USER}",
        "-e", f"SANDBOX_HOST_WORKSPACE={CONTAINER_WORKSPACE}",
        "-e", f"SANDBOX_DEFAULT_TASK_DIR={DEFAULT_TASK_DIR}",
        "-e", f"SANDBOX_SUPERVISOR_SRC={SUPERVISOR_SRC}",
        "-e", f"SANDBOX_SUPERVISOR_VENV={SUPERVISOR_VENV}",
        "-e", f"PYTHONPATH={SUPERVISOR_SRC}",
        "-e", "PYTHONDONTWRITEBYTECODE=1",
        CONTAINER_NAME,
        f"{SUPERVISOR_VENV}/bin/python",
        "-m", "sandbox.supervisor",
    ])
    command.extend(supervisor_args or [])
    return command


def start_container_supervisor() -> subprocess.Popen:
    return subprocess.Popen(
        _supervisor_exec_command(interactive=True),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def healthcheck_container_supervisor(timeout_s: int = 3) -> tuple[bool, str]:
    try:
        result = _run_command(
            _supervisor_exec_command(
                interactive=False, supervisor_args=["--healthcheck"]
            ),
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return False, f"Supervisor healthcheck timed out after {timeout_s}s."
    except Exception as exc:
        return False, str(exc)

    stdout = result.stdout.strip()
    if result.returncode == 0 and stdout == SUPERVISOR_PONG:
        return True, stdout
    if result.returncode == 0 and stdout == "sandbox-supervisor-pong":
        return False, "Supervisor version mismatch: image needs rebuild (got v1 pong, expected v2)."
    message = result.stderr.strip() or stdout or f"Exit code: {result.returncode}"
    return False, message


def _supervisor_python_missing(message: str) -> bool:
    lowered = message.lower()
    expected_python = f"{SUPERVISOR_VENV}/bin/python".lower()
    return expected_python in lowered and "no such file or directory" in lowered


def ensure_supervisor_ready(
    *,
    attempts: int = 2,
    timeout_s: int = 3,
    recreate_on_failure: bool = True,
) -> tuple[bool, str]:
    last_error = ""
    for attempt in range(max(1, attempts)):
        ok, message = healthcheck_container_supervisor(timeout_s=timeout_s)
        if ok:
            return True, message
        last_error = message
        if attempt + 1 < attempts:
            time.sleep(0.2)

    if not recreate_on_failure:
        return False, last_error

    with _container_lock:
        removed, remove_message = _force_remove()
        if not removed:
            return False, f"Supervisor failed ({last_error}); recreate failed: {remove_message}"
        created, create_message = _create_container(SANDBOX_IMAGE)
        if not created:
            return False, f"Supervisor failed ({last_error}); recreate failed: {create_message}"

    running, run_msg = _wait_for_container_running(timeout_s=15.0)
    if not running:
        return False, f"Supervisor failed ({last_error}); container did not start: {run_msg}"

    for attempt in range(max(1, attempts)):
        ok, message = healthcheck_container_supervisor(timeout_s=timeout_s)
        if ok:
            return True, f"Supervisor recovered after recreate: {message}"
        last_error = message
        if attempt + 1 < attempts:
            time.sleep(0.2)

    if _supervisor_python_missing(last_error):
        with _container_lock:
            removed, remove_message = _force_remove()
            if not removed:
                return False, (
                    f"Supervisor venv is missing ({last_error}); "
                    f"container remove failed: {remove_message}"
                )
            rebuilt, rebuild_message = _ensure_image(force_rebuild=True)
            if not rebuilt:
                return False, (
                    f"Supervisor venv is missing ({last_error}); "
                    f"image rebuild failed: {rebuild_message}"
                )
            created, create_message = _create_container(SANDBOX_IMAGE)
            if not created:
                return False, (
                    f"Supervisor venv is missing ({last_error}); "
                    f"container recreate after rebuild failed: {create_message}"
                )

        running, run_msg = _wait_for_container_running(timeout_s=15.0)
        if not running:
            return False, (
                f"Supervisor venv was missing ({last_error}); "
                f"container did not start after rebuild: {run_msg}"
            )

        for attempt in range(max(1, attempts)):
            ok, message = healthcheck_container_supervisor(timeout_s=timeout_s)
            if ok:
                return True, f"Supervisor recovered after image rebuild: {message}"
            last_error = message
            if attempt + 1 < attempts:
                time.sleep(0.2)

    return False, f"Supervisor healthcheck failed after recreate: {last_error}"


def _read_pipe_chunk(source) -> bytes:
    if hasattr(source, "read1"):
        return source.read1(8192)
    if hasattr(source, "readline"):
        return source.readline()
    return source.read(8192)


def _forward_binary_stream(source, sink, close_sink: bool = False) -> None:
    try:
        while True:
            chunk = _read_pipe_chunk(source)
            if not chunk:
                break
            sink.write(chunk)
            sink.flush()
    except (BrokenPipeError, OSError):
        pass
    finally:
        if close_sink:
            try:
                sink.close()
            except OSError:
                pass


def _read_stdin_to_queue(
    input_stream, input_queue: Queue, eof: threading.Event
) -> None:
    try:
        while True:
            chunk = _read_pipe_chunk(input_stream)
            if not chunk:
                break
            input_queue.put(chunk)
    finally:
        eof.set()
        input_queue.put(None)


def _write_queue_to_process(
    input_queue: Queue,
    process: subprocess.Popen,
    eof: threading.Event,
) -> None:
    if process.stdin is None:
        return
    try:
        while process.poll() is None:
            try:
                chunk = input_queue.get(timeout=0.1)
            except Empty:
                continue
            if chunk is None:
                try:
                    process.stdin.close()
                except OSError:
                    pass
                return
            try:
                process.stdin.write(chunk)
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                return
    finally:
        if eof.is_set():
            try:
                process.stdin.close()
            except OSError:
                pass


# Pipe host stdio to the in-container MCP supervisor with reconnect.
def pipe_to_container_supervisor(
    *,
    max_restarts: int | None = None,
    input_stream=None,
    output_stream=None,
    error_stream=None,
) -> int:
    input_stream = input_stream or sys.stdin.buffer
    output_stream = output_stream or sys.stdout.buffer
    error_stream = error_stream or sys.stderr.buffer
    input_queue: Queue | None = None
    stdin_eof: threading.Event | None = None
    stdin_thread = None

    def _ensure_stdin_reader() -> tuple[Queue, threading.Event]:
        nonlocal input_queue, stdin_eof, stdin_thread
        if input_queue is None or stdin_eof is None:
            input_queue = Queue()
            stdin_eof = threading.Event()
            stdin_thread = threading.Thread(
                target=_read_stdin_to_queue,
                args=(input_stream, input_queue, stdin_eof),
                daemon=True,
            )
            stdin_thread.start()
        return input_queue, stdin_eof

    _ensure_sandbox_config()

    restarts = 0
    last_returncode = 1
    while max_restarts is None or restarts <= max_restarts:
        container_ok, message = _ensure_container_running()
        if not container_ok:
            error_stream.write((message + "\n").encode("utf-8", errors="replace"))
            error_stream.flush()
            return 1

        supervisor_ok, supervisor_message = ensure_supervisor_ready()
        if not supervisor_ok:
            error_stream.write(
                (supervisor_message + "\n").encode("utf-8", errors="replace")
            )
            error_stream.flush()
            return 1

        input_queue, stdin_eof = _ensure_stdin_reader()
        process = start_container_supervisor()
        stdout_thread = threading.Thread(
            target=_forward_binary_stream,
            args=(process.stdout, output_stream),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_forward_binary_stream,
            args=(process.stderr, error_stream),
            daemon=True,
        )
        writer_thread = threading.Thread(
            target=_write_queue_to_process,
            args=(input_queue, process, stdin_eof),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        writer_thread.start()

        last_returncode = process.wait()
        for pipe in (process.stdout, process.stderr):
            try:
                if pipe is not None:
                    pipe.close()
            except OSError:
                pass
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)
        writer_thread.join(timeout=5)

        if stdin_eof.is_set():
            return last_returncode

        restarts += 1
        error_stream.write(
            f"mcp-sandbox supervisor exited with {last_returncode}; restarting.\n"
            .encode("utf-8", errors="replace")
        )
        error_stream.flush()
        time.sleep(0.2)

    return last_returncode


# ── Docker-exec bash backend (host-side fallback) ────────────────────
# Used when IN_CONTAINER=False. Will be removed after full migration.

def _docker_exec_shell(
    script: str,
    *,
    container_cwd: str | None = None,
    timeout: int = 30,
    user: str | None = None,
):
    args = ["docker", "exec"]
    if user:
        args.extend(["-u", user])
    if container_cwd:
        args.extend(["-w", container_cwd])
    args.extend([CONTAINER_NAME, "bash", "-lc", script])
    return _run_command(args, timeout=timeout)


def _read_docker_job_file(
    job: BackgroundJob, stream: str, *, incremental: bool = True
) -> str:
    if job.container_job_dir is None:
        return ""
    path = f"{job.container_job_dir.rstrip('/')}/{stream}"
    offset_attr = f"{stream}_offset"
    offset = int(getattr(job, offset_attr)) if incremental else 0
    head_ratio = min(0.9, max(0.1, OUTPUT_HEAD_RATIO))
    head_bytes = max(1, int(MAX_OUTPUT_BYTES * head_ratio))
    tail_bytes = max(1, MAX_OUTPUT_BYTES - head_bytes)
    reader = (
        "python3 - "
        f"{shlex.quote(path)} {offset} {MAX_OUTPUT_BYTES} {head_bytes} {tail_bytes} <<'PY'\n"
        "from __future__ import annotations\n"
        "import os, sys\n"
        "path = sys.argv[1]\n"
        "start = max(0, int(sys.argv[2]))\n"
        "max_bytes = max(1, int(sys.argv[3]))\n"
        "head_bytes = max(1, int(sys.argv[4]))\n"
        "tail_bytes = max(1, int(sys.argv[5]))\n"
        "try:\n"
        "    size = os.path.getsize(path)\n"
        "except OSError:\n"
        "    raise SystemExit(0)\n"
        "start = min(start, size)\n"
        "remaining = size - start\n"
        "with open(path, 'rb') as fh:\n"
        "    if remaining <= max_bytes:\n"
        "        fh.seek(start)\n"
        "        data = fh.read(remaining)\n"
        "    else:\n"
        "        fh.seek(start)\n"
        "        head = fh.read(head_bytes)\n"
        "        fh.seek(max(start, size - tail_bytes))\n"
        "        tail = fh.read(tail_bytes)\n"
        "        marker = (\n"
        "            b'\\n\\n[output truncated while reading docker job spool: '\n"
        "            + f'showed first {head_bytes} bytes and last {tail_bytes} bytes '\n"
        "              f'of {remaining} new bytes'.encode('ascii')\n"
        "            + b']\\n\\n'\n"
        "        )\n"
        "        data = head + marker + tail\n"
        "sys.stdout.write(data.decode('utf-8', errors='replace').replace('\\r\\n', '\\n'))\n"
        "PY"
    )
    result = _docker_exec_shell(
        reader,
        timeout=10,
    )
    content = result.stdout if result.returncode == 0 else ""
    if not incremental:
        return content
    size_result = _docker_exec_shell(
        f"wc -c < {shlex.quote(path)} 2>/dev/null || true",
        timeout=10,
    )
    try:
        new_offset = int((size_result.stdout or "").strip())
    except ValueError:
        new_offset = offset
    setattr(job, offset_attr, max(offset, new_offset))
    return content


def _refresh_docker_job(job: BackgroundJob) -> BackgroundJob:
    if job.runtime != "docker" or job.container_job_dir is None:
        return job
    job_dir = shlex.quote(job.container_job_dir)
    result = _docker_exec_shell(
        (
            f"status=$(cat {job_dir}/status 2>/dev/null || echo running); "
            f"exit_code=$(cat {job_dir}/exit_code 2>/dev/null || true); "
            f"pid=$(cat {job_dir}/pid 2>/dev/null || true); "
            f"pgid=$(cat {job_dir}/pgid 2>/dev/null || true); "
            "alive=unknown; "
            "case \"$pgid\" in ''|*[!0-9]*) pgid='' ;; esac; "
            "case \"$pid\" in ''|*[!0-9]*) pid='' ;; esac; "
            "if [ \"$status\" = running ]; then "
            "  if [ -n \"$pgid\" ]; then "
            "    if kill -0 -- -\"$pgid\" >/dev/null 2>&1; then alive=yes; else alive=no; fi; "
            "  elif [ -n \"$pid\" ]; then "
            "    if kill -0 \"$pid\" >/dev/null 2>&1; then alive=yes; else alive=no; fi; "
            "  fi; "
            "fi; "
            "printf '%s\\n%s\\n%s\\n%s\\n%s\\n' \"$status\" \"$exit_code\" \"$pid\" \"$pgid\" \"$alive\""
        ),
        timeout=10,
    )
    if result.returncode != 0:
        return job
    lines = result.stdout.splitlines()
    status = lines[0].strip() if lines else job.status
    exit_code = job.exit_code
    if len(lines) > 1 and lines[1].strip():
        try:
            exit_code = int(lines[1].strip())
        except ValueError:
            exit_code = job.exit_code
    pid = job.pid
    if len(lines) > 2 and lines[2].strip():
        try:
            pid = int(lines[2].strip())
        except ValueError:
            pid = job.pid
    if pid is not None:
        job.pid = pid
    alive = lines[4].strip() if len(lines) > 4 else "unknown"
    if status == "done":
        JOB_REGISTRY.mark_done(job.job_id, exit_code)
    elif status == "killed":
        JOB_REGISTRY.mark_killed(job.job_id)
    elif status == "running" and alive == "no":
        JOB_REGISTRY.mark_done(job.job_id, exit_code)
    elif status:
        job.status = status
    return job


def list_background_jobs() -> dict:
    for job in JOB_REGISTRY.list_jobs():
        if job.get("runtime") == "docker" and job.get("status") == "running":
            try:
                _refresh_docker_job(JOB_REGISTRY.get(job["job_id"]))
            except Exception:
                pass
    return {"jobs": JOB_REGISTRY.list_jobs()}


def foreground_background_job(job_id: str) -> dict:
    job = _refresh_docker_job(JOB_REGISTRY.get(job_id))
    stdout = _read_docker_job_file(job, "stdout", incremental=True)
    stderr = _read_docker_job_file(job, "stderr", incremental=True)
    spool_trunc_out = "[output truncated while reading docker job spool:" in stdout
    spool_trunc_err = "[output truncated while reading docker job spool:" in stderr
    stdout, trunc_out = _truncate(stdout)
    stderr, trunc_err = _truncate(stderr)
    return {
        **job.to_result(),
        "new_stdout": stdout,
        "new_stderr": stderr,
        "truncated": spool_trunc_out or spool_trunc_err or trunc_out or trunc_err,
    }


def kill_background_job(job_id: str) -> dict:
    job = JOB_REGISTRY.get(job_id)
    if job.runtime == "docker" and job.container_job_dir:
        job_dir = shlex.quote(job.container_job_dir)
        _docker_exec_shell(
            f"pid=$(cat {job_dir}/pid 2>/dev/null || true); "
            f"pgid=$(cat {job_dir}/pgid 2>/dev/null || true); "
            "case \"$pgid\" in ''|*[!0-9]*) pgid='' ;; esac; "
            "case \"$pid\" in ''|*[!0-9]*) pid='' ;; esac; "
            "if [ -n \"$pgid\" ]; then "
            "  kill -TERM -- -\"$pgid\" >/dev/null 2>&1 || true; "
            "  sleep 0.5; "
            "  kill -KILL -- -\"$pgid\" >/dev/null 2>&1 || true; "
            "elif [ -n \"$pid\" ]; then "
            "  kill -TERM \"$pid\" >/dev/null 2>&1 || true; "
            "  sleep 0.5; "
            "  kill -KILL \"$pid\" >/dev/null 2>&1 || true; "
            "fi; "
            f"echo killed > {job_dir}/status",
            timeout=10,
        )
    JOB_REGISTRY.mark_killed(job.job_id)
    return job.to_result()


def _exec_bash_docker_background(
    command: str,
    cwd: str,
    timeout_s: int,
    container_cwd: str,
    on_progress: Callable[[float, str], None] | None = None,
) -> dict:
    start_time = time.time()
    job_id = _new_background_job_id()
    job_dir: str | None = None
    candidate_roots = " ".join(shlex.quote(item) for item in _docker_job_root_candidates())
    command_quoted = shlex.quote(command)
    wrapper = (
        f"bash -lc {command_quoted}; "
        "code=$?; "
        "echo $code > \"$job_dir/exit_code\"; "
        "echo done > \"$job_dir/status\""
    )
    setup_script = (
        f"job_root=''; "
        f"for candidate in {candidate_roots}; do "
        f"  home_dollar='$HOME/'; home_braced='${{HOME}}/'; home_tilde='~/'; "
        f"  case \"$candidate\" in "
        f"    \"$home_dollar\"*) candidate=\"$HOME/${{candidate#$home_dollar}}\" ;; "
        f"    \"$home_braced\"*) candidate=\"$HOME/${{candidate#$home_braced}}\" ;; "
        f"    \"$home_tilde\"*) candidate=\"$HOME/${{candidate#$home_tilde}}\" ;; "
        f"  esac; "
        f"  if mkdir -p \"$candidate\" >/dev/null 2>&1 && [ -w \"$candidate\" ]; then "
        f"    job_root=\"$candidate\"; break; "
        f"  fi; "
        f"done; "
        f"if [ -z \"$job_root\" ]; then "
        f"  echo 'Failed to find writable background job root.' >&2; exit 1; "
        f"fi; "
        f"job_dir=\"$job_root/{job_id}\"; "
        f"mkdir -p \"$job_dir\"; "
        f": > \"$job_dir/stdout\"; "
        f": > \"$job_dir/stderr\"; "
        f"echo running > \"$job_dir/status\"; "
        f"export job_dir; "
        f"if command -v setsid >/dev/null 2>&1; then "
        f"  setsid bash -lc {shlex.quote(wrapper)} "
        f"> \"$job_dir/stdout\" 2> \"$job_dir/stderr\" < /dev/null & "
        f"  child_pid=$!; echo \"$child_pid\" > \"$job_dir/pgid\"; "
        f"else "
        f"  nohup bash -lc {shlex.quote(wrapper)} "
        f"> \"$job_dir/stdout\" 2> \"$job_dir/stderr\" < /dev/null & "
        f"  child_pid=$!; : > \"$job_dir/pgid\"; "
        f"fi; "
        f"echo \"$job_dir\"; "
        f"echo \"$child_pid\" > \"$job_dir/pid\"; "
        f"cat \"$job_dir/pid\""
    )
    setup = _docker_exec_shell(
        setup_script,
        container_cwd=container_cwd,
        timeout=30,
        user=COMMAND_USER if COMMAND_USER and COMMAND_USER != "root" else None,
    )
    if setup.returncode != 0:
        return {
            "exit_code": None,
            "stdout": setup.stdout,
            "stderr": setup.stderr,
            "error": setup.stderr.strip() or "Failed to start background job.",
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "truncated": False,
            "cwd": normalize_model_relative_path(cwd),
        }

    setup_lines = [line.strip() for line in setup.stdout.splitlines() if line.strip()]
    if setup_lines:
        maybe_job_dir = setup_lines[0]
        if "/" in maybe_job_dir:
            job_dir = maybe_job_dir
    try:
        pid = int(setup_lines[-1])
    except (ValueError, IndexError):
        pid = None

    job = JOB_REGISTRY.create(
        command=command,
        cwd=normalize_model_relative_path(cwd),
        runtime="docker",
        pid=pid,
        container_job_dir=job_dir or f"{_resolve_docker_job_root()}/{job_id}",
        job_id=job_id,
    )

    while True:
        job = _refresh_docker_job(job)
        if job.status != "running":
            stdout = _read_docker_job_file(job, "stdout", incremental=False)
            stderr = _read_docker_job_file(job, "stderr", incremental=False)
            spool_trunc_out = "[output truncated while reading docker job spool:" in stdout
            spool_trunc_err = "[output truncated while reading docker job spool:" in stderr
            stdout, trunc_out = _truncate(stdout)
            stderr, trunc_err = _truncate(stderr)
            if on_progress is not None:
                on_progress(100.0, f"Bash finished in {normalize_model_relative_path(cwd)}")
            return {
                "exit_code": job.exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "error": None if job.exit_code == 0 else f"Exit code: {job.exit_code}",
                "elapsed_ms": int((time.time() - start_time) * 1000),
                "truncated": spool_trunc_out or spool_trunc_err or trunc_out or trunc_err,
                "cwd": normalize_model_relative_path(cwd),
            }

        elapsed_s = time.time() - start_time
        if elapsed_s >= timeout_s:
            stdout = _read_docker_job_file(job, "stdout", incremental=True)
            stderr = _read_docker_job_file(job, "stderr", incremental=True)
            spool_trunc_out = "[output truncated while reading docker job spool:" in stdout
            spool_trunc_err = "[output truncated while reading docker job spool:" in stderr
            stdout, trunc_out = _truncate(stdout)
            stderr, trunc_err = _truncate(stderr)
            return _background_error_result(
                job=job,
                stdout=stdout,
                stderr=stderr,
                start_time=start_time,
                timeout_s=timeout_s,
                cwd=cwd,
                truncated=spool_trunc_out or spool_trunc_err or trunc_out or trunc_err,
            )
        if on_progress is not None:
            progress = min(95.0, max(5.0, (elapsed_s / timeout_s) * 90.0))
            on_progress(progress, f"Running bash in {normalize_model_relative_path(cwd)}")
        time.sleep(0.2)


# Execute a bash command inside the sandbox container via docker exec.
def _exec_bash_docker(
    command: str,
    cwd: str = ".",
    timeout_s: int = DEFAULT_TIMEOUT,
    stdin: str | None = None,
    on_stdout: Callable[[str], None] | None = None,
    on_stderr: Callable[[str], None] | None = None,
    on_progress: Callable[[float, str], None] | None = None,
    background: str | bool | None = "never",
) -> dict:
    from sandbox.exec import should_use_background

    command = _rewrite_legacy_upload_paths(command)

    if timeout_s <= 0:
        raise ValueError("timeout_s must be greater than 0.")

    target_dir = get_secure_task_path(cwd, kind="cwd")
    if not target_dir.exists():
        raise FileNotFoundError(f"cwd not found: {normalize_model_relative_path(cwd)}")
    if not target_dir.is_dir():
        raise NotADirectoryError(
            f"cwd is not a directory: {normalize_model_relative_path(cwd)}"
        )

    container_ok, container_message = _ensure_container_running()
    if not container_ok:
        return {
            "ok": False,
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error": container_message,
            "elapsed_ms": 0,
            "truncated": False,
        }

    relative_cwd = (
        target_dir.relative_to(task_root()).as_posix()
        if target_dir != task_root()
        else ""
    )
    container_cwd = MODEL_WORKSPACE_CONTAINER
    if relative_cwd:
        container_cwd = f"{MODEL_WORKSPACE_CONTAINER}/{relative_cwd}"

    if stdin is None and should_use_background(command, timeout_s, background):
        return _exec_bash_docker_background(
            command=command,
            cwd=cwd,
            timeout_s=timeout_s,
            container_cwd=container_cwd,
            on_progress=on_progress,
        )

    start_time = time.time()
    exec_cmd = ["docker", "exec", "-i"]
    if COMMAND_USER and COMMAND_USER != "root":
        exec_cmd.extend(["-u", COMMAND_USER])
    exec_cmd.extend([
        "-w", container_cwd,
        "-e", "PYTHONIOENCODING=utf-8",
        "-e", "LANG=C.UTF-8",
        "-e", f"SANDBOX_DEFAULT_TASK_DIR={DEFAULT_TASK_DIR}",
        CONTAINER_NAME, "bash", "-lc", command,
    ])

    try:
        process = subprocess.Popen(
            exec_cmd,
            stdin=subprocess.PIPE if stdin is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except Exception as exc:
        return {
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "error": str(exc),
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "truncated": False,
            "cwd": normalize_model_relative_path(cwd),
        }

    stdout_chunks = BoundedOutputCollector()
    stderr_chunks = BoundedOutputCollector()
    stdout_thread = threading.Thread(
        target=_read_stream_chunks,
        args=(process.stdout, stdout_chunks, on_stdout),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_read_stream_chunks,
        args=(process.stderr, stderr_chunks, on_stderr),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    if stdin is not None and process.stdin is not None:
        try:
            process.stdin.write(stdin)
        except BrokenPipeError:
            pass
        finally:
            process.stdin.close()

    try:
        while process.poll() is None:
            elapsed_s = time.time() - start_time
            if elapsed_s >= timeout_s:
                process.kill()
                raise subprocess.TimeoutExpired(exec_cmd, timeout_s)
            if on_progress is not None:
                progress = min(95.0, max(5.0, (elapsed_s / timeout_s) * 90.0))
                on_progress(
                    progress, f"Running bash in {normalize_model_relative_path(cwd)}"
                )
            time.sleep(0.2)

        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout_thread.join(timeout=2)
        stderr_thread.join(timeout=2)
        stdout_value, trunc_out = stdout_chunks.value()
        stderr_value, trunc_err = stderr_chunks.value()
        restart_container()
        return {
            "exit_code": None,
            "stdout": stdout_value,
            "stderr": stderr_value,
            "error": (
                f"Execution timed out after {timeout_s} seconds. "
                "Container restarted to free resources."
            ),
            "elapsed_ms": int((time.time() - start_time) * 1000),
            "truncated": trunc_out or trunc_err,
            "cwd": normalize_model_relative_path(cwd),
        }

    stdout_value, trunc_out = stdout_chunks.value()
    stderr_value, trunc_err = stderr_chunks.value()

    if on_progress is not None:
        on_progress(100.0, f"Bash finished in {normalize_model_relative_path(cwd)}")

    return {
        "exit_code": process.returncode,
        "stdout": stdout_value,
        "stderr": stderr_value,
        "error": None if process.returncode == 0 else f"Exit code: {process.returncode}",
        "elapsed_ms": int((time.time() - start_time) * 1000),
        "truncated": trunc_out or trunc_err,
        "cwd": normalize_model_relative_path(cwd),
    }
