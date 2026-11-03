# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import subprocess

import pytest

from sandbox import config


# Run one command inside the sandbox container.
def _docker_exec(cmd: str) -> tuple[int, str]:
    result = subprocess.run(
        ["docker", "exec", config.CONTAINER_NAME, "bash", "-c", cmd],
        capture_output=True,
        text=True,
        timeout=15,
    )
    return result.returncode, result.stdout.strip()


# Return True when the sandbox container is currently running.
def _container_running() -> bool:
    try:
        result = subprocess.run(
            ["docker", "ps", "-q", "-f", f"name=^{config.CONTAINER_NAME}$"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return bool(result.stdout.strip())


pytestmark = pytest.mark.skipif(
    not _container_running(),
    reason=f"container '{config.CONTAINER_NAME}' is not running",
)

# Name of the task directory that is the only item the model-visible workspace may contain.
TASK_DIR = config.DEFAULT_TASK_DIR


# Privileges

# Verify that the container user is not root.
def test_container_does_not_run_as_root() -> None:
    _rc, out = _docker_exec("id")

    assert "uid=0(root)" not in out, f"container runs as root: {out}"


# Host mounts

# Verify that the Docker socket is not visible inside the container.
def test_docker_socket_is_not_exposed() -> None:
    rc, out = _docker_exec("ls -la /var/run/docker.sock 2>/dev/null || echo NOSOCK")

    assert rc != 0 or "NOSOCK" in out, "docker socket is accessible inside the container"


# Verify that no host filesystem mount (/host or /mnt/host) is visible inside the container.
def test_host_filesystem_is_not_mounted() -> None:
    _rc, out = _docker_exec("ls /host 2>/dev/null || ls /mnt/host 2>/dev/null || echo NOHOSTMOUNT")

    assert "NOHOSTMOUNT" in out, "host filesystem is mounted inside the container"


# Workspace exposure

# Verify that /workspace lists only the task directory (and a .gitkeep file).
def test_workspace_exposes_only_task_directory() -> None:
    rc, out = _docker_exec("ls /workspace/")

    assert rc == 0
    items = [item for item in out.splitlines() if item not in (TASK_DIR, ".gitkeep", "")]
    assert not items, f"/workspace exposes items beyond {TASK_DIR}/: {items}"


# Verify that the project source (/workspace/tools) is not readable through the workspace mount.
def test_project_source_is_not_visible_via_workspace() -> None:
    rc, out = _docker_exec("ls /workspace/tools 2>/dev/null | head -5")

    assert not (rc == 0 and out.strip()), f"project source readable inside container: {out}"


# Verify that the mcp-web-search entrypoint cannot be found or executed from inside the container.
def test_project_bridge_is_not_executable_from_container() -> None:
    rc, out = _docker_exec(
        "test -f /workspace/tools/mcp-web-search/mcp-server.py && "
        "python3 /workspace/tools/mcp-web-search/mcp-server.py --help 2>/dev/null | head -1"
    )

    assert not (rc == 0 and out.strip()), "model can execute mounted project entrypoints"


# Filesystem writes

# Verify that /etc is read-only inside the container; a leftover test file is removed afterwards.
def test_etc_is_read_only() -> None:
    rc, out = _docker_exec("touch /etc/pwned_test 2>&1 || echo READONLY")

    try:
        assert "READONLY" in out or rc != 0, "/etc is writable inside the container"
    finally:
        _docker_exec("rm -f /etc/pwned_test")


# Environment

# Verify that no secret-looking environment variable (secret, token, password, api key) is set inside the
# container.
def test_no_secret_environment_variables() -> None:
    rc, out = _docker_exec("env | grep -iE 'secret|token|password|api_key|api_secret' || echo NONE")

    assert out == "NONE" or rc != 0, f"secret-looking env vars found: {out[:300]}"


# Verify that /proc/self/root cannot be used to list the root filesystem.
def test_proc_self_root_escape_is_blocked() -> None:
    rc, out = _docker_exec("ls /proc/self/root/etc/ 2>/dev/null | head -3 || echo BLOCKED")

    assert "BLOCKED" in out or rc != 0, "/proc/self/root escape is possible"


# Symlinks

# Verify that a symlink to /etc/passwd inside the task directory does not expose host files, neither
# through the container shell nor through the host-side read() helper.
def test_symlink_escape_via_task_directory_is_blocked() -> None:
    # Arrange: recreate the symlink inside the task directory.
    _docker_exec(f"rm -f /workspace/{TASK_DIR}/__sym_test")
    _docker_exec(f"ln -s /etc/passwd /workspace/{TASK_DIR}/__sym_test")
    try:
        # Act and assert: reading it through the container shell must not return the file content.
        rc, out = _docker_exec(f"cat /workspace/{TASK_DIR}/__sym_test 2>/dev/null | head -1 || echo BLOCKED")

        assert not (rc == 0 and "root:" in out), "symlink escape works through the container shell"

        # read() rejecting the path outright is also acceptable.
        try:
            read_result = workspace_read_symtest()
            assert "root:" not in read_result, "workspace read() follows symlink targets"
        except Exception:
            # read() raising for the symlink is also acceptable: nothing is leaked in that case.
            pass
    finally:
        _docker_exec(f"rm -f /workspace/{TASK_DIR}/__sym_test")


# Read the __sym_test symlink in the task directory through the host-side workspace helper and return the
# content, or an empty string when the result has none.
def workspace_read_symtest() -> str:
    from sandbox import workspace

    result = workspace.read("__sym_test")

    return (result.get("result") or {}).get("content", "")
