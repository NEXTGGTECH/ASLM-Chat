# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import pytest


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_DESTRUCTIVE_SANDBOX_TESTS", "").strip().lower() not in {"1", "true", "yes"},
    reason="destructive disposable-container test; set RUN_DESTRUCTIVE_SANDBOX_TESTS=1 to run",
)


# Sandbox image used for the disposable container; overridable through SANDBOX_IMAGE.
IMAGE = os.getenv("SANDBOX_IMAGE", "nggtlightkeeper/aslm-chat-sandbox:latest")


# One throwaway sandbox container with its own temporary task directory, removed again by cleanup().
class DisposableSandbox:
    # Choose a unique container name and create a temporary host directory with an empty _sandbox task root.
    def __init__(self) -> None:
        self.name = f"mcp-sandbox-destructive-{uuid.uuid4().hex[:10]}"
        self.tmp = Path(tempfile.mkdtemp(prefix="mcp-sandbox-destructive-"))
        self.task_root = self.tmp / "_sandbox"
        self.task_root.mkdir(parents=True, exist_ok=True)

    # Run a host command and return the completed process; with check=True a non-zero exit code raises an
    # AssertionError that includes the command, stdout and stderr.
    def run(self, args: list[str], *, timeout: int = 60, check: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode != 0:
            raise AssertionError(
                f"command failed ({result.returncode}): {' '.join(args)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return result

    # Remove any container with the same name, then start a detached one with the temporary task root
    # mounted at /workspace/_sandbox.
    def start(self) -> None:
        self.run(["docker", "rm", "-f", self.name], timeout=30, check=False)
        self.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                self.name,
                "-e",
                "SANDBOX_DEFAULT_TASK_DIR=_sandbox",
                "-v",
                f"{self.task_root}:/workspace/_sandbox",
                IMAGE,
            ],
            timeout=120,
        )

    # Run a bash login script inside the container through docker exec, optionally as another user and
    # from another working directory (default: the task root).
    def exec(
        self,
        script: str,
        *,
        user: str | None = None,
        workdir: str = "/workspace/_sandbox",
        timeout: int = 60,
        check: bool = True,
    ) -> subprocess.CompletedProcess[str]:
        args = ["docker", "exec", "-w", workdir]
        if user:
            args.extend(["-u", user])
        args.extend([self.name, "bash", "-lc", script])
        return self.run(args, timeout=timeout, check=check)

    # Remove the container and the temporary host directory.
    def cleanup(self) -> None:
        self.run(["docker", "rm", "-f", self.name], timeout=30, check=False)
        shutil.rmtree(self.tmp, ignore_errors=True)


# Run the supervisor healthcheck inside the container, using its bundled virtualenv and sources.
def _healthcheck(container: DisposableSandbox, *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return container.exec(
        "PYTHONPATH=/opt/sandbox-src /opt/sandbox-venv/bin/python -m sandbox.supervisor --healthcheck",
        timeout=30,
        check=check,
    )


# Verify that a disposable sandbox container survives escalating actions (ordinary work, package install,
# system file changes, killing the supervisor and wiping the workspace) and that the supervisor healthcheck
# still passes after the risky stages.
def test_disposable_container_survives_escalating_destructive_actions() -> None:
    sandbox = DisposableSandbox()
    try:
        sandbox.start()

        # Stage 1: ordinary model-like work as sandbox_user.
        ordinary = sandbox.exec(
            "id -un; pwd; echo ordinary-ok > ordinary.txt; test -f ordinary.txt; cat ordinary.txt",
            user="sandbox_user",
            timeout=30,
        )
        assert "sandbox_user" in ordinary.stdout
        assert "ordinary-ok" in ordinary.stdout
        assert "sandbox-supervisor-pong-v2" in _healthcheck(sandbox).stdout

        # Stage 2: package manager use. This intentionally exercises sudo and networked apt.
        package_install = sandbox.exec(
            "sudo apt-get update >/tmp/destructive-apt.log 2>&1 && "
            "sudo apt-get install -y --no-install-recommends jq >>/tmp/destructive-apt.log 2>&1 && "
            "jq --version",
            user="sandbox_user",
            timeout=300,
        )
        assert "jq-" in package_install.stdout

        # Stage 3: read system logs/metadata from inside the container.
        logs = sandbox.exec(
            "sudo find /var/log -maxdepth 2 -type f -printf '%p\\n' | sort | head -20",
            user="sandbox_user",
            timeout=30,
        )
        assert "/var/log" in logs.stdout

        # Stage 4: modify container system state.
        modified = sandbox.exec(
            "echo changed | sudo tee /etc/mcp-sandbox-destructive-marker >/dev/null && "
            "sudo test -f /etc/mcp-sandbox-destructive-marker && echo system-modified",
            user="sandbox_user",
            timeout=30,
        )
        assert "system-modified" in modified.stdout
        assert "sandbox-supervisor-pong-v2" in _healthcheck(sandbox).stdout

        # Stage 5: start and kill a real long-lived supervisor process, then verify a fresh
        # supervisor healthcheck can still run in the damaged container.
        supervisor = sandbox.exec(
            "PYTHONPATH=/opt/sandbox-src nohup /opt/sandbox-venv/bin/python -m sandbox.supervisor "
            ">/tmp/destructive-supervisor.out 2>/tmp/destructive-supervisor.err & "
            "echo $!",
            timeout=30,
        )
        supervisor_pid = supervisor.stdout.strip().splitlines()[-1]
        assert supervisor_pid.isdigit()
        sandbox.exec(
            "sudo pkill -TERM -f 'python.*sandbox.supervisor|mcp-supervisor' || true; "
            "sleep 1; "
            "sudo pkill -KILL -f 'python.*sandbox.supervisor|mcp-supervisor' || true",
            user="sandbox_user",
            timeout=30,
            check=False,
        )
        assert "sandbox-supervisor-pong-v2" in _healthcheck(sandbox).stdout

        # Stage 6: try the literal dangerous command first. GNU rm refuses to
        # delete "." directly, then we remove all workspace contents and confirm
        # the container can still accept fresh workspace commands afterward.
        literal_rm = sandbox.exec(
            "cd /workspace/_sandbox && sudo rm -rf ./",
            user="sandbox_user",
            timeout=30,
            check=False,
        )
        assert literal_rm.returncode != 0
        assert "refusing to remove" in literal_rm.stderr
        workspace_destroy = sandbox.exec(
            "cd /workspace/_sandbox && sudo rm -rf -- ./* ./.??* && mkdir -p /workspace/_sandbox && "
            "cd /workspace/_sandbox && echo after-rm > after.txt && cat after.txt",
            user="sandbox_user",
            timeout=30,
        )
        assert "after-rm" in workspace_destroy.stdout
        assert "sandbox-supervisor-pong-v2" in _healthcheck(sandbox).stdout
    finally:
        sandbox.cleanup()
