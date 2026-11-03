# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from Services import docker_status


pytestmark = [pytest.mark.unit]


# Redirect the Docker status state file into a temporary directory and return its path.
@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    state_file = tmp_path / "docker_status.json"
    monkeypatch.setattr(docker_status, "STATE_FILE", state_file)
    return state_file


# Probe

# Verify that probe() reports Docker as available when the docker client binary exits with code 0.
def test_probe_true_on_zero_exit(monkeypatch):
    # Pretend the docker binary ran and exited with code 0.
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="Docker version 27.0.0")

    monkeypatch.setattr(docker_status.subprocess, "run", fake_run)

    assert docker_status.probe() is True


# Verify that probe() reports Docker as unavailable when the docker binary exits with a non-zero code.
def test_probe_false_on_nonzero_exit(monkeypatch):
    # Pretend the docker binary ran and exited with code 1.
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(docker_status.subprocess, "run", fake_run)

    assert docker_status.probe() is False


# Verify that probe() reports Docker as unavailable when the docker binary is not installed.
def test_probe_false_when_docker_binary_is_missing(monkeypatch):
    # Pretend the docker executable cannot be found.
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(docker_status.subprocess, "run", fake_run)

    assert docker_status.probe() is False


# Verify that probe() reports Docker as unavailable when the docker command times out instead of
# blocking the caller.
def test_probe_false_on_subprocess_timeout(monkeypatch):
    # Pretend the docker command hung until its timeout expired.
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=6)

    monkeypatch.setattr(docker_status.subprocess, "run", fake_run)

    assert docker_status.probe() is False


# Status persistence

# Verify that a missing state file reports Docker as unavailable with no check time.
def test_missing_state_file_reports_unavailable(isolated_state):
    status = docker_status.get_status()

    assert status == {"available": False, "checked_at": None}
    assert docker_status.is_available() is False


# Verify that a corrupt state file reports Docker as unavailable instead of raising.
def test_corrupt_state_file_reports_unavailable(isolated_state):
    isolated_state.parent.mkdir(parents=True, exist_ok=True)
    isolated_state.write_text("{broken", encoding="utf-8")

    assert docker_status.get_status()["available"] is False


# Verify that refresh() writes each fresh probe result, with a check time, to the state file and that
# a later negative probe flips the stored availability.
def test_refresh_persists_probe_result(isolated_state, monkeypatch):
    monkeypatch.setattr(docker_status, "probe", lambda: True)

    status = docker_status.refresh()

    assert status["available"] is True
    assert status["checked_at"] is not None
    persisted = json.loads(isolated_state.read_text(encoding="utf-8"))
    assert persisted["available"] is True

    monkeypatch.setattr(docker_status, "probe", lambda: False)
    assert docker_status.refresh()["available"] is False
    assert docker_status.is_available() is False
