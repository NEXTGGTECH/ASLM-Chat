# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

from Services import docker_status


pytestmark = [pytest.mark.unit]


@pytest.fixture
def isolated_state(tmp_path, monkeypatch):
    state_file = tmp_path / "docker_status.json"
    monkeypatch.setattr(docker_status, "STATE_FILE", state_file)
    return state_file


# Probe.


# probe() reports availability from the docker client binary exit code.
def test_probe_true_on_zero_exit(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout="Docker version 27.0.0")

    monkeypatch.setattr(docker_status.subprocess, "run", fake_run)

    assert docker_status.probe() is True


# A failing docker binary reports as unavailable.
def test_probe_false_on_nonzero_exit(monkeypatch):
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(docker_status.subprocess, "run", fake_run)

    assert docker_status.probe() is False


# A missing docker binary reports as unavailable.
def test_probe_false_when_docker_binary_is_missing(monkeypatch):
    def fake_run(*args, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(docker_status.subprocess, "run", fake_run)

    assert docker_status.probe() is False


# A hanging docker probe times out instead of blocking the caller.
def test_probe_false_on_subprocess_timeout(monkeypatch):
    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=6)

    monkeypatch.setattr(docker_status.subprocess, "run", fake_run)

    assert docker_status.probe() is False


# Status persistence.


# A missing state file reports Docker as unavailable.
def test_missing_state_file_reports_unavailable(isolated_state):
    status = docker_status.get_status()

    assert status == {"available": False, "checked_at": None}
    assert docker_status.is_available() is False


# A corrupt state file degrades to "unavailable" instead of raising.
def test_corrupt_state_file_reports_unavailable(isolated_state):
    isolated_state.parent.mkdir(parents=True, exist_ok=True)
    isolated_state.write_text("{broken", encoding="utf-8")

    assert docker_status.get_status()["available"] is False


# refresh() persists each fresh probe result to disk.
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
