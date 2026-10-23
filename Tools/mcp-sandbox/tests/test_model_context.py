# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from sandbox import config, model_context
from sandbox import docker_host


ROOT = Path(__file__).resolve().parents[1]


def test_context_is_allowlisted_and_distinguishes_unverified_config(monkeypatch):
    monkeypatch.setenv("SANDBOX_API_KEY", "must-not-leak")
    monkeypatch.setattr(model_context, "_matching_container", lambda: None)
    prompt = model_context.get_model_context()
    assert "Configured limits for newly created containers" in prompt
    assert "Live limits were not verified" in prompt
    assert config.MODEL_WORKSPACE_CONTAINER in prompt
    assert "processes AND threads" in prompt
    assert "must-not-leak" not in prompt
    assert config.HOST_WORKSPACE not in prompt
    assert config.CONFIG_FILE_PATH not in prompt


def test_context_uses_verified_running_limits_without_exporting_other_env(monkeypatch):
    runtime = {
        "HostConfig": {"NanoCpus": 4_000_000_000, "Memory": 3221225472,
                       "MemorySwap": 4294967296, "PidsLimit": 256},
        "Config": {"Env": ["API_KEY=secret", "SANDBOX_THREAD_LIMIT=4", "SANDBOX_DEFAULT_TIMEOUT=90"]},
    }
    monkeypatch.setattr(model_context, "_matching_container", lambda: runtime)
    prompt = model_context.get_model_context()
    line = next(line for line in prompt.splitlines() if line.startswith("Verified running"))
    actual = json.loads(line.split(": ", 1)[1])
    assert actual["cpu_quota_cores"] == 4
    assert actual["processes_and_threads_total"] == 256
    assert actual["library_threads"] == 4
    assert actual["default_timeout_s"] == 90
    assert actual["read_bytes"] == "unknown"
    assert "secret" not in prompt
    assert "Use the running limits" in prompt


@pytest.mark.parametrize("status", ["wrong_workspace", "stopped", "matching"])
def test_runtime_inspection_requires_this_workspace_and_never_starts_docker(monkeypatch, status):
    expected = os.path.join(config.HOST_WORKSPACE, config.DEFAULT_TASK_DIR)
    info = {"State": {"Running": status != "stopped"}, "Mounts": [{
        "Source": expected if status != "wrong_workspace" else "/unrelated/workspace",
        "Destination": config.MODEL_WORKSPACE_CONTAINER,
    }]}
    calls = []

    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps([info]))

    monkeypatch.setattr(model_context.subprocess, "run", run)
    assert (model_context._matching_container() is not None) == (status == "matching")
    assert calls[0][0] == ["docker", "inspect", config.CONTAINER_NAME]
    assert calls[0][1]["timeout"] == 2


def test_docker_unavailable_does_not_block_context(monkeypatch):
    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker", 2)
    monkeypatch.setattr(model_context.subprocess, "run", unavailable)
    assert "Live limits were not verified" in model_context.get_model_context()


@pytest.mark.parametrize("enabled", [True, False])
def test_internet_setting_is_reported_and_applied_without_connectivity_probes(monkeypatch, tmp_path, enabled):
    monkeypatch.setattr(config, "NETWORK_ENABLED", enabled)
    monkeypatch.setattr(model_context, "_matching_container", lambda: None)
    monkeypatch.setattr(docker_host, "NETWORK_ENABLED", enabled)
    monkeypatch.setattr(docker_host, "NETWORK_LIMIT_MBIT", 100)
    monkeypatch.setattr(docker_host, "DEV_BIND", False)
    monkeypatch.setattr(docker_host, "SUPERVISOR_VENV_HOST", "")
    def no_probe(*args, **kwargs):
        raise AssertionError("Network probes must not run")
    monkeypatch.setattr(model_context.subprocess, "run", no_probe)
    prompt = model_context.get_model_context()
    state = "enabled" if enabled else "disabled"
    assert f"Internet access (configuration): {state}." in prompt
    settings = json.loads(next(line.split(": ", 1)[1] for line in prompt.splitlines()
                               if line.startswith("Configured limits")))
    assert settings["internet_enabled"] is enabled
    command = docker_host._build_run_command("image:test", task_host_path=str(tmp_path))
    assert f"SANDBOX_NETWORK_ENABLED={int(enabled)}" in command
    if enabled:
        assert "--network" not in command
    else:
        assert command[command.index("--network") + 1] == "none"
        assert "--cap-add" not in command


def test_new_resources_are_passed_to_docker_and_library_threads(tmp_path, monkeypatch):
    monkeypatch.setattr(docker_host, "CPU_LIMIT", "8")
    monkeypatch.setattr(docker_host, "PIDS_LIMIT", "1024")
    monkeypatch.setattr(docker_host, "MEMORY_LIMIT", "6g")
    monkeypatch.setattr(docker_host, "MEMORY_SWAP_LIMIT", "8g")
    monkeypatch.setattr(docker_host, "THREAD_LIMIT", 8)
    monkeypatch.setattr(docker_host, "DEV_BIND", False)
    monkeypatch.setattr(docker_host, "SUPERVISOR_VENV_HOST", "")
    command = docker_host._build_run_command("image:test", task_host_path=str(tmp_path))
    for flag, value in (("--cpus", "8"), ("--pids-limit", "1024"),
                        ("--memory", "6g"), ("--memory-swap", "8g")):
        assert command[command.index(flag) + 1] == value
    for variable in ("SANDBOX_THREAD_LIMIT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        assert f"{variable}=8" in command


def test_worker_reads_json_config_fresh_without_exporting_secrets(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SANDBOX_")}
    cfg = tmp_path / "sandbox.env"
    json_cfg = cfg.with_suffix(".json")
    env.update(SANDBOX_CONFIG_FILE=str(cfg), SANDBOX_CONTAINER_NAME="missing-context-test-container")
    worker = ROOT.parents[1] / "Services" / "tool_worker.py"
    server = ROOT / "mcp-server.py"
    for cores in (5, 6):
        json_cfg.write_text(json.dumps({"SANDBOX_CPU_LIMIT": cores, "SANDBOX_PIDS_LIMIT": 777,
                                        "SANDBOX_THREAD_LIMIT": 3, "SANDBOX_NETWORK_ENABLED": False,
                                        "SECRET_TOKEN": "hidden"}), encoding="utf-8")
        result = subprocess.run([sys.executable, str(worker), "model_context", str(server)],
                                input="{}", capture_output=True, text=True, env=env, timeout=15, check=True)
        response = json.loads(result.stdout)
        assert response["ok"]
        prompt = response["result"]
        assert f'"cpu_quota_cores": "{cores}"' in prompt
        assert '"processes_and_threads_total": "777"' in prompt
        assert '"library_threads": 3' in prompt
        assert '"internet_enabled": false' in prompt
        assert "Internet access (configuration): disabled." in prompt
        assert "hidden" not in prompt
    described = subprocess.run([sys.executable, str(worker), "describe", str(server)],
                               input="{}", capture_output=True, text=True, env=env, timeout=15, check=True)
    assert json.loads(described.stdout)["result"]["has_model_context"] is True


def test_default_resources_and_module_settings_match(tmp_path):
    env = {key: value for key, value in os.environ.items() if not key.startswith("SANDBOX_")}
    env["SANDBOX_CONFIG_FILE"] = str(tmp_path / "absent.env")
    script = (
        "from sandbox import config; import json; "
        "print(json.dumps([config.CPU_LIMIT,config.THREAD_LIMIT,config.PIDS_LIMIT,"
        "config.MEMORY_LIMIT,config.MEMORY_SWAP_LIMIT]))"
    )
    env["PYTHONPATH"] = str(ROOT / "supervisor")
    result = subprocess.run([sys.executable, "-c", script], env=env, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == ["8", 8, "1024", "6g", "8g"]
    from Settings import settings
    expected = {"sandbox-cpu-limit": 8, "sandbox-thread-limit": 8, "sandbox-pids-limit": 1024,
                "sandbox-memory-limit": 6, "sandbox-memory-swap-limit": 8, "sandbox-network-enabled": True}
    manifest = json.loads((ROOT.parents[1] / "ASLM_Module.json").read_text(encoding="utf-8"))
    manifest_settings = {entry["key"]: entry for entry in manifest["settings"]}
    for key, value in expected.items():
        assert settings.DEFAULTS[key] == value
        assert manifest_settings[key]["default"] == value
