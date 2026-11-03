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


# Package root of mcp-sandbox.
ROOT = Path(__file__).resolve().parents[1]


# Verify that, with no matching running container, the model context contains only allowlisted values
# (no secret environment variable, host workspace or config file path) and states that the configured
# limits were not verified live.
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


# Verify that, with a matching running container, the "Verified running" line reports its real CPU, process,
# thread and timeout limits (other limits "unknown") and that unrelated environment variables are not exported.
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


# Verify that a container is only returned when it is running and mounts this workspace (not for a wrong
# workspace or a stopped container), and that the lookup issues a single docker inspect call with a 2
# second timeout.
@pytest.mark.parametrize("status", ["wrong_workspace", "stopped", "matching"])
def test_runtime_inspection_requires_this_workspace_and_never_starts_docker(monkeypatch, status):
    expected = os.path.join(config.HOST_WORKSPACE, config.DEFAULT_TASK_DIR)
    info = {"State": {"Running": status != "stopped"}, "Mounts": [{
        "Source": expected if status != "wrong_workspace" else "/unrelated/workspace",
        "Destination": config.MODEL_WORKSPACE_CONTAINER,
    }]}
    calls = []

    # Record the docker call and return the prepared inspect data.
    def run(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout=json.dumps([info]))

    monkeypatch.setattr(model_context.subprocess, "run", run)
    assert (model_context._matching_container() is not None) == (status == "matching")
    assert calls[0][0] == ["docker", "inspect", config.CONTAINER_NAME]
    assert calls[0][1]["timeout"] == 2


# Verify that a docker call timing out does not break the model context: it still reports that live limits
# were not verified.
def test_docker_unavailable_does_not_block_context(monkeypatch):
    # Simulate docker not answering within the timeout.
    def unavailable(*args, **kwargs):
        raise subprocess.TimeoutExpired("docker", 2)
    monkeypatch.setattr(model_context.subprocess, "run", unavailable)
    assert "Live limits were not verified" in model_context.get_model_context()


# Verify that the internet setting is reported in the model context as configuration only (no connectivity
# probe is run) and applied to the docker run command: enabled adds no --network option, disabled uses
# --network none without extra capabilities.
@pytest.mark.parametrize("enabled", [True, False])
def test_internet_setting_is_reported_and_applied_without_connectivity_probes(monkeypatch, tmp_path, enabled):
    # Arrange: apply the setting to the config and the docker host, with no live container and no dev bind.
    monkeypatch.setattr(config, "NETWORK_ENABLED", enabled)
    monkeypatch.setattr(model_context, "_matching_container", lambda: None)
    monkeypatch.setattr(docker_host, "NETWORK_ENABLED", enabled)
    monkeypatch.setattr(docker_host, "NETWORK_LIMIT_MBIT", 100)
    monkeypatch.setattr(docker_host, "DEV_BIND", False)
    monkeypatch.setattr(docker_host, "SUPERVISOR_VENV_HOST", "")

    # Fail the test if any subprocess (such as a network probe) is started.
    def no_probe(*args, **kwargs):
        raise AssertionError("Network probes must not run")
    monkeypatch.setattr(model_context.subprocess, "run", no_probe)

    # Assert on the model context: the internet line and the configured settings JSON.
    prompt = model_context.get_model_context()
    state = "enabled" if enabled else "disabled"
    assert f"Internet access (configuration): {state}." in prompt
    settings = json.loads(next(line.split(": ", 1)[1] for line in prompt.splitlines()
                               if line.startswith("Configured limits")))
    assert settings["internet_enabled"] is enabled

    # Assert on the docker run command built for the same setting.
    command = docker_host._build_run_command("image:test", task_host_path=str(tmp_path))
    assert f"SANDBOX_NETWORK_ENABLED={int(enabled)}" in command
    if enabled:
        assert "--network" not in command
    else:
        assert command[command.index("--network") + 1] == "none"
        assert "--cap-add" not in command


# Verify that the configured CPU, process, memory and swap limits become docker run options and that the
# thread limit is exported through the library thread environment variables.
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


# Verify that the tool worker process reads the JSON config on every call (two different CPU limits are
# both picked up) and reports the allowlisted values in the model context without exporting the secret
# entry, and that the module describes itself as having a model context.
def test_worker_reads_json_config_fresh_without_exporting_secrets(tmp_path):
    # Arrange: a clean environment without SANDBOX_ variables, pointing at a temporary config file.
    env = {key: value for key, value in os.environ.items() if not key.startswith("SANDBOX_")}
    cfg = tmp_path / "sandbox.env"
    json_cfg = cfg.with_suffix(".json")
    env.update(SANDBOX_CONFIG_FILE=str(cfg), SANDBOX_CONTAINER_NAME="missing-context-test-container")
    worker = ROOT.parents[1] / "Services" / "tool_worker.py"
    server = ROOT / "mcp-server.py"

    # Act and assert for two config versions: write the JSON, run the worker and check its context.
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

    # The worker's describe command must also report that the module provides a model context.
    described = subprocess.run([sys.executable, str(worker), "describe", str(server)],
                               input="{}", capture_output=True, text=True, env=env, timeout=15, check=True)
    assert json.loads(described.stdout)["result"]["has_model_context"] is True


# Verify that the sandbox config defaults (with no config file) equal the defaults of the module settings
# and the defaults declared in the ASLM_Module.json manifest.
def test_default_resources_and_module_settings_match(tmp_path, monkeypatch):
    # Read the defaults of the sandbox config in a clean subprocess with an absent config file.
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

    # Compare the expected defaults with the module settings and with the manifest settings.
    monkeypatch.syspath_prepend(str(ROOT.parents[1]))
    from Settings import settings
    expected = {"sandbox-cpu-limit": 8, "sandbox-thread-limit": 8, "sandbox-pids-limit": 1024,
                "sandbox-memory-limit": 6, "sandbox-memory-swap-limit": 8, "sandbox-network-enabled": True}
    manifest = json.loads((ROOT.parents[1] / "ASLM_Module.json").read_text(encoding="utf-8"))
    manifest_settings = {entry["key"]: entry for entry in manifest["settings"]}
    for key, value in expected.items():
        assert settings.DEFAULTS[key] == value
        assert manifest_settings[key]["default"] == value
