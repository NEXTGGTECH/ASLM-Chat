# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json

from sandbox.config_bridge import sync_sandbox_env


# Verify that sandbox.env is generated from the JSON file with active assignments for known keys
# (booleans written as 0/1) and without unknown keys.
def test_sync_sandbox_env_generates_active_assignments(tmp_path) -> None:
    json_path = tmp_path / "sandbox.json"
    env_path = tmp_path / "sandbox.env"
    payload = {
        "SANDBOX_CPU_LIMIT": 6,
        "SANDBOX_NETWORK_ENABLED": False,
        "SANDBOX_WORKSPACE_CLEANUP_ENABLED": False,
        "IGNORED_VALUE": "ignored",
    }
    json_path.write_text(json.dumps(payload), encoding="utf-8")

    sync_sandbox_env(json_path, env_path)
    content = env_path.read_text(encoding="utf-8")

    assert "SANDBOX_CPU_LIMIT=6" in content
    assert "SANDBOX_NETWORK_ENABLED=0" in content
    assert "SANDBOX_WORKSPACE_CLEANUP_ENABLED=0" in content
    assert "IGNORED_VALUE" not in content


# Verify that an existing sandbox.env is removed when the JSON source no longer exists.
def test_sync_sandbox_env_removes_stale_env_without_json(tmp_path) -> None:
    json_path = tmp_path / "sandbox.json"
    env_path = tmp_path / "sandbox.env"
    env_path.write_text("SANDBOX_CPU_LIMIT=99\n", encoding="utf-8")

    sync_sandbox_env(json_path, env_path)

    assert not env_path.exists()


# Verify that a second sync with unchanged JSON does not rewrite sandbox.env (its mtime stays the same).
def test_sync_sandbox_env_keeps_unchanged_file(tmp_path) -> None:
    json_path = tmp_path / "sandbox.json"
    env_path = tmp_path / "sandbox.env"
    json_path.write_text(json.dumps({"SANDBOX_THREAD_LIMIT": 4}), encoding="utf-8")
    sync_sandbox_env(json_path, env_path)
    first_mtime = env_path.stat().st_mtime_ns

    sync_sandbox_env(json_path, env_path)

    assert env_path.stat().st_mtime_ns == first_mtime
