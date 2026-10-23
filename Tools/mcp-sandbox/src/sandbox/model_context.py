# Copyright NEXTGGTECH. Elastic License 2.0.

"""Small, allowlisted environment summary for the model; never export the raw config."""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from sandbox import config


_EXECUTION_FIELDS = {
    "default_timeout_s": ("SANDBOX_DEFAULT_TIMEOUT", "DEFAULT_TIMEOUT"),
    "output_bytes_per_stream": ("SANDBOX_MAX_OUTPUT_BYTES", "MAX_OUTPUT_BYTES"),
    "read_bytes": ("SANDBOX_MAX_READ_BYTES", "MAX_READ_BYTES"),
    "plain_cat_preview_bytes": ("SANDBOX_MAX_CAT_FILE_BYTES", "MAX_CAT_FILE_BYTES"),
    "image_preview_bytes": ("SANDBOX_MAX_IMAGE_PREVIEW_BYTES", "MAX_IMAGE_PREVIEW_BYTES"),
    "directory_entries": ("SANDBOX_MAX_LS_ENTRIES", "MAX_LS_ENTRIES"),
    "find_results": ("SANDBOX_MAX_FIND_RESULTS", "MAX_FIND_RESULTS"),
    "grep_results": ("SANDBOX_MAX_GREP_RESULTS", "MAX_GREP_RESULTS"),
    "background_after_s": ("SANDBOX_BACKGROUND_TIMEOUT_THRESHOLD", "BACKGROUND_TIMEOUT_THRESHOLD"),
    "library_threads": ("SANDBOX_THREAD_LIMIT", "THREAD_LIMIT"),
    "network_limit_mbit": ("SANDBOX_NETWORK_LIMIT_MBIT", "NETWORK_LIMIT_MBIT"),
    "workspace_idle_cleanup_s": ("SANDBOX_WORKSPACE_CLEANUP_IDLE_SECONDS", "WORKSPACE_CLEANUP_IDLE_SECONDS"),
}


def _execution_limits(environment: dict[str, str] | None = None) -> dict[str, Any]:
    limits = {}
    for field, (env_key, attr) in _EXECUTION_FIELDS.items():
        value = getattr(config, attr)
        if environment is not None and env_key in environment:
            try:
                value = int(environment[env_key])
            except (TypeError, ValueError):
                value = "unknown"
        elif environment is not None:
            value = "unknown"
        limits[field] = value
    cleanup = config.WORKSPACE_CLEANUP_ENABLED
    if environment is not None:
        raw = environment.get("SANDBOX_WORKSPACE_CLEANUP_ENABLED")
        cleanup = raw.strip().lower() not in {"0", "false", "no", "off"} if raw is not None else "unknown"
    limits["workspace_cleanup_enabled"] = cleanup
    return limits


def _matching_container() -> dict[str, Any] | None:
    """Read-only inspection: never start Docker, create a container, or expose host paths."""
    try:
        result = subprocess.run(
            ["docker", "inspect", config.CONTAINER_NAME],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=2,
        )
        if result.returncode:
            return None
        info = json.loads(result.stdout)[0]
        expected = os.path.normcase(os.path.normpath(
            os.path.join(config.HOST_WORKSPACE, config.DEFAULT_TASK_DIR)
        ))
        matching_mount = any(
            mount.get("Destination") == config.MODEL_WORKSPACE_CONTAINER
            and os.path.normcase(os.path.normpath(str(mount.get("Source", "")))) == expected
            for mount in info.get("Mounts", [])
        )
        return info if matching_mount and info.get("State", {}).get("Running") else None
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, IndexError, TypeError):
        return None


def get_model_context() -> str:
    """Return configured limits and, when verifiable, the matching live container's limits."""
    settings = {
        "cpu_quota_cores": config.CPU_LIMIT,
        "memory": config.MEMORY_LIMIT,
        "memory_plus_swap": config.MEMORY_SWAP_LIMIT,
        "processes_and_threads_total": config.PIDS_LIMIT,
        "requested_writable_layer_size": config.STORAGE_LIMIT,
        "internet_enabled": config.NETWORK_ENABLED,
        **_execution_limits(),
    }
    lines = [
        "Sandbox environment (only applies to sandbox tools):",
        f"Linux container; bash; workspace root: {config.MODEL_WORKSPACE_CONTAINER}; cwd '.' means this root.",
        "Internet access (configuration): " + ("enabled." if config.NETWORK_ENABLED else "disabled.")
        + " This setting is not a connectivity test; network_limit_mbit=0 means no bandwidth cap, not disabled internet.",
        "Configured limits for newly created containers: " + json.dumps(settings, ensure_ascii=False),
    ]
    runtime = _matching_container() if not config.IN_CONTAINER else None
    if runtime is None:
        lines.append(
            "Live limits were not verified for this workspace. Configuration is not proof that a "
            "running container has these limits; inspect cgroups before resource-heavy work."
        )
    else:
        host = runtime.get("HostConfig", {})
        environment = dict(
            item.split("=", 1) for item in runtime.get("Config", {}).get("Env", []) if "=" in item
        )
        nano_cpus = host.get("NanoCpus", 0)
        quota, period = host.get("CpuQuota", 0), host.get("CpuPeriod", 0)
        cpu = nano_cpus / 1_000_000_000 if nano_cpus else (
            quota / period if quota and quota > 0 and period else "no explicit quota"
        )
        actual = {
            "cpu_quota_cores": cpu,
            "memory_bytes": host.get("Memory", 0),
            "memory_plus_swap_bytes": host.get("MemorySwap", 0),
            "processes_and_threads_total": host.get("PidsLimit"),
            **_execution_limits(environment),
        }
        lines.append("Verified running container limits: " + json.dumps(actual, ensure_ascii=False))
        lines.append("Use the running limits for this session when they differ from configuration.")
    lines.extend([
        "The PID limit counts processes AND threads together; it is not the number of parallel jobs. "
        "CPU quota is capacity, not reserved physical cores. Bound worker pools by both CPU and RAM; "
        "avoid multiplying library threads across many worker processes.",
        "For archive extraction, check free disk space and unpack incrementally; writable-layer "
        "storage settings do not guarantee free space in the mounted workspace. Extra CPUs only "
        "help decompressors that support parallelism.",
        "Use a larger timeout_s (for example 300) for installs, builds, or large archives. "
        "Use background jobs and poll them for long work. Output is truncated, so save large "
        "results to workspace files and inspect relevant ranges. Workspace cleanup may remove "
        "idle files; save deliverables before the idle deadline.",
    ])
    return "\n".join(lines)
