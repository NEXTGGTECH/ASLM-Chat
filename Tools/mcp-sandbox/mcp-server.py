# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parent
for _p in (_ROOT / "supervisor", _ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# MCP_SERVER / TOOL_HANDLERS / TOOLS are re-exported by name: ASLM's tool_worker
# and API.mcp read them as module attributes from this entry module.
from sandbox.api import MCP_SERVER, TOOL_HANDLERS, TOOLS, handle_tool  # noqa: F401
from sandbox.config import IN_CONTAINER
from sandbox.model_context import get_model_context  # noqa: F401
from sandbox.temporal import run_temporal_bash

# When imported host-side (e.g. by ASLM's tool_worker), wire bash execution
# through docker instead of trying to run /bin/bash natively on the host.
if not IN_CONTAINER:
    from sandbox.docker_host import (
        _exec_bash_docker,
        foreground_background_job,
        kill_background_job,
        list_background_jobs,
    )
    import sandbox.container as _container
    import sandbox.api as _api
    _container.exec_bash = _exec_bash_docker
    _container.foreground_background_job = foreground_background_job
    _container.kill_background_job = kill_background_job
    _container.list_background_jobs = list_background_jobs
    _api.exec_bash = _exec_bash_docker
    _api.foreground_background_job = foreground_background_job
    _api.kill_background_job = kill_background_job
    _api.list_background_jobs = list_background_jobs


# Expose this tool server for engines that support tool-calling.
def supports(engine: str | None = None, model_name: str | None = None) -> bool:
    return engine in ("ollama-service", "lms", "openai", "google-genai")


# Dispatch one sandbox v2 tool.
def call_tool(
    tool_id: str,
    arguments: dict[str, Any] | None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if (
        tool_id == "bash"
        and isinstance(context, dict)
        and isinstance(context.get("temporal_sandbox"), dict)
    ):
        return run_temporal_bash(arguments or {}, context)
    return handle_tool(tool_id, arguments or {}, context or {})


# Register sandbox tools on a FastMCP instance.
def register_tools(mcp) -> None:
    from sandbox.tools import register_tools as register_fastmcp_tools

    register_fastmcp_tools(mcp)
