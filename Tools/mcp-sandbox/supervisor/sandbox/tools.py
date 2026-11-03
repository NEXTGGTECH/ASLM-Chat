# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import inspect
from typing import Annotated, Any

from mcp.server.fastmcp import Context
from pydantic import Field

from sandbox.api import TOOLS, handle_tool

# Metadata attached to every registered MCP tool: server identity, runtime and workspace root.
TOOL_META = {
    "x-aslm-server": "aslm-chat-sandbox",
    "x-aslm-runtime": "fastmcp-stdio",
    "x-aslm-workspace-root": ".",
}


# Send a best-effort progress notification when the client supports it.
async def _report_progress(ctx: Context | None, progress: float, message: str) -> None:
    if ctx is None:
        return

    try:
        await ctx.report_progress(progress, 100, message)
    except Exception:
        # Progress is optional; a client that cannot take it must not fail the tool call.
        pass


# Build a FastMCP annotation that preserves parameter guidance in its public schema.
def _parameter_annotation(schema: dict[str, Any]) -> Any:
    field_options: dict[str, Any] = {}
    if schema.get("description"):
        field_options["description"] = str(schema["description"])
    if schema.get("minLength") is not None:
        field_options["min_length"] = int(schema["minLength"])
    if schema.get("maxLength") is not None:
        field_options["max_length"] = int(schema["maxLength"])
    if not field_options:
        return Any
    return Annotated[Any, Field(**field_options)]


# Register sandbox tools on a FastMCP instance. For every entry of TOOLS a handler is built whose
# signature, annotations and docstring are generated from the tool definition, so FastMCP publishes
# the tool's parameter schema; the handler runs the blocking tool code in a worker thread.
def register_tools(mcp) -> None:
    for tool_definition in TOOLS:
        tool_id = tool_definition["id"]
        parameters = tool_definition.get("parameters", {}).get("properties", {})

        # Handler of one tool: bash gets live progress callbacks, other tools a start notification.
        async def _tool_impl(
            _tool_id: str = tool_id,
            ctx: Context | None = None,
            **kwargs: Any,
        ) -> dict[str, Any]:
            loop = asyncio.get_running_loop()

            # Forward progress reported from the worker thread back to the event loop.
            def _progress(progress: float, message: str) -> None:
                asyncio.run_coroutine_threadsafe(
                    _report_progress(ctx, progress, message),
                    loop,
                )

            if _tool_id == "bash":
                await _report_progress(ctx, 5, f"Starting {_tool_id}")
                return await asyncio.to_thread(
                    handle_tool,
                    _tool_id,
                    kwargs,
                    {},
                    progress_callback=_progress,
                )

            # Other tools: announce the target (path, source, pattern or the tool name) and run.
            label = kwargs.get("path") or kwargs.get("src") or kwargs.get("pattern") or _tool_id
            await _report_progress(ctx, 10, f"Running {_tool_id}: {label}")
            return await asyncio.to_thread(handle_tool, _tool_id, kwargs, {})

        # Give the handler the tool's name, description, annotations and an explicit keyword-only
        # signature (required parameters without default) so FastMCP derives the input schema from them.
        _tool_impl.__name__ = f"sandbox_{tool_id}"
        _tool_impl.__doc__ = tool_definition["description"]
        parameter_annotations = {key: _parameter_annotation(schema) for key, schema in parameters.items()}
        _tool_impl.__annotations__ = dict(parameter_annotations)
        _tool_impl.__annotations__["ctx"] = Context | None
        _tool_impl.__annotations__["return"] = dict[str, Any]
        signature_parameters = []
        required = set(tool_definition.get("parameters", {}).get("required", []))
        for key, schema in parameters.items():
            default = inspect._empty if key in required else schema.get("default", None)
            signature_parameters.append(
                inspect.Parameter(
                    key,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=default,
                    annotation=parameter_annotations[key],
                )
            )
        signature_parameters.append(
            inspect.Parameter(
                "ctx",
                inspect.Parameter.KEYWORD_ONLY,
                default=None,
                annotation=Context | None,
            )
        )
        _tool_impl.__signature__ = inspect.Signature(
            parameters=signature_parameters,
            return_annotation=dict[str, Any],
        )

        # Register the handler as an MCP tool.
        mcp.tool(name=tool_id, meta=TOOL_META)(_tool_impl)
