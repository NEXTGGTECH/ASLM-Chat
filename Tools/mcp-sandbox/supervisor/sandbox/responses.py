# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


# Structured error raised by sandbox helpers. error_type is the machine-readable code reported in the
# error envelope; result, warnings and truncated carry optional partial output through to the response.
@dataclass
class SandboxToolError(Exception):
    error_type: str
    message: str
    result: dict[str, Any] | None = None
    warnings: list[str] | None = None
    truncated: bool = False

    # Use the message as the string form of the exception.
    def __str__(self) -> str:
        return self.message


# Wrap a successful tool result in the sandbox v2 envelope.
def success_response(
    tool: str,
    result: dict[str, Any] | None = None,
    *,
    warnings: list[str] | None = None,
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "ok": True,
        "tool": tool,
        "result": result or {},
        "error": None,
        "warnings": list(warnings or []),
        "truncated": bool(truncated),
    }


# Wrap a failed tool result in the sandbox v2 envelope.
def error_response(
    tool: str,
    error_type: str,
    message: str,
    *,
    result: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    truncated: bool = False,
) -> dict[str, Any]:
    return {
        "ok": False,
        "tool": tool,
        "result": result or {},
        "error": {
            "type": error_type,
            "message": message,
        },
        "warnings": list(warnings or []),
        "truncated": bool(truncated),
    }


# Map Python exceptions into typed sandbox v2 errors: SandboxToolError keeps its own type and payload;
# common filesystem and value errors get specific codes; anything else is an internal_error.
def exception_response(tool: str, exc: Exception) -> dict[str, Any]:
    if isinstance(exc, SandboxToolError):
        return error_response(
            tool,
            exc.error_type,
            exc.message,
            result=exc.result,
            warnings=exc.warnings,
            truncated=exc.truncated,
        )

    if isinstance(exc, FileNotFoundError):
        return error_response(tool, "not_found", str(exc))
    if isinstance(exc, NotADirectoryError):
        return error_response(tool, "not_a_directory", str(exc))
    if isinstance(exc, IsADirectoryError):
        return error_response(tool, "is_a_directory", str(exc))
    if isinstance(exc, PermissionError):
        return error_response(tool, "permission_denied", str(exc))
    if isinstance(exc, ValueError):
        return error_response(tool, "invalid_arguments", str(exc))

    return error_response(tool, "internal_error", str(exc))
