# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

# Project root, the MCP folder inside it and the user-editable mcp.json file.
BASE_DIR = Path(__file__).resolve().parent.parent
MCP_DIR = BASE_DIR / "MCP"
MCP_JSON_PATH = MCP_DIR / "mcp.json"

# Content of a freshly created mcp.json: an empty mcpServers object.
DEFAULT_MCP_JSON = (
    "{\n"
    '  "mcpServers": {\n'
    "  }\n"
    "}\n"
)

# Transport of a user MCP server: a local process over stdio or a remote HTTP endpoint.
TransportKind = Literal["stdio", "http"]


# One normalized MCP server entry from mcpServers. An entry is either stdio (command, args,
# env, cwd set) or http (url, headers set); the fields of the other transport stay empty.
@dataclass(frozen=True)
class UserMcpServerEntry:
    config_key: str
    server_id: str
    display_name: str
    transport: TransportKind
    command: str | None
    args: list[str]
    env: dict[str, str] | None
    cwd: str | None
    url: str | None
    headers: dict[str, str] | None


# Build a stable slug from a display name for MCP server ids.
def _slugify(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")
    return normalized or "mcp"


# Create MCP/ and a default mcp.json when the file is missing.
def ensure_default_mcp_json() -> None:
    MCP_DIR.mkdir(parents=True, exist_ok=True)
    if MCP_JSON_PATH.is_file():
        return
    MCP_JSON_PATH.write_text(DEFAULT_MCP_JSON, encoding="utf-8")


# Return path and mtime for cache invalidation, or None when the file is absent.
def mcp_json_signature_for(module_dir: Path | None = None) -> tuple[str, int] | None:
    path = (Path(module_dir) if module_dir is not None else BASE_DIR) / "MCP" / "mcp.json"
    if not path.is_file():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return (str(path.resolve()), stat.st_mtime_ns)


# Return path and mtime for cache invalidation, or None when the file is absent.
def mcp_json_signature() -> tuple[str, int] | None:
    return mcp_json_signature_for(BASE_DIR)


# Pick a unique server id that does not collide with reserved ids.
def _unique_server_id(base: str, taken: set[str]) -> str:
    candidate = base
    if candidate not in taken:
        return candidate
    prefixed = f"user_{base}"
    if prefixed not in taken:
        return prefixed
    index = 2
    while True:
        probe = f"user_{base}_{index}"
        if probe not in taken:
            return probe
        index += 1


# Parse one module's mcp.json and return user server entries with stable ids.
def iter_user_mcp_entries_for(module_dir: Path, reserved_ids: set[str]) -> list[UserMcpServerEntry]:
    # Read the module's MCP/mcp.json; a missing, unreadable or malformed file yields no entries.
    path = Path(module_dir) / "MCP" / "mcp.json"
    if not path.is_file():
        return []

    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError, ValueError):
        return []

    if not isinstance(data, dict):
        return []

    servers = data.get("mcpServers")
    if not isinstance(servers, dict):
        return []

    taken: set[str] = set(reserved_ids)
    result: list[UserMcpServerEntry] = []

    for raw_key, raw_entry in servers.items():
        # Skip entries that are not objects, have no name, or are neither or both of url/command.
        if not isinstance(raw_entry, dict):
            continue
        key = str(raw_key or "").strip()
        if not key:
            continue

        url = str(raw_entry.get("url") or "").strip() or None
        command = str(raw_entry.get("command") or "").strip() or None

        if url and command:
            continue
        if not url and not command:
            continue

        # Derive a stable server id from the entry name that avoids reserved and earlier ids.
        base_id = _slugify(key)
        server_id = _unique_server_id(base_id, taken)
        taken.add(server_id)

        # Normalize the optional args, env, cwd and headers fields, dropping values of the wrong type.
        args_raw = raw_entry.get("args")
        args: list[str] = []
        if isinstance(args_raw, list):
            args = [str(a) for a in args_raw if str(a).strip()]

        env: dict[str, str] | None = None
        if isinstance(raw_entry.get("env"), dict):
            env = {str(k): str(v) for k, v in raw_entry["env"].items()}

        cwd_raw = raw_entry.get("cwd")
        cwd = str(cwd_raw).strip() if isinstance(cwd_raw, str) and str(cwd_raw).strip() else None

        headers: dict[str, str] | None = None
        if isinstance(raw_entry.get("headers"), dict):
            headers = {str(k): str(v) for k, v in raw_entry["headers"].items()}

        # An http entry keeps only url and headers; a stdio entry keeps command, args, env and cwd.
        if url:
            result.append(
                UserMcpServerEntry(
                    config_key=key,
                    server_id=server_id,
                    display_name=key,
                    transport="http",
                    command=None,
                    args=[],
                    env=None,
                    cwd=None,
                    url=url,
                    headers=headers,
                )
            )
            continue

        result.append(
            UserMcpServerEntry(
                config_key=key,
                server_id=server_id,
                display_name=key,
                transport="stdio",
                command=command,
                args=args,
                env=env,
                cwd=cwd,
                url=None,
                headers=None,
            )
        )

    return result


# Parse mcp.json and return user server entries with stable ids.
def iter_user_mcp_entries(reserved_ids: set[str]) -> list[UserMcpServerEntry]:
    ensure_default_mcp_json()
    return iter_user_mcp_entries_for(BASE_DIR, reserved_ids)


# Return mcp.json contents, creating a default file when needed.
def load_raw_text() -> str:
    ensure_default_mcp_json()
    return MCP_JSON_PATH.read_text(encoding="utf-8")


# Parse mcp.json into a dictionary.
def load_parsed() -> dict[str, Any]:
    ensure_default_mcp_json()
    raw = MCP_JSON_PATH.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {MCP_JSON_PATH}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("mcp.json root must be a JSON object")
    return data


# Raise ValueError when the document is not a valid MCP configuration.
def validate_mcp_document(data: dict[str, Any]) -> None:
    # The top level must contain an mcpServers object.
    servers = data.get("mcpServers")
    if servers is None:
        raise ValueError("Missing top-level key 'mcpServers'")
    if not isinstance(servers, dict):
        raise ValueError("'mcpServers' must be an object")

    for key, entry in servers.items():
        # Each entry must define exactly one transport: url (HTTP) or command (stdio).
        if not isinstance(entry, dict):
            raise ValueError(f"mcpServers[{key!r}] must be an object")
        has_url = bool(str(entry.get("url") or "").strip())
        has_cmd = bool(str(entry.get("command") or "").strip())
        if has_url and has_cmd:
            raise ValueError(f"mcpServers[{key!r}]: specify either 'url' or 'command', not both")
        if not has_url and not has_cmd:
            raise ValueError(f"mcpServers[{key!r}]: need 'url' (HTTP) or 'command' (stdio)")

        # A stdio entry may carry args (list of strings) and env (object of strings).
        if has_cmd:
            args = entry.get("args")
            if args is not None and not isinstance(args, list):
                raise ValueError(f"mcpServers[{key!r}]: 'args' must be an array of strings")
            if isinstance(args, list):
                for item in args:
                    if not isinstance(item, str):
                        raise ValueError(f"mcpServers[{key!r}]: each arg must be a string")
            env = entry.get("env")
            if env is not None and not isinstance(env, dict):
                raise ValueError(f"mcpServers[{key!r}]: 'env' must be an object of strings")
            if isinstance(env, dict):
                for env_k, env_v in env.items():
                    if not isinstance(env_k, str) or not isinstance(env_v, str):
                        raise ValueError(f"mcpServers[{key!r}]: env keys and values must be strings")

        # An http entry may carry headers (object of strings).
        if has_url:
            headers = entry.get("headers")
            if headers is not None and not isinstance(headers, dict):
                raise ValueError(f"mcpServers[{key!r}]: 'headers' must be an object of strings")
            if isinstance(headers, dict):
                for hk, hv in headers.items():
                    if not isinstance(hk, str) or not isinstance(hv, str):
                        raise ValueError(f"mcpServers[{key!r}]: header keys and values must be strings")


# Validate JSON and atomically write mcp.json.
def save_raw_text(text: str) -> None:
    # Reject text that is not a JSON object or not a valid MCP document before touching the file.
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Root value must be a JSON object")
    validate_mcp_document(data)

    # Write the normalized JSON to a temporary file in the same folder and replace mcp.json with it,
    # so readers never see a half-written file.
    ensure_default_mcp_json()
    normalized = json.dumps(data, ensure_ascii=False, indent=2) + "\n"

    fd, tmp_path = tempfile.mkstemp(prefix="mcp_", suffix=".json", dir=str(MCP_DIR))
    try:
        import os

        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(normalized)
        Path(tmp_path).replace(MCP_JSON_PATH)
    except Exception:
        try:
            Path(tmp_path).unlink(missing_ok=True)
        except OSError:
            # The leftover temporary file is harmless; the original write error is re-raised below.
            pass
        raise
