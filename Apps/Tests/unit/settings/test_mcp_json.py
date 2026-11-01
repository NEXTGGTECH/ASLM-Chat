# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json

import pytest

from Settings import mcp_json


pytestmark = [pytest.mark.unit]


@pytest.fixture
def isolated_mcp(tmp_path, monkeypatch):
    mcp_dir = tmp_path / "MCP"
    monkeypatch.setattr(mcp_json, "MCP_DIR", mcp_dir)
    monkeypatch.setattr(mcp_json, "MCP_JSON_PATH", mcp_dir / "mcp.json")
    return mcp_dir


# Bootstrap.


# ensure_default_mcp_json creates the directory and a valid empty document.
def test_ensure_default_creates_empty_document(isolated_mcp):
    mcp_json.ensure_default_mcp_json()

    document = json.loads((isolated_mcp / "mcp.json").read_text(encoding="utf-8"))
    assert document == {"mcpServers": {}}


# An existing file is never overwritten by the bootstrap.
def test_ensure_default_preserves_existing_document(isolated_mcp):
    path = isolated_mcp / "mcp.json"
    path.parent.mkdir(parents=True)
    path.write_text('{"mcpServers": {"custom": {"command": "run"}}}', encoding="utf-8")

    mcp_json.ensure_default_mcp_json()

    assert json.loads(path.read_text(encoding="utf-8"))["mcpServers"]["custom"]["command"] == "run"


# Entry parsing.


# Stdio entries normalize command, args, env, and cwd.
def test_iter_entries_parses_stdio_server(isolated_mcp):
    module_dir = isolated_mcp.parent
    (module_dir / "MCP").mkdir(exist_ok=True)
    (module_dir / "MCP" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "Filesystem": {
                        "command": "npx",
                        "args": ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
                        "env": {"NODE_ENV": "production"},
                        "cwd": "/srv/tools",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    entries = mcp_json.iter_user_mcp_entries_for(module_dir, set())

    assert len(entries) == 1
    entry = entries[0]
    assert entry.config_key == "Filesystem"
    assert entry.server_id == "filesystem"
    assert entry.transport == "stdio"
    assert entry.command == "npx"
    assert entry.args == ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    assert entry.env == {"NODE_ENV": "production"}
    assert entry.cwd == "/srv/tools"
    assert entry.url is None
    assert entry.headers is None


# HTTP entries carry url and headers instead of a command.
def test_iter_entries_parses_http_server(isolated_mcp):
    module_dir = isolated_mcp.parent
    (module_dir / "MCP").mkdir(exist_ok=True)
    (module_dir / "MCP" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "remote": {
                        "url": "https://mcp.example.com/stream",
                        "headers": {"Authorization": "Bearer token"},
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    entries = mcp_json.iter_user_mcp_entries_for(module_dir, set())

    assert len(entries) == 1
    entry = entries[0]
    assert entry.transport == "http"
    assert entry.url == "https://mcp.example.com/stream"
    assert entry.headers == {"Authorization": "Bearer token"}
    assert entry.command is None


# Entries specifying both url and command are ambiguous and skipped.
def test_iter_entries_skips_ambiguous_and_empty_entries(isolated_mcp):
    module_dir = isolated_mcp.parent
    (module_dir / "MCP").mkdir(exist_ok=True)
    (module_dir / "MCP" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "both": {"url": "https://a.example.com", "command": "run"},
                    "neither": {"args": ["x"]},
                    "": {"command": "run"},
                    "not_an_object": "oops",
                    "good": {"command": "run"},
                }
            }
        ),
        encoding="utf-8",
    )

    entries = mcp_json.iter_user_mcp_entries_for(module_dir, set())

    assert [entry.config_key for entry in entries] == ["good"]


# Server ids avoid collisions with reserved ids by prefixing.
def test_iter_entries_generates_unique_ids_against_reserved(isolated_mcp):
    module_dir = isolated_mcp.parent
    (module_dir / "MCP").mkdir(exist_ok=True)
    (module_dir / "MCP" / "mcp.json").write_text(
        json.dumps(
            {
                "mcpServers": {
                    "Echo": {"command": "a"},
                    "echo two": {"command": "b"},
                }
            }
        ),
        encoding="utf-8",
    )

    entries = mcp_json.iter_user_mcp_entries_for(module_dir, {"echo"})

    assert [entry.server_id for entry in entries] == ["user_echo", "echo_two"]


# Broken documents degrade to an empty entry list instead of raising.
def test_iter_entries_returns_empty_for_invalid_document(isolated_mcp):
    module_dir = isolated_mcp.parent
    (module_dir / "MCP").mkdir(exist_ok=True)
    (module_dir / "MCP" / "mcp.json").write_text("{broken", encoding="utf-8")

    assert mcp_json.iter_user_mcp_entries_for(module_dir, set()) == []


# A missing document simply yields no entries.
def test_iter_entries_returns_empty_for_missing_file(tmp_path):
    assert mcp_json.iter_user_mcp_entries_for(tmp_path, set()) == []


# Document validation.


# Document validation rejects malformed server definitions.
@pytest.mark.parametrize(
    "document",
    [
        {},
        {"mcpServers": []},
        {"mcpServers": {"bad": "text"}},
        {"mcpServers": {"both": {"url": "https://a.b", "command": "run"}}},
        {"mcpServers": {"neither": {"args": []}}},
        {"mcpServers": {"args": {"command": "run", "args": "nope"}}},
        {"mcpServers": {"argtype": {"command": "run", "args": [1]}}},
        {"mcpServers": {"envtype": {"command": "run", "env": ["x"]}}},
        {"mcpServers": {"envval": {"command": "run", "env": {"K": 1}}}},
        {"mcpServers": {"hdrtype": {"url": "https://a.b", "headers": ["x"]}}},
        {"mcpServers": {"hdrval": {"url": "https://a.b", "headers": {"K": 3}}}},
    ],
)
def test_validate_mcp_document_rejects_invalid_documents(document):
    with pytest.raises(ValueError):
        mcp_json.validate_mcp_document(document)


# Well-formed stdio and HTTP documents pass validation.
def test_validate_mcp_document_accepts_valid_servers():
    mcp_json.validate_mcp_document(
        {
            "mcpServers": {
                "stdio": {"command": "run", "args": ["-v"], "env": {"K": "V"}},
                "http": {"url": "https://a.b", "headers": {"K": "V"}},
            }
        }
    )


# Persistence.


# save_raw_text normalizes JSON and persists it atomically.
def test_save_raw_text_persists_normalized_document(isolated_mcp):
    mcp_json.save_raw_text('{"mcpServers":{"srv":{"command":"run"}}}')

    saved = json.loads((isolated_mcp / "mcp.json").read_text(encoding="utf-8"))
    assert saved == {"mcpServers": {"srv": {"command": "run"}}}
    assert (isolated_mcp / "mcp.json").read_text(encoding="utf-8").endswith("\n")


# Invalid JSON or invalid documents leave the stored file untouched.
def test_save_raw_text_rejects_invalid_input(isolated_mcp):
    mcp_json.save_raw_text('{"mcpServers":{"srv":{"command":"run"}}}')
    before = (isolated_mcp / "mcp.json").read_text(encoding="utf-8")

    with pytest.raises(ValueError):
        mcp_json.save_raw_text("{broken")
    with pytest.raises(ValueError):
        mcp_json.save_raw_text('{"mcpServers":{"bad":{"url":"https://a","command":"r"}}}')

    assert (isolated_mcp / "mcp.json").read_text(encoding="utf-8") == before


# load_parsed returns the parsed document or raises a readable error.
def test_load_parsed_roundtrip_and_errors(isolated_mcp):
    mcp_json.save_raw_text('{"mcpServers": {}}')

    assert mcp_json.load_parsed() == {"mcpServers": {}}

    (isolated_mcp / "mcp.json").write_text("[1]", encoding="utf-8")
    with pytest.raises(ValueError, match="root must be a JSON object"):
        mcp_json.load_parsed()

    (isolated_mcp / "mcp.json").write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        mcp_json.load_parsed()


# The signature changes with the file state so caches can invalidate.
def test_signature_tracks_file_presence(isolated_mcp):
    assert mcp_json.mcp_json_signature_for(isolated_mcp.parent) is None

    mcp_json.ensure_default_mcp_json()
    signature = mcp_json.mcp_json_signature_for(isolated_mcp.parent)

    assert signature is not None
    assert signature[0].endswith("mcp.json")
