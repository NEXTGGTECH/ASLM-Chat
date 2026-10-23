# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_FILE = Path(__file__).resolve().parents[1] / "mcp-server.py"
WORKER_FILE = REPO_ROOT / "Services" / "tool_worker.py"


def _worker(operation: str, payload: dict[str, Any]) -> dict[str, Any]:
    completed = subprocess.run(
        [sys.executable, str(WORKER_FILE), operation, str(SERVER_FILE)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        cwd=REPO_ROOT,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    envelope = json.loads(completed.stdout)
    assert envelope["ok"] is True, envelope
    return envelope["result"]


def test_worker_publishes_string_and_two_query_batch_inputs():
    described = _worker("describe", {})

    web_tool = next(tool for tool in described["tools"] if tool["id"] == "web_search")
    read_tool = next(tool for tool in described["tools"] if tool["id"] == "read_page")
    assert web_tool["parameters"]["required"] == ["query"]
    web_string, web_batch = web_tool["parameters"]["properties"]["query"]["oneOf"]
    assert web_string["type"] == "string"
    assert "pattern" not in web_string
    assert web_batch["type"] == "array"
    assert web_batch["maxItems"] == 2
    assert read_tool["parameters"]["properties"]["url"]["oneOf"][1]["type"] == "array"


def test_worker_preflight_matches_the_published_batch_contract():
    good = _worker("prepare", {
        "tool_id": "web_search",
        "arguments": {
            "description": "Check search",
            "query": "github rate limit fallback",
        },
    })
    assert good["ok"] is True
    assert good["arguments"]["query"] == "github rate limit fallback"

    bad_cases = [
        (
            {
                "query": ["one", "two", "three"],
                "description": "Check search",
            },
            "batch permits at most 2",
        ),
    ]
    for arguments, expected_message in bad_cases:
        rejected = _worker("prepare", {
            "tool_id": "web_search",
            "arguments": arguments,
        })
        assert rejected["ok"] is False
        context = rejected["error_result"]["model_context"]
        assert expected_message in context
        assert "Never repeat the rejected shape" in context

    recovered = _worker("prepare", {
        "tool_id": "web_search",
        "arguments": {
            "call_description": "Check search",
            "web": {"item": ["first query", "second query"]},
        },
    })
    assert recovered["ok"] is True
    assert recovered["arguments"]["web"] == ["first query", "second query"]
