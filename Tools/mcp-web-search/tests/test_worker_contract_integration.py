# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


# Repository root, the web search server script and the tool worker script that loads it.
REPO_ROOT = Path(__file__).resolve().parents[3]
SERVER_FILE = Path(__file__).resolve().parents[1] / "mcp-server.py"
WORKER_FILE = REPO_ROOT / "Services" / "tool_worker.py"


# Run the tool worker as a subprocess for one operation with the JSON payload on stdin, assert it exits
# cleanly with an ok envelope, and return the envelope's result.
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


# Verify that the worker describes web_search with a required query that is either a string or an array
# of at most three queries, and read_page with a url that can also be an array.
def test_worker_publishes_string_and_three_query_batch_inputs():
    described = _worker("describe", {})

    web_tool = next(tool for tool in described["tools"] if tool["id"] == "web_search")
    read_tool = next(tool for tool in described["tools"] if tool["id"] == "read_page")
    assert web_tool["parameters"]["required"] == ["query"]
    web_string, web_batch = web_tool["parameters"]["properties"]["query"]["oneOf"]
    assert web_string["type"] == "string"
    assert "pattern" not in web_string
    assert web_batch["type"] == "array"
    assert web_batch["maxItems"] == 3
    assert read_tool["parameters"]["properties"]["url"]["oneOf"][1]["type"] == "array"


# Verify that the worker's prepare step matches the published batch contract: a single query and a
# three-query batch are accepted, a four-query batch is rejected with a message telling not to repeat the
# shape, and an alternative argument shape (call_description and web) is recovered into web arguments.
def test_worker_preflight_matches_the_published_batch_contract():
    # A single query string is accepted unchanged.
    good = _worker("prepare", {
        "tool_id": "web_search",
        "arguments": {
            "description": "Check search",
            "query": "github rate limit fallback",
        },
    })
    assert good["ok"] is True
    assert good["arguments"]["query"] == "github rate limit fallback"

    # A three-query batch is accepted and produces three planned queries.
    batch = _worker("prepare", {
        "tool_id": "web_search",
        "arguments": {"query": ["first query", "second query", "third query"]},
    })
    assert batch["ok"] is True
    assert len(batch["search_request"]["queries"]) == 3

    # A batch over the limit is rejected with a guidance message for the model.
    bad_cases = [
        (
            {
                "query": ["one", "two", "three", "four"],
                "description": "Check search",
            },
            "batch permits at most 3",
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

    # An alternative argument shape is recovered into the web arguments.
    recovered = _worker("prepare", {
        "tool_id": "web_search",
        "arguments": {
            "call_description": "Check search",
            "web": {"item": ["first query", "second query"]},
        },
    })
    assert recovered["ok"] is True
    assert recovered["arguments"]["web"] == ["first query", "second query"]
