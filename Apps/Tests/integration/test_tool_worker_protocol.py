# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest


pytestmark = [pytest.mark.integration]


WORKER_FILE = Path(__file__).resolve().parents[3] / "Services" / "tool_worker.py"


FAKE_SERVER = """
import time

MCP_SERVER = {
    "id": "protocol_suite",
    "name": "Protocol Suite",
    "description": "Fake tool server for worker protocol tests",
}

TOOLS = [
    {
        "id": "echo",
        "name": "Echo",
        "description": "Return the provided value.",
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
    },
    {
        "id": "slow",
        "name": "Slow",
        "description": "Sleep for a while before answering.",
        "parameters": {"type": "object", "properties": {}},
    },
]


def supports(engine=None, model_name=None):
    return engine == "ollama-service"


def echo(arguments, context=None):
    return {
        "value": arguments.get("value"),
        "chat_id": context.get("chat_id"),
    }


async def async_echo(arguments, context=None):
    return {"value": arguments.get("value"), "async": True}


def slow(arguments, context=None):
    time.sleep(6)
    return "finally done"


def call_tool(tool_id, arguments, context=None):
    return {"tool_id": tool_id, "via": "dispatcher"}


TOOL_HANDLERS = {
    "echo": echo,
    "async_echo": async_echo,
    "slow": slow,
}
"""


# Minimal protocol client speaking the real worker envelope format.
class WorkerClient:
    def __init__(self, server_file: Path):
        self.process = subprocess.Popen(
            [sys.executable, str(WORKER_FILE), "serve", str(server_file)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        self._lock = threading.Lock()

    # Send one request and read the next response envelope.
    def request(self, operation: str, payload: dict | None = None, *, raw: str | None = None) -> dict:
        line = raw if raw is not None else json.dumps({"operation": operation, "payload": payload or {}})
        with self._lock:
            assert self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()

            # Heartbeat events are skipped until the answer envelope arrives.
            while True:
                response_line = self.process.stdout.readline()
                assert response_line, "worker terminated before answering"
                envelope = json.loads(response_line)
                if "event" in envelope:
                    continue
                return envelope

    # Send one tool call and count the heartbeats emitted while it runs.
    def call_collecting_heartbeats(self, payload: dict) -> tuple[dict, int]:
        with self._lock:
            assert self.process.stdin is not None and self.process.stdout is not None
            self.process.stdin.write(json.dumps({"operation": "call", "payload": payload}) + "\n")
            self.process.stdin.flush()

            heartbeats = 0
            while True:
                response_line = self.process.stdout.readline()
                assert response_line, "worker terminated before answering"
                envelope = json.loads(response_line)
                if "event" in envelope:
                    heartbeats += 1
                    continue
                return envelope, heartbeats

    # Close stdin and wait for a clean worker exit.
    def close(self) -> int:
        if self.process.stdin:
            self.process.stdin.close()
        return self.process.wait(timeout=30)

    def terminate(self) -> None:
        self.process.kill()
        self.process.wait(timeout=30)


@pytest.fixture
def worker(tmp_path):
    server_file = tmp_path / "protocol_suite" / "mcp-server.py"
    server_file.parent.mkdir(parents=True)
    server_file.write_text(textwrap.dedent(FAKE_SERVER), encoding="utf-8")
    client = WorkerClient(server_file)
    yield client
    if client.process.poll() is None:
        client.terminate()


# describe.


# describe returns the server metadata and tool surface.
def test_describe_returns_metadata_and_tools(worker):
    envelope = worker.request("describe")

    assert envelope["ok"] is True
    metadata = envelope["result"]
    assert metadata["id"] == "protocol_suite"
    assert metadata["name"] == "Protocol Suite"
    tool_ids = [tool["id"] for tool in metadata["tools"]]
    assert tool_ids == ["echo", "slow"]


# supports.


# supports gates engines through the server's own predicate.
def test_supports_respects_server_predicate(worker):
    assert worker.request("supports", {"engine": "ollama-service"})["result"] is True
    assert worker.request("supports", {"engine": "openai"})["result"] is False


# Tool calls.


# Sync handlers receive arguments and context.
def test_call_executes_sync_handler_with_context(worker):
    envelope = worker.request(
        "call",
        {"tool_id": "echo", "arguments": {"value": "hello"}, "context": {"chat_id": "chat-9"}},
    )

    assert envelope["ok"] is True
    assert envelope["result"]["value"] == "hello"
    assert envelope["result"]["chat_id"] == "chat-9"


# Async handlers run through the worker's async runner.
def test_call_executes_async_handler(worker):
    envelope = worker.request(
        "call",
        {"tool_id": "async_echo", "arguments": {"value": "async"}},
    )

    assert envelope["ok"] is True
    assert envelope["result"] == {"value": "async", "async": True}


# Unregistered tools fall back to the generic dispatcher.
def test_call_falls_back_to_server_dispatcher(worker):
    envelope = worker.request("call", {"tool_id": "unregistered", "arguments": {}})

    assert envelope["ok"] is True
    assert envelope["result"] == {"tool_id": "unregistered", "via": "dispatcher"}


# Error envelopes.


# Broken JSON lines answer with a structured error envelope.
def test_broken_json_line_returns_error_envelope(worker):
    envelope = worker.request("", raw="{not json")

    assert envelope["ok"] is False
    assert "JSONDecodeError" in envelope["error"]


# Unknown operations answer with a structured error envelope.
def test_unknown_operation_returns_error_envelope(worker):
    envelope = worker.request("teleport")

    assert envelope["ok"] is False
    assert "Unknown worker operation" in envelope["error"]


# Heartbeats.


# Long-running calls emit heartbeats so the parent can distinguish a busy
# worker from a dead one.
def test_long_call_emits_heartbeats(worker):
    envelope, heartbeats = worker.call_collecting_heartbeats({"tool_id": "slow"})

    assert envelope["ok"] is True
    assert envelope["result"] == "finally done"
    assert heartbeats >= 1, "expected at least one heartbeat during the 6s call"


# Shutdown.


# The worker shuts down cleanly on stdin EOF.
def test_worker_exits_cleanly_on_stdin_close(worker):
    assert worker.close() == 0


# One-shot mode.


# One-shot mode answers a single request from stdin.
def test_one_shot_describe(tmp_path):
    server_file = tmp_path / "one_shot" / "mcp-server.py"
    server_file.parent.mkdir(parents=True)
    server_file.write_text(textwrap.dedent(FAKE_SERVER), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, str(WORKER_FILE), "describe", str(server_file)],
        input="{}",
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )

    assert completed.returncode == 0
    envelope = json.loads(completed.stdout.strip().splitlines()[-1])
    assert envelope["ok"] is True
    assert envelope["result"]["id"] == "protocol_suite"
