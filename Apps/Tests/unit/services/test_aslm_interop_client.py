# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from Services import aslm_interop_client


pytestmark = [pytest.mark.unit]


# Local HTTP server standing in for the ASLM host interop endpoint: it records every request
# (method, path, content type, parsed JSON body) and answers every GET and POST with the scripted
# status and body.
class _InteropServer:
    # Start the server on a free localhost port in a daemon thread.
    def __init__(self):
        # Recorded requests and the scripted response.
        self.requests: list[dict] = []
        self.response_status = 200
        self.response_body: bytes = b"{}"

        server = self

        # Request handler that records the request and replies from the server's scripted response.
        class Handler(BaseHTTPRequestHandler):
            # Answer one interop request and record it.
            def _respond(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else b""
                server.requests.append(
                    {
                        "method": self.command,
                        "path": self.path,
                        "content_type": self.headers.get("Content-Type"),
                        "body": json.loads(body) if body else None,
                    }
                )
                self.send_response(server.response_status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(server.response_body)))
                self.end_headers()
                self.wfile.write(server.response_body)

            # GET and POST are handled identically.
            do_GET = _respond
            do_POST = _respond

            # Silence the default per-request access log.
            def log_message(self, *args):
                pass

        # Bind port 0 so the OS picks a free port, then serve in the background.
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    # Script the next response with a JSON payload or raw text.
    def respond_with(self, status: int, payload: dict | str) -> None:
        self.response_status = status
        self.response_body = payload.encode("utf-8") if isinstance(payload, str) else json.dumps(payload).encode("utf-8")

    # Shut the HTTP server down and release its socket.
    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


# Provide a running interop server and point the module interop base URL environment variable at it;
# stop the server afterwards.
@pytest.fixture
def interop_server(monkeypatch):
    server = _InteropServer()
    monkeypatch.setenv("ASLM_MODULE_INTEROP_BASE_URL", f"http://127.0.0.1:{server.port}")
    yield server
    server.stop()


# Base URL resolution

# Verify that a missing host-provided base URL environment variable raises a RuntimeError naming it.
def test_base_url_requires_host_environment(monkeypatch):
    monkeypatch.delenv("ASLM_MODULE_INTEROP_BASE_URL", raising=False)

    with pytest.raises(RuntimeError, match="ASLM_MODULE_INTEROP_BASE_URL"):
        aslm_interop_client.get_registry()


# Registry

# Verify that the registry request is a GET to /v1/registry and its JSON response is returned parsed.
def test_get_registry_parses_response(interop_server):
    interop_server.respond_with(200, {"modules": [{"id": "aslm-chat"}]})

    registry = aslm_interop_client.get_registry()

    assert registry["modules"][0]["id"] == "aslm-chat"
    request = interop_server.requests[0]
    assert request["method"] == "GET"
    assert request["path"] == "/v1/registry"


# Module starts

# Verify that a start request is a JSON POST to /v1/modules/start carrying the caller and requested
# module ids.
def test_request_start_sends_payload(interop_server):
    interop_server.respond_with(200, {"started": ["other-module"]})

    result = aslm_interop_client.request_start(
        caller_module_id="aslm-chat",
        module_ids=["other-module", "another"],
    )

    assert result == {"started": ["other-module"]}
    request = interop_server.requests[0]
    assert request["method"] == "POST"
    assert request["path"] == "/v1/modules/start"
    assert request["content_type"].startswith("application/json")
    assert request["body"] == {
        "callerModuleId": "aslm-chat",
        "moduleIds": ["other-module", "another"],
    }


# Error handling

# Verify that a JSON error body from the host (HTTP 409) is returned as the call result.
def test_request_start_returns_json_error_payload(interop_server):
    interop_server.respond_with(409, {"error": "module already running"})

    result = aslm_interop_client.request_start(caller_module_id="aslm-chat", module_ids=["x"])

    assert result == {"error": "module already running"}


# Verify that a non-JSON error body (HTTP 500) raises a RuntimeError that names the status code.
def test_request_start_raises_on_non_json_error(interop_server):
    interop_server.respond_with(500, "internal error")

    with pytest.raises(RuntimeError, match="HTTP 500"):
        aslm_interop_client.request_start(caller_module_id="aslm-chat", module_ids=["x"])
