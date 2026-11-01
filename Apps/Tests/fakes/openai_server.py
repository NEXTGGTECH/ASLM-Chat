# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


# In-process fake of an OpenAI-compatible endpoint speaking SSE streaming.
class FakeOpenAiServer:
    def __init__(self):
        self.chat_requests: list[dict] = []
        self.auth_headers: list[str] = []
        self.models_payload: dict = {"object": "list", "data": []}
        self.chat_status = 200
        self.chat_chunks: list[dict] = []
        self._lock = threading.Lock()

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), self._make_handler())
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    # Build the request handler bound to this server instance.
    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            # Answer one JSON request with the given status and payload.
            def _send_json(self, status: int, payload: dict):
                body = json.dumps(payload).encode("utf-8")
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/v1/models"):
                    server.auth_headers.append(self.headers.get("Authorization") or "")
                    self._send_json(200, server.models_payload)
                    return
                self._send_json(404, {"error": {"message": "not found"}})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")

                if self.path.startswith("/v1/chat/completions"):
                    with server._lock:
                        server.chat_requests.append(body)
                        server.auth_headers.append(self.headers.get("Authorization") or "")

                    # Failures answer as JSON errors instead of a stream.
                    if server.chat_status != 200:
                        self._send_json(server.chat_status, {"error": {"message": "boom"}})
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    for chunk in server.chat_chunks:
                        self.wfile.write(b"data: " + json.dumps(chunk, ensure_ascii=False).encode("utf-8") + b"\n\n")
                    self.wfile.write(b"data: [DONE]\n\n")
                    return

                self._send_json(404, {"error": {"message": "not found"}})

        return Handler

    # Queue one streaming delta chunk.
    def enqueue_delta(self, delta: dict, *, finish_reason=None, model: str = "fake-model") -> None:
        self.chat_chunks.append(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion.chunk",
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
        )

    # Queue visible text deltas and the final done chunk.
    def enqueue_text(self, *parts: str, model: str = "fake-model") -> None:
        for part in parts:
            self.enqueue_delta({"content": part}, model=model)

        self.enqueue_delta({}, finish_reason="stop", model=model)

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def openai_server():
    server = FakeOpenAiServer()
    yield server
    server.stop()
