# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


# In-process fake of the managed Ollama HTTP API speaking NDJSON streaming.
class FakeOllamaServer:
    def __init__(self):
        self.chat_requests: list[dict] = []
        self.tags_payload: dict = {"models": []}
        self.show_payload: dict = {
            "license": "MIT",
            "modelfile": "# Modelfile",
            "parameters": "num_keep 5",
            "template": "{{ .Prompt }}",
            "details": {"families": ["llama"], "parameter_size": "8B"},
            "model_info": {"general.architecture": "llama"},
        }
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
                if self.path.startswith("/api/tags"):
                    self._send_json(200, server.tags_payload)
                    return
                self._send_json(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(length) or b"{}")

                if self.path.startswith("/api/chat"):
                    with server._lock:
                        server.chat_requests.append(body)

                    # Failures answer as JSON errors instead of a stream.
                    if server.chat_status != 200:
                        self._send_json(server.chat_status, {"error": "boom"})
                        return

                    self.send_response(200)
                    self.send_header("Content-Type", "application/x-ndjson")
                    self.end_headers()
                    for chunk in server.chat_chunks:
                        line = json.dumps(chunk, ensure_ascii=False)
                        self.wfile.write(line.encode("utf-8") + b"\n")
                    return

                if self.path.startswith("/api/show"):
                    with server._lock:
                        server.show_payload["_request"] = body
                    self._send_json(200, server.show_payload)
                    return

                self._send_json(404, {"error": "not found"})

        return Handler

    # Queue one raw chat response line.
    def enqueue_chunk(self, chunk: dict) -> None:
        self.chat_chunks.append(chunk)

    # Queue visible content chunks followed by the final done chunk.
    def enqueue_text(self, *parts: str, model: str = "fake-model") -> None:
        for part in parts:
            self.enqueue_chunk({"model": model, "message": {"role": "assistant", "content": part}, "done": False})

        self.enqueue_chunk(
            {
                "model": model,
                "message": {"role": "assistant", "content": ""},
                "done": True,
                "prompt_eval_count": 11,
                "eval_count": 22,
            }
        )

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


@pytest.fixture
def ollama_server():
    server = FakeOllamaServer()
    yield server
    server.stop()
