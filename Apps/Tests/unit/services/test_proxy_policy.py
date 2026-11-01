# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import socket
import threading
import pytest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch
from django.test import SimpleTestCase
from Settings.proxy_policy import apply_loopback_proxy_bypass, build_proxy_environment_overlay, urlopen_direct, urlopen_with_loopback_bypass


pytestmark = [pytest.mark.unit]


# Verify loopback proxy bypass configuration and transport behavior.
class LoopbackProxyPolicyTests(SimpleTestCase):
    # Start a small HTTP server that identifies which endpoint received a request.
    def _start_http_server(self, label):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                body = label.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, requests

    # Stop one test HTTP server and wait for its thread to exit.
    def _stop_http_server(self, server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    def test_policy_merges_existing_bypass_and_loopback_hosts_idempotently(self):
        environ = {
            "NO_PROXY": "corp.example;127.0.0.1;[::1]",
            "no_proxy": "internal.example,LOCALHOST",
        }

        first = apply_loopback_proxy_bypass(
            environ,
            system_proxies={},
            system_bypass=[],
        )
        second = apply_loopback_proxy_bypass(
            environ,
            system_proxies={},
            system_bypass=[],
        )

        self.assertEqual(first, second)
        self.assertEqual(environ["NO_PROXY"], environ["no_proxy"])
        tokens = {token.casefold() for token in first.split(",")}
        self.assertEqual(
            tokens,
            {
                "corp.example",
                "internal.example",
                "localhost",
                "127.0.0.1",
                "::1",
            },
        )

    def test_policy_preserves_external_system_proxy_and_native_bypass(self):
        environ = {}

        apply_loopback_proxy_bypass(
            environ,
            system_proxies={
                "http": "http://proxy.example:8080",
                "https": "http://proxy.example:8080",
            },
            system_bypass=["<local>;corp.internal"],
        )

        self.assertEqual(environ["HTTP_PROXY"], "http://proxy.example:8080")
        self.assertEqual(environ["http_proxy"], "http://proxy.example:8080")
        self.assertEqual(environ["HTTPS_PROXY"], "http://proxy.example:8080")
        self.assertIn("corp.internal", environ["NO_PROXY"])
        self.assertIn("localhost", environ["NO_PROXY"])

    def test_policy_does_not_replace_explicit_empty_proxy_or_wildcard(self):
        environ = {"HTTP_PROXY": "", "NO_PROXY": "*"}

        merged = apply_loopback_proxy_bypass(
            environ,
            system_proxies={"http": "http://system-proxy.example:8080"},
            system_bypass=[],
        )

        self.assertEqual(environ["HTTP_PROXY"], "")
        self.assertNotIn("http_proxy", environ)
        self.assertEqual(merged, "*")
        self.assertEqual(environ["no_proxy"], "*")

    def test_ipv6_loopback_builds_valid_httpx_bypass_pattern(self):
        import httpx._utils

        proxy_url = "http://proxy.example:8080"
        with patch.dict(
            os.environ,
            {"HTTP_PROXY": proxy_url, "HTTPS_PROXY": proxy_url},
            clear=True,
        ):
            apply_loopback_proxy_bypass(system_proxies={}, system_bypass=[])
            mounts = httpx._utils.get_environment_proxies()

            self.assertIsNone(mounts["all://[::1]"])

    def test_user_process_overlay_keeps_custom_env_and_only_parent_proxy_keys(self):
        parent_environment = {
            "HTTP_PROXY": "http://parent-proxy.example:8080",
            "HTTPS_PROXY": "http://parent-proxy.example:8080",
            "NO_PROXY": "corp.example",
            "UNRELATED_SECRET": "must-not-leak",
        }
        with patch.dict(os.environ, parent_environment, clear=True):
            overlay = build_proxy_environment_overlay(
                {
                    "CUSTOM_VALUE": "kept",
                    "HTTPS_PROXY": "http://custom-proxy.example:8443",
                    "NO_PROXY": "mcp.internal",
                }
            )

        self.assertEqual(overlay["CUSTOM_VALUE"], "kept")
        self.assertEqual(overlay["HTTP_PROXY"], "http://parent-proxy.example:8080")
        self.assertEqual(overlay["HTTPS_PROXY"], "http://custom-proxy.example:8443")
        self.assertNotIn("UNRELATED_SECRET", overlay)
        self.assertIn("corp.example", overlay["NO_PROXY"])
        self.assertIn("mcp.internal", overlay["NO_PROXY"])
        self.assertIn("::1", overlay["NO_PROXY"])

    def test_httpx_bypasses_loopback_but_keeps_proxy_for_external_urls(self):
        import httpx

        origin, origin_thread, origin_requests = self._start_http_server("origin")
        proxy, proxy_thread, proxy_requests = self._start_http_server("proxy")
        origin_port = origin.server_address[1]
        proxy_port = proxy.server_address[1]

        # Local DNS setups (fake-IP proxies) can break "localhost" and
        # "example.invalid" resolution; pin both so only proxy routing decides.
        real_getaddrinfo = socket.getaddrinfo

        def pinned_getaddrinfo(host, *args, **kwargs):
            if host == "localhost":
                return real_getaddrinfo("127.0.0.1", *args, **kwargs)
            if host == "example.invalid":
                raise socket.gaierror(11003, "pinned: routed through the proxy")
            return real_getaddrinfo(host, *args, **kwargs)

        try:
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            with (
                patch.dict(
                    os.environ,
                    {
                        "HTTP_PROXY": proxy_url,
                        "HTTPS_PROXY": proxy_url,
                        "ALL_PROXY": proxy_url,
                    },
                    clear=True,
                ),
                patch("socket.getaddrinfo", pinned_getaddrinfo),
            ):
                apply_loopback_proxy_bypass(system_proxies={}, system_bypass=[])
                with httpx.Client(timeout=3) as client:
                    by_ip = client.get(f"http://127.0.0.1:{origin_port}/by-ip")
                    by_name = client.get(f"http://localhost:{origin_port}/by-name")
                    external = client.get("http://example.invalid/through-proxy")

                with urlopen_direct(f"http://127.0.0.1:{origin_port}/urllib", timeout=3) as response:
                    direct_body = response.read().decode("utf-8")

                with urlopen_with_loopback_bypass(
                    f"http://localhost:{origin_port}/urllib-policy",
                    timeout=3,
                ) as response:
                    policy_local_body = response.read().decode("utf-8")

                with urlopen_with_loopback_bypass(
                    "http://example.invalid/urllib-external",
                    timeout=3,
                ) as response:
                    policy_external_body = response.read().decode("utf-8")

            self.assertEqual(by_ip.text, "origin")
            self.assertEqual(by_name.text, "origin")
            self.assertEqual(external.text, "proxy")
            self.assertEqual(direct_body, "origin")
            self.assertEqual(policy_local_body, "origin")
            self.assertEqual(policy_external_body, "proxy")
            self.assertEqual(
                origin_requests,
                ["/by-ip", "/by-name", "/urllib", "/urllib-policy"],
            )
            self.assertEqual(
                proxy_requests,
                [
                    "http://example.invalid/through-proxy",
                    "http://example.invalid/urllib-external",
                ],
            )
        finally:
            self._stop_http_server(origin, origin_thread)
            self._stop_http_server(proxy, proxy_thread)
