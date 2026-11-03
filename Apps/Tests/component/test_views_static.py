# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import pytest
from unittest.mock import patch
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from Apps.Tests.support import ToolRegistryTestMixin


pytestmark = [pytest.mark.component]


# Cover the per-process static cache busting of templates and ES modules.
class StaticCacheVersionTests(SimpleTestCase):
    # Verify that the static cache version is a 14-digit timestamp string.
    def test_static_cache_version_format(self):
        from Apps.UI import STATIC_CACHE_VERSION

        self.assertRegex(STATIC_CACHE_VERSION, r"^\d{14}$")

    # Verify that the static template tag appends the cache version as a query parameter to the static URL.
    def test_static_template_tag_appends_cache_bust_query(self):
        from Apps.UI import STATIC_CACHE_VERSION
        from django.template import Context, Template

        rendered = Template(
            "{% load i18n_tags %}{% static 'css/main/main.css' %}"
        ).render(Context({}))
        self.assertEqual(rendered, f"/static/css/main/main.css?v={STATIC_CACHE_VERSION}")

    # Verify that the static import map covers the local ES modules and maps every one of them to its URL
    # with the same cache version.
    def test_static_import_map_versions_the_complete_module_graph(self):
        import json

        from Apps.UI import STATIC_CACHE_VERSION
        from django.template import Context, Template

        rendered = Template(
            "{% load i18n_tags %}{% static_import_map %}"
        ).render(Context({}))
        imports = json.loads(rendered)["imports"]

        self.assertGreater(len(imports), 20)
        self.assertEqual(
            imports["/static/js/ui/deep-research-ui.js"],
            f"/static/js/ui/deep-research-ui.js?v={STATIC_CACHE_VERSION}",
        )
        self.assertEqual(
            imports["/static/js/main/api.js"],
            f"/static/js/main/api.js?v={STATIC_CACHE_VERSION}",
        )
        self.assertTrue(
            all(target == f"{source}?v={STATIC_CACHE_VERSION}" for source, target in imports.items())
        )


# Cover the main page: the configured engine, the local tool servers and the versioned static assets.
class MainViewTests(ToolRegistryTestMixin, TestCase):
    # Verify that the main page context carries the engine, the runtime settings and the local tool servers,
    # and that the page links only versioned local static assets.
    @patch("Apps.API.views._resolve_request_engine", return_value="ollama-service")
    def test_main_view_includes_runtime_settings_and_local_servers(self, _mock_engine):
        # Arrange: one local tool server.
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite", "description": "Time helpers"}
            TOOLS = [{"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}}]
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )

        # Act
        response = self.client.get(reverse("main"))

        # Assert: the context holds the engine and the tool server.
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["models"], [])
        self.assertEqual(response.context["llm_engine"], "ollama-service")
        self.assertIn("runtime_settings", response.context)
        self.assertEqual(
            response.context["available_tool_servers"],
            [{
                "id": "time_suite",
                "name": "Time Suite",
                "description": "Time helpers",
                "tool_count": 1,
                "tools": [{"id": "time_now", "name": "Current Time", "description": ""}],
            }],
        )

        # Assert: the markup has the import map, versioned static URLs without a doubled version, and no CDN
        # or account links.
        self.assertContains(response, 'id="group-load"')
        self.assertContains(response, 'type="importmap"')
        self.assertRegex(
            response.content.decode("utf-8"),
            r'"/static/js/ui/deep-research-ui\.js":"/static/js/ui/deep-research-ui\.js\?v=\d{14}"',
        )
        self.assertRegex(
            response.content.decode("utf-8"),
            r"/static/js/main/main\.js\?v=\d{14}",
        )
        self.assertNotRegex(
            response.content.decode("utf-8"),
            r"/static/[^\"']+\?v=\d{14}\?v=",
        )
        self.assertNotContains(response, "cdn.jsdelivr.net")
        self.assertContains(response, "/static/css/vendor/katex.min.css?v=")
        self.assertContains(response, "/static/js/vendor/katex.min.js?v=")
        self.assertContains(response, "/static/js/vendor/mermaid.min.js?v=")
        self.assertRegex(
            response.content.decode("utf-8"),
            r"/static/img/ui/tools/code\.svg\?v=\d{14}",
        )
        self.assertRegex(
            response.content.decode("utf-8"),
            r"/static/img/ui/copy\.svg\?v=\d{14}",
        )
        self.assertRegex(
            response.content.decode("utf-8"),
            r"/static/img/ui/refresh\.svg\?v=\d{14}",
        )
        self.assertRegex(
            response.content.decode("utf-8"),
            r"/static/img/ui/check\.svg\?v=\d{14}",
        )
        self.assertNotContains(response, "/profile/")
        self.assertNotContains(response, "account-btn")
