# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import tempfile
import textwrap
from pathlib import Path
from unittest.mock import patch

from django.test import TestCase

from API import mcp as tool_registry
from Apps.API.views import _clear_model_metadata_caches


# Provide an isolated Tools directory and metadata cache reset around tests.
class ToolRegistryTestMixin:
    # Create an isolated tools directory.
    def setUp(self):
        super().setUp()
        self._tools_dir_context = tempfile.TemporaryDirectory()
        self.tools_dir = Path(self._tools_dir_context.name)
        self.tools_patch = patch.object(tool_registry, "TOOLS_DIR", self.tools_dir)
        self.tools_patch.start()
        tool_registry.reset_cache()
        _clear_model_metadata_caches()

    # Restore the original registry state.
    def tearDown(self):
        tool_registry.reset_cache()
        _clear_model_metadata_caches()
        self.tools_patch.stop()
        self._tools_dir_context.cleanup()
        super().tearDown()

    # Write a temporary MCP server.
    def write_server(self, folder: str, body: str) -> None:
        server_dir = self.tools_dir / folder
        server_dir.mkdir(parents=True, exist_ok=True)
        (server_dir / "mcp-server.py").write_text(
            textwrap.dedent(body).strip() + "\n",
            encoding="utf-8",
        )
        tool_registry.reset_cache()


# Ready-to-use TestCase base combining the mixin with the Django TestCase.
class ToolRegistryTestCase(ToolRegistryTestMixin, TestCase):
    pass
