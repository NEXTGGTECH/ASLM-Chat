# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from unittest.mock import patch
from django.test import TestCase
from django.urls import reverse
from Apps.Data.models import OllamaPreset
from Apps.Tests.support import ToolRegistryTestMixin


pytestmark = [pytest.mark.component]


# Cover Ollama preset API endpoints and model-info integration.
class OllamaPresetApiTests(ToolRegistryTestMixin, TestCase):
    def setUp(self):
        super().setUp()
        self._engine_enabled_patch = patch(
            "Apps.API.views.settings.is_engine_enabled",
            return_value=True,
        )
        self._engine_enabled_patch.start()

    def tearDown(self):
        self._engine_enabled_patch.stop()
        super().tearDown()

    # Test model info includes active Ollama preset defaults and servers.
    @patch("Apps.API.views.llm_api.get_model_settings")
    # Verify model info includes active ollama preset defaults and servers.
    def test_model_info_includes_active_ollama_preset_defaults_and_servers(self, mock_get_model_settings):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite", "description": "Time helpers"}
            TOOLS = [
                {"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}},
                {"id": "timezone_name", "name": "Timezone Name", "parameters": {"type": "object", "properties": {}}},
            ]
            def supports(engine=None, model_name=None):
                return engine == "ollama-service"
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )
        OllamaPreset.objects.create(
            model_name="llama3",
            name="Default",
            config={"num_ctx": 32768, "num_predict": 8192},
            is_default=True,
            is_active=False,
        )
        custom_preset = OllamaPreset.objects.create(
            model_name="llama3",
            name="Custom",
            config={"num_ctx": 65536, "mirostat": 2, "numa": True},
            is_default=False,
            is_active=True,
        )
        mock_get_model_settings.return_value = {
            "modelinfo": {"general.architecture.context_length": 131072},
            "parameters": "temperature 0.8\nPARAMETER mirostat 2\nnuma true",
            "template": "",
            "capabilities": ["tools"],
        }

        response = self.client.get(
            reverse("model_info_api"),
            {"engine": "ollama-service", "model": "llama3"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("ollama_presets", payload)
        self.assertTrue(payload["supports_tool_calling"])
        self.assertEqual(payload["available_tool_servers"][0]["id"], "time_suite")
        self.assertEqual(payload["available_tool_servers"][0]["tool_count"], 2)
        self.assertEqual(payload["defaults"]["num_ctx"], 65536)
        self.assertEqual(payload["defaults"]["temperature"], 0.8)
        self.assertEqual(payload["ollama_presets"]["active_preset_id"], str(custom_preset.id))
        self.assertNotIn("mirostat", payload["defaults"])
        self.assertNotIn("numa", payload["defaults"])

    # Test sync endpoint clones default preset on first change.
    def test_sync_endpoint_clones_default_preset_on_first_change(self):
        response = self.client.post(
            reverse("sync_ollama_preset_api"),
            data='{"model":"llama3","config":{"num_ctx":65536,"num_predict":4096,"think":true,"think_level":"high"}}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["presets"]), 2)
        self.assertEqual(OllamaPreset.objects.filter(model_name="llama3").count(), 2)

    # Test create rename delete endpoints manage custom preset.
    def test_create_rename_delete_endpoints_manage_custom_preset(self):
        created = self.client.post(
            reverse("create_ollama_preset_api"),
            data='{"model":"llama3","name":"Research","config":{"num_ctx":49152}}',
            content_type="application/json",
        )
        self.assertEqual(created.status_code, 200)
        created_payload = created.json()
        active_preset_id = created_payload["active_preset_id"]

        renamed = self.client.post(
            reverse("rename_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{active_preset_id}","name":"Research v2"}}',
            content_type="application/json",
        )
        self.assertEqual(renamed.status_code, 200)

        deleted = self.client.post(
            reverse("delete_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{active_preset_id}"}}',
            content_type="application/json",
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(OllamaPreset.objects.filter(model_name="llama3").count(), 1)

    # Test duplicate preset name returns validation error.
    def test_duplicate_preset_name_returns_validation_error(self):
        self.client.post(
            reverse("create_ollama_preset_api"),
            data='{"model":"llama3","name":"Research","config":{"num_ctx":49152}}',
            content_type="application/json",
        )

        duplicate = self.client.post(
            reverse("create_ollama_preset_api"),
            data='{"model":"llama3","name":"Research","config":{"num_ctx":32768}}',
            content_type="application/json",
        )

        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("already exists", duplicate.json()["error"])

    # Test select endpoint activates an existing custom preset.
    def test_select_endpoint_activates_custom_preset(self):
        default_preset = OllamaPreset.objects.create(
            model_name="llama3",
            name="Default",
            config={"num_ctx": 32768},
            is_default=True,
            is_active=True,
        )
        custom_preset = OllamaPreset.objects.create(
            model_name="llama3",
            name="Research",
            config={"num_ctx": 65536},
            is_default=False,
            is_active=False,
        )

        response = self.client.post(
            reverse("select_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{custom_preset.id}"}}',
            content_type="application/json",
        )

        default_preset.refresh_from_db()
        custom_preset.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_preset_id"], str(custom_preset.id))
        self.assertFalse(default_preset.is_active)
        self.assertTrue(custom_preset.is_active)

    # Test default preset mutation errors are returned as validation responses.
    def test_default_preset_mutation_errors_return_400(self):
        default_preset = OllamaPreset.objects.create(
            model_name="llama3",
            name="Default",
            config={"num_ctx": 32768},
            is_default=True,
            is_active=True,
        )

        renamed = self.client.post(
            reverse("rename_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{default_preset.id}","name":"Renamed"}}',
            content_type="application/json",
        )
        deleted = self.client.post(
            reverse("delete_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{default_preset.id}"}}',
            content_type="application/json",
        )

        self.assertEqual(renamed.status_code, 400)
        self.assertEqual(deleted.status_code, 400)
        self.assertIn("default preset", renamed.json()["error"])
        self.assertIn("default preset", deleted.json()["error"])
