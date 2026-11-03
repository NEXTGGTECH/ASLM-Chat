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
    # Treat every engine as enabled, so the API views accept Ollama requests.
    def setUp(self):
        super().setUp()
        self._engine_enabled_patch = patch(
            "Apps.API.views.settings.is_engine_enabled",
            return_value=True,
        )
        self._engine_enabled_patch.start()

    # Remove the engine-enabled patch.
    def tearDown(self):
        self._engine_enabled_patch.stop()
        super().tearDown()

    # Verify that model info returns the Ollama presets and the tool servers available to the engine, and
    # applies the active custom preset's supported values over the model defaults.
    @patch("Apps.API.views.llm_api.get_model_settings")
    def test_model_info_includes_active_ollama_preset_defaults_and_servers(self, mock_get_model_settings):
        # Arrange: a tool server that supports the engine, a default preset and an active custom preset
        # with supported and unsupported options, and model settings that report tool support.
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

        # Act
        response = self.client.get(
            reverse("model_info_api"),
            {"engine": "ollama-service", "model": "llama3"},
        )

        # Assert: the preset values win over the model parameters, the unsupported options are dropped and
        # the tool server is listed.
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

    # Verify that the first sync of a changed config keeps the default Ollama preset and saves the change
    # as a new custom preset.
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

    # Verify that a custom Ollama preset can be created, renamed and deleted, and that deleting it leaves
    # only the default preset.
    def test_create_rename_delete_endpoints_manage_custom_preset(self):
        # Create a custom preset; it becomes the active one.
        created = self.client.post(
            reverse("create_ollama_preset_api"),
            data='{"model":"llama3","name":"Research","config":{"num_ctx":49152}}',
            content_type="application/json",
        )
        self.assertEqual(created.status_code, 200)
        created_payload = created.json()
        active_preset_id = created_payload["active_preset_id"]

        # Rename it.
        renamed = self.client.post(
            reverse("rename_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{active_preset_id}","name":"Research v2"}}',
            content_type="application/json",
        )
        self.assertEqual(renamed.status_code, 200)

        # Delete it; only the default preset remains.
        deleted = self.client.post(
            reverse("delete_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{active_preset_id}"}}',
            content_type="application/json",
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(OllamaPreset.objects.filter(model_name="llama3").count(), 1)

    # Verify that creating a second Ollama preset with an existing name returns a validation error.
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

    # Verify that selecting a custom Ollama preset activates it and deactivates the default one.
    def test_select_endpoint_activates_custom_preset(self):
        # Arrange: the default preset is active and a custom preset is not.
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

        # Act
        response = self.client.post(
            reverse("select_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{custom_preset.id}"}}',
            content_type="application/json",
        )

        # Assert: the active flag moved from the default preset to the custom one.
        default_preset.refresh_from_db()
        custom_preset.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_preset_id"], str(custom_preset.id))
        self.assertFalse(default_preset.is_active)
        self.assertTrue(custom_preset.is_active)

    # Verify that renaming or deleting the default Ollama preset returns a 400 validation error.
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
