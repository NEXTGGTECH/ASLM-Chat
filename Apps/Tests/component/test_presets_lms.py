# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from unittest.mock import patch
from django.test import TestCase
from django.urls import reverse
from Apps.Data.models import LmsPreset


pytestmark = [pytest.mark.component]


# Cover LM Studio preset API endpoints and model-info integration.
class LmsPresetApiTests(TestCase):
    # Treat every engine as enabled, so the API views accept LM Studio requests.
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

    # Verify that model info returns the LM Studio presets and applies the active custom preset's
    # values over the model defaults.
    @patch("Apps.API.views.llm_api.get_model_settings")
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_model_info_includes_active_lms_preset_defaults(self, mock_preset_settings, mock_model_settings):
        # Arrange: model settings with thinking enabled, a default preset and an active custom
        # preset that lowers the temperature and turns thinking off.
        base_settings = {
            "context_length": 65536,
            "defaults": {"temperature": 0.7, "think": True},
            "supports_thinking": True,
            "supports_think_level": False,
            "think_param_name": "think",
            "think_level_param_name": "reasoning_effort",
            "supports_vision": True,
            "supports_tool_calling": True,
            "supports_files": True,
            "runtime_limits": {"gpu_devices": [{"id": 0, "name": "RTX"}], "gpu_count": 1, "main_gpu_max": 0},
            "custom_fields": [],
            "capabilities": ["vision", "tools", "thinking", "files"],
        }
        mock_model_settings.return_value = base_settings
        mock_preset_settings.return_value = base_settings

        default_preset = LmsPreset.objects.create(
            model_name="qwen3",
            name="Default",
            config={"operation": {"temperature": 0.7}},
            is_default=True,
            is_active=False,
        )
        custom_preset = LmsPreset.objects.create(
            model_name="qwen3",
            name="Reasoning Off",
            config={"operation": {"temperature": 0.2, "think": False}},
            is_default=False,
            is_active=True,
        )

        # Act
        response = self.client.get(
            reverse("model_info_api"),
            {"engine": "lms", "model": "qwen3"},
        )

        # Assert: the custom preset values win and the preset list is part of the payload.
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("lms_presets", payload)
        self.assertEqual(payload["defaults"]["temperature"], 0.2)
        self.assertFalse(payload["defaults"]["think"])
        self.assertEqual(payload["lms_presets"]["active_preset_id"], str(custom_preset.id))
        self.assertEqual(default_preset.name, "Default")

    # Verify that the first sync of a changed config keeps the default LM Studio preset and saves
    # the change as a new custom preset.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_sync_endpoint_clones_default_lms_preset_on_first_change(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }

        response = self.client.post(
            reverse("sync_lms_preset_api"),
            data='{"model":"qwen3","config":{"operation":{"temperature":0.2,"think":false}}}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["presets"]), 2)
        self.assertEqual(LmsPreset.objects.filter(model_name="qwen3").count(), 2)

    # Verify that the LM Studio presets endpoint rejects a request without a model.
    def test_get_lms_presets_requires_model(self):
        response = self.client.get(reverse("lms_presets_api"))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Model parameter is required")

    # Verify that a custom LM Studio preset can be created, renamed and deleted, and that deleting
    # the active preset leaves only the default one.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_create_rename_delete_endpoints_manage_custom_lms_preset(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }

        # Create a custom preset; it becomes the active one.
        created = self.client.post(
            reverse("create_lms_preset_api"),
            data='{"model":"qwen3","name":"Research","config":{"operation":{"temperature":0.2}}}',
            content_type="application/json",
        )
        self.assertEqual(created.status_code, 200)
        active_preset_id = created.json()["active_preset_id"]

        # Rename it.
        renamed = self.client.post(
            reverse("rename_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{active_preset_id}","name":"Research v2"}}',
            content_type="application/json",
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(LmsPreset.objects.get(id=active_preset_id).name, "Research v2")

        # Delete it; only the default preset remains.
        deleted = self.client.post(
            reverse("delete_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{active_preset_id}"}}',
            content_type="application/json",
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(LmsPreset.objects.filter(model_name="qwen3").count(), 1)
        self.assertTrue(LmsPreset.objects.get(model_name="qwen3").is_default)

    # Verify that creating a second LM Studio preset with an existing name returns a validation error.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_duplicate_lms_preset_name_returns_validation_error(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        self.client.post(
            reverse("create_lms_preset_api"),
            data='{"model":"qwen3","name":"Research","config":{"operation":{"temperature":0.2}}}',
            content_type="application/json",
        )

        duplicate = self.client.post(
            reverse("create_lms_preset_api"),
            data='{"model":"qwen3","name":"Research","config":{"operation":{"temperature":0.4}}}',
            content_type="application/json",
        )

        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("already exists", duplicate.json()["error"])

    # Verify that selecting a custom LM Studio preset activates it and deactivates the default one.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_select_endpoint_activates_custom_lms_preset(self, mock_get_model_settings):
        # Arrange: the default preset is active and a custom preset is not.
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        default_preset = LmsPreset.objects.create(
            model_name="qwen3",
            name="Default",
            config={"operation": {"temperature": 0.7}},
            is_default=True,
            is_active=True,
        )
        custom_preset = LmsPreset.objects.create(
            model_name="qwen3",
            name="Research",
            config={"operation": {"temperature": 0.2}},
            is_default=False,
            is_active=False,
        )

        # Act
        response = self.client.post(
            reverse("select_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{custom_preset.id}"}}',
            content_type="application/json",
        )

        # Assert: the active flag moved from the default preset to the custom one.
        default_preset.refresh_from_db()
        custom_preset.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_preset_id"], str(custom_preset.id))
        self.assertFalse(default_preset.is_active)
        self.assertTrue(custom_preset.is_active)

    # Verify that renaming or deleting the default LM Studio preset returns a 400 validation error.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_default_lms_preset_mutation_errors_return_400(self, mock_get_model_settings):
        # Arrange: only the default preset exists.
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        default_preset = LmsPreset.objects.create(
            model_name="qwen3",
            name="Default",
            config={"operation": {"temperature": 0.7}},
            is_default=True,
            is_active=True,
        )

        # Act: try both forbidden mutations.
        renamed = self.client.post(
            reverse("rename_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{default_preset.id}","name":"Renamed"}}',
            content_type="application/json",
        )
        deleted = self.client.post(
            reverse("delete_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{default_preset.id}"}}',
            content_type="application/json",
        )

        # Assert: both are rejected with an error that names the default preset.
        self.assertEqual(renamed.status_code, 400)
        self.assertEqual(deleted.status_code, 400)
        self.assertIn("default preset", renamed.json()["error"])
        self.assertIn("default preset", deleted.json()["error"])
