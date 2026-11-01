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

    # Test model info includes active LM Studio preset defaults.
    @patch("Apps.API.views.llm_api.get_model_settings")
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify model info includes active lms preset defaults.
    def test_model_info_includes_active_lms_preset_defaults(self, mock_preset_settings, mock_model_settings):
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

        response = self.client.get(
            reverse("model_info_api"),
            {"engine": "lms", "model": "qwen3"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("lms_presets", payload)
        self.assertEqual(payload["defaults"]["temperature"], 0.2)
        self.assertFalse(payload["defaults"]["think"])
        self.assertEqual(payload["lms_presets"]["active_preset_id"], str(custom_preset.id))
        self.assertEqual(default_preset.name, "Default")

    # Test sync endpoint clones default LM Studio preset on first change.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify sync endpoint clones default lms preset on first change.
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

    # Test get LM Studio presets endpoint requires a model.
    def test_get_lms_presets_requires_model(self):
        response = self.client.get(reverse("lms_presets_api"))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Model parameter is required")

    # Test create rename delete endpoints manage a custom LM Studio preset.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify create rename delete endpoints manage custom lms preset.
    def test_create_rename_delete_endpoints_manage_custom_lms_preset(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }

        created = self.client.post(
            reverse("create_lms_preset_api"),
            data='{"model":"qwen3","name":"Research","config":{"operation":{"temperature":0.2}}}',
            content_type="application/json",
        )
        self.assertEqual(created.status_code, 200)
        active_preset_id = created.json()["active_preset_id"]

        renamed = self.client.post(
            reverse("rename_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{active_preset_id}","name":"Research v2"}}',
            content_type="application/json",
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(LmsPreset.objects.get(id=active_preset_id).name, "Research v2")

        deleted = self.client.post(
            reverse("delete_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{active_preset_id}"}}',
            content_type="application/json",
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(LmsPreset.objects.filter(model_name="qwen3").count(), 1)
        self.assertTrue(LmsPreset.objects.get(model_name="qwen3").is_default)

    # Test duplicate LM Studio preset names return validation errors.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify duplicate lms preset name returns validation error.
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

    # Test select endpoint activates an existing custom LM Studio preset.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify select endpoint activates custom lms preset.
    def test_select_endpoint_activates_custom_lms_preset(self, mock_get_model_settings):
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

        response = self.client.post(
            reverse("select_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{custom_preset.id}"}}',
            content_type="application/json",
        )

        default_preset.refresh_from_db()
        custom_preset.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_preset_id"], str(custom_preset.id))
        self.assertFalse(default_preset.is_active)
        self.assertTrue(custom_preset.is_active)

    # Test default LM Studio preset mutation errors are returned as validation responses.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify default lms preset mutation errors return 400.
    def test_default_lms_preset_mutation_errors_return_400(self, mock_get_model_settings):
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

        self.assertEqual(renamed.status_code, 400)
        self.assertEqual(deleted.status_code, 400)
        self.assertIn("default preset", renamed.json()["error"])
        self.assertIn("default preset", deleted.json()["error"])
