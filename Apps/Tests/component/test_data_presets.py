# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from unittest.mock import patch
from django.test import TestCase
from Apps.Data.lms_presets import activate_lms_preset, create_lms_preset, delete_lms_preset, ensure_lms_preset_state, normalize_lms_preset_config, rename_lms_preset, sync_active_lms_preset
from Apps.Data.models import LmsPreset, OllamaPreset
from Apps.Data.ollama_presets import DEFAULT_OLLAMA_PRESET_CONFIG, activate_ollama_preset, create_ollama_preset, delete_ollama_preset, ensure_ollama_preset_state, normalize_ollama_preset_config, rename_ollama_preset, sync_active_ollama_preset


pytestmark = [pytest.mark.component]


# Ollama preset tests.
# Verify the Ollama preset lifecycle for one model.
class OllamaPresetTests(TestCase):
    """Verify per-model Ollama preset lifecycle helpers."""

    # Create the default preset when no saved state exists.
    def test_ensure_state_creates_default_preset(self):
        presets, active_preset = ensure_ollama_preset_state("llama3")

        self.assertEqual(len(presets), 1)
        self.assertTrue(active_preset.is_default)
        self.assertTrue(active_preset.is_active)
        self.assertEqual(active_preset.config["num_ctx"], DEFAULT_OLLAMA_PRESET_CONFIG["num_ctx"])
        self.assertEqual(active_preset.config["num_predict"], DEFAULT_OLLAMA_PRESET_CONFIG["num_predict"])

    # Promote the default preset when saved state has no active preset.
    def test_ensure_state_promotes_default_when_no_active_preset_exists(self):
        default = OllamaPreset.objects.create(
            model_name="llama3",
            name="Default",
            config=DEFAULT_OLLAMA_PRESET_CONFIG,
            is_default=True,
            is_active=False,
        )
        OllamaPreset.objects.create(
            model_name="llama3",
            name="Custom",
            config={"num_ctx": 49152},
            is_default=False,
            is_active=False,
        )

        presets, active_preset = ensure_ollama_preset_state("llama3")

        self.assertEqual(active_preset.id, default.id)
        self.assertEqual(sum(1 for preset in presets if preset.is_active), 1)

    # Deactivate duplicate active presets to keep runtime state deterministic.
    def test_ensure_state_keeps_only_one_active_preset(self):
        OllamaPreset.objects.create(
            model_name="llama3",
            name="Default",
            config=DEFAULT_OLLAMA_PRESET_CONFIG,
            is_default=True,
            is_active=True,
        )
        duplicate = OllamaPreset.objects.create(
            model_name="llama3",
            name="Coding",
            config={"num_ctx": 49152},
            is_default=False,
            is_active=True,
        )

        presets, active_preset = ensure_ollama_preset_state("llama3")

        duplicate.refresh_from_db()
        self.assertTrue(active_preset.is_default)
        self.assertFalse(duplicate.is_active)
        self.assertEqual(sum(1 for preset in presets if preset.is_active), 1)

    # Clone the default preset when the active config changes.
    def test_sync_from_default_creates_custom_active_preset(self):
        payload = sync_active_ollama_preset(
            "llama3",
            {
                "num_ctx": 65536,
                "num_predict": 4096,
                "think": True,
                "think_level": "high",
            },
        )

        self.assertEqual(OllamaPreset.objects.filter(model_name="llama3").count(), 2)
        active = OllamaPreset.objects.get(id=payload["active_preset_id"])
        self.assertFalse(active.is_default)
        self.assertEqual(active.config["num_ctx"], 65536)
        self.assertEqual(active.config["think_level"], "high")

    # Keep the default preset active when the config has not changed.
    def test_sync_unchanged_default_config_does_not_create_custom_preset(self):
        payload = sync_active_ollama_preset("llama3", DEFAULT_OLLAMA_PRESET_CONFIG)

        active = OllamaPreset.objects.get(id=payload["active_preset_id"])
        self.assertTrue(active.is_default)
        self.assertEqual(OllamaPreset.objects.filter(model_name="llama3").count(), 1)

    # Update custom active presets in place.
    def test_sync_custom_active_preset_updates_in_place(self):
        created = create_ollama_preset(
            "llama3",
            name="Coding",
            config={"num_ctx": 49152, "num_predict": 2048},
            activate=True,
        )

        payload = sync_active_ollama_preset(
            "llama3",
            {"num_ctx": 65536, "num_predict": 4096},
        )

        self.assertEqual(payload["active_preset_id"], created["active_preset_id"])
        self.assertEqual(OllamaPreset.objects.filter(model_name="llama3").count(), 1)
        active = OllamaPreset.objects.get(id=payload["active_preset_id"])
        self.assertEqual(active.config["num_ctx"], 65536)

    # Fall back to the default preset after deleting the active custom one.
    def test_delete_active_custom_preset_falls_back_to_default(self):
        created = create_ollama_preset(
            "llama3",
            name="Coding",
            config={"num_ctx": 49152, "num_predict": 4096},
            activate=True,
        )
        delete_ollama_preset("llama3", created["active_preset_id"])

        active = OllamaPreset.objects.get(model_name="llama3", is_active=True)
        self.assertTrue(active.is_default)

    # Select a custom preset and deactivate the previous active one.
    def test_activate_preset_switches_active_record(self):
        default_payload = sync_active_ollama_preset("llama3", DEFAULT_OLLAMA_PRESET_CONFIG)
        custom = create_ollama_preset(
            "llama3",
            name="Coding",
            config={"num_ctx": 49152},
            activate=False,
        )
        custom_preset = OllamaPreset.objects.get(name="Coding")

        payload = activate_ollama_preset("llama3", str(custom_preset.id))

        self.assertEqual(payload["active_preset_id"], str(custom_preset.id))
        self.assertFalse(OllamaPreset.objects.get(id=default_payload["active_preset_id"]).is_active)
        self.assertTrue(OllamaPreset.objects.get(id=custom_preset.id).is_active)
        self.assertEqual(len(custom["presets"]), 2)

    # Reject unsafe operations against the default preset.
    def test_default_preset_cannot_be_renamed_or_deleted(self):
        presets, default_preset = ensure_ollama_preset_state("llama3")

        with self.assertRaisesMessage(ValueError, "default preset cannot be renamed"):
            rename_ollama_preset("llama3", str(default_preset.id), "Renamed")
        with self.assertRaisesMessage(ValueError, "default preset cannot be deleted"):
            delete_ollama_preset("llama3", str(default_preset.id))
        self.assertEqual(len(presets), 1)

    # Drop unsupported runtime keys before saving a preset.
    def test_sync_drops_unsupported_runtime_keys_from_preset_config(self):
        payload = sync_active_ollama_preset(
            "llama3",
            {
                "num_ctx": 65536,
                "think": True,
                "mirostat": 2,
                "numa": True,
                "vocab_only": True,
            },
        )

        active = OllamaPreset.objects.get(id=payload["active_preset_id"])
        self.assertEqual(active.config["num_ctx"], 65536)
        self.assertTrue(active.config["think"])
        self.assertNotIn("mirostat", active.config)
        self.assertNotIn("numa", active.config)
        self.assertNotIn("vocab_only", active.config)

    # Normalize compact configs and keep only supported runtime keys.
    def test_normalize_config_removes_empty_and_unsupported_values(self):
        payload = normalize_ollama_preset_config(
            {
                "num_ctx": 65536,
                "think": True,
                "stop": ["", "END"],
                "mirostat": 2,
                "nested": {"empty": ""},
            }
        )

        self.assertEqual(payload["num_ctx"], 65536)
        self.assertEqual(payload["stop"], ["END"])
        self.assertTrue(payload["think"])
        self.assertNotIn("mirostat", payload)
        self.assertNotIn("nested", payload)

# LM Studio preset tests.
# Verify the LM Studio preset lifecycle for one model.
class LmsPresetTests(TestCase):
    """Verify per-model LM Studio preset lifecycle helpers."""

    # Create the default preset from model defaults when no state exists.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_ensure_state_creates_default_preset(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7, "think": True},
        }

        presets, active_preset = ensure_lms_preset_state("qwen3")

        self.assertEqual(len(presets), 1)
        self.assertTrue(active_preset.is_default)
        self.assertTrue(active_preset.is_active)
        self.assertEqual(active_preset.config["operation"]["temperature"], 0.7)

    # Promote the default preset when no preset is active.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_ensure_state_promotes_default_when_no_active_preset_exists(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        default = LmsPreset.objects.create(
            model_name="qwen3",
            name="Default",
            config={"operation": {"temperature": 0.7}},
            is_default=True,
            is_active=False,
        )
        LmsPreset.objects.create(
            model_name="qwen3",
            name="Coding",
            config={"operation": {"temperature": 0.2}},
            is_default=False,
            is_active=False,
        )

        presets, active_preset = ensure_lms_preset_state("qwen3")

        self.assertEqual(active_preset.id, default.id)
        self.assertEqual(sum(1 for preset in presets if preset.is_active), 1)

    # Clone the default preset when the active config changes.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_sync_from_default_creates_custom_active_preset(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }

        payload = sync_active_lms_preset(
            "qwen3",
            {
                "operation": {"temperature": 0.2, "think": False},
            },
        )

        self.assertEqual(LmsPreset.objects.filter(model_name="qwen3").count(), 2)
        active = LmsPreset.objects.get(id=payload["active_preset_id"])
        self.assertFalse(active.is_default)
        self.assertFalse(active.config["operation"]["think"])

    # Keep the default preset when the config matches model defaults.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_sync_unchanged_default_config_does_not_create_custom_preset(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }

        payload = sync_active_lms_preset("qwen3", {"operation": {"temperature": 0.7}})

        active = LmsPreset.objects.get(id=payload["active_preset_id"])
        self.assertTrue(active.is_default)
        self.assertEqual(LmsPreset.objects.filter(model_name="qwen3").count(), 1)

    # Update a custom active preset in place.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_sync_custom_active_preset_updates_in_place(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        created = create_lms_preset(
            "qwen3",
            name="Coding",
            config={"operation": {"temperature": 0.2}},
            activate=True,
        )

        payload = sync_active_lms_preset(
            "qwen3",
            {"operation": {"temperature": 0.4, "think": False}},
        )

        self.assertEqual(payload["active_preset_id"], created["active_preset_id"])
        self.assertEqual(LmsPreset.objects.filter(model_name="qwen3").count(), 1)
        active = LmsPreset.objects.get(id=payload["active_preset_id"])
        self.assertEqual(active.config["operation"]["temperature"], 0.4)

    # Fall back to the default preset after deleting the active custom one.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_delete_active_custom_preset_falls_back_to_default(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        created = create_lms_preset(
            "qwen3",
            name="Coding",
            config={"operation": {"temperature": 0.2}},
            activate=True,
        )
        delete_lms_preset("qwen3", created["active_preset_id"])

        active = LmsPreset.objects.get(model_name="qwen3", is_active=True)
        self.assertTrue(active.is_default)

    # Select a custom preset and deactivate the previous active one.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_activate_preset_switches_active_record(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        default_payload = sync_active_lms_preset("qwen3", {"operation": {"temperature": 0.7}})
        custom = create_lms_preset(
            "qwen3",
            name="Coding",
            config={"operation": {"temperature": 0.2}},
            activate=False,
        )
        custom_preset = LmsPreset.objects.get(name="Coding")

        payload = activate_lms_preset("qwen3", str(custom_preset.id))

        self.assertEqual(payload["active_preset_id"], str(custom_preset.id))
        self.assertFalse(LmsPreset.objects.get(id=default_payload["active_preset_id"]).is_active)
        self.assertTrue(LmsPreset.objects.get(id=custom_preset.id).is_active)
        self.assertEqual(len(custom["presets"]), 2)

    # Reject unsafe operations against the default preset.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    def test_default_preset_cannot_be_renamed_or_deleted(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        presets, default_preset = ensure_lms_preset_state("qwen3")

        with self.assertRaisesMessage(ValueError, "default preset cannot be renamed"):
            rename_lms_preset("qwen3", str(default_preset.id), "Renamed")
        with self.assertRaisesMessage(ValueError, "default preset cannot be deleted"):
            delete_lms_preset("qwen3", str(default_preset.id))
        self.assertEqual(len(presets), 1)

    # Normalize legacy top-level options into the operation block.
    def test_normalize_config_moves_top_level_options_into_operation(self):
        payload = normalize_lms_preset_config(
            {
                "temperature": 0.2,
                "stop": ["", "END"],
                "load": {"contextLength": 32768},
            }
        )

        self.assertEqual(payload, {"operation": {"temperature": 0.2, "stop": ["END"]}})
