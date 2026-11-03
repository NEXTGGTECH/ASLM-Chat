# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from unittest.mock import Mock, patch
from django.test import SimpleTestCase
from API import ollama as ollama_api
from API.ollama import _prepare_chat_kwargs, prepare_runtime as prepare_ollama_runtime
from Apps.API.views import _extract_ollama_model_info


pytestmark = [pytest.mark.unit]


# Ensure Ollama-only thinking parameters are normalized before request dispatch.
class OllamaOptionMappingTests(SimpleTestCase):
    # Verify that a think_level replaces the boolean think flag and is removed from the chat kwargs.
    def test_prepare_chat_kwargs_maps_think_level_into_think(self):
        payload = _prepare_chat_kwargs(
            {
                "stream": True,
                "think": True,
                "think_level": "high",
                "options": {"temperature": 0.7},
            }
        )

        self.assertNotIn("think_level", payload)
        self.assertEqual(payload["think"], "high")
        self.assertEqual(payload["options"]["temperature"], 0.7)

    # Verify that chat kwargs drop the runtime options current Ollama no longer supports (mirostat, numa, tfs_z)
    # and keep the supported ones.
    def test_prepare_chat_kwargs_drops_runtime_options_unsupported_by_current_ollama(self):
        payload = _prepare_chat_kwargs(
            {
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 32768,
                    "mirostat": 2,
                    "numa": False,
                    "tfs_z": 1.0,
                }
            }
        )

        self.assertEqual(payload["options"]["temperature"], 0.7)
        self.assertEqual(payload["options"]["num_ctx"], 32768)
        self.assertNotIn("mirostat", payload["options"])
        self.assertNotIn("numa", payload["options"])
        self.assertNotIn("tfs_z", payload["options"])

    # Verify that chat kwargs drop the keys used only by the LM Studio engine and keep think and options.
    def test_prepare_chat_kwargs_ignores_lms_only_internal_keys(self):
        payload = _prepare_chat_kwargs(
            {
                "stream": True,
                "think": True,
                "think_param_name": "ext.virtualModel.customField.qwen.enableThinking",
                "think_level_param_name": "ext.virtualModel.customField.openai.reasoningEffort",
                "load_config": {"contextLength": 32768},
                "sync_operation_defaults": {"ext.virtualModel.customField.qwen.enableThinking": False},
                "options": {"temperature": 0.7},
            }
        )

        self.assertNotIn("think_param_name", payload)
        self.assertNotIn("think_level_param_name", payload)
        self.assertNotIn("load_config", payload)
        self.assertNotIn("sync_operation_defaults", payload)
        self.assertEqual(payload["think"], True)
        self.assertEqual(payload["options"]["temperature"], 0.7)

    # Verify that preparing the runtime starts the managed Ollama service for the requested engine.
    @patch("API.ollama._get_ollama_service_module")
    def test_prepare_runtime_passes_requested_engine_to_managed_service(self, mock_get_service):
        mock_service = Mock()
        mock_get_service.return_value = mock_service

        prepare_ollama_runtime("ollama-service")

        mock_service.start_ollama.assert_called_once_with(engine="ollama-service")

    # Verify that the local Ollama client connects to the configured loopback port and ignores proxy settings from
    # the environment.
    @patch("API.ollama.ollama.Client")
    @patch("API.ollama.settings.get", return_value=20003)
    def test_ollama_client_disables_environment_proxy(self, _mock_port, mock_client):
        mock_client.return_value = Mock()

        ollama_api.get_client()

        mock_client.assert_called_once_with(
            host="http://127.0.0.1:20003",
            trust_env=False,
        )


# Ensure Ollama tool support follows Ollama model metadata.
class OllamaModelInfoTests(SimpleTestCase):
    # Verify that an explicit capabilities list without tools disables tool support even when the template has
    # tool markers.
    def test_ollama_capabilities_without_tools_disable_tool_support(self):
        payload = _extract_ollama_model_info({
            "capabilities": ["completion"],
            "template": "{{ if .Tools }}tools{{ end }}{{ if .ToolCalls }}calls{{ end }}",
        })

        self.assertFalse(payload["supports_tool_calling"])

    # Verify that the tools capability enables tool support even without tool markers in the template.
    def test_ollama_tools_capability_enables_tool_support(self):
        payload = _extract_ollama_model_info({
            "capabilities": ["completion", "tools"],
            "template": "{{ if .Messages }}{{ end }}",
        })

        self.assertTrue(payload["supports_tool_calling"])

    # Verify that without a capabilities list tool support is inferred from the template: absent without tool
    # markers, present with them.
    def test_ollama_tool_template_fallback_when_capabilities_are_missing(self):
        payload = _extract_ollama_model_info({
            "template": "{{ if .Messages }}{{ end }}",
        })

        self.assertFalse(payload["supports_tool_calling"])

        payload = _extract_ollama_model_info({
            "template": "{{ if .Tools }}tools{{ end }}{{ if .ToolCalls }}calls{{ end }}",
        })

        self.assertTrue(payload["supports_tool_calling"])
