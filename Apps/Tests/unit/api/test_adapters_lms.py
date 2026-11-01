# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import pytest
from unittest.mock import Mock, patch
from django.test import SimpleTestCase
from API import lms as lms_api
from API.lms import _prepare_native_prediction_options, _prepare_openai_prediction_options, _serialize_model_info, get_model_settings as get_lms_model_settings


pytestmark = [pytest.mark.unit]


# Cover LM Studio metadata normalization and capability fallback.
class LmsAdapterTests(SimpleTestCase):
    # Test native prediction options never forward unsupported raw config.
    def test_native_prediction_options_drop_raw_config(self):
        payload = _prepare_native_prediction_options(
            {"raw": {"fields": [{"key": "custom", "value": True}]}, "maxTokens": 32},
            think=False,
            think_param_name="ext.virtualModel.customField.reasoning",
        )

        self.assertNotIn("raw", payload)
        self.assertEqual(payload["maxTokens"], 32)

    # Test serialize model info reads nested info wrapper.
    def test_serialize_model_info_reads_nested_info_wrapper(self):
        # Define info.
        class Info:
            model_key = "qwen3"
            display_name = "Qwen 3"
            vision = True
            trained_for_tool_use = True
            max_context_length = 65536

        # Define wrapper.
        class Wrapper:
            info = Info()

        payload = _serialize_model_info(Wrapper())

        self.assertEqual(payload["modelKey"], "qwen3")
        self.assertTrue(payload["vision"])
        self.assertTrue(payload["trainedForToolUse"])
        self.assertEqual(payload["maxContextLength"], 65536)

    # Test get model settings uses loaded model info when direct lookup fails.
    @patch("API.lms._get_catalog_model_info", return_value={})
    @patch("API.lms._close_client")
    @patch("API.lms._get_client")
    # Verify get model settings uses loaded model info when direct lookup fails.
    def test_get_model_settings_uses_loaded_model_info_when_direct_lookup_fails(
        self,
        mock_get_client,
        _mock_close_client,
        _mock_catalog,
    ):
        # Define loaded info.
        class LoadedInfo:
            model_key = "qwen3"
            vision = True
            trained_for_tool_use = True
            max_context_length = 65536

        # Define loaded model.
        class LoadedModel:
            info = LoadedInfo()

        client = Mock()
        client.llm.get_model_info.side_effect = RuntimeError("not loaded")
        client.list_loaded_models.return_value = [LoadedModel()]
        mock_get_client.return_value = (Mock(), client)

        payload = get_lms_model_settings("qwen3")

        self.assertTrue(payload["supports_vision"])
        self.assertTrue(payload["supports_tool_calling"])
        self.assertEqual(payload["context_length"], 65536)

    @patch("API.lms._get_client")
    @patch("API.lms._get_local_gpu_devices", return_value=[])
    @patch("API.lms._get_disk_model_operation_defaults", return_value={})
    @patch("API.lms._get_model_index_record", return_value={})
    @patch("API.lms._get_catalog_model_info")
    def test_unloaded_model_uses_catalog_capabilities_without_sdk_load(
        self, catalog, _index, _defaults, _gpu, client,
    ):
        catalog.return_value = {
            "key": "qwen/model", "loaded_instances": [], "max_context_length": 262144,
            "capabilities": {"vision": True, "trained_for_tool_use": True, "reasoning": {"allowed_options": ["off", "on"]}},
        }
        payload = get_lms_model_settings("qwen/model")
        self.assertTrue(payload["supports_vision"])
        self.assertTrue(payload["supports_tool_calling"])
        self.assertTrue(payload["supports_thinking"])
        self.assertFalse(payload["metadata_fallback"])
        self.assertEqual(payload["capabilities_source"], "lms-catalog")
        self.assertEqual(payload["context_length"], 262144)
        client.assert_not_called()

    @patch("API.lms.urlopen_with_loopback_bypass")
    @patch("API.lms.settings.get_engine_url", return_value="http://localhost:1234/v1")
    def test_catalog_resolves_model_keys_and_loaded_instance_aliases(self, _url, request):
        models = [
            {"type": "embedding", "key": "embedding"},
            {"type": "llm", "key": "owner/model", "loaded_instances": [{"id": "instance"}]},
            {"type": "llm", "key": "other/model", "variants": ["other/model@q4"]},
        ]
        request.return_value.__enter__.return_value.read.return_value = json.dumps({"models": models}).encode()
        self.assertEqual(lms_api._get_catalog_model_info("instance"), models[1])
        self.assertEqual(lms_api._get_catalog_model_info("owner/model"), models[1])
        self.assertEqual(lms_api._get_catalog_model_info("other/model@q4"), models[2])
        self.assertEqual(lms_api._get_catalog_model_info("model"), {})
        self.assertEqual(request.call_args.args[0].full_url, "http://localhost:1234/api/v1/models")

    # Test prepare OpenAI prediction options keeps LM Studio custom values in extra body.
    def test_prepare_openai_prediction_options_keeps_lms_custom_values_in_extra_body(self):
        payload = _prepare_openai_prediction_options(
            {
                "temperature": 0.2,
                "contextOverflowPolicy": "truncateMiddle",
                "draftModel": "qwen/qwen3.5-0.5b",
            },
            think_level="high",
        )

        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertNotIn("contextOverflowPolicy", payload)
        self.assertEqual(payload["extra_body"]["contextOverflowPolicy"], "truncateMiddle")
        self.assertEqual(payload["extra_body"]["draftModel"], "qwen/qwen3.5-0.5b")
        self.assertIn("reasoningParsing", payload["extra_body"])
