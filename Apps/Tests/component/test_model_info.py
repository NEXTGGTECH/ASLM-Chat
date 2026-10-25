# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from unittest.mock import patch
from django.test import TestCase
from Apps.API.views import _build_model_info_payload, _clear_model_metadata_caches


pytestmark = [pytest.mark.component]


class ModelInfoCacheTests(TestCase):
    # Clear metadata caches around each test.
    def setUp(self):
        super().setUp()
        _clear_model_metadata_caches()

    # Restore metadata cache state after the test.
    def tearDown(self):
        _clear_model_metadata_caches()
        super().tearDown()

    # Test cached model info is returned as a defensive copy.
    @patch("Apps.API.views.llm_api.get_model_settings")
    # Verify model info payload cache returns detached copies.
    def test_model_info_payload_cache_returns_detached_copies(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "context_length": 32768,
            "defaults": {"temperature": 0.7},
            "supports_tool_calling": False,
        }

        first_payload = _build_model_info_payload("openai", "gpt-test")
        first_payload["defaults"]["temperature"] = 99
        second_payload = _build_model_info_payload("openai", "gpt-test")

        self.assertEqual(second_payload["defaults"]["temperature"], 0.7)
        self.assertEqual(second_payload["context_length"], 32768)
        mock_get_model_settings.assert_called_once_with("openai", "gpt-test")
