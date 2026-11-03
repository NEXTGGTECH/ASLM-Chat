# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from unittest.mock import patch
from django.test import TestCase
from Apps.API.views import _build_model_info_payload, _clear_model_metadata_caches


pytestmark = [pytest.mark.component]


# Cover the model info payload cache.
class ModelInfoCacheTests(TestCase):
    # Clear the model metadata caches before each test.
    def setUp(self):
        super().setUp()
        _clear_model_metadata_caches()

    # Clear the model metadata caches again so no cached payload leaks into other tests.
    def tearDown(self):
        _clear_model_metadata_caches()
        super().tearDown()

    # Verify that a cached model info payload is returned as a detached copy: mutating the first result does
    # not change the second one, and the engine settings are loaded only once.
    @patch("Apps.API.views.llm_api.get_model_settings")
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
