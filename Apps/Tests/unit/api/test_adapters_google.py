# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import pytest
from unittest.mock import Mock, patch
from django.test import SimpleTestCase
from API import google_genai as google_genai_api
from API.google_genai import generate as generate_google_genai, get_model_settings as get_google_genai_model_settings, get_models as get_google_genai_models


pytestmark = [pytest.mark.unit]


# Small structured error helper for Google GenAI adapter tests.
class FakeGoogleError(Exception):
    def __init__(
        self,
        code: int,
        status: str,
        message: str,
        *,
        details: list[dict[str, object]] | None = None,
    ) -> None:
        self.code = code
        self.status = status
        self.message = message
        self.details = {
            "error": {
                "code": code,
                "status": status,
                "message": message,
                "details": details or [],
            }
        }
        super().__init__(f"{code} {status}. {self.details}")

# Cover Google GenAI filtering, capability learning, and thinking fallback.
class GoogleGenAiAdapterTests(SimpleTestCase):
    # Set up the test fixture.
    def setUp(self):
        super().setUp()
        google_genai_api._reset_runtime_caches()

    # Tear down the test fixture.
    def tearDown(self):
        google_genai_api._reset_runtime_caches()
        super().tearDown()

    # Gemini receives supported audio/video as inline_data content parts.
    def test_google_contents_include_media_inline_data(self):
        _system_instruction, contents = google_genai_api._build_google_contents(
            [
                {
                    "role": "user",
                    "content": "Describe",
                    "media": [
                        {
                            "kind": "video",
                            "name": "clip.mp4",
                            "mime_type": "video/mp4",
                            "data": "ZmFrZQ==",
                        }
                    ],
                }
            ]
        )

        self.assertEqual(contents[0]["parts"][1]["inline_data"]["data"], b"fake")
        self.assertEqual(contents[0]["parts"][1]["inline_data"]["mime_type"], "video/mp4")

    # Test Gemini function-call replay preserves thought signatures.
    def test_function_call_history_preserves_thought_signature(self):
        raw_part = {
            "thought_signature": b"signature-bytes",
            "function_call": {
                "name": "web_search",
                "args": {"query": "latest ai news"},
            },
        }

        history_part = google_genai_api._build_google_history_part(raw_part, include_text=False)
        replay_parts = google_genai_api._normalize_google_request_parts([history_part])

        self.assertEqual(history_part["function_call"]["name"], "web_search")
        self.assertIn("thought_signature", history_part)
        self.assertEqual(replay_parts[0]["thought_signature"], b"signature-bytes")
        self.assertEqual(replay_parts[0]["function_call"]["name"], "web_search")
        self.assertEqual(replay_parts[0]["function_call"]["args"], {"query": "latest ai news"})

    # Test fallback function-call reconstruction is skipped for preserved Gemini parts.
    def test_preserved_function_call_parts_avoid_unsigned_duplicate(self):
        preserved_parts = [
            {
                "thought_signature": "c2lnbmF0dXJlLWJ5dGVz",
                "function_call": {
                    "name": "web_search",
                    "args": {"query": "latest ai news"},
                },
            }
        ]

        self.assertTrue(google_genai_api._history_parts_have_function_call(preserved_parts))
        content = google_genai_api._assistant_message_to_content(
            {
                "role": "assistant",
                "content": "",
                "google_parts": preserved_parts,
                "tool_calls": [
                    {
                        "id": "call_1_web_search",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": "latest ai news"}),
                        },
                    }
                ],
            }
        )

        function_call_parts = [
            part for part in content["parts"] if isinstance(part.get("function_call"), dict)
        ]
        self.assertEqual(len(function_call_parts), 1)
        self.assertEqual(function_call_parts[0]["thought_signature"], b"signature-bytes")

    # Test legacy unsigned Gemini tool-call transcript is not replayed.
    def test_unsigned_legacy_function_call_history_is_skipped(self):
        _system_instruction, contents = google_genai_api._build_google_contents(
            [
                {"role": "user", "content": "Search this"},
                {
                    "role": "assistant",
                    "content": "",
                    "google_parts": [
                        {
                            "function_call": {
                                "name": "web_search",
                                "args": {"query": "latest ai news"},
                            }
                        }
                    ],
                    "tool_calls": [
                        {
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": json.dumps({"query": "latest ai news"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "name": "web_search", "content": "{\"result\": \"ok\"}"},
                {"role": "user", "content": "Continue"},
            ]
        )

        self.assertEqual(contents[0], {"role": "user", "parts": [{"text": "Search this"}]})
        self.assertEqual(contents[-1], {"role": "user", "parts": [{"text": "Continue"}]})
        self.assertFalse(
            any(
                isinstance(part.get("function_call"), dict)
                for content in contents
                for part in content.get("parts", [])
            )
        )
        self.assertFalse(
            any(
                isinstance(part.get("function_response"), dict)
                for content in contents
                for part in content.get("parts", [])
            )
        )

    # Test get models filters out non generate content models.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    # Verify get models filters out non generate content models.
    def test_get_models_filters_out_non_generate_content_models(self, mock_get_client, _mock_close_client):
        client = Mock()
        payloads = [
            {"name": "models/gemini-2.5-flash", "supported_actions": ["generateContent"]},
            {"name": "models/veo-3.0-generate-001", "supported_actions": ["generateVideos"]},
        ]
        client.models.list.side_effect = lambda config=None: payloads
        mock_get_client.return_value = client

        models = get_google_genai_models()

        self.assertEqual([entry["model"] for entry in models], ["gemini-2.5-flash"])
        client.models.generate_content.assert_not_called()

    # Test get models hides zero quota models for current key after runtime learning.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    @patch("API.google_genai.settings.get_engine_url", return_value="https://generativelanguage.googleapis.com")
    @patch("API.google_genai.settings.get_google_genai_api_key", return_value="key-a")
    # Verify get models hides zero quota models for current key after runtime learning.
    def test_get_models_hides_zero_quota_models_for_current_key_after_runtime_learning(
        self,
        _mock_api_key,
        _mock_engine_url,
        mock_get_client,
        _mock_close_client,
    ):
        client = Mock()
        payloads = [{"name": "models/gemini-3.1-pro", "supported_actions": ["generateContent"]}]
        client.models.list.side_effect = lambda config=None: payloads
        client.models.generate_content.side_effect = FakeGoogleError(
            429,
            "RESOURCE_EXHAUSTED",
            "Quota exceeded for model gemini-3.1-pro. limit: 0.",
            details=[{"violations": [{"quotaDimensions": {"model": "gemini-3.1-pro"}}]}],
        )
        mock_get_client.return_value = client

        with self.assertRaises(FakeGoogleError):
            list(
                generate_google_genai(
                    "gemini-3.1-pro",
                    [{"role": "user", "content": "Hi"}],
                    stream=False,
                )
            )

        self.assertEqual(get_google_genai_models(), [])
        self.assertEqual(client.models.generate_content.call_count, 1)

    # Test get models keeps temporarily rate limited models visible.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    @patch("API.google_genai.settings.get_engine_url", return_value="https://generativelanguage.googleapis.com")
    @patch("API.google_genai.settings.get_google_genai_api_key", return_value="key-a")
    # Verify get models keeps temporarily rate limited models visible.
    def test_get_models_keeps_temporarily_rate_limited_models_visible(
        self,
        _mock_api_key,
        _mock_engine_url,
        mock_get_client,
        _mock_close_client,
    ):
        client = Mock()
        payloads = [{"name": "models/gemini-2.5-pro", "supported_actions": ["generateContent"]}]
        client.models.list.side_effect = lambda config=None: payloads
        client.models.generate_content.side_effect = FakeGoogleError(
            429,
            "RESOURCE_EXHAUSTED",
            "Quota exceeded for model gemini-2.5-pro. limit: 8. Please retry later.",
            details=[{"violations": [{"quotaDimensions": {"model": "gemini-2.5-pro"}}]}],
        )
        mock_get_client.return_value = client

        with self.assertRaises(FakeGoogleError):
            list(
                generate_google_genai(
                    "gemini-2.5-pro",
                    [{"role": "user", "content": "Hi"}],
                    stream=False,
                )
            )

        models = get_google_genai_models()
        cached_models = get_google_genai_models()

        self.assertEqual([entry["model"] for entry in models], ["gemini-2.5-pro"])
        self.assertEqual([entry["model"] for entry in cached_models], ["gemini-2.5-pro"])
        self.assertEqual(client.models.generate_content.call_count, 1)

    # Test published thinking metadata is used without an inference request.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    def test_get_model_settings_uses_published_thinking_metadata_without_generation(
        self,
        mock_get_client,
        _mock_close_client,
    ):
        client = Mock()
        client.models.get.return_value = {
            "name": "models/gemini-2.5-flash",
            "supported_actions": ["generateContent"],
            "thinking": True,
            "tools": True,
            "output_token_limit": 65536,
        }
        mock_get_client.return_value = client

        payload = get_google_genai_model_settings("gemini-2.5-flash")

        self.assertTrue(payload["supports_thinking"])
        self.assertTrue(payload["supports_think_toggle"])
        self.assertFalse(payload["supports_think_level"])
        self.assertTrue(payload["supports_vision"])
        self.assertTrue(payload["supports_audio_input"])
        self.assertTrue(payload["supports_video_input"])
        self.assertEqual(payload["think_level_options"], [])
        self.assertTrue(payload["defaults"]["include_thoughts"])
        self.assertEqual(payload["defaults"]["max_output_tokens"], 8192)
        self.assertEqual(payload["runtime_limits"]["output_token_limit"], 65536)
        self.assertNotIn("thinking_level", payload["supported_parameters"])
        client.models.generate_content.assert_not_called()

    # Test generate retries without thinking level when model rejects it.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    # Verify generate retries without thinking level when model rejects it.
    def test_generate_retries_without_thinking_level_when_model_rejects_it(
        self,
        mock_get_client,
        _mock_close_client,
    ):
        client = Mock()
        captured_configs: list[dict[str, object]] = []

        # Simulate a retry flow that rejects thinking_level once.
        def generate_content(*, model, contents, config):
            captured_configs.append(config)
            thinking_config = dict(config.get("thinking_config", {}) or {})
            if thinking_config.get("thinking_level") is not None:
                raise FakeGoogleError(
                    400,
                    "INVALID_ARGUMENT",
                    "Thinking level is not supported for this model.",
                )
            return {"candidates": [{"content": {"parts": [{"text": "Final answer"}]}}]}

        client.models.generate_content.side_effect = generate_content
        mock_get_client.return_value = client

        chunks = list(
            generate_google_genai(
                "gemini-2.5-flash",
                [{"role": "user", "content": "Hi"}],
                stream=False,
                think_level="high",
            )
        )

        self.assertTrue(any(chunk.get("message", {}).get("content") == "Final answer" for chunk in chunks))
        self.assertEqual(len(captured_configs), 2)
        self.assertEqual(
            captured_configs[0]["thinking_config"]["thinking_level"],
            "HIGH",
        )
        self.assertNotIn("thinking_level", captured_configs[1]["thinking_config"])
        cached_capabilities = google_genai_api._get_cached_model_capabilities("gemini-2.5-flash")
        self.assertFalse(cached_capabilities["supports_think_level"])

    # Test learned availability is scoped to API key.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    @patch("API.google_genai.settings.get_engine_url", return_value="https://generativelanguage.googleapis.com")
    # Verify learned availability is scoped to api key.
    def test_learned_availability_is_scoped_to_api_key(
        self,
        _mock_engine_url,
        mock_get_client,
        _mock_close_client,
    ):
        client_blocked = Mock()
        client_allowed = Mock()
        payloads = [{"name": "models/gemini-3.1-pro", "supported_actions": ["generateContent"]}]
        client_blocked.models.list.side_effect = lambda config=None: payloads
        client_allowed.models.list.side_effect = lambda config=None: payloads
        client_blocked.models.generate_content.side_effect = FakeGoogleError(
            429,
            "RESOURCE_EXHAUSTED",
            "Quota exceeded for model gemini-3.1-pro. limit: 0.",
            details=[{"violations": [{"quotaDimensions": {"model": "gemini-3.1-pro"}}]}],
        )
        client_allowed.models.generate_content.return_value = {
            "candidates": [{"content": {"parts": [{"text": "OK"}]}}]
        }
        key_state = {"value": "key-a"}

        # Get API key.
        def get_api_key():
            return key_state["value"]

        # Get client for key.
        def get_client_for_key():
            return client_blocked if key_state["value"] == "key-a" else client_allowed

        mock_get_client.side_effect = get_client_for_key

        with patch("API.google_genai.settings.get_google_genai_api_key", side_effect=get_api_key):
            with self.assertRaises(FakeGoogleError):
                list(
                    generate_google_genai(
                        "gemini-3.1-pro",
                        [{"role": "user", "content": "Hi"}],
                        stream=False,
                    )
                )
            self.assertEqual(get_google_genai_models(), [])
            key_state["value"] = "key-b"
            models = get_google_genai_models()

        self.assertEqual([entry["model"] for entry in models], ["gemini-3.1-pro"])
        self.assertEqual(client_blocked.models.generate_content.call_count, 1)
        client_allowed.models.generate_content.assert_not_called()
