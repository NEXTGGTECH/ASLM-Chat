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
    # Store the error code, status, message and a Google-style nested details payload.
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
    # Clear the runtime model caches so learned availability does not leak between tests.
    def setUp(self):
        super().setUp()
        google_genai_api._reset_runtime_caches()

    # Clear the runtime model caches again after the test.
    def tearDown(self):
        google_genai_api._reset_runtime_caches()
        super().tearDown()

    # Verify that supported audio/video media is sent to Gemini as inline_data content parts.
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

    # Verify that replaying a Gemini function-call part keeps its thought signature.
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

    # Verify that fallback function-call reconstruction is skipped when preserved Gemini parts already carry the
    # call, so the content has a single signed function call instead of an unsigned duplicate.
    def test_preserved_function_call_parts_avoid_unsigned_duplicate(self):
        # Arrange: a preserved Gemini part with a signature.
        preserved_parts = [
            {
                "thought_signature": "c2lnbmF0dXJlLWJ5dGVz",
                "function_call": {
                    "name": "web_search",
                    "args": {"query": "latest ai news"},
                },
            }
        ]

        # Act: convert an assistant message that has both the preserved parts and an OpenAI-style tool call.
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

        # Assert: exactly one function call remains and it keeps the decoded signature.
        function_call_parts = [
            part for part in content["parts"] if isinstance(part.get("function_call"), dict)
        ]
        self.assertEqual(len(function_call_parts), 1)
        self.assertEqual(function_call_parts[0]["thought_signature"], b"signature-bytes")

    # Verify that a legacy unsigned Gemini tool-call transcript is not replayed: only the surrounding user
    # messages remain and no function_call or function_response part is sent.
    def test_unsigned_legacy_function_call_history_is_skipped(self):
        # Act: build contents from a history whose assistant function call has no thought signature.
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

        # Assert: the user turns survive and no function call or response part is left.
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

    # Verify that get_models drops models that do not support generateContent and does not probe the remaining
    # model with a generation request.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
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

    # Verify that a model whose 429 error reports a zero quota limit is learned as unavailable and hidden from
    # get_models for the current API key.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    @patch("API.google_genai.settings.get_engine_url", return_value="https://generativelanguage.googleapis.com")
    @patch("API.google_genai.settings.get_google_genai_api_key", return_value="key-a")
    def test_get_models_hides_zero_quota_models_for_current_key_after_runtime_learning(
        self,
        _mock_api_key,
        _mock_engine_url,
        mock_get_client,
        _mock_close_client,
    ):
        # Arrange: a listed model whose generation fails with a zero-limit quota error.
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

        # Act: the failed generation teaches the adapter that the model has no quota.
        with self.assertRaises(FakeGoogleError):
            list(
                generate_google_genai(
                    "gemini-3.1-pro",
                    [{"role": "user", "content": "Hi"}],
                    stream=False,
                )
            )

        # Assert: the model disappears from the list and was probed only once.
        self.assertEqual(get_google_genai_models(), [])
        self.assertEqual(client.models.generate_content.call_count, 1)

    # Verify that a model hitting a temporary (non-zero) rate limit stays visible in get_models, also on the
    # cached second call.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    @patch("API.google_genai.settings.get_engine_url", return_value="https://generativelanguage.googleapis.com")
    @patch("API.google_genai.settings.get_google_genai_api_key", return_value="key-a")
    def test_get_models_keeps_temporarily_rate_limited_models_visible(
        self,
        _mock_api_key,
        _mock_engine_url,
        mock_get_client,
        _mock_close_client,
    ):
        # Arrange: a listed model whose generation fails with a retryable quota error (limit above zero).
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

        # Act: trigger the rate limit, then list models twice.
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

        # Assert: the model stays listed and was probed only once.
        self.assertEqual([entry["model"] for entry in models], ["gemini-2.5-pro"])
        self.assertEqual([entry["model"] for entry in cached_models], ["gemini-2.5-pro"])
        self.assertEqual(client.models.generate_content.call_count, 1)

    # Verify that get_model_settings derives thinking support and limits from the published model metadata
    # without sending an inference request.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    def test_get_model_settings_uses_published_thinking_metadata_without_generation(
        self,
        mock_get_client,
        _mock_close_client,
    ):
        # Arrange: published metadata for a thinking-capable model.
        client = Mock()
        client.models.get.return_value = {
            "name": "models/gemini-2.5-flash",
            "supported_actions": ["generateContent"],
            "thinking": True,
            "tools": True,
            "output_token_limit": 65536,
        }
        mock_get_client.return_value = client

        # Act: read the model settings.
        payload = get_google_genai_model_settings("gemini-2.5-flash")

        # Assert: thinking toggle, media inputs, defaults and limits come from the metadata.
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

    # Verify that generate retries without the thinking level when the model rejects it, and remembers that the
    # model does not support thinking levels.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    def test_generate_retries_without_thinking_level_when_model_rejects_it(
        self,
        mock_get_client,
        _mock_close_client,
    ):
        # Arrange: a client that rejects any request carrying a thinking_level.
        client = Mock()
        captured_configs: list[dict[str, object]] = []

        # Record the request config and reject it while it still carries a thinking_level.
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

        # Act: generate with a requested think level.
        chunks = list(
            generate_google_genai(
                "gemini-2.5-flash",
                [{"role": "user", "content": "Hi"}],
                stream=False,
                think_level="high",
            )
        )

        # Assert: the second request drops the level, the answer arrives and the capability is cached as missing.
        self.assertTrue(any(chunk.get("message", {}).get("content") == "Final answer" for chunk in chunks))
        self.assertEqual(len(captured_configs), 2)
        self.assertEqual(
            captured_configs[0]["thinking_config"]["thinking_level"],
            "HIGH",
        )
        self.assertNotIn("thinking_level", captured_configs[1]["thinking_config"])
        cached_capabilities = google_genai_api._get_cached_model_capabilities("gemini-2.5-flash")
        self.assertFalse(cached_capabilities["supports_think_level"])

    # Verify that availability learned from a zero-quota error applies only to the API key that triggered it:
    # after switching keys the model is listed again.
    @patch("API.google_genai._close_client")
    @patch("API.google_genai._get_client")
    @patch("API.google_genai.settings.get_engine_url", return_value="https://generativelanguage.googleapis.com")
    def test_learned_availability_is_scoped_to_api_key(
        self,
        _mock_engine_url,
        mock_get_client,
        _mock_close_client,
    ):
        # Arrange: one client per key; key-a hits a zero quota, key-b generates normally.
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

        # Return the API key currently selected by the test.
        def get_api_key():
            return key_state["value"]

        # Return the blocked client for key-a and the allowed client for any other key.
        def get_client_for_key():
            return client_blocked if key_state["value"] == "key-a" else client_allowed

        mock_get_client.side_effect = get_client_for_key

        # Act: learn the zero quota under key-a, then switch to key-b and list again.
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

        # Assert: key-b sees the model and neither client was probed again.
        self.assertEqual([entry["model"] for entry in models], ["gemini-3.1-pro"])
        self.assertEqual(client_blocked.models.generate_content.call_count, 1)
        client_allowed.models.generate_content.assert_not_called()
