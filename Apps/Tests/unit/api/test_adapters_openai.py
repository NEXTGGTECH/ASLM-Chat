# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import importlib
import pytest
from unittest.mock import Mock, patch
from django.test import SimpleTestCase
from API import openai as openai_api
from API.openai import _build_openai_request_options, generate as generate_openai, get_model_settings as get_openai_model_settings


pytestmark = [pytest.mark.unit]


# Keep literal inline reasoning tags visible while preserving real control blocks.
class ReasoningTextParserBoundaryTests(SimpleTestCase):
    # Adapter modules that each define their own _ReasoningTextParser; every test runs against all of them.
    PARSER_MODULES = ("API.openai", "API.lms", "API.google_genai")

    # Yield the module name and a fresh reasoning parser for every adapter module.
    def _parsers(self):
        for module_name in self.PARSER_MODULES:
            module = importlib.import_module(module_name)
            yield module_name, module._ReasoningTextParser()

    # Verify that a think tag quoted in the middle of a sentence is left in the visible text.
    def test_inline_quoted_reasoning_block_stays_visible(self):
        source = "The token <think>quoted text</think> stays visible."

        for module_name, parser in self._parsers():
            with self.subTest(module=module_name):
                thinking, visible = parser.feed(source)
                self.assertEqual(thinking, "")
                self.assertEqual(visible, source)

    # Verify that an inline think tag split across two stream chunks is still kept visible and produces no
    # thinking text.
    def test_inline_tag_boundary_survives_stream_chunking(self):
        for module_name, parser in self._parsers():
            with self.subTest(module=module_name):
                first_thinking, first_visible = parser.feed("The token <thi")
                second_thinking, second_visible = parser.feed("nk>quoted</think> stays visible.")
                self.assertEqual(first_thinking + second_thinking, "")
                self.assertEqual(
                    first_visible + second_visible,
                    "The token <think>quoted</think> stays visible.",
                )

    # Verify that a think block is parsed into thinking text when it follows a newline or a colon (optionally
    # followed by a quote) and is removed from the visible text.
    def test_reasoning_block_after_clear_separator_is_parsed(self):
        for module_name, parser in self._parsers():
            with self.subTest(module=module_name):
                source = "Preface:\n  <think>plan</think>Answer"
                thinking, visible = parser.feed(source)
                self.assertEqual(thinking, "plan")
                self.assertEqual(visible, "Preface:\n  Answer")

        for prefix in ("Preface: ", 'Preface: "'):
            for module_name, parser in self._parsers():
                with self.subTest(module=module_name, prefix=prefix):
                    thinking, visible = parser.feed(f"{prefix}<think>plan</think>Answer")
                    self.assertEqual(thinking, "plan")
                    self.assertEqual(visible, f"{prefix}Answer")

    # Verify that a think block at the very start of the response is parsed into thinking text.
    def test_reasoning_block_at_response_start_is_parsed(self):
        for module_name, parser in self._parsers():
            with self.subTest(module=module_name):
                thinking, visible = parser.feed("<think>plan</think>Answer")
                self.assertEqual(thinking, "plan")
                self.assertEqual(visible, "Answer")


# Ensure generic runtime options are safely mapped for OpenAI-compatible APIs.
class OpenAiOptionMappingTests(SimpleTestCase):
    # Verify that supported options are mapped to OpenAI names (num_predict to max_tokens, think_level to
    # reasoning_effort) and unsupported ones are kept in extra_body.
    def test_maps_supported_options_and_keeps_custom_values_in_extra_body(self):
        payload = _build_openai_request_options(
            {
                "temperature": 0.7,
                "num_predict": 256,
                "num_ctx": 4096,
                "top_k": 40,
            },
            think_level="high",
        )

        self.assertEqual(payload["temperature"], 0.7)
        self.assertEqual(payload["max_tokens"], 256)
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["extra_body"]["num_ctx"], 4096)
        self.assertEqual(payload["extra_body"]["top_k"], 40)

    # Verify that top_logprobs is dropped unless logprobs output is enabled.
    def test_top_logprobs_requires_logprobs(self):
        disabled_payload = _build_openai_request_options({"top_logprobs": 5})
        enabled_payload = _build_openai_request_options({"logprobs": True, "top_logprobs": 5})

        self.assertNotIn("top_logprobs", disabled_payload)
        self.assertEqual(enabled_payload["top_logprobs"], 5)

    # Verify that the OpenAI client is created with the placeholder API key when no key is configured.
    @patch("openai.OpenAI")
    @patch("API.openai.settings.get_engine_url", return_value="http://127.0.0.1:1234/v1")
    @patch("API.openai.settings.get_openai_api_key", return_value="")
    def test_openai_client_uses_placeholder_api_key_when_not_configured(
        self,
        _mock_api_key,
        _mock_engine_url,
        mock_openai_client,
    ):
        from API.openai import _get_client

        mock_openai_client.return_value = Mock()
        _get_client()

        self.assertEqual(mock_openai_client.call_args.kwargs["api_key"], "not-needed")


# Cover extended OpenAI-compatible capability parsing and reasoning output.
class OpenAiAdapterTests(SimpleTestCase):
    # Verify that OpenAI-compatible messages encode supported audio media as a native input_audio part.
    def test_openai_messages_include_audio_media_part(self):
        payload = openai_api._build_openai_messages(
            [
                {
                    "role": "user",
                    "content": "Transcribe",
                    "media": [
                        {
                            "kind": "audio",
                            "name": "voice.mp3",
                            "mime_type": "audio/mpeg",
                            "data": "ZmFrZQ==",
                        }
                    ],
                }
            ]
        )

        self.assertEqual(payload[0]["content"][1]["type"], "input_audio")
        self.assertEqual(payload[0]["content"][1]["input_audio"]["format"], "mp3")

    # Verify that get_model_settings reads tool, vision and reasoning capabilities, effort options, context length
    # and supported parameters from the listed and retrieved model payloads.
    @patch("API.openai._get_client")
    def test_get_model_settings_reads_openai_capabilities_and_reasoning(self, mock_get_client):
        # Arrange: a model list entry with capabilities and reasoning, and a retrieved model with parameters.
        client = Mock()
        client.models.list.return_value = Mock(
            data=[
                {
                    "id": "gpt-test",
                    "capabilities": {"tools": True, "vision": True},
                    "reasoning": {
                        "enabled": True,
                        "effort": {
                            "default": "high",
                            "options": ["low", "medium", "high"],
                        },
                    },
                    "context_length": 65536,
                }
            ]
        )
        client.models.retrieve.return_value = {
            "id": "gpt-test",
            "supported_parameters": {
                "tools": {"type": "array"},
                "tool_choice": {"type": "string"},
                "reasoning_effort": {"enum": ["low", "medium", "high"]},
            },
            "defaults": {"temperature": 0.2},
        }
        mock_get_client.return_value = client

        # Act: read the model settings.
        payload = get_openai_model_settings("gpt-test")

        # Assert: capabilities, reasoning effort options, defaults and limits come from the two payloads.
        self.assertTrue(payload["supports_tool_calling"])
        self.assertTrue(payload["supports_vision"])
        self.assertTrue(payload["supports_thinking"])
        self.assertTrue(payload["supports_think_toggle"])
        self.assertTrue(payload["supports_think_level"])
        self.assertTrue(payload["supports_files"])
        self.assertEqual(payload["think_level_param_name"], "reasoning_effort")
        self.assertEqual(payload["defaults"]["temperature"], 0.2)
        self.assertEqual(payload["defaults"]["reasoning_effort"], "high")
        self.assertEqual(payload["think_level_options"], ["low", "medium", "high"])
        self.assertEqual(payload["context_length"], 65536)
        self.assertIn("tool_choice", payload["supported_parameters"])

    # Verify that get_model_settings reads direct feature flags, input modalities and a supported_parameters list
    # of plain names; a bare reasoning flag with a reasoning_effort parameter gives a think level but no toggle.
    @patch("API.openai._get_client")
    def test_get_model_settings_reads_direct_feature_flags_and_scalar_supported_parameters(self, mock_get_client):
        # Arrange: a model list entry with plain boolean flags and modalities, and a parameter name list.
        client = Mock()
        client.models.list.return_value = Mock(
            data=[
                {
                    "id": "gpt-test",
                    "vision": True,
                    "tool_calling": True,
                    "reasoning": True,
                    "input_modalities": ["text", "image", "audio", "video"],
                }
            ]
        )
        client.models.retrieve.return_value = {
            "id": "gpt-test",
            "supported_parameters": ["temperature", "tools", "tool_choice", "reasoning_effort"],
        }
        mock_get_client.return_value = client

        # Act: read the model settings.
        payload = get_openai_model_settings("gpt-test")

        # Assert: the flags and modalities are mapped to the capability keys.
        self.assertTrue(payload["supports_tool_calling"])
        self.assertTrue(payload["supports_vision"])
        self.assertTrue(payload["supports_audio_input"])
        self.assertTrue(payload["supports_video_input"])
        self.assertTrue(payload["supports_thinking"])
        self.assertTrue(payload["supports_think_level"])
        self.assertFalse(payload["supports_think_toggle"])
        self.assertIn("reasoning_effort", payload["supported_parameters"])

    # Verify that a streamed response yields reasoning_content as thinking and regular content as visible text.
    @patch("API.openai._get_client")
    def test_generate_stream_parses_reasoning_and_visible_content(self, mock_get_client):
        client = Mock()
        client.chat.completions.create.return_value = [
            {"choices": [{"delta": {"reasoning_content": "Plan first."}}]},
            {"choices": [{"delta": {"content": "Final answer"}}]},
        ]
        mock_get_client.return_value = client

        chunks = list(
            generate_openai(
                "gpt-test",
                [{"role": "user", "content": "Hi"}],
                stream=True,
            )
        )

        self.assertEqual(chunks[0]["message"]["thinking"], "Plan first.")
        self.assertEqual(chunks[0]["message"]["content"], "")
        self.assertEqual(chunks[1]["message"]["content"], "Final answer")

    # Verify that a streamed response with only plain content yields that content unchanged and never adds a
    # thinking field.
    @patch("API.openai._get_client")
    def test_generate_stream_does_not_duplicate_plain_content_into_thinking(self, mock_get_client):
        client = Mock()
        client.chat.completions.create.return_value = [
            {"choices": [{"delta": {"content": "Hello"}}]},
            {"choices": [{"delta": {"content": " world"}}]},
        ]
        mock_get_client.return_value = client

        chunks = list(
            generate_openai(
                "gpt-test",
                [{"role": "user", "content": "Hi"}],
                stream=True,
            )
        )

        self.assertEqual([chunk["message"]["content"] for chunk in chunks], ["Hello", " world"])
        self.assertTrue(all("thinking" not in chunk["message"] for chunk in chunks))

    # Verify that get_model_settings takes capabilities from the companion model payload when the model list has
    # none, without sending a generation request.
    @patch("API.openai._get_companion_model_payload")
    @patch("API.openai._get_client")
    def test_get_model_settings_reads_companion_metadata_without_generation(
        self,
        mock_get_client,
        mock_get_companion_payload,
    ):
        # Arrange: the provider payloads carry no capabilities; only the companion payload describes the model.
        client = Mock()
        client.models.list.return_value = Mock(data=[{"id": "gpt-probe", "object": "model", "owned_by": "org"}])
        client.models.retrieve.return_value = {"id": "gpt-probe", "object": "model", "owned_by": "org"}
        mock_get_companion_payload.return_value = {
            "key": "gpt-probe",
            "type": "llm",
            "max_context_length": 131072,
            "capabilities": {
                "vision": True,
                "trained_for_tool_use": True,
                "reasoning": {
                    "allowed_options": ["off", "on"],
                    "default": "on",
                },
            },
        }
        mock_get_client.return_value = client

        # Act: read the model settings.
        payload = get_openai_model_settings("gpt-probe")

        # Assert: the companion capabilities are used and no generation request was sent.
        self.assertTrue(payload["supports_tool_calling"])
        self.assertTrue(payload["supports_thinking"])
        self.assertTrue(payload["supports_vision"])
        self.assertFalse(payload["supports_think_level"])
        self.assertTrue(payload["supports_think_toggle"])
        self.assertEqual(payload["think_level_options"], [])
        self.assertEqual(payload["defaults"]["think"], True)
        self.assertIn("tool_choice", payload["supported_parameters"])
        self.assertIn("tools", payload["supported_parameters"])
        client.chat.completions.create.assert_not_called()
