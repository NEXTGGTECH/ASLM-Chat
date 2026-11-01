# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from unittest.mock import patch
from django.test import SimpleTestCase
from API import google_genai as google_genai_api
from API import lms as lms_api
from API import mcp as tool_registry
from API import ollama as ollama_api
from API import openai as openai_api
from Tools.system_prompts import get_system_prompt, is_instant_generation
from Apps.API.views import _build_generate_kwargs, _compose_system_prompt


pytestmark = [pytest.mark.unit]


class DefaultSystemPromptTests(SimpleTestCase):
    def test_tool_use_is_mandatory_in_both_modes(self):
        for instant in (False, True):
            prompt = get_system_prompt(instant_mode=instant)
            self.assertIn("using relevant available tools is REQUIRED, not optional", prompt)
            self.assertIn("inspect the supplied tool names", prompt)
            self.assertIn("computation/materials tools", prompt)
            self.assertIn("never invent a calculation", prompt)

    def test_search_instructions_rely_on_the_actual_schema(self):
        prompt = _compose_system_prompt("")
        self.assertIn("Read the supplied web_search schema", prompt)
        self.assertNotIn("Count every whitespace-separated token", prompt)
        self.assertNotIn("detailed internal research plan", prompt)

    def test_capability_prompt_uses_only_live_tools_and_preserves_history(self):
        history = [{"role": "system", "content": "Original"}, {"role": "user", "content": "Calculate"}]
        lookup = {"materials__compute": {"tool": {"description": "Compute material properties.\nDetails"}}}
        result = tool_registry.with_available_tools_prompt(history, lookup)
        self.assertIn("materials__compute: Compute material properties.", result[0]["content"])
        self.assertIn("MUST call the relevant tools", result[0]["content"])
        self.assertNotIn("Details", result[0]["content"])
        self.assertEqual(history[0]["content"], "Original")
        self.assertEqual(tool_registry.available_tools_prompt({}), "")

    def test_parallel_searches_do_not_silently_lose_vertical_routing(self):
        result = tool_registry._merge_parallel_web_search_calls([
            {"name": "web_search", "arguments": {"query": "steel", "vertical": "academic"}},
            {"name": "web_search", "arguments": {"query": "steel", "vertical": "shopping"}},
        ])
        self.assertIn("preflight_error_result", result)
        merged = tool_registry._merge_parallel_web_search_calls([
            {"name": "web_search", "arguments": {"query": "steel density", "vertical": "academic"}},
            {"name": "web_search", "arguments": {"query": "steel modulus", "vertical": "academic"}},
        ])
        self.assertEqual(merged["arguments"]["vertical"], "academic")
        self.assertEqual(merged["arguments"]["query"], ["steel density", "steel modulus"])

    def test_all_adapters_send_live_capabilities_to_the_model(self):
        tool = {"type": "function", "function": {
            "name": "materials__compute", "description": "Compute material properties.",
            "parameters": {"type": "object", "properties": {}},
        }}
        lookup = {"materials__compute": {"server": {"id": "materials"}, "tool": {
            "id": "compute", "description": "Compute material properties.",
        }}}
        history = [{"role": "user", "content": "Calculate steel properties"}]
        cases = ((ollama_api, "_stream_round"), (openai_api, "_stream_openai_round"),
                 (lms_api, "_stream_openai_round"), (google_genai_api, "_stream_google_round"))
        for adapter, stream_name in cases:
            with self.subTest(adapter=adapter.__name__), patch.object(
                tool_registry, "build_ollama_tools", return_value=([tool], lookup)
            ), patch.object(adapter, stream_name, return_value=iter(())) as stream:
                list(adapter._run_tool_loop(None, "model", history, {}, ["materials"], {}))
                args = stream.call_args.args
                if adapter is google_genai_api:
                    context = args[3]["system_instruction"]
                else:
                    context = args[2][0]["content"]
                self.assertIn("materials__compute: Compute material properties.", context)
                self.assertIn("MUST call the relevant tools", context)
        self.assertEqual(history, [{"role": "user", "content": "Calculate steel properties"}])

    def test_instant_prompt_replaces_thinking_search_contract(self):
        prompt = _compose_system_prompt("", instant_mode=True)

        self.assertIn("Instant/no-thinking web lookup rules", prompt)
        self.assertIn("at most two searches and two page reads", prompt)
        self.assertNotIn("`shopping` 4", prompt)

    def test_instant_mode_detection_tracks_reasoning_controls(self):
        self.assertTrue(is_instant_generation(False, None))
        self.assertTrue(is_instant_generation(None, "off"))
        self.assertFalse(is_instant_generation(True, None))
        self.assertFalse(is_instant_generation(None, "medium"))

    def test_tool_context_resets_when_switching_back_to_thinking(self):
        common = {
            "engine": "openai",
            "model_name": "test-model",
            "llm_messages": [],
            "think_value": None,
            "think_level_value": None,
            "clean_options": {},
            "session_id": "chat-1",
            "selected_tool_servers": [{"id": "web_search"}],
        }

        instant = _build_generate_kwargs(**common, instant_mode=True)
        thinking = _build_generate_kwargs(**common, instant_mode=False)

        self.assertTrue(instant["tool_context"]["instant_mode"])
        self.assertNotIn("max_tool_rounds", instant["tool_context"])
        self.assertNotIn("instant_mode", thinking["tool_context"])
        self.assertNotIn("max_tool_rounds", thinking["tool_context"])
