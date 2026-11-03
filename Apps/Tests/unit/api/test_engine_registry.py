# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import Mock, patch
from django.test import SimpleTestCase
from API import llm_api
from API import mcp as tool_registry
from Apps.API.views import _extract_model_name


pytestmark = [pytest.mark.unit]


# Cover the sandbox model context that is added to the available-tools prompt.
class SandboxModelContextTests(SimpleTestCase):
    # Verify that the context of an enabled sandbox server is requested once per prompt build, shared by all its
    # tools, and requested again on the next build so values stay current.
    def test_enabled_server_context_is_fetched_once_per_generation(self):
        server = {"id": "sandbox", "has_model_context": True, "external": True,
                  "server_file": Path("Tools/mcp-sandbox/mcp-server.py")}
        lookup = {
            "sandbox__bash": {"server": server, "tool": {"description": "Run Bash"}},
            "sandbox__read": {"server": server, "tool": {"description": "Read a file"}},
        }
        with patch.object(tool_registry, "_run_worker", side_effect=["Sandbox CPU=5", "Sandbox CPU=6"]) as worker:
            self.assertIn("Sandbox CPU=5", tool_registry.available_tools_prompt(lookup))
            self.assertEqual(worker.call_count, 1)
            self.assertIn("Sandbox CPU=6", tool_registry.available_tools_prompt(lookup))
            self.assertEqual(worker.call_count, 2)
        self.assertEqual(worker.call_args.args[1], "model_context")

    # Verify that without a sandbox tool no worker is started, the prompt has no sandbox section and an empty
    # tool lookup gives an empty prompt.
    def test_disabled_sandbox_is_not_inspected_or_included(self):
        lookup = {"web__search": {"server": {"id": "web"}, "tool": {"description": "Search"}}}
        with patch.object(tool_registry, "_run_worker") as worker:
            self.assertNotIn("Sandbox environment", tool_registry.available_tools_prompt(lookup))
            self.assertEqual(tool_registry.available_tools_prompt({}), "")
        worker.assert_not_called()

    # Verify that a failing context worker only adds an "environment details unavailable" note to the prompt and
    # never leaks the error text.
    def test_context_failure_does_not_break_generation(self):
        lookup = {"sandbox__bash": {"server": {
            "id": "sandbox", "has_model_context": True, "external": True,
            "server_file": Path("sandbox/mcp-server.py"),
        }, "tool": {"description": "Run Bash"}}}
        with patch.object(tool_registry, "_run_worker", side_effect=RuntimeError("private host details")):
            prompt = tool_registry.available_tools_prompt(lookup)
        self.assertIn("environment details unavailable", prompt)
        self.assertNotIn("private host details", prompt)


# Cover adapter-specific model list formats.
class ModelNameExtractionTests(SimpleTestCase):
    # Verify that a plain string entry is returned as the model name.
    def test_extracts_name_from_string(self):
        self.assertEqual(_extract_model_name("llama3"), "llama3")

    # Verify that the model name is read from the model, name or model_key key of a mapping entry.
    def test_extracts_name_from_mapping(self):
        self.assertEqual(_extract_model_name({"model": "qwen"}), "qwen")
        self.assertEqual(_extract_model_name({"name": "gpt-oss"}), "gpt-oss")
        self.assertEqual(_extract_model_name({"model_key": "mistral-nemo"}), "mistral-nemo")

    # Verify that the id of a mapping entry wins over its friendly name.
    def test_prefers_id_over_friendly_name(self):
        self.assertEqual(
            _extract_model_name({"id": "openai/gpt-oss-20b", "name": "OpenAI: GPT OSS 20B"}),
            "openai/gpt-oss-20b",
        )


# Cover generic engine registry behavior for optional capabilities.
class EngineRegistryTests(SimpleTestCase):
    # Verify that a title request truncates the source text to 2048 characters, limits the context and output
    # tokens, streams, and returns the generated title stripped of whitespace and backticks.
    @patch("API.llm_api.generate")
    def test_generate_chat_title_builds_bounded_ollama_request(self, mock_generate):
        mock_generate.return_value = [{"message": {"content": "  `Useful title`  "}}]

        title = llm_api.generate_chat_title(
            "ollama-service",
            "test-model",
            "x" * 3000,
            {"context_length": 8192},
        )

        self.assertEqual(title, "Useful title")
        request = mock_generate.call_args.kwargs
        self.assertEqual(len(request["messages"][1]["content"]), 2048)
        self.assertEqual(request["options"], {"num_ctx": 4096, "num_predict": 32})
        self.assertTrue(request["stream"])

    # Verify that a first attempt without thinking that only returns thought text is retried with a low thinking
    # level and a larger output limit, and the second answer becomes the title.
    @patch("API.llm_api.generate")
    def test_generate_chat_title_retries_thought_only_response(self, mock_generate):
        mock_generate.side_effect = [
            [{"message": {"thinking": "Draft", "content": ""}}],
            [{"message": {"thinking": "Draft", "content": "Generated title"}}],
        ]

        title = llm_api.generate_chat_title(
            "google-genai",
            "test-model",
            "Describe the problem",
            {
                "supports_thinking": True,
                "supports_think_toggle": True,
            },
        )

        self.assertEqual(title, "Generated title")
        self.assertEqual(mock_generate.call_count, 2)
        first_request = mock_generate.call_args_list[0].kwargs
        second_request = mock_generate.call_args_list[1].kwargs
        self.assertEqual(first_request["options"]["thinking_budget"], 0)
        self.assertEqual(first_request["options"]["max_output_tokens"], 32)
        self.assertEqual(second_request["options"]["thinking_level"], "low")
        self.assertEqual(second_request["options"]["max_output_tokens"], 1024)

    # Verify that reloading a model on an engine without reload support raises NotImplementedError.
    def test_reload_model_raises_for_engines_without_reload_support(self):
        with self.assertRaises(NotImplementedError):
            llm_api.reload_model("openai", "gpt-oss")

    # Verify that listing models prepares the engine runtime first and returns the engine module's list.
    @patch("API.llm_api.prepare_runtime")
    @patch("API.llm_api._get_engine_module")
    def test_get_models_prepares_runtime_before_listing(self, mock_get_engine_module, mock_prepare_runtime):
        mock_module = Mock()
        mock_module.get_models.return_value = ["llama3"]
        mock_get_engine_module.return_value = mock_module

        self.assertEqual(llm_api.get_models("ollama-service"), ["llama3"])
        mock_prepare_runtime.assert_called_once_with("ollama-service")

    # Verify that loading model settings prepares the engine runtime first and returns the engine module's
    # settings.
    @patch("API.llm_api.prepare_runtime")
    @patch("API.llm_api._get_engine_module")
    def test_get_model_settings_prepares_runtime_before_loading_metadata(
        self,
        mock_get_engine_module,
        mock_prepare_runtime,
    ):
        mock_module = Mock()
        mock_module.get_model_settings.return_value = {"model": "llama3"}
        mock_get_engine_module.return_value = mock_module

        self.assertEqual(llm_api.get_model_settings("ollama-service", "llama3"), {"model": "llama3"})
        mock_prepare_runtime.assert_called_once_with("ollama-service")
