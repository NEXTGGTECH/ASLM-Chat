# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import tempfile
import pytest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock, patch
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from API import llm_api
from Settings import settings as project_settings
from Apps.API.views import RequestEngineResolutionError, _resolve_request_engine


pytestmark = [pytest.mark.component]


# Cover settings-driven engine availability.
class EngineAvailabilitySettingsTests(SimpleTestCase):
    # Clear the settings cache between mocked settings snapshots.
    def tearDown(self):
        project_settings._invalidate_settings_cache()

    # Run one assertion block against an isolated settings payload.
    def _with_settings_payload(self, payload, assertion):
        with patch.dict(os.environ, {}, clear=True):
            with patch("Settings.settings._load_settings_from_disk", return_value=payload):
                with patch("Settings.settings._get_settings_mtime_ns", return_value=1):
                    project_settings._invalidate_settings_cache()
                    assertion()

    # Verify that the supported engine list contains only the engines whose settings flag is enabled.
    def test_supported_engines_only_includes_enabled_flags(self):
        # Check the supported engine list against the two enabled engines.
        def assertion():
            self.assertEqual(
                project_settings.get_supported_engines(),
                [
                    {"id": "ollama-service", "label": "Ollama"},
                    {"id": "openai", "label": "OpenAI-Compatible"},
                ],
            )

        self._with_settings_payload(
            {
                "llm-engine": "ollama-service",
                "ollama-service": True,
                "lms": False,
                "openai": True,
                "google-genai": False,
            },
            assertion,
        )

    # Verify that a configured but disabled active engine falls back to the first enabled engine.
    def test_active_engine_falls_back_when_configured_engine_is_disabled(self):
        # Check the active engine against the only enabled engine.
        def assertion():
            self.assertEqual(project_settings.get_llm_engine(), "ollama-service")

        self._with_settings_payload(
            {
                "llm-engine": "openai",
                "ollama-service": True,
                "lms": False,
                "openai": False,
                "google-genai": False,
            },
            assertion,
        )


# Cover the synchronization of engine runtimes with the enabled engines.
class LlmApiRuntimeSyncTests(SimpleTestCase):
    # Verify that syncing prepares the runtime of every enabled engine and cleans up the disabled ones.
    @patch("API.llm_api.cleanup_runtime")
    @patch("API.llm_api.prepare_runtime")
    @patch("Settings.settings.get_enabled_engine_ids", return_value=["ollama-service", "lms"])
    def test_sync_prepares_enabled_and_cleans_up_disabled(self, _mock_enabled, mock_prepare, mock_cleanup):
        llm_api.sync_enabled_engine_runtimes()

        mock_prepare.assert_any_call("ollama-service")
        mock_prepare.assert_any_call("lms")
        mock_cleanup.assert_any_call("openai")
        mock_cleanup.assert_any_call("google-genai")

    # Verify that handling an engine transition triggers a runtime sync.
    @patch("API.llm_api.sync_enabled_engine_runtimes")
    def test_handle_engine_transition_calls_sync(self, mock_sync):
        llm_api.handle_engine_transition("ollama-service", "lms")

        mock_sync.assert_called_once()


# Cover the resolution of the engine for one API request.
class RequestEngineResolutionTests(SimpleTestCase):
    # Build a mock request whose GET parameters are the given query dict.
    def _build_request(self, query=None):
        request = Mock()
        request.GET = query or {}
        return request

    # Verify that a request without an engine parameter resolves to the active engine.
    @patch("Settings.settings.get_llm_engine", return_value="ollama-service")
    def test_resolve_defaults_to_active_engine(self, _mock_active):
        engine = _resolve_request_engine(self._build_request())
        self.assertEqual(engine, "ollama-service")

    # Verify that the engine named in the query is used when that engine is enabled.
    @patch("Settings.settings.is_engine_enabled", return_value=True)
    def test_resolve_query_engine_when_enabled(self, _mock_enabled):
        engine = _resolve_request_engine(self._build_request({"engine": "openai"}))
        self.assertEqual(engine, "openai")

    # Verify that an explicitly requested engine that is disabled raises RequestEngineResolutionError.
    @patch("Settings.settings.is_engine_enabled", return_value=False)
    def test_resolve_rejects_disabled_engine(self, _mock_enabled):
        with self.assertRaises(RequestEngineResolutionError):
            _resolve_request_engine(self._build_request({"engine": "openai"}))

    # Verify that the engine in the request body wins over the engine in the query.
    @patch("Settings.settings.is_engine_enabled", return_value=True)
    def test_body_engine_takes_priority_over_query(self, _mock_enabled):
        engine = _resolve_request_engine(
            self._build_request({"engine": "lms"}),
            {"engine": "openai"},
        )
        self.assertEqual(engine, "openai")


# Cover how the HTTP layer treats the explicit engine of a request when engines are enabled or disabled.
class DisabledEngineApiTests(TestCase):
    # Save the payload to a temporary settings file with every related config file redirected into the same
    # temporary directory, then yield and invalidate the settings cache afterwards.
    @contextmanager
    def isolated_settings_payload(self, payload):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "settings.json"

            # Generated tool configs must land in the same temporary
            # directory, otherwise save_settings writes into the repository.
            with (
                patch("Settings.settings.SETTINGS_FILE", settings_file),
                patch("Settings.settings._apply_environment_overrides", side_effect=lambda data: data),
                patch("Settings.settings._sync_module_manifest_setting"),
                patch("Settings.settings.WEB_SEARCH_API_KEYS_FILE", Path(temp_dir) / "web_search_api_keys.json"),
                patch("Settings.settings.WEB_SEARCH_CONFIG_FILE", Path(temp_dir) / "search_config.json"),
                patch("Settings.settings.BROWSER_AGENT_CONFIG_FILE", Path(temp_dir) / "browser_agent.json"),
                patch("Settings.settings.SANDBOX_CONFIG_FILE", Path(temp_dir) / "sandbox.json"),
            ):
                project_settings._invalidate_settings_cache()
                project_settings.save_settings(payload)
                try:
                    yield
                finally:
                    project_settings._invalidate_settings_cache()

    # Verify that the models endpoint rejects a request for a disabled engine with a 400 error.
    def test_models_api_rejects_disabled_engine(self):
        with self.isolated_settings_payload(
            {
                "llm-engine": "ollama-service",
                "ollama-service": True,
                "lms": False,
                "openai": False,
                "google-genai": False,
            }
        ):
            response = self.client.get(reverse("models_api"), {"engine": "lms"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("not enabled", response.json()["error"])

    # Verify that the chat endpoint uses the enabled engine from the query string instead of the active
    # engine and streams the generated text.
    @patch("Apps.API.views.llm_api.prepare_runtime")
    @patch("Apps.API.views.llm_api.generate")
    def test_chat_api_accepts_engine_query_param(self, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Hi there"}}]

        with self.isolated_settings_payload(
            {
                "llm-engine": "ollama-service",
                "ollama-service": True,
                "lms": False,
                "openai": True,
                "google-genai": False,
            }
        ):
            response = self.client.post(
                f"{reverse('chat_api')}?engine=openai",
                data='{"message":"Hello","model":"gpt-test"}',
                content_type="application/json",
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(b"".join(response.streaming_content), b"Hi there")
        self.assertEqual(mock_generate.call_args.kwargs["engine"], "openai")


# Cover the runtime settings, model list, model info and inference info endpoints.
class RuntimeSettingsApiTests(TestCase):
    # Runtime settings that report a saved OpenAI API key, used to check that the key itself is never exposed.
    RUNTIME_SETTINGS_WITH_API_KEY = {
        "llm-engine": "openai",
        "lms_url": "127.0.0.1:1234",
        "openai_url": "openrouter.ai/api/v1",
        "has_openai_api_key": True,
        "engine_urls": {"openai": "https://openrouter.ai/api/v1"},
    }

    # Treat every engine as enabled, so the API views accept any engine in the requests.
    def setUp(self):
        super().setUp()
        self._engine_enabled_patch = patch(
            "Apps.API.views.settings.is_engine_enabled",
            return_value=True,
        )
        self._engine_enabled_patch.start()

    # Remove the engine-enabled patch.
    def tearDown(self):
        self._engine_enabled_patch.stop()
        super().tearDown()

    # Save the payload to a temporary settings file with every related config file redirected into the same
    # temporary directory, then yield and invalidate the settings cache afterwards.
    @contextmanager
    def isolated_settings_payload(self, payload):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings_file = Path(temp_dir) / "settings.json"

            # Generated tool configs must land in the same temporary
            # directory, otherwise save_settings writes into the repository.
            with (
                patch("Settings.settings.SETTINGS_FILE", settings_file),
                patch("Settings.settings._apply_environment_overrides", side_effect=lambda data: data),
                patch("Settings.settings._sync_module_manifest_setting"),
                patch("Settings.settings.WEB_SEARCH_API_KEYS_FILE", Path(temp_dir) / "web_search_api_keys.json"),
                patch("Settings.settings.WEB_SEARCH_CONFIG_FILE", Path(temp_dir) / "search_config.json"),
                patch("Settings.settings.BROWSER_AGENT_CONFIG_FILE", Path(temp_dir) / "browser_agent.json"),
                patch("Settings.settings.SANDBOX_CONFIG_FILE", Path(temp_dir) / "sandbox.json"),
            ):
                project_settings._invalidate_settings_cache()
                project_settings.save_settings(payload)
                try:
                    yield
                finally:
                    project_settings._invalidate_settings_cache()

    # Verify that the runtime settings endpoint returns a 200 payload without the model list.
    def test_get_runtime_settings_payload(self):
        response = self.client.get(reverse("runtime_settings_api"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("models", response.json())

    # Verify that posting malformed JSON to the runtime settings endpoint returns a 400 error.
    def test_runtime_settings_rejects_invalid_json(self):
        response = self.client.post(
            reverse("runtime_settings_api"),
            data="{",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Invalid JSON format")

    # Verify that posting runtime settings switches the active engine, normalizes the saved URL (scheme
    # removed), reports no saved API key and triggers an engine transition to the new engine.
    @patch("Apps.UI.views.llm_api.handle_engine_transition")
    def test_post_runtime_settings_updates_engine(self, mock_transition):
        with self.isolated_settings_payload(
            {
                "llm-engine": "ollama-service",
                "ollama-service": True,
                "lms": False,
                "openai": True,
                "google-genai": False,
                "openai_url": "127.0.0.1:8000/v1",
            }
        ):
            response = self.client.post(
                reverse("runtime_settings_api"),
                data='{"llm-engine":"openai","openai_url":"http://127.0.0.1:1234/v1"}',
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["llm-engine"], "openai")
            self.assertEqual(payload["openai_url"], "127.0.0.1:1234/v1")
            self.assertNotIn("models", payload)
            self.assertFalse(payload["has_openai_api_key"])
            mock_transition.assert_called_once()
            self.assertEqual(mock_transition.call_args.args[1], "openai")

    # Verify that posting a disabled engine as the active one is ignored: the active engine stays on the
    # enabled engine, the engine options list only it, and the transition is from and to that engine.
    @patch("Apps.UI.views.llm_api.handle_engine_transition")
    def test_post_runtime_settings_ignores_disabled_engine(self, mock_transition):
        with self.isolated_settings_payload(
            {
                "llm-engine": "ollama-service",
                "ollama-service": True,
                "lms": False,
                "openai": False,
                "google-genai": False,
            }
        ):
            response = self.client.post(
                reverse("runtime_settings_api"),
                data='{"llm-engine":"openai"}',
                content_type="application/json",
            )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertEqual(payload["llm-engine"], "ollama-service")
            self.assertEqual(payload["engine_options"], [{"id": "ollama-service", "label": "Ollama"}])
            mock_transition.assert_called_once_with("ollama-service", "ollama-service")

    # Verify that the models endpoint returns the models loaded for the requested engine.
    @patch("Apps.API.views._load_models_for_engine", return_value=["llama3"])
    def test_models_api_returns_engine_specific_models(self, mock_models):
        response = self.client.get(reverse("models_api"), {"engine": "lms"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"engine": "lms", "models": ["llama3"]})
        mock_models.assert_called_once_with("lms")

    # Verify that the model info endpoint returns a 400 error when the model parameter is missing.
    def test_model_info_api_requires_model_parameter(self):
        response = self.client.get(reverse("model_info_api"), {"engine": "lms"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Model parameter is required")

    # Verify that the model info endpoint maps a NotImplementedError from the payload builder to a 501 error.
    @patch("Apps.API.views._build_model_info_payload", side_effect=NotImplementedError("Not supported"))
    def test_model_info_api_returns_501_for_unimplemented_engines(self, mock_build_payload):
        response = self.client.get(reverse("model_info_api"), {"engine": "ollama-service", "model": "model"})

        self.assertEqual(response.status_code, 501)
        self.assertEqual(response.json()["error"], "Not supported")
        mock_build_payload.assert_called_once()

    # Verify that the inference info endpoint turns an engine specific model info payload into the compact
    # engine independent payload: context window, output limits, capabilities, tool servers and source.
    @patch("Apps.API.views._build_model_info_payload")
    def test_inference_info_api_returns_unified_payload(self, mock_build_payload):
        # Arrange: the engine specific model info that the endpoint converts.
        mock_build_payload.return_value = {
            "context_length": 131072,
            "defaults": {"num_ctx": 65536, "num_predict": 8192, "temperature": 0.7},
            "supports_thinking": True,
            "supports_think_toggle": True,
            "supports_think_level": True,
            "supports_vision": False,
            "supports_tool_calling": True,
            "supports_files": False,
            "capabilities": ["tools", "thinking"],
            "runtime_limits": {"output_token_limit": 32768},
            "available_tool_servers": [{"id": "time_suite", "name": "Time Suite"}],
        }

        # Act
        response = self.client.get(
            reverse("inference_info_api"),
            {"engine": "ollama-service", "model": "llama3"},
        )

        # Assert: the compact payload maps the context length, defaults, limits and capabilities.
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["engine"], "ollama-service")
        self.assertEqual(payload["engine_label"], "Ollama")
        self.assertEqual(payload["model"], "llama3")
        self.assertEqual(payload["context_window"], 65536)
        self.assertEqual(payload["model_context_limit"], 131072)
        self.assertEqual(payload["max_output_tokens"], 8192)
        self.assertEqual(payload["output_token_limit"], 32768)
        self.assertTrue(payload["capabilities"]["supports_tool_calling"])
        self.assertEqual(payload["tool_servers"][0]["id"], "time_suite")
        self.assertEqual(payload["source"]["model"], "request")
        mock_build_payload.assert_called_once_with("ollama-service", "llama3")

    # Verify that the inference info endpoint falls back to the model last selected through the model info
    # endpoint when the request names no model.
    @patch("Apps.API.views._build_model_info_payload")
    def test_inference_info_api_uses_runtime_selected_model(self, mock_build_payload):
        mock_build_payload.return_value = {
            "context_length": 32768,
            "defaults": {"max_completion_tokens": 2048},
            "supports_tool_calling": False,
        }

        # Select the model through the model info endpoint.
        selected = self.client.get(reverse("model_info_api"), {"engine": "openai", "model": "gpt-test"})
        self.assertEqual(selected.status_code, 200)

        # Ask for the inference info without a model.
        response = self.client.get(reverse("inference_info_api"), {"engine": "openai"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["engine"], "openai")
        self.assertEqual(payload["model"], "gpt-test")
        self.assertEqual(payload["context_window"], 32768)
        self.assertEqual(payload["max_output_tokens"], 2048)
        self.assertEqual(payload["source"]["model"], "runtime_selection")

    # Verify that the runtime settings payload reports whether an API key exists but never includes the key.
    @patch("Apps.UI.views.settings.get_supported_engines", return_value=[])
    @patch("Apps.UI.views.settings.get_runtime_engine_settings", return_value=RUNTIME_SETTINGS_WITH_API_KEY)
    def test_runtime_settings_payload_does_not_expose_api_key(
        self,
        _mock_runtime_settings,
        _mock_engines,
    ):
        response = self.client.get(reverse("runtime_settings_api"))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["has_openai_api_key"])
        self.assertNotIn("openai_api_key", payload)
