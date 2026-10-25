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

    # Test supported engines only includes enabled engine flags.
    def test_supported_engines_only_includes_enabled_flags(self):
        # Assertion.
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

    # Test disabled active engine falls back to the first enabled engine.
    def test_active_engine_falls_back_when_configured_engine_is_disabled(self):
        # Assertion.
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

# Verify enabled-engine runtime synchronization.
class LlmApiRuntimeSyncTests(SimpleTestCase):
    @patch("API.llm_api.cleanup_runtime")
    @patch("API.llm_api.prepare_runtime")
    @patch("Settings.settings.get_enabled_engine_ids", return_value=["ollama-service", "lms"])
    def test_sync_prepares_enabled_and_cleans_up_disabled(self, _mock_enabled, mock_prepare, mock_cleanup):
        llm_api.sync_enabled_engine_runtimes()

        mock_prepare.assert_any_call("ollama-service")
        mock_prepare.assert_any_call("lms")
        mock_cleanup.assert_any_call("openai")
        mock_cleanup.assert_any_call("google-genai")

    @patch("API.llm_api.sync_enabled_engine_runtimes")
    def test_handle_engine_transition_calls_sync(self, mock_sync):
        llm_api.handle_engine_transition("ollama-service", "lms")

        mock_sync.assert_called_once()

# Verify request-level engine resolution.
class RequestEngineResolutionTests(SimpleTestCase):
    def _build_request(self, query=None):
        request = Mock()
        request.GET = query or {}
        return request

    @patch("Settings.settings.get_llm_engine", return_value="ollama-service")
    def test_resolve_defaults_to_active_engine(self, _mock_active):
        engine = _resolve_request_engine(self._build_request())
        self.assertEqual(engine, "ollama-service")

    @patch("Settings.settings.is_engine_enabled", return_value=True)
    def test_resolve_query_engine_when_enabled(self, _mock_enabled):
        engine = _resolve_request_engine(self._build_request({"engine": "openai"}))
        self.assertEqual(engine, "openai")

    @patch("Settings.settings.is_engine_enabled", return_value=False)
    def test_resolve_rejects_disabled_engine(self, _mock_enabled):
        with self.assertRaises(RequestEngineResolutionError):
            _resolve_request_engine(self._build_request({"engine": "openai"}))

    @patch("Settings.settings.is_engine_enabled", return_value=True)
    def test_body_engine_takes_priority_over_query(self, _mock_enabled):
        engine = _resolve_request_engine(
            self._build_request({"engine": "lms"}),
            {"engine": "openai"},
        )
        self.assertEqual(engine, "openai")

# Verify disabled explicit engines are rejected at the HTTP layer.
class DisabledEngineApiTests(TestCase):
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

# Verify runtime settings and dynamic model selection endpoints.
class RuntimeSettingsApiTests(TestCase):
    RUNTIME_SETTINGS_WITH_API_KEY = {
        "llm-engine": "openai",
        "lms_url": "127.0.0.1:1234",
        "openai_url": "openrouter.ai/api/v1",
        "has_openai_api_key": True,
        "engine_urls": {"openai": "https://openrouter.ai/api/v1"},
    }

    def setUp(self):
        super().setUp()
        self._engine_enabled_patch = patch(
            "Apps.API.views.settings.is_engine_enabled",
            return_value=True,
        )
        self._engine_enabled_patch.start()

    def tearDown(self):
        self._engine_enabled_patch.stop()
        super().tearDown()

    # Run runtime settings API tests against a temporary settings file.
    @contextmanager
    # Isolated settings payload.
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

    # Test get runtime settings payload.
    def test_get_runtime_settings_payload(self):
        response = self.client.get(reverse("runtime_settings_api"))

        self.assertEqual(response.status_code, 200)
        self.assertNotIn("models", response.json())

    # Test runtime settings rejects invalid JSON.
    def test_runtime_settings_rejects_invalid_json(self):
        response = self.client.post(
            reverse("runtime_settings_api"),
            data="{",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Invalid JSON format")

    # Test post runtime settings updates engine.
    @patch("Apps.UI.views.llm_api.handle_engine_transition")
    # Verify post runtime settings updates engine.
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

    # Test disabled engine selection falls back to an enabled engine.
    @patch("Apps.UI.views.llm_api.handle_engine_transition")
    # Verify post runtime settings ignores disabled engine.
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

    # Test models API returns engine specific models.
    @patch("Apps.API.views._load_models_for_engine", return_value=["llama3"])
    # Verify models api returns engine specific models.
    def test_models_api_returns_engine_specific_models(self, mock_models):
        response = self.client.get(reverse("models_api"), {"engine": "lms"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"engine": "lms", "models": ["llama3"]})
        mock_models.assert_called_once_with("lms")

    # Test model info API requires a model query parameter.
    def test_model_info_api_requires_model_parameter(self):
        response = self.client.get(reverse("model_info_api"), {"engine": "lms"})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Model parameter is required")

    # Test model info API maps unsupported engines to 501.
    @patch("Apps.API.views._build_model_info_payload", side_effect=NotImplementedError("Not supported"))
    # Verify model info api returns 501 for unimplemented engines.
    def test_model_info_api_returns_501_for_unimplemented_engines(self, mock_build_payload):
        response = self.client.get(reverse("model_info_api"), {"engine": "ollama-service", "model": "model"})

        self.assertEqual(response.status_code, 501)
        self.assertEqual(response.json()["error"], "Not supported")
        mock_build_payload.assert_called_once()

    # Test inference info API returns a compact engine-independent payload.
    @patch("Apps.API.views._build_model_info_payload")
    # Verify inference info api returns unified payload.
    def test_inference_info_api_returns_unified_payload(self, mock_build_payload):
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

        response = self.client.get(
            reverse("inference_info_api"),
            {"engine": "ollama-service", "model": "llama3"},
        )

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

    # Test inference info can use the latest model selected through model info.
    @patch("Apps.API.views._build_model_info_payload")
    # Verify inference info api uses runtime selected model.
    def test_inference_info_api_uses_runtime_selected_model(self, mock_build_payload):
        mock_build_payload.return_value = {
            "context_length": 32768,
            "defaults": {"max_completion_tokens": 2048},
            "supports_tool_calling": False,
        }

        selected = self.client.get(reverse("model_info_api"), {"engine": "openai", "model": "gpt-test"})
        self.assertEqual(selected.status_code, 200)

        response = self.client.get(reverse("inference_info_api"), {"engine": "openai"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["engine"], "openai")
        self.assertEqual(payload["model"], "gpt-test")
        self.assertEqual(payload["context_window"], 32768)
        self.assertEqual(payload["max_output_tokens"], 2048)
        self.assertEqual(payload["source"]["model"], "runtime_selection")

    # Test runtime settings payload does not expose API key.
    @patch("Apps.UI.views.settings.get_supported_engines", return_value=[])
    @patch("Apps.UI.views.settings.get_runtime_engine_settings", return_value=RUNTIME_SETTINGS_WITH_API_KEY)
    # Verify runtime settings payload does not expose api key.
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
