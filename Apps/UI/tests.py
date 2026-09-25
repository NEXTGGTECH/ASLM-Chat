# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import io
import os
import subprocess
import tempfile
import textwrap
import threading
import zipfile
from contextlib import asynccontextmanager, contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse

from API import google_genai as google_genai_api
from API import llm_api
from API import lms as lms_api
from API import mcp as tool_registry
from API import ollama as ollama_api
from API import openai as openai_api
from Apps.Data import model_capabilities
from Tools.system_prompts import get_system_prompt, is_instant_generation
from API.google_genai import (
    generate as generate_google_genai,
    get_model_settings as get_google_genai_model_settings,
    get_models as get_google_genai_models,
)
from API.lms import (
    _prepare_native_prediction_options,
    _prepare_openai_prediction_options,
    _serialize_model_info,
    get_model_settings as get_lms_model_settings,
)
from API.ollama import _prepare_chat_kwargs, prepare_runtime as prepare_ollama_runtime
from API.openai import (
    _build_openai_request_options,
    generate as generate_openai,
    get_model_settings as get_openai_model_settings,
)
from Services import user_mcp_client
from Settings import settings as project_settings
from Settings.proxy_policy import (
    apply_loopback_proxy_bypass,
    build_proxy_environment_overlay,
    urlopen_direct,
    urlopen_with_loopback_bypass,
)
from Settings.mcp_json import UserMcpServerEntry
from Settings import skills as skills_config
from Apps.Data.models import (
    Chat,
    ChatBranch,
    LmsPreset,
    Message,
    MessageAttachment,
    MessageAttachmentKind,
    MessageImage,
    OllamaPreset,
)
from Apps.UI import upload_storage
from Apps.UI.file_manifests import (
    TEXT_PREVIEW_CHAR_LIMIT,
    build_uploaded_file_manifest,
    normalize_upload_name,
)
from Apps.UI.upload_storage import display_kind_for_upload
from Apps.UI.views import (
    RequestEngineResolutionError,
    _build_activity_segments,
    _build_chat_history,
    _build_chat_title,
    _build_generate_kwargs,
    _build_model_info_payload,
    _build_uploaded_file_context_entry,
    _build_uploaded_file_prompt_block,
    _apply_uploaded_file_manifests_to_llm_entry,
    _clear_model_metadata_caches,
    _chat_is_first_user_turn,
    _compose_system_prompt,
    _extract_attachment_text,
    _extract_uploaded_file_ids_from_message,
    _extract_ollama_model_info,
    _extract_model_name,
    _format_runtime_error,
    _is_active_browser_portal_state,
    _load_model_upload_manifests,
    _normalize_request_attachments,
    _normalize_uploaded_file_ids,
    _parse_active_tool_slugs,
    _resolve_request_engine,
    _reveal_file_in_file_manager,
    _resolve_history_char_budget,
    _serialize_attachment_record,
    _serialize_tool_call_marker,
    _serialize_tool_activity_marker,
    _selected_tools_include_sandbox,
    _stream_chat_response,
    _strip_llm_control_tokens,
)


class ToolActivityStreamTests(SimpleTestCase):
    def test_activity_marker_keeps_structured_realtime_event(self):
        activity = {
            "alias": "deep_research__research",
            "server_id": "deep_research",
            "tool_id": "research",
            "event": {
                "sequence": 7,
                "phase": "research",
                "iteration": 2,
                "type": "model_output_delta",
                "data": {"channel": "reasoning", "content": "Checking sources"},
            },
        }

        marker = _serialize_tool_activity_marker(activity)

        self.assertTrue(marker.startswith("<tool_activity>"))
        self.assertTrue(marker.endswith("</tool_activity>"))
        payload = json.loads(marker.removeprefix("<tool_activity>").removesuffix("</tool_activity>"))
        self.assertEqual(payload, activity)


class DefaultSystemPromptTests(SimpleTestCase):
    def test_thinking_prompt_exactly_preserves_the_legacy_system_prompt(self):
        # SHA-256 of the complete former Tools/SYSTEM_PROMPT.md content.
        digest = hashlib.sha256(get_system_prompt(instant_mode=False).encode("utf-8")).hexdigest()

        self.assertEqual(
            digest,
            "258f06452eef4abcf17b397614f3fe623d6c6e96002585300a208da7af77a49c",
        )

    def test_search_query_word_limits_are_explicit(self):
        prompt = _compose_system_prompt("")

        self.assertIn("`web` 10 words per string", prompt)
        self.assertIn("`shopping` 4", prompt)
        self.assertIn("`academic` 8", prompt)
        self.assertIn("`onion` 7", prompt)
        self.assertIn("Count every whitespace-separated token", prompt)
        self.assertIn("Search operators count toward the corresponding string's limit", prompt)

    def test_instant_prompt_replaces_thinking_search_contract(self):
        prompt = _compose_system_prompt("", instant_mode=True)

        self.assertIn("Instant/no-thinking web lookup rules", prompt)
        self.assertIn("at most twice", prompt)
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


# Keep literal inline reasoning tags visible while preserving real control blocks.
class ReasoningTextParserBoundaryTests(SimpleTestCase):
    PARSER_MODULES = ("API.openai", "API.lms", "API.google_genai")

    def _parsers(self):
        for module_name in self.PARSER_MODULES:
            module = importlib.import_module(module_name)
            yield module_name, module._ReasoningTextParser()

    def test_inline_quoted_reasoning_block_stays_visible(self):
        source = "The token <think>quoted text</think> stays visible."

        for module_name, parser in self._parsers():
            with self.subTest(module=module_name):
                thinking, visible = parser.feed(source)
                self.assertEqual(thinking, "")
                self.assertEqual(visible, source)

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

    def test_reasoning_block_at_response_start_is_parsed(self):
        for module_name, parser in self._parsers():
            with self.subTest(module=module_name):
                thinking, visible = parser.feed("<think>plan</think>Answer")
                self.assertEqual(thinking, "plan")
                self.assertEqual(visible, "Answer")


# Shared test helpers.

# Patch the local tools directory for endpoint tests.
class ToolRegistryTestMixin:
    # Create an isolated tools directory.
    def setUp(self):
        super().setUp()
        self._tools_dir_context = tempfile.TemporaryDirectory()
        self.tools_dir = Path(self._tools_dir_context.name)
        self.tools_patch = patch.object(tool_registry, "TOOLS_DIR", self.tools_dir)
        self.tools_patch.start()
        tool_registry.reset_cache()
        _clear_model_metadata_caches()

    # Restore the original registry state.
    def tearDown(self):
        tool_registry.reset_cache()
        _clear_model_metadata_caches()
        self.tools_patch.stop()
        self._tools_dir_context.cleanup()
        super().tearDown()

    # Write a temporary MCP server.
    def write_server(self, folder: str, body: str) -> None:
        server_dir = self.tools_dir / folder
        server_dir.mkdir(parents=True, exist_ok=True)
        (server_dir / "mcp-server.py").write_text(
            textwrap.dedent(body).strip() + "\n",
            encoding="utf-8",
        )
        tool_registry.reset_cache()


# Persistent user MCP connection tests.
class UserMcpPersistentSessionTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        user_mcp_client.shutdown_all()

    def tearDown(self):
        user_mcp_client.shutdown_all()
        super().tearDown()

    @staticmethod
    def _entry() -> UserMcpServerEntry:
        return UserMcpServerEntry(
            config_key="test",
            server_id="test",
            display_name="Test MCP",
            transport="stdio",
            command="fake-mcp",
            args=[],
            env=None,
            cwd=None,
            url=None,
            headers=None,
        )

    # list_tools and repeated calls must share one initialized transport.
    def test_reuses_connection_until_shutdown(self):
        counters = {"entered": 0, "exited": 0, "calls": 0}

        class FakeSession:
            async def list_tools(self):
                return SimpleNamespace(
                    tools=[
                        SimpleNamespace(
                            name="echo",
                            description="Echo input",
                            inputSchema={"type": "object", "properties": {}},
                        )
                    ]
                )

            async def call_tool(self, name, arguments):
                counters["calls"] += 1
                return SimpleNamespace(
                    isError=False,
                    structuredContent=None,
                    content=[SimpleNamespace(text=f"{name}:{arguments['value']}")],
                )

        @asynccontextmanager
        async def fake_connect(_entry):
            counters["entered"] += 1
            try:
                yield FakeSession()
            finally:
                counters["exited"] += 1

        with patch.object(user_mcp_client, "_connect_session", fake_connect):
            definitions, error = user_mcp_client.fetch_tool_definitions(self._entry())
            first = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {"value": "one"})
            second = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {"value": "two"})

            self.assertIsNone(error)
            self.assertEqual(definitions[0]["mcp_tool_name"], "echo")
            self.assertEqual(first, "echo:one")
            self.assertEqual(second, "echo:two")
            self.assertEqual(counters, {"entered": 1, "exited": 0, "calls": 2})

            user_mcp_client.shutdown_all()
            self.assertEqual(counters["exited"], 1)

    # A failed transport is discarded so the next invocation reconnects.
    def test_reconnects_after_transport_failure(self):
        counters = {"entered": 0}

        class FakeSession:
            async def call_tool(self, _name, _arguments):
                if counters["entered"] == 1:
                    raise ConnectionError("transport stopped")
                return SimpleNamespace(
                    isError=False,
                    structuredContent=None,
                    content=[SimpleNamespace(text="reconnected")],
                )

        @asynccontextmanager
        async def fake_connect(_entry):
            counters["entered"] += 1
            yield FakeSession()

        with patch.object(user_mcp_client, "_connect_session", fake_connect):
            failed = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {})
            recovered = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {})

        self.assertIn("transport stopped", failed)
        self.assertEqual(recovered, "reconnected")
        self.assertEqual(counters["entered"], 2)

    def test_cancel_entry_stops_active_call(self):
        started = threading.Event()
        exited = threading.Event()
        result = {}

        class FakeSession:
            async def call_tool(self, _name, _arguments):
                started.set()
                await asyncio.Event().wait()

        @asynccontextmanager
        async def fake_connect(_entry):
            try:
                yield FakeSession()
            finally:
                exited.set()

        def invoke() -> None:
            result["value"] = user_mcp_client.call_user_mcp_tool(self._entry(), "echo", {})

        with patch.object(user_mcp_client, "_connect_session", fake_connect):
            caller = threading.Thread(target=invoke, daemon=True)
            caller.start()
            self.assertTrue(started.wait(timeout=1))

            user_mcp_client.cancel_entry(self._entry())
            caller.join(timeout=1)

        self.assertFalse(caller.is_alive())
        self.assertTrue(exited.is_set())
        self.assertIn("closed", result.get("value", "").lower())


# MCP reload endpoint tests.
class McpReloadApiTests(SimpleTestCase):
    @patch("Apps.UI.views._list_tool_servers_cached")
    @patch("Apps.UI.views._clear_tool_server_cache")
    @patch.object(tool_registry, "reset_cache")
    def test_reload_restarts_sessions_and_returns_fresh_servers(
        self,
        reset_cache_mock,
        clear_cache_mock,
        list_servers_mock,
    ):
        servers = [{"id": "custom", "name": "Custom", "user_mcp": True, "tools": []}]
        list_servers_mock.return_value = servers

        response = self.client.post(
            reverse("mcp_reload_api"),
            data=json.dumps({"model": "test-model"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["tool_servers"], servers)
        reset_cache_mock.assert_called_once_with()
        clear_cache_mock.assert_called_once_with()
        list_servers_mock.assert_called_once_with(project_settings.get_llm_engine(), "test-model")

    def test_reload_rejects_non_post_requests(self):
        response = self.client.get(reverse("mcp_reload_api"))
        self.assertEqual(response.status_code, 405)


# Skills API tests.

class SkillsApiTests(TestCase):
    # Prepare shared fixtures for each test case.
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.skills_dir = self.root / "Skills"
        self.sandbox_skills_dir = self.root / "Tools" / "mcp-sandbox" / "_sandbox" / "Skills"
        self._patches = [
            patch.object(skills_config, "BASE_DIR", self.root),
            patch.object(skills_config, "SKILLS_DIR", self.skills_dir),
            patch.object(skills_config, "SANDBOX_SKILLS_DIR", self.sandbox_skills_dir),
            patch.object(skills_config, "_PENDING_NOTIFY_PATH", self.root / ".aslm" / "skills-pending-notify.json"),
        ]
        for patcher in self._patches:
            patcher.start()
        self.client = Client()
        skills_config.clear_skill_config_refresh_pending()

    # Clean up fixtures created for each test case.
    def tearDown(self):
        skills_config.clear_skill_config_refresh_pending()
        for patcher in reversed(self._patches):
            patcher.stop()
        self._tmp.cleanup()
        super().tearDown()

    # Verify skills root created and crud validates paths.
    def test_skills_root_created_and_crud_validates_paths(self):
        response = self.client.get(reverse("skills_api"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.skills_dir.is_dir())

        created = self.client.post(
            reverse("skills_api"),
            data=json.dumps({"name": "skill-creator"}),
            content_type="application/json",
        )
        self.assertEqual(created.status_code, 200)
        self.assertTrue((self.skills_dir / "skill-creator" / "SKILL.md").is_file())

        saved = self.client.put(
            reverse("skills_file_api"),
            data=json.dumps({
                "folder": "skill-creator",
                "file": "agents/grader.md",
                "content": "---\nname: skill-creator\ndescription: Make skills\nenabled: true\n---\n\n# Skill\n",
            }),
            content_type="application/json",
        )
        self.assertEqual(saved.status_code, 200)
        self.assertTrue((self.skills_dir / "skill-creator" / "agents" / "grader.md").is_file())

        renamed = self.client.patch(
            reverse("skills_folder_api"),
            data=json.dumps({"old_name": "skill-creator", "new_name": "renamed-skill"}),
            content_type="application/json",
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertTrue((self.skills_dir / "renamed-skill" / "SKILL.md").is_file())
        self.assertEqual(renamed.json()["folders"][0]["title"], "renamed-skill")

        rejected = self.client.put(
            reverse("skills_file_api"),
            data=json.dumps({"folder": "renamed-skill", "file": "../bad.md", "content": "x"}),
            content_type="application/json",
        )
        self.assertEqual(rejected.status_code, 400)

    # Verify front matter summary and prompt inventory.
    def test_front_matter_summary_and_prompt_inventory(self):
        skill_root = self.skills_dir / "skill-creator"
        skill_root.mkdir(parents=True)
        (skill_root / "SKILL.md").write_text(
            "---\n"
            "name: skill-creator\n"
            "description: Create and improve skills.\n"
            "trigger: Slash command + auto\n"
            "added_by: Anthropic\n"
            "enabled: true\n"
            "---\n\n# Skill Creator\n",
            encoding="utf-8",
        )
        agents_dir = skill_root / "agents"
        agents_dir.mkdir()
        (agents_dir / "grader.md").write_text("# Grader\n", encoding="utf-8")

        payload = self.client.get(reverse("skills_api")).json()
        folder = payload["folders"][0]
        self.assertEqual(folder["title"], "skill-creator")
        self.assertEqual(folder["description"], "Create and improve skills.")
        self.assertEqual(folder["source"], "download")
        self.assertIsInstance(folder["created_at"], float)
        self.assertGreater(folder["created_at"], 0)

        prompt = _compose_system_prompt("", include_skills_baseline=True)
        self.assertIn("Your skills:", prompt)
        self.assertIn("/workspace/_sandbox/Skills/skill-creator", prompt)
        self.assertIn("agents/grader.md", prompt)

        follow_up_prompt = _compose_system_prompt("")
        self.assertNotIn("Your skills:", follow_up_prompt)

        disabled_root = self.skills_dir / "disabled-skill"
        disabled_root.mkdir()
        (disabled_root / "SKILL.md").write_text(
            "---\nname: disabled\ndescription: Off\nenabled: false\n---\n",
            encoding="utf-8",
        )
        prompt_with_disabled = _compose_system_prompt("")
        self.assertNotIn("disabled-skill", prompt_with_disabled)

    # Verify disable skill queues refreshed inventory for next prompt.
    def test_disable_skill_queues_refreshed_inventory_for_next_prompt(self):
        skill_root = self.skills_dir / "pdf"
        skill_root.mkdir(parents=True)
        (skill_root / "SKILL.md").write_text(
            "---\nname: pdf\ndescription: PDF skill\nenabled: true\n---\n",
            encoding="utf-8",
        )

        baseline_prompt = _compose_system_prompt("", include_skills_baseline=True)
        self.assertIn("Your skills:", baseline_prompt)
        self.assertNotIn("Skill configuration update:", baseline_prompt)

        response = self.client.patch(
            reverse("skills_enabled_api"),
            data=json.dumps({"folder": "pdf", "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        prompt_after_disable = _compose_system_prompt("")
        self.assertIn("Skill configuration update:", prompt_after_disable)
        self.assertIn("no enabled project skills", prompt_after_disable)
        self.assertNotIn("Your skills:", prompt_after_disable)

        prompt_after_consume = _compose_system_prompt("")
        self.assertNotIn("Skill configuration update:", prompt_after_consume)
        self.assertNotIn("Your skills:", prompt_after_consume)

    # Verify sync mirrors skills and overrides sandbox.
    def test_sync_mirrors_skills_and_overrides_sandbox(self):
        skill_root = self.skills_dir / "writer"
        skill_root.mkdir(parents=True)
        (skill_root / "SKILL.md").write_text("# Writer\n", encoding="utf-8")
        stale_root = self.sandbox_skills_dir / "writer"
        stale_root.mkdir(parents=True)
        (stale_root / "SKILL.md").write_text("stale\n", encoding="utf-8")
        (self.sandbox_skills_dir / "extra.txt").write_text("remove\n", encoding="utf-8")

        result = skills_config.sync_skills_to_sandbox()

        self.assertGreaterEqual(result["copied"], 1)
        self.assertEqual((self.sandbox_skills_dir / "writer" / "SKILL.md").read_text(encoding="utf-8"), "# Writer\n")
        self.assertFalse((self.sandbox_skills_dir / "extra.txt").exists())

    # Verify sync excludes disabled skills from sandbox.
    def test_sync_excludes_disabled_skills_from_sandbox(self):
        enabled_root = self.skills_dir / "writer"
        enabled_root.mkdir(parents=True)
        (enabled_root / "SKILL.md").write_text("# Writer\n", encoding="utf-8")

        disabled_root = self.skills_dir / "disabled-skill"
        disabled_root.mkdir(parents=True)
        (disabled_root / "SKILL.md").write_text(
            "---\nname: disabled\ndescription: Off\nenabled: false\n---\n",
            encoding="utf-8",
        )

        stale_disabled = self.sandbox_skills_dir / "disabled-skill"
        stale_disabled.mkdir(parents=True)
        (stale_disabled / "SKILL.md").write_text("stale\n", encoding="utf-8")

        skills_config.sync_skills_to_sandbox()

        self.assertTrue((self.sandbox_skills_dir / "writer" / "SKILL.md").is_file())
        self.assertFalse(stale_disabled.exists())

    # Verify disable skill removes folder from sandbox.
    def test_disable_skill_removes_folder_from_sandbox(self):
        skill_root = self.skills_dir / "toggle-skill"
        skill_root.mkdir(parents=True)
        (skill_root / "SKILL.md").write_text(
            "---\nname: toggle-skill\ndescription: Toggle\nenabled: true\n---\n",
            encoding="utf-8",
        )
        skills_config.sync_skills_to_sandbox()
        self.assertTrue((self.sandbox_skills_dir / "toggle-skill" / "SKILL.md").is_file())

        response = self.client.patch(
            reverse("skills_enabled_api"),
            data=json.dumps({"folder": "toggle-skill", "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["folders"][0]["enabled"])
        self.assertFalse((self.sandbox_skills_dir / "toggle-skill").exists())

    # Verify create skill subdirectory.
    def test_create_skill_subdirectory(self):
        skill_root = self.skills_dir / "my-skill"
        skill_root.mkdir(parents=True)
        (skill_root / "SKILL.md").write_text("# My Skill\n", encoding="utf-8")

        response = self.client.post(
            reverse("skills_directory_api"),
            data=json.dumps({"folder": "my-skill", "path": "agents"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue((skill_root / "agents").is_dir())

        nested_response = self.client.post(
            reverse("skills_directory_api"),
            data=json.dumps({"folder": "my-skill", "path": "agents/tools"}),
            content_type="application/json",
        )
        self.assertEqual(nested_response.status_code, 200)
        self.assertTrue((skill_root / "agents" / "tools").is_dir())

    # Verify create skill subdirectory rejects duplicate.
    def test_create_skill_subdirectory_rejects_duplicate(self):
        skill_root = self.skills_dir / "my-skill"
        skill_root.mkdir(parents=True)
        (skill_root / "agents").mkdir()

        response = self.client.post(
            reverse("skills_directory_api"),
            data=json.dumps({"folder": "my-skill", "path": "agents"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    # Verify create skill subdirectory rejects traversal.
    def test_create_skill_subdirectory_rejects_traversal(self):
        skill_root = self.skills_dir / "my-skill"
        skill_root.mkdir(parents=True)

        response = self.client.post(
            reverse("skills_directory_api"),
            data=json.dumps({"folder": "my-skill", "path": "../escape"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    # Verify create skill subdirectory rejects file extension.
    def test_create_skill_subdirectory_rejects_file_extension(self):
        skill_root = self.skills_dir / "my-skill"
        skill_root.mkdir(parents=True)

        response = self.client.post(
            reverse("skills_directory_api"),
            data=json.dumps({"folder": "my-skill", "path": "agents.md"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    # Verify rename and delete skill subdirectory.
    def test_rename_and_delete_skill_subdirectory(self):
        skill_root = self.skills_dir / "my-skill"
        skill_root.mkdir(parents=True)
        (skill_root / "agents").mkdir()
        (skill_root / "agents" / "grader.md").write_text("# Grader\n", encoding="utf-8")

        renamed = self.client.patch(
            reverse("skills_path_api"),
            data=json.dumps({
                "folder": "my-skill",
                "old_path": "agents",
                "new_path": "reviewers",
                "kind": "directory",
            }),
            content_type="application/json",
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertTrue((skill_root / "reviewers" / "grader.md").is_file())
        self.assertFalse((skill_root / "agents").exists())

        deleted = self.client.delete(
            reverse("skills_directory_api"),
            data=json.dumps({"folder": "my-skill", "path": "reviewers"}),
            content_type="application/json",
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse((skill_root / "reviewers").exists())

    # Verify rename skill file.
    def test_rename_skill_file(self):
        skill_root = self.skills_dir / "my-skill"
        skill_root.mkdir(parents=True)
        (skill_root / "notes.md").write_text("# Notes\n", encoding="utf-8")

        renamed = self.client.patch(
            reverse("skills_path_api"),
            data=json.dumps({
                "folder": "my-skill",
                "old_path": "notes.md",
                "new_path": "journal.md",
                "kind": "file",
            }),
            content_type="application/json",
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertTrue((skill_root / "journal.md").is_file())
        self.assertFalse((skill_root / "notes.md").exists())

    # Verify import skill creates folder and files.
    def test_import_skill_creates_folder_and_files(self):
        files = [
            {"path": "SKILL.md", "content": "---\nname: imported\ndescription: Test import\nenabled: true\n---\n\n# Imported\n"},
            {"path": "agents/grader.md", "content": "# Grader\n"},
            {"path": "../escape.md", "content": "bad"},   # should be silently skipped
            {"path": "binary\x00.md", "content": "bad"},  # should be skipped
        ]
        response = self.client.post(
            reverse("skills_import_api"),
            data=json.dumps({"name": "imported-skill", "files": files}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        skill_root = self.skills_dir / "imported-skill"
        self.assertTrue((skill_root / "SKILL.md").is_file())
        self.assertTrue((skill_root / "agents" / "grader.md").is_file())
        self.assertFalse((self.skills_dir / "escape.md").exists())
        payload = response.json()
        folder_names = [f["name"] for f in payload["folders"]]
        self.assertIn("imported-skill", folder_names)
        imported = next(f for f in payload["folders"] if f["name"] == "imported-skill")
        self.assertEqual(imported["description"], "Test import")

    # Verify import skill merges into existing folder.
    def test_import_skill_merges_into_existing_folder(self):
        skill_root = self.skills_dir / "duplicate-skill"
        skill_root.mkdir(parents=True)
        (skill_root / "SKILL.md").write_text("# Duplicate\n", encoding="utf-8")

        response = self.client.post(
            reverse("skills_import_api"),
            data=json.dumps({
                "name": "duplicate-skill",
                "files": [{"path": "notes.md", "content": "# Notes\n"}],
            }),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue((skill_root / "notes.md").is_file())
        payload = response.json()
        folder_names = [f["name"] for f in payload["folders"]]
        self.assertIn("duplicate-skill", folder_names)


# Verify what the model receives in system context under skills toggles.
class SkillsModelContextTests(TestCase):
    # Prepare shared fixtures for each test case.
    def setUp(self):
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.skills_dir = self.root / "Skills"
        self.sandbox_skills_dir = self.root / "Tools" / "mcp-sandbox" / "_sandbox" / "Skills"
        self._patches = [
            patch.object(skills_config, "BASE_DIR", self.root),
            patch.object(skills_config, "SKILLS_DIR", self.skills_dir),
            patch.object(skills_config, "SANDBOX_SKILLS_DIR", self.sandbox_skills_dir),
            patch.object(skills_config, "_PENDING_NOTIFY_PATH", self.root / ".aslm" / "skills-pending-notify.json"),
        ]
        for patcher in self._patches:
            patcher.start()
        self.client = Client()
        skills_config.clear_skill_config_refresh_pending()

    # Clean up fixtures created for each test case.
    def tearDown(self):
        skills_config.clear_skill_config_refresh_pending()
        for patcher in reversed(self._patches):
            patcher.stop()
        self._tmp.cleanup()
        super().tearDown()

    # Assert has skills inventory.
    def _assert_has_skills_inventory(self, text: str, *folders: str) -> None:
        self.assertIn("Your skills:", text)
        for folder in folders:
            self.assertIn(f"/workspace/_sandbox/Skills/{folder}", text)

    # Assert no skill context.
    def _assert_no_skill_context(self, text: str) -> None:
        self.assertNotIn("Your skills:", text)
        self.assertNotIn("Skill configuration update:", text)
        self.assertNotIn("/workspace/_sandbox/Skills/", text)

    # Assert config update header.
    def _assert_config_update_header(self, text: str) -> None:
        self.assertIn("Skill configuration update:", text)
        self.assertIn("changed which project skills are enabled", text)

    # Write skill.
    def _write_skill(self, folder: str, *, enabled: bool = True, title: str | None = None) -> None:
        skill_root = self.skills_dir / folder
        skill_root.mkdir(parents=True, exist_ok=True)
        display = title or folder
        (skill_root / "SKILL.md").write_text(
            f"---\nname: {display}\ndescription: Test\nenabled: {'true' if enabled else 'false'}\n---\n",
            encoding="utf-8",
        )

    # Compose.
    def _compose(self, *, consume: bool = True, include_baseline: bool = False, user_prompt: str = "") -> str:
        return _compose_system_prompt(
            user_prompt,
            consume_skill_notifications=consume,
            include_skills_baseline=include_baseline,
        )

    # System message for chat.
    def _system_message_for_chat(self, system_prompt: str, user_text: str = "hello") -> str:
        chat = Chat.objects.create(title="Skills model context")
        user_record = Message.objects.create(chat=chat, role="user", content=user_text)
        llm_messages, _compression = _build_chat_history(
            chat,
            user_record,
            user_text,
            system_prompt,
            "ollama-service",
            "test-model",
        )
        system_entries = [entry for entry in llm_messages if entry.get("role") == "system"]
        self.assertEqual(len(system_entries), 1)
        return str(system_entries[0].get("content") or "")

    # Disable via api.
    def _disable_via_api(self, folder: str) -> None:
        response = self.client.patch(
            reverse("skills_enabled_api"),
            data=json.dumps({"folder": folder, "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

    # Verify enabled skills appear only on first chat turn.
    def test_enabled_skills_appear_only_on_first_chat_turn(self):
        self._write_skill("writer", enabled=True)
        self._write_skill("pdf", enabled=True, title="PDF helper")

        first_turn = self._compose(include_baseline=True)
        history = self._system_message_for_chat(first_turn)

        self._assert_has_skills_inventory(first_turn, "writer", "pdf")
        self._assert_has_skills_inventory(history, "writer", "pdf")
        self.assertNotIn("Skill configuration update:", first_turn)

        follow_up = self._compose(include_baseline=False)
        self._assert_no_skill_context(follow_up)

    # Verify chat api first turn includes skills baseline.
    def test_chat_api_first_turn_includes_skills_baseline(self):
        self._write_skill("writer", enabled=True)
        chat = Chat.objects.create(title="Skills baseline chat")
        prompt = _compose_system_prompt("", include_skills_baseline=_chat_is_first_user_turn(chat))
        self._assert_has_skills_inventory(prompt, "writer")

        chat.messages.create(role="user", content="hello")
        prompt_after_history = _compose_system_prompt("", include_skills_baseline=_chat_is_first_user_turn(chat))
        self._assert_no_skill_context(prompt_after_history)

    # Verify static disabled skill is omitted from inventory.
    def test_static_disabled_skill_is_omitted_from_inventory(self):
        self._write_skill("writer", enabled=True)
        self._write_skill("legacy-off", enabled=False, title="Legacy")

        composed = self._compose(include_baseline=True)
        self._assert_has_skills_inventory(composed, "writer")
        self.assertNotIn("legacy-off", composed)

    # Verify disable sends updated inventory once.
    def test_disable_sends_updated_inventory_once(self):
        self._write_skill("writer", enabled=True)
        self._write_skill("pdf", enabled=True, title="PDF skill")
        self._disable_via_api("pdf")

        first_compose = self._compose(consume=True)
        self._assert_config_update_header(first_compose)
        self._assert_has_skills_inventory(first_compose, "writer")
        self.assertNotIn("/workspace/_sandbox/Skills/pdf", first_compose)

        second_compose = self._compose(consume=True)
        self.assertNotIn("Skill configuration update:", second_compose)
        self._assert_no_skill_context(second_compose)

    # Verify context usage style compose does not consume pending refresh.
    def test_context_usage_style_compose_does_not_consume_pending_refresh(self):
        self._write_skill("writer", enabled=True)
        self._write_skill("docx", enabled=True, title="DOCX skill")
        self._disable_via_api("docx")

        peek = self._compose(consume=False)
        self._assert_config_update_header(peek)
        self._assert_has_skills_inventory(peek, "writer")
        self.assertTrue(skills_config._peek_config_refresh_pending())

        generation = self._compose(consume=True)
        self._assert_config_update_header(generation)
        self.assertFalse(skills_config._peek_config_refresh_pending())

        follow_up = self._compose(consume=True)
        self.assertNotIn("Skill configuration update:", follow_up)
        self._assert_no_skill_context(follow_up)

    # Verify enable queues refreshed inventory with enabled skill.
    def test_enable_queues_refreshed_inventory_with_enabled_skill(self):
        self._write_skill("pdf", enabled=True, title="PDF")
        self._disable_via_api("pdf")
        self._compose(consume=True)

        enable_response = self.client.patch(
            reverse("skills_enabled_api"),
            data=json.dumps({"folder": "pdf", "enabled": True}),
            content_type="application/json",
        )
        self.assertEqual(enable_response.status_code, 200)

        composed = self._compose(consume=True)
        self._assert_config_update_header(composed)
        self._assert_has_skills_inventory(composed, "pdf")

        follow_up = self._compose(consume=True)
        self.assertNotIn("Skill configuration update:", follow_up)
        self._assert_no_skill_context(follow_up)

    # Verify re toggle without change does not queue refresh.
    def test_re_toggle_without_change_does_not_queue_refresh(self):
        self._write_skill("pdf", enabled=False, title="PDF")

        response = self.client.patch(
            reverse("skills_enabled_api"),
            data=json.dumps({"folder": "pdf", "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        composed = self._compose(consume=True)
        self._assert_no_skill_context(composed)
        self.assertFalse(skills_config._peek_config_refresh_pending())

    @patch.object(skills_config, "sync_skills_to_sandbox", side_effect=RuntimeError("sandbox unavailable"))
    # Verify toggle still queues inventory when sandbox sync fails.
    def test_toggle_still_queues_inventory_when_sandbox_sync_fails(self, _sync_mock):
        self._write_skill("writer", enabled=True)
        self._write_skill("pdf", enabled=True, title="PDF skill")
        response = self.client.patch(
            reverse("skills_enabled_api"),
            data=json.dumps({"folder": "pdf", "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        prompt = self._compose(consume=True)
        self._assert_config_update_header(prompt)
        self._assert_has_skills_inventory(prompt, "writer")


# Ensure sandbox dispatch refreshes Skills before executing a tool.
class SkillsSandboxDispatchTests(SimpleTestCase):
    # Verify sandbox tool dispatch syncs skills first.
    def test_sandbox_tool_dispatch_syncs_skills_first(self):
        server = {
            "id": "sandbox",
            "name": "Sandbox",
            "description": "",
            "tools": [{"id": "write", "alias": "sandbox__write", "name": "Write", "description": ""}],
            "module": None,
            "supports": None,
            "server_callable": None,
            "tool_handlers": {"write": lambda arguments, context=None: {"ok": True, "result": arguments}},
            "server_file": Path("Tools/mcp-sandbox/mcp-server.py"),
            "external": False,
        }
        lookup = {"sandbox__write": {"server": server, "tool": server["tools"][0]}}

        with patch.object(skills_config, "sync_skills_to_sandbox") as sync_mock:
            result = tool_registry.call_ollama_tool(lookup, "sandbox__write", {"path": "x.txt"})

        sync_mock.assert_called_once()
        self.assertIn("x.txt", str(result))


class ToolCancellationTests(SimpleTestCase):
    def tearDown(self):
        tool_registry.abort_active_tools()
        with tool_registry._ACTIVE_TOOL_EXECUTIONS_LOCK:
            tool_registry._ACTIVE_TOOL_EXECUTIONS.clear()

    def test_abort_active_tools_only_cancels_matching_generation(self):
        first = tool_registry.ActiveToolExecution("generation-a")
        second = tool_registry.ActiveToolExecution("generation-b")
        first_callback = Mock()
        second_callback = Mock()
        first.set_cancel_callback(first_callback)
        second.set_cancel_callback(second_callback)
        tool_registry._register_active_tool(first)
        tool_registry._register_active_tool(second)

        cancelled_count = tool_registry.abort_active_tools("generation-a")

        self.assertEqual(cancelled_count, 1)
        self.assertTrue(first.cancelled.is_set())
        self.assertFalse(second.cancelled.is_set())
        first_callback.assert_called_once_with()
        second_callback.assert_not_called()

    def test_abort_active_research_only_cancels_matching_session(self):
        first = tool_registry.ActiveToolExecution(
            "generation-a",
            research_session_id="deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        )
        second = tool_registry.ActiveToolExecution(
            "generation-a",
            research_session_id="deep-research:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        )
        unrelated = tool_registry.ActiveToolExecution("generation-a")
        first_callback = Mock()
        second_callback = Mock()
        unrelated_callback = Mock()
        first.set_cancel_callback(first_callback)
        second.set_cancel_callback(second_callback)
        unrelated.set_cancel_callback(unrelated_callback)
        tool_registry._register_active_tool(first)
        tool_registry._register_active_tool(second)
        tool_registry._register_active_tool(unrelated)

        cancelled_count = tool_registry.abort_active_research(
            "deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        )

        self.assertEqual(cancelled_count, 1)
        self.assertTrue(first.cancelled.is_set())
        self.assertFalse(second.cancelled.is_set())
        self.assertFalse(unrelated.cancelled.is_set())
        first_callback.assert_called_once_with()
        second_callback.assert_not_called()
        unrelated_callback.assert_not_called()

    def test_streamed_in_process_tool_receives_cancel_event_and_stops(self):
        started = threading.Event()
        stopped = threading.Event()

        def blocking_handler(_arguments, context):
            started.set()
            context["cancel_event"].wait(timeout=2)
            stopped.set()
            return "finished"

        tool = {
            "id": "wait",
            "alias": "local__wait",
            "name": "Wait",
            "description": "",
        }
        server = {
            "id": "local",
            "name": "Local",
            "external": False,
            "server_file": Path("Tools/local/mcp-server.py"),
            "tool_handlers": {"wait": blocking_handler},
            "server_callable": None,
        }
        lookup = {"local__wait": {"server": server, "tool": tool}}
        outcome = {}

        def consume() -> None:
            stream = tool_registry.stream_ollama_tool(
                lookup,
                "local__wait",
                {},
                context={"generation_id": "generation-a"},
            )
            try:
                while True:
                    next(stream)
            except StopIteration as exc:
                outcome["result"] = exc.value

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(started.wait(timeout=1))

        self.assertEqual(tool_registry.abort_active_tools("generation-a"), 1)
        consumer.join(timeout=1)

        self.assertFalse(consumer.is_alive())
        self.assertTrue(stopped.wait(timeout=1))
        self.assertTrue(tool_registry.is_tool_execution_cancelled(outcome.get("result")))

    def test_research_stop_maps_interrupted_worker_error_to_cancelled_result(self):
        session_id = "deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        started = threading.Event()
        outcome = {}

        def interrupted_call(*_args, **kwargs):
            cancellation = kwargs["cancellation"]
            started.set()
            cancellation.cancelled.wait(timeout=2)
            raise RuntimeError("worker process was terminated")

        def consume() -> None:
            with patch.object(tool_registry, "call_ollama_tool", side_effect=interrupted_call):
                stream = tool_registry.stream_ollama_tool(
                    {},
                    "deep_research__deep_research",
                    {"session_id": session_id},
                    context={"generation_id": "generation-a"},
                )
                try:
                    while True:
                        next(stream)
                except StopIteration as exc:
                    outcome["result"] = exc.value

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(started.wait(timeout=1))

        self.assertEqual(tool_registry.abort_active_research(session_id), 1)
        consumer.join(timeout=1)

        self.assertFalse(consumer.is_alive())
        self.assertTrue(tool_registry.is_tool_execution_cancelled(outcome.get("result")))

    def test_research_worker_monitors_durable_stop_while_provider_is_blocked(self):
        session_id = "deep-research:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        provider_started = threading.Event()
        stop_requested = threading.Event()
        cancellation_checkpointed = threading.Event()
        outcome = {}

        def blocked_provider(*_args, **kwargs):
            cancellation = kwargs["cancellation"]
            provider_started.set()
            cancellation.cancelled.wait(timeout=2)
            raise RuntimeError("blocked provider interrupted")

        def consume() -> None:
            def checkpoint_cancelled(*_args, **_kwargs):
                cancellation_checkpointed.set()
                return {}

            with (
                patch.object(tool_registry, "call_ollama_tool", side_effect=blocked_provider),
                patch(
                    "Tools.deep_research.control.latest_cancel_requested",
                    side_effect=lambda _session_id: stop_requested.is_set(),
                ),
                patch(
                    "Tools.deep_research.control.update_state",
                    side_effect=checkpoint_cancelled,
                ),
            ):
                stream = tool_registry.stream_ollama_tool(
                    {},
                    "deep_research__deep_research",
                    {"session_id": session_id},
                    context={"generation_id": "generation-a"},
                )
                try:
                    while True:
                        next(stream)
                except StopIteration as exc:
                    outcome["result"] = exc.value

        consumer = threading.Thread(target=consume, daemon=True)
        consumer.start()
        self.assertTrue(provider_started.wait(timeout=1))

        stop_requested.set()
        consumer.join(timeout=2)

        self.assertFalse(consumer.is_alive())
        self.assertTrue(cancellation_checkpointed.wait(timeout=1))
        self.assertTrue(tool_registry.is_tool_execution_cancelled(outcome.get("result")))

    def test_ollama_cancellation_forces_the_next_round_to_run_without_tools(self):
        ollama_api._abort_event.clear()
        self.addCleanup(ollama_api._abort_event.clear)
        round_tools = []
        round_prompts = []
        round_results = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "deep_research__deep_research",
                        "arguments": {"topic": "Test"},
                    }
                }],
            },
            {"role": "assistant", "content": "Stopped as requested."},
        ]

        def fake_stream_round(_client, _model, conversation, _kwargs, *, tools=None):
            round_tools.append(tools)
            round_prompts.append(str(conversation[-1].get("content") or ""))
            result = round_results[len(round_tools) - 1]

            def stream():
                if False:
                    yield {}
                return result

            return stream()

        def cancelled_tool_stream(*_args, **_kwargs):
            if False:
                yield {}
            return tool_registry.TOOL_EXECUTION_CANCELLED

        event = {
            "alias": "deep_research__deep_research__0",
            "server_id": "deep-research",
            "server_name": "Deep Research",
            "tool_id": "deep_research",
            "tool_name": "Deep Research",
            "arguments": {"topic": "Test"},
        }
        with (
            patch.object(ollama_api, "_stream_round", side_effect=fake_stream_round),
            patch.object(
                tool_registry,
                "build_ollama_tools",
                return_value=([{"type": "function"}], {}),
            ),
            patch.object(tool_registry, "prepare_tool_calls", side_effect=lambda _lookup, calls: calls),
            patch.object(
                tool_registry,
                "limit_tool_calls_from_context",
                side_effect=lambda _context, calls, _lookup: calls,
            ),
            patch.object(ollama_api, "_build_tool_event", return_value=event),
            patch.object(tool_registry, "stream_ollama_tool", side_effect=cancelled_tool_stream),
        ):
            list(
                ollama_api._run_tool_loop(
                    SimpleNamespace(),
                    "gemma-test",
                    [{"role": "user", "content": "Research this"}],
                    {},
                    ["deep-research"],
                    {},
                )
            )

        self.assertIsNotNone(round_tools[0])
        self.assertIsNone(round_tools[1])
        self.assertIn("Do not call any tool again", round_prompts[1])

    def test_main_chat_abort_does_not_start_a_cancellation_acknowledgement_round(self):
        ollama_api._abort_event.set()
        self.addCleanup(ollama_api._abort_event.clear)
        round_count = 0

        def fake_stream_round(_client, _model, _conversation, _kwargs, *, tools=None):
            nonlocal round_count
            round_count += 1
            if round_count > 1:
                raise AssertionError("main abort must not start another provider request")

            def stream():
                if False:
                    yield {}
                return {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "deep_research__deep_research",
                            "arguments": {"topic": "Test"},
                        }
                    }],
                }

            return stream()

        def cancelled_tool_stream(*_args, **_kwargs):
            if False:
                yield {}
            return tool_registry.TOOL_EXECUTION_CANCELLED

        event = {
            "alias": "deep_research__deep_research__0",
            "server_id": "deep-research",
            "server_name": "Deep Research",
            "tool_id": "deep_research",
            "tool_name": "Deep Research",
            "arguments": {"topic": "Test"},
        }
        with (
            patch.object(ollama_api, "_stream_round", side_effect=fake_stream_round),
            patch.object(
                tool_registry,
                "build_ollama_tools",
                return_value=([{"type": "function"}], {}),
            ),
            patch.object(tool_registry, "prepare_tool_calls", side_effect=lambda _lookup, calls: calls),
            patch.object(
                tool_registry,
                "limit_tool_calls_from_context",
                side_effect=lambda _context, calls, _lookup: calls,
            ),
            patch.object(ollama_api, "_build_tool_event", return_value=event),
            patch.object(tool_registry, "stream_ollama_tool", side_effect=cancelled_tool_stream),
        ):
            list(
                ollama_api._run_tool_loop(
                    SimpleNamespace(),
                    "gemma-test",
                    [{"role": "user", "content": "Research this"}],
                    {},
                    ["deep-research"],
                    {},
                )
            )

        self.assertEqual(round_count, 1)

    def test_cancelled_persistent_worker_is_not_restarted(self):
        cancellation = tool_registry.ActiveToolExecution("generation-a")
        session = tool_registry.ExternalWorkerSession(Path("Tools/example/mcp-server.py"), Path("python"))
        session.process = SimpleNamespace(stdin=Mock(), stdout=Mock())

        def cancelled_read(_timeout):
            cancellation.cancelled.set()
            return ""

        with (
            patch.object(session, "_start") as start_mock,
            patch.object(session, "_read_response_line", side_effect=cancelled_read),
            patch.object(session, "close") as close_mock,
        ):
            with self.assertRaises(tool_registry.ToolExecutionCancelled):
                session.request("call", {}, timeout_s=1, cancellation=cancellation)

        start_mock.assert_called_once_with()
        close_mock.assert_called_once_with()

    def test_cancelled_during_persistent_worker_start_is_closed_before_request(self):
        cancellation = tool_registry.ActiveToolExecution("generation-a")
        session = tool_registry.ExternalWorkerSession(Path("Tools/example/mcp-server.py"), Path("python"))

        def cancel_during_start():
            cancellation.cancelled.set()

        with (
            patch.object(session, "_start", side_effect=cancel_during_start) as start_mock,
            patch.object(session, "close") as close_mock,
        ):
            with self.assertRaises(tool_registry.ToolExecutionCancelled):
                session.request("call", {}, timeout_s=1, cancellation=cancellation)

        start_mock.assert_called_once_with()
        close_mock.assert_called_once_with()

    @patch("API.llm_api.tool_registry.abort_active_tools")
    @patch("API.llm_api._get_engine_module")
    def test_llm_abort_forwards_generation_id_to_tools(self, get_module, abort_tools):
        adapter = SimpleNamespace(abort_generation=Mock())
        get_module.return_value = adapter

        llm_api.abort_generation("ollama-service", generation_id="generation-a")

        abort_tools.assert_called_once_with("generation-a")
        adapter.abort_generation.assert_called_once_with()


# Cover per-response tool quota guardrails.
class ToolQuotaTests(SimpleTestCase):
    # High-effort web search is expensive, so keep it bounded per response.
    def test_high_effort_web_search_limits_to_three_calls(self):
        tool_event = {"tool_id": "web_search", "tool_name": "Web search"}
        counters: dict[str, int] = {}
        arguments = {"query": "cheap coding model", "effort": "high"}

        self.assertIsNone(tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments))
        self.assertIsNone(tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments))
        self.assertIsNone(tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments))

        error = tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments)
        self.assertIsNotNone(error)
        self.assertIn("high mode is unavailable", str(error))
        self.assertIn("use medium or low", str(error))

    # Lower-effort searches keep the existing broader budget.
    def test_normal_web_search_keeps_default_quota(self):
        tool_event = {"tool_id": "web_search", "tool_name": "Web search"}
        counters: dict[str, int] = {}
        arguments = {"query": "cheap coding model", "effort": "medium"}

        for _index in range(4):
            self.assertIsNone(tool_registry.consume_tool_quota(tool_event, counters, arguments=arguments))

    def test_instant_search_and_read_page_each_allow_two_calls(self):
        counters: dict[str, int] = {}
        for tool_id in ("web_search", "read_page"):
            event = {"tool_id": tool_id, "tool_name": tool_id, "instant_mode": True}
            self.assertIsNone(tool_registry.consume_tool_quota(event, counters, arguments={}))
            self.assertIsNone(tool_registry.consume_tool_quota(event, counters, arguments={}))
            self.assertIn(
                "at most 2 times",
                str(tool_registry.consume_tool_quota(event, counters, arguments={})),
            )


class InstantToolSchemaTests(SimpleTestCase):
    def test_all_engine_adapters_forward_forced_instant_batch_size(self):
        context = {"instant_mode": True, "instant_search_batch_size": 3}
        adapter_cases = (
            (
                ollama_api,
                lambda: list(ollama_api._run_tool_loop(
                    None, "model", [{"role": "user", "content": "Find"}], {}, ["web_search"], context
                )),
                "_stream_round",
            ),
            (
                openai_api,
                lambda: list(openai_api._run_tool_loop(
                    None, "model", [{"role": "user", "content": "Find"}], {}, ["web_search"], context
                )),
                "_stream_openai_round",
            ),
            (
                lms_api,
                lambda: list(lms_api._run_tool_loop(
                    None, "model", [{"role": "user", "content": "Find"}], {}, ["web_search"], context
                )),
                "_stream_openai_round",
            ),
        )

        for adapter, invoke, stream_name in adapter_cases:
            with self.subTest(adapter=adapter.__name__):
                with (
                    patch.object(tool_registry, "build_ollama_tools", return_value=([], {})) as builder,
                    patch.object(adapter, stream_name, return_value=iter(())),
                ):
                    invoke()
                self.assertEqual(builder.call_args.kwargs["instant_search_batch_size"], 3)

        with patch.object(tool_registry, "build_ollama_tools", return_value=([], {})) as builder:
            google_genai_api._build_google_tools(["web_search"], "model", context)
        self.assertEqual(builder.call_args.kwargs["instant_search_batch_size"], 3)

    def test_instant_search_schema_is_minimal_and_thinking_schema_is_unchanged(self):
        server = {
            "id": "web_search",
            "name": "Web Search",
            "description": "Search tools",
            "tools": [
                {
                    "id": "web_search",
                    "alias": "web_search__web_search",
                    "name": "Web Search",
                    "description": "Full search",
                    "parameters": {
                        "type": "object",
                        "properties": {"web": {}, "shopping": {}, "effort": {}},
                    },
                    "prepares_arguments": True,
                },
                {
                    "id": "read_page",
                    "alias": "web_search__read_page",
                    "name": "Read Page",
                    "description": "Read pages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "items": {"type": "string"}},
                                ]
                            }
                        },
                    },
                    "prepares_arguments": False,
                },
            ],
        }
        with patch.object(tool_registry, "get_server", return_value=server):
            instant_tools, instant_lookup = tool_registry.build_ollama_tools(
                ["web_search"], instant_mode=True
            )
            forced_instant_tools, forced_instant_lookup = tool_registry.build_ollama_tools(
                ["web_search"],
                instant_mode=True,
                instant_search_batch_size=3,
            )
            thinking_tools, _thinking_lookup = tool_registry.build_ollama_tools(["web_search"])

        instant_schema = instant_tools[0]["function"]["parameters"]
        self.assertEqual(set(instant_schema["properties"]), {"query", "description", "operators"})
        self.assertEqual(instant_schema["properties"]["query"]["oneOf"][1]["maxItems"], 3)
        self.assertTrue(instant_lookup["web_search__web_search"]["tool"]["instant_mode"])
        forced_query_schema = forced_instant_tools[0]["function"]["parameters"]["properties"]["query"]
        self.assertEqual(forced_query_schema["type"], "array")
        self.assertEqual(forced_query_schema["minItems"], 3)
        self.assertEqual(forced_query_schema["maxItems"], 3)
        self.assertEqual(
            forced_instant_lookup["web_search__web_search"]["tool"]["instant_search_batch_size"],
            3,
        )
        self.assertIn("effort", thinking_tools[0]["function"]["parameters"]["properties"])

    def test_forced_instant_search_preflight_requires_exactly_three_queries(self):
        preparer = Mock(return_value={
            "ok": True,
            "arguments": {
                "query": ["alpha", "beta", "gamma"],
                "description": "Checking sources",
            },
            "tool_ui": {"kind": "web_search", "status": "pending", "instant_mode": True},
        })
        tool = {
            "id": "web_search",
            "alias": "web_search__web_search",
            "instant_mode": True,
            "instant_search_batch_size": 3,
            "prepares_arguments": True,
        }
        server = {
            "id": "web_search",
            "external": False,
            "tool_preparers": {"web_search": preparer},
        }
        lookup = {tool["alias"]: {"server": server, "tool": tool}}

        rejected = tool_registry.prepare_tool_call(lookup, {
            "name": tool["alias"],
            "arguments": {"query": "alpha", "description": "Checking sources"},
        })
        accepted = tool_registry.prepare_tool_call(lookup, {
            "name": tool["alias"],
            "arguments": {
                "query": ["alpha", "beta", "gamma"],
                "description": "Checking sources",
            },
        })

        self.assertEqual(
            rejected["preflight_error_result"]["ui"]["error"]["code"],
            "INVALID_INSTANT_SEARCH_BATCH",
        )
        self.assertEqual(rejected["tool_ui"]["query_count"], 0)
        self.assertNotIn("preflight_error_result", accepted)
        preparer.assert_called_once()

        preparer.reset_mock()
        preparer.return_value = {
            "ok": True,
            "arguments": {
                "query": ["alpha", "beta", "gamma"],
                "description": "Checking alpha · Checking beta · Checking gamma",
            },
            "tool_ui": {"kind": "web_search", "status": "pending", "instant_mode": True},
        }
        collapsed = tool_registry.prepare_tool_calls(lookup, [
            {"name": tool["alias"], "arguments": {"query": "alpha", "description": "Checking alpha"}},
            {"name": tool["alias"], "arguments": {"query": "beta", "description": "Checking beta"}},
            {"name": tool["alias"], "arguments": {"query": "gamma", "description": "Checking gamma"}},
        ])

        self.assertEqual(len(collapsed), 1)
        self.assertEqual(collapsed[0]["arguments"]["query"], ["alpha", "beta", "gamma"])
        self.assertEqual(collapsed[0]["tool_ui"]["collapsed_parallel_calls"], 3)
        preparer.assert_called_once()

    def test_in_process_tool_preparer_is_discovered_and_executed(self):
        preparer = Mock(return_value={
            "ok": True,
            "arguments": {"query": "alpha", "description": "Checking alpha"},
            "tool_ui": {"kind": "web_search", "status": "pending", "instant_mode": True},
        })
        module = SimpleNamespace(
            MCP_SERVER={"id": "web_search", "name": "Web Search"},
            TOOLS=[{
                "id": "web_search",
                "name": "Web Search",
                "description": "Search",
                "parameters": {"type": "object", "properties": {}},
            }],
            TOOL_PREPARERS={"web_search": preparer},
            call_tool=lambda *_args, **_kwargs: {},
        )
        server = tool_registry._extract_server_definition(
            module,
            "mcp-web-search",
            Path("Tools/mcp-web-search/mcp-server.py"),
        )
        tool = dict(server["tools"][0])
        tool["instant_mode"] = True
        lookup = {tool["alias"]: {"server": server, "tool": tool}}

        prepared = tool_registry.prepare_tool_calls(lookup, [{
            "id": "call-1",
            "name": tool["alias"],
            "arguments": {"query": "alpha", "description": "Checking alpha"},
        }])

        self.assertTrue(server["tools"][0]["prepares_arguments"])
        preparer.assert_called_once()
        self.assertTrue(preparer.call_args.args[0]["_instant_mode"])
        self.assertTrue(prepared[0]["tool_ui"]["instant_mode"])

    def test_ollama_instant_search_read_and_retry_still_reaches_final_answer(self):
        server = {"id": "web_search", "name": "Web Search"}
        search_tool = {
            "id": "web_search",
            "alias": "web_search__web_search",
            "name": "Web Search",
            "instant_mode": True,
        }
        read_tool = {
            "id": "read_page",
            "alias": "web_search__read_page",
            "name": "Read Page",
            "instant_mode": True,
        }
        lookup = {
            search_tool["alias"]: {"server": server, "tool": search_tool},
            read_tool["alias"]: {"server": server, "tool": read_tool},
        }
        round_results = [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": search_tool["alias"],
                        "arguments": {"query": "first", "description": "Finding sources"},
                    }
                }],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": read_tool["alias"],
                        "arguments": {"url": "https://example.com/source"},
                    }
                }],
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": search_tool["alias"],
                        "arguments": {"query": "retry", "description": "Retrying search"},
                    }
                }],
            },
            {"role": "assistant", "content": "Final answer"},
        ]

        def fake_stream_round(_client, _model, _conversation, _kwargs, *, tools=None):
            result = round_results.pop(0)

            def stream():
                if False:
                    yield {}
                return result

            return stream()

        def fake_tool_stream(*_args, **_kwargs):
            if False:
                yield {}
            return {
                "model_context": "Tool result",
                "sources": [],
                "ui": {"kind": "web_search", "status": "done", "instant_mode": True},
            }

        with (
            patch.object(ollama_api, "_stream_round", side_effect=fake_stream_round),
            patch.object(tool_registry, "build_ollama_tools", return_value=([{"type": "function"}], lookup)),
            patch.object(tool_registry, "stream_ollama_tool", side_effect=fake_tool_stream),
        ):
            chunks = list(
                ollama_api._run_tool_loop(
                    SimpleNamespace(),
                    "test-model",
                    [{"role": "user", "content": "Find this"}],
                    {},
                    ["web_search"],
                    {"instant_mode": True},
                )
            )

        self.assertFalse(round_results)
        self.assertTrue(any(
            chunk.get("transcript_message", {}).get("content") == "Final answer"
            for chunk in chunks
            if isinstance(chunk, dict)
        ))
        self.assertFalse(any(
            "tool loop exceeded" in str(chunk).lower()
            for chunk in chunks
        ))


class ToolCooldownTests(SimpleTestCase):
    def setUp(self):
        tool_registry._TOOL_COOLDOWNS.clear()

    def tearDown(self):
        tool_registry._TOOL_COOLDOWNS.clear()

    # Identical tool arguments in separate chats must not share a cooldown.
    def test_web_search_cooldown_is_scoped_by_chat_id(self):
        tool_event = {"tool_id": "web_search", "tool_name": "Web search"}
        arguments = {"query": "same query", "effort": "medium"}
        first_chat = {"chat_id": "chat-a"}
        second_chat = {"chat_id": "chat-b"}

        tool_registry.remember_tool_cooldown(tool_event, arguments, context=first_chat)

        self.assertIsNotNone(
            tool_registry.consume_tool_cooldown(tool_event, arguments, context=first_chat)
        )
        self.assertIsNone(
            tool_registry.consume_tool_cooldown(tool_event, arguments, context=second_chat)
        )

    # The scope applies to page reads as well as searches.
    def test_read_page_cooldown_is_scoped_by_chat_id(self):
        tool_event = {"tool_id": "read_page", "tool_name": "Read page"}
        arguments = {"url": "https://example.com/page"}

        tool_registry.remember_tool_cooldown(
            tool_event, arguments, context={"chat_id": "chat-a"}
        )

        self.assertIsNone(
            tool_registry.consume_tool_cooldown(
                tool_event, arguments, context={"chat_id": "chat-b"}
            )
        )


class ToolPreflightTests(SimpleTestCase):
    def _lookup(self):
        tool = {
            "id": "web_search",
            "alias": "web_search__web_search",
            "name": "Web Search",
            "prepares_arguments": True,
        }
        server = {
            "id": "web_search",
            "name": "Web Search",
            "external": True,
            "server_file": Path("Tools/mcp-web-search/mcp-server.py"),
        }
        return {"web_search__web_search": {"server": server, "tool": tool}}

    @patch.object(tool_registry, "_run_worker")
    def test_preflight_replaces_raw_arguments_before_tool_event(self, worker_mock):
        worker_mock.return_value = {
            "ok": True,
            "arguments": {
                "call_description": "Verify canonical sources",
                "web": "canonical",
                "effort": "medium",
            },
            "tool_ui": {
                "kind": "web_search",
                "status": "pending",
                "description": "Verify canonical sources",
                "search_request": {
                    "schema_mode": "advanced",
                    "description": "Verify canonical sources",
                    "effort": "medium",
                    "queries": [{"vertical": "web", "compiled_query": "canonical", "operators": {}}],
                },
            },
        }
        raw = {
            "call_description": "Verify raw sources",
            "web": "raw model value",
        }
        call = tool_registry.prepare_tool_call(
            self._lookup(),
            {"name": "web_search__web_search", "arguments": raw},
        )
        event = tool_registry.build_tool_event(self._lookup(), call)

        self.assertEqual(call["raw_arguments"], raw)
        self.assertNotIn("raw model value", json.dumps(event))
        self.assertEqual(event["arguments"]["web"], "canonical")
        self.assertEqual(event["tool_ui"]["description"], "Verify canonical sources")
        self.assertEqual(event["tool_ui"]["search_request"]["queries"][0]["compiled_query"], "canonical")
        marker = _serialize_tool_call_marker(event)
        self.assertIn('"tool_ui"', marker)
        self.assertNotIn("raw model value", marker)

        transcript = tool_registry.canonicalize_transcript_tool_calls(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "web_search__web_search",
                            "arguments": json.dumps(raw),
                        },
                    }
                ],
            },
            [call],
        )
        self.assertNotIn("raw model value", json.dumps(transcript))
        transcript_arguments = transcript["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(json.loads(transcript_arguments), event["arguments"])

    @patch.object(tool_registry, "_run_worker")
    def test_rejected_preflight_keeps_structured_error_for_adapter_short_circuit(self, worker_mock):
        error_result = {
            "model_context": "INVALID_SEARCH_PLAN: $: must include a vertical query",
            "sources": [],
            "ui": {"kind": "web_search", "status": "rejected"},
        }
        worker_mock.return_value = {
            "ok": False,
            "arguments": {},
            "tool_ui": error_result["ui"],
            "error_result": error_result,
        }
        raw_arguments = {"query": "legacy"}
        call = tool_registry.prepare_tool_call(
            self._lookup(),
            {"name": "web_search__web_search", "arguments": raw_arguments},
        )

        self.assertEqual(call["arguments"], {})
        self.assertEqual(call["preflight_error_result"], error_result)
        self.assertEqual(call["tool_ui"]["status"], "rejected")
        self.assertEqual(call["tool_ui"]["rejected_arguments"], raw_arguments)
        self.assertTrue(tool_registry.is_blocking_tool_result(call["preflight_error_result"]))

    @patch.object(tool_registry, "_run_worker")
    def test_parallel_advanced_search_calls_are_preflighted_as_one_batch(self, worker_mock):
        def prepare_batch(_server_file, _operation, payload, persistent=False):
            self.assertTrue(persistent)
            arguments = payload["arguments"]
            self.assertEqual(arguments["web"], ["first query", "second query"])
            return {
                "ok": True,
                "arguments": arguments,
                "tool_ui": {
                    "kind": "web_search",
                    "status": "pending",
                    "query_count": 2,
                },
            }

        worker_mock.side_effect = prepare_batch
        calls = [
            {
                "id": "call-1",
                "name": "web_search__web_search",
                "arguments": {
                    "call_description": "Check first source",
                    "web": "first query",
                    "effort": "medium",
                },
            },
            {
                "id": "call-2",
                "name": "web_search__web_search",
                "arguments": {
                    "call_description": "Check second source",
                    "web": "second query",
                    "effort": "medium",
                },
            },
        ]

        prepared = tool_registry.prepare_tool_calls(self._lookup(), calls)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["id"], "call-1")
        self.assertEqual(prepared[0]["parallel_batch_size"], 2)
        self.assertEqual([call["id"] for call in prepared[0]["absorbed_tool_calls"]], ["call-2"])
        worker_mock.assert_called_once()

    @patch.object(tool_registry, "_run_worker")
    def test_parallel_legacy_search_batch_over_server_limit_is_one_atomic_rejection(self, worker_mock):
        def reject_oversized_batch(_server_file, _operation, payload, persistent=False):
            self.assertTrue(persistent)
            query_count = len(payload["arguments"]["query"])
            self.assertEqual(query_count, 4)
            error_result = {
                "error": {
                    "code": "INVALID_SEARCH_PLAN",
                    "issues": [{"path": "$", "message": "batch permits at most 2 queries total"}],
                },
                "sources": [],
                "model_context": "INVALID_SEARCH_PLAN: $: batch permits at most 2 queries total",
                "ui": {"kind": "web_search", "status": "rejected", "query_count": 0},
            }
            return {
                "ok": False,
                "arguments": {},
                "tool_ui": error_result["ui"],
                "error_result": error_result,
            }

        worker_mock.side_effect = reject_oversized_batch
        calls = [
            {
                "id": f"call-{index}",
                "name": "web_search__web_search",
                "arguments": {
                    "call_description": f"Check source {index}",
                    "query": f"query {index}",
                    "effort": "medium",
                },
            }
            for index in range(1, 5)
        ]

        prepared = tool_registry.prepare_tool_calls(self._lookup(), calls)

        self.assertEqual(len(prepared), 1)
        self.assertEqual(prepared[0]["tool_ui"]["status"], "rejected")
        self.assertIn("4 parallel web_search calls", prepared[0]["preflight_error_result"]["model_context"])
        self.assertIn("batch permits at most 2 queries total", prepared[0]["preflight_error_result"]["model_context"])
        worker_mock.assert_called_once()

        transcript = tool_registry.canonicalize_transcript_tool_calls(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": call["id"],
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                    for call in calls
                ],
                "google_parts": [
                    {
                        "function_call": {
                            "name": call["name"],
                            "args": call["arguments"],
                        },
                        "thought_signature": f"signature-{call['id']}",
                    }
                    for call in calls
                ],
            },
            prepared,
        )
        self.assertEqual([call["id"] for call in transcript["tool_calls"]], ["call-1"])
        self.assertEqual(len(transcript["google_parts"]), 1)

# Cover adapter-specific model list formats.
class ModelNameExtractionTests(SimpleTestCase):
    # Test extracts name from string.
    def test_extracts_name_from_string(self):
        self.assertEqual(_extract_model_name("llama3"), "llama3")

    # Test extracts name from mapping.
    def test_extracts_name_from_mapping(self):
        self.assertEqual(_extract_model_name({"model": "qwen"}), "qwen")
        self.assertEqual(_extract_model_name({"name": "gpt-oss"}), "gpt-oss")
        self.assertEqual(_extract_model_name({"model_key": "mistral-nemo"}), "mistral-nemo")

    # Test prefers id over friendly name.
    def test_prefers_id_over_friendly_name(self):
        self.assertEqual(
            _extract_model_name({"id": "openai/gpt-oss-20b", "name": "OpenAI: GPT OSS 20B"}),
            "openai/gpt-oss-20b",
        )


# Attachment normalization tests.

# Cover fast attachment validation helpers.
class AttachmentNormalizationTests(SimpleTestCase):
    # Test invalid base64 attachments are ignored before persistence.
    def test_invalid_base64_attachments_are_ignored(self):
        self.assertEqual(
            _normalize_request_attachments({
                "attachments": [{"name": "bad.txt", "mime_type": "text/plain", "data": "not valid !!!"}],
            }),
            [],
        )

    # Test data URL attachments keep MIME, filename and decoded size.
    def test_data_url_attachments_are_normalized_for_storage(self):
        attachments = _normalize_request_attachments({
            "attachments": [
                {
                    "name": "note.txt",
                    "data_url": "data:text/plain;base64,SGVsbG8=",
                },
            ],
        })

        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]["kind"], MessageAttachmentKind.FILE)
        self.assertEqual(attachments[0]["name"], "note.txt")
        self.assertEqual(attachments[0]["mime_type"], "text/plain")
        self.assertEqual(attachments[0]["data"], "SGVsbG8=")
        self.assertEqual(attachments[0]["size_bytes"], 5)
        self.assertEqual(attachments[0]["order"], 0)

    # Test legacy image payloads are detected and named.
    def test_legacy_image_payloads_are_normalized_with_detected_mime(self):
        attachments = _normalize_request_attachments({
            "images": ["iVBORw0KGgo="],
        })

        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]["kind"], MessageAttachmentKind.IMAGE)
        self.assertEqual(attachments[0]["name"], "image-1")
        self.assertEqual(attachments[0]["mime_type"], "image/png")
        self.assertEqual(attachments[0]["order"], 0)

    # Test empty entries are skipped without breaking later order values.
    def test_attachment_order_uses_surviving_items_only(self):
        attachments = _normalize_request_attachments({
            "attachments": [
                {"name": "bad.txt", "mime_type": "text/plain", "data": ""},
                {"name": "ok.txt", "mime_type": "text/plain", "data": "T0s="},
            ],
        })

        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]["name"], "ok.txt")
        self.assertEqual(attachments[0]["order"], 0)


# Attachment extraction tests.
# Cover prompt text extraction and database caching for stored files.
class AttachmentExtractionTests(TestCase):
    # Cache extracted text back onto the attachment record.
    def test_text_attachment_extraction_is_cached_on_record(self):
        chat = Chat.objects.create(title="Chat")
        message = Message.objects.create(chat=chat, role="user", content="See file")
        attachment = MessageAttachment.objects.create(
            message=message,
            kind=MessageAttachmentKind.FILE,
            name="note.txt",
            mime_type="text/plain",
            data="SGVsbG8gZnJvbSBmaWxl",
            size_bytes=15,
        )
        payload = _serialize_attachment_record(attachment)

        extracted_text = _extract_attachment_text(payload)

        attachment.refresh_from_db()
        self.assertEqual(extracted_text, "Hello from file")
        self.assertTrue(attachment.extracted_text_ready)
        self.assertEqual(attachment.extracted_text, "Hello from file")

    # Reuse cached text without trying to decode a broken payload.
    def test_cached_attachment_text_is_reused(self):
        chat = Chat.objects.create(title="Chat")
        message = Message.objects.create(chat=chat, role="user", content="See file")
        attachment = MessageAttachment.objects.create(
            message=message,
            kind=MessageAttachmentKind.FILE,
            name="note.txt",
            mime_type="text/plain",
            data="not valid !!!",
            extracted_text="Cached text",
            extracted_text_ready=True,
        )
        payload = _serialize_attachment_record(attachment)

        self.assertEqual(_extract_attachment_text(payload), "Cached text")


# Uploaded file manifest tests.

# Cover the standalone manifest builder used by the upload layer.
class UploadedFileManifestTests(SimpleTestCase):
    # Test text files expose bounded previews instead of unbounded content.
    def test_text_manifest_uses_bounded_preview(self):
        content = ("hello\n" * (TEXT_PREVIEW_CHAR_LIMIT // 6 + 100)).encode("utf-8")

        manifest = build_uploaded_file_manifest(
            content,
            name="notes.md",
            mime="text/markdown",
            sandbox_path="/workspace/_sandbox/User/chat/file__notes.md",
            file_id="file-1",
        )

        self.assertEqual(manifest.file_id, "file-1")
        self.assertEqual(manifest.name, "notes.md")
        self.assertEqual(manifest.mime, "text/markdown")
        self.assertTrue(manifest.text_available)
        self.assertTrue(manifest.text_truncated)
        self.assertLessEqual(len(manifest.text_preview or ""), TEXT_PREVIEW_CHAR_LIMIT + 20)
        self.assertEqual(manifest.sandbox_path, "/workspace/_sandbox/User/chat/file__notes.md")
        self.assertIn("sandbox", manifest.recommended_tools)
        self.assertIn("file_search", manifest.recommended_tools)

    # Test binary-looking files do not get decoded through permissive encodings.
    def test_binary_manifest_does_not_expose_text_preview(self):
        payload = b"MZ\x00\x00\x03\x00" + bytes(range(32)) * 8

        manifest = build_uploaded_file_manifest(
            payload,
            name="tool.exe",
            mime="application/octet-stream",
            sandbox_path="/workspace/_sandbox/User/chat/file__tool.exe",
        )

        self.assertFalse(manifest.text_available)
        self.assertIsNone(manifest.text_preview)
        self.assertIsNone(manifest.text_total_chars)
        self.assertEqual(manifest.archive_tree, None)
        self.assertEqual(manifest.recommended_tools, ["sandbox"])

    # Test uploaded names are reduced to safe basenames.
    def test_upload_name_is_normalized_to_basename(self):
        self.assertEqual(normalize_upload_name("../secrets/.env"), ".env")
        self.assertEqual(normalize_upload_name(r"..\..\report.pdf"), "report.pdf")

    # Test zip files include a bounded archive tree without unpacking.
    def test_zip_manifest_includes_archive_tree(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("docs/readme.txt", "hello")
            archive.writestr("src/app.py", "print('ok')")

        manifest = build_uploaded_file_manifest(
            buffer.getvalue(),
            name="bundle.zip",
            mime="application/zip",
            sandbox_path="/workspace/_sandbox/User/chat/file__bundle.zip",
        )

        self.assertEqual(manifest.archive_tree, ["docs/readme.txt", "src/app.py"])
        self.assertIn("archive", manifest.recommended_tools)

    # Test PDF files with a text layer expose a model-readable preview.
    def test_pdf_manifest_extracts_text_layer(self):
        import fitz

        document = fitz.open()
        page = document.new_page()
        page.insert_text((72, 72), "PDF upload text layer")
        payload = document.tobytes()
        document.close()

        manifest = build_uploaded_file_manifest(
            payload,
            name="statement.pdf",
            mime="application/pdf",
        )

        self.assertTrue(manifest.text_available)
        self.assertIn("PDF upload text layer", manifest.text_preview or "")

    # Test docx files expose text from their document XML.
    def test_docx_manifest_extracts_document_xml_text(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "word/document.xml",
                """
                <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
                  <w:body><w:p><w:r><w:t>Word upload text</w:t></w:r></w:p></w:body>
                </w:document>
                """,
            )

        manifest = build_uploaded_file_manifest(
            buffer.getvalue(),
            name="report.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        )

        self.assertTrue(manifest.text_available)
        self.assertIn("Word upload text", manifest.text_preview or "")

    # Test pptx files expose slide text from their slide XML.
    def test_pptx_manifest_extracts_slide_text(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "ppt/slides/slide1.xml",
                """
                <p:sld xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"
                       xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">
                  <p:cSld><p:spTree><p:sp><p:txBody><a:p><a:r><a:t>Slide upload text</a:t></a:r></a:p></p:txBody></p:sp></p:spTree></p:cSld>
                </p:sld>
                """,
            )

        manifest = build_uploaded_file_manifest(
            buffer.getvalue(),
            name="slides.pptx",
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        )

        self.assertTrue(manifest.text_available)
        self.assertIn("Slide upload text", manifest.text_preview or "")

    # Test xlsx files expose a small table preview from worksheet XML.
    def test_xlsx_manifest_extracts_sheet_text(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(
                "xl/sharedStrings.xml",
                """
                <sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                  <si><t>Name</t></si><si><t>Alice</t></si>
                </sst>
                """,
            )
            archive.writestr(
                "xl/worksheets/sheet1.xml",
                """
                <worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
                  <sheetData>
                    <row><c t="s"><v>0</v></c><c><v>42</v></c></row>
                    <row><c t="s"><v>1</v></c><c><v>7</v></c></row>
                  </sheetData>
                </worksheet>
                """,
            )

        manifest = build_uploaded_file_manifest(
            buffer.getvalue(),
            name="sheet.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        self.assertTrue(manifest.text_available)
        self.assertIn("Name | 42", manifest.text_preview or "")
        self.assertIn("Alice | 7", manifest.table_preview or "")

    # Test non-vision image uploads keep metadata and sandbox access only.
    def test_non_vision_image_manifest_keeps_sandbox_without_text(self):
        manifest = build_uploaded_file_manifest(
            b"\x89PNG\r\n\x1a\n",
            name="photo.png",
            mime="image/png",
            sandbox_path="/workspace/_sandbox/User/chat/file__photo.png",
            model_supports_vision=False,
        )

        self.assertFalse(manifest.vision_available)
        self.assertFalse(manifest.text_available)
        self.assertEqual(manifest.sandbox_path, "/workspace/_sandbox/User/chat/file__photo.png")
        self.assertEqual(manifest.recommended_tools, ["sandbox"])


# Upload API tests.

# Cover the public upload contract without exposing model-only manifests.
class UploadFilesApiTests(SimpleTestCase):
    # Isolate sandbox writes in a temporary directory.
    def setUp(self):
        super().setUp()
        self._upload_root_context = tempfile.TemporaryDirectory()
        self._manifest_root_context = tempfile.TemporaryDirectory()
        self.upload_root = Path(self._upload_root_context.name)
        self.manifest_root = Path(self._manifest_root_context.name)
        self.upload_root_patch = patch.object(upload_storage, "USER_UPLOAD_ROOT", self.upload_root)
        self.manifest_root_patch = patch.object(upload_storage, "USER_FILE_MANIFEST_ROOT", self.manifest_root)
        self.upload_root_patch.start()
        self.manifest_root_patch.start()

    # Clean up the temporary sandbox.
    def tearDown(self):
        self.manifest_root_patch.stop()
        self.upload_root_patch.stop()
        self._manifest_root_context.cleanup()
        self._upload_root_context.cleanup()
        super().tearDown()

    # Test the upload API returns only card-safe fields while storing a private manifest.
    def test_upload_api_returns_public_file_card_payload_only(self):
        upload = SimpleUploadedFile("notes.txt", b"Hello from upload", content_type="text/plain")

        response = self.client.post(reverse("uploads_api"), {"files": [upload], "scope": "chat-1"})

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["files"]), 1)
        public_file = payload["files"][0]
        self.assertEqual(public_file["name"], "notes.txt")
        self.assertEqual(public_file["status"], "ready")
        self.assertEqual(public_file["display_kind"], "text")
        self.assertEqual(public_file["type_label"], "Text file")
        self.assertNotIn("sha256", public_file)
        self.assertNotIn("sandbox_path", public_file)
        self.assertNotIn("text_preview", public_file)

        self.assertEqual(list(self.upload_root.glob("chat-1/*.manifest.json")), [])
        self.assertEqual(list(self.upload_root.glob("pending/*.manifest.json")), [])
        manifests = list(self.manifest_root.glob("*/*.manifest.json"))
        self.assertEqual(len(manifests), 1)
        private_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(manifests[0].name, f"{public_file['file_id']}.manifest.json")
        self.assertEqual(private_manifest["text_preview"], "Hello from upload")
        self.assertEqual(manifests[0].parent.name, private_manifest["sha256"])
        self.assertTrue(private_manifest["sandbox_path"].startswith(f"/workspace/_sandbox/User/{private_manifest['sha256']}/"))

    # Test archive uploads get a simple English card label.
    def test_upload_api_labels_zip_archive_for_card(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("readme.txt", "hello")
        upload = SimpleUploadedFile("bundle.zip", buffer.getvalue(), content_type="application/zip")

        response = self.client.post(reverse("uploads_api"), {"files": [upload]})

        self.assertEqual(response.status_code, 200)
        public_file = response.json()["files"][0]
        self.assertEqual(public_file["display_kind"], "archive")
        self.assertEqual(public_file["type_label"], "ZIP archive")

    # Test unusual extensions are accepted and routed as generic files.
    def test_upload_api_accepts_unknown_extension_as_generic_file(self):
        upload = SimpleUploadedFile(
            "sample.abc",
            b"custom binary-ish payload",
            content_type="application/x-abc",
        )

        response = self.client.post(reverse("uploads_api"), {"files": [upload], "scope": "chat-abc"})

        self.assertEqual(response.status_code, 200)
        public_file = response.json()["files"][0]
        self.assertEqual(public_file["name"], "sample.abc")
        self.assertEqual(public_file["status"], "ready")
        self.assertEqual(public_file["display_kind"], "file")
        self.assertEqual(public_file["type_label"], "File")

        self.assertEqual(list(self.upload_root.glob("chat-abc/*.manifest.json")), [])
        self.assertEqual(list(self.upload_root.glob("pending/*.manifest.json")), [])
        manifests = list(self.manifest_root.glob("*/*.manifest.json"))
        self.assertEqual(len(manifests), 1)
        private_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(manifests[0].parent.name, private_manifest["sha256"])
        self.assertEqual(private_manifest["name"], "sample.abc")
        self.assertEqual(private_manifest["mime"], "application/x-abc")
        self.assertFalse(private_manifest["text_available"])
        self.assertTrue(private_manifest["sandbox_path"].startswith(f"/workspace/_sandbox/User/{private_manifest['sha256']}/"))

    # Test the configured upload ceiling matches the advertised large-video contract.
    def test_upload_limit_is_16_gb(self):
        self.assertEqual(upload_storage.MAX_UPLOAD_BYTES, 16 * 1024 * 1024 * 1024)

    # Test uploads beyond the inline manifest threshold are stored without full in-memory extraction.
    def test_upload_api_uses_lightweight_manifest_after_inline_threshold(self):
        upload = SimpleUploadedFile("clip.mp4", b"12345", content_type="video/mp4")

        with patch.object(upload_storage, "INLINE_MANIFEST_MAX_BYTES", 4):
            response = self.client.post(reverse("uploads_api"), {"files": [upload], "scope": "chat-video"})

        self.assertEqual(response.status_code, 200)
        public_file = response.json()["files"][0]
        self.assertEqual(public_file["status"], "ready")
        self.assertEqual(public_file["display_kind"], "video")
        self.assertEqual(public_file["type_label"], "Video")

        manifests = list(self.manifest_root.glob("*/*.manifest.json"))
        self.assertEqual(len(manifests), 1)
        private_manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        self.assertEqual(private_manifest["name"], "clip.mp4")
        self.assertEqual(private_manifest["mime"], "video/mp4")
        self.assertEqual(private_manifest["size_bytes"], 5)
        self.assertFalse(private_manifest["text_available"])
        self.assertIsNone(private_manifest["text_preview"])
        stored_files = [path for path in self.upload_root.glob("*/*") if path.is_file()]
        self.assertEqual(len(stored_files), 1)
        self.assertEqual(stored_files[0].read_bytes(), b"12345")

    # Test oversize uploads are rejected before being stored.
    def test_upload_api_reports_oversized_files(self):
        upload = SimpleUploadedFile("too-big.mp4", b"12345", content_type="video/mp4")

        with patch.object(upload_storage, "MAX_UPLOAD_BYTES", 4):
            response = self.client.post(reverse("uploads_api"), {"files": [upload]})

        self.assertEqual(response.status_code, 200)
        public_file = response.json()["files"][0]
        self.assertEqual(public_file["status"], "error")
        self.assertIn("File is too large", public_file["error"])
        self.assertEqual(list(self.manifest_root.glob("*/*.manifest.json")), [])

    # Test media content endpoint supports suffix ranges needed by MP4 metadata reads.
    def test_uploaded_file_content_supports_suffix_byte_range(self):
        upload = SimpleUploadedFile("clip.mp4", b"0123456789", content_type="video/mp4")
        upload_response = self.client.post(reverse("uploads_api"), {"files": [upload]})
        content_url = upload_response.json()["files"][0]["content_url"]
        stored_file = next(path for path in self.upload_root.glob("*/*") if path.is_file())

        with patch("Apps.UI.views._resolve_uploaded_file_content_path", return_value=stored_file):
            response = self.client.get(content_url, HTTP_RANGE="bytes=-4")

        self.assertEqual(response.status_code, 206)
        self.assertEqual(response["Content-Range"], "bytes 6-9/10")
        self.assertEqual(b"".join(response.streaming_content), b"6789")

    # Test open-ended media ranges are chunked so playback can start without reading the rest of a large file.
    def test_uploaded_file_content_chunks_open_ended_range(self):
        upload = SimpleUploadedFile("clip.mp4", b"0123456789", content_type="video/mp4")
        upload_response = self.client.post(reverse("uploads_api"), {"files": [upload]})
        content_url = upload_response.json()["files"][0]["content_url"]
        stored_file = next(path for path in self.upload_root.glob("*/*") if path.is_file())

        with (
            patch("Apps.UI.views.MEDIA_RANGE_CHUNK_BYTES", 4),
            patch("Apps.UI.views._resolve_uploaded_file_content_path", return_value=stored_file),
        ):
            response = self.client.get(content_url, HTTP_RANGE="bytes=2-")

        self.assertEqual(response.status_code, 206)
        self.assertEqual(response["Content-Range"], "bytes 2-5/10")
        self.assertEqual(b"".join(response.streaming_content), b"2345")

    # Test model-shared files use the same range streaming path as uploaded files.
    def test_shared_file_download_supports_byte_range(self):
        with tempfile.NamedTemporaryFile(delete=False) as handle:
            handle.write(b"abcdefghij")
            temp_path = Path(handle.name)
        try:
            with patch("Apps.UI.views._resolve_shared_file_path", return_value=temp_path):
                response = self.client.get(
                    reverse("shared_file_download_api"),
                    {"path": str(temp_path), "preview": "1"},
                    HTTP_RANGE="bytes=-3",
                )

            self.assertEqual(response.status_code, 206)
            self.assertEqual(response["Content-Range"], "bytes 7-9/10")
            self.assertEqual(b"".join(response.streaming_content), b"hij")
        finally:
            temp_path.unlink(missing_ok=True)

    # Test reveal endpoint applies path validation before opening the file manager.
    @patch("Apps.UI.views._reveal_file_in_file_manager")
    @patch("Apps.UI.views._resolve_shared_file_path")
    def test_shared_file_reveal_opens_validated_target(self, resolve_mock, reveal_mock):
        target = Path("C:/sandbox/User/report.svg")
        resolve_mock.return_value = target

        response = self.client.post(
            reverse("shared_file_reveal_api"),
            data=json.dumps({"path": "/workspace/_sandbox/User/report.svg"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"ok": True})
        resolve_mock.assert_called_once_with("/workspace/_sandbox/User/report.svg")
        reveal_mock.assert_called_once_with(target)

    # Test missing or forbidden shared files are never sent to the file manager.
    @patch("Apps.UI.views._reveal_file_in_file_manager")
    @patch("Apps.UI.views._resolve_shared_file_path", side_effect=FileNotFoundError)
    def test_shared_file_reveal_rejects_invalid_target(self, _resolve_mock, reveal_mock):
        response = self.client.post(
            reverse("shared_file_reveal_api"),
            data=json.dumps({"path": "C:/outside/private.txt"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 404)
        reveal_mock.assert_not_called()

    # Test Windows reveal uses Explorer's select-file argument.
    @patch("Apps.UI.views.subprocess.Popen")
    def test_reveal_file_manager_selects_file_on_windows(self, popen_mock):
        target = Path("C:/sandbox/User/report.svg")
        with patch("Apps.UI.views.sys.platform", "win32"):
            _reveal_file_in_file_manager(target)

        popen_mock.assert_called_once_with(
            ["explorer.exe", f"/select,{target}"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
        )

    # Test dynamic shared images render inline while the card link remains a download.
    def test_shared_file_dynamic_image_preview_is_inline(self):
        animated_gif = bytes.fromhex(
            "47494638396101000100800000000000ffffff"
            "21ff0b4e45545343415045322e300301000000"
            "21f904000a0000002c0000000001000100000202440100"
            "21f904000a0000002c0000000001000100000202440100"
            "3b"
        )
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as handle:
            handle.write(animated_gif)
            temp_path = Path(handle.name)
        try:
            with patch("Apps.UI.views._resolve_shared_file_path", return_value=temp_path):
                preview_response = self.client.get(
                    reverse("shared_file_download_api"),
                    {"path": str(temp_path), "name": "animated.gif", "preview": "1"},
                )
                download_response = self.client.get(
                    reverse("shared_file_download_api"),
                    {"path": str(temp_path), "name": "animated.gif"},
                )

            self.assertEqual(preview_response.status_code, 200)
            self.assertEqual(preview_response["Content-Type"], "image/gif")
            self.assertTrue(preview_response["Content-Disposition"].startswith("inline;"))
            self.assertEqual(b"".join(preview_response.streaming_content), animated_gif)
            self.assertTrue(download_response["Content-Disposition"].startswith("attachment;"))
            download_response.close()
        finally:
            temp_path.unlink(missing_ok=True)

    # Test shared-file downloads are limited to the sandbox workspace.
    def test_shared_file_download_rejects_project_absolute_path(self):
        response = self.client.get(
            reverse("shared_file_download_api"),
            {"path": str(Path(__file__).resolve())},
        )

        self.assertEqual(response.status_code, 404)

    # Test container-style sandbox paths are still mapped to the host sandbox.
    def test_shared_file_download_allows_container_sandbox_path(self):
        sandbox_file = Path("Tools/mcp-sandbox/_sandbox/User/shared-test.txt")
        sandbox_file.parent.mkdir(parents=True, exist_ok=True)
        sandbox_file.write_text("shared ok", encoding="utf-8")
        try:
            response = self.client.get(
                reverse("shared_file_download_api"),
                {"path": "/workspace/_sandbox/User/shared-test.txt"},
            )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(b"".join(response.streaming_content), b"shared ok")
        finally:
            sandbox_file.unlink(missing_ok=True)

    # Test empty upload requests fail before returning a card payload.
    def test_upload_api_requires_files(self):
        response = self.client.post(reverse("uploads_api"), {})

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "No files uploaded")

    # Test model-facing upload manifests do not expose sandbox paths unless selected.
    def test_model_upload_manifest_respects_sandbox_selection(self):
        upload = SimpleUploadedFile("notes.txt", b"Hello from upload", content_type="text/plain")
        response = self.client.post(reverse("uploads_api"), {"files": [upload], "scope": "chat-1"})
        file_id = response.json()["files"][0]["file_id"]

        without_sandbox = _load_model_upload_manifests([file_id], sandbox_enabled=False)[0]
        with_sandbox = _load_model_upload_manifests([file_id], sandbox_enabled=True)[0]

        self.assertIsNone(without_sandbox["sandbox_path"])
        self.assertNotIn("sandbox", without_sandbox["recommended_tools"])
        self.assertTrue(with_sandbox["sandbox_path"].startswith(f"/workspace/_sandbox/User/{with_sandbox['sha256']}/"))
        self.assertIn("sandbox", with_sandbox["recommended_tools"])

    # JSON uploads are sent as complete text context rather than a short preview.
    def test_uploaded_json_is_added_to_model_context(self):
        json_payload = json.dumps({"items": list(range(8000)), "tail": "context-tail"})
        upload = SimpleUploadedFile("payload.json", json_payload.encode("utf-8"), content_type="application/json")
        response = self.client.post(reverse("uploads_api"), {"files": [upload]})
        file_id = response.json()["files"][0]["file_id"]
        manifest = _load_model_upload_manifests([file_id], sandbox_enabled=False)[0]

        entry = _apply_uploaded_file_manifests_to_llm_entry(
            {"role": "user", "content": "Inspect this JSON"},
            [manifest],
        )

        self.assertIn("JSON content:\n```json", entry["content"])
        self.assertIn('"tail": "context-tail"', entry["content"])
        self.assertNotIn("Text preview:", entry["content"])

    # Uploaded media bytes are attached only for model-supported modalities.
    def test_uploaded_media_is_gated_by_model_capabilities(self):
        upload = SimpleUploadedFile("voice.mp3", b"fake-mp3-bytes", content_type="audio/mpeg")
        response = self.client.post(reverse("uploads_api"), {"files": [upload]})
        file_id = response.json()["files"][0]["file_id"]
        manifest = _load_model_upload_manifests([file_id], sandbox_enabled=False)[0]

        unsupported_entry = _apply_uploaded_file_manifests_to_llm_entry(
            {"role": "user", "content": "Listen"},
            [manifest],
            supported_media_kinds={"image"},
        )
        supported_entry = _apply_uploaded_file_manifests_to_llm_entry(
            {"role": "user", "content": "Listen"},
            [manifest],
            supported_media_kinds={"audio"},
        )

        self.assertNotIn("media", unsupported_entry)
        self.assertEqual(supported_entry["media"][0]["kind"], "audio")
        self.assertEqual(supported_entry["media"][0]["data"], "ZmFrZS1tcDMtYnl0ZXM=")

    # Test the private prompt block only includes sandbox path when allowed.
    def test_uploaded_file_prompt_block_hides_disabled_sandbox_path(self):
        manifest = {
            "file_id": "file-1",
            "name": "notes.txt",
            "mime": "text/plain",
            "size_bytes": 5,
            "sandbox_path": None,
            "text_preview": "Hello",
            "archive_tree": None,
            "table_preview": None,
        }

        block = _build_uploaded_file_prompt_block(manifest)

        self.assertIn("[Uploaded file: notes.txt]", block)
        self.assertIn("Text preview:\nHello", block)
        self.assertNotIn("Sandbox path:", block)

    # Verify uploaded archive prompt block says preview not extracted.
    def test_uploaded_archive_prompt_block_says_preview_not_extracted(self):
        manifest = {
            "file_id": "file-zip",
            "name": "bundle.zip",
            "mime": "application/zip",
            "size_bytes": 123,
            "sandbox_path": "/workspace/_sandbox/User/chat/file__bundle.zip",
            "text_preview": None,
            "archive_tree": ["bundle/", "bundle/manage.py"],
            "table_preview": None,
        }

        block = _build_uploaded_file_prompt_block(manifest)

        self.assertIn("Archive preview", block)
        self.assertIn("has not been extracted", block)
        self.assertNotIn("Archive tree:", block)
        self.assertIn("- bundle/manage.py", block)

    # Test upload file ids can be read from current and future request shapes.
    def test_uploaded_file_ids_are_normalized_from_request_shapes(self):
        self.assertEqual(
            _normalize_uploaded_file_ids({
                "uploaded_file_ids": ["a", "b", "a"],
                "attachments": [{"file_id": "c"}, {"name": "legacy.txt"}],
            }),
            ["a", "b", "c"],
        )

    # Test upload ids can be persisted on a user message for regenerate/history replay.
    def test_uploaded_file_context_entry_round_trips_file_ids(self):
        entry = _build_uploaded_file_context_entry(["file-1", "file-2", "file-1"])
        message = Message(role="user", content="read this", llm_transcript=[entry])

        self.assertEqual(entry["type"], "uploaded_file_context")
        self.assertEqual(_extract_uploaded_file_ids_from_message(message), ["file-1", "file-2"])

    # Test sandbox state is derived only from resolved tool servers.
    def test_selected_tools_include_sandbox_only_when_resolved(self):
        self.assertTrue(_selected_tools_include_sandbox([{"id": "sandbox"}]))
        self.assertFalse(_selected_tools_include_sandbox([{"id": "other"}]))
        self.assertFalse(_selected_tools_include_sandbox([]))


# Upload routing tests.

# Cover file type classification used by public upload cards.
class UploadRoutingTests(SimpleTestCase):
    # Test routing common file types to stable card labels.
    def test_display_kind_routes_known_file_types(self):
        cases = [
            ("photo.png", "image/png", ("image", "Image")),
            ("notes.md", "text/markdown", ("text", "Text file")),
            ("script.py", "text/x-python", ("code", "Code file")),
            ("report.pdf", "application/pdf", ("document", "PDF document")),
            ("sheet.csv", "text/csv", ("table", "CSV table")),
            ("bundle.zip", "application/zip", ("archive", "ZIP archive")),
            ("voice_note.mp3", "audio/mpeg", ("audio", "Audio")),
            ("demo_clip.mp4", "video/mp4", ("video", "Video")),
            ("slides.pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", ("presentation", "PowerPoint presentation")),
        ]

        for name, mime, expected in cases:
            with self.subTest(name=name):
                self.assertEqual(display_kind_for_upload(name, mime), expected)

    # Test unknown extensions fall back to generic File, not rejection.
    def test_display_kind_routes_unknown_extension_to_file(self):
        self.assertEqual(display_kind_for_upload("mystery.abc", "application/x-abc"), ("file", "File"))
        self.assertEqual(display_kind_for_upload("no-extension", "application/octet-stream"), ("file", "File"))


# View and runtime mapping tests.

# Verify per-process static cache busting for templates and ES modules.
class StaticCacheVersionTests(SimpleTestCase):
    # Verify static cache version format.
    def test_static_cache_version_format(self):
        from Apps.UI import STATIC_CACHE_VERSION

        self.assertRegex(STATIC_CACHE_VERSION, r"^\d{14}$")

    # Verify static template tag appends the cache-bust query.
    def test_static_template_tag_appends_cache_bust_query(self):
        from Apps.UI import STATIC_CACHE_VERSION
        from django.template import Context, Template

        rendered = Template(
            "{% load i18n_tags %}{% static 'css/main/main.css' %}"
        ).render(Context({}))
        self.assertEqual(rendered, f"/static/css/main/main.css?v={STATIC_CACHE_VERSION}")

    # Verify every local ES module is mapped to the same backend-generated key.
    def test_static_import_map_versions_the_complete_module_graph(self):
        import json

        from Apps.UI import STATIC_CACHE_VERSION
        from django.template import Context, Template

        rendered = Template(
            "{% load i18n_tags %}{% static_import_map %}"
        ).render(Context({}))
        imports = json.loads(rendered)["imports"]

        self.assertGreater(len(imports), 20)
        self.assertEqual(
            imports["/static/js/ui/deep-research-ui.js"],
            f"/static/js/ui/deep-research-ui.js?v={STATIC_CACHE_VERSION}",
        )
        self.assertEqual(
            imports["/static/js/main/api.js"],
            f"/static/js/main/api.js?v={STATIC_CACHE_VERSION}",
        )
        self.assertTrue(
            all(target == f"{source}?v={STATIC_CACHE_VERSION}" for source, target in imports.items())
        )

# Verify that the main page uses the configured engine and local server helpers.
class MainViewTests(ToolRegistryTestMixin, TestCase):
    # Test main view includes runtime settings and local servers.
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify main view includes runtime settings and local servers.
    def test_main_view_includes_runtime_settings_and_local_servers(self, _mock_engine):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite", "description": "Time helpers"}
            TOOLS = [{"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}}]
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )

        response = self.client.get(reverse("main"))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context["models"], [])
        self.assertEqual(response.context["llm_engine"], "ollama-service")
        self.assertIn("runtime_settings", response.context)
        self.assertEqual(
            response.context["available_tool_servers"],
            [{
                "id": "time_suite",
                "name": "Time Suite",
                "description": "Time helpers",
                "tool_count": 1,
                "tools": [{"id": "time_now", "name": "Current Time", "description": ""}],
            }],
        )
        self.assertContains(response, 'id="group-load"')
        self.assertContains(response, 'type="importmap"')
        self.assertRegex(
            response.content.decode("utf-8"),
            r'"/static/js/ui/deep-research-ui\.js":"/static/js/ui/deep-research-ui\.js\?v=\d{14}"',
        )
        self.assertRegex(
            response.content.decode("utf-8"),
            r"/static/js/main/main\.js\?v=\d{14}",
        )
        self.assertNotRegex(
            response.content.decode("utf-8"),
            r"/static/[^\"']+\?v=\d{14}\?v=",
        )
        self.assertNotContains(response, "cdn.jsdelivr.net")
        self.assertContains(response, "/static/css/vendor/katex.min.css?v=")
        self.assertContains(response, "/static/js/vendor/katex.min.js?v=")
        self.assertContains(response, "/static/js/vendor/mermaid.min.js?v=")
        self.assertRegex(
            response.content.decode("utf-8"),
            r"/static/img/ui/tools/code\.svg\?v=\d{14}",
        )
        self.assertRegex(
            response.content.decode("utf-8"),
            r"/static/img/ui/copy\.svg\?v=\d{14}",
        )
        self.assertRegex(
            response.content.decode("utf-8"),
            r"/static/img/ui/refresh\.svg\?v=\d{14}",
        )
        self.assertRegex(
            response.content.decode("utf-8"),
            r"/static/img/ui/check\.svg\?v=\d{14}",
        )
        self.assertNotContains(response, "/profile/")
        self.assertNotContains(response, "account-btn")


# Ensure Ollama-only thinking parameters are normalized before request dispatch.
class OllamaOptionMappingTests(SimpleTestCase):
    # Test prepare chat kwargs maps think level into think.
    def test_prepare_chat_kwargs_maps_think_level_into_think(self):
        payload = _prepare_chat_kwargs(
            {
                "stream": True,
                "think": True,
                "think_level": "high",
                "options": {"temperature": 0.7},
            }
        )

        self.assertNotIn("think_level", payload)
        self.assertEqual(payload["think"], "high")
        self.assertEqual(payload["options"]["temperature"], 0.7)

    # Test prepare chat kwargs drops runtime options unsupported by current Ollama.
    def test_prepare_chat_kwargs_drops_runtime_options_unsupported_by_current_ollama(self):
        payload = _prepare_chat_kwargs(
            {
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 32768,
                    "mirostat": 2,
                    "numa": False,
                    "tfs_z": 1.0,
                }
            }
        )

        self.assertEqual(payload["options"]["temperature"], 0.7)
        self.assertEqual(payload["options"]["num_ctx"], 32768)
        self.assertNotIn("mirostat", payload["options"])
        self.assertNotIn("numa", payload["options"])
        self.assertNotIn("tfs_z", payload["options"])

    # Test prepare chat kwargs ignores LM Studio only internal keys.
    def test_prepare_chat_kwargs_ignores_lms_only_internal_keys(self):
        payload = _prepare_chat_kwargs(
            {
                "stream": True,
                "think": True,
                "think_param_name": "ext.virtualModel.customField.qwen.enableThinking",
                "think_level_param_name": "ext.virtualModel.customField.openai.reasoningEffort",
                "load_config": {"contextLength": 32768},
                "sync_operation_defaults": {"ext.virtualModel.customField.qwen.enableThinking": False},
                "options": {"temperature": 0.7},
            }
        )

        self.assertNotIn("think_param_name", payload)
        self.assertNotIn("think_level_param_name", payload)
        self.assertNotIn("load_config", payload)
        self.assertNotIn("sync_operation_defaults", payload)
        self.assertEqual(payload["think"], True)
        self.assertEqual(payload["options"]["temperature"], 0.7)

    # Test prepare runtime passes requested engine to managed service.
    @patch("API.ollama._get_ollama_service_module")
    # Verify prepare runtime passes requested engine to managed service.
    def test_prepare_runtime_passes_requested_engine_to_managed_service(self, mock_get_service):
        mock_service = Mock()
        mock_get_service.return_value = mock_service

        prepare_ollama_runtime("ollama-service")

        mock_service.start_ollama.assert_called_once_with(engine="ollama-service")

    # Test the fixed local Ollama client never trusts system proxy settings.
    @patch("API.ollama.ollama.Client")
    @patch("API.ollama.settings.get", return_value=20003)
    def test_ollama_client_disables_environment_proxy(self, _mock_port, mock_client):
        mock_client.return_value = Mock()

        ollama_api.get_client()

        mock_client.assert_called_once_with(
            host="http://127.0.0.1:20003",
            trust_env=False,
        )


# Ensure Ollama tool support follows Ollama model metadata.
class OllamaModelInfoTests(SimpleTestCase):
    # Test an explicit Ollama capabilities list without tools disables tool support.
    def test_ollama_capabilities_without_tools_disable_tool_support(self):
        payload = _extract_ollama_model_info({
            "capabilities": ["completion"],
            "template": "{{ if .Tools }}tools{{ end }}{{ if .ToolCalls }}calls{{ end }}",
        })

        self.assertFalse(payload["supports_tool_calling"])

    # Test Ollama's tools capability enables support without template markers.
    def test_ollama_tools_capability_enables_tool_support(self):
        payload = _extract_ollama_model_info({
            "capabilities": ["completion", "tools"],
            "template": "{{ if .Messages }}{{ end }}",
        })

        self.assertTrue(payload["supports_tool_calling"])

    # Test old/custom Ollama responses can still infer tools from the template.
    def test_ollama_tool_template_fallback_when_capabilities_are_missing(self):
        payload = _extract_ollama_model_info({
            "template": "{{ if .Messages }}{{ end }}",
        })

        self.assertFalse(payload["supports_tool_calling"])

        payload = _extract_ollama_model_info({
            "template": "{{ if .Tools }}tools{{ end }}{{ if .ToolCalls }}calls{{ end }}",
        })

        self.assertTrue(payload["supports_tool_calling"])


# Ensure generic runtime options are safely mapped for OpenAI-compatible APIs.
class OpenAiOptionMappingTests(SimpleTestCase):
    # Test maps supported options and keeps custom values in extra body.
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

    # Test top logprobs is dropped unless logprobs output is enabled.
    def test_top_logprobs_requires_logprobs(self):
        disabled_payload = _build_openai_request_options({"top_logprobs": 5})
        enabled_payload = _build_openai_request_options({"logprobs": True, "top_logprobs": 5})

        self.assertNotIn("top_logprobs", disabled_payload)
        self.assertEqual(enabled_payload["top_logprobs"], 5)

    # Test OpenAI client uses placeholder API key when not configured.
    @patch("openai.OpenAI")
    @patch("API.openai.settings.get_engine_url", return_value="http://127.0.0.1:1234/v1")
    @patch("API.openai.settings.get_openai_api_key", return_value="")
    # Verify openai client uses placeholder api key when not configured.
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
    # OpenAI-compatible messages encode supported audio as native input_audio.
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

    # Test get model settings reads OpenAI capabilities and reasoning.
    @patch("API.openai._get_client")
    # Verify get model settings reads openai capabilities and reasoning.
    def test_get_model_settings_reads_openai_capabilities_and_reasoning(self, mock_get_client):
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

        payload = get_openai_model_settings("gpt-test")

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

    # Test get model settings reads direct feature flags and scalar supported parameters.
    @patch("API.openai._get_client")
    # Verify get model settings reads direct feature flags and scalar supported parameters.
    def test_get_model_settings_reads_direct_feature_flags_and_scalar_supported_parameters(self, mock_get_client):
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

        payload = get_openai_model_settings("gpt-test")

        self.assertTrue(payload["supports_tool_calling"])
        self.assertTrue(payload["supports_vision"])
        self.assertTrue(payload["supports_audio_input"])
        self.assertTrue(payload["supports_video_input"])
        self.assertTrue(payload["supports_thinking"])
        self.assertTrue(payload["supports_think_level"])
        self.assertFalse(payload["supports_think_toggle"])
        self.assertIn("reasoning_effort", payload["supported_parameters"])

    # Test generate stream parses reasoning and visible content.
    @patch("API.openai._get_client")
    # Verify generate stream parses reasoning and visible content.
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

    # Test generate stream does not duplicate plain content into thinking.
    @patch("API.openai._get_client")
    # Verify generate stream does not duplicate plain content into thinking.
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

    # Test get model settings reads companion metadata without generation.
    @patch("API.openai._get_companion_model_payload")
    @patch("API.openai._get_client")
    # Verify get model settings reads companion metadata without generation.
    def test_get_model_settings_reads_companion_metadata_without_generation(
        self,
        mock_get_client,
        mock_get_companion_payload,
    ):
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

        payload = get_openai_model_settings("gpt-probe")

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


# Cover generic engine registry behavior for optional capabilities.
class EngineRegistryTests(SimpleTestCase):
    # Test title requests use a bounded source, context, and output.
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

    # Test a thought-only instant attempt retries with a bounded reasoning budget.
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

    # Test reload model raises for engines without reload support.
    def test_reload_model_raises_for_engines_without_reload_support(self):
        with self.assertRaises(NotImplementedError):
            llm_api.reload_model("openai", "gpt-oss")

    # Test get models prepares runtime before listing.
    @patch("API.llm_api.prepare_runtime")
    @patch("API.llm_api._get_engine_module")
    # Verify get models prepares runtime before listing.
    def test_get_models_prepares_runtime_before_listing(self, mock_get_engine_module, mock_prepare_runtime):
        mock_module = Mock()
        mock_module.get_models.return_value = ["llama3"]
        mock_get_engine_module.return_value = mock_module

        self.assertEqual(llm_api.get_models("ollama-service"), ["llama3"])
        mock_prepare_runtime.assert_called_once_with("ollama-service")

    # Test get model settings prepares runtime before loading metadata.
    @patch("API.llm_api.prepare_runtime")
    @patch("API.llm_api._get_engine_module")
    # Verify get model settings prepares runtime before loading metadata.
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


# Verify loopback proxy bypass configuration and transport behavior.
class LoopbackProxyPolicyTests(SimpleTestCase):
    # Start a small HTTP server that identifies which endpoint received a request.
    def _start_http_server(self, label):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                body = label.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread, requests

    # Stop one test HTTP server and wait for its thread to exit.
    def _stop_http_server(self, server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    def test_policy_merges_existing_bypass_and_loopback_hosts_idempotently(self):
        environ = {
            "NO_PROXY": "corp.example;127.0.0.1;[::1]",
            "no_proxy": "internal.example,LOCALHOST",
        }

        first = apply_loopback_proxy_bypass(
            environ,
            system_proxies={},
            system_bypass=[],
        )
        second = apply_loopback_proxy_bypass(
            environ,
            system_proxies={},
            system_bypass=[],
        )

        self.assertEqual(first, second)
        self.assertEqual(environ["NO_PROXY"], environ["no_proxy"])
        tokens = {token.casefold() for token in first.split(",")}
        self.assertEqual(
            tokens,
            {
                "corp.example",
                "internal.example",
                "localhost",
                "127.0.0.1",
                "::1",
            },
        )

    def test_policy_preserves_external_system_proxy_and_native_bypass(self):
        environ = {}

        apply_loopback_proxy_bypass(
            environ,
            system_proxies={
                "http": "http://proxy.example:8080",
                "https": "http://proxy.example:8080",
            },
            system_bypass=["<local>;corp.internal"],
        )

        self.assertEqual(environ["HTTP_PROXY"], "http://proxy.example:8080")
        self.assertEqual(environ["http_proxy"], "http://proxy.example:8080")
        self.assertEqual(environ["HTTPS_PROXY"], "http://proxy.example:8080")
        self.assertIn("corp.internal", environ["NO_PROXY"])
        self.assertIn("localhost", environ["NO_PROXY"])

    def test_policy_does_not_replace_explicit_empty_proxy_or_wildcard(self):
        environ = {"HTTP_PROXY": "", "NO_PROXY": "*"}

        merged = apply_loopback_proxy_bypass(
            environ,
            system_proxies={"http": "http://system-proxy.example:8080"},
            system_bypass=[],
        )

        self.assertEqual(environ["HTTP_PROXY"], "")
        self.assertNotIn("http_proxy", environ)
        self.assertEqual(merged, "*")
        self.assertEqual(environ["no_proxy"], "*")

    def test_ipv6_loopback_builds_valid_httpx_bypass_pattern(self):
        import httpx._utils

        proxy_url = "http://proxy.example:8080"
        with patch.dict(
            os.environ,
            {"HTTP_PROXY": proxy_url, "HTTPS_PROXY": proxy_url},
            clear=True,
        ):
            apply_loopback_proxy_bypass(system_proxies={}, system_bypass=[])
            mounts = httpx._utils.get_environment_proxies()

            self.assertIsNone(mounts["all://[::1]"])

    def test_user_process_overlay_keeps_custom_env_and_only_parent_proxy_keys(self):
        parent_environment = {
            "HTTP_PROXY": "http://parent-proxy.example:8080",
            "HTTPS_PROXY": "http://parent-proxy.example:8080",
            "NO_PROXY": "corp.example",
            "UNRELATED_SECRET": "must-not-leak",
        }
        with patch.dict(os.environ, parent_environment, clear=True):
            overlay = build_proxy_environment_overlay(
                {
                    "CUSTOM_VALUE": "kept",
                    "HTTPS_PROXY": "http://custom-proxy.example:8443",
                    "NO_PROXY": "mcp.internal",
                }
            )

        self.assertEqual(overlay["CUSTOM_VALUE"], "kept")
        self.assertEqual(overlay["HTTP_PROXY"], "http://parent-proxy.example:8080")
        self.assertEqual(overlay["HTTPS_PROXY"], "http://custom-proxy.example:8443")
        self.assertNotIn("UNRELATED_SECRET", overlay)
        self.assertIn("corp.example", overlay["NO_PROXY"])
        self.assertIn("mcp.internal", overlay["NO_PROXY"])
        self.assertIn("::1", overlay["NO_PROXY"])

    def test_httpx_bypasses_loopback_but_keeps_proxy_for_external_urls(self):
        import httpx

        origin, origin_thread, origin_requests = self._start_http_server("origin")
        proxy, proxy_thread, proxy_requests = self._start_http_server("proxy")
        origin_port = origin.server_address[1]
        proxy_port = proxy.server_address[1]

        try:
            proxy_url = f"http://127.0.0.1:{proxy_port}"
            with patch.dict(
                os.environ,
                {
                    "HTTP_PROXY": proxy_url,
                    "HTTPS_PROXY": proxy_url,
                    "ALL_PROXY": proxy_url,
                },
                clear=True,
            ):
                apply_loopback_proxy_bypass(system_proxies={}, system_bypass=[])
                with httpx.Client(timeout=3) as client:
                    by_ip = client.get(f"http://127.0.0.1:{origin_port}/by-ip")
                    by_name = client.get(f"http://localhost:{origin_port}/by-name")
                    external = client.get("http://example.invalid/through-proxy")

                with urlopen_direct(f"http://127.0.0.1:{origin_port}/urllib", timeout=3) as response:
                    direct_body = response.read().decode("utf-8")

                with urlopen_with_loopback_bypass(
                    f"http://localhost:{origin_port}/urllib-policy",
                    timeout=3,
                ) as response:
                    policy_local_body = response.read().decode("utf-8")

                with urlopen_with_loopback_bypass(
                    "http://example.invalid/urllib-external",
                    timeout=3,
                ) as response:
                    policy_external_body = response.read().decode("utf-8")

            self.assertEqual(by_ip.text, "origin")
            self.assertEqual(by_name.text, "origin")
            self.assertEqual(external.text, "proxy")
            self.assertEqual(direct_body, "origin")
            self.assertEqual(policy_local_body, "origin")
            self.assertEqual(policy_external_body, "proxy")
            self.assertEqual(
                origin_requests,
                ["/by-ip", "/by-name", "/urllib", "/urllib-policy"],
            )
            self.assertEqual(
                proxy_requests,
                [
                    "http://example.invalid/through-proxy",
                    "http://example.invalid/urllib-external",
                ],
            )
        finally:
            self._stop_http_server(origin, origin_thread)
            self._stop_http_server(proxy, proxy_thread)


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


# Keep user-visible LM output clean and actionable.
class ViewFormattingTests(SimpleTestCase):
    # Test strip LLM control tokens removes service markers.
    def test_strip_llm_control_tokens_removes_service_markers(self):
        self.assertEqual(
            _strip_llm_control_tokens("<|start|>assistant<|channel|>final<|message|>Hello"),
            "Hello",
        )

    # Test format runtime error hides LM Studio model load verbosity.
    def test_format_runtime_error_hides_lms_model_load_verbosity(self):
        message = _format_runtime_error(
            "lms",
            RuntimeError(
                "Model get/load error: V Cache Quantization requires flash attention to be enabled."
            ),
        )

        self.assertIn("Flash Attention", message)
        self.assertNotIn("Model get/load error", message)

    # Test chat titles are compact and useful for attachment-only threads.
    def test_build_chat_title_handles_long_and_attachment_only_messages(self):
        self.assertEqual(_build_chat_title("Short title", False), "Short title")
        self.assertEqual(_build_chat_title("x" * 31, False), f"{'x' * 30}...")
        self.assertEqual(_build_chat_title("", True), "Attachment chat")
        self.assertEqual(_build_chat_title("", False), "New Chat")

    # Test active tool slugs support both current JSON and legacy string shapes.
    def test_parse_active_tool_slugs_supports_json_and_legacy_values(self):
        self.assertEqual(_parse_active_tool_slugs('["time_suite", "", "browser"]'), ["time_suite", "browser"])
        self.assertEqual(_parse_active_tool_slugs("time_suite"), ["time_suite"])
        self.assertEqual(_parse_active_tool_slugs(""), [])

    # Test shared files keep their UI render payload after tool result splitting.
    def test_shared_file_tool_result_keeps_ui_metadata(self):
        payload = {
            "kind": "shared_file",
            "path": "/workspace/_sandbox/wave_graph.svg",
            "host_path": str(Path("C:/tmp/oda/wave_graph.svg")),
            "filename": "wave_graph.svg",
            "mime_type": "image/svg+xml",
            "size_bytes": 123,
            "render": {
                "type": "image",
                "mime_type": "image/svg+xml",
                "preview": {"kind": "base64", "data": "abc"},
            },
        }

        model_text, extras = tool_registry.split_tool_result_payload(payload)

        self.assertEqual(model_text, "Shared file ready for download: wave_graph.svg")
        self.assertEqual(extras["structured_content"]["kind"], "shared_file")
        self.assertEqual(extras["structured_content"]["file"]["render"]["type"], "image")
        self.assertIn("/api/shared-file/download/?", extras["structured_content"]["file"]["download_url"])
        self.assertEqual(extras["tool_ui"]["kind"], "shared_file")

    def test_cancelled_tool_result_is_stable_and_tells_model_not_to_retry(self):
        model_text, extras = tool_registry.split_tool_result_payload(
            tool_registry.TOOL_EXECUTION_CANCELLED
        )

        self.assertTrue(model_text.startswith("TOOL_EXECUTION_CANCELLED:"))
        self.assertIn("Do not retry", model_text)
        self.assertNotIn("object at 0x", model_text)
        self.assertEqual(extras["tool_ui"]["status"], "cancelled")
        self.assertTrue(tool_registry.is_blocking_tool_result(tool_registry.TOOL_EXECUTION_CANCELLED))

    # Test repeated tool aliases preserve all shared files in activity segments.
    def test_build_activity_segments_keeps_repeated_share_file_aliases(self):
        class _MessageStub:
            llm_transcript = [
                {
                    "role": "tool",
                    "alias": "sandbox__share_file__0",
                    "tool_id": "share_file",
                    "tool_display_name": "Share File",
                    "arguments": {"path": "a.txt", "filename": "a.txt"},
                    "content": "Shared file ready for download: a.txt",
                    "structured_content": {
                        "kind": "shared_file",
                        "file": {"kind": "shared_file", "path": "a.txt", "filename": "a.txt"},
                    },
                    "tool_ui": {
                        "kind": "shared_file",
                        "status": "done",
                        "file": {"kind": "shared_file", "path": "a.txt", "filename": "a.txt"},
                    },
                },
                {
                    "role": "tool",
                    "alias": "sandbox__share_file__0",
                    "tool_id": "share_file",
                    "tool_display_name": "Share File",
                    "arguments": {"path": "b.txt", "filename": "b.txt"},
                    "content": "Shared file ready for download: b.txt",
                    "structured_content": {
                        "kind": "shared_file",
                        "file": {"kind": "shared_file", "path": "b.txt", "filename": "b.txt"},
                    },
                    "tool_ui": {
                        "kind": "shared_file",
                        "status": "done",
                        "file": {"kind": "shared_file", "path": "b.txt", "filename": "b.txt"},
                    },
                },
            ]

        segments = _build_activity_segments(_MessageStub())
        files = [
            (segment.get("structuredContent") or {}).get("file", {}).get("filename")
            for segment in segments
            if segment.get("type") == "tool"
        ]
        self.assertEqual(files, ["a.txt", "b.txt"])


# Verify wait-for-user portal timing and finish signaling.
class BrowserPortalApiTests(SimpleTestCase):
    # Verify active browser portal state uses deadline when available.
    def test_active_browser_portal_state_uses_deadline_when_available(self):
        with patch("Apps.UI.views.time.time", return_value=1000.0):
            self.assertFalse(
                _is_active_browser_portal_state(
                    {
                        "status": "waiting",
                        "updated_at": 999.0,
                        "timeout_seconds": 45,
                        "deadline_at": 980.0,
                    }
                )
            )
            self.assertTrue(
                _is_active_browser_portal_state(
                    {
                        "status": "waiting",
                        "updated_at": 500.0,
                        "timeout_seconds": 45,
                        "deadline_at": 1005.0,
                    }
                )
            )

    # Verify finish event response reports done and queues event.
    def test_finish_event_response_reports_done_and_queues_event(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "browser_portal"
            events_dir = root / "events"
            events_dir.mkdir(parents=True)
            (root / "state.json").write_text(
                json.dumps(
                    {
                        "ok": True,
                        "status": "waiting",
                        "session_id": "session-a",
                        "updated_at": 999.0,
                        "timeout_seconds": 45,
                        "deadline_at": 1045.0,
                    }
                ),
                encoding="utf-8",
            )

            with patch("Apps.UI.views._browser_portal_roots", return_value=[root]):
                with patch("Apps.UI.views.time.time", return_value=1000.0):
                    response = Client().post(
                        reverse("browser_portal_event_api"),
                        data=json.dumps({"type": "finish", "session_id": "session-a"}),
                        content_type="application/json",
                    )

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertTrue(payload["queued"])
            self.assertEqual(payload["status"], "done")
            queued_events = list(events_dir.glob("event_*.json"))
            self.assertEqual(len(queued_events), 1)
            queued_payload = json.loads(queued_events[0].read_text(encoding="utf-8"))
            self.assertEqual(queued_payload["type"], "finish")
            self.assertEqual(queued_payload["session_id"], "session-a")


# Model metadata cache tests.
# Ensure cached payloads are safe to reuse between requests.
class ModelCapabilitiesCacheTests(SimpleTestCase):
    def setUp(self):
        super().setUp()
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache_dir = Path(temporary.name)
        cache_patch = patch.object(model_capabilities, "CACHE_DIR", self.cache_dir)
        cache_patch.start()
        self.addCleanup(cache_patch.stop)
        enabled_patch = patch("Apps.UI.views.settings.is_engine_enabled", return_value=True)
        enabled_patch.start()
        self.addCleanup(enabled_patch.stop)
        self.endpoint = "https://first.example/v1"
        url_patch = patch("Apps.UI.views.settings.get_engine_url", side_effect=lambda engine: self.endpoint)
        url_patch.start()
        self.addCleanup(url_patch.stop)
        _clear_model_metadata_caches()
        self.addCleanup(_clear_model_metadata_caches)

    def request_capabilities(self, engine="openai", model="same-model", **query):
        return self.client.get(reverse("model_capabilities_api"), {"engine": engine, "model": model, **query})

    @patch("Apps.UI.views.llm_api.get_model_settings")
    def test_success_is_read_back_from_disk_without_provider_or_presets(self, provider):
        provider.return_value = {"supports_vision": True, "supports_thinking": True, "defaults": {"temperature": 0.2}}
        with patch("Apps.UI.views.get_openai_preset_payload") as presets:
            first = self.request_capabilities()
            _clear_model_metadata_caches()
            provider.side_effect = RuntimeError("offline after restart")
            second = self.request_capabilities()
            presets.assert_not_called()
        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.json(), first.json())
        provider.assert_called_once_with("openai", "same-model")
        records = json.loads((self.cache_dir / "openai.json").read_text(encoding="utf-8"))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["endpoint"], self.endpoint)
        self.assertNotIn("defaults", records[0]["capabilities"])
        self.assertTrue(records[0]["capabilities"]["supports_thinking"])

    @patch("Apps.UI.views.llm_api.get_model_settings")
    def test_engine_and_endpoint_isolation_and_ollama_without_url(self, provider):
        for engine in ("lms", "openai", "google-genai"):
            self.endpoint = "https://first.example/v1"
            provider.return_value = {"supports_vision": True, "capabilities_source": "catalog"}
            self.assertEqual(self.request_capabilities(engine).status_code, 200)
            self.endpoint = "https://second.example/v1"
            provider.return_value = {"supports_tool_calling": True, "capabilities_source": "catalog"}
            self.assertEqual(self.request_capabilities(engine).status_code, 200)
            records = json.loads((self.cache_dir / f"{engine}.json").read_text(encoding="utf-8"))
            self.assertEqual(len(records), 2)
            self.assertTrue(records[0]["capabilities"]["supports_vision"])
            self.assertFalse(records[1]["capabilities"]["supports_vision"])
            self.assertTrue(records[1]["capabilities"]["supports_tool_calling"])
            listed = self.request_capabilities(engine, model="").json()["records"]
            self.assertEqual(listed, records[1:])
        provider.return_value = {"capabilities": ["vision", "tools", "thinking"]}
        response = self.request_capabilities("ollama-service", "gemma4:cloud")
        self.assertNotIn("endpoint", response.json())
        self.assertTrue(response.json()["capabilities"]["is_cloud"])
        self.assertEqual(len(list(self.cache_dir.glob("*.json"))), 4)

    @patch("Apps.UI.views.llm_api.get_model_settings")
    def test_only_expired_entries_are_refreshed_and_replaced(self, provider):
        with patch.object(model_capabilities.time, "time", return_value=1):
            model_capabilities.store("openai", "same-model", self.endpoint, {"supports_vision": False})
        provider.return_value = {"supports_vision": True}
        response = self.request_capabilities()
        self.assertTrue(response.json()["capabilities"]["supports_vision"])
        records = model_capabilities.list_cached("openai", self.endpoint)
        self.assertEqual(len(records), 1)
        self.assertTrue(model_capabilities.is_fresh(records[0]))
        provider.assert_called_once()

    @patch("Apps.UI.views.llm_api.get_model_settings", side_effect=RuntimeError("offline"))
    def test_failure_keeps_stale_facts_and_does_not_cache_unknown_as_false(self, provider):
        with patch.object(model_capabilities.time, "time", return_value=1):
            record = model_capabilities.store("openai", "same-model", self.endpoint, {"supports_vision": True})
        response = self.request_capabilities()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["stale"])
        self.assertTrue(response.json()["capabilities"]["supports_vision"])
        missing = self.request_capabilities(model="unknown")
        self.assertEqual(missing.status_code, 503)
        self.assertEqual(model_capabilities.list_cached("openai", self.endpoint), [record])

    @patch("Apps.UI.views.llm_api.get_model_settings")
    def test_invalid_cache_is_rebuilt_from_provider(self, provider):
        (self.cache_dir / "openai.json").write_text("[broken", encoding="utf-8")
        provider.return_value = {"supports_tool_calling": True}
        with self.assertLogs("Apps.Data.model_capabilities", level="WARNING"):
            response = self.request_capabilities()
        self.assertEqual(response.status_code, 200)
        self.assertTrue(model_capabilities.list_cached("openai", self.endpoint)[0]["capabilities"]["supports_tool_calling"])

    @patch("Apps.UI.views.llm_api.get_model_settings")
    def test_endpoint_change_during_fetch_cannot_contaminate_cache(self, provider):
        def change_endpoint(*_args):
            self.endpoint = "https://second.example/v1"
            return {"supports_vision": True}
        provider.side_effect = change_endpoint
        self.assertEqual(self.request_capabilities().status_code, 409)
        self.assertEqual(list(self.cache_dir.iterdir()), [])

    @patch("Apps.UI.views.llm_api.get_model_settings")
    def test_obsolete_queued_endpoint_request_is_rejected_without_fetch(self, provider):
        self.assertEqual(self.request_capabilities(endpoint="https://second.example/v1").status_code, 409)
        provider.assert_not_called()

    @patch("Apps.UI.views.llm_api.get_model_settings")
    def test_genai_fixed_endpoint_can_resolve_unselected_models(self, provider):
        self.endpoint = "https://generativelanguage.googleapis.com"
        provider.return_value = {"supports_vision": True, "supports_tool_calling": True}
        response = self.request_capabilities("google-genai", "unselected", endpoint=self.endpoint)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["capabilities"]["supports_tool_calling"])
        provider.assert_called_once_with("google-genai", "unselected")

    @patch("Apps.UI.views.llm_api.get_model_settings")
    def test_old_lms_snapshots_are_rechecked_without_waiting_a_day(self, provider):
        model_capabilities.store("lms", "same-model", self.endpoint, {"supports_vision": False})
        records = self.request_capabilities("lms", model="").json()["records"]
        self.assertTrue(records[0]["stale"])
        self.assertFalse(model_capabilities.is_fresh(records[0]))
        provider.return_value = {"supports_vision": True, "capabilities_source": "lms-catalog"}
        response = self.request_capabilities("lms")
        self.assertTrue(response.json()["capabilities"]["supports_vision"])
        self.assertTrue(model_capabilities.is_fresh(response.json()))
        self.assertEqual(response.json()["source"], "lms-catalog")
        self.request_capabilities("lms")
        provider.assert_called_once()

    @patch("Apps.UI.views.llm_api.get_model_settings", return_value={"metadata_fallback": True})
    def test_missing_lms_metadata_is_not_persisted_as_unsupported(self, provider):
        response = self.request_capabilities("lms")
        self.assertEqual(response.status_code, 503)
        self.assertEqual(list(self.cache_dir.iterdir()), [])

    @patch("Apps.UI.views.llm_api.get_model_settings")
    def test_missing_layer_metadata_is_not_a_cloud_marker(self, provider):
        provider.return_value = {"capabilities": ["completion"]}
        response = self.request_capabilities("ollama-service", "local-model")
        self.assertFalse(response.json()["capabilities"]["is_cloud"])
        provider.return_value = {"supports_vision": True, "is_cloud": True}
        response = self.request_capabilities("openai", "remote-model")
        self.assertFalse(response.json()["capabilities"]["is_cloud"])

    def test_failed_atomic_write_preserves_previous_file(self):
        record = model_capabilities.store("openai", "same-model", self.endpoint, {"supports_vision": True})
        with patch.object(model_capabilities.os, "replace", side_effect=OSError("locked")):
            with self.assertLogs("Apps.Data.model_capabilities", level="WARNING"):
                model_capabilities.store("openai", "same-model", self.endpoint, {"supports_vision": False})
        self.assertEqual(model_capabilities.list_cached("openai", self.endpoint), [record])
        self.assertEqual(list(self.cache_dir.glob("*.tmp")), [])


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
    @patch("Apps.UI.views.llm_api.get_model_settings")
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


# Cover context compression threshold math.
class ContextCompressionBudgetTests(SimpleTestCase):
    # Verify history budget uses same model token estimator as usage ui.
    def test_history_budget_uses_same_model_token_estimator_as_usage_ui(self):
        payload = {"context_length": 10000}

        self.assertEqual(
            _resolve_history_char_budget(payload, active_engine="lms", active_model="qwen3"),
            20000,
        )
        self.assertEqual(
            _resolve_history_char_budget(payload, active_engine="ollama-service", active_model="llama3"),
            27000,
        )

    # Verify history budget blends observed token ratio.
    def test_history_budget_blends_observed_token_ratio(self):
        payload = {"context_length": 10000}

        self.assertEqual(
            _resolve_history_char_budget(
                payload,
                active_engine="lms",
                active_model="qwen3",
                observed_chars_per_token=1.6,
            ),
            17400,
        )


# Exercise chat API basics without calling a real model backend.
class ChatApiTests(ToolRegistryTestMixin, TestCase):
    # Set up the test fixture.
    def setUp(self):
        super().setUp()
        self.client = Client()
        self._chat_title_patch = patch("Apps.UI.views.llm_api.generate_chat_title", return_value="")
        self.mock_generate_chat_title = self._chat_title_patch.start()
        self.addCleanup(self._chat_title_patch.stop)

    # Test chat API rejects invalid JSON before touching runtime services.
    def test_chat_api_rejects_invalid_json_body(self):
        response = self.client.post(
            reverse("chat_api"),
            data="{",
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Invalid JSON format")

    # Test chat API requires a model name.
    def test_chat_api_rejects_missing_model(self):
        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Missing model parameter")

    # Test chat API creates new chat and streams response.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api creates new chat and streams response.
    def test_chat_api_creates_new_chat_and_streams_response(
        self,
        _mock_engine,
        mock_generate,
        mock_prepare_runtime,
    ):
        mock_generate.return_value = [{"message": {"content": "Hi there"}}]

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.has_header("X-Chat-ID"))
        self.assertEqual(b"".join(response.streaming_content), b"Hi there")
        self.assertEqual(Chat.objects.count(), 1)
        self.assertEqual(Chat.objects.first().messages.count(), 2)
        mock_prepare_runtime.assert_any_call("ollama-service")

    # Test a generated title is stored and returned with a new chat.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._build_model_info_payload", return_value={})
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    def test_chat_api_stores_generated_title(
        self,
        _mock_engine,
        _mock_model_info,
        mock_generate,
        _mock_prepare_runtime,
    ):
        self.mock_generate_chat_title.return_value = "Название чата"
        mock_generate.return_value = [{"message": {"content": "Ответ"}}]

        response = self.client.post(
            reverse("chat_api"),
            data=json.dumps({"message": "Первое сообщение", "model": "test-model"}),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response["X-Chat-Title"],
            "%D0%9D%D0%B0%D0%B7%D0%B2%D0%B0%D0%BD%D0%B8%D0%B5%20%D1%87%D0%B0%D1%82%D0%B0",
        )
        self.assertEqual(Chat.objects.get().title, "Название чата")
        self.mock_generate_chat_title.assert_called_once()

    @patch(
        "Apps.UI.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["tools", "thinking"],
            "template": "{{ if .Tools }}{{ end }}{{ if .ToolCalls }}{{ end }}",
            "think_param_name": "think",
            "think_level_param_name": "think_level",
        },
    )
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    def test_same_chat_switches_from_instant_to_thinking_contract(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
        _mock_model_settings,
    ):
        self.write_server(
            "web_search",
            '''
            MCP_SERVER = {"id": "web_search", "name": "Web Search"}
            TOOLS = [
                {"id": "web_search", "name": "Web Search", "parameters": {"type": "object", "properties": {}}},
                {"id": "read_page", "name": "Read Page", "parameters": {"type": "object", "properties": {}}},
            ]
            def call_tool(tool_id, arguments, context=None):
                return {"ok": True}
            ''',
        )
        mock_generate.return_value = [{"message": {"content": "Instant answer"}}]

        instant_response = self.client.post(
            reverse("chat_api"),
            data=json.dumps({
                "message": "/search First turn",
                "model": "llama3",
                "tool_server_ids": ["web_search"],
                "options": {"think": False},
            }),
            content_type="application/json",
        )
        self.assertEqual(instant_response.status_code, 200)
        b"".join(instant_response.streaming_content)
        instant_kwargs = dict(mock_generate.call_args.kwargs)
        chat_id = instant_response["X-Chat-ID"]

        mock_generate.reset_mock()
        mock_generate.return_value = [{"message": {"content": "Thinking answer"}}]
        thinking_response = self.client.post(
            reverse("chat_api"),
            data=json.dumps({
                "chat_id": chat_id,
                "message": "Second turn",
                "model": "llama3",
                "tool_server_ids": ["web_search"],
                "options": {"think_level": "medium"},
            }),
            content_type="application/json",
        )
        self.assertEqual(thinking_response.status_code, 200)
        b"".join(thinking_response.streaming_content)
        thinking_kwargs = dict(mock_generate.call_args.kwargs)

        self.assertIn("Instant/no-thinking web lookup rules", instant_kwargs["messages"][0]["content"])
        self.assertNotIn("Web-search research planning rules", instant_kwargs["messages"][0]["content"])
        self.assertIn("exactly three complementary query strings", instant_kwargs["messages"][0]["content"])
        self.assertTrue(instant_kwargs["tool_context"]["instant_mode"])
        self.assertEqual(instant_kwargs["tool_context"]["instant_search_batch_size"], 3)
        self.assertEqual(instant_kwargs["tool_context"]["forced_tool_name"], "web_search")
        self.assertNotIn("max_tool_rounds", instant_kwargs["tool_context"])

        self.assertIn("Web-search research planning rules", thinking_kwargs["messages"][0]["content"])
        self.assertNotIn("Instant/no-thinking web lookup rules", thinking_kwargs["messages"][0]["content"])
        self.assertNotIn("instant_mode", thinking_kwargs["tool_context"])
        self.assertNotIn("instant_search_batch_size", thinking_kwargs["tool_context"])
        self.assertNotIn("max_tool_rounds", thinking_kwargs["tool_context"])
        self.assertNotIn("think", thinking_kwargs)
        self.assertEqual(thinking_kwargs["think_level"], "medium")

        thinking_system_messages = [
            message
            for message in thinking_kwargs["messages"]
            if message.get("role") == "system"
        ]
        self.assertEqual(len(thinking_system_messages), 1)
        self.assertTrue(
            thinking_system_messages[0]["content"].startswith(
                get_system_prompt(instant_mode=False)
            )
        )
        self.assertIn(
            {"role": "assistant", "content": "Instant answer"},
            thinking_kwargs["messages"],
        )
        self.assertNotIn(
            "Instant/no-thinking web lookup rules",
            json.dumps(thinking_kwargs["messages"], ensure_ascii=False),
        )

    # Test chat API supports an attachment-only prompt.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="lms")
    # Verify chat api creates attachment only thread.
    def test_chat_api_creates_attachment_only_thread(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = [{"message": {"content": "Done"}}]

        response = self.client.post(
            reverse("chat_api"),
            data=json.dumps(
                {
                    "model": "qwen",
                    "attachments": [
                        {
                            "name": "note.txt",
                            "mime_type": "text/plain",
                            "data": "SGVsbG8=",
                        },
                    ],
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        chat = Chat.objects.get()
        self.assertEqual(chat.title, "Attachment chat")
        self.assertEqual(chat.messages.filter(role="user").get().content, "")

    # Test chat API passes selected tool server to Ollama.
    @patch(
        "Apps.UI.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["tools"],
            "template": "{{ if .Tools }}{{ end }}{{ if .ToolCalls }}{{ end }}",
        },
    )
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api passes selected tool server to ollama.
    def test_chat_api_passes_selected_tool_server_to_ollama(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
        _mock_model_settings,
    ):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite"}
            TOOLS = [
                {"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}},
                {"id": "timezone_name", "name": "Timezone Name", "parameters": {"type": "object", "properties": {}}},
            ]
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )
        mock_generate.return_value = [{"message": {"content": "Done"}}]

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3.1","tool_server_id":"time_suite"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(b''.join(response.streaming_content), b'Done')
        self.assertEqual(mock_generate.call_args.kwargs["tool_server_ids"], ["time_suite"])
        self.assertEqual(mock_generate.call_args.kwargs["tool_context"]["engine"], "ollama-service")
        self.assertEqual(Chat.objects.first().active_tool_slug, '["time_suite"]')

    # Test chat API rejects unknown tool server.
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api rejects unknown tool server.
    def test_chat_api_rejects_unknown_tool_server(self, _mock_engine):
        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3","tool_server_id":"missing"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("Unknown or unsupported tool server", response.json()["error"])

    # Test chat API stream includes server and tool markers.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api stream includes server and tool markers.
    def test_chat_api_stream_includes_server_and_tool_markers(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = iter([
            {"message": {"thinking": "Searching..."}},
            {"tool_event": {"server_id": "time_suite", "server_name": "Time Suite", "tool_id": "time_now", "tool_name": "Current Time", "alias": "time_suite__time_now", "arguments": {"label": "now"}}},
            {"message": {"content": "Done"}},
        ])

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        body = b''.join(response.streaming_content).decode('utf-8')
        self.assertIn('<think>\nSearching...', body)
        self.assertIn('\"server_id\": \"time_suite\"', body)
        self.assertIn('\"tool_id\": \"time_now\"', body)
        self.assertIn('Done', body)

    # Test chat API keeps reasoning-only output instead of deleting it on abort.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api persists reasoning only response.
    def test_chat_api_persists_reasoning_only_response(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = iter([
            {"message": {"thinking": "Planning the answer."}},
        ])

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        body = b"".join(response.streaming_content).decode("utf-8")
        self.assertIn("<think>\nPlanning the answer.", body)
        assistant_message = Message.objects.filter(role="assistant").latest("created_at")
        self.assertEqual(assistant_message.content, "")
        self.assertEqual(assistant_message.llm_transcript[0]["role"], "assistant")
        self.assertEqual(assistant_message.llm_transcript[0]["thinking"], "Planning the answer.")

    # Test streamed reasoning is buffered before the full response completes.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api buffers reasoning while streaming.
    def test_chat_api_buffers_reasoning_while_streaming(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = iter([
            {"message": {"thinking": "Live reasoning buffer."}},
            {"message": {"content": "Done"}},
        ])

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        stream = iter(response.streaming_content)
        first_chunk = next(stream).decode("utf-8")
        self.assertEqual(first_chunk, "<think>\n")
        assistant_message = Message.objects.filter(role="assistant").latest("created_at")
        self.assertEqual(assistant_message.llm_transcript[0]["thinking"], "Live reasoning buffer.")
        b"".join(stream)

    # Test streaming compression can run during reasoning without waiting for the next send.
    @patch("Apps.UI.views._build_manual_compression_event")
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    # Verify stream chat response auto compresses at reasoning safe point.
    def test_stream_chat_response_auto_compresses_at_reasoning_safe_point(
        self,
        mock_generate,
        _mock_prepare_runtime,
        mock_build_compression_event,
    ):
        chat = Chat.objects.create(title="Chat")
        assistant_message = Message.objects.create(chat=chat, role="assistant", content="", llm_transcript=[])
        compression_event = {
            "role": "tool",
            "alias": "context_compression_summary",
            "name": "context_compression_summary",
            "tool_name": "context_compression_summary",
            "tool_id": "context_compression_summary",
            "content": "Compressed history",
            "arguments": {},
        }
        mock_build_compression_event.return_value = compression_event
        mock_generate.return_value = iter([
            {"message": {"thinking": "Live reasoning buffer."}},
            {"message": {"content": "Done"}},
        ])

        body = "".join(_stream_chat_response(
            "ollama-service",
            {
                "engine": "ollama-service",
                "model_name": "llama3",
                "messages": [],
                "stream": True,
            },
            "generation-1",
            chat=chat,
            assistant_message_record=assistant_message,
            session_id=str(chat.id),
            model_info_payload={},
            system_prompt="System",
        ))

        self.assertIn("<context_compression>", body)
        self.assertIn('"auto_trigger": "reasoning"', body)
        self.assertIn('"restart_generation": true', body)
        kwargs = mock_build_compression_event.call_args.kwargs
        self.assertEqual(kwargs["draft_text"], "Live reasoning buffer.")
        self.assertEqual(kwargs["exclude_message_ids"], {assistant_message.id})
        self.assertFalse(kwargs["summarize_with_model_enabled"])
        compression_message = (
            Message.objects
            .filter(role="assistant", llm_transcript__0__alias="context_compression_summary")
            .latest("created_at")
        )
        self.assertNotEqual(compression_message.id, assistant_message.id)
        self.assertEqual(compression_message.llm_transcript[0]["arguments"]["auto_trigger"], "reasoning")
        self.assertTrue(compression_message.llm_transcript[0]["arguments"]["restart_generation"])

    # Test streaming compression also checks the threshold when a tool call starts.
    @patch("Apps.UI.views._build_manual_compression_event")
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    # Verify stream chat response auto compresses at tool call safe point.
    def test_stream_chat_response_auto_compresses_at_tool_call_safe_point(
        self,
        mock_generate,
        _mock_prepare_runtime,
        mock_build_compression_event,
    ):
        chat = Chat.objects.create(title="Chat")
        assistant_message = Message.objects.create(chat=chat, role="assistant", content="", llm_transcript=[])
        compression_event = {
            "role": "tool",
            "alias": "context_compression_summary",
            "name": "context_compression_summary",
            "tool_name": "context_compression_summary",
            "tool_id": "context_compression_summary",
            "content": "Compressed history",
            "arguments": {},
        }
        mock_build_compression_event.return_value = compression_event
        mock_generate.return_value = iter([
            {
                "tool_event": {
                    "server_id": "time_suite",
                    "server_name": "Time Suite",
                    "tool_id": "time_now",
                    "tool_name": "Current Time",
                    "alias": "time_suite__time_now",
                    "arguments": {"label": "now"},
                },
            },
            {"message": {"content": "Done"}},
        ])

        body = "".join(_stream_chat_response(
            "ollama-service",
            {
                "engine": "ollama-service",
                "model_name": "llama3",
                "messages": [],
                "stream": True,
            },
            "generation-1",
            chat=chat,
            assistant_message_record=assistant_message,
            session_id=str(chat.id),
            model_info_payload={},
            system_prompt="System",
        ))

        self.assertIn("<context_compression>", body)
        self.assertIn('"auto_trigger": "tool_call"', body)
        kwargs = mock_build_compression_event.call_args.kwargs
        self.assertIn("time_now", kwargs["draft_text"])
        compression_message = (
            Message.objects
            .filter(role="assistant", llm_transcript__0__alias="context_compression_summary")
            .latest("created_at")
        )
        self.assertEqual(compression_message.llm_transcript[0]["arguments"]["auto_trigger"], "tool_call")

    # Test the server-side history builder uses the current prompt for the 80% trigger.
    @patch("Apps.UI.views.build_structured_history_summary")
    # Verify build chat history compresses when current prompt crosses threshold.
    def test_build_chat_history_compresses_when_current_prompt_crosses_threshold(
        self,
        mock_build_summary,
    ):
        mock_build_summary.return_value = (
            "Compressed prior history",
            {"summary_version": 1, "work_summary": "Compressed prior history"},
        )
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Older user context")
        Message.objects.create(chat=chat, role="assistant", content="Older assistant context")
        current_user = Message.objects.create(chat=chat, role="user", content="x" * 9800)

        llm_messages, compression_event = _build_chat_history(
            chat,
            current_user,
            current_user.content,
            "system",
            "lms",
            "qwen3",
            {"context_length": 4096, "defaults": {"num_ctx": 4096}},
        )

        self.assertIsNotNone(compression_event)
        self.assertEqual(compression_event["arguments"]["context_window_tokens"], 4096)
        self.assertGreaterEqual(
            compression_event["arguments"]["used_history_chars"],
            int(compression_event["arguments"]["history_budget_chars"] * 0.8),
        )
        self.assertEqual(llm_messages[0]["role"], "system")
        self.assertIn("Compressed prior history", llm_messages[1]["content"])
        self.assertEqual(llm_messages[-1]["content"], current_user.content)

    # Test chat API persists generic attachments and builds LM Studio messages.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="lms")
    # Verify chat api persists generic attachments and builds lms messages.
    def test_chat_api_persists_generic_attachments_and_builds_lms_messages(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = [{"message": {"content": "Done"}}]

        response = self.client.post(
            reverse("chat_api"),
            data=json.dumps(
                {
                    "message": "Hello",
                    "model": "qwen",
                    "attachments": [
                        {
                            "kind": "image",
                            "name": "photo.png",
                            "mime_type": "image/png",
                            "data": "iVBORw0KGgo=",
                        },
                        {
                            "kind": "file",
                            "name": "note.txt",
                            "mime_type": "text/plain",
                            "data": "SGVsbG8gZnJvbSBmaWxl",
                        },
                    ],
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        self.assertEqual(MessageAttachment.objects.count(), 2)

        outbound_messages = mock_generate.call_args.kwargs["messages"]
        current_user_message = outbound_messages[-1]
        self.assertEqual(current_user_message["role"], "user")
        self.assertEqual(len(current_user_message["images"]), 1)
        self.assertIn("[Attached file: note.txt]", current_user_message["content"])
        self.assertIn("Hello from file", current_user_message["content"])

    # Test chat API rejects tool server when LM Studio model lacks tool support.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings", return_value={"supports_tool_calling": False, "supports_files": True})
    @patch("Apps.UI.views.llm_api.get_model_settings", return_value={"supports_tool_calling": False, "supports_files": True})
    @patch("Apps.UI.views._resolve_request_engine", return_value="lms")
    # Verify chat api rejects tool server when lms model lacks tool support.
    def test_chat_api_rejects_tool_server_when_lms_model_lacks_tool_support(
        self,
        _mock_engine,
        _mock_model_settings,
        _mock_preset_model_settings,
    ):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite"}
            TOOLS = [{"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}}]
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"qwen","tool_server_id":"time_suite"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("does not support tool calling", response.json()["error"])

    # Test chat API rejects tool server when Ollama capabilities omit tools.
    @patch(
        "Apps.UI.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["completion"],
            "template": "{{ if .Tools }}tools{{ end }}{{ if .ToolCalls }}calls{{ end }}",
        },
    )
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api rejects tool server when ollama capabilities omit tools.
    def test_chat_api_rejects_tool_server_when_ollama_capabilities_omit_tools(
        self,
        _mock_engine,
        _mock_model_settings,
    ):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite"}
            TOOLS = [{"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}}]
            def supports(engine=None, model_name=None):
                return engine == "ollama-service"
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"model-without-tools","tool_server_id":"time_suite"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("does not support tool calling", response.json()["error"])

    # Test chat API rejects tool server when OpenAI model lacks tool support.
    @patch("Apps.UI.views.llm_api.get_model_settings", return_value={"supports_tool_calling": False, "supports_files": True})
    @patch("Apps.UI.views._resolve_request_engine", return_value="openai")
    # Verify chat api rejects tool server when openai model lacks tool support.
    def test_chat_api_rejects_tool_server_when_openai_model_lacks_tool_support(
        self,
        _mock_engine,
        _mock_model_settings,
    ):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite"}
            TOOLS = [{"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}}]
            def supports(engine=None, model_name=None):
                return engine == "openai"
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"gpt-test","tool_server_id":"time_suite"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("does not support tool calling", response.json()["error"])


    # Test chat API saves visible content and machine transcript.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api saves visible content and machine transcript.
    def test_chat_api_saves_visible_content_and_machine_transcript(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = iter([
            {"message": {"thinking": "Plan first."}},
            {"transcript_message": {"role": "assistant", "content": "", "thinking": "Plan first.", "tool_calls": [{"function": {"name": "time_suite__time_now", "arguments": {"label": "now"}}}]}},
            {"tool_event": {"server_id": "time_suite", "server_name": "Time Suite", "tool_id": "time_now", "tool_name": "Current Time", "alias": "time_suite__time_now", "arguments": {"label": "now"}}},
            {"tool_result": {"role": "tool", "name": "time_suite__time_now", "tool_name": "time_suite__time_now", "content": '{"ok": true}', "server_id": "time_suite", "server_name": "Time Suite", "tool_id": "time_now", "tool_display_name": "Current Time", "arguments": {"label": "now"}}},
            {"message": {"content": "Done"}},
            {"transcript_message": {"role": "assistant", "content": "Done"}},
        ])

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        assistant_message = Message.objects.filter(role="assistant").latest("created_at")
        self.assertEqual(assistant_message.content, "Done")
        self.assertEqual(len(assistant_message.llm_transcript), 3)
        self.assertEqual(assistant_message.llm_transcript[1]["role"], "tool")
        self.assertEqual(assistant_message.llm_transcript[1]["server_name"], "Time Suite")

    # Test chat API uses stored transcript for follow up messages.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api uses stored transcript for follow up messages.
    def test_chat_api_uses_stored_transcript_for_follow_up_messages(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Hello")
        Message.objects.create(
            chat=chat,
            role="assistant",
            content="Visible answer",
            llm_transcript=[
                {"role": "assistant", "content": "", "thinking": "Plan", "tool_calls": [{"function": {"name": "time_suite__time_now", "arguments": {"label": "now"}}}]},
                {"role": "tool", "name": "time_suite__time_now", "tool_name": "time_suite__time_now", "content": '{"ok": true}'},
                {"role": "assistant", "content": "Visible answer"},
            ],
        )
        mock_generate.return_value = [{"message": {"content": "Next"}}]

        response = self.client.post(
            reverse("chat_api"),
            data=f'{{"chat_id":"{chat.id}","message":"Follow up","model":"llama3"}}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        history_messages = mock_generate.call_args.kwargs["messages"]
        non_system = [item for item in history_messages if item.get("role") != "system"]
        self.assertEqual([item["role"] for item in non_system[:4]], ["user", "assistant", "tool", "assistant"])
        self.assertEqual(non_system[1]["thinking"], "Plan")
        self.assertEqual(non_system[2]["name"], "time_suite__time_now")
        self.assertEqual(history_messages[-1]["content"], "Follow up")

    # Test chat API strips legacy UI markup when transcript is missing.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api strips legacy ui markup when transcript is missing.
    def test_chat_api_strips_legacy_ui_markup_when_transcript_is_missing(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Hello")
        Message.objects.create(
            chat=chat,
            role="assistant",
            content='<think>\nPlan\n</think>\n<tool_call>{"alias":"time_suite__time_now"}</tool_call>Visible answer',
        )
        mock_generate.return_value = [{"message": {"content": "Next"}}]

        response = self.client.post(
            reverse("chat_api"),
            data=f'{{"chat_id":"{chat.id}","message":"Follow up","model":"llama3"}}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        history_messages = mock_generate.call_args.kwargs["messages"]
        assistant_messages = [item for item in history_messages if item.get("role") == "assistant"]
        self.assertEqual(assistant_messages[0], {"role": "assistant", "content": "Visible answer"})

    # Test chat API strips service control tokens from visible output.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="lms")
    # Verify chat api strips service control tokens from visible output.
    def test_chat_api_strips_service_control_tokens_from_visible_output(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
    ):
        mock_generate.return_value = [
            {"message": {"content": "<|start|>assistant<|channel|>final<|message|>Hello"}},
        ]

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hi","model":"qwen3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(b"".join(response.streaming_content).decode("utf-8"), "Hello")
        assistant_message = Message.objects.filter(role="assistant").latest("created_at")
        self.assertEqual(assistant_message.content, "Hello")


# Exercise stateless generate API without persisting chat rows.
class GenerateApiTests(ToolRegistryTestMixin, TestCase):
    def setUp(self):
        super().setUp()
        self.client = Client()

    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_streams_without_db_writes(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Stateless reply"}}]

        response = self.client.post(
            reverse("generate_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.has_header("X-Chat-ID"))
        self.assertTrue(response.has_header("X-Session-ID"))
        self.assertTrue(response.has_header("X-Generation-ID"))
        self.assertEqual(b"".join(response.streaming_content), b"Stateless reply")
        self.assertEqual(Chat.objects.count(), 0)
        self.assertEqual(Message.objects.count(), 0)

    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_passes_messages_to_generate(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Follow-up"}}]

        response = self.client.post(
            reverse("generate_api"),
            data=json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "Earlier question"},
                        {"role": "assistant", "content": "Earlier answer"},
                    ],
                    "message": "Next question",
                    "model": "llama3",
                    "session_id": "module-session-1",
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        kwargs = mock_generate.call_args.kwargs
        messages = kwargs["messages"]
        self.assertEqual(messages[-1], {"role": "user", "content": "Next question"})
        self.assertIn({"role": "assistant", "content": "Earlier answer"}, messages)
        self.assertEqual(response["X-Session-ID"], "module-session-1")

    def test_generate_api_rejects_missing_model(self):
        response = self.client.post(
            reverse("generate_api"),
            data='{"message":"Hello"}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Missing model parameter")

    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_supports_inline_attachments(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Seen"}}]

        response = self.client.post(
            reverse("generate_api"),
            data=json.dumps(
                {
                    "model": "llama3",
                    "attachments": [
                        {
                            "name": "note.txt",
                            "mime_type": "text/plain",
                            "data": "SGVsbG8=",
                        },
                    ],
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        messages = mock_generate.call_args.kwargs["messages"]
        user_entry = messages[-1]
        self.assertEqual(user_entry["role"], "user")
        self.assertIn("note.txt", str(user_entry.get("content") or ""))

    @patch(
        "Apps.UI.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["tools"],
            "template": "{{ if .Tools }}{{ end }}{{ if .ToolCalls }}{{ end }}",
        },
    )
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_passes_tool_servers_to_generate(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
        _mock_model_settings,
    ):
        self.write_server(
            "time_suite",
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite"}
            TOOLS = [
                {"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}},
            ]
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )
        mock_generate.return_value = [{"message": {"content": "Tool reply"}}]

        response = self.client.post(
            reverse("generate_api"),
            data=json.dumps(
                {
                    "message": "What time is it?",
                    "model": "llama3",
                    "tool_server_ids": ["time_suite"],
                    "session_id": "tool-session",
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        kwargs = mock_generate.call_args.kwargs
        self.assertEqual(kwargs["tool_server_ids"], ["time_suite"])
        self.assertEqual(kwargs["tool_context"]["chat_id"], "tool-session")

    @patch(
        "Apps.UI.views.llm_api.get_model_settings",
        return_value={
            "capabilities": ["tools", "thinking"],
            "template": "{{ if .Tools }}{{ end }}{{ if .ToolCalls }}{{ end }}",
            "think_param_name": "think",
            "think_level_param_name": "think_level",
        },
    )
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_switches_prompt_schema_rounds_and_quotas_with_reasoning_mode(
        self,
        _mock_engine,
        mock_generate,
        _mock_prepare_runtime,
        _mock_model_settings,
    ):
        self.write_server(
            "web_search",
            '''
            MCP_SERVER = {"id": "web_search", "name": "Web Search"}
            TOOLS = [
                {
                    "id": "web_search",
                    "name": "Web Search",
                    "description": "Advanced search",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "web": {"type": "string"},
                            "shopping": {"type": "string"},
                            "academic": {"type": "string"},
                            "onion": {"type": "string"},
                            "effort": {"type": "string"},
                            "call_description": {"type": "string"},
                        },
                    },
                },
                {
                    "id": "read_page",
                    "name": "Read Page",
                    "description": "Read exact pages",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "oneOf": [
                                    {"type": "string"},
                                    {"type": "array", "maxItems": 10, "items": {"type": "string"}},
                                ]
                            }
                        },
                    },
                },
            ]
            def call_tool(tool_id, arguments, context=None):
                return {"ok": True}
            ''',
        )
        mock_generate.return_value = [{"message": {"content": "Done"}}]

        def run_mode(options):
            response = self.client.post(
                reverse("generate_api"),
                data=json.dumps({
                    "message": "Check current information",
                    "model": "llama3",
                    "tool_server_ids": ["web_search"],
                    "include_skills_baseline": False,
                    "options": options,
                }),
                content_type="application/json",
            )
            self.assertEqual(response.status_code, 200)
            b"".join(response.streaming_content)
            kwargs = dict(mock_generate.call_args.kwargs)
            tool_context = dict(kwargs["tool_context"])
            tools, lookup = tool_registry.build_ollama_tools(
                kwargs["tool_server_ids"],
                engine=kwargs["engine"],
                model_name=kwargs["model_name"],
                instant_mode=bool(tool_context.get("instant_mode")),
                instant_search_batch_size=tool_context.get("instant_search_batch_size"),
            )
            mock_generate.reset_mock()
            mock_generate.return_value = [{"message": {"content": "Done"}}]
            return kwargs, tools, lookup

        instant_kwargs, instant_tools, instant_lookup = run_mode({"think": False})
        thinking_kwargs, thinking_tools, thinking_lookup = run_mode({"think_level": "medium"})

        instant_system = instant_kwargs["messages"][0]["content"]
        thinking_system = thinking_kwargs["messages"][0]["content"]
        self.assertTrue(instant_system.startswith(get_system_prompt(instant_mode=True)))
        self.assertTrue(thinking_system.startswith(get_system_prompt(instant_mode=False)))
        self.assertIn("Instant/no-thinking web lookup rules", instant_system)
        self.assertNotIn("Web-search research planning rules", instant_system)
        self.assertIn("Web-search research planning rules", thinking_system)
        self.assertNotIn("Instant/no-thinking web lookup rules", thinking_system)

        self.assertIs(instant_kwargs["think"], False)
        self.assertTrue(instant_kwargs["tool_context"]["instant_mode"])
        self.assertNotIn("instant_search_batch_size", instant_kwargs["tool_context"])
        self.assertNotIn("max_tool_rounds", instant_kwargs["tool_context"])
        self.assertEqual(thinking_kwargs["think_level"], "medium")
        self.assertNotIn("instant_mode", thinking_kwargs["tool_context"])
        self.assertNotIn("instant_search_batch_size", thinking_kwargs["tool_context"])
        self.assertNotIn("max_tool_rounds", thinking_kwargs["tool_context"])

        instant_by_name = {tool["function"]["name"]: tool for tool in instant_tools}
        thinking_by_name = {tool["function"]["name"]: tool for tool in thinking_tools}
        instant_search_schema = instant_by_name["web_search__web_search"]["function"]["parameters"]
        thinking_search_schema = thinking_by_name["web_search__web_search"]["function"]["parameters"]
        self.assertEqual(set(instant_search_schema["properties"]), {"query", "description", "operators"})
        self.assertEqual(instant_search_schema["properties"]["query"]["oneOf"][1]["maxItems"], 3)
        self.assertIn("effort", thinking_search_schema["properties"])
        self.assertIn("shopping", thinking_search_schema["properties"])

        instant_read_schema = instant_by_name["web_search__read_page"]["function"]["parameters"]
        instant_read_array = instant_read_schema["properties"]["url"]["oneOf"][1]
        self.assertEqual(instant_read_array["maxItems"], 3)

        instant_event = tool_registry.build_tool_event(
            instant_lookup,
            {"name": "web_search__web_search", "arguments": {"query": "x", "description": "Checking"}},
        )
        thinking_event = tool_registry.build_tool_event(
            thinking_lookup,
            {"name": "web_search__web_search", "arguments": {"web": "x", "effort": "medium"}},
        )
        instant_counters: dict[str, int] = {}
        thinking_counters: dict[str, int] = {}
        self.assertIsNone(tool_registry.consume_tool_quota(instant_event, instant_counters))
        self.assertIsNone(tool_registry.consume_tool_quota(instant_event, instant_counters))
        self.assertIsNotNone(tool_registry.consume_tool_quota(instant_event, instant_counters))
        self.assertIsNone(tool_registry.consume_tool_quota(thinking_event, thinking_counters))
        self.assertIsNone(tool_registry.consume_tool_quota(thinking_event, thinking_counters))

    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    def test_generate_api_replays_llm_transcript_in_history(self, _mock_engine, mock_generate, _mock_prepare_runtime):
        mock_generate.return_value = [{"message": {"content": "Done"}}]

        response = self.client.post(
            reverse("generate_api"),
            data=json.dumps(
                {
                    "messages": [
                        {
                            "role": "assistant",
                            "content": "Calling tool",
                            "llm_transcript": [
                                {
                                    "role": "assistant",
                                    "content": "",
                                    "tool_calls": [{"id": "call-1", "function": {"name": "time_now", "arguments": "{}"}}],
                                },
                                {
                                    "role": "tool",
                                    "tool_call_id": "call-1",
                                    "content": "12:00",
                                },
                            ],
                        }
                    ],
                    "message": "Continue",
                    "model": "llama3",
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        b"".join(response.streaming_content)
        messages = mock_generate.call_args.kwargs["messages"]
        self.assertTrue(any(entry.get("role") == "tool" and entry.get("content") == "12:00" for entry in messages))


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


# Verify Ollama desired-state policy.
class OllamaDesiredStateTests(SimpleTestCase):
    @patch("Settings.settings.get_llm_engine", return_value="lms")
    @patch("Settings.settings.get", return_value=True)
    def test_desired_state_runs_when_enabled_even_if_active_engine_differs(self, _mock_get, _mock_active):
        ollama_service = importlib.import_module("Services.ollama-service")
        state = ollama_service._get_desired_state("ollama-service")
        self.assertTrue(state.should_run)

    def test_service_never_persists_runtime_output_in_module_settings(self):
        ollama_service = importlib.import_module("Services.ollama-service")
        source = Path(ollama_service.__file__).read_text(encoding="utf-8")

        self.assertFalse(hasattr(ollama_service, "LOG_FILE"))
        self.assertNotIn("ollama-service.log", source)
        self.assertIn("stdout=subprocess.PIPE", source)


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
            with patch("Settings.settings.SETTINGS_FILE", settings_file):
                with patch("Settings.settings._apply_environment_overrides", side_effect=lambda data: data):
                    with patch("Settings.settings._sync_module_manifest_setting"):
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

    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
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
            "Apps.UI.views.settings.is_engine_enabled",
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
            with patch("Settings.settings.SETTINGS_FILE", settings_file):
                with patch("Settings.settings._apply_environment_overrides", side_effect=lambda data: data):
                    with patch("Settings.settings._sync_module_manifest_setting"):
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
    @patch("Apps.UI.views._load_models_for_engine", return_value=["llama3"])
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
    @patch("Apps.UI.views._build_model_info_payload", side_effect=NotImplementedError("Not supported"))
    # Verify model info api returns 501 for unimplemented engines.
    def test_model_info_api_returns_501_for_unimplemented_engines(self, mock_build_payload):
        response = self.client.get(reverse("model_info_api"), {"engine": "ollama-service", "model": "model"})

        self.assertEqual(response.status_code, 501)
        self.assertEqual(response.json()["error"], "Not supported")
        mock_build_payload.assert_called_once()

    # Test inference info API returns a compact engine-independent payload.
    @patch("Apps.UI.views._build_model_info_payload")
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
    @patch("Apps.UI.views._build_model_info_payload")
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


# Cover local tool-server listing and chat persistence endpoints.
class ToolApiTests(ToolRegistryTestMixin, TestCase):
    def setUp(self):
        super().setUp()
        self._engine_enabled_patch = patch(
            "Apps.UI.views.settings.is_engine_enabled",
            return_value=True,
        )
        self._engine_enabled_patch.start()

    def tearDown(self):
        self._engine_enabled_patch.stop()
        super().tearDown()

    # Test tools API returns discovered servers.
    def test_tools_api_returns_discovered_servers(self):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite", "description": "Time helpers"}
            TOOLS = [{"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}}]
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )

        response = self.client.get(reverse("tools_api"), {"engine": "ollama-service", "model": "llama3"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["tool_servers"],
            [{
                "id": "time_suite",
                "name": "Time Suite",
                "description": "Time helpers",
                "tool_count": 1,
                "tools": [{"id": "time_now", "name": "Current Time", "description": ""}],
            }],
        )

    # Test load chat API returns active tool server id.
    def test_load_chat_api_returns_active_tool_server_id(self):
        chat = Chat.objects.create(title="Chat", active_tool_slug="time_suite")
        Message.objects.create(
            chat=chat,
            role="assistant",
            content="Visible answer",
            llm_transcript=[
                {"role": "assistant", "content": "", "thinking": "Plan"},
                {"role": "tool", "name": "time_suite__time_now", "tool_name": "time_suite__time_now", "content": '{"ok": true}', "server_id": "time_suite", "server_name": "Time Suite", "tool_id": "time_now", "tool_display_name": "Current Time", "arguments": {"label": "now"}},
                {"role": "assistant", "content": "Visible answer"},
            ],
        )

        response = self.client.get(reverse("load_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["active_tool_server_id"], "time_suite")
        self.assertEqual(payload["messages"][0]["content"], "Visible answer")
        self.assertEqual(payload["messages"][0]["activity_segments"][0]["type"], "thought")
        self.assertEqual(payload["messages"][0]["activity_segments"][1]["type"], "tool")

    # Test load chat API returns all active tool server ids.
    def test_load_chat_api_returns_multiple_active_tool_server_ids(self):
        chat = Chat.objects.create(title="Chat", active_tool_slug='["time_suite", "browser"]')
        Message.objects.create(chat=chat, role="user", content="Hello")

        response = self.client.get(reverse("load_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["active_tool_server_ids"], ["time_suite", "browser"])
        self.assertEqual(payload["active_tool_server_id"], "time_suite")

    # Test load chat API returns attachment metadata without inline data.
    def test_load_chat_api_returns_attachment_metadata_without_inline_data(self):
        chat = Chat.objects.create(title="Chat")
        message = Message.objects.create(chat=chat, role="user", content="See file")
        attachment = MessageAttachment.objects.create(
            message=message,
            kind="file",
            name="note.txt",
            mime_type="text/plain",
            data="SGVsbG8=",
            size_bytes=5,
        )

        response = self.client.get(reverse("load_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        serialized = payload["messages"][0]["attachments"][0]
        self.assertEqual(serialized["id"], attachment.id)
        self.assertEqual(serialized["content_url"], f"/api/attachment/attachment/{attachment.id}/content/")
        self.assertNotIn("data_url", serialized)
        self.assertNotIn("extracted_text", serialized)

    # Test attachment content API streams stored bytes on demand.
    def test_attachment_content_api_streams_stored_bytes(self):
        chat = Chat.objects.create(title="Chat")
        message = Message.objects.create(chat=chat, role="user", content="See file")
        attachment = MessageAttachment.objects.create(
            message=message,
            kind="file",
            name="note.txt",
            mime_type="text/plain",
            data="SGVsbG8=",
            size_bytes=5,
        )

        response = self.client.get(reverse("attachment_content_api", args=["attachment", attachment.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"Hello")
        self.assertEqual(response["Content-Type"], "text/plain")

    # Test attachment content API streams legacy image records.
    def test_attachment_content_api_streams_legacy_image_bytes(self):
        chat = Chat.objects.create(title="Chat")
        message = Message.objects.create(chat=chat, role="user", content="See image")
        image = MessageImage.objects.create(
            message=message,
            mime_type="image/png",
            data="SGVsbG8=",
            order=2,
        )

        response = self.client.get(reverse("attachment_content_api", args=["image", image.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content, b"Hello")
        self.assertEqual(response["Content-Type"], "image/png")
        self.assertIn('filename="image-3"', response["Content-Disposition"])

    # Test attachment content API rejects unknown record types.
    def test_attachment_content_api_rejects_unknown_record_type(self):
        response = self.client.get(reverse("attachment_content_api", args=["unknown", 1]))

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"], "Unknown attachment type")

    # Test delete last assistant API returns the user message to regenerate.
    def test_delete_last_assistant_api_returns_user_message_for_regeneration(self):
        chat = Chat.objects.create(title="Chat")
        user_message = Message.objects.create(chat=chat, role="user", content="See this")
        MessageAttachment.objects.create(
            message=user_message,
            kind=MessageAttachmentKind.IMAGE,
            name="photo.png",
            mime_type="image/png",
            data="iVBORw0KGgo=",
            size_bytes=8,
        )
        assistant_message = Message.objects.create(chat=chat, role="assistant", content="Answer")

        response = self.client.delete(reverse("delete_last_assistant_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["user_message"]["content"], "See this")
        self.assertEqual(payload["user_message"]["attachments"][0]["name"], "photo.png")
        self.assertEqual(payload["user_message"]["images"], ["data:image/png;base64,iVBORw0KGgo="])
        self.assertFalse(Message.objects.filter(id=assistant_message.id).exists())

    # Test delete last assistant API rejects chats ending with a user message.
    def test_delete_last_assistant_api_rejects_when_last_message_is_user(self):
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Still pending")

        response = self.client.delete(reverse("delete_last_assistant_api", args=[chat.id]))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Last message is not from assistant")

    # Test delete message API removes only the selected message.
    def test_delete_message_api_removes_selected_message(self):
        chat = Chat.objects.create(title="Chat")
        first = Message.objects.create(chat=chat, role="user", content="First")
        second = Message.objects.create(chat=chat, role="assistant", content="Second")

        response = self.client.delete(reverse("delete_message_api", args=[first.id]))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Message.objects.filter(id=first.id).exists())
        self.assertTrue(Message.objects.filter(id=second.id).exists())

    # A branch is a lossless copy through the selected turn with links on both sides.
    def test_branch_message_api_copies_history_and_exposes_bidirectional_markers(self):
        chat = Chat.objects.create(title="Original", active_tool_slug='["sandbox"]')
        first = Message.objects.create(chat=chat, role="user", content="First")
        MessageAttachment.objects.create(
            message=first,
            kind=MessageAttachmentKind.IMAGE,
            name="image.png",
            mime_type="image/png",
            data="aW1hZ2U=",
            size_bytes=5,
        )
        target = Message.objects.create(
            chat=chat,
            role="assistant",
            content="Answer",
            llm_transcript=[{"role": "assistant", "thinking": "Full thought", "content": "Answer"}],
        )
        Message.objects.create(chat=chat, role="user", content="Must not be copied")

        response = self.client.post(reverse("branch_message_api", args=[target.id]), data="{}", content_type="application/json")

        self.assertEqual(response.status_code, 201)
        child = Chat.objects.get(id=response.json()["chat_id"])
        self.assertEqual(child.active_tool_slug, '["sandbox"]')
        copied = list(child.messages.order_by("created_at", "id"))
        self.assertEqual([item.content for item in copied], ["First", "Answer"])
        self.assertEqual(copied[1].llm_transcript[0]["thinking"], "Full thought")
        self.assertEqual(copied[0].attachments.get().data, "aW1hZ2U=")
        branch = ChatBranch.objects.get(child_chat=child)
        self.assertEqual(branch.source_message_id, target.id)
        self.assertEqual(branch.child_message_id, copied[1].id)

        source_payload = self.client.get(reverse("load_chat_api", args=[chat.id])).json()
        child_payload = self.client.get(reverse("load_chat_api", args=[child.id])).json()
        source_target = next(item for item in source_payload["messages"] if item["id"] == target.id)
        child_target = next(item for item in child_payload["messages"] if item["id"] == copied[1].id)
        self.assertEqual(source_target["branch_links"][0]["direction"], "to_branch")
        self.assertEqual(source_target["branch_links"][0]["chat_id"], str(child.id))
        self.assertEqual(child_target["branch_links"][0]["direction"], "to_origin")
        self.assertEqual(child_target["branch_links"][0]["chat_id"], str(chat.id))

    # Editing a user message invalidates every later turn while keeping attachments.
    def test_edit_message_api_replaces_user_text_and_truncates_continuation(self):
        chat = Chat.objects.create(title="Chat")
        target = Message.objects.create(chat=chat, role="user", content="Old text")
        attachment = MessageAttachment.objects.create(
            message=target,
            kind=MessageAttachmentKind.FILE,
            name="note.txt",
            mime_type="text/plain",
            data="bm90ZQ==",
            size_bytes=4,
        )
        stale = Message.objects.create(chat=chat, role="assistant", content="Stale answer")

        response = self.client.patch(
            reverse("edit_message_api", args=[target.id]),
            data=json.dumps({"content": "New\ntext"}),
            content_type="application/json",
        )

        target.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(target.content, "New\ntext")
        self.assertFalse(Message.objects.filter(id=stale.id).exists())
        self.assertTrue(MessageAttachment.objects.filter(id=attachment.id).exists())

    # The archive contains readable linked Markdown, full transcript JSON and binary assets.
    def test_export_chat_api_returns_lossless_zip(self):
        chat = Chat.objects.create(title="Export test")
        message = Message.objects.create(
            chat=chat,
            role="assistant",
            content="Supported claim [cabc-1]",
            llm_transcript=[
                {"role": "assistant", "thinking": "Do not shorten", "content": "Supported claim [cabc-1]"},
                {
                    "role": "tool",
                    "content": "result",
                    "structured_content": {"sources": [{"id": "cabc-1", "url": "https://example.com/source"}]},
                },
            ],
        )
        MessageAttachment.objects.create(
            message=message,
            kind=MessageAttachmentKind.IMAGE,
            name="chart.png",
            mime_type="image/png",
            data="cG5nLWJ5dGVz",
            size_bytes=9,
        )

        response = self.client.get(reverse("export_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/zip")
        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            self.assertEqual(set(archive.namelist()), {"chat.md", "chat.json", "attachments/chart.png"})
            markdown = archive.read("chat.md").decode("utf-8")
            payload = json.loads(archive.read("chat.json"))
            self.assertIn("[CABC-1](https://example.com/source)", markdown)
            self.assertIn('"thinking": "Do not shorten"', markdown)
            self.assertEqual(payload["messages"][0]["llm_transcript"][0]["thinking"], "Do not shorten")
            self.assertEqual(archive.read("attachments/chart.png"), b"png-bytes")

    # Test rename chat API trims and persists the title.
    def test_rename_chat_api_updates_title(self):
        chat = Chat.objects.create(title="Old")

        response = self.client.patch(
            reverse("rename_chat_api", args=[chat.id]),
            data='{"title":"  New title  "}',
            content_type="application/json",
        )

        chat.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "New title")
        self.assertEqual(chat.title, "New title")

    # Test delete chat API removes the whole thread.
    def test_delete_chat_api_removes_thread_and_messages(self):
        chat = Chat.objects.create(title="Chat")
        Message.objects.create(chat=chat, role="user", content="Hello")

        response = self.client.delete(reverse("delete_chat_api", args=[chat.id]))

        self.assertEqual(response.status_code, 200)
        self.assertFalse(Chat.objects.filter(id=chat.id).exists())
        self.assertEqual(Message.objects.count(), 0)


# Cover Ollama preset API endpoints and model-info integration.
class OllamaPresetApiTests(ToolRegistryTestMixin, TestCase):
    def setUp(self):
        super().setUp()
        self._engine_enabled_patch = patch(
            "Apps.UI.views.settings.is_engine_enabled",
            return_value=True,
        )
        self._engine_enabled_patch.start()

    def tearDown(self):
        self._engine_enabled_patch.stop()
        super().tearDown()

    # Test model info includes active Ollama preset defaults and servers.
    @patch("Apps.UI.views.llm_api.get_model_settings")
    # Verify model info includes active ollama preset defaults and servers.
    def test_model_info_includes_active_ollama_preset_defaults_and_servers(self, mock_get_model_settings):
        self.write_server(
            'time_suite',
            '''
            MCP_SERVER = {"id": "time_suite", "name": "Time Suite", "description": "Time helpers"}
            TOOLS = [
                {"id": "time_now", "name": "Current Time", "parameters": {"type": "object", "properties": {}}},
                {"id": "timezone_name", "name": "Timezone Name", "parameters": {"type": "object", "properties": {}}},
            ]
            def supports(engine=None, model_name=None):
                return engine == "ollama-service"
            def call_tool(tool_id, arguments, context=None):
                return "ok"
            ''',
        )
        OllamaPreset.objects.create(
            model_name="llama3",
            name="Default",
            config={"num_ctx": 32768, "num_predict": 8192},
            is_default=True,
            is_active=False,
        )
        custom_preset = OllamaPreset.objects.create(
            model_name="llama3",
            name="Custom",
            config={"num_ctx": 65536, "mirostat": 2, "numa": True},
            is_default=False,
            is_active=True,
        )
        mock_get_model_settings.return_value = {
            "modelinfo": {"general.architecture.context_length": 131072},
            "parameters": "temperature 0.8\nPARAMETER mirostat 2\nnuma true",
            "template": "",
            "capabilities": ["tools"],
        }

        response = self.client.get(
            reverse("model_info_api"),
            {"engine": "ollama-service", "model": "llama3"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("ollama_presets", payload)
        self.assertTrue(payload["supports_tool_calling"])
        self.assertEqual(payload["available_tool_servers"][0]["id"], "time_suite")
        self.assertEqual(payload["available_tool_servers"][0]["tool_count"], 2)
        self.assertEqual(payload["defaults"]["num_ctx"], 65536)
        self.assertEqual(payload["defaults"]["temperature"], 0.8)
        self.assertEqual(payload["ollama_presets"]["active_preset_id"], str(custom_preset.id))
        self.assertNotIn("mirostat", payload["defaults"])
        self.assertNotIn("numa", payload["defaults"])

    # Test sync endpoint clones default preset on first change.
    def test_sync_endpoint_clones_default_preset_on_first_change(self):
        response = self.client.post(
            reverse("sync_ollama_preset_api"),
            data='{"model":"llama3","config":{"num_ctx":65536,"num_predict":4096,"think":true,"think_level":"high"}}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["presets"]), 2)
        self.assertEqual(OllamaPreset.objects.filter(model_name="llama3").count(), 2)

    # Test create rename delete endpoints manage custom preset.
    def test_create_rename_delete_endpoints_manage_custom_preset(self):
        created = self.client.post(
            reverse("create_ollama_preset_api"),
            data='{"model":"llama3","name":"Research","config":{"num_ctx":49152}}',
            content_type="application/json",
        )
        self.assertEqual(created.status_code, 200)
        created_payload = created.json()
        active_preset_id = created_payload["active_preset_id"]

        renamed = self.client.post(
            reverse("rename_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{active_preset_id}","name":"Research v2"}}',
            content_type="application/json",
        )
        self.assertEqual(renamed.status_code, 200)

        deleted = self.client.post(
            reverse("delete_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{active_preset_id}"}}',
            content_type="application/json",
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(OllamaPreset.objects.filter(model_name="llama3").count(), 1)

    # Test duplicate preset name returns validation error.
    def test_duplicate_preset_name_returns_validation_error(self):
        self.client.post(
            reverse("create_ollama_preset_api"),
            data='{"model":"llama3","name":"Research","config":{"num_ctx":49152}}',
            content_type="application/json",
        )

        duplicate = self.client.post(
            reverse("create_ollama_preset_api"),
            data='{"model":"llama3","name":"Research","config":{"num_ctx":32768}}',
            content_type="application/json",
        )

        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("already exists", duplicate.json()["error"])

    # Test select endpoint activates an existing custom preset.
    def test_select_endpoint_activates_custom_preset(self):
        default_preset = OllamaPreset.objects.create(
            model_name="llama3",
            name="Default",
            config={"num_ctx": 32768},
            is_default=True,
            is_active=True,
        )
        custom_preset = OllamaPreset.objects.create(
            model_name="llama3",
            name="Research",
            config={"num_ctx": 65536},
            is_default=False,
            is_active=False,
        )

        response = self.client.post(
            reverse("select_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{custom_preset.id}"}}',
            content_type="application/json",
        )

        default_preset.refresh_from_db()
        custom_preset.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_preset_id"], str(custom_preset.id))
        self.assertFalse(default_preset.is_active)
        self.assertTrue(custom_preset.is_active)

    # Test default preset mutation errors are returned as validation responses.
    def test_default_preset_mutation_errors_return_400(self):
        default_preset = OllamaPreset.objects.create(
            model_name="llama3",
            name="Default",
            config={"num_ctx": 32768},
            is_default=True,
            is_active=True,
        )

        renamed = self.client.post(
            reverse("rename_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{default_preset.id}","name":"Renamed"}}',
            content_type="application/json",
        )
        deleted = self.client.post(
            reverse("delete_ollama_preset_api"),
            data=f'{{"model":"llama3","preset_id":"{default_preset.id}"}}',
            content_type="application/json",
        )

        self.assertEqual(renamed.status_code, 400)
        self.assertEqual(deleted.status_code, 400)
        self.assertIn("default preset", renamed.json()["error"])
        self.assertIn("default preset", deleted.json()["error"])

# Cover LM Studio preset API endpoints and model-info integration.
class LmsPresetApiTests(TestCase):
    def setUp(self):
        super().setUp()
        self._engine_enabled_patch = patch(
            "Apps.UI.views.settings.is_engine_enabled",
            return_value=True,
        )
        self._engine_enabled_patch.start()

    def tearDown(self):
        self._engine_enabled_patch.stop()
        super().tearDown()

    # Test model info includes active LM Studio preset defaults.
    @patch("Apps.UI.views.llm_api.get_model_settings")
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify model info includes active lms preset defaults.
    def test_model_info_includes_active_lms_preset_defaults(self, mock_preset_settings, mock_model_settings):
        base_settings = {
            "context_length": 65536,
            "defaults": {"temperature": 0.7, "think": True},
            "supports_thinking": True,
            "supports_think_level": False,
            "think_param_name": "think",
            "think_level_param_name": "reasoning_effort",
            "supports_vision": True,
            "supports_tool_calling": True,
            "supports_files": True,
            "runtime_limits": {"gpu_devices": [{"id": 0, "name": "RTX"}], "gpu_count": 1, "main_gpu_max": 0},
            "custom_fields": [],
            "capabilities": ["vision", "tools", "thinking", "files"],
        }
        mock_model_settings.return_value = base_settings
        mock_preset_settings.return_value = base_settings

        default_preset = LmsPreset.objects.create(
            model_name="qwen3",
            name="Default",
            config={"operation": {"temperature": 0.7}},
            is_default=True,
            is_active=False,
        )
        custom_preset = LmsPreset.objects.create(
            model_name="qwen3",
            name="Reasoning Off",
            config={"operation": {"temperature": 0.2, "think": False}},
            is_default=False,
            is_active=True,
        )

        response = self.client.get(
            reverse("model_info_api"),
            {"engine": "lms", "model": "qwen3"},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("lms_presets", payload)
        self.assertEqual(payload["defaults"]["temperature"], 0.2)
        self.assertFalse(payload["defaults"]["think"])
        self.assertEqual(payload["lms_presets"]["active_preset_id"], str(custom_preset.id))
        self.assertEqual(default_preset.name, "Default")

    # Test sync endpoint clones default LM Studio preset on first change.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify sync endpoint clones default lms preset on first change.
    def test_sync_endpoint_clones_default_lms_preset_on_first_change(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }

        response = self.client.post(
            reverse("sync_lms_preset_api"),
            data='{"model":"qwen3","config":{"operation":{"temperature":0.2,"think":false}}}',
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["presets"]), 2)
        self.assertEqual(LmsPreset.objects.filter(model_name="qwen3").count(), 2)

    # Test get LM Studio presets endpoint requires a model.
    def test_get_lms_presets_requires_model(self):
        response = self.client.get(reverse("lms_presets_api"))

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["error"], "Model parameter is required")

    # Test create rename delete endpoints manage a custom LM Studio preset.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify create rename delete endpoints manage custom lms preset.
    def test_create_rename_delete_endpoints_manage_custom_lms_preset(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }

        created = self.client.post(
            reverse("create_lms_preset_api"),
            data='{"model":"qwen3","name":"Research","config":{"operation":{"temperature":0.2}}}',
            content_type="application/json",
        )
        self.assertEqual(created.status_code, 200)
        active_preset_id = created.json()["active_preset_id"]

        renamed = self.client.post(
            reverse("rename_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{active_preset_id}","name":"Research v2"}}',
            content_type="application/json",
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertEqual(LmsPreset.objects.get(id=active_preset_id).name, "Research v2")

        deleted = self.client.post(
            reverse("delete_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{active_preset_id}"}}',
            content_type="application/json",
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(LmsPreset.objects.filter(model_name="qwen3").count(), 1)
        self.assertTrue(LmsPreset.objects.get(model_name="qwen3").is_default)

    # Test duplicate LM Studio preset names return validation errors.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify duplicate lms preset name returns validation error.
    def test_duplicate_lms_preset_name_returns_validation_error(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        self.client.post(
            reverse("create_lms_preset_api"),
            data='{"model":"qwen3","name":"Research","config":{"operation":{"temperature":0.2}}}',
            content_type="application/json",
        )

        duplicate = self.client.post(
            reverse("create_lms_preset_api"),
            data='{"model":"qwen3","name":"Research","config":{"operation":{"temperature":0.4}}}',
            content_type="application/json",
        )

        self.assertEqual(duplicate.status_code, 400)
        self.assertIn("already exists", duplicate.json()["error"])

    # Test select endpoint activates an existing custom LM Studio preset.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify select endpoint activates custom lms preset.
    def test_select_endpoint_activates_custom_lms_preset(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        default_preset = LmsPreset.objects.create(
            model_name="qwen3",
            name="Default",
            config={"operation": {"temperature": 0.7}},
            is_default=True,
            is_active=True,
        )
        custom_preset = LmsPreset.objects.create(
            model_name="qwen3",
            name="Research",
            config={"operation": {"temperature": 0.2}},
            is_default=False,
            is_active=False,
        )

        response = self.client.post(
            reverse("select_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{custom_preset.id}"}}',
            content_type="application/json",
        )

        default_preset.refresh_from_db()
        custom_preset.refresh_from_db()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["active_preset_id"], str(custom_preset.id))
        self.assertFalse(default_preset.is_active)
        self.assertTrue(custom_preset.is_active)

    # Test default LM Studio preset mutation errors are returned as validation responses.
    @patch("Apps.Data.lms_presets.lms_api.get_model_settings")
    # Verify default lms preset mutation errors return 400.
    def test_default_lms_preset_mutation_errors_return_400(self, mock_get_model_settings):
        mock_get_model_settings.return_value = {
            "defaults": {"temperature": 0.7},
        }
        default_preset = LmsPreset.objects.create(
            model_name="qwen3",
            name="Default",
            config={"operation": {"temperature": 0.7}},
            is_default=True,
            is_active=True,
        )

        renamed = self.client.post(
            reverse("rename_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{default_preset.id}","name":"Renamed"}}',
            content_type="application/json",
        )
        deleted = self.client.post(
            reverse("delete_lms_preset_api"),
            data=f'{{"model":"qwen3","preset_id":"{default_preset.id}"}}',
            content_type="application/json",
        )

        self.assertEqual(renamed.status_code, 400)
        self.assertEqual(deleted.status_code, 400)
        self.assertIn("default preset", renamed.json()["error"])
        self.assertIn("default preset", deleted.json()["error"])


# Verify the three critical fixes: message IDs in headers, no user duplication
# on regenerate, and chat.updated_at bumped on every mutation.
class MessageIdAndRegenerateTests(ToolRegistryTestMixin, TestCase):
    # Prepare shared fixtures for each test case.
    def setUp(self):
        super().setUp()
        self.client = Client()

    # chat_api must return X-User-Message-ID and X-Assistant-Message-ID headers
    # so the frontend can stamp fresh rows without waiting for a page reload.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat api returns message id headers.
    def test_chat_api_returns_message_id_headers(self, _mock_engine, mock_generate, _mock_runtime):
        mock_generate.return_value = [{"message": {"content": "Hi"}}]

        response = self.client.post(
            reverse("chat_api"),
            data='{"message":"Hello","model":"llama3"}',
            content_type="application/json",
        )
        b"".join(response.streaming_content)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.has_header("X-User-Message-ID"), "X-User-Message-ID header missing")
        self.assertTrue(response.has_header("X-Assistant-Message-ID"), "X-Assistant-Message-ID header missing")

        user_id = int(response["X-User-Message-ID"])
        assistant_id = int(response["X-Assistant-Message-ID"])
        self.assertTrue(Message.objects.filter(id=user_id, role="user").exists())
        self.assertTrue(Message.objects.filter(id=assistant_id, role="assistant").exists())

    # After a normal send + regenerate the chat must contain exactly one user
    # message — not two copies of the same prompt.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify regenerate does not duplicate user message.
    def test_regenerate_does_not_duplicate_user_message(self, _mock_engine, mock_generate, _mock_runtime):
        mock_generate.return_value = [{"message": {"content": "Answer"}}]

        r1 = self.client.post(
            reverse("chat_api"),
            data='{"message":"hello","model":"llama3"}',
            content_type="application/json",
        )
        b"".join(r1.streaming_content)
        chat_id = r1["X-Chat-ID"]

        self.assertEqual(Message.objects.filter(chat__id=chat_id, role="user").count(), 1)

        mock_generate.return_value = [{"message": {"content": "New answer"}}]
        r2 = self.client.post(
            reverse("regenerate_chat_api", args=[chat_id]),
            data='{"model":"llama3"}',
            content_type="application/json",
        )
        b"".join(r2.streaming_content)

        self.assertEqual(r2.status_code, 200)
        user_count = Message.objects.filter(chat__id=chat_id, role="user").count()
        self.assertEqual(user_count, 1, f"Expected 1 user message after regenerate, got {user_count}")
        assistant_count = Message.objects.filter(chat__id=chat_id, role="assistant").count()
        self.assertEqual(assistant_count, 1, f"Expected 1 assistant message after regenerate, got {assistant_count}")

    # chat.updated_at must be bumped when messages are added so the sidebar
    # sort order stays correct.
    @patch("Apps.UI.views.llm_api.prepare_runtime")
    @patch("Apps.UI.views.llm_api.generate")
    @patch("Apps.UI.views._resolve_request_engine", return_value="ollama-service")
    # Verify chat updated at is bumped after generation.
    def test_chat_updated_at_is_bumped_after_generation(self, _mock_engine, mock_generate, _mock_runtime):
        import time

        mock_generate.return_value = [{"message": {"content": "Hi"}}]

        chat = Chat.objects.create(title="Test")
        ts_before = chat.updated_at

        time.sleep(0.05)

        response = self.client.post(
            reverse("chat_api"),
            data=f'{{"message":"Hi","model":"llama3","chat_id":"{chat.id}"}}',
            content_type="application/json",
        )
        b"".join(response.streaming_content)

        chat.refresh_from_db()
        self.assertGreater(chat.updated_at, ts_before, "chat.updated_at was not bumped after generation")
