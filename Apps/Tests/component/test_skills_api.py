# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch
from django.test import Client, SimpleTestCase, TestCase
from django.urls import reverse
from API import mcp as tool_registry
from Settings import skills as skills_config
from Apps.Data.models import Chat, Message
from Apps.UI.views import _build_chat_history, _chat_is_first_user_turn, _compose_system_prompt


pytestmark = [pytest.mark.component]


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
