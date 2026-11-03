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
from Apps.API.views import _build_chat_history, _chat_is_first_user_turn, _compose_system_prompt


pytestmark = [pytest.mark.component]


# Cover the skills REST endpoints, front matter parsing, prompt inventory and sandbox mirroring against a
# temporary Skills folder.
class SkillsApiTests(TestCase):
    # Point the skills configuration at a temporary project root, so tests never touch the real Skills folder,
    # and reset the pending refresh marker.
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

    # Reset the pending refresh marker, undo the path patches and remove the temporary project root.
    def tearDown(self):
        skills_config.clear_skill_config_refresh_pending()
        for patcher in reversed(self._patches):
            patcher.stop()
        self._tmp.cleanup()
        super().tearDown()

    # Verify the skills CRUD flow: listing creates the Skills root, a skill folder can be created, a file can
    # be saved in it, the folder can be renamed, and saving a file through a ".." path returns a 400 error.
    def test_skills_root_created_and_crud_validates_paths(self):
        # Listing skills creates the missing Skills root.
        response = self.client.get(reverse("skills_api"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(self.skills_dir.is_dir())

        # Create a skill folder; it gets a default SKILL.md.
        created = self.client.post(
            reverse("skills_api"),
            data=json.dumps({"name": "skill-creator"}),
            content_type="application/json",
        )
        self.assertEqual(created.status_code, 200)
        self.assertTrue((self.skills_dir / "skill-creator" / "SKILL.md").is_file())

        # Save a file in a new subdirectory of the skill.
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

        # Rename the skill folder; the listing reports the new title.
        renamed = self.client.patch(
            reverse("skills_folder_api"),
            data=json.dumps({"old_name": "skill-creator", "new_name": "renamed-skill"}),
            content_type="application/json",
        )
        self.assertEqual(renamed.status_code, 200)
        self.assertTrue((self.skills_dir / "renamed-skill" / "SKILL.md").is_file())
        self.assertEqual(renamed.json()["folders"][0]["title"], "renamed-skill")

        # Saving a file outside the skill folder through ".." is rejected.
        rejected = self.client.put(
            reverse("skills_file_api"),
            data=json.dumps({"folder": "renamed-skill", "file": "../bad.md", "content": "x"}),
            content_type="application/json",
        )
        self.assertEqual(rejected.status_code, 400)

    # Verify that the skills listing summarizes a skill from its front matter, that the first-turn system
    # prompt lists the enabled skill with its files, and that later prompts and disabled skills omit it.
    def test_front_matter_summary_and_prompt_inventory(self):
        # Arrange: an enabled skill with front matter and one extra markdown file.
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

        # The listing reports the front matter title and description, and a skill added by someone else is
        # marked as downloaded.
        payload = self.client.get(reverse("skills_api")).json()
        folder = payload["folders"][0]
        self.assertEqual(folder["title"], "skill-creator")
        self.assertEqual(folder["description"], "Create and improve skills.")
        self.assertEqual(folder["source"], "download")
        self.assertIsInstance(folder["created_at"], float)
        self.assertGreater(folder["created_at"], 0)

        # The first-turn prompt lists the skill, its sandbox path and its files; a follow-up prompt does not.
        prompt = _compose_system_prompt("", include_skills_baseline=True)
        self.assertIn("Your skills:", prompt)
        self.assertIn("/workspace/_sandbox/Skills/skill-creator", prompt)
        self.assertIn("agents/grader.md", prompt)

        follow_up_prompt = _compose_system_prompt("")
        self.assertNotIn("Your skills:", follow_up_prompt)

        # A skill disabled in its front matter never appears in the prompt.
        disabled_root = self.skills_dir / "disabled-skill"
        disabled_root.mkdir()
        (disabled_root / "SKILL.md").write_text(
            "---\nname: disabled\ndescription: Off\nenabled: false\n---\n",
            encoding="utf-8",
        )
        prompt_with_disabled = _compose_system_prompt("")
        self.assertNotIn("disabled-skill", prompt_with_disabled)

    # Verify that disabling the only enabled skill queues a one-shot configuration update that the next
    # prompt carries (saying no skills are enabled) and that the prompt after it no longer repeats.
    def test_disable_skill_queues_refreshed_inventory_for_next_prompt(self):
        # Arrange: one enabled skill, and a first-turn prompt that lists it without any update notice.
        skill_root = self.skills_dir / "pdf"
        skill_root.mkdir(parents=True)
        (skill_root / "SKILL.md").write_text(
            "---\nname: pdf\ndescription: PDF skill\nenabled: true\n---\n",
            encoding="utf-8",
        )

        baseline_prompt = _compose_system_prompt("", include_skills_baseline=True)
        self.assertIn("Your skills:", baseline_prompt)
        self.assertNotIn("Skill configuration update:", baseline_prompt)

        # Act: disable the skill through the API.
        response = self.client.patch(
            reverse("skills_enabled_api"),
            data=json.dumps({"folder": "pdf", "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

        # Assert: the next prompt carries the update once, and the one after it carries nothing.
        prompt_after_disable = _compose_system_prompt("")
        self.assertIn("Skill configuration update:", prompt_after_disable)
        self.assertIn("no enabled project skills", prompt_after_disable)
        self.assertNotIn("Your skills:", prompt_after_disable)

        prompt_after_consume = _compose_system_prompt("")
        self.assertNotIn("Skill configuration update:", prompt_after_consume)
        self.assertNotIn("Your skills:", prompt_after_consume)

    # Verify that syncing copies a skill into the sandbox, overwrites a stale copy there and removes files
    # in the sandbox that have no counterpart in the project Skills folder.
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

    # Verify that syncing copies an enabled skill into the sandbox and removes the stale sandbox copy of a
    # disabled skill.
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

    # Verify that disabling a skill through the API removes its folder from the sandbox mirror.
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

    # Verify that a subdirectory, including a nested one, can be created inside a skill folder.
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

    # Verify that creating a subdirectory that already exists returns a 400 error.
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

    # Verify that creating a subdirectory with a ".." path that escapes the skill folder returns a 400 error.
    def test_create_skill_subdirectory_rejects_traversal(self):
        skill_root = self.skills_dir / "my-skill"
        skill_root.mkdir(parents=True)

        response = self.client.post(
            reverse("skills_directory_api"),
            data=json.dumps({"folder": "my-skill", "path": "../escape"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    # Verify that creating a subdirectory whose name looks like a file (with an extension) returns a 400 error.
    def test_create_skill_subdirectory_rejects_file_extension(self):
        skill_root = self.skills_dir / "my-skill"
        skill_root.mkdir(parents=True)

        response = self.client.post(
            reverse("skills_directory_api"),
            data=json.dumps({"folder": "my-skill", "path": "agents.md"}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)

    # Verify that a skill subdirectory can be renamed with its files and then deleted.
    def test_rename_and_delete_skill_subdirectory(self):
        # Arrange: a skill with an agents subdirectory that holds one file.
        skill_root = self.skills_dir / "my-skill"
        skill_root.mkdir(parents=True)
        (skill_root / "agents").mkdir()
        (skill_root / "agents" / "grader.md").write_text("# Grader\n", encoding="utf-8")

        # Rename the subdirectory; its file moves with it.
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

        # Delete the renamed subdirectory.
        deleted = self.client.delete(
            reverse("skills_directory_api"),
            data=json.dumps({"folder": "my-skill", "path": "reviewers"}),
            content_type="application/json",
        )
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse((skill_root / "reviewers").exists())

    # Verify that a file inside a skill folder can be renamed.
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

    # Verify that importing creates the skill folder with its valid files, silently skips files with a ".."
    # path or a NUL character in the path, and lists the new skill with its front matter description.
    def test_import_skill_creates_folder_and_files(self):
        # The last two entries are invalid and must be skipped without failing the import.
        files = [
            {"path": "SKILL.md", "content": "---\nname: imported\ndescription: Test import\nenabled: true\n---\n\n# Imported\n"},
            {"path": "agents/grader.md", "content": "# Grader\n"},
            {"path": "../escape.md", "content": "bad"},
            {"path": "binary\x00.md", "content": "bad"},
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

    # Verify that importing into an existing skill folder adds the new files to it and keeps it listed.
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


# Cover what the model receives in its system context when skills are enabled, disabled and toggled.
class SkillsModelContextTests(TestCase):
    # Point the skills configuration at a temporary project root, so tests never touch the real Skills folder,
    # and reset the pending refresh marker.
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

    # Reset the pending refresh marker, undo the path patches and remove the temporary project root.
    def tearDown(self):
        skills_config.clear_skill_config_refresh_pending()
        for patcher in reversed(self._patches):
            patcher.stop()
        self._tmp.cleanup()
        super().tearDown()

    # Assert that the text lists the skills inventory with a sandbox path for each given skill folder.
    def _assert_has_skills_inventory(self, text: str, *folders: str) -> None:
        self.assertIn("Your skills:", text)
        for folder in folders:
            self.assertIn(f"/workspace/_sandbox/Skills/{folder}", text)

    # Assert that the text has no skills inventory, no configuration update notice and no skill sandbox path.
    def _assert_no_skill_context(self, text: str) -> None:
        self.assertNotIn("Your skills:", text)
        self.assertNotIn("Skill configuration update:", text)
        self.assertNotIn("/workspace/_sandbox/Skills/", text)

    # Assert that the text starts the skills configuration update notice that follows an enable toggle.
    def _assert_config_update_header(self, text: str) -> None:
        self.assertIn("Skill configuration update:", text)
        self.assertIn("changed which project skills are enabled", text)

    # Create a skill folder whose SKILL.md front matter uses the given title (default: the folder name) and
    # enabled flag.
    def _write_skill(self, folder: str, *, enabled: bool = True, title: str | None = None) -> None:
        skill_root = self.skills_dir / folder
        skill_root.mkdir(parents=True, exist_ok=True)
        display = title or folder
        (skill_root / "SKILL.md").write_text(
            f"---\nname: {display}\ndescription: Test\nenabled: {'true' if enabled else 'false'}\n---\n",
            encoding="utf-8",
        )

    # Compose the system prompt with the given skill notification and baseline options.
    def _compose(self, *, consume: bool = True, include_baseline: bool = False, user_prompt: str = "") -> str:
        return _compose_system_prompt(
            user_prompt,
            consume_skill_notifications=consume,
            include_skills_baseline=include_baseline,
        )

    # Build the chat history for a new chat from the given system prompt and return the content of its
    # single system message.
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

    # Disable the given skill through the enabled endpoint and check for a 200 response.
    def _disable_via_api(self, folder: str) -> None:
        response = self.client.patch(
            reverse("skills_enabled_api"),
            data=json.dumps({"folder": folder, "enabled": False}),
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 200)

    # Verify that enabled skills are listed in the first-turn prompt and in the system message built from
    # it, and that a follow-up prompt without the baseline lists none.
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

    # Verify that the baseline flag derived from the chat (first user turn) adds the skills inventory to
    # the prompt, and that it is dropped once the chat has a user message.
    def test_chat_api_first_turn_includes_skills_baseline(self):
        self._write_skill("writer", enabled=True)
        chat = Chat.objects.create(title="Skills baseline chat")
        prompt = _compose_system_prompt("", include_skills_baseline=_chat_is_first_user_turn(chat))
        self._assert_has_skills_inventory(prompt, "writer")

        chat.messages.create(role="user", content="hello")
        prompt_after_history = _compose_system_prompt("", include_skills_baseline=_chat_is_first_user_turn(chat))
        self._assert_no_skill_context(prompt_after_history)

    # Verify that a skill disabled in its front matter is left out of the skills inventory.
    def test_static_disabled_skill_is_omitted_from_inventory(self):
        self._write_skill("writer", enabled=True)
        self._write_skill("legacy-off", enabled=False, title="Legacy")

        composed = self._compose(include_baseline=True)
        self._assert_has_skills_inventory(composed, "writer")
        self.assertNotIn("legacy-off", composed)

    # Verify that disabling a skill makes the next consuming prompt carry the update notice with the
    # remaining skills, and that the prompt after it carries no skill context.
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

    # Verify that composing a prompt without consuming (as the context usage view does) shows the pending
    # update but keeps it pending, and that a consuming compose clears it.
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

    # Verify that re-enabling a disabled skill queues an update notice that lists it, and that the notice
    # is sent only once.
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

    # Verify that setting a skill to the state it already has queues no update notice.
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

    # Verify that disabling a skill still returns 200 and queues the update notice when the sandbox sync
    # raises an error.
    @patch.object(skills_config, "sync_skills_to_sandbox", side_effect=RuntimeError("sandbox unavailable"))
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


# Cover the skills refresh that sandbox tool dispatch performs before executing a tool.
class SkillsSandboxDispatchTests(SimpleTestCase):
    # Verify that calling a sandbox tool syncs the skills into the sandbox once and still returns the tool
    # result.
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
