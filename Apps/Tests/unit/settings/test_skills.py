# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Settings import skills


pytestmark = [pytest.mark.unit]


# Standard SKILL.md document used by most tests: front matter (name, description, source, enabled) and a body.
SKILL_BODY = "---\nname: My Skill\ndescription: Does useful things\nsource: local\nenabled: true\n---\n\n# My Skill\n\nBody text.\n"


# Redirect the skills folders, the sandbox mirror and the pending-notify file into a temporary module directory and
# return their paths as base, root and sandbox.
@pytest.fixture
def isolated_skills(tmp_path, monkeypatch) -> SimpleNamespace:
    # Mirror the real module layout so sandbox sync confinement checks hold.
    base = tmp_path / "module"
    skills_root = base / "Skills"
    sandbox_skills = base / "Tools" / "mcp-sandbox" / "_sandbox" / "Skills"
    monkeypatch.setattr(skills, "BASE_DIR", base)
    monkeypatch.setattr(skills, "SKILLS_DIR", skills_root)
    monkeypatch.setattr(skills, "SANDBOX_SKILLS_DIR", sandbox_skills)
    monkeypatch.setattr(skills, "_PENDING_NOTIFY_PATH", base / ".aslm" / "skills-pending-notify.json")
    skills.clear_skill_config_refresh_pending()
    return SimpleNamespace(base=base, root=skills_root, sandbox=sandbox_skills)


# Create one skill folder with the standard body.
def _create_skill(name: str, body: str = SKILL_BODY) -> None:
    skills.create_skill_folder(name)
    skills.write_skill_file(name, "SKILL.md", body)


# Front matter parsing

# Verify that the front matter metadata is extracted (with scalar values converted) and stripped from the body.
def test_parse_front_matter_extracts_metadata_and_body():
    meta, body = skills.parse_front_matter(SKILL_BODY)

    assert meta["name"] == "My Skill"
    assert meta["description"] == "Does useful things"
    assert meta["enabled"] is True
    assert body.lstrip().startswith("# My Skill")


# Verify that a document without a front matter block gives empty metadata and the text unchanged.
def test_parse_front_matter_without_block_returns_text():
    text = "# Just markdown\n\nNo front matter here.\n"

    meta, body = skills.parse_front_matter(text)

    assert meta == {}
    assert body == text


# Verify that front matter keys are normalized to snake case, scalars are mapped to Python values (yes/off to
# booleans, quoted text to a string) and comment or value-less lines are skipped.
def test_parse_front_matter_normalizes_keys_and_scalars():
    content = "---\nDisplay Name: yes\nanswer: \"42\"\nflag: off\n# comment\nnovalue\n---\nbody"

    meta, body = skills.parse_front_matter(content)

    assert meta == {"display_name": True, "answer": "42", "flag": False}
    assert body == "body"


# Name and path validation

# Verify that unsafe skill folder names are rejected with ValueError and nothing is created in the skills root.
@pytest.mark.parametrize(
    "name",
    ["", "   ", "../escape", "bad/name", "bad\\name", ".hidden", "C:\\evil", "bad$name"],
)
def test_create_skill_folder_rejects_unsafe_names(isolated_skills, name):
    with pytest.raises(ValueError):
        skills.create_skill_folder(name)
    assert not isolated_skills.root.exists() or not any(isolated_skills.root.iterdir())


# Verify that a safe name creates a folder with a SKILL.md template rendered for that name, and the returned
# listing reports it as an enabled local skill.
def test_create_skill_folder_accepts_safe_names_and_writes_template(isolated_skills):
    payload = skills.create_skill_folder("Code Review Helper")

    folder = isolated_skills.root / "Code Review Helper"
    assert folder.is_dir()

    created = next(item for item in payload["folders"] if item["name"] == "Code Review Helper")
    assert created["primary_file"] == "SKILL.md"
    assert created["enabled"] is True
    assert created["source"] == "local"

    content = (folder / "SKILL.md").read_text(encoding="utf-8")
    assert content.startswith("---\nname: Code Review Helper")
    assert "# Code Review Helper" in content


# Verify that creating a skill folder that already exists fails with an "already exists" error.
def test_create_duplicate_skill_folder_fails(isolated_skills):
    skills.create_skill_folder("Duplicated")

    with pytest.raises(ValueError, match="already exists"):
        skills.create_skill_folder("Duplicated")


# Verify that writing a file with an unsafe path (escape, disallowed extension, drive path, control character or
# hidden name) inside a skill folder fails with ValueError.
@pytest.mark.parametrize(
    "path",
    ["../escape.txt", "sub/../../escape.txt", "notes.exe", "C:/evil.txt", "file\x00.txt", ".hidden.md"],
)
def test_write_skill_file_rejects_unsafe_paths(isolated_skills, path):
    _create_skill("Guarded")

    with pytest.raises(ValueError):
        skills.write_skill_file("Guarded", path, "content")


# Verify that creating a subdirectory with an unsafe path (escape, extension, drive path or hidden name) inside a
# skill folder fails with ValueError.
@pytest.mark.parametrize(
    "path",
    ["../escape", "nested/with.ext", "C:/evil", ".hidden"],
)
def test_create_subdirectory_rejects_unsafe_paths(isolated_skills, path):
    _create_skill("Guarded")

    with pytest.raises(ValueError):
        skills.create_skill_subdirectory("Guarded", path)


# Editing skill files

# Verify that a file written into a nested path is read back with the same content, empty front matter and the
# normalized relative path.
def test_skill_file_write_read_roundtrip(isolated_skills):
    _create_skill("Editable")

    skills.write_skill_file("Editable", "docs/notes.md", "# Notes\n\nRemember this.\n")
    payload = skills.read_skill_file("Editable", "docs/notes.md")

    assert payload["content"] == "# Notes\n\nRemember this.\n"
    assert payload["front_matter"] == {}
    assert payload["file"] == "docs/notes.md"


# Verify that reading the primary SKILL.md also returns its parsed front matter.
def test_read_skill_file_returns_front_matter(isolated_skills):
    _create_skill("Editable")

    payload = skills.read_skill_file("Editable", "SKILL.md")

    assert payload["front_matter"]["name"] == "My Skill"
    assert payload["front_matter"]["enabled"] is True


# Verify that content larger than the maximum text file size is rejected with a "too large" error.
def test_write_skill_file_rejects_oversized_content(isolated_skills):
    _create_skill("Editable")

    with pytest.raises(ValueError, match="too large"):
        skills.write_skill_file("Editable", "SKILL.md", "x" * (skills.MAX_TEXT_FILE_BYTES + 1))


# Verify that deleting a file also removes the parent directories it leaves empty.
def test_delete_skill_file_prunes_empty_parents(isolated_skills):
    _create_skill("Editable")
    skills.write_skill_file("Editable", "deep/nested/notes.md", "content")

    skills.delete_skill_file("Editable", "deep/nested/notes.md")

    assert not (isolated_skills.root / "Editable" / "deep").exists()


# Folder lifecycle

# Verify that renaming a skill folder moves it and updates the name in the front matter of its primary file.
def test_rename_skill_folder_updates_front_matter_name(isolated_skills):
    _create_skill("Old Name")

    skills.rename_skill_folder("Old Name", "New Name")

    renamed = isolated_skills.root / "New Name"
    assert renamed.is_dir()
    meta, _body = skills.parse_front_matter((renamed / "SKILL.md").read_text(encoding="utf-8"))
    assert meta["name"] == "New Name"


# Verify that renaming a skill folder onto an existing folder fails with an "already exists" error.
def test_rename_skill_folder_rejects_existing_destination(isolated_skills):
    _create_skill("First")
    _create_skill("Second")

    with pytest.raises(ValueError, match="already exists"):
        skills.rename_skill_folder("First", "Second")


# Verify that deleting a skill folder removes the whole folder tree, including its subdirectories.
def test_delete_skill_folder_removes_tree(isolated_skills):
    _create_skill("Doomed")
    skills.create_skill_subdirectory("Doomed", "assets")

    skills.delete_skill_folder("Doomed")

    assert not (isolated_skills.root / "Doomed").exists()


# Verify that renaming a file item moves it inside the skill folder: the new name has the content and the old name
# no longer exists.
def test_rename_skill_item_moves_files(isolated_skills):
    _create_skill("Mover")
    skills.write_skill_file("Mover", "draft.md", "text")

    skills.rename_skill_item("Mover", "draft.md", "final.md", "file")

    payload = skills.read_skill_file("Mover", "final.md")
    assert payload["content"] == "text"
    with pytest.raises(FileNotFoundError):
        skills.read_skill_file("Mover", "draft.md")


# Enable toggles

# Verify that the enabled flag parser accepts the formats arriving from the API: booleans, yes/off strings,
# numbers, an empty string as disabled and unrecognized text as enabled.
def test_parse_enabled_flag_covers_input_formats():
    assert skills.parse_enabled_flag(True) is True
    assert skills.parse_enabled_flag(False) is False
    assert skills.parse_enabled_flag("yes") is True
    assert skills.parse_enabled_flag("off") is False
    assert skills.parse_enabled_flag("") is False
    assert skills.parse_enabled_flag(1) is True
    assert skills.parse_enabled_flag("maybe") is True


# Verify that disabling a skill writes enabled: false into its front matter and removes it from the sandbox
# mirror.
def test_set_skill_enabled_writes_front_matter_and_mirrors_sandbox(isolated_skills):
    _create_skill("Toggle")

    skills.set_skill_enabled("Toggle", False)

    meta, _body = skills.parse_front_matter(
        (isolated_skills.root / "Toggle" / "SKILL.md").read_text(encoding="utf-8")
    )
    assert meta["enabled"] is False
    assert not (isolated_skills.sandbox / "Toggle").exists()


# Verify that syncing mirrors an enabled skill and its nested files into the sandbox skills folder.
def test_set_skill_enabled_syncs_enabled_skills_into_sandbox(isolated_skills):
    _create_skill("Enabled Skill")
    skills.write_skill_file("Enabled Skill", "refs/guide.md", "guide")

    skills.sync_skills_to_sandbox()

    mirrored = isolated_skills.sandbox / "Enabled Skill"
    assert (mirrored / "SKILL.md").is_file()
    assert (mirrored / "refs" / "guide.md").is_file()


# Sandbox sync

# Verify that a sync removes stale files from the sandbox mirror and copies source files that changed.
def test_sync_updates_changed_files_and_removes_stale_ones(isolated_skills):
    _create_skill("Synced")
    skills.sync_skills_to_sandbox()

    # An out-of-band file in the sandbox mirror is pruned on the next sync.
    stale = isolated_skills.sandbox / "Synced" / "stale.md"
    stale.write_text("stale", encoding="utf-8")
    manifest = skills.sync_skills_to_sandbox()

    assert not stale.exists()
    assert manifest["removed"] >= 1

    # Editing the source file re-copies it into the mirror.
    skills.write_skill_file("Synced", "SKILL.md", SKILL_BODY + "\nExtra section.\n")
    manifest = skills.sync_skills_to_sandbox()

    assert manifest["copied"] >= 1
    assert "Extra section." in (isolated_skills.sandbox / "Synced" / "SKILL.md").read_text(encoding="utf-8")


# Browser imports

# Verify that an imported file tree skips unsafe or disallowed files and normalizes the primary file: the declared
# name is kept, legacy attribution keys are dropped and the source becomes local.
def test_import_skill_files_writes_tree_and_normalizes_primary(isolated_skills):
    payload = skills.import_skill_files(
        "Imported",
        [
            {
                "path": "SKILL.md",
                "content": "---\nname: Raw\ndescription: Imported\nsource: download\nadded_by: someuser\nenabled: true\n---\n\n# Raw\n",
            },
            {"path": "scripts/run.py", "content": "print('hi')\n"},
            {"path": "../escape.md", "content": "skip me"},
            {"path": "binary.exe", "content": "skip me"},
        ],
    )

    imported = isolated_skills.root / "Imported"
    assert (imported / "scripts" / "run.py").is_file()
    assert not (imported / "escape.md").exists()

    meta, _body = skills.parse_front_matter((imported / "SKILL.md").read_text(encoding="utf-8"))

    # The imported document's declared name is preserved.
    assert meta["name"] == "Raw"

    # Legacy attribution keys are stripped before storage.
    assert "added_by" not in meta
    assert meta["source"] == "local"
    assert next(item for item in payload["folders"] if item["name"] == "Imported")["primary_file"] == "SKILL.md"


# Verify that an import without a SKILL.md gets a synthesized one named after the folder.
def test_import_skill_files_synthesizes_primary_when_absent(isolated_skills):
    skills.import_skill_files("No Primary", [{"path": "notes.md", "content": "note"}])

    content = (isolated_skills.root / "No Primary" / "SKILL.md").read_text(encoding="utf-8")
    assert content.startswith("---\nname: No Primary")
    assert "Imported skill." in content


# Verify that an import whose files are all invalid fails with a "No valid files" error.
def test_import_skill_files_rejects_empty_input(isolated_skills):
    with pytest.raises(ValueError, match="No valid files"):
        skills.import_skill_files("Empty", [{"path": "../only.md", "content": "x"}])


# System prompt inventory

# Verify that the system prompt inventory lists only enabled skills, with their sandbox paths and files.
def test_inventory_lists_only_enabled_skills(isolated_skills):
    _create_skill("Active Skill")
    _create_skill("Sleeping Skill")
    skills.set_skill_enabled("Sleeping Skill", False)

    inventory = skills.build_system_prompt_inventory()

    assert "Active Skill" in inventory
    assert "Sleeping Skill" not in inventory
    assert "/workspace/_sandbox/Skills/Active Skill" in inventory
    assert "SKILL.md" in inventory


# Verify that the inventory is an empty string when there are no skills.
def test_inventory_empty_when_no_skills(isolated_skills):
    assert skills.build_system_prompt_inventory() == ""


# Verify that nested markdown files appear in the inventory tree, with their directory and relative path.
def test_inventory_includes_tree_files(isolated_skills):
    _create_skill("Tree Skill")
    skills.write_skill_file("Tree Skill", "guides/deep.md", "deep")

    inventory = skills.build_system_prompt_inventory()

    assert "guides/" in inventory
    assert "guides/deep.md" in inventory


# Pending refresh handling

# Verify that the skills section is empty by default and lists the skills only when the baseline is requested,
# and that after a skill is disabled the pending configuration update is reported once and then consumed.
def test_skills_section_reports_pending_config_refresh_once(isolated_skills):
    _create_skill("Promoted")

    assert skills.build_system_prompt_skills_section() == ""
    assert "Promoted" in skills.build_system_prompt_skills_section(include_baseline=True)

    skills.set_skill_enabled("Promoted", False)

    section = skills.build_system_prompt_skills_section()
    assert section.startswith("Skill configuration update:")
    assert "no enabled project skills" in section

    # The pending flag is consumed by the first read.
    assert skills.build_system_prompt_skills_section() == ""


# Verify that a pending refresh stays pending across reads that do not consume it and is removed by an explicit
# clear.
def test_pending_refresh_survives_reload_and_can_be_cleared(isolated_skills):
    _create_skill("Persisted")
    skills.set_skill_enabled("Persisted", False)
    assert skills.build_system_prompt_skills_section(consume=False)

    skills.clear_skill_config_refresh_pending()

    assert skills.build_system_prompt_skills_section(consume=False) == ""


# Verify that a corrupt pending-notify file is treated as "no refresh pending" instead of raising.
def test_notify_state_survives_invalid_file(isolated_skills):
    isolated_skills.base.joinpath(".aslm").mkdir(parents=True, exist_ok=True)
    isolated_skills.base.joinpath(".aslm", "skills-pending-notify.json").write_text("{broken", encoding="utf-8")

    assert skills.build_system_prompt_skills_section(consume=False) == ""


# Verify that a legacy notify file in the skills root is migrated into a pending refresh and removed.
def test_legacy_notify_file_migrates_to_pending_refresh(isolated_skills):
    isolated_skills.root.mkdir(parents=True, exist_ok=True)
    isolated_skills.root.joinpath(".skills-pending-notify.json").write_text("{}", encoding="utf-8")

    section = skills.build_system_prompt_skills_section(consume=False)

    # The legacy file is migrated into a pending refresh notification.
    assert section.startswith("Skill configuration update:")
    assert not isolated_skills.root.joinpath(".skills-pending-notify.json").exists()


# Listing

# Verify that hidden folders and files of a disallowed type are left out of the skill listing.
def test_list_skills_filters_hidden_folders_and_binary_files(isolated_skills):
    isolated_skills.root.mkdir(parents=True, exist_ok=True)
    hidden = isolated_skills.root / ".hidden"
    hidden.mkdir()
    (hidden / "SKILL.md").write_text("x", encoding="utf-8")

    skills.create_skill_folder("Filtered")
    (isolated_skills.root / "Filtered" / "payload.bin").write_bytes(b"\x00\x01")

    payload = skills.list_skills()

    assert [item["name"] for item in payload["folders"]] == ["Filtered"]
    tree_names = [node["name"] for node in payload["folders"][0]["tree"]]
    assert "payload.bin" not in tree_names
    assert "SKILL.md" in tree_names
