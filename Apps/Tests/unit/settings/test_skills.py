# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from types import SimpleNamespace

import pytest

from Settings import skills


pytestmark = [pytest.mark.unit]


SKILL_BODY = "---\nname: My Skill\ndescription: Does useful things\nsource: local\nenabled: true\n---\n\n# My Skill\n\nBody text.\n"


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


# Front matter parsing.


# Front matter metadata is extracted and stripped from the body.
def test_parse_front_matter_extracts_metadata_and_body():
    meta, body = skills.parse_front_matter(SKILL_BODY)

    assert meta["name"] == "My Skill"
    assert meta["description"] == "Does useful things"
    assert meta["enabled"] is True
    assert body.lstrip().startswith("# My Skill")


# Documents without a front matter block return the text unchanged.
def test_parse_front_matter_without_block_returns_text():
    text = "# Just markdown\n\nNo front matter here.\n"

    meta, body = skills.parse_front_matter(text)

    assert meta == {}
    assert body == text


# Keys normalize to snake case and scalars map to Python values.
def test_parse_front_matter_normalizes_keys_and_scalars():
    content = "---\nDisplay Name: yes\nanswer: \"42\"\nflag: off\n# comment\nnovalue\n---\nbody"

    meta, body = skills.parse_front_matter(content)

    assert meta == {"display_name": True, "answer": "42", "flag": False}
    assert body == "body"


# Name and path validation.


# Unsafe folder names are rejected before anything touches the disk.
@pytest.mark.parametrize(
    "name",
    ["", "   ", "../escape", "bad/name", "bad\\name", ".hidden", "C:\\evil", "bad$name"],
)
def test_create_skill_folder_rejects_unsafe_names(isolated_skills, name):
    with pytest.raises(ValueError):
        skills.create_skill_folder(name)
    assert not isolated_skills.root.exists() or not any(isolated_skills.root.iterdir())


# Safe names create a folder with a rendered SKILL.md template.
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


# Duplicate folder names fail with a clear error.
def test_create_duplicate_skill_folder_fails(isolated_skills):
    skills.create_skill_folder("Duplicated")

    with pytest.raises(ValueError, match="already exists"):
        skills.create_skill_folder("Duplicated")


# Unsafe file paths inside a skill folder are rejected.
@pytest.mark.parametrize(
    "path",
    ["../escape.txt", "sub/../../escape.txt", "notes.exe", "C:/evil.txt", "file\x00.txt", ".hidden.md"],
)
def test_write_skill_file_rejects_unsafe_paths(isolated_skills, path):
    _create_skill("Guarded")

    with pytest.raises(ValueError):
        skills.write_skill_file("Guarded", path, "content")


# Unsafe subdirectory paths inside a skill folder are rejected.
@pytest.mark.parametrize(
    "path",
    ["../escape", "nested/with.ext", "C:/evil", ".hidden"],
)
def test_create_subdirectory_rejects_unsafe_paths(isolated_skills, path):
    _create_skill("Guarded")

    with pytest.raises(ValueError):
        skills.create_skill_subdirectory("Guarded", path)


# Editing skill files.


# Files round-trip through write and read with normalized paths.
def test_skill_file_write_read_roundtrip(isolated_skills):
    _create_skill("Editable")

    skills.write_skill_file("Editable", "docs/notes.md", "# Notes\n\nRemember this.\n")
    payload = skills.read_skill_file("Editable", "docs/notes.md")

    assert payload["content"] == "# Notes\n\nRemember this.\n"
    assert payload["front_matter"] == {}
    assert payload["file"] == "docs/notes.md"


# The primary file exposes its parsed front matter.
def test_read_skill_file_returns_front_matter(isolated_skills):
    _create_skill("Editable")

    payload = skills.read_skill_file("Editable", "SKILL.md")

    assert payload["front_matter"]["name"] == "My Skill"
    assert payload["front_matter"]["enabled"] is True


# Oversized content is rejected before it reaches the disk.
def test_write_skill_file_rejects_oversized_content(isolated_skills):
    _create_skill("Editable")

    with pytest.raises(ValueError, match="too large"):
        skills.write_skill_file("Editable", "SKILL.md", "x" * (skills.MAX_TEXT_FILE_BYTES + 1))


# Deleting a file prunes the empty parent directories it leaves behind.
def test_delete_skill_file_prunes_empty_parents(isolated_skills):
    _create_skill("Editable")
    skills.write_skill_file("Editable", "deep/nested/notes.md", "content")

    skills.delete_skill_file("Editable", "deep/nested/notes.md")

    assert not (isolated_skills.root / "Editable" / "deep").exists()


# Folder lifecycle.


# Renaming a folder updates its primary front matter name.
def test_rename_skill_folder_updates_front_matter_name(isolated_skills):
    _create_skill("Old Name")

    skills.rename_skill_folder("Old Name", "New Name")

    renamed = isolated_skills.root / "New Name"
    assert renamed.is_dir()
    meta, _body = skills.parse_front_matter((renamed / "SKILL.md").read_text(encoding="utf-8"))
    assert meta["name"] == "New Name"


# Renaming onto an existing folder fails.
def test_rename_skill_folder_rejects_existing_destination(isolated_skills):
    _create_skill("First")
    _create_skill("Second")

    with pytest.raises(ValueError, match="already exists"):
        skills.rename_skill_folder("First", "Second")


# Deleting a folder removes the whole tree.
def test_delete_skill_folder_removes_tree(isolated_skills):
    _create_skill("Doomed")
    skills.create_skill_subdirectory("Doomed", "assets")

    skills.delete_skill_folder("Doomed")

    assert not (isolated_skills.root / "Doomed").exists()


# Renaming items moves files inside the skill folder.
def test_rename_skill_item_moves_files(isolated_skills):
    _create_skill("Mover")
    skills.write_skill_file("Mover", "draft.md", "text")

    skills.rename_skill_item("Mover", "draft.md", "final.md", "file")

    payload = skills.read_skill_file("Mover", "final.md")
    assert payload["content"] == "text"
    with pytest.raises(FileNotFoundError):
        skills.read_skill_file("Mover", "draft.md")


# Enable toggles.


# The enabled flag parser covers the formats arriving from the API.
def test_parse_enabled_flag_covers_input_formats():
    assert skills.parse_enabled_flag(True) is True
    assert skills.parse_enabled_flag(False) is False
    assert skills.parse_enabled_flag("yes") is True
    assert skills.parse_enabled_flag("off") is False
    assert skills.parse_enabled_flag("") is False
    assert skills.parse_enabled_flag(1) is True
    assert skills.parse_enabled_flag("maybe") is True


# Disabling a skill writes the flag and drops it from the sandbox mirror.
def test_set_skill_enabled_writes_front_matter_and_mirrors_sandbox(isolated_skills):
    _create_skill("Toggle")

    skills.set_skill_enabled("Toggle", False)

    meta, _body = skills.parse_front_matter(
        (isolated_skills.root / "Toggle" / "SKILL.md").read_text(encoding="utf-8")
    )
    assert meta["enabled"] is False
    assert not (isolated_skills.sandbox / "Toggle").exists()


# Enabled skills and their nested files are mirrored into the sandbox.
def test_set_skill_enabled_syncs_enabled_skills_into_sandbox(isolated_skills):
    _create_skill("Enabled Skill")
    skills.write_skill_file("Enabled Skill", "refs/guide.md", "guide")

    skills.sync_skills_to_sandbox()

    mirrored = isolated_skills.sandbox / "Enabled Skill"
    assert (mirrored / "SKILL.md").is_file()
    assert (mirrored / "refs" / "guide.md").is_file()


# Sandbox sync.


# Sync removes stale files and re-copies changed ones.
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


# Browser imports.


# Imported trees are validated and the primary file is normalized.
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


# Imports without a primary file get a synthesized SKILL.md.
def test_import_skill_files_synthesizes_primary_when_absent(isolated_skills):
    skills.import_skill_files("No Primary", [{"path": "notes.md", "content": "note"}])

    content = (isolated_skills.root / "No Primary" / "SKILL.md").read_text(encoding="utf-8")
    assert content.startswith("---\nname: No Primary")
    assert "Imported skill." in content


# Imports with no valid files fail without touching the disk.
def test_import_skill_files_rejects_empty_input(isolated_skills):
    with pytest.raises(ValueError, match="No valid files"):
        skills.import_skill_files("Empty", [{"path": "../only.md", "content": "x"}])


# System prompt inventory.


# The inventory lists enabled skills with their sandbox paths and trees.
def test_inventory_lists_only_enabled_skills(isolated_skills):
    _create_skill("Active Skill")
    _create_skill("Sleeping Skill")
    skills.set_skill_enabled("Sleeping Skill", False)

    inventory = skills.build_system_prompt_inventory()

    assert "Active Skill" in inventory
    assert "Sleeping Skill" not in inventory
    assert "/workspace/_sandbox/Skills/Active Skill" in inventory
    assert "SKILL.md" in inventory


# Without skills the inventory is empty.
def test_inventory_empty_when_no_skills(isolated_skills):
    assert skills.build_system_prompt_inventory() == ""


# Nested markdown files appear as indented tree lines.
def test_inventory_includes_tree_files(isolated_skills):
    _create_skill("Tree Skill")
    skills.write_skill_file("Tree Skill", "guides/deep.md", "deep")

    inventory = skills.build_system_prompt_inventory()

    assert "guides/" in inventory
    assert "guides/deep.md" in inventory


# Pending refresh handling.


# Baseline inventory is only included when explicitly requested.
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


# The pending flag survives a reload and can be cleared explicitly.
def test_pending_refresh_survives_reload_and_can_be_cleared(isolated_skills):
    _create_skill("Persisted")
    skills.set_skill_enabled("Persisted", False)
    assert skills.build_system_prompt_skills_section(consume=False)

    skills.clear_skill_config_refresh_pending()

    assert skills.build_system_prompt_skills_section(consume=False) == ""


# A corrupt notification file degrades to "no refresh pending".
def test_notify_state_survives_invalid_file(isolated_skills):
    isolated_skills.base.joinpath(".aslm").mkdir(parents=True, exist_ok=True)
    isolated_skills.base.joinpath(".aslm", "skills-pending-notify.json").write_text("{broken", encoding="utf-8")

    assert skills.build_system_prompt_skills_section(consume=False) == ""


# A legacy notify file in the skills root is migrated into a pending refresh.
def test_legacy_notify_file_migrates_to_pending_refresh(isolated_skills):
    isolated_skills.root.mkdir(parents=True, exist_ok=True)
    isolated_skills.root.joinpath(".skills-pending-notify.json").write_text("{}", encoding="utf-8")

    section = skills.build_system_prompt_skills_section(consume=False)

    # The legacy file is migrated into a pending refresh notification.
    assert section.startswith("Skill configuration update:")
    assert not isolated_skills.root.joinpath(".skills-pending-notify.json").exists()


# Listing.


# Hidden folders and disallowed file types never reach the listing.
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
