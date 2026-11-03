# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import BASE_DIR

# Project Skills folder, its mirror inside the sandbox, and the path prefix under which the model
# sees that mirror from the sandbox workspace.
SKILLS_DIR = BASE_DIR / "Skills"
SANDBOX_SKILLS_DIR = BASE_DIR / "Tools" / "mcp-sandbox" / "_sandbox" / "Skills"
SANDBOX_MODEL_SKILLS_PREFIX = "/workspace/_sandbox/Skills"

# File names tried, in order, as the primary markdown file of a skill folder.
PRIMARY_SKILL_FILENAMES = ("SKILL.md", "skills.md")

# Text file types the skills manager may list, read and write; anything else is ignored or rejected.
ALLOWED_TEXT_SUFFIXES = {
    ".bat",
    ".css",
    ".html",
    ".js",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
    ".ts",
    ".txt",
    ".yaml",
    ".yml",
}

# Largest text file, in bytes, that the skills manager reads or writes.
MAX_TEXT_FILE_BYTES = 2 * 1024 * 1024

# Allowed skill folder name, and the leading "---" front matter block of a skill markdown file.
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._ -]{0,79}$")
_FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.DOTALL)

# Marker of the pending skills config refresh: the old file inside the Skills folder (migrated away
# and deleted) and the current file under the project's .aslm folder.
_LEGACY_PENDING_NOTIFY_FILE = ".skills-pending-notify.json"
_PENDING_NOTIFY_PATH = BASE_DIR / ".aslm" / "skills-pending-notify.json"

# Re-entrant locks that serialize sandbox sync and changes to the pending refresh marker.
_sync_lock = threading.RLock()
_notify_lock = threading.RLock()
logger = logging.getLogger(__name__)


# Metadata for one file node in a skill folder tree.
@dataclass(frozen=True)
class SkillFile:
    path: str
    name: str
    type: str
    size_bytes: int
    updated_at: float


# Paths and validation

# Ensure the project-level Skills root directory exists.
def ensure_skills_dir() -> Path:
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    return SKILLS_DIR


# Return True when a path segment is hidden or reserved.
def _is_hidden_part(part: str) -> bool:
    return part in {"", ".", ".."} or part.startswith(".")


# Validate and return a safe skill folder name.
def _validate_skill_name(name: str) -> str:
    cleaned = str(name or "").strip()
    if not cleaned:
        raise ValueError("Skill folder name is required.")
    if not _NAME_RE.fullmatch(cleaned):
        raise ValueError("Skill folder name may contain letters, numbers, spaces, dots, underscores, and hyphens.")
    if any(sep in cleaned for sep in ("/", "\\")) or _is_hidden_part(cleaned):
        raise ValueError("Invalid skill folder name.")
    return cleaned


# Normalize and validate a relative file path inside a skill folder.
def _normalize_relative_file_path(path: str) -> str:
    raw = str(path or "").replace("\\", "/").strip().strip("/")
    if not raw:
        raise ValueError("File path is required.")
    if "\x00" in raw or re.match(r"^[A-Za-z]:/", raw) or raw.startswith("/"):
        raise ValueError("File path must be relative.")
    parts = [part for part in raw.split("/") if part]
    if not parts or any(_is_hidden_part(part) for part in parts):
        raise ValueError("Invalid file path.")
    suffix = Path(parts[-1]).suffix.lower()
    if suffix not in ALLOWED_TEXT_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix or '(none)'}")
    return "/".join(parts)


# Normalize and validate a relative directory path inside a skill folder.
def _normalize_relative_dir_path(path: str) -> str:
    raw = str(path or "").replace("\\", "/").strip().strip("/")
    if not raw:
        raise ValueError("Directory path is required.")
    if "\x00" in raw or re.match(r"^[A-Za-z]:/", raw) or raw.startswith("/"):
        raise ValueError("Directory path must be relative.")
    parts = [part for part in raw.split("/") if part]
    if not parts or any(_is_hidden_part(part) for part in parts):
        raise ValueError("Invalid directory path.")
    last = parts[-1]
    if Path(last).suffix:
        raise ValueError("Directory name must not have a file extension.")
    return "/".join(parts)


# Resolve the on-disk directory for one skill name.
def _skill_dir(name: str) -> Path:
    ensure_skills_dir()
    skill_name = _validate_skill_name(name)
    target = (SKILLS_DIR / skill_name).resolve()
    if not target.is_relative_to(SKILLS_DIR.resolve()):
        raise ValueError("Skill path escapes Skills root.")
    return target


# Resolve an absolute path for one file inside a skill folder.
def _skill_file_path(folder: str, file_path: str) -> Path:
    skill_root = _skill_dir(folder)
    rel_path = _normalize_relative_file_path(file_path)
    target = (skill_root / Path(rel_path)).resolve()
    if not target.is_relative_to(skill_root.resolve()):
        raise ValueError("File path escapes skill folder.")
    return target


# Resolve an absolute path for one subdirectory inside a skill folder.
def _skill_subdir_path(folder: str, dir_path: str) -> Path:
    skill_root = _skill_dir(folder)
    rel_path = _normalize_relative_dir_path(dir_path)
    target = (skill_root / Path(rel_path)).resolve()
    if not target.is_relative_to(skill_root.resolve()):
        raise ValueError("Directory path escapes skill folder.")
    return target


# Safe file access

# Read a text file with size and binary guards for the skills manager.
def _safe_read_text(path: Path) -> str:
    size_bytes = path.stat().st_size
    if size_bytes > MAX_TEXT_FILE_BYTES:
        raise ValueError("File is too large to edit in the skills manager.")
    data = path.read_bytes()
    if b"\x00" in data[:8192]:
        raise ValueError("Binary files are not supported by the skills manager.")
    return data.decode("utf-8")


# Write text atomically via a temporary file in the target directory.
def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as handle:
            handle.write(str(content or ""))
        Path(tmp_path).replace(path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)


# Front matter

# Parse one YAML-like front matter scalar value.
def _parse_scalar(value: str) -> Any:
    cleaned = str(value or "").strip().strip('"').strip("'")
    lowered = cleaned.lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    return cleaned


# Parse a small YAML-like front matter block from skill markdown.
def parse_front_matter(content: str) -> tuple[dict[str, Any], str]:
    text = str(content or "")
    match = _FRONT_MATTER_RE.match(text)
    if match is None:
        return {}, text

    meta: dict[str, Any] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        clean_key = re.sub(r"[^a-z0-9_]+", "_", key.strip().lower()).strip("_")
        if clean_key:
            meta[clean_key] = _parse_scalar(value)

    return meta, text[match.end() :]


# Normalize the skill source field from front matter metadata.
def _normalize_skill_source(meta: dict[str, Any]) -> str:
    raw = str(meta.get("source") or meta.get("added_by") or "").strip().lower()
    if raw in {"download", "downloaded", "remote", "import", "imported"}:
        return "download"
    if raw in {"local", "personal", "manual", "user"}:
        return "local"
    if raw and meta.get("added_by"):
        return "download"
    return "local"


# Strip legacy front matter keys before persisting metadata.
def _normalize_meta_for_storage(meta: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(meta)
    cleaned["source"] = _normalize_skill_source(cleaned)
    cleaned.pop("added_by", None)
    cleaned.pop("trigger", None)
    return cleaned


# Return the best available creation timestamp for a skill folder.
def _skill_created_at(stat: os.stat_result) -> float:
    birthtime = getattr(stat, "st_birthtime", None)
    if birthtime is not None:
        return float(birthtime)
    return float(stat.st_ctime)


# Render front matter metadata as a markdown header block.
def _format_front_matter(meta: dict[str, Any]) -> str:
    lines = ["---"]
    for key in ("name", "description", "source", "enabled"):
        if key not in meta:
            continue
        value = meta[key]
        if isinstance(value, bool):
            rendered = "true" if value else "false"
        else:
            rendered = str(value or "").replace("\n", " ").strip()
        lines.append(f"{key}: {rendered}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


# Skill metadata and file tree

# Locate the primary markdown file for one skill folder.
def _primary_file_for_skill(skill_root: Path) -> Path:
    for filename in PRIMARY_SKILL_FILENAMES:
        candidate = skill_root / filename
        if candidate.is_file():
            return candidate
    for candidate in sorted(skill_root.glob("*.md"), key=lambda item: item.name.casefold()):
        if candidate.is_file():
            return candidate
    return skill_root / "SKILL.md"


# Read normalized metadata and the primary file path for one skill.
def _read_skill_meta(skill_root: Path) -> tuple[dict[str, Any], str]:
    primary = _primary_file_for_skill(skill_root)
    meta: dict[str, Any] = {}
    if primary.is_file():
        try:
            meta, _body = parse_front_matter(_safe_read_text(primary))
        except (OSError, UnicodeDecodeError, ValueError):
            meta = {}

    name = str(meta.get("name") or skill_root.name).strip() or skill_root.name
    description = str(meta.get("description") or "").strip()
    source = _normalize_skill_source(meta)
    enabled = meta.get("enabled")
    normalized = {
        "name": name,
        "description": description,
        "source": source,
        "enabled": bool(enabled) if isinstance(enabled, bool) else True,
    }
    primary_rel = primary.relative_to(skill_root).as_posix()
    return normalized, primary_rel


# Build one SkillFile node for the skills tree API.
def _file_node(path: Path, root: Path) -> SkillFile:
    stat = path.stat()
    return SkillFile(
        path=path.relative_to(root).as_posix(),
        name=path.name,
        type="file",
        size_bytes=stat.st_size,
        updated_at=stat.st_mtime,
    )


# Build the nested file tree for one skill folder.
def _build_tree(path: Path, root: Path) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    try:
        entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.casefold()))
    except OSError:
        return nodes

    for entry in entries:
        if _is_hidden_part(entry.name) or entry.is_symlink():
            continue
        if entry.is_dir():
            children = _build_tree(entry, root)
            nodes.append({
                "type": "directory",
                "name": entry.name,
                "path": entry.relative_to(root).as_posix(),
                "children": children,
            })
            continue
        if entry.is_file() and entry.suffix.lower() in ALLOWED_TEXT_SUFFIXES:
            nodes.append(_file_node(entry, root).__dict__)
    return nodes


# Skill folder and file operations

# List all skill folders with metadata and file trees.
def list_skills() -> dict[str, Any]:
    root = ensure_skills_dir()
    folders: list[dict[str, Any]] = []
    for skill_root in sorted(root.iterdir(), key=lambda item: item.name.casefold()):
        # Skip files, symlinks and hidden folders; only real skill folders are listed.
        if not skill_root.is_dir() or skill_root.is_symlink() or _is_hidden_part(skill_root.name):
            continue

        # Describe the folder from its front matter and add timestamps and the file tree.
        meta, primary = _read_skill_meta(skill_root)
        stat = skill_root.stat()
        folders.append({
            "name": skill_root.name,
            "title": meta["name"],
            "description": meta["description"],
            "source": meta["source"],
            "enabled": meta["enabled"],
            "primary_file": primary,
            "sandbox_path": f"{SANDBOX_MODEL_SKILLS_PREFIX}/{skill_root.name}",
            "created_at": _skill_created_at(stat),
            "updated_at": stat.st_mtime,
            "tree": _build_tree(skill_root, skill_root),
        })
    return {
        "folders": folders,
        "root": str(root),
        "sandbox_path_prefix": SANDBOX_MODEL_SKILLS_PREFIX,
        "allowed_extensions": sorted(ALLOWED_TEXT_SUFFIXES),
    }


# Create a new skill folder with a default SKILL.md template.
def create_skill_folder(name: str) -> dict[str, Any]:
    skill_name = _validate_skill_name(name)
    target = _skill_dir(skill_name)
    if target.exists():
        raise ValueError("Skill folder already exists.")
    target.mkdir(parents=True)
    title = skill_name.replace("_", " ").replace("-", " ").strip().title() or skill_name
    _atomic_write_text(
        target / "SKILL.md",
        _format_front_matter({
            "name": skill_name,
            "description": "",
            "source": "local",
            "enabled": True,
        })
        + f"# {title}\n\nDescribe when and how this skill should be used.\n",
    )
    return list_skills()


# Rename a skill folder and update its primary front matter name.
def rename_skill_folder(old_name: str, new_name: str) -> dict[str, Any]:
    source = _skill_dir(old_name)
    destination = _skill_dir(new_name)
    if not source.is_dir():
        raise FileNotFoundError("Skill folder not found.")
    if destination.exists():
        raise ValueError("Destination skill folder already exists.")
    source.rename(destination)
    primary = _primary_file_for_skill(destination)
    if primary.is_file():
        content = _safe_read_text(primary)
        meta, body = parse_front_matter(content)
        meta["name"] = _validate_skill_name(new_name)
        _atomic_write_text(primary, _format_front_matter(_normalize_meta_for_storage(meta)) + body.lstrip("\r\n"))
    return list_skills()


# Delete one skill folder and its contents.
def delete_skill_folder(name: str) -> dict[str, Any]:
    target = _skill_dir(name)
    if not target.is_dir():
        raise FileNotFoundError("Skill folder not found.")
    shutil.rmtree(target)
    return list_skills()


# Create a subdirectory inside one skill folder.
def create_skill_subdirectory(folder: str, dir_path: str) -> dict[str, Any]:
    skill_root = _skill_dir(folder)
    if not skill_root.is_dir():
        raise FileNotFoundError("Skill folder not found.")
    target = _skill_subdir_path(folder, dir_path)
    if target.exists():
        raise ValueError("Directory already exists.")
    target.mkdir(parents=True)
    return list_skills()


# Delete a subdirectory inside one skill folder.
def delete_skill_subdirectory(folder: str, dir_path: str) -> dict[str, Any]:
    target = _skill_subdir_path(folder, dir_path)
    if not target.is_dir():
        raise FileNotFoundError("Directory not found.")
    shutil.rmtree(target)
    return list_skills()


# Rename a file or directory inside one skill folder.
def rename_skill_item(folder: str, old_path: str, new_path: str, kind: str) -> dict[str, Any]:
    skill_root = _skill_dir(folder)
    if not skill_root.is_dir():
        raise FileNotFoundError("Skill folder not found.")
    kind_value = str(kind or "").strip().lower()
    is_dir = kind_value in {"directory", "dir", "folder"}
    old_rel = _normalize_relative_dir_path(old_path) if is_dir else _normalize_relative_file_path(old_path)
    new_rel = _normalize_relative_dir_path(new_path) if is_dir else _normalize_relative_file_path(new_path)
    if old_rel == new_rel:
        return list_skills()
    source = (skill_root / Path(old_rel)).resolve()
    destination = (skill_root / Path(new_rel)).resolve()
    if not source.is_relative_to(skill_root.resolve()) or not destination.is_relative_to(skill_root.resolve()):
        raise ValueError("Path escapes skill folder.")
    if not source.exists():
        raise FileNotFoundError("Item not found.")
    if destination.exists():
        raise ValueError("Destination already exists.")
    source.rename(destination)
    return list_skills()


# Read one skill file with parsed front matter metadata.
def read_skill_file(folder: str, file_path: str) -> dict[str, Any]:
    target = _skill_file_path(folder, file_path)
    if not target.is_file():
        raise FileNotFoundError("Skill file not found.")
    content = _safe_read_text(target)
    meta, _body = parse_front_matter(content)
    return {
        "folder": _validate_skill_name(folder),
        "file": _normalize_relative_file_path(file_path),
        "content": content,
        "front_matter": meta,
        "size_bytes": target.stat().st_size,
        "updated_at": target.stat().st_mtime,
    }


# Write one skill file and return the updated skills listing.
def write_skill_file(folder: str, file_path: str, content: str) -> dict[str, Any]:
    target = _skill_file_path(folder, file_path)
    skill_root = _skill_dir(folder)
    if not skill_root.is_dir():
        raise FileNotFoundError("Skill folder not found.")
    encoded_size = len(str(content or "").encode("utf-8"))
    if encoded_size > MAX_TEXT_FILE_BYTES:
        raise ValueError("File is too large to edit in the skills manager.")
    _atomic_write_text(target, str(content or ""))
    return list_skills()


# Delete one skill file and prune empty parent directories.
def delete_skill_file(folder: str, file_path: str) -> dict[str, Any]:
    target = _skill_file_path(folder, file_path)
    if not target.is_file():
        raise FileNotFoundError("Skill file not found.")
    target.unlink()
    current = target.parent
    skill_root = _skill_dir(folder)
    while current != skill_root and current.is_dir():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent
    return list_skills()


# Parse a loose enabled flag from API or front matter input.
def parse_enabled_flag(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    cleaned = str(value or "").strip().lower()
    if cleaned in {"true", "1", "yes", "on"}:
        return True
    if cleaned in {"false", "0", "no", "off", ""}:
        return False
    return bool(cleaned)


# Toggle whether one skill is enabled and refresh sandbox sync when needed.
def set_skill_enabled(folder: str, enabled: bool) -> dict[str, Any]:
    skill_root = _skill_dir(folder)
    if not skill_root.is_dir():
        raise FileNotFoundError("Skill folder not found.")
    primary = _primary_file_for_skill(skill_root)
    if primary.exists():
        content = _safe_read_text(primary)
        meta, body = parse_front_matter(content)
        was_enabled = _read_skill_meta(skill_root)[0]["enabled"]
    else:
        meta, body = {}, f"# {folder}\n\nDescribe when and how this skill should be used.\n"
        was_enabled = True
    meta.setdefault("name", folder)
    meta.setdefault("description", "")
    meta.setdefault("source", "local")
    new_enabled = parse_enabled_flag(enabled)
    meta["enabled"] = new_enabled
    _atomic_write_text(primary, _format_front_matter(_normalize_meta_for_storage(meta)) + body.lstrip("\r\n"))
    if was_enabled != new_enabled:
        try:
            sync_skills_to_sandbox()
        except Exception:
            logger.exception("Skills sandbox sync failed after toggling %s", folder)
        _queue_skill_config_refresh()
    return list_skills()


# Create a skill folder from browser-imported path/content pairs.
def import_skill_files(skill_name: str, files: list[dict[str, Any]]) -> dict[str, Any]:
    validated_name = _validate_skill_name(skill_name)
    target_root = _skill_dir(validated_name)
    target_resolved = target_root.resolve()

    # Collect and validate all files first so we fail before writing anything.
    validated: list[tuple[str, str]] = []
    for entry in (files or []):
        if not isinstance(entry, dict):
            continue
        raw_path = str(entry.get("path") or "").replace("\\", "/").strip().strip("/")
        content = str(entry.get("content") or "")
        if not raw_path or "\x00" in raw_path:
            continue

        # Normalise and security-check the relative path.
        parts = [p for p in raw_path.split("/") if p]
        if not parts or any(_is_hidden_part(p) for p in parts):
            continue
        suffix = Path(parts[-1]).suffix.lower()
        if suffix not in ALLOWED_TEXT_SUFFIXES:
            continue
        rel = "/".join(parts)
        dest = (target_root / Path(rel)).resolve()
        try:
            if not dest.is_relative_to(target_resolved):
                continue
        except ValueError:
            # A path that cannot be compared with the skill root is outside it; skip the file.
            continue
        validated.append((rel, content))

    if not validated:
        raise ValueError("No valid files to import.")

    target_root.mkdir(parents=True, exist_ok=True)

    # Patch the primary SKILL.md front-matter if present, otherwise inject one.
    primary_rel = next(
        (r for r, _ in validated if Path(r).name in PRIMARY_SKILL_FILENAMES),
        None,
    )

    for rel, content in validated:
        dest = target_root / Path(rel)
        if rel == primary_rel:
            meta, body = parse_front_matter(content)
            meta.setdefault("name", validated_name)
            meta.setdefault("description", "")
            meta["source"] = "local"
            meta.setdefault("enabled", True)
            content = _format_front_matter(_normalize_meta_for_storage(meta)) + body.lstrip("\r\n")
        _atomic_write_text(dest, content)

    if primary_rel is None:
        # No primary file was included — synthesise a minimal SKILL.md.
        title = validated_name.replace("_", " ").replace("-", " ").strip().title() or validated_name
        _atomic_write_text(
            target_root / "SKILL.md",
            _format_front_matter({
                "name": validated_name,
                "description": "",
                "source": "local",
                "enabled": True,
            }) + f"# {title}\n\nImported skill.\n",
        )

    return list_skills()


# System prompt inventory

# Render enabled skill tree nodes as indented lines for the system prompt.
def _format_skill_tree_lines(nodes: list[dict[str, Any]], *, indent: int = 2) -> list[str]:
    lines: list[str] = []
    pad = " " * indent
    for node in nodes:
        if not isinstance(node, dict):
            continue
        node_type = str(node.get("type") or "").strip().lower()
        if node_type == "directory":
            dir_name = str(node.get("name") or node.get("path") or "").strip()
            if not dir_name:
                continue
            lines.append(f"{pad}{dir_name}/")
            lines.extend(
                _format_skill_tree_lines(node.get("children") or [], indent=indent + 2)
            )
            continue
        if node_type != "file":
            continue
        rel_path = str(node.get("path") or node.get("name") or "").strip()
        if rel_path.lower().endswith(".md"):
            lines.append(f"{pad}{rel_path}")
    return lines


# Pending refresh notification

# Return the path used to queue a one-shot skills config refresh.
def _pending_notify_path() -> Path:
    path = _PENDING_NOTIFY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Load the pending skills notification state from disk.
def _load_notify_state() -> dict[str, Any]:
    path = _pending_notify_path()
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError):
        logger.warning("Ignoring invalid skills notification file at %s", path, exc_info=True)
        return {}
    return payload if isinstance(payload, dict) else {}


# Persist or clear the pending skills notification state.
def _save_notify_state(state: dict[str, Any]) -> None:
    path = _pending_notify_path()
    if not state.get("config_refresh_pending"):
        if path.exists():
            path.unlink(missing_ok=True)
        return
    _atomic_write_text(
        path,
        json.dumps({"config_refresh_pending": True}, ensure_ascii=False, indent=2) + "\n",
    )


# Clear any queued one-shot skills summary refresh (tests and tooling).
def clear_skill_config_refresh_pending() -> None:
    with _notify_lock:
        _save_notify_state({})


# Migrate legacy notification files and re-queue refresh when needed.
def _migrate_legacy_notify_state() -> None:
    legacy_path = ensure_skills_dir() / _LEGACY_PENDING_NOTIFY_FILE
    if legacy_path.is_file():
        legacy_path.unlink(missing_ok=True)
        _queue_skill_config_refresh()
        return

    payload = _load_notify_state()
    if payload.get("disabled"):
        _queue_skill_config_refresh()


# Return whether a skills config refresh is pending without consuming it.
def _peek_config_refresh_pending() -> bool:
    _migrate_legacy_notify_state()
    return bool(_load_notify_state().get("config_refresh_pending"))


# Return and clear the pending skills config refresh flag.
def _consume_config_refresh_pending() -> bool:
    with _notify_lock:
        _migrate_legacy_notify_state()
        pending = bool(_load_notify_state().get("config_refresh_pending"))
        if pending:
            _save_notify_state({})
        return pending


# Queue a one-shot skills summary refresh for the next chat turn.
def _queue_skill_config_refresh() -> None:
    with _notify_lock:
        _save_notify_state({"config_refresh_pending": True})


# System prompt sections

# Build a system-prompt block listing enabled skills and their on-disk layout.
def build_system_prompt_inventory() -> str:
    # Only enabled skills are listed; no enabled skills means no block at all.
    payload = list_skills()
    folders = [folder for folder in payload["folders"] if folder.get("enabled", True)]
    if not folders:
        return ""

    # Header lines tell the model where to read the skill files in the sandbox and on the host.
    sandbox_prefix = str(payload.get("sandbox_path_prefix") or SANDBOX_MODEL_SKILLS_PREFIX).rstrip("/")
    project_root = str(payload.get("root") or SKILLS_DIR)

    lines = [
        "Your skills:",
        "",
        "Read these skill files with sandbox tools when they are relevant to the user's request.",
        f"Sandbox path: {sandbox_prefix}",
        f"Project folder: {project_root}",
        "",
    ]

    # One entry per skill: folder name, optional title and description, sandbox path, then its markdown files.
    for folder in folders:
        name = str(folder.get("name") or "").strip()
        if not name:
            continue
        title = str(folder.get("title") or name).strip()
        description = str(folder.get("description") or "").strip()
        skill_sandbox = f"{sandbox_prefix}/{name}"
        header = f"- {name}/"
        details: list[str] = []
        if title and title.casefold() != name.casefold():
            details.append(title)
        if description:
            details.append(description)
        if details:
            header += f" ({'; '.join(details)})"
        header += f" → {skill_sandbox}"
        lines.append(header)

        # Fall back to the primary file name when the tree has no markdown file to list.
        tree_lines = _format_skill_tree_lines(folder.get("tree") or [])
        if tree_lines:
            lines.extend(tree_lines)
        else:
            primary = str(folder.get("primary_file") or "SKILL.md").strip()
            lines.append(f"  {primary}")

    return "\n".join(lines)


# Inject skills into the system prompt on the first turn or after enable toggles.
def build_system_prompt_skills_section(
    *,
    consume: bool = True,
    include_baseline: bool = False,
) -> str:
    pending = _consume_config_refresh_pending() if consume else _peek_config_refresh_pending()
    inventory = build_system_prompt_inventory()
    if pending:
        if inventory:
            return (
                "Skill configuration update:\n\n"
                "The user changed which project skills are enabled. "
                "Use only the skills listed below.\n\n"
                + inventory
            )
        return (
            "Skill configuration update:\n\n"
            "The user changed which project skills are enabled. "
            "There are currently no enabled project skills."
        )
    if include_baseline and inventory:
        return inventory
    return ""


# Backward-compatible alias for build_system_prompt_skills_section.
def build_system_prompt_skill_delta(*, consume: bool = True, include_baseline: bool = False) -> str:
    return build_system_prompt_skills_section(consume=consume, include_baseline=include_baseline)


# Sandbox sync

# Compute the SHA-256 digest of one file.
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# List skill folder names that are currently enabled.
def _enabled_skill_names(root: Path) -> set[str]:
    names: set[str] = set()
    if not root.is_dir():
        return names
    for skill_root in root.iterdir():
        if not skill_root.is_dir() or skill_root.is_symlink() or _is_hidden_part(skill_root.name):
            continue
        meta, _primary = _read_skill_meta(skill_root)
        if meta.get("enabled", True):
            names.add(skill_root.name)
    return names


# Enumerate files under a skills root with size and hash metadata for sync.
def _iter_sync_files(root: Path, skill_names: set[str] | None = None) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    if not root.exists():
        return files
    for path in sorted(root.rglob("*"), key=lambda item: str(item).casefold()):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            rel_parts = path.relative_to(root).parts
        except ValueError:
            # A path outside the root cannot be part of the sync set.
            continue
        if any(_is_hidden_part(part) for part in rel_parts):
            continue
        if skill_names is not None:
            top_level = rel_parts[0] if rel_parts else ""
            if top_level not in skill_names:
                continue
        rel_path = Path(*rel_parts).as_posix()
        stat = path.stat()
        files[rel_path] = {
            "path": path,
            "size": stat.st_size,
            "sha256": _sha256_file(path),
        }
    return files


# Remove one file or directory during sandbox sync cleanup.
def _remove_tree_entry(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink(missing_ok=True)
    elif path.is_dir():
        shutil.rmtree(path)


# Mirror project Skills into the sandbox Skills directory.
def sync_skills_to_sandbox() -> dict[str, Any]:
    with _sync_lock:
        # Resolve both roots and make sure the sandbox target stays inside the sandbox folder.
        source_root = ensure_skills_dir().resolve()
        target_root = SANDBOX_SKILLS_DIR.resolve()
        if target_root.exists() and target_root.is_symlink():
            target_root.unlink()
        target_root.mkdir(parents=True, exist_ok=True)
        if not target_root.is_relative_to((BASE_DIR / "Tools" / "mcp-sandbox" / "_sandbox").resolve()):
            raise RuntimeError("Sandbox Skills target escapes sandbox root.")

        # Only enabled skills are mirrored; compare the source and target file sets by size and hash.
        enabled_names = _enabled_skill_names(source_root)
        source_files = _iter_sync_files(source_root, enabled_names)
        target_files = _iter_sync_files(target_root)

        removed = 0
        copied = 0

        # Drop target files that are no longer present in the enabled source set.
        for rel_path, target_info in target_files.items():
            if rel_path in source_files:
                continue
            target_path = Path(target_info["path"])
            target_path.unlink(missing_ok=True)
            removed += 1

        # Remove stale directories, manifests, and paths outside the source tree.
        source_prefixes = set()
        for rel_path in source_files:
            parts = Path(rel_path).parts
            for index in range(1, len(parts)):
                source_prefixes.add(Path(*parts[:index]).as_posix())

        for target_entry in sorted(target_root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
            if target_entry == target_root:
                continue
            try:
                rel_path = target_entry.relative_to(target_root).as_posix()
            except ValueError:
                # An entry that is not under the target root is not ours to clean up.
                continue
            if rel_path.startswith(".skills_sync_manifest"):
                _remove_tree_entry(target_entry)
                removed += 1
                continue
            if rel_path in source_files or rel_path in source_prefixes:
                continue
            _remove_tree_entry(target_entry)
            removed += 1

        # Copy or update files that changed size or content hash.
        for rel_path, source_info in source_files.items():
            source_path = Path(source_info["path"])
            target_path = (target_root / rel_path).resolve()
            if not target_path.is_relative_to(target_root):
                continue
            target_info = target_files.get(rel_path)
            if target_info and target_info.get("size") == source_info["size"] and target_info.get("sha256") == source_info["sha256"]:
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            copied += 1

        # Prune empty directories left after removals.
        for directory in sorted(
            [path for path in target_root.rglob("*") if path.is_dir()],
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            if directory == target_root:
                continue
            try:
                directory.rmdir()
            except OSError:
                # rmdir only succeeds on empty directories; a non-empty one is still in use and stays.
                pass

        # Summarize what the sync did.
        manifest = {
            "source": str(source_root),
            "target": str(target_root),
            "files": sorted(source_files),
            "copied": copied,
            "removed": removed,
        }
        return manifest
