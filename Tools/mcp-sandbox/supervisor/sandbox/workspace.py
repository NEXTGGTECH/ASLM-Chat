# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import base64
import difflib
import fnmatch
import mimetypes
import os
import posixpath
import re
import shutil
import struct
from pathlib import Path
from typing import Any

from sandbox.config import (
    ALLOWED_IMPORT_ROOTS,
    COMMAND_USER,
    CONTAINER_WORKSPACE,
    DEFAULT_TASK_DIR,
    HOST_WORKSPACE,
    IN_CONTAINER,
    IGNORED_DIR_NAMES,
    MAX_FIND_RESULTS,
    MAX_GREP_RESULTS,
    MAX_IMAGE_PREVIEW_BYTES,
    MAX_LS_ENTRIES,
    MAX_READ_BYTES,
)
from sandbox.responses import SandboxToolError

# Context lines shown around a match in previews, the number of leading bytes sampled to tell text from
# binary, the MIME prefixes of images and text, and the largest file the supervisor loads into memory.
CONTEXT_LINES = 3
TEXT_SAMPLE_BYTES = 8192
IMAGE_MIME_PREFIX = "image/"
TEXT_MIME_PREFIX = "text/"
MAX_FILE_READ_BYTES = max(MAX_READ_BYTES, MAX_IMAGE_PREVIEW_BYTES, TEXT_SAMPLE_BYTES)

# Path prefixes of an older upload convention; such paths are mapped to "User/" in the task workspace.
LEGACY_UPLOAD_ROOT_PREFIXES = ("mnt/data/User",)


# Return accepted model-facing workspace root aliases: the default task directory name, cleaned of
# slashes and duplicates (empty names are dropped).
def model_root_aliases() -> tuple[str, ...]:
    aliases = [DEFAULT_TASK_DIR]
    seen: set[str] = set()
    result: list[str] = []
    for alias in aliases:
        cleaned = alias.strip().strip("/")
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        result.append(cleaned)
    return tuple(result)


# Encoding and newlines

# Decode bytes with a small fallback chain: UTF-16 (with or without BOM), then utf-8, cp866, cp1251 and
# latin1. Return the text and the encoding that worked; undecodable data falls back to utf-8 with
# replacement characters.
def smart_decode(bytes_data: bytes) -> tuple[str, str | None]:
    if not bytes_data:
        return "", "utf-8"

    # UTF-16 BOM detection — must happen before latin1, which never raises and
    # would silently misread UTF-16 as garbage.
    if bytes_data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return bytes_data.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            # Not valid UTF-16 after all: continue with the other detection steps.
            pass

    # UTF-16 without BOM: count null bytes at even vs odd byte positions in a
    # sample window. ASCII-range UTF-16-LE has nulls at odd positions; UTF-16-BE
    # at even. latin1 text almost never has this density of nulls.
    sample = bytes_data[:512]
    if len(sample) >= 4:
        odd_nulls = sum(1 for i in range(1, len(sample), 2) if sample[i] == 0)
        even_nulls = sum(1 for i in range(0, len(sample), 2) if sample[i] == 0)
        threshold = len(sample) // 4
        if odd_nulls > threshold:
            try:
                return bytes_data.decode("utf-16-le"), "utf-16-le"
            except UnicodeDecodeError:
                # The null pattern was a coincidence; try the next candidate.
                pass
        if even_nulls > threshold:
            try:
                return bytes_data.decode("utf-16-be"), "utf-16-be"
            except UnicodeDecodeError:
                # The null pattern was a coincidence; fall through to the single-byte encodings.
                pass

    # Single-byte and UTF-8 candidates, strictest first (latin1 accepts anything).
    for encoding in ("utf-8", "cp866", "cp1251", "latin1"):
        try:
            return bytes_data.decode(encoding), encoding
        except UnicodeDecodeError:
            # Wrong encoding: try the next one.
            continue

    return bytes_data.decode("utf-8", errors="replace"), "utf-8"


# Detect the dominant newline style in text: CRLF, lone CR or LF (LF when there is a tie or no newline).
def detect_newline_style(text: str) -> str:
    crlf = text.count("\r\n")
    normalized = text.replace("\r\n", "")
    cr = normalized.count("\r")
    lf = normalized.count("\n")

    if crlf >= lf and crlf >= cr and crlf > 0:
        return "\r\n"
    if cr > lf and cr > 0:
        return "\r"
    return "\n"


# Normalize all newline variants to LF.
def normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


# Workspace roots

# Return the resolved host workspace root.
def workspace_root() -> Path:
    return Path(HOST_WORKSPACE).resolve()


# Return the sandbox workspace root exposed to the model.
def task_root() -> Path:
    return (workspace_root() / DEFAULT_TASK_DIR).resolve()


# Path normalization

# Normalize a workspace-relative path to POSIX form (forward slashes, no leading slash).
def normalize_relative_path(path: str) -> str:
    normalized = str(path or ".").replace("\\", "/").strip()
    if not normalized:
        normalized = "."
    if normalized == ".":
        return "."
    return normalized.lstrip("/")


# Normalize model-facing paths and tolerate workspace-root aliases: legacy upload paths are mapped,
# and a leading task directory (plain or under the container workspace) is stripped, so the result
# is relative to the task root.
def normalize_model_relative_path(path: str) -> str:
    normalized = normalize_relative_path(path)
    legacy_upload = _legacy_upload_relative_path(normalized)
    if legacy_upload is not None:
        return legacy_upload

    # Strip the task directory prefix in its plain and container-absolute forms.
    for alias in model_root_aliases():
        if normalized == alias:
            return "."

        prefix = f"{alias}/"
        if normalized.startswith(prefix):
            return normalized[len(prefix) :] or "."

        container_task_prefix = f"{CONTAINER_WORKSPACE.strip('/')}/{alias}"
        if normalized == container_task_prefix:
            return "."

        container_task_prefix_with_slash = f"{container_task_prefix}/"
        if normalized.startswith(container_task_prefix_with_slash):
            return normalized[len(container_task_prefix_with_slash) :] or "."

    return normalized


# Map old upload paths from the prompt contract into the task workspace.
def _legacy_upload_relative_path(normalized_path: str) -> str | None:
    normalized = normalize_relative_path(normalized_path)
    for prefix in LEGACY_UPLOAD_ROOT_PREFIXES:
        if normalized == prefix:
            return "User"
        prefix_with_slash = f"{prefix}/"
        if normalized.startswith(prefix_with_slash):
            return f"User/{normalized[len(prefix_with_slash):]}" or "User"
    return None


# Path validation and resolution

# Reject unsafe paths (null bytes, Windows drive paths); allow absolute Linux paths inside the
# container, and outside it only the workspace-root aliases. Raise ValueError for rejected paths.
def validate_model_path(rel_path: str, kind: str = "path") -> None:
    raw = str(rel_path or ".").replace("\\", "/").strip()
    if not raw or raw == ".":
        return

    if "\x00" in raw:
        raise ValueError(f"{kind} must not contain null bytes.")

    # os.path.splitdrive() only detects drive letters on Windows; inside the
    # Linux container it always returns ("", ...).  Use a portable regex instead.
    if re.match(r'^[a-zA-Z]:[/\\]', raw):
        raise ValueError(
            f"{kind} must not be a Windows path. Got: '{raw}'."
        )

    if raw.startswith("/"):
        legacy_upload = _legacy_upload_relative_path(raw)
        if legacy_upload is not None:
            return

        # Always allow workspace-root aliases (e.g. /workspace/_sandbox/...).
        for alias in model_root_aliases():
            container_task_root = f"{CONTAINER_WORKSPACE.rstrip('/')}/{alias}"
            if raw == container_task_root or raw.startswith(f"{container_task_root}/"):
                return

        # Inside the container any absolute Linux path is legal — the container
        # filesystem is the security boundary, not path restrictions.
        if IN_CONTAINER:
            return
        raise ValueError(
            f"{kind} must be relative to the model workspace root. "
            f"Use '.' or a relative path like 'script.py'. "
            f"Optional prefixes like '{DEFAULT_TASK_DIR}/' are tolerated, never '{raw}'."
        )


# Resolve a model-facing path relative to the provided cwd. Absolute workspace paths and paths that
# already carry a workspace alias are kept; other absolute paths are accepted only inside the
# container; relative paths are joined with the cwd.
def resolve_model_path(path: str, cwd: str = ".") -> str:
    raw = str(path or ".").replace("\\", "/").strip()
    normalized_cwd = normalize_model_relative_path(cwd)
    if not raw or raw == ".":
        return normalized_cwd

    # Absolute paths: legacy uploads, workspace aliases, or any container path.
    if raw.startswith("/"):
        legacy_upload = _legacy_upload_relative_path(raw)
        if legacy_upload is not None:
            return legacy_upload

        for alias in model_root_aliases():
            container_task_root = f"{CONTAINER_WORKSPACE.rstrip('/')}/{alias}"
            if raw == container_task_root or raw.startswith(f"{container_task_root}/"):
                return raw
        if IN_CONTAINER:
            return posixpath.normpath(raw)
        raise ValueError(f"Access denied: Absolute path '{raw}' is outside the task workspace.")

    # Relative paths that already start with a workspace alias are kept as they are.
    for alias in model_root_aliases():
        if raw == alias or raw.startswith(f"{alias}/"):
            return raw
        container_prefix = f"{CONTAINER_WORKSPACE.rstrip('/')}/{alias}"
        if raw == container_prefix or raw.startswith(f"{container_prefix}/"):
            return raw

    # Everything else is relative to the cwd.
    if normalized_cwd in ("", "."):
        return posixpath.normpath(raw)

    return posixpath.normpath(f"{normalized_cwd}/{raw}")


# The supervisor runs as root while bash commands are demoted to COMMAND_USER;
# anything the supervisor creates must be handed to that user, or later bash
# commands cannot modify or delete the model's own files. Best-effort, root-only.
def _chown_to_command_user(*paths: Path) -> None:
    if os.name != "posix" or not COMMAND_USER or COMMAND_USER == "root":
        return
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None or geteuid() != 0:
        return
    try:
        import pwd

        user_info = pwd.getpwnam(COMMAND_USER)
    except Exception:
        # Unknown user (or no pwd module): leave the ownership as it is.
        return
    for path in paths:
        try:
            os.chown(path, user_info.pw_uid, user_info.pw_gid)
        except OSError:
            # Best-effort: a path that vanished or cannot be re-owned is skipped.
            continue


# Resolve a path to a sandbox-safe location (container vs host rules). Paths are validated first;
# inside the container absolute Linux paths are returned as they are, other paths are resolved under
# the task root and must not escape it.
def get_secure_task_path(rel_path: str, kind: str = "path") -> Path:
    validate_model_path(rel_path, kind=kind)

    raw = str(rel_path or ".").replace("\\", "/").strip()

    # Inside the container, absolute Linux paths bypass task_root resolution.
    if IN_CONTAINER and raw.startswith("/"):
        return Path(posixpath.normpath(raw))

    normalized = normalize_model_relative_path(rel_path)

    if normalized == ".":
        return task_root()

    # Resolve under the task root and reject results that escape it (for example via "..").
    full_path = (task_root() / normalized).resolve()
    if not full_path.is_relative_to(task_root()):
        raise ValueError(f"Access denied: {rel_path} is outside sandbox workspace.")

    return full_path


# Return True when path is inside root.
def is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# Raise if path is a symlink whose target escapes the task root.
def _reject_symlink_escape(path: Path, original: str) -> None:
    if not path.is_symlink():
        return

    resolved = path.resolve()
    if not resolved.is_relative_to(task_root()):
        raise ValueError(
            f"Access denied: {original!r} is a symlink pointing outside the sandbox workspace."
        )


# Return True when an absolute path is under the model workspace.
def _is_workspace_absolute_path(path: str) -> bool:
    raw = str(path or "").replace("\\", "/").strip()
    if not raw.startswith("/"):
        return False
    legacy_upload = _legacy_upload_relative_path(raw)
    if legacy_upload is not None:
        return True
    for alias in model_root_aliases():
        container_task_root = f"{CONTAINER_WORKSPACE.rstrip('/')}/{alias}"
        if raw == container_task_root or raw.startswith(f"{container_task_root}/"):
            return True
    return False


# Reject absolute paths outside the model workspace for image-only tools.
def _reject_non_workspace_absolute_path(path: str, kind: str) -> None:
    raw = str(path or "").replace("\\", "/").strip()
    if raw.startswith("/") and not _is_workspace_absolute_path(raw):
        raise ValueError(
            f"{kind} only accepts absolute paths under the model workspace. "
            "Use bash for direct container/system paths."
        )


# Return file size or raise FileNotFoundError with a normalized path.
def _stat_size(path: Path, original: str) -> int:
    try:
        return path.stat().st_size
    except OSError as exc:
        raise FileNotFoundError(f"File not found: {normalize_model_relative_path(original)}") from exc


# Reject reads above max_bytes with a structured file_too_large error.
def _reject_oversized_read(path: Path, original: str, max_bytes: int | None = None) -> int:
    limit = MAX_FILE_READ_BYTES if max_bytes is None else max(1, int(max_bytes))
    size_bytes = _stat_size(path, original)
    if size_bytes > limit:
        raise SandboxToolError(
            "file_too_large",
            (
                f"File is too large to load into the supervisor safely: "
                f"{normalize_model_relative_path(original)} ({size_bytes} bytes, limit {limit} bytes). "
                "Use bash with streaming tools for large files."
            ),
            result={
                "path": normalize_model_relative_path(original),
                "size_bytes": size_bytes,
                "max_bytes": limit,
            },
        )
    return size_bytes


# Host import

# Return True when a host path is allowed for import: inside the task root or one of the allowed
# import roots.
def is_allowed_host_import(source: Path) -> bool:
    resolved = source.resolve()
    if is_within(resolved, task_root()):
        return True

    for root in ALLOWED_IMPORT_ROOTS:
        if is_within(resolved, Path(root)):
            return True

    return False


# Text preview helpers

# Render a numbered text excerpt: lines start_line..end_line plus CONTEXT_LINES lines on each side.
def render_numbered_context(text: str, start_line: int, end_line: int) -> str:
    lines = text.split("\n")
    total_lines = len(lines)
    context_start = max(1, start_line - CONTEXT_LINES)
    context_end = min(total_lines, end_line + CONTEXT_LINES)

    return "\n".join(
        f"{line_no:>4} | {lines[line_no - 1]}"
        for line_no in range(context_start, context_end + 1)
    )


# Build preview metadata for one text match.
def build_match_preview(text: str, start_index: int, needle_len: int) -> dict[str, Any]:
    start_line = text[:start_index].count("\n") + 1
    end_line = start_line + text[start_index : start_index + needle_len].count("\n")
    return {
        "line_start": start_line,
        "line_end": end_line,
        "context": render_numbered_context(text, start_line, end_line),
    }


# MIME, binary, and image helpers

# Guess MIME type from file extension.
def _guess_mime(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


# Heuristic check whether sample bytes look binary: text and image MIME types decide directly; otherwise
# null bytes mean binary, valid UTF-8 means text, and anything else is judged by the share of
# printable bytes (under 85% means binary).
def _is_probably_binary(data: bytes, mime_type: str) -> bool:
    if mime_type.startswith(TEXT_MIME_PREFIX):
        return False
    if mime_type.startswith(IMAGE_MIME_PREFIX):
        return True
    if not data:
        return False
    if b"\x00" in data[:TEXT_SAMPLE_BYTES]:
        return True

    sample = data[:TEXT_SAMPLE_BYTES]
    try:
        sample.decode("utf-8")
        return False
    except UnicodeDecodeError:
        # Not valid UTF-8: fall back to the printable-byte ratio below.
        pass

    text_like = 0
    for byte in sample:
        if byte in (9, 10, 13) or 32 <= byte <= 126 or byte >= 128:
            text_like += 1

    return (text_like / len(sample)) < 0.85


# Return basic image dimensions for common formats without full decoding.
def _image_dimensions(data: bytes, mime_type: str) -> dict[str, int] | None:
    try:
        # PNG: width/height stored as big-endian uint32 at bytes 16-23.
        if mime_type == "image/png" and data.startswith(b"\x89PNG\r\n\x1a\n") and len(data) >= 24:
            width, height = struct.unpack(">II", data[16:24])
            return {"width": width, "height": height}

        # GIF: logical screen dimensions at bytes 6-9.
        if mime_type == "image/gif" and data[:6] in (b"GIF87a", b"GIF89a") and len(data) >= 10:
            width, height = struct.unpack("<HH", data[6:10])
            return {"width": width, "height": height}

        # JPEG: scan markers until a Start-of-Frame segment is found.
        if mime_type == "image/jpeg" and data.startswith(b"\xff\xd8"):
            offset = 2
            sof_markers = {
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            }
            standalone_markers = {0x01, *range(0xD0, 0xD9)}
            while offset + 4 <= len(data):
                if data[offset] != 0xFF:
                    offset += 1
                    continue
                while offset < len(data) and data[offset] == 0xFF:
                    offset += 1
                if offset >= len(data):
                    break
                marker = data[offset]
                offset += 1
                if marker in standalone_markers:
                    continue
                if offset + 2 > len(data):
                    break
                segment_length = int.from_bytes(data[offset : offset + 2], "big")
                if segment_length < 2 or offset + segment_length > len(data):
                    break
                if marker in sof_markers and segment_length >= 7:
                    height = int.from_bytes(data[offset + 3 : offset + 5], "big")
                    width = int.from_bytes(data[offset + 5 : offset + 7], "big")
                    return {"width": width, "height": height}
                offset += segment_length

        # WebP: VP8X extended header or VP8L lossless bit-packed dimensions.
        if mime_type == "image/webp" and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            chunk = data[12:16]
            if chunk == b"VP8X" and len(data) >= 30:
                width = int.from_bytes(data[24:27], "little") + 1
                height = int.from_bytes(data[27:30], "little") + 1
                return {"width": width, "height": height}
            if chunk == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
                bits = int.from_bytes(data[21:25], "little")
                width = (bits & 0x3FFF) + 1
                height = ((bits >> 14) & 0x3FFF) + 1
                return {"width": width, "height": height}
    except (struct.error, ValueError):
        return None

    return None


# Build read_image result dict with optional base64 preview. The preview is inlined only when the file
# is no larger than max_preview_bytes; otherwise a warning says it was omitted.
def _image_payload(
    path: str,
    data: bytes,
    mime_type: str,
    *,
    include_preview: bool = True,
    max_preview_bytes: int = MAX_IMAGE_PREVIEW_BYTES,
) -> dict[str, Any]:
    warnings: list[str] = []
    preview_data = None
    max_preview_bytes = max(0, int(max_preview_bytes))

    # Inline the image as base64 when it fits the preview limit.
    if include_preview:
        if len(data) <= max_preview_bytes:
            preview_data = {
                "type": "inline_base64",
                "mime_type": mime_type,
                "data_base64": base64.b64encode(data).decode("utf-8"),
            }
        else:
            warnings.append(
                f"Image preview omitted because the file is larger than {max_preview_bytes} bytes."
            )

    # Assemble the result, adding width and height when the format is recognized.
    result: dict[str, Any] = {
        "path": path,
        "kind": "image",
        "mime": mime_type,
        "size_bytes": len(data),
        "encoding": None,
        "preview": preview_data,
    }
    dimensions = _image_dimensions(data, mime_type)
    if dimensions is not None:
        result.update(dimensions)

    return {
        "result": result,
        "warnings": warnings,
        "truncated": False,
    }


# Return True when a directory listing entry should be skipped (ignored names, and dotfiles unless
# hidden entries are requested).
def _should_skip_entry(name: str, include_hidden: bool) -> bool:
    if name in IGNORED_DIR_NAMES:
        return True
    if not include_hidden and name.startswith("."):
        return True
    return False


# Extract a 1-based line range from normalized text. Return the sliced text, the effective start and
# end lines (clamped to the file) and the total number of lines.
def _line_slice(
    text: str,
    start_line: int | None = None,
    end_line: int | None = None,
) -> tuple[str, int, int, int]:
    normalized = normalize_newlines(text)
    lines = normalized.split("\n")
    total_lines = len(lines)
    start = max(1, int(start_line or 1))
    end = min(total_lines, int(end_line or total_lines))
    if end < start:
        end = start
    return "\n".join(lines[start - 1 : end]), start, end, total_lines


# Truncate UTF-8 text to max_bytes (a character cut in half is dropped). Return the text and whether it
# was cut.
def _truncate_text(text: str, max_bytes: int) -> tuple[str, bool]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text, False

    truncated = encoded[:max_bytes].decode("utf-8", errors="ignore")
    return truncated, True


# File operations

# List files and directories inside the task workspace. Directories come first, then files, each
# sorted by name; the walk stops at max_entries entries (truncated=True) and descends at most depth
# levels below path.
def ls(
    path: str = ".",
    depth: int = 1,
    max_entries: int = MAX_LS_ENTRIES,
    include_hidden: bool = False,
) -> dict[str, Any]:
    # Validate the target directory.
    target = get_secure_task_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {normalize_model_relative_path(path)}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {normalize_model_relative_path(path)}")

    # Clamp the limits; entry depth is counted relative to the listed directory.
    depth = max(0, int(depth))
    max_entries = max(1, int(max_entries))
    entries: list[dict[str, Any]] = []
    truncated = False
    warnings: list[str] = []

    base_depth = len(target.relative_to(task_root()).parts)

    # Depth-limited recursive walk with entry cap.
    def _walk(current: Path) -> None:
        nonlocal truncated
        if truncated:
            return

        # Directories first, then files, each alphabetically (symlinks sort with the files).
        children = sorted(
            current.iterdir(),
            key=lambda item: (
                not (item.is_dir() and not item.is_symlink()),
                item.name.lower(),
            ),
        )
        for child in children:
            if _should_skip_entry(child.name, include_hidden):
                continue

            child_depth = len(child.relative_to(task_root()).parts) - base_depth
            if child_depth > depth:
                continue

            # Describe the entry; symlinks are listed but never followed.
            is_symlink = child.is_symlink()
            is_dir = child.is_dir() and not is_symlink
            entry: dict[str, Any] = {
                "path": child.relative_to(task_root()).as_posix(),
                "name": child.name,
                "type": "symlink" if is_symlink else ("directory" if is_dir else "file"),
                "depth": child_depth,
            }
            if child.is_file() and not is_symlink:
                try:
                    entry["size_bytes"] = child.stat().st_size
                    entry["mime"] = _guess_mime(child)
                except OSError:
                    # The file vanished or is unreadable: list it without size and MIME type.
                    pass

            # Stop at the entry cap; otherwise recurse into directories within the depth limit.
            entries.append(entry)
            if len(entries) >= max_entries:
                truncated = True
                warnings.append(f"Entry limit reached at {max_entries} items.")
                return

            if is_dir and child_depth < depth:
                _walk(child)

    # Walk the tree and report the entries found.
    _walk(target)
    return {
        "result": {
            "path": normalize_model_relative_path(path),
            "depth": depth,
            "entries": entries,
        },
        "warnings": warnings,
        "truncated": truncated,
    }


# Read a file from the task workspace. Images return an image payload, binary files metadata only, and
# text files the decoded content (newlines normalized to LF), optionally sliced by 1-based line range
# and cut to max_bytes.
def read(
    path: str,
    start_line: int | None = None,
    end_line: int | None = None,
    max_bytes: int = MAX_READ_BYTES,
) -> dict[str, Any]:
    # Validate the path and load the file (oversized files are rejected).
    target = get_secure_task_path(path)
    _reject_symlink_escape(target, path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {normalize_model_relative_path(path)}")

    size_bytes = _reject_oversized_read(target, path)
    data = target.read_bytes()
    mime_type = _guess_mime(target)
    normalized_path = normalize_model_relative_path(path)
    warnings: list[str] = []
    truncated = False

    if mime_type.startswith(IMAGE_MIME_PREFIX):
        return _image_payload(normalized_path, data, mime_type)

    # Binary files return metadata only; text files are decoded and optionally sliced.
    if _is_probably_binary(data, mime_type):
        return {
            "result": {
                "path": normalized_path,
                "kind": "binary",
                "mime": mime_type,
                "size_bytes": size_bytes,
                "encoding": None,
            },
            "warnings": warnings,
            "truncated": False,
        }

    # Decode the text and take the requested line range (the whole file when none was given).
    text, encoding = smart_decode(data)
    if start_line is not None or end_line is not None:
        content, line_start, line_end, total_lines = _line_slice(text, start_line, end_line)
    else:
        normalized_text = normalize_newlines(text)
        total_lines = len(normalized_text.split("\n"))
        content = normalized_text
        line_start = 1
        line_end = total_lines

    # Cap the returned content at max_bytes.
    content, truncated = _truncate_text(content, max(1, int(max_bytes)))
    if truncated:
        warnings.append(f"Read output truncated to {max_bytes} bytes.")

    return {
        "result": {
            "path": normalized_path,
            "kind": "text",
            "mime": mime_type,
            "encoding": encoding,
            "content": content,
            "start_line": line_start,
            "end_line": line_end,
            "total_lines": total_lines,
            "size_bytes": size_bytes,
        },
        "warnings": warnings,
        "truncated": truncated,
    }


# Return file metadata without loading the full file into memory: path, kind (image, binary or text),
# MIME type, size and, for text, the encoding detected from the first TEXT_SAMPLE_BYTES bytes.
def describe(path: str) -> dict[str, Any]:
    target = get_secure_task_path(path)
    _reject_symlink_escape(target, path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {normalize_model_relative_path(path)}")

    size_bytes = target.stat().st_size
    mime_type = _guess_mime(target)
    normalized_path = normalize_model_relative_path(path)

    # Only a small sample is read to classify the file.
    with target.open("rb") as handle:
        sample = handle.read(TEXT_SAMPLE_BYTES)

    if mime_type.startswith(IMAGE_MIME_PREFIX):
        kind = "image"
        encoding = None
    elif _is_probably_binary(sample, mime_type):
        kind = "binary"
        encoding = None
    else:
        _text, encoding = smart_decode(sample)
        kind = "text"

    return {
        "result": {
            "path": normalized_path,
            "kind": kind,
            "mime": mime_type,
            "size_bytes": size_bytes,
            "encoding": encoding,
        },
        "warnings": [],
        "truncated": False,
    }


# Read image metadata and, when small enough, an inline preview. Only workspace paths are accepted;
# a file that is not an image raises SandboxToolError("not_image") carrying its detected kind.
def read_image(
    path: str,
    include_preview: bool = True,
    max_preview_bytes: int = MAX_IMAGE_PREVIEW_BYTES,
) -> dict[str, Any]:
    # Validate the path and the file size.
    _reject_non_workspace_absolute_path(path, "view_image")
    target = get_secure_task_path(path)
    _reject_symlink_escape(target, path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {normalize_model_relative_path(path)}")

    size_bytes = _reject_oversized_read(target, path)
    mime_type = _guess_mime(target)
    normalized_path = normalize_model_relative_path(path)

    # Report non-image files with their detected kind instead of previewing them.
    if not mime_type.startswith(IMAGE_MIME_PREFIX):
        with target.open("rb") as handle:
            sample = handle.read(TEXT_SAMPLE_BYTES)
        raise SandboxToolError(
            "not_image",
            f"File is not an image: {normalized_path}",
            result={
                "path": normalized_path,
                "kind": "binary" if _is_probably_binary(sample, mime_type) else "text",
                "mime": mime_type,
                "size_bytes": size_bytes,
            },
        )

    # Load the image and build the payload with its optional inline preview.
    data = target.read_bytes()
    return _image_payload(
        normalized_path,
        data,
        mime_type,
        include_preview=bool(include_preview),
        max_preview_bytes=max_preview_bytes,
    )


# Write a UTF-8 text file inside the task workspace, creating missing parent directories; the content
# is written as given (no newline translation).
def write(path: str, content: str) -> dict[str, Any]:
    target = get_secure_task_path(path)

    # Note: _reject_symlink_escape is intentionally not called here.
    # For a new (non-existent) file it would be a no-op (is_symlink() = False).
    # For an existing file, get_secure_task_path already resolved and validated
    # the final path via .resolve() + is_relative_to(task_root()), so a symlink
    # pointing outside would have raised there.
    existed = target.exists()
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"Cannot overwrite directory: {normalize_model_relative_path(path)}")

    # Directories this write is about to create, restricted to the task workspace.
    # Absolute Linux paths deliberately bypass task_root (get_secure_task_path) so
    # the model can create root-owned files outside the workspace — we must NOT
    # chown those away from root. Inside task_root, bash runs as the command user
    # and must be able to modify/delete files that write created, so those are
    # handed to that user.
    inside_workspace = is_within(target, task_root())
    created_dirs: list[Path] = []
    if inside_workspace:
        probe = target.parent
        while not probe.exists() and probe != probe.parent and is_within(probe, task_root()):
            created_dirs.append(probe)
            probe = probe.parent

    # Create the file; newly created files and directories go to the command user.
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8", newline="")
    if not existed and inside_workspace:
        _chown_to_command_user(target, *created_dirs)

    return {
        "result": {
            "path": normalize_model_relative_path(path),
            "bytes_written": target.stat().st_size,
            "created": not existed,
            "overwrote": existed,
        },
        "warnings": [],
        "truncated": False,
    }


# Replace exact text matches inside a text file. Newlines are compared in normalized (LF) form and the
# file's original newline style is restored on write. A missing match or an ambiguous one without
# replace_all raises SandboxToolError. The result uses short keys to save tokens: p path, m match
# count, rep replacements made, all replace_all flag, cx/cxr/cxt compact context with its line range
# and truncation flag, ud unified diff.
def edit(path: str, old_str: str, new_str: str, replace_all: bool = False) -> dict[str, Any]:
    target = get_secure_task_path(path)
    _reject_symlink_escape(target, path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {normalize_model_relative_path(path)}")
    if old_str == "":
        raise ValueError("old_str must not be empty.")

    # Load the file as text; binary files cannot be edited.
    original_bytes = target.read_bytes()
    mime_type = _guess_mime(target)
    if _is_probably_binary(original_bytes, mime_type):
        raise SandboxToolError("binary_not_supported", "edit only supports text files.")

    # Normalize newlines and count the matches of old_str.
    original_text, _encoding = smart_decode(original_bytes)
    newline_style = detect_newline_style(original_text)
    normalized_content = normalize_newlines(original_text)
    normalized_old = normalize_newlines(old_str)
    normalized_new = normalize_newlines(new_str)
    match_count = normalized_content.count(normalized_old)

    if match_count == 0:
        raise SandboxToolError(
            "match_not_found",
            "old_str not found in file.",
            result={
                "path": normalize_model_relative_path(path),
                "match_count": 0,
            },
        )

    # Ambiguous multi-match edits require replace_all or more context.
    if match_count > 1 and not replace_all:
        previews = []
        start = 0
        while len(previews) < 3:
            match_index = normalized_content.find(normalized_old, start)
            if match_index < 0:
                break
            previews.append(build_match_preview(normalized_content, match_index, len(normalized_old)))
            start = match_index + len(normalized_old)

        raise SandboxToolError(
            "ambiguous_match",
            f"old_str found {match_count} times. Set replace_all=true or add more context.",
            result={
                "path": normalize_model_relative_path(path),
                "match_count": match_count,
                "matches": previews,
            },
        )

    # Apply the replacement (all matches with replace_all, otherwise the only/first one) and write it back
    # in the original newline style.
    replaced_count = match_count if replace_all else 1
    match_index = normalized_content.find(normalized_old)
    updated_normalized = normalized_content.replace(normalized_old, normalized_new, replaced_count)

    final_text = updated_normalized if newline_style == "\n" else updated_normalized.replace("\n", newline_style)
    target.write_text(final_text, encoding="utf-8", newline="")

    # Build the compact context around the first change and a unified diff for the result.
    updated_lines = updated_normalized.splitlines()
    change_start = normalized_content[:match_index].count("\n") + 1
    change_end = change_start + normalized_new.count("\n")
    context_text, context_start, context_end, context_truncated = _render_compact_line_edit_context(
        updated_lines, change_start, change_end,
    )

    diff_text = "\n".join(
        difflib.unified_diff(
            normalized_content.splitlines(),
            updated_normalized.splitlines(),
            fromfile=f"{normalize_model_relative_path(path)} (before)",
            tofile=f"{normalize_model_relative_path(path)} (after)",
            lineterm="",
            n=CONTEXT_LINES,
        )
    )

    return {
        "result": {
            "p": normalize_model_relative_path(path),
            "m": match_count,
            "rep": replaced_count,
            "all": bool(replace_all),
            "cx": context_text,
            "cxr": f"{context_start}:{context_end}",
            "cxt": context_truncated,
            "ud": diff_text,
        },
        "warnings": [],
        "truncated": False,
    }


# Line-range editing helpers

# Split normalized text into lines preserving trailing newline flag.
def _split_edit_lines(text: str) -> tuple[list[str], bool]:
    normalized = normalize_newlines(text)
    has_trailing_newline = normalized.endswith("\n")
    if normalized == "":
        return [], False
    lines = normalized.split("\n")
    if has_trailing_newline:
        lines = lines[:-1]
    return lines, has_trailing_newline


# Parse a 1-based line range string into start, end, and insert-mode flag. Accepted forms are "N",
# "start:end" or a two-item list; an end below the start (for example "12:11") means insert before
# line start. Raise ValueError for malformed or out-of-range input.
def _parse_line_range(range_str: Any, total_lines: int) -> tuple[int, int, bool]:
    # Turn the argument into a "left:right" string.
    if isinstance(range_str, (list, tuple)):
        if len(range_str) != 2:
            raise ValueError("range list must contain exactly two line numbers.")
        raw = f"{range_str[0]}:{range_str[1]}"
    else:
        raw = str(range_str or "").strip()
    if not raw:
        raise ValueError("range is required for mode='lines'.")

    if ":" in raw:
        left, right = raw.split(":", 1)
    else:
        left, right = raw, raw

    # Parse the bounds as integers.
    try:
        start = int(left)
        end = int(right)
    except ValueError as exc:
        raise ValueError("range must be a 1-based line number or 'start:end'.") from exc

    if start < 1 or end < 0:
        raise ValueError("range must use positive 1-based line numbers.")

    # Insert ranges may point one line past the end of the file; replace ranges must lie inside it.
    is_insert = end < start
    if is_insert:
        if start > total_lines + 1:
            raise ValueError(f"insert start line {start} is beyond end of file.")
        return start, end, True

    if total_lines == 0:
        raise ValueError("cannot replace lines in an empty file; use insert range '1:0' (or '1:1' with empty content).")
    if start > total_lines or end > total_lines:
        raise ValueError(f"range {start}:{end} is outside file with {total_lines} lines.")

    return start, end, False


# Normalize edit content into a list of lines.
def _new_edit_lines(content: str) -> list[str]:
    normalized = normalize_newlines(content)
    if normalized == "":
        return []
    lines = normalized.split("\n")
    if normalized.endswith("\n"):
        lines = lines[:-1]
    return lines


# Build compact numbered context around edited lines. Lines highlight_start..highlight_end are marked
# with "+"; the context is widened by up to max_radius lines on each side and narrowed down to
# min_radius until it fits max_chars, then individual lines are clipped as a last resort. Return the
# rendered lines, the first and last line shown, and whether anything was shortened.
def _render_compact_line_edit_context(
    lines: list[str],
    highlight_start: int,
    highlight_end: int,
    max_chars: int = 900,
    min_radius: int = 2,
    max_radius: int = 6,
) -> tuple[list[str], int, int, bool]:
    if not lines:
        return [], 1, 0, False

    line_count = len(lines)
    highlight_start = max(1, min(line_count, highlight_start))
    highlight_end = max(highlight_start, min(line_count, highlight_end))

    # Cut text to limit characters, ending with an ellipsis; return the text and whether it was cut.
    def shorten(text: str, limit: int) -> tuple[str, bool]:
        if len(text) <= limit:
            return text, False
        if limit <= 1:
            return "…", True
        return text[: limit - 1].rstrip() + "…", True

    # Render lines start_line..end_line as "<marker>L<number> <text>", with the per-line text budget
    # derived from max_chars; return the lines and whether any was shortened.
    def render(start_line: int, end_line: int) -> tuple[list[str], bool]:
        visible_count = max(1, end_line - start_line + 1)

        # Keep every visible line represented. Long individual lines are
        # shortened before the whole context is shortened.
        text_budget = max(48, (max_chars // visible_count) - 8)
        rendered = []
        truncated_any = False
        for line_no in range(start_line, end_line + 1):
            marker = "+" if highlight_start <= line_no <= highlight_end else " "
            text, was_truncated = shorten(lines[line_no - 1], text_budget)
            truncated_any = truncated_any or was_truncated
            rendered.append(f"{marker}L{line_no} {text}")
        return rendered, truncated_any

    # Return the first and last line shown for a context radius around the highlighted lines.
    def bounds(radius: int) -> tuple[int, int]:
        return (
            max(1, highlight_start - radius),
            min(line_count, highlight_end + radius),
        )

    chosen_start, chosen_end = bounds(max_radius)
    chosen_text, chosen_truncated = render(chosen_start, chosen_end)
    truncated = False

    # Try radii from largest to smallest and return the first context that fits the budget.
    for radius in range(max_radius, min_radius - 1, -1):
        start_line, end_line = bounds(radius)
        text, line_truncated = render(start_line, end_line)
        if sum(len(line) for line in text) + max(0, len(text) - 1) <= max_chars:
            return text, start_line, end_line, line_truncated or radius < max_radius
        chosen_start, chosen_end, chosen_text = start_line, end_line, text
        chosen_truncated = line_truncated

    # Even the smallest context is too long: clip lines until the character budget is used up.
    chosen_len = sum(len(line) for line in chosen_text) + max(0, len(chosen_text) - 1)
    if chosen_len > max_chars and chosen_text:
        truncated = True
        remaining = max_chars
        clipped = []
        for line in chosen_text:
            if remaining <= 0:
                break
            line_budget = min(len(line), remaining)
            if line_budget < len(line):
                line, _ = shorten(line, line_budget)
            clipped.append(line)
            remaining -= len(line) + 1
        chosen_text = clipped

    return chosen_text, chosen_start, chosen_end, truncated or chosen_truncated


# Replace or insert text by 1-based line range inside a text file. An optional anchor must equal the
# line at the start of the range (ignoring surrounding whitespace); on mismatch SandboxToolError
# reports the actual line and suggests a nearby range. The result uses short keys to save tokens:
# p path, r range, rm lines removed, add lines added, n new line count, d line count change,
# cx/cxr/cxt compact context with its line range and truncation flag, ud unified diff.
def edit_lines(
    path: str,
    range_str: Any,
    content: str,
    anchor: str | None = None,
    context_lines: int = 2,
) -> dict[str, Any]:
    target = get_secure_task_path(path)
    _reject_symlink_escape(target, path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {normalize_model_relative_path(path)}")

    # Load the file as text (binary files cannot be edited) and split it into lines.
    original_bytes = target.read_bytes()
    mime_type = _guess_mime(target)
    if _is_probably_binary(original_bytes, mime_type):
        raise SandboxToolError("binary_not_supported", "edit only supports text files.")

    original_text, _encoding = smart_decode(original_bytes)
    newline_style = detect_newline_style(original_text)
    original_lines, had_trailing_newline = _split_edit_lines(original_text)
    total_before = len(original_lines)
    start, end, is_insert = _parse_line_range(range_str, total_before)

    # Optional anchor line check with nearby-line suggestion on mismatch.
    if anchor is not None:
        if is_insert:
            actual = original_lines[start - 1] if start <= total_before else ""
        else:
            actual = original_lines[start - 1]
        expected = str(anchor)
        if actual.strip() != expected.strip():
            suggestion = None
            radius_start = max(1, start - 20)
            radius_end = min(total_before, start + 20)
            for line_no in range(radius_start, radius_end + 1):
                if original_lines[line_no - 1].strip() == expected.strip():
                    old_len = 0 if is_insert else end - start + 1
                    suggestion_end = line_no - 1 if is_insert else line_no + old_len - 1
                    suggestion = f"{line_no}:{suggestion_end}"
                    break

            result: dict[str, Any] = {
                "path": normalize_model_relative_path(path),
                "range": f"{start}:{end}",
                f"actual_line_{start}": actual,
            }
            if suggestion:
                result["suggestion"] = f"Use range {suggestion}."

            raise SandboxToolError(
                "anchor_mismatch",
                f"Line {start} is {actual!r}, expected {expected!r}.",
                result=result,
            )

    # Insert before start line, or replace the inclusive start:end range.
    replacement_lines = _new_edit_lines(content)
    if is_insert:
        updated_lines = (
            original_lines[: start - 1]
            + replacement_lines
            + original_lines[start - 1 :]
        )
        lines_before = 0
    else:
        updated_lines = original_lines[: start - 1] + replacement_lines + original_lines[end:]
        lines_before = end - start + 1

    total_after = len(updated_lines)
    lines_after = len(replacement_lines)
    shift = total_after - total_before

    # Write the file back with its original trailing newline and newline style.
    final_text = "\n".join(updated_lines)
    if had_trailing_newline:
        final_text += "\n"
    if newline_style != "\n":
        final_text = final_text.replace("\n", newline_style)
    target.write_text(final_text, encoding="utf-8", newline="")

    # Build the compact context around the changed lines and a unified diff for the result.
    highlight_start = start
    highlight_end = start + max(lines_after, 1) - 1
    context_text, context_start, context_end, context_truncated = _render_compact_line_edit_context(
        updated_lines,
        highlight_start,
        highlight_end,
        min_radius=max(1, min(2, context_lines)),
        max_radius=max(3, context_lines + 2),
    )

    diff_text = "\n".join(
        difflib.unified_diff(
            original_lines,
            updated_lines,
            fromfile=f"{normalize_model_relative_path(path)} (before)",
            tofile=f"{normalize_model_relative_path(path)} (after)",
            lineterm="",
            n=context_lines,
        )
    )

    return {
        "result": {
            "p": normalize_model_relative_path(path),
            "r": f"{start}:{end}",
            "rm": lines_before,
            "add": lines_after,
            "n": total_after,
            "d": shift,
            "cx": context_text,
            "cxr": f"{context_start}:{context_end}",
            "cxt": context_truncated,
            "ud": diff_text,
        },
        "warnings": [],
        "truncated": False,
    }


# Search

# Find files and directories inside the task workspace by name pattern (fnmatch on the entry name),
# optionally limited to files or directories. Ignored and hidden entries are skipped, the walk goes
# at most max_depth levels deep and stops at max_results matches.
def find(
    path: str = ".",
    name_pattern: str | None = None,
    type_filter: str | None = None,
    max_depth: int = 8,
    max_results: int = MAX_FIND_RESULTS,
) -> dict[str, Any]:
    # Validate the start directory and the type filter.
    target = get_secure_task_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {normalize_model_relative_path(path)}")
    if not target.is_dir():
        raise NotADirectoryError(f"Not a directory: {normalize_model_relative_path(path)}")

    normalized_type = str(type_filter or "any").strip().lower()
    if normalized_type not in {"any", "file", "directory"}:
        raise ValueError("type must be one of: any, file, directory.")

    name_pattern = str(name_pattern or "*")
    max_depth = max(0, int(max_depth))
    max_results = max(1, int(max_results))
    matches: list[dict[str, Any]] = []
    warnings: list[str] = []
    truncated = False

    base_depth = len(target.relative_to(task_root()).parts)

    # Walk the tree top-down; pruning dirnames in place controls which subdirectories os.walk visits.
    for root, dirnames, filenames in os.walk(target):
        root_path = Path(root)
        current_depth = len(root_path.relative_to(task_root()).parts) - base_depth
        dirnames[:] = [
            name
            for name in sorted(dirnames)
            if not _should_skip_entry(name, include_hidden=False)
        ]

        # Do not descend below max_depth.
        if current_depth >= max_depth:
            dirnames[:] = []

        # Directory matches of this level.
        if normalized_type in {"any", "directory"}:
            for dirname in dirnames:
                if not fnmatch.fnmatch(dirname, name_pattern):
                    continue
                child = root_path / dirname
                matches.append(
                    {
                        "path": child.relative_to(task_root()).as_posix(),
                        "name": dirname,
                        "type": "directory",
                        "depth": len(child.relative_to(task_root()).parts) - base_depth,
                    }
                )
                if len(matches) >= max_results:
                    truncated = True
                    warnings.append(f"Result limit reached at {max_results} items.")
                    return {
                        "result": {
                            "path": normalize_model_relative_path(path),
                            "matches": matches,
                        },
                        "warnings": warnings,
                        "truncated": True,
                    }

        # File matches of this level.
        if normalized_type in {"any", "file"}:
            for filename in sorted(filenames):
                if _should_skip_entry(filename, include_hidden=False):
                    continue
                if not fnmatch.fnmatch(filename, name_pattern):
                    continue
                child = root_path / filename
                matches.append(
                    {
                        "path": child.relative_to(task_root()).as_posix(),
                        "name": filename,
                        "type": "file",
                        "depth": len(child.relative_to(task_root()).parts) - base_depth,
                    }
                )
                if len(matches) >= max_results:
                    truncated = True
                    warnings.append(f"Result limit reached at {max_results} items.")
                    return {
                        "result": {
                            "path": normalize_model_relative_path(path),
                            "matches": matches,
                        },
                        "warnings": warnings,
                        "truncated": True,
                    }

    return {
        "result": {
            "path": normalize_model_relative_path(path),
            "matches": matches,
        },
        "warnings": warnings,
        "truncated": truncated,
    }


# Search text files inside the task workspace for a regular expression (case-insensitive unless
# case_sensitive is set), optionally limited by a glob on the workspace-relative path. Binary and
# oversized files are skipped (oversized ones with a warning); each match carries up to
# context_before/context_after surrounding lines; the search stops at max_results matches.
def grep(
    pattern: str,
    path: str = ".",
    glob: str | None = None,
    case_sensitive: bool = False,
    context_before: int = 0,
    context_after: int = 0,
    max_results: int = MAX_GREP_RESULTS,
) -> dict[str, Any]:
    target = get_secure_task_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {normalize_model_relative_path(path)}")

    # Compile the pattern; an invalid expression is reported as a ValueError.
    regex_flags = re.MULTILINE if case_sensitive else (re.MULTILINE | re.IGNORECASE)
    try:
        regex = re.compile(pattern, regex_flags)
    except re.error as exc:
        raise ValueError(f"Invalid regular expression: {exc}") from exc

    # Clamp the limits.
    max_results = max(1, int(max_results))
    context_before = max(0, int(context_before))
    context_after = max(0, int(context_after))
    matches: list[dict[str, Any]] = []
    warnings: list[str] = []
    truncated = False

    # Single file: grep directly; directory: collect candidates then scan each file.
    if target.is_file():
        _reject_symlink_escape(target, path)
        _reject_oversized_read(target, path)
        candidates = [target]
    else:
        candidates = []
        for root, dirnames, filenames in os.walk(target):
            dirnames[:] = [
                name
                for name in sorted(dirnames)
                if not _should_skip_entry(name, include_hidden=False)
            ]
            for filename in sorted(filenames):
                if _should_skip_entry(filename, include_hidden=False):
                    continue
                file_path = Path(root) / filename
                rel_path = file_path.relative_to(task_root()).as_posix()
                if glob and not fnmatch.fnmatch(rel_path, glob):
                    continue
                candidates.append(file_path)

    # Scan every candidate line by line; unsafe symlinks abort, oversized and binary files are skipped.
    for candidate in candidates:
        _reject_symlink_escape(candidate, candidate.relative_to(task_root()).as_posix())
        try:
            _reject_oversized_read(candidate, candidate.relative_to(task_root()).as_posix())
        except SandboxToolError as exc:
            if exc.error_type != "file_too_large":
                raise
            warnings.append(exc.message)
            continue
        data = candidate.read_bytes()
        mime_type = _guess_mime(candidate)
        if _is_probably_binary(data, mime_type):
            continue

        text, _encoding = smart_decode(data)
        lines = normalize_newlines(text).split("\n")
        for index, line in enumerate(lines, start=1):
            if regex.search(line) is None:
                continue

            # Record the match with its context lines.
            before_start = max(1, index - context_before)
            after_end = min(len(lines), index + context_after)
            matches.append(
                {
                    "path": candidate.relative_to(task_root()).as_posix(),
                    "line_number": index,
                    "line": line,
                    "before": lines[before_start - 1 : index - 1],
                    "after": lines[index:after_end],
                }
            )

            # Stop as soon as the result cap is reached.
            if len(matches) >= max_results:
                truncated = True
                warnings.append(f"Result limit reached at {max_results} items.")
                return {
                    "result": {
                        "path": normalize_model_relative_path(path),
                        "pattern": pattern,
                        "matches": matches,
                    },
                    "warnings": warnings,
                    "truncated": True,
                }

    return {
        "result": {
            "path": normalize_model_relative_path(path),
            "pattern": pattern,
            "matches": matches,
        },
        "warnings": warnings,
        "truncated": truncated,
    }


# Directory operations

# Create a directory inside the task workspace; an existing directory is not an error, an existing
# non-directory is.
def mkdir(path: str, parents: bool = True) -> dict[str, Any]:
    target = get_secure_task_path(path)
    existed = target.exists()
    if existed and not target.is_dir():
        raise SandboxToolError(
            "path_conflict",
            f"Path already exists and is not a directory: {normalize_model_relative_path(path)}",
        )

    target.mkdir(parents=parents, exist_ok=True)

    return {
        "result": {
            "path": normalize_model_relative_path(path),
            "created": not existed,
            "already_exists": existed,
        },
        "warnings": [],
        "truncated": False,
    }


# Move or rename a file or directory inside the task workspace. An existing destination is removed
# first when overwrite is set, otherwise it is an error.
def move(src: str, dst: str, overwrite: bool = False) -> dict[str, Any]:
    # Resolve both paths and require an existing source.
    source = get_secure_task_path(src, kind="src")
    destination = get_secure_task_path(dst, kind="dst")
    if not source.exists():
        raise FileNotFoundError(f"Path not found: {normalize_model_relative_path(src)}")

    # Handle an existing destination.
    if destination.exists():
        if not overwrite:
            raise SandboxToolError(
                "destination_exists",
                f"Destination already exists: {normalize_model_relative_path(dst)}",
            )
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink()

    # Create missing parent directories and move.
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), str(destination))
    return {
        "result": {
            "src": normalize_model_relative_path(src),
            "dst": normalize_model_relative_path(dst),
            "overwrote": bool(overwrite),
        },
        "warnings": [],
        "truncated": False,
    }


# Delete a file or directory inside the task workspace; directories need recursive=True.
def delete(path: str, recursive: bool = False) -> dict[str, Any]:
    target = get_secure_task_path(path)
    if not target.exists():
        raise FileNotFoundError(f"Path not found: {normalize_model_relative_path(path)}")

    if target.is_dir():
        if not recursive:
            raise SandboxToolError(
                "recursive_required",
                f"Refusing to delete directory without recursive=true: {normalize_model_relative_path(path)}",
            )
        shutil.rmtree(target)
        deleted_type = "directory"
    else:
        target.unlink()
        deleted_type = "file"

    return {
        "result": {
            "path": normalize_model_relative_path(path),
            "type": deleted_type,
        },
        "warnings": [],
        "truncated": False,
    }


# Workspace reset

# Clear the dedicated sandbox workspace without deleting its root; return the names removed.
def clear_workspace() -> dict[str, Any]:
    root = task_root()
    cleared: list[str] = []

    for child in root.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)
        cleared.append(child.name)

    return {"ok": True, "cleared": cleared}
