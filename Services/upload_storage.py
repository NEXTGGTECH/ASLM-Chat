# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import re
import uuid
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from Settings import settings

from Services.file_manifests import UploadedFileManifest, build_uploaded_file_manifest, normalize_upload_name
from Services.file_manifests import guess_upload_mime


# Host locations: the mcp-sandbox tree, the folder inside it that receives user uploads, and the
# separate folder that holds the private manifests of those uploads.
SANDBOX_ROOT = settings.BASE_DIR / "Tools" / "mcp-sandbox" / "_sandbox"
USER_UPLOAD_ROOT = SANDBOX_ROOT / "User"
USER_FILE_MANIFEST_ROOT = settings.BASE_DIR / "Tools" / "user_files"

# Path of USER_UPLOAD_ROOT as the model sees it inside the sandbox, and the suffix of manifest files.
SANDBOX_MODEL_PREFIX = "/workspace/_sandbox/User"
UPLOAD_MANIFEST_SUFFIX = ".manifest.json"

# Size limits in bytes: the largest accepted upload, and the largest upload still kept in memory
# so that its full manifest (text preview, archive tree) can be built.
MAX_UPLOAD_BYTES = 16 * 1024 * 1024 * 1024
INLINE_MANIFEST_MAX_BYTES = 64 * 1024 * 1024


# Host and model paths for one upload destination: the tool server id, the host upload folder
# and the path prefix under which the model sees that folder.
@dataclass(frozen=True)
class UploadStorageTarget:
    server_id: str
    upload_root: Path
    model_prefix: str


# Return a stable list of tool server ids from request payloads.
def normalize_tool_server_ids(tool_server_ids: list[str] | None) -> list[str]:
    if not tool_server_ids:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_value in tool_server_ids:
        value = str(raw_value or "").strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        normalized.append(value)
    return normalized


# Route UI uploads to the mcp-sandbox tree. The tool server ids are accepted for the caller's
# convenience but do not change the target: every upload goes to the sandbox.
def resolve_upload_storage_target(tool_server_ids: list[str] | None = None) -> UploadStorageTarget:
    _ = tool_server_ids
    return UploadStorageTarget("sandbox", USER_UPLOAD_ROOT, SANDBOX_MODEL_PREFIX)


# Turn a scope value into a safe directory name: unsafe characters become underscores, the result
# is at most 96 characters, and an empty value maps to "pending".
def _safe_scope(value: str | None) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return "pending"
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", raw_value)[:96] or "pending"


# Build the on-disk file name "<file id>__<sanitized original name>" so equal names never collide.
def _stored_file_name(file_id: str, original_name: str) -> str:
    safe_name = normalize_upload_name(original_name)
    safe_name = re.sub(r"[\x00-\x1f<>:\"|?*]+", "_", safe_name).strip(" .") or "uploaded-file"
    normalized_id = re.sub(r"[^a-zA-Z0-9-]+", "", str(file_id or "")) or "upload"
    return f"{normalized_id}__{safe_name}"


# Return UI-facing kind and label for one upload.
def display_kind_for_upload(name: str, mime: str) -> tuple[str, str]:
    suffix = Path(normalize_upload_name(name)).suffix.lower()
    normalized_mime = str(mime or "").lower()

    # Media first (by MIME or extension), then archives, Office/PDF documents, code, plain text,
    # and a generic "file" fallback.
    if normalized_mime.startswith("image/"):
        return "image", "Image"
    if normalized_mime.startswith("audio/") or suffix in {".mp3", ".wav", ".ogg", ".oga", ".m4a", ".aac", ".flac", ".opus"}:
        return "audio", "Audio"
    if normalized_mime.startswith("video/") or suffix in {".mp4", ".webm", ".mov", ".m4v", ".ogv", ".avi", ".mkv"}:
        return "video", "Video"
    if suffix == ".zip" or normalized_mime in {"application/zip", "application/x-zip-compressed"}:
        return "archive", "ZIP archive"
    if suffix in {".rar", ".7z"}:
        return "archive", "Archive"
    if suffix == ".pdf" or normalized_mime == "application/pdf":
        return "document", "PDF document"
    if suffix == ".docx":
        return "document", "Word document"
    if suffix == ".xlsx":
        return "table", "Excel spreadsheet"
    if suffix == ".csv":
        return "table", "CSV table"
    if suffix == ".pptx":
        return "presentation", "PowerPoint presentation"
    if suffix in {".py", ".js", ".ts", ".css", ".html", ".sql", ".sh", ".ps1"}:
        return "code", "Code file"
    if normalized_mime.startswith("text/") or suffix in {".txt", ".md", ".log", ".json", ".yaml", ".yml", ".xml"}:
        return "text", "Text file"
    return "file", "File"


# Return the small user-facing upload payload (identity, size, display kind and content URL)
# for a manifest object or a stored manifest dict.
def public_upload_payload(manifest: UploadedFileManifest | dict[str, Any], *, status: str = "ready") -> dict[str, Any]:
    if isinstance(manifest, UploadedFileManifest):
        file_id = manifest.file_id
        name = manifest.name
        mime = manifest.mime
        size_bytes = manifest.size_bytes
    else:
        file_id = str((manifest or {}).get("file_id") or "")
        name = str((manifest or {}).get("name") or "uploaded-file")
        mime = str((manifest or {}).get("mime") or "application/octet-stream")
        size_bytes = int((manifest or {}).get("size_bytes") or 0)

    display_kind, type_label = display_kind_for_upload(name, mime)
    return {
        "file_id": file_id,
        "name": name,
        "mime_type": mime,
        "size_bytes": size_bytes,
        "status": status,
        "display_kind": display_kind,
        "type_label": type_label,
        "content_url": f"/api/uploads/{file_id}/content/" if file_id else "",
    }


# Return a model-facing manifest that respects the selected sandbox state: when the sandbox is
# off, the sandbox path is hidden and the sandbox tool is removed from the recommended tools.
def model_upload_payload(manifest: dict[str, Any], *, sandbox_enabled: bool = False) -> dict[str, Any]:
    payload = dict(manifest or {})
    if not sandbox_enabled:
        payload["sandbox_path"] = None
        recommended_tools = payload.get("recommended_tools")
        if isinstance(recommended_tools, list):
            payload["recommended_tools"] = [
                tool for tool in recommended_tools if tool != "sandbox"
            ]
    return payload


# Return the legacy sidecar manifest path that sits next to an uploaded file.
def _manifest_sidecar_path(file_path: Path) -> Path:
    return file_path.with_name(f"{file_path.name}{UPLOAD_MANIFEST_SUFFIX}")


# Return the manifest folder for a content hash (one folder per sha256, "unknown" when it is empty).
def _manifest_storage_dir(sha256: str) -> Path:
    safe_sha = re.sub(r"[^a-fA-F0-9]+", "", str(sha256 or "").lower())
    return USER_FILE_MANIFEST_ROOT / (safe_sha or "unknown")


# Return the path of a manifest inside the hash folder, named after the sanitized file id.
def _manifest_storage_path(manifest: UploadedFileManifest | dict[str, Any]) -> Path:
    if isinstance(manifest, UploadedFileManifest):
        sha256 = manifest.sha256
        file_id = manifest.file_id
    else:
        sha256 = str(manifest.get("sha256") or "")
        file_id = str(manifest.get("file_id") or "")
    safe_id = re.sub(r"[^a-fA-F0-9-]+", "", str(file_id or "")) or "manifest"
    return _manifest_storage_dir(sha256) / f"{safe_id}{UPLOAD_MANIFEST_SUFFIX}"


# Write the manifest as JSON into external manifest storage, creating folders as needed.
def _write_manifest(manifest: UploadedFileManifest) -> None:
    manifest_path = _manifest_storage_path(manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Build the path of a stored upload as the model sees it: prefix, scope folder and stored name.
def _model_sandbox_path(model_prefix: str, scope: str, stored_name: str) -> str:
    return f"{model_prefix}/{scope}/{stored_name}".replace("\\", "/")


# Map a stored manifest sandbox_path back to a host file path.
# Raise FileNotFoundError when the path is missing, outside the upload prefix or not a file,
# and ValueError when it would escape the upload root.
def resolve_uploaded_file_host_path(manifest: dict[str, Any]) -> Path:
    sandbox_path = str((manifest or {}).get("sandbox_path") or "").strip().replace("\\", "/")
    if not sandbox_path:
        raise FileNotFoundError("Uploaded file content is not available")

    # Strip the model prefix and resolve under the upload root, refusing path traversal.
    mcp_prefix = f"{SANDBOX_MODEL_PREFIX}/"
    if not sandbox_path.startswith(mcp_prefix):
        raise FileNotFoundError("Uploaded file content is not available")
    relative = sandbox_path[len(mcp_prefix):].lstrip("/")
    root = USER_UPLOAD_ROOT.resolve()

    target = (root / relative).resolve()
    if root != target and root not in target.parents:
        raise ValueError("Unsafe uploaded file path")
    if not target.is_file():
        raise FileNotFoundError("Uploaded file not found")
    return target


# Return the SHA-256 hex digest of the file bytes.
def _file_sha256(file_bytes: bytes) -> str:
    return hashlib.sha256(file_bytes or b"").hexdigest()


# Format a byte count for error messages as GB, MB or bytes.
def _format_upload_size(size_bytes: int) -> str:
    if size_bytes >= 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes} bytes"


# Stream the upload chunk by chunk into a temporary file under <incoming_root>/_incoming while
# hashing it. Return the temp path, the size, the SHA-256 hex digest and the full bytes (None when
# the file exceeds INLINE_MANIFEST_MAX_BYTES). Raise ValueError above MAX_UPLOAD_BYTES; the temp
# file is removed on any failure.
def _stream_upload_to_temp(
    uploaded_file: Any,
    *,
    incoming_root: Path,
) -> tuple[Path, int, str, bytes | None]:
    # Prepare the temp target, the running hash and the in-memory copy of the file.
    incoming_dir = incoming_root / "_incoming"
    incoming_dir.mkdir(parents=True, exist_ok=True)
    temp_path = incoming_dir / f"{uuid.uuid4().hex}.upload"
    sha256 = hashlib.sha256()
    size_bytes = 0
    inline_chunks: list[bytes] = []
    keep_inline = True

    # Copy the chunks to disk, enforcing the size limit and dropping the in-memory copy once the
    # file is too big to keep inline.
    chunks = uploaded_file.chunks() if hasattr(uploaded_file, "chunks") else [uploaded_file.read()]
    try:
        with temp_path.open("wb") as handle:
            for chunk in chunks:
                if not chunk:
                    continue
                chunk_bytes = bytes(chunk)
                size_bytes += len(chunk_bytes)
                if size_bytes > MAX_UPLOAD_BYTES:
                    raise ValueError(f"File is too large (max {_format_upload_size(MAX_UPLOAD_BYTES)})")
                sha256.update(chunk_bytes)
                handle.write(chunk_bytes)
                if keep_inline:
                    if size_bytes <= INLINE_MANIFEST_MAX_BYTES:
                        inline_chunks.append(chunk_bytes)
                    else:
                        inline_chunks.clear()
                        keep_inline = False
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            # The temp file is already gone or locked; the original error is re-raised below.
            pass
        raise

    inline_bytes = b"".join(inline_chunks) if keep_inline else None
    return temp_path, size_bytes, sha256.hexdigest(), inline_bytes


# Read a manifest JSON file. Return None when it is unreadable, invalid or not a JSON object.
def _load_manifest_from_sidecar(sidecar_path: Path) -> dict[str, Any] | None:
    try:
        manifest = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return manifest if isinstance(manifest, dict) else None


# Yield the stored manifest files in the folder of the given content hash.
def _iter_manifest_paths_for_sha(sha256: str):
    manifest_dir = _manifest_storage_dir(sha256)
    if not manifest_dir.exists():
        return
    yield from manifest_dir.glob(f"*{UPLOAD_MANIFEST_SUFFIX}")


# Find a stored manifest whose hash, size and name (and sandbox path, when given) all match.
# Return None when there is none.
def _find_stored_manifest(
    *,
    sha256: str,
    size_bytes: int,
    clean_name: str,
    sandbox_path: str | None = None,
) -> dict[str, Any] | None:
    for manifest_path in _iter_manifest_paths_for_sha(sha256):
        manifest = _load_manifest_from_sidecar(manifest_path)
        if not manifest:
            continue
        if str(manifest.get("sha256") or "") != sha256:
            continue
        if int(manifest.get("size_bytes") or 0) != size_bytes:
            continue
        if str(manifest.get("name") or "") != clean_name:
            continue
        if sandbox_path and str(manifest.get("sandbox_path") or "") != sandbox_path:
            continue
        return manifest
    return None


# Find an already stored copy of the same upload in the target folder, using the in-memory bytes.
# Return the stored file path and its manifest (None when no manifest exists and the match was
# made by comparing the file content), or None when there is no copy.
def _find_existing_upload(
    target_dir: Path,
    *,
    model_prefix: str,
    safe_scope: str,
    clean_name: str,
    file_bytes: bytes,
) -> tuple[Path, dict[str, Any] | None] | None:
    expected_sha256 = _file_sha256(file_bytes)
    expected_size = len(file_bytes)
    expected_name = str(clean_name or "").strip()

    for file_path in target_dir.iterdir():
        if not file_path.is_file() or file_path.name.endswith(UPLOAD_MANIFEST_SUFFIX):
            continue

        # First choice: a manifest in external storage that describes exactly this file.
        expected_sandbox_path = _model_sandbox_path(model_prefix, safe_scope, file_path.name)
        manifest = _find_stored_manifest(
            sha256=expected_sha256,
            size_bytes=expected_size,
            clean_name=expected_name,
            sandbox_path=expected_sandbox_path,
        )
        if manifest:
            return file_path, manifest

        # Second choice: a legacy sidecar manifest next to the file. A sidecar that describes a
        # different file rules this file out.
        sidecar_path = _manifest_sidecar_path(file_path)
        manifest = _load_manifest_from_sidecar(sidecar_path) if sidecar_path.exists() else None
        if manifest:
            manifest_sha = str(manifest.get("sha256") or "")
            manifest_size = int(manifest.get("size_bytes") or 0)
            manifest_name = str(manifest.get("name") or "")
            if (
                manifest_sha == expected_sha256
                and manifest_size == expected_size
                and manifest_name == expected_name
            ):
                return file_path, manifest
            continue

        # No manifest at all: fall back to comparing the file content byte for byte.
        try:
            existing_bytes = file_path.read_bytes()
        except OSError:
            # An unreadable candidate cannot be the same upload; check the next file.
            continue
        if existing_bytes == file_bytes:
            return file_path, None
    return None


# Find an already stored copy of a large upload that was not kept in memory, through its stored
# manifest alone. Return the stored file path and manifest, or None when there is no such copy.
def _find_existing_upload_by_hash(
    target_dir: Path,
    *,
    sha256: str,
    size_bytes: int,
    clean_name: str,
) -> tuple[Path, dict[str, Any] | None] | None:
    manifest = _find_stored_manifest(
        sha256=sha256,
        size_bytes=size_bytes,
        clean_name=clean_name,
    )
    if not manifest:
        return None

    sandbox_path = str(manifest.get("sandbox_path") or "").strip()
    stored_name = sandbox_path.rsplit("/", 1)[-1] if sandbox_path else ""
    if not stored_name:
        return None

    existing_path = target_dir / stored_name
    if existing_path.is_file():
        return existing_path, manifest
    return None


# Build a manifest object from a stored dict. Return None when its fields do not match the dataclass.
def _manifest_from_dict(manifest: dict[str, Any]) -> UploadedFileManifest | None:
    try:
        return UploadedFileManifest(**manifest)
    except TypeError:
        return None


# Parse a stored manifest and make sure its sandbox_path points at the given stored file,
# patching it when the manifest was written for another location. Return None when it is invalid.
def _normalize_existing_manifest_for_path(
    manifest: dict[str, Any],
    *,
    model_prefix: str,
    safe_scope: str,
    stored_name: str,
) -> UploadedFileManifest | None:
    parsed = _manifest_from_dict(manifest)
    if parsed is None:
        return None

    expected_sandbox_path = _model_sandbox_path(model_prefix, safe_scope, stored_name)
    if str(parsed.sandbox_path or "") == expected_sandbox_path:
        return parsed

    patched = dict(manifest)
    patched["sandbox_path"] = expected_sandbox_path
    return _manifest_from_dict(patched)


# Build a manifest without any text, archive or table preview, for uploads too large to read
# into memory. Only identity, size, hash, sandbox path and image vision support are filled in.
def _build_lightweight_upload_manifest(
    *,
    file_id: str,
    clean_name: str,
    mime: str,
    size_bytes: int,
    sha256: str,
    sandbox_path: str,
    model_supports_vision: bool,
    tool_server_id: str = "sandbox",
) -> UploadedFileManifest:
    # Only the MIME type is derived; every preview field stays empty.
    clean_mime = guess_upload_mime(clean_name, mime)
    recommended_tools = [tool_server_id] if sandbox_path else []
    return UploadedFileManifest(
        file_id=file_id,
        name=clean_name,
        mime=clean_mime,
        size_bytes=size_bytes,
        sha256=sha256,
        sandbox_path=sandbox_path or None,
        text_available=False,
        text_preview=None,
        text_total_chars=None,
        text_truncated=False,
        vision_available=bool(clean_mime.startswith("image/") and model_supports_vision),
        archive_tree=None,
        table_preview=None,
        recommended_tools=recommended_tools,
    )


# Persist one Django uploaded file and return its private manifest plus public payload.
# The file is stored in a folder named after its content hash, so a repeated upload of the same
# file reuses the existing copy (and its manifest) instead of storing it again.
def save_upload_to_sandbox(
    uploaded_file: Any,
    *,
    scope: str | None = None,
    model_supports_vision: bool = False,
    tool_server_ids: list[str] | None = None,
) -> tuple[UploadedFileManifest, dict[str, Any]]:
    # Reject oversized files by their declared size before reading anything.
    clean_name = normalize_upload_name(getattr(uploaded_file, "name", "") or "uploaded-file")
    declared_size_bytes = int(getattr(uploaded_file, "size", 0) or 0)
    if declared_size_bytes > MAX_UPLOAD_BYTES:
        raise ValueError(f"File is too large (max {_format_upload_size(MAX_UPLOAD_BYTES)})")

    # Stream the upload to a temp file and derive the destination folder from its hash.
    storage_target = resolve_upload_storage_target(tool_server_ids)
    mime = str(getattr(uploaded_file, "content_type", "") or "")
    temp_path, size_bytes, file_sha256, file_bytes = _stream_upload_to_temp(
        uploaded_file,
        incoming_root=storage_target.upload_root,
    )
    safe_scope = _safe_scope(file_sha256)
    target_dir = storage_target.upload_root / safe_scope
    target_dir.mkdir(parents=True, exist_ok=True)

    # Look for an existing copy: by content when the bytes are in memory, by manifest otherwise.
    try:
        existing_upload = (
            _find_existing_upload(
                target_dir,
                model_prefix=storage_target.model_prefix,
                safe_scope=safe_scope,
                clean_name=clean_name,
                file_bytes=file_bytes,
            )
            if file_bytes is not None
            else _find_existing_upload_by_hash(
                target_dir,
                sha256=file_sha256,
                size_bytes=size_bytes,
                clean_name=clean_name,
            )
        )

        # Reuse the existing copy: drop the fresh temp file and return its manifest, rebuilding the
        # manifest when it is missing or invalid.
        if existing_upload:
            existing_path, existing_manifest = existing_upload
            temp_path.unlink(missing_ok=True)
            if existing_manifest:
                parsed_manifest = _normalize_existing_manifest_for_path(
                    existing_manifest,
                    model_prefix=storage_target.model_prefix,
                    safe_scope=safe_scope,
                    stored_name=existing_path.name,
                )
                if parsed_manifest is not None:
                    _write_manifest(parsed_manifest)
                    return parsed_manifest, public_upload_payload(parsed_manifest)

            # Existing binary matched but sidecar is missing/corrupted. Rebuild one
            # manifest bound to the existing file path, then reuse it.
            rebuilt_file_id = uuid.uuid4().hex
            sandbox_path = _model_sandbox_path(storage_target.model_prefix, safe_scope, existing_path.name)
            rebuilt_manifest = (
                build_uploaded_file_manifest(
                    file_bytes,
                    name=clean_name,
                    mime=mime,
                    sandbox_path=sandbox_path,
                    model_supports_vision=model_supports_vision,
                    file_id=rebuilt_file_id,
                    tool_server_id=storage_target.server_id,
                )
                if file_bytes is not None
                else _build_lightweight_upload_manifest(
                    file_id=rebuilt_file_id,
                    clean_name=clean_name,
                    mime=mime,
                    size_bytes=size_bytes,
                    sha256=file_sha256,
                    sandbox_path=sandbox_path,
                    model_supports_vision=model_supports_vision,
                    tool_server_id=storage_target.server_id,
                )
            )
            _write_manifest(rebuilt_manifest)
            return rebuilt_manifest, public_upload_payload(rebuilt_manifest)

        # New file: move the temp file into place and write a full manifest (or a lightweight one
        # for large files).
        file_id = uuid.uuid4().hex
        stored_name = _stored_file_name(file_id, clean_name)
        target_path = target_dir / stored_name
        temp_path.replace(target_path)
        sandbox_path = _model_sandbox_path(storage_target.model_prefix, safe_scope, stored_name)

        manifest = (
            build_uploaded_file_manifest(
                file_bytes,
                name=clean_name,
                mime=mime,
                sandbox_path=sandbox_path,
                model_supports_vision=model_supports_vision,
                file_id=file_id,
                tool_server_id=storage_target.server_id,
            )
            if file_bytes is not None
            else _build_lightweight_upload_manifest(
                file_id=file_id,
                clean_name=clean_name,
                mime=mime,
                size_bytes=size_bytes,
                sha256=file_sha256,
                sandbox_path=sandbox_path,
                model_supports_vision=model_supports_vision,
                tool_server_id=storage_target.server_id,
            )
        )
        _write_manifest(manifest)
        return manifest, public_upload_payload(manifest)
    except Exception:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            # The temp file is already gone or locked; the original error is re-raised below.
            pass
        raise


# Load a private manifest by file id from external storage or legacy sidecars.
def load_upload_manifest(file_id: str) -> dict[str, Any] | None:
    normalized_id = re.sub(r"[^a-fA-F0-9-]+", "", str(file_id or ""))
    if not normalized_id:
        return None

    for manifest_path in USER_FILE_MANIFEST_ROOT.glob(f"**/{normalized_id}{UPLOAD_MANIFEST_SUFFIX}"):
        manifest = _load_manifest_from_sidecar(manifest_path)
        if manifest and str(manifest.get("file_id") or "") == normalized_id:
            return manifest

    # Legacy fallback for uploads created before manifests were moved out of
    # the sandbox file tree.
    for upload_root in (USER_UPLOAD_ROOT,):
        for sidecar in upload_root.glob(f"**/*{UPLOAD_MANIFEST_SUFFIX}"):
            manifest = _load_manifest_from_sidecar(sidecar)
            if manifest and str(manifest.get("file_id") or "") == normalized_id:
                return manifest
    return None
