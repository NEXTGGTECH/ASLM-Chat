# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json

import pytest

from Services import upload_storage
from Services.upload_storage import UploadStorageTarget, save_upload_to_sandbox
from Services.upload_storage import (
    display_kind_for_upload,
    load_upload_manifest,
    model_upload_payload,
    normalize_tool_server_ids,
    public_upload_payload,
    resolve_uploaded_file_host_path,
)


pytestmark = [pytest.mark.unit]


# Minimal stand-in for a Django uploaded file.
class _FakeUpload:
    # Build one upload with a declared name and content type.
    def __init__(self, name: str, content: bytes, content_type: str = ""):
        self.name = name
        self.content = content
        self.size = len(content)
        self.content_type = content_type

    # Return the whole payload (Django files also expose chunks()).
    def read(self) -> bytes:
        return self.content


# Redirect the sandbox, upload and manifest roots into a temporary directory; return the upload root
# and the external manifest root.
@pytest.fixture
def isolated_storage(tmp_path, monkeypatch):
    sandbox_root = tmp_path / "_sandbox"
    upload_root = sandbox_root / "User"
    manifest_root = tmp_path / "user_files"
    monkeypatch.setattr(upload_storage, "SANDBOX_ROOT", sandbox_root)
    monkeypatch.setattr(upload_storage, "USER_UPLOAD_ROOT", upload_root)
    monkeypatch.setattr(upload_storage, "USER_FILE_MANIFEST_ROOT", manifest_root)
    return upload_root, manifest_root


# Request normalization and routing

# Verify that tool server ids are trimmed, lowercased, de-duplicated and stripped of empty values.
def test_normalize_tool_server_ids_dedupes_and_lowercases():
    assert normalize_tool_server_ids(["Sandbox", "sandbox", " other ", "", None]) == ["sandbox", "other"]
    assert normalize_tool_server_ids(None) == []


# Verify that uploads for the sandbox tool server are routed into the sandbox User tree with the
# matching model path prefix.
def test_resolve_upload_storage_target_routes_to_sandbox(isolated_storage):
    target = upload_storage.resolve_upload_storage_target(["sandbox"])

    assert isinstance(target, UploadStorageTarget)
    assert target.server_id == "sandbox"
    assert target.model_prefix == "/workspace/_sandbox/User"


# Stored names and scopes

# Verify that a run of characters that are unsafe in file names collapses to one underscore in the
# stored name, which is prefixed with the file id.
def test_stored_file_name_sanitizes_hostile_names():
    name = upload_storage._stored_file_name("abc123", 'weird<>:"|?*name.txt')

    assert name == "abc123__weird_name.txt"


# Verify that scopes are normalized to path-safe segments and empty or missing scopes fall back to
# "pending".
def test_safe_scope_normalizes_and_falls_back_to_pending():
    assert upload_storage._safe_scope("chat/1") == "chat_1"
    assert upload_storage._safe_scope("") == "pending"
    assert upload_storage._safe_scope(None) == "pending"


# Saving uploads

# Verify that saving an upload stores one file under the sandbox root, writes the manifest under the
# external root by content hash, and returns the public payload.
def test_save_upload_persists_file_and_manifest(isolated_storage):
    upload_root, manifest_root = isolated_storage
    upload = _FakeUpload("notes.txt", b"hello upload", "text/plain")

    # Act: save the upload.
    manifest, payload = save_upload_to_sandbox(upload)

    # Assert: the file landed under the sandbox User tree and the manifest describes it.
    stored_files = [path for path in upload_root.rglob("*") if path.is_file()]
    assert len(stored_files) == 1
    assert manifest.name == "notes.txt"
    assert manifest.sandbox_path.startswith("/workspace/_sandbox/User/")
    assert manifest.sandbox_path.endswith("__notes.txt")
    assert manifest.recommended_tools == ["sandbox"]

    # The manifest is stored per content hash under the external root.
    assert (manifest_root / manifest.sha256).is_dir()
    reloaded = load_upload_manifest(manifest.file_id)
    assert reloaded is not None
    assert reloaded["sha256"] == manifest.sha256

    # Assert: the returned public payload identifies the file and links its content.
    assert payload["file_id"] == manifest.file_id
    assert payload["content_url"] == f"/api/uploads/{manifest.file_id}/content/"
    assert payload["display_kind"] == "text"


# Verify that saving the same content twice yields the same file id and a single stored file.
def test_save_upload_is_deduplicated_by_content(isolated_storage):
    upload_root, _manifest_root = isolated_storage
    first = save_upload_to_sandbox(_FakeUpload("a.txt", b"same bytes", "text/plain"))
    second = save_upload_to_sandbox(_FakeUpload("a.txt", b"same bytes", "text/plain"))

    assert first[0].file_id == second[0].file_id
    stored_files = [path for path in upload_root.rglob("*") if path.is_file()]
    assert len(stored_files) == 1


# Verify that a path-traversal upload name is reduced to its safe basename in the returned payload.
def test_save_upload_sanitizes_hostile_names(isolated_storage):
    _manifest, payload = save_upload_to_sandbox(
        _FakeUpload("../../etc/passwd.txt", b"x", "text/plain")
    )

    assert payload["name"] == "passwd.txt"
    assert ".." not in payload["name"]


# Verify that an upload whose declared size exceeds the limit is rejected with a "too large" error.
def test_save_upload_rejects_oversized_declared_size(monkeypatch):
    upload = _FakeUpload("big.bin", b"x")
    upload.size = upload_storage.MAX_UPLOAD_BYTES + 1

    with pytest.raises(ValueError, match="too large"):
        save_upload_to_sandbox(upload)


# Verify that the chunked streaming path enforces the size limit while reading even when the declared
# size is zero.
def test_save_upload_enforces_size_limit_while_streaming(isolated_storage, monkeypatch):
    # Upload stand-in that only offers chunks() and declares no size.
    class _ChunkedUpload:
        name = "stream.bin"
        size = 0
        content_type = "application/octet-stream"

        # Yield two chunks that together exceed the configured limit.
        def chunks(self):
            yield b"x" * 8
            yield b"y" * 8

    monkeypatch.setattr(upload_storage, "MAX_UPLOAD_BYTES", 10)

    with pytest.raises(ValueError, match="too large"):
        save_upload_to_sandbox(_ChunkedUpload())


# Host path resolution

# Verify that a model sandbox path maps back to the stored host file inside the upload root.
def test_resolve_host_path_maps_model_path_back_to_disk(isolated_storage):
    upload_root, _manifest_root = isolated_storage
    manifest, _payload = save_upload_to_sandbox(_FakeUpload("findme.txt", b"content", "text/plain"))

    host_path = resolve_uploaded_file_host_path({"sandbox_path": manifest.sandbox_path})

    assert host_path.is_file()
    assert host_path.read_bytes() == b"content"
    assert host_path.parent.parent == upload_root.resolve()


# Verify that empty paths, paths outside the sandbox User tree and paths that escape it with ".."
# raise an error instead of resolving.
def test_resolve_host_path_rejects_escapes_and_missing_files():
    with pytest.raises(FileNotFoundError):
        resolve_uploaded_file_host_path({"sandbox_path": ""})
    with pytest.raises(FileNotFoundError):
        resolve_uploaded_file_host_path({"sandbox_path": "/workspace/Other/file.txt"})
    with pytest.raises((FileNotFoundError, ValueError)):
        resolve_uploaded_file_host_path({"sandbox_path": "/workspace/_sandbox/User/../secret.txt"})


# Legacy sidecar manifests

# Verify that a manifest stored next to the file in the legacy sidecar layout still loads by file id
# and unknown or empty ids return None.
def test_load_upload_manifest_falls_back_to_legacy_sidecars(isolated_storage):
    upload_root, _manifest_root = isolated_storage

    sidecar = upload_root / "legacy"
    sidecar.mkdir(parents=True)
    document = {"file_id": "abc123", "sha256": "abc", "name": "old.txt"}
    sidecar.joinpath("abc123__old.txt.manifest.json").write_text(
        json.dumps(document), encoding="utf-8"
    )

    assert load_upload_manifest("abc123") == document
    assert load_upload_manifest("missing-id") is None
    assert load_upload_manifest("") is None


# Display kinds and payloads

# Verify that the display kind and label of an upload follow its MIME type and file extension.
@pytest.mark.parametrize(
    ("name", "mime", "kind", "label"),
    [
        ("photo.PNG", "image/png", "image", "Image"),
        ("song.mp3", "", "audio", "Audio"),
        ("clip.mkv", "", "video", "Video"),
        ("bundle.zip", "", "archive", "ZIP archive"),
        ("docs.rar", "", "archive", "Archive"),
        ("paper.pdf", "", "document", "PDF document"),
        ("report.docx", "", "document", "Word document"),
        ("table.xlsx", "", "table", "Excel spreadsheet"),
        ("rows.csv", "", "table", "CSV table"),
        ("deck.pptx", "", "presentation", "PowerPoint presentation"),
        ("main.py", "", "code", "Code file"),
        ("notes.md", "", "text", "Text file"),
        ("mystery.xyz", "", "file", "File"),
    ],
)
def test_display_kind_for_upload(name, mime, kind, label):
    assert display_kind_for_upload(name, mime) == (kind, label)


# Verify that the public payload is identical whether built from a manifest object or its dictionary.
def test_public_payload_accepts_manifest_or_dict():
    from Services.file_manifests import build_uploaded_file_manifest

    manifest = build_uploaded_file_manifest(b"content", name="a.txt", mime="text/plain")
    from_manifest = public_upload_payload(manifest)
    from_dict = public_upload_payload(manifest.to_dict())

    assert from_manifest == from_dict
    assert from_dict["display_kind"] == "text"


# Verify that the model payload hides the sandbox path and the sandbox tool recommendation when the
# sandbox is disabled and passes the manifest through unchanged when it is enabled.
def test_model_payload_hides_sandbox_state_when_disabled():
    manifest = {
        "sandbox_path": "/workspace/_sandbox/User/a.txt",
        "recommended_tools": ["sandbox", "archive"],
    }

    disabled = model_upload_payload(manifest, sandbox_enabled=False)
    enabled = model_upload_payload(manifest, sandbox_enabled=True)

    assert disabled["sandbox_path"] is None
    assert disabled["recommended_tools"] == ["archive"]
    assert enabled == manifest
