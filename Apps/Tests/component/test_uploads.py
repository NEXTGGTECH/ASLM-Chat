# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import io
import subprocess
import tempfile
import zipfile
import pytest
from pathlib import Path
from unittest.mock import patch
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import SimpleTestCase, TestCase
from django.urls import reverse
from Apps.Data.models import Chat, Message, MessageAttachment, MessageAttachmentKind
from Apps.UI import upload_storage
from Apps.UI.file_manifests import TEXT_PREVIEW_CHAR_LIMIT, build_uploaded_file_manifest, normalize_upload_name
from Apps.UI.upload_storage import display_kind_for_upload
from Apps.UI.views import _build_uploaded_file_context_entry, _build_uploaded_file_prompt_block, _apply_uploaded_file_manifests_to_llm_entry, _extract_attachment_text, _extract_uploaded_file_ids_from_message, _load_model_upload_manifests, _normalize_request_attachments, _normalize_uploaded_file_ids, _reveal_file_in_file_manager, _serialize_attachment_record, _selected_tools_include_sandbox


pytestmark = [pytest.mark.component]


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

# Cover the public upload contract without exposing model-only manifests.
class UploadFilesApiTests(TestCase):
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
