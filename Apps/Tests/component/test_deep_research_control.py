# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
from zipfile import ZipFile

import pytest
from django.test import SimpleTestCase, TestCase

from Tools.deep_research import control, export, orchestrator
from Apps.UI import views


pytestmark = [pytest.mark.component]


class DeepResearchDirectStreamTests(SimpleTestCase):
    def test_artifact_and_planning_card_have_no_hover_highlight(self) -> None:
        source = Path("Apps/UI/static/css/main/deep-research.css").read_text(encoding="utf-8")
        self.assertNotIn(".msg-deep-research-card:hover", source)
        self.assertNotIn(".deep-research-report-preview:hover", source)
        self.assertIn(".deep-research-card-activity:hover", source)

    def test_model_stage_forwards_attached_media(self) -> None:
        events = mock.Mock()
        with mock.patch.object(
            orchestrator.llm_api,
            "generate",
            return_value=[{"message": {"content": "OK"}}],
        ) as generate:
            result = orchestrator._run_model_stage(
                engine="openai",
                model="test-model",
                system_prompt="System",
                user_prompt="Research attachments",
                generation_options={},
                session_id="deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                events=events,
                phase="planning",
                iteration=0,
                plan_version=0,
                request_message={
                    "role": "user",
                    "content": "Source context",
                    "media": [{"kind": "audio", "mime_type": "audio/mpeg", "data": "ZmFrZQ=="}],
                },
            )

        outbound_user = generate.call_args.args[2][1]
        self.assertEqual(result, "OK")
        self.assertEqual(outbound_user["content"], "Research attachments")
        self.assertEqual(outbound_user["media"][0]["kind"], "audio")

    def test_attachment_only_request_builds_topic_and_multimodal_context(self) -> None:
        pasted_text = base64.b64encode(b"Compare these attached materials").decode("ascii")
        image_data = base64.b64encode(b"fake-image").decode("ascii")

        topic, request_message = views._build_deep_research_request_message(
            "",
            [
                {
                    "kind": "file",
                    "name": "Pasted text",
                    "mime_type": "text/plain",
                    "size_bytes": 32,
                    "data": pasted_text,
                },
                {
                    "kind": "image",
                    "name": "screenshot.png",
                    "mime_type": "image/png",
                    "size_bytes": 10,
                    "data": image_data,
                },
            ],
            [],
            engine="openai",
            model_info_payload={"supports_vision": True},
        )

        self.assertIn("Pasted text", topic)
        self.assertIn("screenshot.png", topic)
        self.assertIn("Compare these attached materials", request_message["content"])
        self.assertEqual(request_message["images"], [image_data])

    def test_report_preview_is_not_a_button_containing_mermaid_controls(self) -> None:
        source = Path("Apps/UI/static/js/ui/deep-research-ui.js").read_text(encoding="utf-8")
        self.assertIn('<div class="deep-research-report-preview"', source)
        self.assertNotIn('<button type="button" class="deep-research-report-preview"', source)

    def test_direct_stream_persists_report_context_and_system_ready_text(self) -> None:
        assistant = SimpleNamespace(content="", llm_transcript=[], save=mock.Mock())
        chat = SimpleNamespace(id="11111111-1111-1111-1111-111111111111", pk="chat-pk")
        prepared = {
            "arguments": {
                "topic": "Test direct research",
                "session_id": "deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "max_rounds": 6,
            },
            "tool_ui": {"kind": "deep_research", "status": "planning"},
        }
        result = {
            "model_context": "Report [source](https://example.com)",
            "report": "Report [source](https://example.com)",
            "sources": [{"id": "c1", "url": "https://example.com"}],
            "ui": {"kind": "deep_research", "status": "completed"},
        }

        with (
            mock.patch.object(views.deep_research_service, "run_research", return_value=result) as run_research,
            mock.patch.object(views.Chat.objects, "filter") as filter_chat,
        ):
            streamed = "".join(
                views._stream_deep_research_response(
                    chat=chat,
                    assistant_message_record=assistant,
                    prepared_research=prepared,
                    engine="openai",
                    model_name="test-model",
                    request_message={"role": "user", "content": "Attached context", "images": ["aW1hZ2U="]},
                )
            )

        self.assertIn("<tool_call>", streamed)
        self.assertIn("<tool_result>", streamed)
        self.assertTrue(streamed.endswith("Your research is ready"))
        self.assertEqual(assistant.content, "Your research is ready")
        self.assertEqual(assistant.llm_transcript[1]["content"], result["model_context"])
        self.assertEqual(assistant.llm_transcript[2]["content"], "Your research is ready")
        self.assertEqual(assistant.save.call_count, 2)
        self.assertEqual(filter_chat.call_count, 2)
        filter_chat.assert_called_with(pk="chat-pk")
        run_context = run_research.call_args.args[1]
        self.assertEqual(run_context["request_message"]["content"], "Attached context")


class DeepResearchAttachmentChatApiTests(TestCase):
    def test_attachment_only_chat_starts_research_with_file_context(self) -> None:
        prepared = {
            "arguments": {
                "topic": "Research the user-provided attachments: screenshot.png, Pasted text.",
                "session_id": "deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                "max_rounds": 6,
            },
            "tool_ui": {"kind": "deep_research", "status": "planning"},
        }
        result = {
            "model_context": "Report",
            "report": "Report",
            "sources": [],
            "ui": {"kind": "deep_research", "status": "completed"},
        }
        image_data = base64.b64encode(b"fake-image").decode("ascii")
        pasted_data = base64.b64encode(b"Compare the attached materials").decode("ascii")

        with (
            mock.patch.object(views, "_resolve_request_engine_or_response", return_value=("openai", None)),
            mock.patch.object(
                views,
                "_build_model_info_payload",
                return_value={
                    "supports_vision": True,
                    "supports_audio_input": False,
                    "supports_video_input": False,
                    "supports_tool_calling": False,
                    "defaults": {},
                    "runtime_limits": {},
                },
            ),
            mock.patch.object(views, "_sync_runtime_model_metadata"),
            mock.patch.object(views.deep_research_service, "prepare_research", return_value=prepared) as prepare,
            mock.patch.object(views.deep_research_service, "run_research", return_value=result) as run,
        ):
            response = self.client.post(
                "/api/chat/",
                data=json.dumps(
                    {
                        "message": "",
                        "model": "test-model",
                        "deep_research": True,
                        "attachments": [
                            {
                                "kind": "image",
                                "name": "screenshot.png",
                                "mime_type": "image/png",
                                "data": image_data,
                            },
                            {
                                "kind": "file",
                                "name": "Pasted text",
                                "mime_type": "text/plain",
                                "data": pasted_data,
                            },
                        ],
                    }
                ),
                content_type="application/json",
            )
            streamed = b"".join(response.streaming_content).decode("utf-8")

        self.assertEqual(response.status_code, 200)
        self.assertIn("Your research is ready", streamed)
        topic = prepare.call_args.args[0]
        self.assertIn("screenshot.png", topic)
        self.assertIn("Pasted text", topic)
        request_message = run.call_args.args[1]["request_message"]
        self.assertIn("Compare the attached materials", request_message["content"])
        self.assertEqual(request_message["images"], [image_data])


class DeepResearchExportApiTests(TestCase):
    RICH_REPORT = """# Architecture report

## Decision

Use **modular services** with the latency model $T = T_{network} + T_{compute}$.

$$R = \\frac{requests}{second}$$

| Component | Decision | Evidence |
| --- | --- | --- |
| Gateway | Stateless | [Primary source](https://example.com/source) |

```mermaid
flowchart LR
    Client --> Gateway
    Gateway --> Service
```
"""

    def setUp(self) -> None:
        self.control_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.control_directory.cleanup)
        root = Path(self.control_directory.name)
        self.enterContext(mock.patch.object(control, "STATE_ROOT", root / "state"))
        self.enterContext(mock.patch.object(control, "COMMAND_ROOT", root / "commands"))
        self.session_id = control.new_session_id()
        control.create_session(self.session_id, topic="Export report")
        control.update_state(
            self.session_id,
            status="completed",
            report="# Result\nClaim [c0001-1](https://example.com/source)",
            model_context="# Result\nClaim [c0001-1](https://example.com/source)",
        )

    def export(self, export_format: str):
        return self.client.post(
            "/api/deep-research/export/",
            data=json.dumps({"session_id": self.session_id, "format": export_format}),
            content_type="application/json",
        )

    def test_markdown_export_keeps_resolved_citation_link(self) -> None:
        response = self.export("md")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"https://example.com/source", response.content)
        self.assertIn(".md", response["Content-Disposition"])

    def test_markdown_export_uses_transcript_fallback_after_snapshot_retention(self) -> None:
        control._state_path(self.session_id).unlink()

        response = self.client.post(
            "/api/deep-research/export/",
            data=json.dumps(
                {
                    "session_id": self.session_id,
                    "format": "md",
                    "title": "Recovered report",
                    "report": "# Recovered\nClaim [source](https://example.com/recovered)",
                }
            ),
            content_type="application/json",
        )

        self.assertEqual(response.status_code, 200)
        self.assertIn(b"https://example.com/recovered", response.content)
        self.assertIn("Recovered-report.md", response["Content-Disposition"])

    def test_docx_export_contains_clickable_external_relationship(self) -> None:
        response = self.export("docx")
        self.assertEqual(response.status_code, 200)
        with ZipFile(BytesIO(response.content)) as archive:
            relationships = archive.read("word/_rels/document.xml.rels").decode("utf-8")
        self.assertIn("https://example.com/source", relationships)
        self.assertIn('TargetMode="External"', relationships)

    def test_pdf_export_returns_pdf_document(self) -> None:
        response = self.export("pdf")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b"%PDF-"))

    def test_docx_renders_markdown_math_and_mermaid(self) -> None:
        control.update_state(
            self.session_id,
            report=self.RICH_REPORT,
            model_context=self.RICH_REPORT,
        )

        # The diagram renderer shells out to a local browser binary; replace
        # only that boundary so the DOCX assembly pipeline stays exercised.
        def fake_render(source, output_path, workspace) -> None:
            from PIL import Image, ImageDraw

            image = Image.new("RGB", (1600, 400), "white")
            ImageDraw.Draw(image).rectangle([40, 80, 700, 320], outline="black", width=6)
            image.save(output_path, "PNG")

        with mock.patch.object(export, "_render_mermaid_png", side_effect=fake_render):
            response = self.export("docx")

        self.assertEqual(response.status_code, 200)
        with ZipFile(BytesIO(response.content)) as archive:
            document_xml = archive.read("word/document.xml").decode("utf-8")
            styles_xml = archive.read("word/styles.xml")
            media = [name for name in archive.namelist() if name.startswith("word/media/")]
            from PIL import Image
            diagram_bytes = archive.read(media[0]) if media else b""
            diagram_size = Image.open(BytesIO(diagram_bytes)).size if diagram_bytes else (0, 0)
        self.assertIn("<w:tbl", document_xml)
        self.assertIn("<m:oMath", document_xml)
        self.assertNotIn("**modular services**", document_xml)
        self.assertNotIn("0563C1", document_xml)
        from xml.etree import ElementTree
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        styles_root = ElementTree.fromstring(styles_xml)
        title_style = styles_root.find(".//w:style[@w:styleId='Title']", namespace)
        self.assertIsNotNone(title_style)
        self.assertIsNone(title_style.find("./w:pPr/w:pBdr", namespace))
        self.assertEqual(title_style.find("./w:rPr/w:color", namespace).get(f"{{{namespace['w']}}}val"), "000000")
        self.assertEqual(len(media), 1)
        self.assertEqual(diagram_size, (1600, 400))
        # 1600 px wide is clamped to the 6.35 in display width (5806440 EMU).
        self.assertIn('cx="5806440"', document_xml)

    def test_pdf_renders_report_pages_and_keeps_links(self) -> None:
        control.update_state(
            self.session_id,
            report=self.RICH_REPORT,
            model_context=self.RICH_REPORT,
        )

        response = self.export("pdf")

        self.assertEqual(response.status_code, 200)
        import fitz
        document = fitz.open(stream=response.content, filetype="pdf")
        try:
            rendered_text = "\n".join(page.get_text() for page in document)
            links = [link for page in document for link in page.get_links()]
            sample = document[0].get_pixmap(colorspace=fitz.csRGB, alpha=False)
        finally:
            document.close()
        self.assertIn("Component", rendered_text)
        self.assertIn("Gateway", rendered_text)
        self.assertNotIn("| --- |", rendered_text)
        self.assertTrue(any(link.get("uri") == "https://example.com/source" for link in links))
        from PIL import Image, ImageChops
        rendered_page = Image.frombytes("RGB", (sample.width, sample.height), sample.samples)
        red, green, blue = rendered_page.split()
        self.assertIsNone(ImageChops.difference(red, green).getbbox())
        self.assertIsNone(ImageChops.difference(red, blue).getbbox())


class DeepResearchControlApiTests(SimpleTestCase):
    def setUp(self) -> None:
        self.control_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.control_directory.cleanup)
        root = Path(self.control_directory.name)
        self.enterContext(mock.patch.object(control, "STATE_ROOT", root / "state"))
        self.enterContext(mock.patch.object(control, "COMMAND_ROOT", root / "commands"))
        self.session_id = control.new_session_id()
        control.create_session(
            self.session_id,
            topic="Test topic",
            status="awaiting_approval",
            extra={"plan": "- [ ] Verify the claim", "plan_version": 1},
        )

    def post(self, payload: dict):
        return self.client.post(
            "/api/deep-research/control/",
            data=json.dumps(payload),
            content_type="application/json",
        )

    def test_state_can_be_read_and_current_plan_approved(self) -> None:
        response = self.client.get(
            "/api/deep-research/control/",
            {"session_id": self.session_id},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["state"]["plan_version"], 1)

        response = self.post(
            {
                "session_id": self.session_id,
                "action": "approve",
                "expected_plan_version": 1,
            }
        )
        self.assertEqual(response.status_code, 202)
        commands = control.read_commands(self.session_id)
        self.assertEqual(commands[0][1]["action"], "approve")

    def test_stale_revision_is_rejected_with_latest_state(self) -> None:
        response = self.post(
            {
                "session_id": self.session_id,
                "action": "revise",
                "plan": "- [ ] Use the revised evidence goal",
                "expected_plan_version": 0,
            }
        )
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json()["state"]["plan_version"], 1)
        self.assertEqual(control.read_commands(self.session_id), [])

    def test_cancel_is_idempotent_after_terminal_state(self) -> None:
        control.update_state(self.session_id, status="cancelled")
        response = self.post(
            {
                "session_id": self.session_id,
                "action": "cancel",
                "expected_plan_version": 1,
            }
        )
        self.assertEqual(response.status_code, 202)
        self.assertTrue(response.json()["terminal"])

    @mock.patch.object(views.tool_registry, "abort_active_research", return_value=1)
    def test_cancel_immediately_interrupts_matching_active_worker(self, abort_research) -> None:
        response = self.post(
            {
                "session_id": self.session_id,
                "action": "cancel",
                "expected_plan_version": 1,
            }
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertTrue(payload["terminal"])
        self.assertEqual(payload["interrupted_active_tools"], 1)
        self.assertEqual(payload["state"]["status"], "cancelled")
        self.assertFalse(payload["state"]["can_stop"])
        abort_research.assert_called_once_with(self.session_id)

    @mock.patch.object(views.tool_registry, "abort_active_research", return_value=0)
    def test_cancel_hides_stop_while_worker_stops_cooperatively(self, abort_research) -> None:
        response = self.post(
            {
                "session_id": self.session_id,
                "action": "cancel",
                "expected_plan_version": 1,
            }
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertFalse(payload["terminal"])
        self.assertEqual(payload["state"]["status"], "stopping")
        self.assertFalse(payload["state"]["can_stop"])
        persisted = control.read_state(self.session_id)
        self.assertEqual(persisted["status"], "stopping")
        self.assertFalse(persisted["can_stop"])
        abort_research.assert_called_once_with(self.session_id)

    def test_invalid_session_id_is_rejected(self) -> None:
        response = self.post({"session_id": "../../outside", "action": "cancel"})
        self.assertEqual(response.status_code, 400)

    def test_revision_is_rejected_after_edit_window_closes(self) -> None:
        control.update_state(self.session_id, can_edit=False)

        response = self.post(
            {
                "session_id": self.session_id,
                "action": "revise",
                "plan": "- [ ] Too late",
                "expected_plan_version": 1,
            }
        )

        self.assertEqual(response.status_code, 409)
        self.assertEqual(control.read_commands(self.session_id), [])

    def test_processed_commands_are_removed_after_acknowledgement(self) -> None:
        control.submit_command(
            self.session_id,
            "approve",
            expected_plan_version=1,
        )
        commands = control.read_commands(self.session_id)
        filename = commands[0][0]
        command_directory = control.COMMAND_ROOT / control.session_key(self.session_id)

        self.assertEqual(
            control.read_commands(self.session_id, processed={filename}),
            [],
        )
        self.assertFalse((command_directory / filename).exists())
        self.assertFalse(command_directory.exists())

    def test_terminal_snapshot_removes_obsolete_command_inbox(self) -> None:
        control.submit_command(
            self.session_id,
            "approve",
            expected_plan_version=1,
        )
        command_directory = control.COMMAND_ROOT / control.session_key(self.session_id)
        self.assertTrue(command_directory.exists())

        control.update_state(self.session_id, status="completed", report="Done")

        self.assertFalse(command_directory.exists())

    def test_stale_worker_checkpoint_cannot_resurrect_a_stopping_run(self) -> None:
        control.update_state(
            self.session_id,
            status="stopping",
            phase="stopping",
            can_stop=False,
            can_edit=False,
            can_approve=False,
        )

        state = control.update_state(
            self.session_id,
            status="researching",
            phase="search",
            can_stop=True,
            can_edit=True,
            can_approve=True,
            source_count=7,
        )

        self.assertEqual(state["status"], "stopping")
        self.assertEqual(state["phase"], "stopping")
        self.assertFalse(state["can_stop"])
        self.assertFalse(state["can_edit"])
        self.assertFalse(state["can_approve"])
        self.assertEqual(state["source_count"], 7)

        terminal = control.update_state(
            self.session_id,
            status="cancelled",
            phase="cancelled",
        )
        self.assertEqual(terminal["status"], "cancelled")
        self.assertEqual(terminal["phase"], "cancelled")


class DeepResearchStorageTests(SimpleTestCase):
    def _patch_storage(self, runtime_root: Path, legacy_root: Path) -> None:
        self.enterContext(mock.patch.object(control, "CONTROL_ROOT", runtime_root))
        self.enterContext(mock.patch.object(control, "STATE_ROOT", runtime_root / "state"))
        self.enterContext(mock.patch.object(control, "COMMAND_ROOT", runtime_root / "commands"))
        self.enterContext(mock.patch.object(control, "LEGACY_CONTROL_ROOT", legacy_root))
        self.enterContext(mock.patch.object(control, "_last_retention_check", 0.0))

    def test_runtime_directory_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"ASLM_DEEP_RESEARCH_RUNTIME_DIR": directory},
                clear=False,
            ):
                self.assertEqual(control._default_control_root(), Path(directory))

    def test_windows_default_is_outside_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ASLM_DEEP_RESEARCH_RUNTIME_DIR", None)
                os.environ["LOCALAPPDATA"] = directory
                with mock.patch.object(control.platform, "system", return_value="Windows"):
                    expected = Path(directory) / "ASLM-Chat" / "runtime" / "deep-research"
                    self.assertEqual(control._default_control_root(), expected)
                    self.assertFalse(expected.is_relative_to(control.PROJECT_ROOT))

    def test_legacy_store_is_atomically_moved_and_payload_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_root = root / "runtime"
            legacy_root = root / "project" / "Data" / "deep_research"
            session_key = "a" * 32
            session_id = f"deep-research:{session_key}"
            state_payload = {
                "schema_version": 1,
                "session_id": session_id,
                "status": "awaiting_approval",
                "updated_at": time.time(),
                "report": "Evidence [caaaaaaaa-7]",
                "citations": [{"handle": "caaaaaaaa-7", "url": "https://example.test"}],
                "citation_next": 8,
            }
            command_payload = {
                "schema_version": 1,
                "command_id": "command-1",
                "session_id": session_id,
                "action": "approve",
            }
            legacy_state = legacy_root / "state" / f"{session_key}.json"
            legacy_command = legacy_root / "commands" / session_key / "command.json"
            legacy_state.parent.mkdir(parents=True)
            legacy_command.parent.mkdir(parents=True)
            legacy_state.write_text(json.dumps(state_payload), encoding="utf-8")
            legacy_command.write_text(json.dumps(command_payload), encoding="utf-8")
            self._patch_storage(runtime_root, legacy_root)

            control._ensure_roots()

            self.assertEqual(control._read_json(runtime_root / "state" / legacy_state.name), state_payload)
            self.assertEqual(
                control._read_json(runtime_root / "commands" / session_key / legacy_command.name),
                command_payload,
            )
            self.assertTrue((runtime_root / control._MIGRATION_MARKER).exists())
            self.assertFalse(legacy_root.exists())

    def test_migration_keeps_newer_runtime_state_and_legacy_only_fields(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_root = root / "runtime"
            legacy_root = root / "legacy"
            session_key = "b" * 32
            legacy_state = legacy_root / "state" / f"{session_key}.json"
            current_state = runtime_root / "state" / f"{session_key}.json"
            legacy_state.parent.mkdir(parents=True)
            current_state.parent.mkdir(parents=True)
            legacy_state.write_text(
                json.dumps(
                    {"status": "completed", "updated_at": time.time() - 10, "citation_next": 12}
                ),
                encoding="utf-8",
            )
            current_state.write_text(
                json.dumps({"status": "completed", "updated_at": time.time(), "report": "new"}),
                encoding="utf-8",
            )
            self._patch_storage(runtime_root, legacy_root)

            control._ensure_roots()

            merged = control._read_json(current_state)
            self.assertGreater(merged["updated_at"], time.time() - 5)
            self.assertEqual(merged["report"], "new")
            self.assertEqual(merged["citation_next"], 12)
            self.assertFalse(legacy_root.exists())

    def test_locked_legacy_artifact_does_not_break_runtime_storage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_root = root / "runtime"
            legacy_root = root / "legacy"
            session_key = "c" * 32
            legacy_state = legacy_root / "state" / f"{session_key}.json"
            legacy_state.parent.mkdir(parents=True)
            legacy_state.write_text(
                json.dumps({"status": "researching", "updated_at": time.time()}),
                encoding="utf-8",
            )
            self._patch_storage(runtime_root, legacy_root)
            original_unlink = Path.unlink

            def fail_for_legacy(path: Path, *args, **kwargs):
                if path == legacy_state:
                    raise PermissionError("locked")
                return original_unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", new=fail_for_legacy):
                control._ensure_roots()

            self.assertTrue(legacy_state.exists())
            self.assertTrue((runtime_root / "state" / legacy_state.name).exists())
            self.assertFalse((runtime_root / control._MIGRATION_MARKER).exists())

    def test_restart_cleanup_removes_persisted_acknowledgements_and_terminal_inboxes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_root = root / "runtime"
            self._patch_storage(runtime_root, root / "missing-legacy")
            active_key = "d" * 32
            terminal_key = "e" * 32
            processed_name = "0001-processed.json"
            pending_name = "0002-pending.json"
            control._atomic_write_json(
                runtime_root / "state" / f"{active_key}.json",
                {
                    "status": "researching",
                    "processed_commands": [processed_name],
                    "updated_at": time.time(),
                },
            )
            control._atomic_write_json(
                runtime_root / "state" / f"{terminal_key}.json",
                {"status": "completed", "updated_at": time.time()},
            )
            active_commands = runtime_root / "commands" / active_key
            terminal_commands = runtime_root / "commands" / terminal_key
            active_commands.mkdir(parents=True)
            terminal_commands.mkdir(parents=True)
            (active_commands / processed_name).write_text("{}", encoding="utf-8")
            (active_commands / pending_name).write_text("{}", encoding="utf-8")
            (terminal_commands / pending_name).write_text("{}", encoding="utf-8")

            control._apply_terminal_retention(force=True)

            self.assertFalse((active_commands / processed_name).exists())
            self.assertTrue((active_commands / pending_name).exists())
            self.assertFalse(terminal_commands.exists())

    def test_terminal_retention_bounds_age_and_count_without_touching_active_runs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime_root = root / "runtime"
            legacy_root = root / "missing-legacy"
            self._patch_storage(runtime_root, legacy_root)
            (runtime_root / "state").mkdir(parents=True)
            (runtime_root / "commands").mkdir(parents=True)
            now = time.time()
            keys = ["1" * 32, "2" * 32, "3" * 32, "4" * 32]
            snapshots = [
                (keys[0], "completed", now - 10),
                (keys[1], "failed", now - 20),
                (keys[2], "cancelled", now - (3 * 86400)),
                (keys[3], "researching", now - (90 * 86400)),
            ]
            for key, status, updated_at in snapshots:
                control._atomic_write_json(
                    runtime_root / "state" / f"{key}.json",
                    {"status": status, "updated_at": updated_at},
                )
                command_dir = runtime_root / "commands" / key
                command_dir.mkdir(parents=True)
                (command_dir / "command.json").write_text("{}", encoding="utf-8")

            with mock.patch.dict(
                os.environ,
                {
                    "ASLM_DEEP_RESEARCH_TERMINAL_RETENTION_DAYS": "1",
                    "ASLM_DEEP_RESEARCH_TERMINAL_RETENTION_MAX": "1",
                },
                clear=False,
            ):
                control._apply_terminal_retention(force=True)

            self.assertTrue((runtime_root / "state" / f"{keys[0]}.json").exists())
            self.assertFalse((runtime_root / "state" / f"{keys[1]}.json").exists())
            self.assertFalse((runtime_root / "state" / f"{keys[2]}.json").exists())
            self.assertTrue((runtime_root / "state" / f"{keys[3]}.json").exists())
            self.assertFalse((runtime_root / "commands" / keys[1]).exists())
            self.assertFalse((runtime_root / "commands" / keys[2]).exists())
            self.assertTrue((runtime_root / "commands" / keys[3]).exists())


class DeepResearchTranscriptReloadTests(SimpleTestCase):
    def test_pending_tool_call_restores_pollable_research_card(self) -> None:
        session_id = "deep-research:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        message = SimpleNamespace(
            llm_transcript=[
                {
                    "role": "assistant",
                    "content": "",
                    "thinking": "I will start research.",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "deep_research__deep_research",
                                "arguments": {
                                    "topic": "Reload-safe research",
                                    "max_rounds": 2,
                                    "session_id": session_id,
                                },
                            }
                        }
                    ],
                }
            ]
        )

        segments = views._build_activity_segments(message)

        research = next(segment for segment in segments if segment["type"] == "tool")
        self.assertEqual(research["toolUi"]["kind"], "deep_research")
        self.assertEqual(research["toolUi"]["session_id"], session_id)
        self.assertEqual(research["toolUi"]["query_budget"], 4)
        self.assertIsNone(research["result"])

    def test_completed_tool_call_and_result_merge_without_duplicate_card(self) -> None:
        message = SimpleNamespace(
            llm_transcript=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "deep_research__deep_research",
                                "arguments": {"topic": "Completed research", "max_rounds": 2},
                            }
                        }
                    ],
                },
                {
                    "role": "tool",
                    "alias": "deep_research__deep_research__0",
                    "content": "Final report",
                    "tool_ui": {"kind": "deep_research", "status": "completed"},
                },
            ]
        )

        tools = [
            segment
            for segment in views._build_activity_segments(message)
            if segment["type"] == "tool"
        ]

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["result"], "Final report")
        self.assertEqual(tools[0]["toolUi"]["status"], "completed")
