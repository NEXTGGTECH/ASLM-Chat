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
from Apps.API import views as api_views


pytestmark = [pytest.mark.component]


# Cover the first-class Deep Research stream: card styling, attachment context, media forwarding and the
# persistence of the report in the assistant message.
class DeepResearchDirectStreamTests(SimpleTestCase):
    # Verify that the stylesheet defines no hover highlight for the research card and the report preview, while
    # the card activity row keeps its own hover rule.
    def test_artifact_and_planning_card_have_no_hover_highlight(self) -> None:
        source = Path("Apps/UI/static/css/main/deep-research.css").read_text(encoding="utf-8")
        self.assertNotIn(".msg-deep-research-card:hover", source)
        self.assertNotIn(".deep-research-report-preview:hover", source)
        self.assertIn(".deep-research-card-activity:hover", source)

    # Verify that a model stage sends the media of the request message along with the stage prompt: the
    # outbound user message keeps the stage prompt as content and carries the attached audio.
    def test_model_stage_forwards_attached_media(self) -> None:
        # Act: run a planning stage whose request message carries one audio attachment, with the model call mocked.
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

        # Assert: the user message sent to the model has the stage prompt and the audio.
        outbound_user = generate.call_args.args[2][1]
        self.assertEqual(result, "OK")
        self.assertEqual(outbound_user["content"], "Research attachments")
        self.assertEqual(outbound_user["media"][0]["kind"], "audio")

    # Verify that a request with attachments but no text gets a topic naming the attachments, a request
    # message containing the pasted text, and the image passed on for a vision model.
    def test_attachment_only_request_builds_topic_and_multimodal_context(self) -> None:
        # Arrange: a pasted text file and an image, both as base64 attachment data.
        pasted_text = base64.b64encode(b"Compare these attached materials").decode("ascii")
        image_data = base64.b64encode(b"fake-image").decode("ascii")

        # Act: build the research request from an empty message and the two attachments.
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

        # Assert: the topic names both attachments, the text goes into the content and the image is passed on.
        self.assertIn("Pasted text", topic)
        self.assertIn("screenshot.png", topic)
        self.assertIn("Compare these attached materials", request_message["content"])
        self.assertEqual(request_message["images"], [image_data])

    # Verify that the report preview is rendered as a div, not a button, in the front-end source, so the
    # Mermaid controls it contains are not nested inside a button.
    def test_report_preview_is_not_a_button_containing_mermaid_controls(self) -> None:
        source = Path("Apps/UI/static/js/ui/deep-research-ui.js").read_text(encoding="utf-8")
        self.assertIn('<div class="deep-research-report-preview"', source)
        self.assertNotIn('<button type="button" class="deep-research-report-preview"', source)

    # Verify that the direct stream emits the tool call and tool result markers followed by the ready text, and
    # persists the report as model context and the ready text in the assistant message.
    def test_direct_stream_persists_report_context_and_system_ready_text(self) -> None:
        # Arrange: stand-ins for the chat and the assistant message, a prepared research and its result.
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

        # Act: consume the whole stream with the research run and the chat update mocked.
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

        # Assert: the markers and ready text are streamed, the message is saved twice (pending, then final),
        # the chat is touched twice and the run received the request message.
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


# Cover the chat endpoint when a Deep Research request carries attachments but no message text.
class DeepResearchAttachmentChatApiTests(TestCase):
    # Verify that a Deep Research chat request with only an image and a pasted text attachment prepares the
    # research with a topic naming both attachments and runs it with the pasted text and image as context.
    def test_attachment_only_chat_starts_research_with_file_context(self) -> None:
        # Arrange: a prepared research and result, and the two attachments.
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

        # Act: post the chat request with the engine, model metadata and research service mocked, and read the
        # whole stream.
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

        # Assert: the research was prepared with both attachment names in the topic and run with the
        # attachment context.
        self.assertEqual(response.status_code, 200)
        self.assertIn("Your research is ready", streamed)
        topic = prepare.call_args.args[0]
        self.assertIn("screenshot.png", topic)
        self.assertIn("Pasted text", topic)
        request_message = run.call_args.args[1]["request_message"]
        self.assertIn("Compare the attached materials", request_message["content"])
        self.assertEqual(request_message["images"], [image_data])


# Cover the export endpoint of a completed research report in Markdown, DOCX and PDF formats.
class DeepResearchExportApiTests(TestCase):
    # Markdown report with a heading, bold text, inline and display math, a table with a link, and a Mermaid
    # diagram, used to check the rich rendering of the DOCX and PDF exports.
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

    # Redirect the control state and command folders into a temporary directory and create a completed session
    # with a short report.
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

    # Post an export request for the test session in the given format and return the response.
    def export(self, export_format: str):
        return self.client.post(
            "/api/deep-research/export/",
            data=json.dumps({"session_id": self.session_id, "format": export_format}),
            content_type="application/json",
        )

    # Verify that the Markdown export contains the resolved citation URL and is served as an .md download.
    def test_markdown_export_keeps_resolved_citation_link(self) -> None:
        response = self.export("md")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"https://example.com/source", response.content)
        self.assertIn(".md", response["Content-Disposition"])

    # Verify that when the session snapshot has been removed, the Markdown export falls back to the title and
    # report sent in the request and names the file after that title.
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

    # Verify that the DOCX export stores the citation URL as an external relationship, so the link is
    # clickable in Word.
    def test_docx_export_contains_clickable_external_relationship(self) -> None:
        response = self.export("docx")
        self.assertEqual(response.status_code, 200)
        with ZipFile(BytesIO(response.content)) as archive:
            relationships = archive.read("word/_rels/document.xml.rels").decode("utf-8")
        self.assertIn("https://example.com/source", relationships)
        self.assertIn('TargetMode="External"', relationships)

    # Verify that the PDF export returns a document that starts with the PDF header.
    def test_pdf_export_returns_pdf_document(self) -> None:
        response = self.export("pdf")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.content.startswith(b"%PDF-"))

    # Verify that the DOCX export renders the rich report as Word content: a table, Office math and no raw
    # Markdown, a black borderless title style, and the Mermaid diagram as one picture clamped to the page width.
    def test_docx_renders_markdown_math_and_mermaid(self) -> None:
        # Arrange: store the rich report in the session.
        control.update_state(
            self.session_id,
            report=self.RICH_REPORT,
            model_context=self.RICH_REPORT,
        )

        # The diagram renderer shells out to a local browser binary; replace
        # only that boundary so the DOCX assembly pipeline stays exercised.
        # Write a plain 1600x400 PNG with a rectangle instead of rendering the Mermaid source.
        def fake_render(source, output_path, workspace) -> None:
            from PIL import Image, ImageDraw

            image = Image.new("RGB", (1600, 400), "white")
            ImageDraw.Draw(image).rectangle([40, 80, 700, 320], outline="black", width=6)
            image.save(output_path, "PNG")

        # Act: export the DOCX with the diagram renderer replaced.
        with mock.patch.object(export, "_render_mermaid_png", side_effect=fake_render):
            response = self.export("docx")

        # Assert: read the document XML, styles and embedded diagram image from the DOCX archive.
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

        # Assert: the Title style has no border and a black color.
        from xml.etree import ElementTree
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        styles_root = ElementTree.fromstring(styles_xml)
        title_style = styles_root.find(".//w:style[@w:styleId='Title']", namespace)
        self.assertIsNotNone(title_style)
        self.assertIsNone(title_style.find("./w:pPr/w:pBdr", namespace))
        self.assertEqual(title_style.find("./w:rPr/w:color", namespace).get(f"{{{namespace['w']}}}val"), "000000")

        # Assert: exactly one diagram picture, at its original size, scaled down on the page.
        self.assertEqual(len(media), 1)
        self.assertEqual(diagram_size, (1600, 400))

        # 1600 px wide is clamped to the 6.35 in display width (5806440 EMU).
        self.assertIn('cx="5806440"', document_xml)

    # Verify that the PDF export renders the rich report as readable text and a table without raw Markdown
    # markers, keeps the citation link clickable and renders the page in grayscale.
    def test_pdf_renders_report_pages_and_keeps_links(self) -> None:
        # Arrange and act: store the rich report in the session and export the PDF.
        control.update_state(
            self.session_id,
            report=self.RICH_REPORT,
            model_context=self.RICH_REPORT,
        )

        response = self.export("pdf")

        # Assert: read the text, links and a pixel sample of the first page from the PDF.
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

        # Assert: the rendered page has equal red, green and blue channels, so it is grayscale.
        from PIL import Image, ImageChops
        rendered_page = Image.frombytes("RGB", (sample.width, sample.height), sample.samples)
        red, green, blue = rendered_page.split()
        self.assertIsNone(ImageChops.difference(red, green).getbbox())
        self.assertIsNone(ImageChops.difference(red, blue).getbbox())


# Cover the control endpoint that reads the state of a research session and queues approve, revise and
# cancel commands, plus the command inbox and snapshot rules behind it.
class DeepResearchControlApiTests(SimpleTestCase):
    # Redirect the control state and command folders into a temporary directory and create a session that is
    # awaiting approval of plan version 1.
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

    # Post the payload as JSON to the control endpoint and return the response.
    def post(self, payload: dict):
        return self.client.post(
            "/api/deep-research/control/",
            data=json.dumps(payload),
            content_type="application/json",
        )

    # Verify that the control endpoint returns the session state and accepts approval of the current plan
    # version with a 202 response and a queued approve command.
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

    # Verify that a revision made against an outdated plan version is rejected with a 409 response that
    # returns the latest state and queues no command.
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

    # Verify that cancelling a session that is already cancelled is accepted with a 202 response and reported
    # as terminal.
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

    # Verify that cancelling a session whose active worker is aborted right away marks it cancelled and
    # terminal, reports one interrupted tool and disables stopping.
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

    # Verify that cancelling a session with no tool to abort sets it to stopping (not terminal) and persists
    # that state with the stop control disabled, while the worker stops cooperatively.
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

    # Verify that a session id that is not a valid deep research id (a path traversal) returns a 400 error.
    def test_invalid_session_id_is_rejected(self) -> None:
        response = self.post({"session_id": "../../outside", "action": "cancel"})
        self.assertEqual(response.status_code, 400)

    # Verify that a revision sent after editing was disabled returns a 409 response and queues no command.
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

    # Verify that reading commands with a processed file name returns nothing, deletes that command file and
    # removes the then empty command folder.
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

    # Verify that storing a terminal (completed) snapshot removes the session's command folder with its
    # unprocessed command.
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

    # Verify that a late worker checkpoint cannot move a stopping session back to researching or re-enable its
    # controls, while its other fields are still saved and a final cancelled state is accepted.
    def test_stale_worker_checkpoint_cannot_resurrect_a_stopping_run(self) -> None:
        # Arrange: the session is stopping, with every control disabled.
        control.update_state(
            self.session_id,
            status="stopping",
            phase="stopping",
            can_stop=False,
            can_edit=False,
            can_approve=False,
        )

        # Act: a late worker checkpoint reports the session as researching with the controls enabled.
        state = control.update_state(
            self.session_id,
            status="researching",
            phase="search",
            can_stop=True,
            can_edit=True,
            can_approve=True,
            source_count=7,
        )

        # Assert: status, phase and controls stay in the stopping state, but the source count is saved.
        self.assertEqual(state["status"], "stopping")
        self.assertEqual(state["phase"], "stopping")
        self.assertFalse(state["can_stop"])
        self.assertFalse(state["can_edit"])
        self.assertFalse(state["can_approve"])
        self.assertEqual(state["source_count"], 7)

        # The final cancelled state is still accepted.
        terminal = control.update_state(
            self.session_id,
            status="cancelled",
            phase="cancelled",
        )
        self.assertEqual(terminal["status"], "cancelled")
        self.assertEqual(terminal["phase"], "cancelled")


# Cover the runtime storage of research sessions: the storage location, the one-time migration of the legacy
# repository-local store and the retention of old snapshots.
class DeepResearchStorageTests(SimpleTestCase):
    # Point the control storage roots at the given runtime and legacy folders and reset the retention
    # throttle, for the duration of the test.
    def _patch_storage(self, runtime_root: Path, legacy_root: Path) -> None:
        self.enterContext(mock.patch.object(control, "CONTROL_ROOT", runtime_root))
        self.enterContext(mock.patch.object(control, "STATE_ROOT", runtime_root / "state"))
        self.enterContext(mock.patch.object(control, "COMMAND_ROOT", runtime_root / "commands"))
        self.enterContext(mock.patch.object(control, "LEGACY_CONTROL_ROOT", legacy_root))
        self.enterContext(mock.patch.object(control, "_last_retention_check", 0.0))

    # Verify that the ASLM_DEEP_RESEARCH_RUNTIME_DIR environment variable overrides the default control root.
    def test_runtime_directory_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(
                os.environ,
                {"ASLM_DEEP_RESEARCH_RUNTIME_DIR": directory},
                clear=False,
            ):
                self.assertEqual(control._default_control_root(), Path(directory))

    # Verify that on Windows, without an override, the control root is under LOCALAPPDATA and outside the
    # project repository.
    def test_windows_default_is_outside_the_repository(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("ASLM_DEEP_RESEARCH_RUNTIME_DIR", None)
                os.environ["LOCALAPPDATA"] = directory
                with mock.patch.object(control.platform, "system", return_value="Windows"):
                    expected = Path(directory) / "ASLM-Chat" / "runtime" / "deep-research"
                    self.assertEqual(control._default_control_root(), expected)
                    self.assertFalse(expected.is_relative_to(control.PROJECT_ROOT))

    # Verify that preparing the storage moves a legacy state snapshot and command file into the runtime
    # folders unchanged, writes the migration marker and removes the legacy folder.
    def test_legacy_store_is_atomically_moved_and_payload_is_preserved(self) -> None:
        # Arrange: a legacy store with one state snapshot and one command file.
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

            # Act: preparing the storage runs the migration.
            control._ensure_roots()

            # Assert: both files arrived unchanged, the marker exists and the legacy folder is gone.
            self.assertEqual(control._read_json(runtime_root / "state" / legacy_state.name), state_payload)
            self.assertEqual(
                control._read_json(runtime_root / "commands" / session_key / legacy_command.name),
                command_payload,
            )
            self.assertTrue((runtime_root / control._MIGRATION_MARKER).exists())
            self.assertFalse(legacy_root.exists())

    # Verify that when a session exists in both stores, the migration keeps the newer runtime snapshot, adds
    # the fields only the legacy snapshot has and removes the legacy folder.
    def test_migration_keeps_newer_runtime_state_and_legacy_only_fields(self) -> None:
        # Arrange: the same session in both stores, an older legacy snapshot with an extra field and a newer
        # runtime snapshot with a report.
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

            # Act: preparing the storage runs the migration.
            control._ensure_roots()

            # Assert: the runtime snapshot keeps its newer values and gains the legacy-only field.
            merged = control._read_json(current_state)
            self.assertGreater(merged["updated_at"], time.time() - 5)
            self.assertEqual(merged["report"], "new")
            self.assertEqual(merged["citation_next"], 12)
            self.assertFalse(legacy_root.exists())

    # Verify that a legacy snapshot that cannot be deleted (locked) is still copied to the runtime folder,
    # stays in place, and keeps the migration unmarked so it is retried later.
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

            # Simulate a locked file: fail with PermissionError when the legacy snapshot is deleted and delete
            # every other path normally.
            def fail_for_legacy(path: Path, *args, **kwargs):
                if path == legacy_state:
                    raise PermissionError("locked")
                return original_unlink(path, *args, **kwargs)

            with mock.patch.object(Path, "unlink", new=fail_for_legacy):
                control._ensure_roots()

            self.assertTrue(legacy_state.exists())
            self.assertTrue((runtime_root / "state" / legacy_state.name).exists())
            self.assertFalse((runtime_root / control._MIGRATION_MARKER).exists())

    # Verify that the cleanup after a restart removes the command files a running session already acknowledged
    # (keeping its pending commands) and the whole command folder of a finished session.
    def test_restart_cleanup_removes_persisted_acknowledgements_and_terminal_inboxes(self) -> None:
        # Arrange: a running session with one processed and one pending command, and a completed session with
        # a pending command.
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

            # Act and assert: a forced retention sweep applies the cleanup.
            control._apply_terminal_retention(force=True)

            self.assertFalse((active_commands / processed_name).exists())
            self.assertTrue((active_commands / pending_name).exists())
            self.assertFalse(terminal_commands.exists())

    # Verify that the retention sweep deletes finished snapshots (and their command folders) that exceed the
    # maximum count or age, keeps the newest ones, and never deletes an active run however old.
    def test_terminal_retention_bounds_age_and_count_without_touching_active_runs(self) -> None:
        # Arrange: three finished sessions of different age and one active session that is 90 days old, each
        # with a command folder.
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

            # Act: sweep with a limit of one day and one finished snapshot.
            with mock.patch.dict(
                os.environ,
                {
                    "ASLM_DEEP_RESEARCH_TERMINAL_RETENTION_DAYS": "1",
                    "ASLM_DEEP_RESEARCH_TERMINAL_RETENTION_MAX": "1",
                },
                clear=False,
            ):
                control._apply_terminal_retention(force=True)

            # Assert: the newest finished session and the active one remain; the older finished ones are gone.
            self.assertTrue((runtime_root / "state" / f"{keys[0]}.json").exists())
            self.assertFalse((runtime_root / "state" / f"{keys[1]}.json").exists())
            self.assertFalse((runtime_root / "state" / f"{keys[2]}.json").exists())
            self.assertTrue((runtime_root / "state" / f"{keys[3]}.json").exists())
            self.assertFalse((runtime_root / "commands" / keys[1]).exists())
            self.assertFalse((runtime_root / "commands" / keys[2]).exists())
            self.assertTrue((runtime_root / "commands" / keys[3]).exists())


# Cover how a stored assistant transcript is turned back into activity segments after a page reload.
class DeepResearchTranscriptReloadTests(SimpleTestCase):
    # Verify that a transcript with a research tool call but no result yet rebuilds a research card that
    # carries the session id and query budget and has no result, so the UI can keep polling the session.
    def test_pending_tool_call_restores_pollable_research_card(self) -> None:
        # Arrange: a transcript with only the assistant's research tool call.
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

        # Act and assert: the rebuilt tool segment is a pollable research card without a result.
        segments = api_views._build_activity_segments(message)

        research = next(segment for segment in segments if segment["type"] == "tool")
        self.assertEqual(research["toolUi"]["kind"], "deep_research")
        self.assertEqual(research["toolUi"]["session_id"], session_id)
        self.assertEqual(research["toolUi"]["query_budget"], 4)
        self.assertIsNone(research["result"])

    # Verify that a research tool call and its tool result in the transcript merge into one tool segment
    # that holds the final report and the completed status.
    def test_completed_tool_call_and_result_merge_without_duplicate_card(self) -> None:
        # Arrange: a transcript with the research tool call followed by its tool result.
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

        # Act: keep only the tool segments of the rebuilt activity.
        tools = [
            segment
            for segment in api_views._build_activity_segments(message)
            if segment["type"] == "tool"
        ]

        # Assert: a single card holds the report and the completed status.
        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0]["result"], "Final report")
        self.assertEqual(tools[0]["toolUi"]["status"], "completed")
