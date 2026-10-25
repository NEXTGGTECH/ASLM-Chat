# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import hashlib
import io
import zipfile

import pytest

from Services.file_manifests import (
    UploadedFileManifest,
    build_uploaded_file_manifest,
    guess_upload_mime,
    is_probably_text_upload,
    normalize_upload_name,
)


pytestmark = [pytest.mark.unit]


# Fixture builders for real binary documents.


# Build a genuine DOCX through python-docx.
def _build_docx(paragraphs: list[str]) -> bytes:
    from docx import Document

    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)

    stream = io.BytesIO()
    document.save(stream)
    return stream.getvalue()


# Build a genuine one-page PDF through PyMuPDF.
def _build_pdf(text: str) -> bytes:
    import fitz

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    return document.tobytes()


# Build a minimal XLSX with shared strings and one sheet.
def _build_xlsx(shared_strings: list[str], rows: list[list[str]]) -> bytes:
    sst_items = "".join(f"<si><t>{value}</t></si>" for value in shared_strings)
    sheet_rows = []
    for row in rows:
        cells = []
        for cell in row:
            # "#str:" marks a shared-string cell, anything else a value cell.
            if cell.startswith("#str:"):
                index = shared_strings.index(cell[5:])
                cells.append(f'<c t="s"><v>{index}</v></c>')
            else:
                cells.append(f"<c><v>{cell}</v></c>")
        sheet_rows.append("<row>" + "".join(cells) + "</row>")

    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">{sst_items}</sst>',
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<sheetData>" + "".join(sheet_rows) + "</sheetData></worksheet>",
        )
    return stream.getvalue()


# Build a minimal PPTX with one text slide per entry.
def _build_pptx(slides: list[str]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for index, slide_text in enumerate(slides, start=1):
            archive.writestr(
                f"ppt/slides/slide{index}.xml",
                '<p:sld xmlns:p="urn:schemas-microsoft-com:office:powerpoint" '
                'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                f"<p:txBody><a:p><a:t>{slide_text}</a:t></a:p></p:txBody></p:sld>",
            )
    return stream.getvalue()


# Text uploads.


# A plain text upload produces a complete manifest.
def test_text_upload_manifest_fields():
    content = "Hello from ASLM.\nSecond line.\n"

    manifest = build_uploaded_file_manifest(
        content.encode("utf-8"),
        name="notes.txt",
        mime="text/plain",
        sandbox_path="/workspace/_sandbox/User/notes.txt",
    )

    assert isinstance(manifest, UploadedFileManifest)
    assert manifest.name == "notes.txt"
    assert manifest.mime == "text/plain"
    assert manifest.size_bytes == len(content.encode("utf-8"))
    assert manifest.sha256 == hashlib.sha256(content.encode("utf-8")).hexdigest()
    assert manifest.text_available is True
    assert "Hello from ASLM." in manifest.text_preview
    assert manifest.text_truncated is False
    assert manifest.sandbox_path == "/workspace/_sandbox/User/notes.txt"
    assert manifest.recommended_tools == ["sandbox"]
    assert manifest.vision_available is False


# Binary content is never decoded as text.
def test_binary_upload_has_no_text():
    payload = bytes(range(256)) * 64

    manifest = build_uploaded_file_manifest(payload, name="blob.bin", mime="application/octet-stream")

    assert manifest.text_available is False
    assert manifest.text_preview is None


# Non-UTF-8 text encodings are recovered for declared text uploads.
def test_cp1251_text_decodes_for_declared_text_uploads():
    raw_text = "Привет из ASLM"

    manifest = build_uploaded_file_manifest(
        raw_text.encode("cp1251"),
        name="letter.txt",
        mime="text/plain",
    )

    assert manifest.text_available is True
    assert "Привет из ASLM" in manifest.text_preview


# Oversized text is truncated with a marker and recommends file_search.
def test_oversized_text_is_truncated_and_recommends_file_search():
    content = ("line\n" * 60_000).encode("utf-8")

    manifest = build_uploaded_file_manifest(content, name="huge.log", mime="text/plain")

    assert manifest.text_truncated is True
    assert manifest.text_preview.endswith("...[truncated]")
    assert manifest.text_total_chars > len(manifest.text_preview)
    assert "file_search" in manifest.recommended_tools


# Table previews.


# CSV uploads render a pipe-separated table preview.
def test_csv_upload_builds_table_preview():
    content = "name,city\nAlice,New York\nBob,San Francisco\n"

    manifest = build_uploaded_file_manifest(content.encode("utf-8"), name="people.csv", mime="text/csv")

    assert manifest.table_preview is not None
    assert "name | city" in manifest.table_preview
    assert "Alice | New York" in manifest.table_preview
    assert "Bob | San Francisco" in manifest.table_preview


# TSV previews use the tab delimiter.
def test_tsv_upload_uses_tab_delimiter():
    content = "col1\tcol2\none\ttwo\n"

    manifest = build_uploaded_file_manifest(
        content.encode("utf-8"), name="data.tsv", mime="text/tab-separated-values"
    )

    assert manifest.table_preview is not None
    assert "col1 | col2" in manifest.table_preview
    assert "one | two" in manifest.table_preview


# Archives.


# Zip uploads expose their tree and recommend the archive tool.
def test_zip_upload_builds_tree_and_recommends_archive_tool():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("src/main.py", "print('hi')\n")
        archive.writestr("docs/readme.md", "readme")

    manifest = build_uploaded_file_manifest(
        stream.getvalue(),
        name="project.zip",
        mime="application/zip",
        sandbox_path="/workspace/_sandbox/User/project.zip",
    )

    assert manifest.archive_tree == ["src/main.py", "docs/readme.md"]
    assert manifest.recommended_tools == ["sandbox", "archive"]
    assert manifest.text_available is False


# Office documents.


# DOCX uploads extract their paragraph text.
def test_docx_upload_extracts_paragraph_text():
    docx_bytes = _build_docx(["ASLM chat export", "Second paragraph"])

    manifest = build_uploaded_file_manifest(docx_bytes, name="report.docx", mime=None)

    assert manifest.mime == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    assert manifest.text_available is True
    assert "ASLM chat export" in manifest.text_preview
    assert "Second paragraph" in manifest.text_preview


# XLSX uploads resolve shared strings into a table preview.
def test_xlsx_upload_builds_table_preview():
    xlsx_bytes = _build_xlsx(
        ["Name", "City"],
        [["#str:Name", "#str:City"], ["1", "2"]],
    )

    manifest = build_uploaded_file_manifest(xlsx_bytes, name="table.xlsx")

    assert manifest.text_available is True
    assert manifest.table_preview is not None
    assert "Name | City" in manifest.table_preview
    assert "1 | 2" in manifest.table_preview
    assert manifest.recommended_tools == []


# PPTX uploads extract slide text with slide numbers.
def test_pptx_upload_extracts_slide_text():
    pptx_bytes = _build_pptx(["Title slide", "Bullet slide"])

    manifest = build_uploaded_file_manifest(pptx_bytes, name="deck.pptx")

    assert manifest.text_available is True
    assert "Slide 1:" in manifest.text_preview
    assert "Title slide" in manifest.text_preview
    assert "Slide 2:" in manifest.text_preview


# PDF uploads extract the embedded text layer.
def test_pdf_upload_extracts_text():
    pdf_bytes = _build_pdf("ASLM PDF fixture text")

    manifest = build_uploaded_file_manifest(pdf_bytes, name="document.pdf")

    assert manifest.text_available is True
    assert "ASLM PDF fixture text" in manifest.text_preview


# Vision gating.


# Image vision availability follows the model capability flag.
def test_image_upload_vision_depends_on_model_support():
    png_bytes = b"\x89PNG\r\n\x1a\nfake-image-bytes"

    without_vision = build_uploaded_file_manifest(png_bytes, name="photo.png", mime="image/png")
    with_vision = build_uploaded_file_manifest(
        png_bytes, name="photo.png", mime="image/png", model_supports_vision=True
    )

    assert without_vision.vision_available is False
    assert with_vision.vision_available is True


# Name and MIME normalization.


# Upload names reduce to a safe basename.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("report.pdf", "report.pdf"),
        ("C:/Users/someone/notes.txt", "notes.txt"),
        ("..\\..\\escape.md", "escape.md"),
        ("", "uploaded-file"),
        ("   ", "uploaded-file"),
    ],
)
def test_normalize_upload_name(raw, expected):
    assert normalize_upload_name(raw) == expected


# Explicit MIME wins, then the extension, then octet-stream.
def test_guess_upload_mime_prefers_explicit_then_extension():
    assert guess_upload_mime("a.txt", "text/x-custom") == "text/x-custom"
    assert guess_upload_mime("a.json") == "application/json"
    assert guess_upload_mime("a.unknownext") == "application/octet-stream"


# Text detection heuristics.


# Text detection follows MIME type, known names, and extensions.
@pytest.mark.parametrize(
    ("name", "mime", "expected"),
    [
        ("notes.txt", "", True),
        ("data.json", "application/json", True),
        (".gitignore", "application/octet-stream", True),
        ("README.md", "", True),
        ("archive.zip", "application/zip", False),
        ("photo.png", "image/png", False),
    ],
)
def test_is_probably_text_upload(name, mime, expected):
    assert is_probably_text_upload(name, mime) is expected
