# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import io

import pytest

from Services import markitdown_extractor


pytestmark = [pytest.mark.unit]


# Build a genuine DOCX payload for real conversion.
def _build_docx(paragraphs: list[str]) -> bytes:
    from docx import Document

    document = Document()
    for paragraph in paragraphs:
        document.add_paragraph(paragraph)

    stream = io.BytesIO()
    document.save(stream)
    return stream.getvalue()


# Conversion

# Verify that a genuine DOCX converts to markdown text containing its paragraphs.
def test_extract_markdown_converts_docx():
    docx_bytes = _build_docx(["MarkItDown heading", "Body paragraph"])

    text = markitdown_extractor.extract_markdown(docx_bytes, name="report.docx", mime="")

    assert "MarkItDown heading" in text
    assert "Body paragraph" in text


# Verify that an empty payload converts to an empty string.
def test_extract_markdown_rejects_empty_payload():
    assert markitdown_extractor.extract_markdown(b"", name="a.docx", mime="") == ""


# Verify that a payload above the input size limit converts to an empty string.
def test_extract_markdown_rejects_oversized_payload(monkeypatch):
    monkeypatch.setattr(markitdown_extractor, "MARKITDOWN_MAX_INPUT_BYTES", 10)

    assert markitdown_extractor.extract_markdown(b"x" * 11, name="a.docx", mime="") == ""


# Verify that an unavailable converter instance yields an empty string.
def test_extract_markdown_returns_empty_without_converter(monkeypatch):
    monkeypatch.setattr(markitdown_extractor, "_get_markitdown_instance", lambda: None)

    assert markitdown_extractor.extract_markdown(b"payload", name="a.docx", mime="") == ""


# Verify that an exception raised during conversion is swallowed and yields an empty string.
def test_extract_markdown_swallows_conversion_errors(monkeypatch):
    # Fail every conversion attempt with a RuntimeError.
    def broken_once(file_bytes, *, name, mime):
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(markitdown_extractor, "_extract_once", broken_once)

    assert markitdown_extractor.extract_markdown(b"payload", name="a.docx", mime="") == ""


# Converter instance caching

# Verify that the shared converter instance is created on first use and reused afterwards.
def test_markitdown_instance_is_cached(monkeypatch):
    monkeypatch.setattr(markitdown_extractor, "_MARKITDOWN_INSTANCE", None)
    monkeypatch.setattr(markitdown_extractor, "_MARKITDOWN_IMPORT_ERROR", False)

    first = markitdown_extractor._get_markitdown_instance()
    second = markitdown_extractor._get_markitdown_instance()

    assert first is not None
    assert second is first
