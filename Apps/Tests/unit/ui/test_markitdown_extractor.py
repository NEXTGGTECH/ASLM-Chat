# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import io

import pytest

from Apps.UI import markitdown_extractor


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


# Conversion.


# A genuine DOCX converts to markdown text.
def test_extract_markdown_converts_docx():
    docx_bytes = _build_docx(["MarkItDown heading", "Body paragraph"])

    text = markitdown_extractor.extract_markdown(docx_bytes, name="report.docx", mime="")

    assert "MarkItDown heading" in text
    assert "Body paragraph" in text


# Empty payloads convert to an empty string.
def test_extract_markdown_rejects_empty_payload():
    assert markitdown_extractor.extract_markdown(b"", name="a.docx", mime="") == ""


# Oversized payloads are rejected without starting a conversion.
def test_extract_markdown_rejects_oversized_payload(monkeypatch):
    monkeypatch.setattr(markitdown_extractor, "MARKITDOWN_MAX_INPUT_BYTES", 10)

    assert markitdown_extractor.extract_markdown(b"x" * 11, name="a.docx", mime="") == ""


# A missing converter yields an empty string.
def test_extract_markdown_returns_empty_without_converter(monkeypatch):
    monkeypatch.setattr(markitdown_extractor, "_get_markitdown_instance", lambda: None)

    assert markitdown_extractor.extract_markdown(b"payload", name="a.docx", mime="") == ""


# Converter failures never propagate to the caller.
def test_extract_markdown_swallows_conversion_errors(monkeypatch):
    def broken_once(file_bytes, *, name, mime):
        raise RuntimeError("converter exploded")

    monkeypatch.setattr(markitdown_extractor, "_extract_once", broken_once)

    assert markitdown_extractor.extract_markdown(b"payload", name="a.docx", mime="") == ""


# Converter instance caching.


# The shared converter instance is created lazily and reused.
def test_markitdown_instance_is_cached(monkeypatch):
    monkeypatch.setattr(markitdown_extractor, "_MARKITDOWN_INSTANCE", None)
    monkeypatch.setattr(markitdown_extractor, "_MARKITDOWN_IMPORT_ERROR", False)

    first = markitdown_extractor._get_markitdown_instance()
    second = markitdown_extractor._get_markitdown_instance()

    assert first is not None
    assert second is first
