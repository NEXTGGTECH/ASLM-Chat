# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from urllib.parse import urlparse


# Largest PDF download that is processed, and the cap on the extracted markdown length.
MAX_PDF_BYTES = 10 * 1024 * 1024
MAX_PDF_TEXT_CHARS = 80_000

# Three or more consecutive newlines, collapsed to a single blank line in the final document.
_BLANK_LINES_RE = re.compile(r"\n{3,}")

# Tokens of a raw PDF file; several of them in decoded text mean it is a PDF dump, not prose.
_PDF_DUMP_MARKERS = (
    "%PDF-",
    "endobj",
    "/FlateDecode",
    "xref",
    "startxref",
)


# Return True when the URL path or query indicates a PDF resource (.pdf suffix, an arXiv /pdf/
# path, or format=pdf in the query).
def looks_like_pdf_url(url: str) -> bool:
    parsed = urlparse(url)
    path = parsed.path.lower()
    if path.endswith(".pdf"):
        return True
    if parsed.netloc.lower().endswith("arxiv.org") and path.startswith("/pdf/"):
        return True
    if "format=pdf" in parsed.query.lower():
        return True
    return False


# Return True when the raw bytes start (after leading whitespace) with the PDF magic header.
def looks_like_pdf_bytes(data: bytes) -> bool:
    return data.lstrip().startswith(b"%PDF-")


# Return True when decoded text looks like a PDF object stream, not article body: it starts with
# the PDF header or its first 12000 characters hold at least three of the PDF dump markers.
def looks_like_pdf_text_dump(text: str) -> bool:
    if not text:
        return False
    head = text.lstrip()[:12000]
    if head.startswith("%PDF-"):
        return True
    marker_count = sum(1 for marker in _PDF_DUMP_MARKERS if marker in head)
    return marker_count >= 3


# Return True when text is likely arbitrary binary decoded as UTF-8: a PDF dump, or a sample of
# the first 12000 characters with more than 0.5% control characters or 1% replacement characters.
def looks_like_decoded_binary(text: str) -> bool:
    if looks_like_pdf_text_dump(text):
        return True
    sample = text[:12000]
    if not sample:
        return False
    printable_controls = {"\t", "\n", "\r"}
    control_count = sum(1 for ch in sample if ord(ch) < 32 and ch not in printable_controls)
    replacement_count = sample.count("\ufffd")
    length = max(1, len(sample))
    return control_count / length > 0.005 or replacement_count / length > 0.01


# Write PDF bytes to a temp file, extract markdown, then delete the temp file. Return "" for empty,
# oversized or non-PDF data.
def pdf_bytes_to_markdown(
    *,
    url: str,
    data: bytes,
    title: str = "",
    max_chars: int = MAX_PDF_TEXT_CHARS,
) -> str:
    if not data:
        return ""
    if len(data) > MAX_PDF_BYTES:
        return ""
    if not looks_like_pdf_bytes(data):
        return ""

    # The extractor needs a real file path, so spool the bytes to disk and always remove the file.
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as handle:
            handle.write(data)
            tmp_path = Path(handle.name)
        return pdf_file_to_markdown(url=url, path=tmp_path, title=title, max_chars=max_chars)
    finally:
        if tmp_path is not None:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                # Failing to delete a temp file must not mask the extraction result.
                pass


# Extract a local PDF file into a bounded markdown document: a small header (site, URL, source
# type) followed by the body. Return "" when no text could be extracted.
def pdf_file_to_markdown(
    *,
    url: str,
    path: Path,
    title: str = "",
    max_chars: int = MAX_PDF_TEXT_CHARS,
) -> str:
    # Preferred path: pymupdf4llm layout-aware markdown, with its console output silenced and
    # the result cut at the last full line before max_chars.
    try:
        import pymupdf4llm  # type: ignore
        import contextlib
        import os
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                body = pymupdf4llm.to_markdown(str(path))
        if not body or not body.strip():
            raise ValueError("empty")
        body = body.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rsplit("\n", 1)[0].rstrip() + "\n\n[...truncated]"
    except Exception:
        # Fallback: plain text per page via PyMuPDF (fitz), stopping once max_chars is reached.
        import fitz
        text_parts: list[str] = []
        doc = fitz.open(str(path))
        try:
            meta = doc.metadata or {}
            title = (title or str(meta.get("title") or "")).strip() or "PDF Document"
            for index, page in enumerate(doc, 1):
                page_text = (page.get_text("text") or "").strip()
                if not page_text:
                    continue
                text_parts.append(f"\n\n## Page {index}\n\n{page_text}")
                if sum(len(part) for part in text_parts) >= max_chars:
                    break
        finally:
            doc.close()
        body = "\n".join(text_parts).strip()

    if not body:
        return ""

    # Fill a missing title from the PDF metadata, with a generic name as the last resort.
    if not title:
        try:
            import fitz
            doc = fitz.open(str(path))
            meta = doc.metadata or {}
            doc.close()
            title = str(meta.get("title") or "").strip() or "PDF Document"
        except Exception:
            title = "PDF Document"

    # Assemble the document with a source header and collapse runs of blank lines.
    site = urlparse(url).netloc.removeprefix("www.")
    markdown = (
        f"**Site:** {site}\n"
        f"**URL:** {url}\n"
        "**Source type:** PDF\n\n"
        "---\n\n"
        f"{body}"
    )
    return _BLANK_LINES_RE.sub("\n\n", markdown).strip()
