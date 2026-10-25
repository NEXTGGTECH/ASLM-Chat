# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import csv
import hashlib
import io
import mimetypes
import uuid
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from Services.markitdown_extractor import extract_markdown


TEXT_FILE_EXTENSIONS = {
    ".adoc", ".ahk", ".asm", ".asciidoc", ".bash", ".bat", ".bib", ".c", ".cc", ".cfg",
    ".clj", ".cljc", ".cljs", ".cmd", ".cmake", ".conf", ".config", ".cpp", ".cs",
    ".cshtml", ".csproj", ".css", ".csv", ".cts", ".cue", ".cxx", ".dart", ".diff",
    ".dockerfile", ".edn", ".ejs", ".env", ".erb", ".erl", ".ex", ".exs", ".fish",
    ".fs", ".fsi", ".fsx", ".geojson", ".go", ".gql", ".gradle", ".graphql", ".groovy",
    ".gvy", ".h", ".handlebars", ".haml", ".hbs", ".hpp", ".hrl", ".hs", ".htm", ".html",
    ".http", ".hxx", ".ini", ".java", ".jinja", ".jinja2", ".jl", ".js", ".json",
    ".json5", ".jsonc", ".jsonl", ".jsx", ".kt", ".kts", ".ksh", ".latex", ".less",
    ".liquid", ".log", ".lua", ".m", ".mako", ".markdown", ".md", ".mdown", ".mdx",
    ".mjml", ".mkd", ".mm", ".mod", ".mts", ".mustache", ".nix", ".njk", ".patch",
    ".php", ".pl", ".pm", ".pod", ".properties", ".proto", ".ps1", ".ps1xml", ".psd1",
    ".psm1", ".pug", ".py", ".pyi", ".r", ".rb", ".rego", ".rest", ".rst", ".rs",
    ".sass", ".scala", ".scss", ".service", ".sh", ".shtml", ".sol", ".sql", ".srt",
    ".styl", ".svelte", ".sum", ".svg", ".swift", ".tcl", ".templ", ".tex", ".tf",
    ".tfvars", ".toml", ".ts", ".tsv", ".tsx", ".twig", ".txt", ".vb", ".vbs", ".vue",
    ".vtt", ".xhtml", ".xml", ".xsd", ".xsl", ".yaml", ".yml", ".zsh",
}

TEXT_FILE_NAMES = {
    ".bash_profile", ".bashrc", ".dockerignore", ".editorconfig", ".env", ".env.example",
    ".eslintignore", ".eslintrc", ".gitattributes", ".gitconfig", ".gitignore", ".gitkeep",
    ".gitmodules", ".npmrc", ".nvmrc", ".prettierignore", ".prettierrc", ".python-version",
    ".stylelintrc", ".tool-versions", ".yamllint", ".zshenv", ".zshrc", "brewfile",
    "cmakelists.txt", "containerfile", "dockerfile", "gemfile", "jenkinsfile", "justfile",
    "makefile", "procfile", "rakefile", "tiltfile", "vagrantfile",
}

TEXT_MIME_TYPES = {
    "application/ecmascript",
    "application/json",
    "application/ld+json",
    "application/javascript",
    "application/sql",
    "application/toml",
    "application/vnd.api+json",
    "application/xml",
    "application/x-httpd-php",
    "application/x-javascript",
    "application/x-ndjson",
    "application/x-sh",
    "application/x-shellscript",
    "application/x-toml",
    "application/x-yaml",
    "application/yaml",
    "image/svg+xml",
}

TEXT_PREVIEW_CHAR_LIMIT = 24_000
TEXT_FULL_CHAR_LIMIT = 256 * 1024
TEXT_MAX_SCAN_BYTES = 2 * 1024 * 1024
ARCHIVE_TREE_LIMIT = 250
TABLE_PREVIEW_ROWS = 20
TABLE_PREVIEW_CELL_LIMIT = 120
OFFICE_XML_MEMBER_LIMIT = 80


@dataclass(frozen=True)
class UploadedFileManifest:
    file_id: str
    name: str
    mime: str
    size_bytes: int
    sha256: str
    sandbox_path: str | None
    text_available: bool
    text_preview: str | None
    text_total_chars: int | None
    text_truncated: bool
    vision_available: bool
    archive_tree: list[str] | None
    table_preview: str | None
    recommended_tools: list[str]

    # Return a JSON-serializable dict for the dataclass.
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Return a display-safe basename for an uploaded file.
def normalize_upload_name(name: str) -> str:
    clean_name = Path(str(name or "").replace("\\", "/")).name.strip()
    return clean_name or "uploaded-file"


# Return a stable MIME value for an uploaded file.
def guess_upload_mime(name: str, mime: str | None = None) -> str:
    clean_mime = str(mime or "").strip().lower()
    if clean_mime:
        return clean_mime
    guessed_mime, _encoding = mimetypes.guess_type(name)
    return guessed_mime or "application/octet-stream"


# Return whether an upload should be treated as text-like.
def is_probably_text_upload(name: str, mime: str) -> bool:
    normalized_mime = str(mime or "").strip().lower()
    normalized_name = normalize_upload_name(name).lower()
    upload_path = Path(normalized_name)
    if normalized_mime.startswith("text/") or normalized_mime in TEXT_MIME_TYPES:
        return True
    if normalized_name in TEXT_FILE_NAMES:
        return True
    return any(suffix.lower() in TEXT_FILE_EXTENSIONS for suffix in upload_path.suffixes)


# Upload text extraction helpers.

# Return whether a byte sample contains binary markers.
def _has_binary_markers(sample: bytes) -> bool:
    if b"\x00" in sample:
        return True
    if not sample:
        return False
    allowed_controls = {9, 10, 12, 13, 27}
    control_count = sum(1 for byte in sample if byte < 32 and byte not in allowed_controls)
    return control_count / len(sample) > 0.02


# Decode text bytes.
def _decode_text_bytes(file_bytes: bytes, *, explicit_text: bool) -> tuple[str, bool]:
    sample = file_bytes[:8192]
    if _has_binary_markers(sample):
        return "", False

    candidate_bytes = file_bytes[:TEXT_MAX_SCAN_BYTES]
    encodings = ("utf-8", "utf-8-sig", "cp1251")
    if explicit_text:
        encodings = (*encodings, "latin-1")

    for encoding in encodings:
        try:
            text = candidate_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
        if _looks_like_text(text):
            return text.replace("\r\n", "\n").replace("\r", "\n"), len(file_bytes) > len(candidate_bytes)
    return "", False


# Return whether decoded text looks printable enough to treat as text.
def _looks_like_text(text: str) -> bool:
    if not text:
        return True
    sample = text[:8192]
    printable = sum(1 for char in sample if char.isprintable() or char in "\n\t")
    return printable / max(len(sample), 1) >= 0.92


# Trim text preview.
def _trim_text_preview(text: str, *, source_truncated: bool) -> tuple[str | None, int, bool]:
    normalized = str(text or "").strip()
    total_chars = len(normalized)
    truncated = source_truncated or total_chars > TEXT_FULL_CHAR_LIMIT or total_chars > TEXT_PREVIEW_CHAR_LIMIT
    if total_chars == 0:
        return "", 0, source_truncated
    preview = normalized[:TEXT_PREVIEW_CHAR_LIMIT].rstrip()
    if truncated:
        preview = f"{preview}\n...[truncated]"
    return preview, total_chars, truncated


# Build table preview.
def _build_table_preview(name: str, text: str) -> str | None:
    suffix = Path(normalize_upload_name(name)).suffix.lower()
    if suffix not in {".csv", ".tsv"}:
        return None

    delimiter = "\t" if suffix == ".tsv" else ","
    rows: list[str] = []
    try:
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        for index, row in enumerate(reader):
            if index >= TABLE_PREVIEW_ROWS:
                rows.append("...[truncated]")
                break
            cells = [cell.replace("\n", " ")[:TABLE_PREVIEW_CELL_LIMIT] for cell in row[:12]]
            rows.append(" | ".join(cells))
    except csv.Error:
        return None
    return "\n".join(rows) if rows else None


# Build zip tree.
def _build_zip_tree(file_bytes: bytes) -> list[str] | None:
    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            names = []
            for info in archive.infolist()[:ARCHIVE_TREE_LIMIT]:
                suffix = "/" if info.is_dir() and not info.filename.endswith("/") else ""
                names.append(f"{info.filename}{suffix}")
            if len(archive.infolist()) > ARCHIVE_TREE_LIMIT:
                names.append("...[truncated]")
            return names
    except (zipfile.BadZipFile, OSError):
        return None


# Extract xml text.
def _extract_xml_text(xml_bytes: bytes) -> str:
    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError:
        return ""

    parts = []
    for text in root.itertext():
        clean_text = str(text or "").strip()
        if clean_text:
            parts.append(clean_text)
    return " ".join(parts)


# Extract pdf text.
def _extract_pdf_text(file_bytes: bytes) -> str:
    try:
        import fitz

        with fitz.open(stream=file_bytes, filetype="pdf") as document:
            pages = []
            for page in document:
                pages.append(page.get_text("text"))
                if sum(len(item) for item in pages) >= TEXT_MAX_SCAN_BYTES:
                    break
        return "\n".join(pages).strip()
    except Exception:
        return ""


# Extract docx text.
def _extract_docx_text(archive: zipfile.ZipFile) -> str:
    try:
        document_xml = archive.read("word/document.xml")
    except KeyError:
        return ""
    return _extract_xml_text(document_xml)


# Extract pptx text.
def _extract_pptx_text(archive: zipfile.ZipFile) -> str:
    slide_names = sorted(
        name for name in archive.namelist()
        if name.startswith("ppt/slides/slide") and name.endswith(".xml")
    )
    slides = []
    for index, slide_name in enumerate(slide_names[:OFFICE_XML_MEMBER_LIMIT], start=1):
        slide_text = _extract_xml_text(archive.read(slide_name))
        if slide_text:
            slides.append(f"Slide {index}: {slide_text}")
    return "\n".join(slides)


# Extract xlsx text.
def _extract_xlsx_text(archive: zipfile.ZipFile) -> str:
    shared_strings: list[str] = []
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
        for item in root:
            text = " ".join(part.strip() for part in item.itertext() if part.strip())
            if text:
                shared_strings.append(text)
    except (KeyError, ElementTree.ParseError):
        shared_strings = []

    sheet_names = sorted(
        name for name in archive.namelist()
        if name.startswith("xl/worksheets/sheet") and name.endswith(".xml")
    )
    rows = []
    for sheet_index, sheet_name in enumerate(sheet_names[:OFFICE_XML_MEMBER_LIMIT], start=1):
        try:
            root = ElementTree.fromstring(archive.read(sheet_name))
        except ElementTree.ParseError:
            continue

        sheet_rows = []
        for row in root.iter():
            if not row.tag.endswith("row"):
                continue
            cells = []
            for cell in row:
                if not cell.tag.endswith("c"):
                    continue
                cell_type = cell.attrib.get("t", "")
                value = ""
                for child in cell:
                    if child.tag.endswith("v") or child.tag.endswith("t"):
                        value = str(child.text or "").strip()
                        break
                if cell_type == "s" and value.isdigit():
                    string_index = int(value)
                    if 0 <= string_index < len(shared_strings):
                        value = shared_strings[string_index]
                if value:
                    cells.append(value)
            if cells:
                sheet_rows.append(" | ".join(cells[:12]))
            if len(sheet_rows) >= TABLE_PREVIEW_ROWS:
                break
        if sheet_rows:
            rows.append(f"Sheet {sheet_index}:\n" + "\n".join(sheet_rows))
    return "\n\n".join(rows)


# Extract document text.
def _extract_document_text(file_bytes: bytes, name: str, mime: str) -> str:
    suffix = Path(normalize_upload_name(name)).suffix.lower()
    normalized_mime = str(mime or "").lower()
    if suffix in {".pdf", ".docx", ".pptx", ".xlsx"} or normalized_mime == "application/pdf":
        # Prefer MarkItDown in safe stream mode, then fall back to local parsers.
        markdown_text = extract_markdown(file_bytes, name=name, mime=mime)
        if markdown_text:
            return markdown_text

    if suffix == ".pdf" or normalized_mime == "application/pdf":
        return _extract_pdf_text(file_bytes)

    if suffix not in {".docx", ".pptx", ".xlsx"}:
        return ""

    try:
        with zipfile.ZipFile(io.BytesIO(file_bytes)) as archive:
            if suffix == ".docx":
                return _extract_docx_text(archive)
            if suffix == ".pptx":
                return _extract_pptx_text(archive)
            if suffix == ".xlsx":
                return _extract_xlsx_text(archive)
    except (zipfile.BadZipFile, OSError):
        return ""
    return ""


# Build the model-facing manifest for one uploaded file.
def build_uploaded_file_manifest(
    file_bytes: bytes,
    *,
    name: str,
    mime: str | None = None,
    sandbox_path: str | None = None,
    model_supports_vision: bool = False,
    file_id: str | None = None,
    tool_server_id: str = "sandbox",
) -> UploadedFileManifest:
    raw_bytes = bytes(file_bytes or b"")
    clean_name = normalize_upload_name(name)
    clean_mime = guess_upload_mime(clean_name, mime)
    upload_id = str(file_id or uuid.uuid4())
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    is_image = clean_mime.startswith("image/")
    explicit_text = is_probably_text_upload(clean_name, clean_mime)

    text_preview = None
    text_total_chars = None
    text_truncated = False
    table_preview = None
    if explicit_text and raw_bytes:
        extracted_text, source_truncated = _decode_text_bytes(raw_bytes, explicit_text=True)
        if extracted_text or raw_bytes == b"":
            text_preview, text_total_chars, text_truncated = _trim_text_preview(
                extracted_text,
                source_truncated=source_truncated,
            )
            table_preview = _build_table_preview(clean_name, extracted_text)
    elif raw_bytes:
        extracted_text = _extract_document_text(raw_bytes, clean_name, clean_mime)
        if extracted_text:
            text_preview, text_total_chars, text_truncated = _trim_text_preview(
                extracted_text,
                source_truncated=len(extracted_text) >= TEXT_MAX_SCAN_BYTES,
            )
            if Path(clean_name).suffix.lower() == ".xlsx":
                table_preview = text_preview

    archive_tree = None
    if clean_name.lower().endswith(".zip") or clean_mime in {"application/zip", "application/x-zip-compressed"}:
        archive_tree = _build_zip_tree(raw_bytes)

    recommended_tools: list[str] = []
    if sandbox_path:
        recommended_tools.append(str(tool_server_id or "sandbox").strip() or "sandbox")
    if archive_tree:
        recommended_tools.append("archive")
    if text_preview and text_truncated:
        recommended_tools.append("file_search")

    return UploadedFileManifest(
        file_id=upload_id,
        name=clean_name,
        mime=clean_mime,
        size_bytes=len(raw_bytes),
        sha256=sha256,
        sandbox_path=str(sandbox_path or "") or None,
        text_available=bool(text_preview),
        text_preview=text_preview,
        text_total_chars=text_total_chars,
        text_truncated=text_truncated,
        vision_available=bool(is_image and model_supports_vision),
        archive_tree=archive_tree,
        table_preview=table_preview,
        recommended_tools=recommended_tools,
    )
