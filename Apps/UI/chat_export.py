# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import base64
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

from Apps.Data.models import Chat, ChatBranch, MessageAttachment, MessageImage
from Services.upload_storage import load_upload_manifest, resolve_uploaded_file_host_path
from Tools.deep_research.export import safe_filename


# Opaque citation handle formats: S1, source-3, source-abc-3 and c<id>-<n>.
_CITATION_ID = r"(?:S\d+|source-(?:[a-z0-9]+-)?\d+|c[a-z0-9]{3,}-\d+)"
# A bracketed group of one or more comma-separated handles, such as "[S1, S2]".
_CITATION_GROUP_RE = re.compile(rf"\[\s*({_CITATION_ID}(?:\s*,\s*{_CITATION_ID})*)\s*\]", re.I)
# One "Citation handle: ..." block of tool output, up to the next block or the end of the text.
_CITATION_BLOCK_RE = re.compile(
    r"(?:^|\n)Citation handle:\s*\[?([A-Za-z0-9-]+)\]?\s*([\s\S]*?)(?=\nCitation handle:|$)",
    re.I,
)
# Template for matching one "<name>: value" field inside a citation block, up to the next known field.
_FIELD_RE_TEMPLATE = r"(?:^|\n){name}:\s*([\s\S]*?)(?=\n(?:Citation handle|Evidence kind|Title|Domain|URL|Date|Preview|Content):|$)"


# Return the stripped value when it is an http(s) URL with a host, otherwise an empty string.
def _safe_external_url(value: Any) -> str:
    raw = str(value or "").strip()
    try:
        parsed = urlparse(raw)
    except ValueError:
        return ""
    return raw if parsed.scheme in {"http", "https"} and parsed.netloc else ""


# Yield every valid citation handle of a source record, normalized to upper case: first the id-like
# fields, then the aliases.
def _citation_ids(source: dict[str, Any]) -> Iterable[str]:
    for key in (
        "id", "source_id", "sourceId", "citation_id", "citationId",
        "citation_handle", "citationHandle", "handle",
    ):
        value = str(source.get(key) or "").strip().strip("[]").upper()
        if re.fullmatch(_CITATION_ID, value, re.I):
            yield value
    for value in source.get("citation_aliases") or []:
        normalized = str(value or "").strip().strip("[]").upper()
        if re.fullmatch(_CITATION_ID, normalized, re.I):
            yield normalized


# Recursively yield every dict in a transcript that has both an external URL and a citation handle.
def _walk_source_records(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        url = _safe_external_url(value.get("url") or value.get("link") or value.get("href") or value.get("source_url"))
        if url and any(_citation_ids(value)):
            yield value
        for nested in value.values():
            yield from _walk_source_records(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_source_records(nested)


# Yield (citation handle, URL) pairs parsed from plain-text "Citation handle:" blocks of tool output.
def _text_block_sources(value: Any) -> Iterable[tuple[str, str]]:
    if not isinstance(value, str) or "Citation handle:" not in value:
        return
    for match in _CITATION_BLOCK_RE.finditer(value):
        citation_id, block = match.groups()
        url_match = re.search(_FIELD_RE_TEMPLATE.format(name="URL"), block, re.I)
        url = _safe_external_url(url_match.group(1).strip() if url_match else "")
        if url:
            yield citation_id.strip().upper(), url


# Build a map from citation handle to source URL out of the structured source records and the text
# blocks found in the given message transcripts. Later entries overwrite earlier ones.
def citation_registry(transcripts: Iterable[Any]) -> dict[str, str]:
    registry: dict[str, str] = {}
    for transcript in transcripts:
        # Structured source records.
        for source in _walk_source_records(transcript):
            url = _safe_external_url(source.get("url") or source.get("link") or source.get("href") or source.get("source_url"))
            for citation_id in _citation_ids(source):
                registry[citation_id] = url

        # Plain-text citation blocks inside any string of the transcript.
        for value in _walk_strings(transcript):
            for citation_id, url in _text_block_sources(value):
                registry[citation_id] = url
    return registry


# Recursively yield every string found in nested dicts and lists.
def _walk_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _walk_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_strings(nested)


# Turn registered opaque citation handles into ordinary Markdown links. A bracketed group is rewritten
# only when at least one of its handles is in the registry; unknown handles stay as plain "[ID]".
def link_citations(markdown: str, registry: dict[str, str]) -> str:
    # Rewrite one matched bracket group, linking the handles that have a known URL.
    def replace(match: re.Match[str]) -> str:
        rendered: list[str] = []
        changed = False
        for raw_id in match.group(1).split(","):
            citation_id = raw_id.strip().upper()
            url = registry.get(citation_id, "")
            if url:
                rendered.append(f"[{citation_id}]({url})")
                changed = True
            else:
                rendered.append(f"[{citation_id}]")
        return " ".join(rendered) if changed else match.group(0)

    return _CITATION_GROUP_RE.sub(replace, str(markdown or ""))


# Return a safe, archive-unique file name for an attachment: strip directories and unsafe characters,
# fall back to the given name when nothing is left, and add a "-N" suffix on case-insensitive collisions
# with the names already recorded in used (the chosen name is added to it).
def _unique_asset_name(name: str, used: set[str], fallback: str) -> str:
    cleaned = re.sub(r"[^\w .()\-]+", "_", Path(name or fallback).name, flags=re.UNICODE).strip(" .") or fallback
    stem, suffix = Path(cleaned).stem, Path(cleaned).suffix
    candidate = cleaned
    index = 2
    while candidate.casefold() in used:
        candidate = f"{stem}-{index}{suffix}"
        index += 1
    used.add(candidate.casefold())
    return candidate


# Decode a base64 attachment payload, accepting both raw base64 and a "data:" URL.
def _decode_payload(value: str) -> bytes:
    raw = str(value or "")
    if raw.startswith("data:") and "," in raw:
        raw = raw.split(",", 1)[1]
    return base64.b64decode(raw, validate=False)


# Describe a stored attachment or legacy image for the JSON export; archive_path is the file's location
# inside the ZIP.
def _serialize_attachment(record: MessageAttachment | MessageImage, archive_path: str) -> dict[str, Any]:
    payload = {
        "id": record.id,
        "archive_path": archive_path,
        "mime_type": record.mime_type,
        "order": record.order,
    }
    if isinstance(record, MessageAttachment):
        payload.update({
            "kind": record.kind,
            "name": record.name,
            "size_bytes": record.size_bytes,
            "extracted_text": record.extracted_text,
            "extracted_text_ready": record.extracted_text_ready,
        })
    else:
        payload.update({"kind": "image", "name": Path(archive_path).name, "legacy": True})
    return payload


# Collect the distinct uploaded-file ids referenced by "uploaded_file_context" transcript entries, in
# order of first appearance. Ids may be plain values or dicts with "file_id"/"id".
def _uploaded_file_ids(transcript: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for entry in transcript if isinstance(transcript, list) else []:
        if not isinstance(entry, dict) or str(entry.get("type") or entry.get("kind") or "") != "uploaded_file_context":
            continue
        values: list[Any] = []
        for key in ("uploaded_file_ids", "uploaded_files", "file_ids"):
            raw = entry.get(key)
            values.extend(raw if isinstance(raw, list) else ([raw] if raw else []))
        for value in values:
            if isinstance(value, dict):
                value = value.get("file_id") or value.get("id")
            file_id = str(value or "").strip()
            if file_id and file_id not in seen:
                seen.add(file_id)
                result.append(file_id)
    return result


# Build the portable chat export and return it as ZIP bytes: a readable chat.md, a lossless chat.json
# and every attachment, image and uploaded file under attachments/.
def build_chat_archive(chat: Chat) -> bytes:
    # Collect the messages and the accumulators shared by all export parts.
    messages = list(chat.messages.order_by("created_at", "id").prefetch_related("attachments", "images"))
    registry = citation_registry(message.llm_transcript for message in messages)
    used_names: set[str] = set()
    archive_assets: list[tuple[str, bytes]] = []
    json_messages: list[dict[str, Any]] = []
    markdown_parts = [f"# {chat.title}", ""]

    # Record the branch origin, when this chat was created by branching another one.
    try:
        incoming = chat.incoming_branch
    except ChatBranch.DoesNotExist:
        incoming = None
    lineage: dict[str, Any] | None = None
    if incoming is not None:
        lineage = {
            "source_chat_id": str(incoming.source_chat_id),
            "source_chat_title": incoming.source_chat.title,
            "source_message_id": incoming.source_message_id,
            "child_message_id": incoming.child_message_id,
        }
        markdown_parts.extend([
            f"> Branched from [{incoming.source_chat.title}](/chat/{incoming.source_chat_id}/).",
            "",
        ])

    # Render each message into the Markdown text and the JSON structure, collecting its files.
    for message in messages:
        role_label = "User" if message.role == "user" else ("Assistant" if message.role == "assistant" else "System")
        timestamp = message.created_at.isoformat()
        markdown_parts.extend([f"## {role_label} — {timestamp}", "", link_citations(message.content, registry), ""])
        attachment_payloads: list[dict[str, Any]] = []

        # Stored attachments and legacy images: decode each payload into an archive file.
        records: list[MessageAttachment | MessageImage] = [*message.attachments.all(), *message.images.all()]
        for index, record in enumerate(records, start=1):
            if isinstance(record, MessageAttachment):
                fallback = f"attachment-{message.id}-{index}"
                original_name = record.name or fallback
            else:
                suffix = mimetype_suffix(record.mime_type)
                fallback = f"image-{message.id}-{index}{suffix}"
                original_name = fallback
            asset_name = _unique_asset_name(original_name, used_names, fallback)
            archive_path = f"attachments/{asset_name}"
            archive_assets.append((archive_path, _decode_payload(record.data)))
            attachment_payloads.append(_serialize_attachment(record, archive_path))
            markdown_parts.append(f"- Attachment: [{asset_name}]({archive_path})")

        # Uploaded files referenced by the transcript: copy those that still exist on disk.
        for file_id in _uploaded_file_ids(message.llm_transcript):
            manifest = load_upload_manifest(file_id)
            if not manifest:
                continue
            try:
                host_path = resolve_uploaded_file_host_path(manifest)
                file_bytes = host_path.read_bytes()
            except (FileNotFoundError, OSError, ValueError):
                # The upload was deleted or is unreadable; the export simply omits that file.
                continue
            original_name = str(manifest.get("name") or host_path.name or f"upload-{file_id}")
            asset_name = _unique_asset_name(original_name, used_names, f"upload-{file_id}")
            archive_path = f"attachments/{asset_name}"
            archive_assets.append((archive_path, file_bytes))
            attachment_payloads.append({
                "file_id": file_id,
                "kind": "uploaded_file",
                "name": original_name,
                "mime_type": str(manifest.get("mime") or "application/octet-stream"),
                "size_bytes": int(manifest.get("size_bytes") or len(file_bytes)),
                "archive_path": archive_path,
            })
            markdown_parts.append(f"- Uploaded file: [{asset_name}]({archive_path})")

        # Close the attachment list and append the raw transcript as a collapsible block.
        if attachment_payloads:
            markdown_parts.append("")
        if message.llm_transcript:
            markdown_parts.extend([
                "<details>",
                "<summary>Full reasoning and tool transcript (JSON)</summary>",
                "",
                "```json",
                json.dumps(message.llm_transcript, ensure_ascii=False, indent=2),
                "```",
                "",
                "</details>",
                "",
            ])
        json_messages.append({
            "id": message.id,
            "role": message.role,
            "content": message.content,
            "created_at": timestamp,
            "llm_transcript": message.llm_transcript,
            "attachments": attachment_payloads,
        })

    # Assemble the lossless JSON document.
    export_payload = {
        "schema": "aslm.chat.export",
        "schema_version": 1,
        "chat": {
            "id": str(chat.id),
            "title": chat.title,
            "active_tool_slug": chat.active_tool_slug,
            "created_at": chat.created_at.isoformat(),
            "updated_at": chat.updated_at.isoformat(),
            "lineage": lineage,
        },
        "citations": registry,
        "messages": json_messages,
    }

    # Write chat.md, chat.json and the collected files into an in-memory ZIP.
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("chat.md", "\n".join(markdown_parts).rstrip() + "\n")
        archive.writestr("chat.json", json.dumps(export_payload, ensure_ascii=False, indent=2) + "\n")
        for archive_path, payload in archive_assets:
            archive.writestr(archive_path, payload)
    return buffer.getvalue()


# Return the file extension for an image MIME type, or ".bin" when the type is unknown.
def mimetype_suffix(mime_type: str) -> str:
    return {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/gif": ".gif",
        "image/webp": ".webp",
        "image/svg+xml": ".svg",
    }.get(str(mime_type or "").lower(), ".bin")


# Return the download file name of the chat archive: the sanitized chat title with a .zip extension.
def archive_filename(chat: Chat) -> str:
    return safe_filename(chat.title, "zip")
