# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import copy
import hashlib
import ipaddress
import json
import logging
import mimetypes
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin, urlparse

from django.db import transaction
from django.http import FileResponse, HttpResponse, JsonResponse, StreamingHttpResponse
from django.utils import timezone
from django.views.generic import TemplateView

try:
    import requests
except Exception:  # pragma: no cover - optional dependency guard.
    requests = None

try:
    from bs4 import BeautifulSoup
except Exception:  # pragma: no cover - optional dependency guard.
    BeautifulSoup = None

from API import llm_api, mcp as tool_registry
from Apps.API.views import (
    _append_command_directives_to_system_prompt,
    _apply_attachments_to_llm_entry,
    _apply_command_tool_context,
    _apply_uploaded_file_manifests_to_llm_entry,
    _build_chat_history,
    _build_generate_kwargs,
    _build_lms_sync_operation_defaults,
    _build_model_info_payload,
    _build_uploaded_file_context_entry,
    _chat_is_first_user_turn,
    _clear_model_metadata_caches,
    _clear_tool_server_cache,
    _command_directives_for_mode,
    _compose_system_prompt,
    _decode_base64_payload,
    _get_active_engine,
    _get_message_attachments,
    _list_tool_servers_cached,
    _load_model_upload_manifests,
    _message_has_context_compression_summary,
    _normalize_request_attachments,
    _normalize_uploaded_file_ids,
    _parse_active_tool_slugs,
    _parse_chat_command_plan,
    _read_json_request_body,
    _remember_active_model,
    _resolve_request_engine_or_response,
    _resolve_tool_servers,
    _selected_tools_include_sandbox,
    _serialize_message,
    _serialize_tool_call_marker,
    _serialize_tool_result_marker,
    _split_generation_options,
    _stream_chat_response,
    _supported_model_media_kinds,
    _sync_runtime_model_metadata,
    _upload_manifest_file_ids,
    _validate_tool_server_support,
)
from Apps.Data.ollama_presets import (
    activate_ollama_preset,
    create_ollama_preset,
    delete_ollama_preset,
    get_ollama_preset_payload,
    rename_ollama_preset,
    sync_active_ollama_preset,
)
from Apps.Data.lms_presets import (
    activate_lms_preset,
    create_lms_preset,
    delete_lms_preset,
    get_lms_preset_payload,
    rename_lms_preset,
    sync_active_lms_preset,
)
from Apps.Data.openai_presets import (
    activate_openai_preset,
    create_openai_preset,
    delete_openai_preset,
    get_openai_preset_payload,
    normalize_openai_endpoint_url,
    rename_openai_preset,
    sync_active_openai_preset,
)
from Apps.Data.google_genai_presets import (
    activate_google_genai_preset,
    create_google_genai_preset,
    delete_google_genai_preset,
    get_google_genai_preset_payload,
    rename_google_genai_preset,
    sync_active_google_genai_preset,
)
from Apps.Data.models import (
    Chat,
    ChatBranch,
    GoogleGenAiPreset,
    LmsPreset,
    Message,
    MessageAttachment,
    MessageAttachmentKind,
    MessageImage,
    OllamaPreset,
    OpenAiPreset,
)
from Apps.UI.chat_export import archive_filename, build_chat_archive
from Apps.UI import STATIC_CACHE_VERSION
from Tools.system_prompts import is_instant_generation
from Apps.UI.host_theme_bridge import build_host_theme_template_context
from Apps.UI.host_locale_bridge import build_host_locale_template_context
from Services.upload_storage import (
    load_upload_manifest,
    resolve_uploaded_file_host_path,
    save_upload_to_sandbox,
)
from Settings import mcp_json, settings
from Settings import skills as skills_config
from Tools.deep_research import control as deep_research_control
from Tools.deep_research import export as research_export
from Tools.deep_research import service as deep_research_service

logger = logging.getLogger(__name__)

TOOLS_PATH = settings.BASE_DIR / "Tools"
if str(TOOLS_PATH) not in sys.path:
    sys.path.insert(0, str(TOOLS_PATH))

# For saving read results to user_files in the same style as uploads
import re as _re
import hashlib as _hashlib
import uuid as _uuid

# Direct access to WebSearch read_page for URL attachments.
# Performed for every URL attachment regardless of whether "web_search" tool is enabled in UI.
try:
    _mcp_ws_dir = TOOLS_PATH / "mcp-web-search"
    if str(_mcp_ws_dir) not in sys.path:
        sys.path.insert(0, str(_mcp_ws_dir))
    from core.read.service import run_read_page
    from core.search_io_logger import write_search_io_event
except Exception:
    run_read_page = None
    write_search_io_event = lambda event: None
    logger.warning("WebSearch read_page / io_logger not importable — URL attachments will have limited support")


def _save_url_read_result_to_user_files(url: str, content: str) -> None:
    """Save read_page result to Tools/user_files the same way search/read_page results are persisted.

    Writes the markdown content and a manifest so the result is available like other user/search files.
    """
    if not content or str(content).lower().startswith(("error", "warning")):
        return
    try:
        from Services.upload_storage import (
            USER_FILE_MANIFEST_ROOT,
            _safe_scope,
            _write_manifest,
            build_uploaded_file_manifest,
        )
        from pathlib import Path as _PPath
        import json as _json

        safe_hash = _re.sub(r"[^a-fA-F0-9]+", "", _hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]) or "url"
        scope = _safe_scope(safe_hash)
        target_dir = USER_FILE_MANIFEST_ROOT / scope
        target_dir.mkdir(parents=True, exist_ok=True)

        file_id = _uuid.uuid4().hex
        safe_name = _re.sub(r"[^a-zA-Z0-9._-]", "_", url)[:80] + ".md"
        stored_name = f"{file_id}__{safe_name}"
        file_path = target_dir / stored_name
        file_bytes = content.encode("utf-8")
        file_path.write_bytes(file_bytes)

        # Create manifest using the same builder as uploads/user files
        manifest = build_uploaded_file_manifest(
            file_bytes,
            name=safe_name,
            mime="text/markdown",
            sandbox_path=f"/workspace/_sandbox/User/{scope}/{stored_name}",
            model_supports_vision=False,
            file_id=file_id,
            tool_server_id="web_search",
        )
        _write_manifest(manifest)

        # Augment manifest with URL source info (written next to the .manifest.json)
        try:
            mpath = file_path.with_name(file_path.name + ".manifest.json")
            if mpath.exists():
                data = _json.loads(mpath.read_text(encoding="utf-8"))
            else:
                data = {}
            data.update({
                "source_url": url,
                "source_kind": "url_attachment_read_page",
                "text_available": True,
                "text_total_chars": len(content),
            })
            mpath.write_text(_json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    except Exception as e:
        logger.warning("Failed to save URL read result to user_files for %s: %s", url, e)


# Favicon resolution helpers.

# Normalize one domain string for favicon lookup.
def _normalize_favicon_domain(value: str) -> str:
    raw_value = str(value or "").strip()
    if not raw_value:
        return ""

    if not re.match(r"^https?://", raw_value, flags=re.IGNORECASE):
        raw_value = f"https://{raw_value}"

    parsed = urlparse(raw_value)
    host = (parsed.hostname or "").strip().rstrip(".")
    if not host:
        return ""

    try:
        host = host.encode("idna").decode("ascii")
    except UnicodeError:
        return ""

    host = host.lower()
    if len(host) > 253 or not re.match(r"^[a-z0-9.-]+$", host):
        return ""
    if host in {"localhost", "localhost.localdomain"}:
        return ""

    return host


# Return whether the host resolves to a public address.
def _is_public_favicon_host(host: str) -> bool:
    try:
        addresses = socket.getaddrinfo(host, None)
    except OSError:
        return False

    for address in addresses:
        try:
            ip = ipaddress.ip_address(address[4][0])
        except ValueError:
            return False
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_reserved
            or ip.is_unspecified
        ):
            return False

    return bool(addresses)


# Return whether one favicon URL is safe to fetch.
def _favicon_url_is_safe(url: str) -> bool:
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return False
    return _is_public_favicon_host(parsed.hostname)


# Infer the favicon MIME type from headers and bytes.
def _favicon_content_type(response: Any, content: bytes) -> str:
    content_type = str(response.headers.get("content-type", "") or "").split(";")[0].strip().lower()
    if content_type in FAVICON_IMAGE_TYPES:
        return content_type

    probe = content[:256].lstrip()
    if content.startswith(b"\x00\x00\x01\x00"):
        return "image/x-icon"
    if content.startswith(b"\x89PNG"):
        return "image/png"
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if probe.startswith(b"<svg") or b"<svg" in probe[:80]:
        return "image/svg+xml"
    if content.startswith(b"RIFF") and b"WEBP" in content[:16]:
        return "image/webp"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"

    return ""


# Perform one bounded HTTP GET with redirect validation.
def _favicon_safe_get(session: Any, url: str, *, stream: bool = False) -> Any | None:
    current_url = url
    for _redirect_index in range(5):
        if not _favicon_url_is_safe(current_url):
            return None

        try:
            response = session.get(current_url, timeout=(3, 7), allow_redirects=False, stream=stream)
        except Exception:
            return None

        if response.status_code not in {301, 302, 303, 307, 308}:
            return response

        location = response.headers.get("location", "")
        response.close()
        if not location:
            return None
        current_url = urljoin(current_url, location)

    return None


# Download one favicon candidate when it is within size limits.
def _fetch_favicon_candidate(session: Any, url: str) -> tuple[str, bytes] | None:
    response = _favicon_safe_get(session, url, stream=True)
    if response is None:
        return None

    try:
        final_url = str(response.url or url)
        if response.status_code >= 400 or not _favicon_url_is_safe(final_url):
            return None

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=16384):
            if not chunk:
                continue
            total += len(chunk)
            if total > FAVICON_MAX_BYTES:
                return None
            chunks.append(chunk)

        content = b"".join(chunks)
        content_type = _favicon_content_type(response, content)
        if not content_type:
            return None
        return content_type, content
    finally:
        response.close()


# Rank one favicon candidate by rel, size, and format.
def _score_favicon_candidate(candidate: dict[str, str]) -> int:
    rel = candidate.get("rel", "")
    sizes = candidate.get("sizes", "")
    href = candidate.get("href", "")
    score = 0

    if "apple-touch-icon" in rel:
        score += 30
    if rel == "icon" or "shortcut icon" in rel:
        score += 25
    if href.lower().endswith(".svg"):
        score += 20
    if href.lower().endswith(".png"):
        score += 15
    if href.lower().endswith(".ico"):
        score += 5

    match = re.search(r"(\d+)x(\d+)", sizes)
    if match:
        score += min(int(match.group(1)), 256)

    return score


# Collect favicon candidates from HTML, manifests, and fallbacks.
def _collect_favicon_candidates(session: Any, base_url: str) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    final_url = base_url

    if not _favicon_url_is_safe(base_url):
        return candidates

    try:
        response = _favicon_safe_get(session, base_url)
    except Exception:
        response = None

    if response is not None:
        try:
            final_url = str(response.url or base_url)
            if response.status_code < 400 and _favicon_url_is_safe(final_url):
                html = response.text[:400000]
                if BeautifulSoup is not None:
                    soup = BeautifulSoup(html, "html.parser")
                    for tag in soup.find_all("link"):
                        rels = tag.get("rel") or []
                        rel = " ".join(rels).lower() if isinstance(rels, list) else str(rels).lower()
                        if any(icon_rel in rel for icon_rel in FAVICON_ICON_RELS):
                            href = tag.get("href")
                            if href:
                                candidates.append({
                                    "href": urljoin(final_url, href),
                                    "rel": rel,
                                    "sizes": str(tag.get("sizes", "") or ""),
                                })

                    manifest_tag = soup.find("link", rel=lambda value: value and "manifest" in str(value).lower())
                    manifest_href = manifest_tag.get("href") if manifest_tag else ""
                    if manifest_href:
                        manifest_url = urljoin(final_url, manifest_href)
                        if _favicon_url_is_safe(manifest_url):
                            manifest_response = None
                            try:
                                manifest_response = _favicon_safe_get(session, manifest_url)
                                if manifest_response is None or manifest_response.status_code >= 400:
                                    manifest = {}
                                else:
                                    manifest = manifest_response.json()
                            except Exception:
                                manifest = {}
                            finally:
                                if manifest_response is not None:
                                    manifest_response.close()
                            for icon in manifest.get("icons", []) if isinstance(manifest, dict) else []:
                                src = icon.get("src") if isinstance(icon, dict) else ""
                                if src:
                                    candidates.append({
                                        "href": urljoin(manifest_url, src),
                                        "rel": "manifest-icon",
                                        "sizes": str(icon.get("sizes", "") or ""),
                                    })
        finally:
            response.close()

    parsed = urlparse(final_url)
    if parsed.scheme and parsed.netloc:
        origin = f"{parsed.scheme}://{parsed.netloc}"
        for path in FAVICON_ROOT_FALLBACKS:
            candidates.append({"href": urljoin(origin, path), "rel": "root-fallback", "sizes": ""})

    return sorted(candidates, key=_score_favicon_candidate, reverse=True)


# Resolve the best favicon bytes for one domain.
def _resolve_favicon_content(domain: str) -> tuple[str, bytes] | None:
    if requests is None:
        return None
    if not domain or not _is_public_favicon_host(domain):
        return None

    base_url = f"https://{domain}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; ASLM-FaviconResolver/1.0)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/*;q=0.8,*/*;q=0.5",
    }

    with requests.Session() as session:
        session.headers.update(headers)
        seen: set[str] = set()
        for candidate in _collect_favicon_candidates(session, base_url):
            href = candidate.get("href", "")
            if not href or href in seen:
                continue
            seen.add(href)

            resolved = _fetch_favicon_candidate(session, href)
            if resolved:
                return resolved

        for fallback_url in (
            f"https://www.google.com/s2/favicons?domain={domain}&sz=128",
            f"https://icons.duckduckgo.com/ip3/{domain}.ico",
        ):
            resolved = _fetch_favicon_candidate(session, fallback_url)
            if resolved:
                return resolved

    return None


# Return on-disk cache paths for one favicon domain.
def _favicon_cache_paths(domain: str) -> tuple[Path, Path]:
    key = hashlib.sha256(domain.encode("utf-8")).hexdigest()
    return FAVICON_CACHE_DIR / f"{key}.bin", FAVICON_CACHE_DIR / f"{key}.json"


# Read a cached favicon when the entry is still valid.
def _read_favicon_disk_cache(domain: str, now: float) -> tuple[str, bytes] | None:
    content_path, meta_path = _favicon_cache_paths(domain)
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if float(meta.get("expires_at", 0) or 0) <= now:
            return None
        content_type = str(meta.get("content_type", "") or "").strip()
        if content_type not in FAVICON_IMAGE_TYPES:
            return None
        content = content_path.read_bytes()
    except Exception:
        return None

    if not content or len(content) > FAVICON_MAX_BYTES:
        return None
    return content_type, content


# Persist one favicon payload to the disk cache.
def _write_favicon_disk_cache(domain: str, content_type: str, content: bytes, expires_at: float) -> None:
    if not content or len(content) > FAVICON_MAX_BYTES:
        return

    try:
        FAVICON_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        content_path, meta_path = _favicon_cache_paths(domain)
        content_path.write_bytes(content)
        meta_path.write_text(
            json.dumps(
                {
                    "domain": domain,
                    "content_type": content_type,
                    "expires_at": expires_at,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        logger.debug("Failed to write favicon disk cache for %s", domain, exc_info=True)


_favicon_cache_lock = threading.RLock()
_favicon_cache: dict[str, tuple[float, str, bytes]] = {}
FAVICON_CACHE_DIR = settings.BASE_DIR / "Data" / "favicon_cache"
FAVICON_CACHE_TTL_SECONDS = 7 * 24 * 60 * 60
FAVICON_NEGATIVE_CACHE_TTL_SECONDS = 60 * 60
FAVICON_MAX_BYTES = 384 * 1024
FAVICON_IMAGE_TYPES = {
    "image/x-icon",
    "image/vnd.microsoft.icon",
    "image/png",
    "image/jpeg",
    "image/svg+xml",
    "image/webp",
    "image/gif",
}
FAVICON_ICON_RELS = {
    "icon",
    "shortcut icon",
    "apple-touch-icon",
    "apple-touch-icon-precomposed",
    "mask-icon",
    "fluid-icon",
}
FAVICON_ROOT_FALLBACKS = (
    "/favicon.ico",
    "/favicon.png",
    "/apple-touch-icon.png",
    "/apple-touch-icon-precomposed.png",
)


# Build shared template context
def _build_base_context() -> dict[str, Any]:
    runtime_settings = settings.get_runtime_engine_settings()
    engine = _get_active_engine(runtime_settings.get("llm-engine"))
    base = {
        "llm_engine": engine,
        "models": [],
        "engine_options": settings.get_supported_engines(),
        "runtime_settings": runtime_settings,
        "available_tool_servers": _list_tool_servers_cached(engine=engine),
        "chats": Chat.objects.all(),
        "static_cache_version": STATIC_CACHE_VERSION,
    }
    base.update(build_host_theme_template_context())
    base.update(build_host_locale_template_context())
    return base


# Build runtime settings payload
def _build_runtime_settings_payload() -> dict[str, Any]:
    runtime_settings = settings.get_runtime_engine_settings()
    active_engine = _get_active_engine(runtime_settings.get("llm-engine"))
    runtime_settings["llm-engine"] = active_engine
    runtime_settings["active_url"] = runtime_settings["engine_urls"].get(active_engine, "")
    engine_api_keys = runtime_settings.get("engine_api_keys", {})
    runtime_settings["active_has_api_key"] = bool(
        engine_api_keys.get(active_engine, runtime_settings.get("active_has_api_key", False))
    )
    runtime_settings["engine_options"] = settings.get_supported_engines()
    return runtime_settings


# Build chat title
def _build_chat_title(message: str, has_attachments: bool) -> str:
    from Apps.UI.locale_catalog import translate

    if message:
        return message[:30] + ("..." if len(message) > 30 else "")
    if has_attachments:
        return translate("chat.attachmentChat")
    return translate("chat.newChat")


# Resolve chat instance
def _resolve_chat(chat_id: str, user_message: str, attachments: list[dict[str, Any]]) -> Chat:
    if not chat_id:
        return Chat.objects.create(title=_build_chat_title(user_message, bool(attachments)))

    try:
        return Chat.objects.get(id=chat_id)
    except Chat.DoesNotExist as exc:
        raise LookupError("Chat not found") from exc

# Save message images
def _store_message_attachments(message_record: Message, attachments: list[dict[str, Any]]) -> None:
    for attachment in attachments:
        MessageAttachment.objects.create(
            message=message_record,
            kind=attachment["kind"],
            name=attachment["name"],
            mime_type=attachment["mime_type"],
            data=attachment["data"],
            size_bytes=int(attachment.get("size_bytes") or 0),
            order=int(attachment.get("order") or 0),
        )


# Chat page views.

# Render the main chat page.
class MainView(TemplateView):
    template_name = "main/main.html"

    # Build the main page context.
    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context.update(_build_base_context())
        return context


# Chat session APIs.

# Store uploaded files and return UI-facing file cards.
def upload_files_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    uploaded_files = request.FILES.getlist("files") or request.FILES.getlist("file")
    if not uploaded_files:
        return JsonResponse({"error": "No files uploaded"}, status=400)

    scope = request.POST.get("chat_id") or request.POST.get("scope") or "pending"
    model_supports_vision = str(request.POST.get("supports_vision", "")).lower() in {"1", "true", "yes"}
    tool_server_ids = request.POST.getlist("tool_server_ids")
    if not tool_server_ids:
        raw_tool_server_ids = str(request.POST.get("tool_server_ids") or "").strip()
        if raw_tool_server_ids.startswith("["):
            try:
                parsed_tool_server_ids = json.loads(raw_tool_server_ids)
            except json.JSONDecodeError:
                parsed_tool_server_ids = []
            if isinstance(parsed_tool_server_ids, list):
                tool_server_ids = [str(item) for item in parsed_tool_server_ids if str(item).strip()]
    public_files: list[dict[str, Any]] = []
    for uploaded_file in uploaded_files:
        try:
            _manifest, public_payload = save_upload_to_sandbox(
                uploaded_file,
                scope=scope,
                model_supports_vision=model_supports_vision,
                tool_server_ids=tool_server_ids,
            )
            public_files.append(public_payload)
        except Exception as exc:
            logger.exception("Failed to store uploaded file")
            public_files.append({
                "file_id": "",
                "name": str(getattr(uploaded_file, "name", "") or "uploaded-file"),
                "size_bytes": int(getattr(uploaded_file, "size", 0) or 0),
                "status": "error",
                "display_kind": "file",
                "type_label": "Upload failed",
                "error": str(exc),
            })

    return JsonResponse({"files": public_files})


# Build a non-empty Deep Research topic and preserve attached model context.
def _build_deep_research_request_message(
    user_message: str,
    attachments: list[dict[str, Any]],
    upload_manifests: list[dict[str, Any]],
    *,
    engine: str,
    model_info_payload: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    attachment_names = [
        str(item.get("name") or "").strip()
        for item in [*(attachments or []), *(upload_manifests or [])]
        if isinstance(item, dict) and str(item.get("name") or "").strip()
    ]
    attachment_names = list(dict.fromkeys(attachment_names))
    topic = str(user_message or "").strip()
    if not topic:
        listed_names = ", ".join(attachment_names[:6])
        topic = (
            f"Research the user-provided attachments: {listed_names}."
            if listed_names
            else "Research the user-provided attachments."
        )

    request_message: dict[str, Any] = {"role": "user", "content": topic}
    request_message = _apply_attachments_to_llm_entry(request_message, attachments or [])
    request_message = _apply_uploaded_file_manifests_to_llm_entry(
        request_message,
        upload_manifests or [],
        supported_media_kinds=_supported_model_media_kinds(engine, model_info_payload),
    )
    return topic, request_message


# Stream a first-class Deep Research run without routing the request through the
# parent model's tool-selection loop.
def _stream_deep_research_response(
    *,
    chat: Chat,
    assistant_message_record: Message,
    prepared_research: dict[str, Any],
    engine: str,
    model_name: str,
    request_message: dict[str, Any] | None = None,
):
    arguments = dict(prepared_research.get("arguments") or {})
    initial_ui = dict(prepared_research.get("tool_ui") or {})
    alias = "research__deep_research"
    call_id = f"research-{uuid.uuid4().hex}"
    call_entry = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": alias, "arguments": arguments},
            }
        ],
    }
    # Persist the pending artifact before the blocking orchestrator begins so a
    # reload can reconstruct the card and continue polling the durable session.
    assistant_message_record.llm_transcript = [call_entry]
    assistant_message_record.save(update_fields=["llm_transcript"])
    Chat.objects.filter(pk=chat.pk).update(updated_at=timezone.now())
    yield _serialize_tool_call_marker(
        {
            "alias": alias,
            "server_id": "research",
            "server_name": "Deep Research",
            "tool_id": "deep_research",
            "tool_name": "Deep Research",
            "arguments": arguments,
            "tool_ui": initial_ui,
        }
    )

    result: dict[str, Any]
    try:
        result = deep_research_service.run_research(
            arguments,
            {
                "chat_id": str(chat.id),
                "engine": engine,
                "model_name": model_name,
                "request_message": dict(request_message or {}),
            },
        )
        if not isinstance(result, dict):
            raise RuntimeError("Deep Research returned an invalid result.")
    except Exception as exc:
        logger.exception("First-class Deep Research failed")
        session_id = str(arguments.get("session_id") or "")
        snapshot = deep_research_control.read_state(session_id) if session_id else {}
        public_error = str((snapshot or {}).get("error") or exc).strip()
        result = {
            "model_context": public_error,
            "report": "",
            "sources": [],
            "ui": {
                **initial_ui,
                **(snapshot if isinstance(snapshot, dict) else {}),
                "kind": "deep_research",
                "status": "failed",
                "error": public_error,
                "can_approve": False,
                "can_edit": False,
                "can_stop": False,
            },
        }

    tool_ui = result.get("ui") if isinstance(result.get("ui"), dict) else initial_ui
    model_context = str(result.get("model_context") or result.get("report") or "").strip()
    tool_entry = {
        "role": "tool",
        "tool_call_id": call_id,
        "alias": alias,
        "server_id": "research",
        "server_name": "Deep Research",
        "tool_id": "deep_research",
        "tool_name": "Deep Research",
        "content": model_context,
        "structured_content": result,
        "tool_ui": tool_ui,
    }
    yield _serialize_tool_result_marker(
        alias,
        model_context,
        tool_ui=tool_ui,
        structured_content=result,
    )

    status = str(tool_ui.get("status") or "").strip().lower()
    ready_text = "Your research is ready" if status in {"completed", "partial"} else "Research could not be completed"
    visible_entry = {"role": "assistant", "content": ready_text}
    assistant_message_record.content = ready_text
    assistant_message_record.llm_transcript = [call_entry, tool_entry, visible_entry]
    assistant_message_record.save(update_fields=["content", "llm_transcript"])
    Chat.objects.filter(pk=chat.pk).update(updated_at=timezone.now())
    yield f"\n{ready_text}"


# Handle a chat generation request.
def chat_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)

        # Read request payload and validate required inputs.
        user_message = data.get("message", "")
        model_name = data.get("model", "")
        options = data.get("options", {}) or {}
        chat_id = data.get("chat_id", "")
        is_new_chat = not str(chat_id or "").strip()
        attachments = _normalize_request_attachments(data)
        uploaded_file_ids = _normalize_uploaded_file_ids(data)
        deep_research_requested = data.get("deep_research") is True
        engine, engine_error = _resolve_request_engine_or_response(request, data)
        if engine_error is not None:
            return engine_error
        raw_tool_ids = data.get("tool_server_ids") or data.get("tool_server_id") or data.get("tool_id") or []
        if isinstance(raw_tool_ids, str):
            raw_tool_ids = [raw_tool_ids] if raw_tool_ids.strip() else []
        tool_server_ids = [
            str(s).strip()
            for s in raw_tool_ids
            if str(s).strip() and str(s).strip().lower() != "deep_research"
        ]
        command_plan = _parse_chat_command_plan(
            user_message,
            tool_server_ids,
            is_known_tool_id=lambda tool_id: (
                str(tool_id or "").strip().lower() != "deep_research"
                and tool_registry.get_server(tool_id, engine=engine, model_name=model_name) is not None
            ),
        )
        # The stored message keeps the raw slash commands the user typed; the model
        # receives the command_plan.message variant with inline command markers.
        user_message = str(user_message or "")
        llm_user_message = command_plan.message

        if not model_name:
            return JsonResponse({"error": "Missing model parameter"}, status=400)
        if not user_message and not attachments and not uploaded_file_ids:
            return JsonResponse({"error": "Missing message or attachments"}, status=400)

        _remember_active_model(engine, model_name)
        selected_tool_servers = _resolve_tool_servers(
            engine,
            model_name,
            command_plan.tool_server_ids,
        )
        model_info_payload = _build_model_info_payload(engine, model_name, allow_fallback=True)
        _sync_runtime_model_metadata(
            engine,
            model_name,
            model_info_payload,
            source="generation",
            route="/api/chat/",
        )
        has_image_attachments = any(
            attachment.get("kind") == MessageAttachmentKind.IMAGE
            for attachment in attachments
            if isinstance(attachment, dict)
        )
        vision_supported = bool(model_info_payload.get("supports_vision", False))
        vision_source = str(model_info_payload.get("supports_vision_source") or "").strip().lower()
        if has_image_attachments and not vision_supported and vision_source == "explicit_false":
            return JsonResponse(
                {
                    "error": (
                        f"Model '{model_name}' does not support image input on '{engine}'. "
                        "Choose a vision-capable model."
                    )
                },
                status=400,
            )
        _validate_tool_server_support(
            engine,
            model_name,
            [server["id"] for server in selected_tool_servers],
            payload=model_info_payload,
        )
        sandbox_enabled = _selected_tools_include_sandbox(selected_tool_servers)
        upload_manifests = _load_model_upload_manifests(uploaded_file_ids, sandbox_enabled=sandbox_enabled)
        think_value, think_level_value, clean_options = _split_generation_options(
            options,
            think_param_name=str(model_info_payload.get("think_param_name", "think") or "think"),
            think_level_param_name=str(model_info_payload.get("think_level_param_name", "think_level") or "think_level"),
        )
        instant_mode = is_instant_generation(think_value, think_level_value)

        persisted_upload_file_ids = _upload_manifest_file_ids(upload_manifests)
        upload_context_entry = _build_uploaded_file_context_entry(persisted_upload_file_ids)

        # Reuse the existing chat when provided, otherwise create a new one.
        try:
            chat = _resolve_chat(chat_id, user_message, attachments or upload_manifests)
        except LookupError as exc:
            return JsonResponse({"error": str(exc)}, status=404)

        system_prompt = _compose_system_prompt(
            data.get("system_prompt", ""),
            include_skills_baseline=_chat_is_first_user_turn(chat),
            forced_skill_names=command_plan.forced_skill_names,
            instant_mode=instant_mode,
        )
        system_prompt = _append_command_directives_to_system_prompt(
            system_prompt,
            _command_directives_for_mode(command_plan, instant_mode=instant_mode),
        )

        import json as _json
        active_slug = _json.dumps([s["id"] for s in selected_tool_servers], ensure_ascii=False)
        if chat.active_tool_slug != active_slug:
            chat.active_tool_slug = active_slug
            chat.save(update_fields=["active_tool_slug", "updated_at"])

        # Persist the incoming user message and its attachments.
        user_message_record = Message.objects.create(
            chat=chat,
            role="user",
            content=user_message,
            llm_transcript=[upload_context_entry] if upload_context_entry else [],
        )
        _store_message_attachments(user_message_record, attachments)
        Chat.objects.filter(pk=chat.pk).update(updated_at=timezone.now())

        if is_new_chat and settings.get("generate-chat-titles", True):
            generated_title = llm_api.generate_chat_title(
                engine,
                model_name,
                llm_user_message or user_message,
                model_info_payload,
            )
            if generated_title:
                chat.title = generated_title
                chat.save(update_fields=["title", "updated_at"])

        # URL attachments: fetch page content via WebSearch read_page (always, even if tool not selected).
        # This runs AFTER the user message is persisted but BEFORE we build LLM history / call the model.
        # Uses built-in search IO logging and saves result to Tools/user_files like the search tool itself.
        # The markdown is stored in extracted_text so prompt builders pick it up.
        url_atts = [a for a in (attachments or []) if isinstance(a, dict) and (str(a.get("kind") or "").lower() == "url" or a.get("url"))]
        if url_atts:
            for ua in url_atts:
                u = str(ua.get("url") or ua.get("name") or "").strip()
                if not u:
                    continue
                write_search_io_event({
                    "layer": "url_attachment",
                    "phase": "read_page.request",
                    "tool_id": "read_page",
                    "url": u,
                })
                started = time.perf_counter()
                content = ""
                try:
                    # Call via the MCP tool worker (like the search tool itself does).
                    # This ensures correct python env, deps (selectolax etc), browser daemon, and built-in logging.
                    server_file = TOOLS_PATH / "mcp-web-search" / "mcp-server.py"
                    worker_payload = {
                        "tool_id": "read_page",
                        "arguments": {"url": u},
                        "context": {"chat_id": str(chat.id) if chat else ""},
                    }
                    try:
                        from API.mcp import _get_worker_session
                        session = _get_worker_session(server_file)
                        envelope = session.request(
                            "call",
                            worker_payload,
                            timeout_s=60,
                        )
                    except Exception as worker_exc:
                        logger.warning("Persistent worker call failed for read_page, falling back to one-shot: %s", worker_exc)
                        # Fallback one-shot using the worker script
                        python_path = None
                        try:
                            from API.mcp import _get_worker_python
                            python_path = _get_worker_python(server_file)
                        except Exception:
                            pass
                        if python_path is None:
                            python_path = sys.executable
                        import subprocess, json as _json
                        req = _json.dumps(worker_payload, ensure_ascii=False)
                        res = subprocess.run(
                            [str(python_path), str(TOOLS_PATH.parent / "Services" / "tool_worker.py"), "call", str(server_file)],
                            input=req,
                            capture_output=True,
                            text=True,
                            encoding="utf-8",
                            cwd=str(server_file.parent),
                            env=tool_registry._venv_subprocess_env(Path(python_path)),
                            timeout=60,
                            check=False,
                        )
                        stdout = (res.stdout or "").strip()
                        envelope = None
                        for line in reversed(stdout.splitlines()):
                            try:
                                envelope = _json.loads(line)
                                break
                            except Exception:
                                continue
                        if not isinstance(envelope, dict) or not envelope.get("ok"):
                            raise RuntimeError(f"Worker failed: {stdout[:300]} {res.stderr[:300]}")
                        envelope = envelope.get("result") or envelope

                    # Extract content from the read_page payload (same shape as mcp-server _call_read_page)
                    if isinstance(envelope, dict):
                        if "model_context" in envelope:
                            content = envelope.get("model_context") or ""
                        elif "result" in envelope and isinstance(envelope["result"], dict):
                            content = envelope["result"].get("model_context") or ""
                        else:
                            content = str(envelope)
                    else:
                        content = str(envelope)

                    elapsed = time.perf_counter() - started
                    write_search_io_event({
                        "layer": "url_attachment",
                        "phase": "read_page.result",
                        "tool_id": "read_page",
                        "url": u,
                        "result": {"model_context": (content or "")[:2000]},
                        "elapsed_seconds": elapsed,
                    })
                    try:
                        from API.mcp import log_search_tool_io
                        log_search_tool_io(
                            phase="result",
                            tool_event={"tool_id": "read_page", "tool_name": "Read Page (URL attachment)"},
                            arguments={"url": u},
                            result=(content or "")[:1000],
                            elapsed_seconds=elapsed,
                        )
                    except Exception:
                        pass

                    # Save the result to Tools/user_files like the search tool itself does for read_page results.
                    try:
                        _save_url_read_result_to_user_files(u, content)
                    except Exception as save_exc:
                        logger.warning("Failed to persist read_page result to user_files for %s: %s", u, save_exc)

                    # Update DB attachment so it gets injected into prompt context via extracted_text
                    MessageAttachment.objects.filter(
                        message=user_message_record,
                        name=u,
                        kind="url",
                    ).update(extracted_text=content or "", extracted_text_ready=True)
                except Exception as exc:
                    logger.warning("Failed to read_page for URL attachment %s: %s", u, exc)
                    write_search_io_event({
                        "layer": "url_attachment",
                        "phase": "read_page.error",
                        "tool_id": "read_page",
                        "url": u,
                        "error": str(exc),
                    })
                    try:
                        MessageAttachment.objects.filter(message=user_message_record, name=u, kind="url").update(
                            extracted_text=f"Error: could not fetch page content: {exc}", extracted_text_ready=True
                        )
                    except Exception:
                        pass

        # Pre-create the assistant row so the client knows its ID up front
        # and can target it for delete/regenerate without waiting for reload.
        assistant_message_record = Message.objects.create(
            chat=chat,
            role="assistant",
            content="",
            llm_transcript=[],
        )

        if deep_research_requested:
            current_attachments = _get_message_attachments(user_message_record)
            research_topic, research_request_message = _build_deep_research_request_message(
                llm_user_message or user_message,
                current_attachments,
                upload_manifests or [],
                engine=engine,
                model_info_payload=model_info_payload,
            )
            prepared_research = deep_research_service.prepare_research(
                research_topic,
                instructions=str(data.get("deep_research_instructions") or ""),
                max_rounds=data.get("deep_research_max_rounds") or 6,
            )
            generation_id = uuid.uuid4().hex
            response = StreamingHttpResponse(
                _stream_deep_research_response(
                    chat=chat,
                    assistant_message_record=assistant_message_record,
                    prepared_research=prepared_research,
                    engine=engine,
                    model_name=model_name,
                    request_message=research_request_message,
                ),
                content_type="text/plain; charset=utf-8",
            )
            response["X-Chat-ID"] = str(chat.id)
            if is_new_chat:
                response["X-Chat-Title"] = quote(str(chat.title), safe="")
            response["X-LLM-Engine"] = engine
            response["X-User-Message-ID"] = str(user_message_record.id)
            response["X-Assistant-Message-ID"] = str(assistant_message_record.id)
            response["X-Generation-ID"] = generation_id
            response["X-Deep-Research-Session-ID"] = str(
                prepared_research.get("arguments", {}).get("session_id") or ""
            )
            return response

        # Rebuild the message history expected by the selected backend.
        llm_messages, compression_event = _build_chat_history(
            chat,
            user_message_record,
            llm_user_message,
            system_prompt,
            engine,
            model_name,
            model_info_payload,
            upload_manifests,
            sandbox_enabled=sandbox_enabled,
        )
        # Build the final generation payload for the adapter layer.
        external_tool_context = {}
        _apply_command_tool_context(
            external_tool_context,
            command_plan,
            instant_mode=instant_mode,
        )
        generate_kwargs = _build_generate_kwargs(
            engine,
            model_name,
            llm_messages,
            think_value,
            think_level_value,
            clean_options,
            str(chat.id),
            selected_tool_servers,
            think_param_name=str(model_info_payload.get("think_param_name", "think") or "think"),
            think_level_param_name=str(model_info_payload.get("think_level_param_name", "think_level") or "think_level"),
            sync_operation_defaults=_build_lms_sync_operation_defaults(
                engine,
                model_info_payload,
                think_value,
                think_level_value,
            ),
            external_tool_context=external_tool_context,
            instant_mode=instant_mode,
        )

        generation_id = uuid.uuid4().hex
        response = StreamingHttpResponse(
            _stream_chat_response(
                engine,
                generate_kwargs,
                generation_id,
                chat=chat,
                assistant_message_record=assistant_message_record,
                session_id=str(chat.id),
                compression_event=compression_event,
                model_info_payload=model_info_payload,
                system_prompt=system_prompt,
                current_user_message_id=user_message_record.id,
            ),
            content_type="text/plain; charset=utf-8",
        )
        response["X-Chat-ID"] = str(chat.id)
        if is_new_chat:
            response["X-Chat-Title"] = quote(str(chat.title), safe="")
        response["X-LLM-Engine"] = engine
        response["X-User-Message-ID"] = str(user_message_record.id)
        response["X-Assistant-Message-ID"] = str(assistant_message_record.id)
        response["X-Generation-ID"] = generation_id
        return response

    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Unhandled exception in chat_api")
        return JsonResponse({"error": str(exc)}, status=500)


# Message and chat management APIs.

def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


# Shared file allowed roots.
def _shared_file_allowed_roots() -> list[Path]:
    roots = [
        settings.BASE_DIR / "Tools" / "mcp-sandbox" / "_sandbox",
    ]

    sandbox_host_workspace = os.getenv("SANDBOX_HOST_WORKSPACE", "").strip()
    sandbox_default_task_dir = os.getenv("SANDBOX_DEFAULT_TASK_DIR", "_sandbox").strip() or "_sandbox"
    if sandbox_host_workspace:
        roots.append(Path(sandbox_host_workspace) / sandbox_default_task_dir)

    normalized: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            continue
        key = str(resolved).lower()
        if key not in seen:
            normalized.append(resolved)
            seen.add(key)
    return normalized


# Resolve shared file path.
def _resolve_shared_file_path(raw_path: str) -> Path:
    cleaned = str(raw_path or "").strip().strip('"')
    if not cleaned or "\x00" in cleaned:
        raise ValueError("Missing shared file path.")

    raw = Path(cleaned)
    allowed_roots = _shared_file_allowed_roots()
    sandbox_root = settings.BASE_DIR / "Tools" / "mcp-sandbox" / "_sandbox"
    normalized_cleaned = cleaned.replace("\\", "/").strip("/")
    normalized_parts = [part for part in normalized_cleaned.split("/") if part not in {"", "."}]
    mapped_parts = normalized_parts

    if normalized_parts and normalized_parts[0] in {"workspace", "home"}:
        if len(normalized_parts) >= 2 and normalized_parts[0] == "workspace" and normalized_parts[1] == "_sandbox":
            mapped_parts = normalized_parts[2:]
        elif len(normalized_parts) >= 2 and normalized_parts[0] == "home" and normalized_parts[1] == "sandbox":
            mapped_parts = normalized_parts[2:]

    if mapped_parts and mapped_parts[0] == "_sandbox":
        mapped_parts = mapped_parts[1:]

    if mapped_parts and mapped_parts[0].lower() == "user":
        mapped_relative = Path(*mapped_parts)
    else:
        mapped_relative = Path("User") / Path(*mapped_parts) if mapped_parts else Path("User")

    candidate_set: list[Path] = []
    seen: set[str] = set()

    # Track one shared-file path candidate without duplicates.
    def push_candidate(path: Path) -> None:
        key = str(path).lower()
        if key not in seen:
            seen.add(key)
            candidate_set.append(path)

    if raw.is_absolute():
        push_candidate(raw)
    push_candidate(settings.BASE_DIR / cleaned)
    push_candidate(sandbox_root / cleaned)
    push_candidate(sandbox_root / mapped_relative)
    for root in allowed_roots:
        push_candidate(root / cleaned)
        push_candidate(root / mapped_relative)
        push_candidate(root / Path(*mapped_parts) if mapped_parts else root)

    for candidate in candidate_set:
        try:
            resolved = candidate.resolve()
        except OSError:
            continue
        if not resolved.is_file():
            continue
        if any(_path_is_within(resolved, root) for root in allowed_roots):
            return resolved

    raise FileNotFoundError("Shared file not found or not allowed.")


# Return the local path for one uploaded file manifest.
def _resolve_uploaded_file_content_path(manifest: dict[str, Any]) -> Path:
    return resolve_uploaded_file_host_path(manifest)


MEDIA_RANGE_CHUNK_BYTES = 64 * 1024 * 1024
MEDIA_STREAM_READ_BYTES = 1024 * 1024


# Range not satisfiable response.
def _range_not_satisfiable_response(file_size: int) -> HttpResponse:
    resp = HttpResponse(status=416)
    resp["Content-Range"] = "bytes */" + str(max(0, int(file_size or 0)))
    resp["Accept-Ranges"] = "bytes"
    return resp


# Return one satisfiable byte range, including suffix ranges.
def _parse_single_byte_range(range_header: str, file_size: int) -> tuple[int, int] | None:
    if file_size <= 0:
        return None
    range_match = re.fullmatch(r"bytes=(\d*)-(\d*)", str(range_header or "").strip())
    if not range_match:
        return None

    raw_start, raw_end = range_match.group(1), range_match.group(2)
    if not raw_start and not raw_end:
        return None

    if not raw_start:
        suffix_length = int(raw_end)
        if suffix_length <= 0:
            return None
        start = max(file_size - suffix_length, 0)
        end = file_size - 1
        return start, end

    start = int(raw_start)
    if start >= file_size:
        return None

    if raw_end:
        end = min(int(raw_end), file_size - 1)
    else:
        end = min(start + MEDIA_RANGE_CHUNK_BYTES - 1, file_size - 1)
    if start > end:
        return None
    return start, end


# Stream a local file with HTTP Range support for media playback.
def _stream_local_file_response(
    request,
    target: Path,
    *,
    mime_type: str,
    safe_name: str,
    disposition: str = "inline",
):
    file_size = target.stat().st_size
    range_header = request.META.get("HTTP_RANGE", "").strip()
    is_head = request.method == "HEAD"

    if range_header:
        parsed_range = _parse_single_byte_range(range_header, file_size)
        if parsed_range is None:
            return _range_not_satisfiable_response(file_size)
        start, end = parsed_range
        chunk_size = end - start + 1
        if is_head:
            response = HttpResponse(status=206, content_type=mime_type)
        else:
            fh = target.open("rb")
            fh.seek(start)

            # Yield file bytes for one HTTP Range response chunk.
            def _range_iter(fh, remaining):
                try:
                    while remaining > 0:
                        data = fh.read(min(MEDIA_STREAM_READ_BYTES, remaining))
                        if not data:
                            break
                        remaining -= len(data)
                        yield data
                finally:
                    fh.close()

            response = StreamingHttpResponse(_range_iter(fh, chunk_size), status=206, content_type=mime_type)
        response["Content-Range"] = "bytes " + str(start) + "-" + str(end) + "/" + str(file_size)
        response["Content-Length"] = str(chunk_size)
    else:
        if is_head:
            response = HttpResponse(content_type=mime_type)
        else:
            response = FileResponse(target.open("rb"), content_type=mime_type)
        response["Content-Length"] = str(file_size)

    response["Accept-Ranges"] = "bytes"
    response["Cache-Control"] = "private, max-age=300"
    response["Content-Disposition"] = f'{disposition}; filename="{safe_name}"'
    return response


# Return uploaded file bytes on demand.
def uploaded_file_content_api(request, file_id: str):
    if request.method not in {"GET", "HEAD"}:
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        manifest = load_upload_manifest(file_id)
        if not manifest:
            return JsonResponse({"error": "Uploaded file not found"}, status=404)
        target = _resolve_uploaded_file_content_path(manifest)
        safe_name = re.sub(r"[\"\\\r\n]", "_", Path(str(manifest.get("name") or target.name)).name or "download")
        mime_type = str(manifest.get("mime") or mimetypes.guess_type(str(target))[0] or "application/octet-stream")
        return _stream_local_file_response(request, target, mime_type=mime_type, safe_name=safe_name)
    except FileNotFoundError:
        return JsonResponse({"error": "Uploaded file not found"}, status=404)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Failed to stream uploaded file")
        return JsonResponse({"error": str(exc)}, status=500)


# Download a model-shared local file after validating its workspace path.
def shared_file_download_api(request):
    if request.method not in {"GET", "HEAD"}:
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        target = _resolve_shared_file_path(request.GET.get("path", ""))
        safe_name = re.sub(r'["\\\r\n]', "_", Path(request.GET.get("name") or target.name).name or "download")
        mime_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        disposition = "inline" if str(request.GET.get("preview") or "").lower() in {"1", "true", "yes"} else "attachment"
        return _stream_local_file_response(
            request,
            target,
            mime_type=mime_type,
            safe_name=safe_name,
            disposition=disposition,
        )
    except FileNotFoundError:
        return JsonResponse({"error": "Shared file not found"}, status=404)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Failed to download shared file")
        return JsonResponse({"error": str(exc)}, status=500)


# Open the host file manager and select one validated shared file.
def _reveal_file_in_file_manager(target: Path) -> None:
    if sys.platform == "win32":
        command = ["explorer.exe", f"/select,{target}"]
    elif sys.platform == "darwin":
        command = ["open", "-R", str(target)]
    else:
        command = ["xdg-open", str(target.parent)]

    subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
    )


# Reveal a model-shared local file after applying the download path policy.
def shared_file_reveal_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        payload = _read_json_request_body(request)
        target = _resolve_shared_file_path(str(payload.get("path") or ""))
        _reveal_file_in_file_manager(target)
        return JsonResponse({"ok": True})
    except FileNotFoundError:
        return JsonResponse({"error": "Shared file not found"}, status=404)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Failed to reveal shared file")
        return JsonResponse({"error": str(exc)}, status=500)


# Download a completed Deep Research report in a portable format. The durable
# report already contains resolved Markdown links for every citation handle.
def deep_research_export_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)
    try:
        data = _read_json_request_body(request)
        session_id = str(data.get("session_id") or "").strip()
        export_format = str(data.get("format") or "md").strip().lower()
        if export_format not in {"md", "pdf", "docx"}:
            return JsonResponse({"error": "Unsupported research export format."}, status=400)
        snapshot = deep_research_control.read_state(session_id)
        fallback_report = str(data.get("report") or "").strip()
        # Older chats can retain a complete report in their transcript after
        # the retention policy removes the durable control snapshot.
        report = str(
            (snapshot or {}).get("report")
            or (snapshot or {}).get("model_context")
            or fallback_report
        ).strip()
        if not report:
            return JsonResponse({"error": "The research report is not ready yet."}, status=409)
        title = str((snapshot or {}).get("topic") or data.get("title") or "Research report").strip()
        if export_format == "md":
            payload = research_export.markdown_bytes(report)
            content_type = "text/markdown; charset=utf-8"
        elif export_format == "docx":
            payload = research_export.docx_bytes(report, title)
            content_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            payload = research_export.pdf_bytes(report, title)
            content_type = "application/pdf"
        response = HttpResponse(payload, content_type=content_type)
        response["Content-Disposition"] = (
            f'attachment; filename="{research_export.safe_filename(title, export_format)}"'
        )
        return response
    except deep_research_control.InvalidResearchSession as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Failed to export Deep Research report")
        return JsonResponse({"error": str(exc)}, status=500)


# Read or control one durable Deep Research session.
def deep_research_control_api(request):
    """Approve, revise, stop, or inspect an isolated Deep Research run."""

    try:
        if request.method == "GET":
            session_id = str(request.GET.get("session_id") or "").strip()
            if not session_id:
                return JsonResponse({"error": "Missing session_id."}, status=400)
            state = deep_research_control.read_state(session_id)
            if not state:
                return JsonResponse({"error": "Deep research session was not found."}, status=404)
            return JsonResponse({"ok": True, "state": state})

        if request.method != "POST":
            return JsonResponse({"error": "Invalid request method"}, status=405)

        data = _read_json_request_body(request)
        session_id = str(data.get("session_id") or "").strip()
        action = str(data.get("action") or "").strip().lower()
        if not session_id:
            return JsonResponse({"error": "Missing session_id."}, status=400)
        if action not in {"approve", "revise", "cancel"}:
            return JsonResponse({"error": "Unsupported deep research action."}, status=400)

        current_state = deep_research_control.read_state(session_id)
        if not current_state:
            return JsonResponse({"error": "Deep research session was not found."}, status=404)
        if action == "approve" and current_state.get("can_approve") is False:
            return JsonResponse(
                {"error": "This research plan is not awaiting approval.", "state": current_state},
                status=409,
            )
        if action == "revise" and current_state.get("can_edit") is False:
            return JsonResponse(
                {"error": "This research plan can no longer be edited.", "state": current_state},
                status=409,
            )

        expected_version = data.get("expected_plan_version")
        if expected_version in (None, ""):
            expected_plan_version = None
        else:
            try:
                expected_plan_version = int(expected_version)
            except (TypeError, ValueError):
                return JsonResponse({"error": "expected_plan_version must be an integer."}, status=400)

        payload: dict[str, Any] = {}
        if action in {"approve", "revise"} and data.get("plan") is not None:
            plan = str(data.get("plan") or "").strip()
            if len(plan) > 20_000:
                return JsonResponse({"error": "Research plan is too long."}, status=400)
            if action == "revise" and not plan:
                return JsonResponse({"error": "A revised plan cannot be empty."}, status=400)
            if plan:
                payload["plan"] = plan

        result = deep_research_control.submit_command(
            session_id,
            action,
            payload=payload,
            expected_plan_version=expected_plan_version,
        )
        if action == "cancel" and result.get("accepted") and not result.get("terminal"):
            # Persist the user's intent before waiting for the worker to reach a
            # cooperative cancellation point. This keeps the Stop control gone
            # across polls and page reloads instead of briefly reviving it from
            # the last running snapshot.
            stopping_state = deep_research_control.update_state(
                session_id,
                status="stopping",
                phase="stopping",
                latest_action="Stopping research safely",
                can_approve=False,
                can_edit=False,
                can_stop=False,
                stop_requested_at=time.time(),
            )
            result = {**result, "state": stopping_state}
            # The durable command remains the cooperative cancellation path.
            # When this Django process also owns the active tool worker, stop
            # that exact process immediately so a hung provider/read cannot
            # leave the card in "Stopping" until its network timeout.
            interrupted = tool_registry.abort_active_research(session_id)
            if interrupted:
                cancelled_at = time.time()
                terminal_state = deep_research_control.update_state(
                    session_id,
                    status="cancelled",
                    phase="cancelled",
                    latest_action="Research stopped by user",
                    can_approve=False,
                    can_edit=False,
                    can_stop=False,
                    cancelled_at=cancelled_at,
                    completed_at=cancelled_at,
                )
                result = {
                    **result,
                    "terminal": True,
                    "state": terminal_state,
                    "interrupted_active_tools": interrupted,
                }
        return JsonResponse({"ok": True, **result}, status=202)
    except deep_research_control.InvalidResearchSession as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except FileNotFoundError as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except RuntimeError as exc:
        if str(exc) == "STALE_PLAN_VERSION":
            state = deep_research_control.read_state(
                str(request.GET.get("session_id") or "")
                if request.method == "GET"
                else str(data.get("session_id") or "")
            )
            return JsonResponse(
                {"error": "The research plan changed. Review the latest revision.", "state": state},
                status=409,
            )
        if str(exc) == "ACTION_NOT_AVAILABLE":
            state = deep_research_control.read_state(str(data.get("session_id") or ""))
            return JsonResponse(
                {"error": "That research action is no longer available.", "state": state},
                status=409,
            )
        raise
    except Exception as exc:
        logger.exception("Failed to control Deep Research")
        return JsonResponse({"error": str(exc)}, status=500)


# Return stored attachment bytes on demand.
def attachment_content_api(request, record_type: str, attachment_id: int):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        normalized_type = str(record_type or "").strip().lower()
        if normalized_type == "attachment":
            attachment = MessageAttachment.objects.get(id=attachment_id)
            mime_type = attachment.mime_type
            name = attachment.name or f"attachment-{attachment.id}"
            encoded = attachment.data
        elif normalized_type == "image":
            attachment = MessageImage.objects.get(id=attachment_id)
            mime_type = attachment.mime_type
            name = f"image-{attachment.order + 1}"
            encoded = attachment.data
        else:
            return JsonResponse({"error": "Unknown attachment type"}, status=404)

        safe_name = re.sub(r'["\\\r\n]', "_", Path(name).name or "attachment")
        response = HttpResponse(_decode_base64_payload(encoded), content_type=mime_type)
        response["Cache-Control"] = "private, max-age=3600"
        response["Content-Disposition"] = f'inline; filename="{safe_name}"'
        return response
    except (MessageAttachment.DoesNotExist, MessageImage.DoesNotExist):
        return JsonResponse({"error": "Attachment not found"}, status=404)
    except Exception as exc:
        logger.exception("Failed to load attachment %s/%s", record_type, attachment_id)
        return JsonResponse({"error": str(exc)}, status=500)


# Delete a specific message by ID.
def delete_message_api(request, message_id):
    if request.method != "DELETE":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        msg = Message.objects.get(id=message_id)
        msg.delete()
        return JsonResponse({"ok": True})
    except Message.DoesNotExist:
        return JsonResponse({"error": "Message not found"}, status=404)
    except Exception as exc:
        logger.exception("Failed to delete message %s", message_id)
        return JsonResponse({"error": str(exc)}, status=500)


# Replace one user message and discard the now-stale continuation.
def edit_message_api(request, message_id):
    if request.method != "PATCH":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        content = str(data.get("content", ""))
        with transaction.atomic():
            message = Message.objects.select_related("chat").get(id=message_id)
            if message.role != "user":
                return JsonResponse({"error": "Only user messages can be edited"}, status=400)
            if not content.strip() and not message.attachments.exists() and not message.images.exists():
                return JsonResponse({"error": "Message content is required"}, status=400)

            later_ids = list(
                message.chat.messages.filter(
                    created_at__gt=message.created_at,
                ).values_list("id", flat=True)
            )
            # IDs are a stable tie-breaker for records created in the same timestamp tick.
            later_ids.extend(
                message.chat.messages.filter(created_at=message.created_at, id__gt=message.id)
                .values_list("id", flat=True)
            )
            if later_ids:
                Message.objects.filter(id__in=set(later_ids)).delete()
            message.content = content
            message.save(update_fields=["content"])
            Chat.objects.filter(id=message.chat_id).update(updated_at=timezone.now())

        return JsonResponse({"ok": True, "message": _serialize_message(message), "deleted_message_ids": later_ids})
    except Message.DoesNotExist:
        return JsonResponse({"error": "Message not found"}, status=404)
    except Exception as exc:
        logger.exception("Failed to edit message %s", message_id)
        return JsonResponse({"error": str(exc)}, status=500)


# Clone one conversation through a selected message and link both sides.
def branch_message_api(request, message_id):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        with transaction.atomic():
            source_message = Message.objects.select_related("chat").get(id=message_id)
            source_chat = source_message.chat
            source_messages = list(
                source_chat.messages.filter(created_at__lt=source_message.created_at)
                .prefetch_related("attachments", "images")
            )
            source_messages.extend(
                source_chat.messages.filter(created_at=source_message.created_at, id__lte=source_message.id)
                .prefetch_related("attachments", "images")
            )
            source_messages.sort(key=lambda item: (item.created_at, item.id))

            child_title = f"Branch: {source_chat.title}"[:255]
            child_chat = Chat.objects.create(
                title=child_title,
                active_tool_slug=source_chat.active_tool_slug,
            )
            cloned_by_source_id: dict[int, Message] = {}
            for original in source_messages:
                cloned = Message.objects.create(
                    chat=child_chat,
                    role=original.role,
                    content=original.content,
                    llm_transcript=copy.deepcopy(original.llm_transcript),
                )
                Message.objects.filter(id=cloned.id).update(created_at=original.created_at)
                cloned.created_at = original.created_at
                cloned_by_source_id[original.id] = cloned

                MessageAttachment.objects.bulk_create([
                    MessageAttachment(
                        message=cloned,
                        kind=attachment.kind,
                        name=attachment.name,
                        mime_type=attachment.mime_type,
                        data=attachment.data,
                        size_bytes=attachment.size_bytes,
                        extracted_text=attachment.extracted_text,
                        extracted_text_ready=attachment.extracted_text_ready,
                        order=attachment.order,
                    )
                    for attachment in original.attachments.all()
                ])
                MessageImage.objects.bulk_create([
                    MessageImage(
                        message=cloned,
                        mime_type=image.mime_type,
                        data=image.data,
                        order=image.order,
                    )
                    for image in original.images.all()
                ])

            child_message = cloned_by_source_id[source_message.id]
            ChatBranch.objects.create(
                source_chat=source_chat,
                source_message=source_message,
                child_chat=child_chat,
                child_message=child_message,
            )

        return JsonResponse({
            "ok": True,
            "chat_id": str(child_chat.id),
            "title": child_chat.title,
            "source_chat_id": str(source_chat.id),
            "source_chat_title": source_chat.title,
        }, status=201)
    except Message.DoesNotExist:
        return JsonResponse({"error": "Message not found"}, status=404)
    except Exception as exc:
        logger.exception("Failed to branch from message %s", message_id)
        return JsonResponse({"error": str(exc)}, status=500)


# Delete the last assistant reply for regeneration.
def delete_last_assistant_api(request, chat_id):
    if request.method != "DELETE":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        chat = Chat.objects.get(id=chat_id)
        messages = list(chat.messages.order_by("created_at"))
        if not messages:
            return JsonResponse({"error": "No messages"}, status=400)

        last = messages[-1]
        if last.role != "assistant":
            return JsonResponse({"error": "Last message is not from assistant"}, status=400)

        last.delete()

        # Find the preceding user message to replay.
        user_message = next((m for m in reversed(messages[:-1]) if m.role == "user"), None)
        if not user_message:
            return JsonResponse({"ok": True, "user_message": None})

        attachments = _get_message_attachments(user_message)
        return JsonResponse({
            "ok": True,
            "user_message": {
                "content": user_message.content,
                "attachments": attachments,
                "images": [item["data_url"] for item in attachments if item.get("kind") == MessageAttachmentKind.IMAGE],
            }
        })
    except Chat.DoesNotExist:
        return JsonResponse({"error": "Chat not found"}, status=404)
    except Exception as exc:
        logger.exception("Failed to delete last assistant message for chat %s", chat_id)
        return JsonResponse({"error": str(exc)}, status=500)


# Regenerate the assistant reply for an existing user message without duplicating it.
def regenerate_chat_api(request, chat_id):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)

        model_name = data.get("model", "")
        options = data.get("options", {}) or {}
        engine, engine_error = _resolve_request_engine_or_response(request, data)
        if engine_error is not None:
            return engine_error
        raw_tool_ids = data.get("tool_server_ids") or data.get("tool_server_id") or data.get("tool_id") or []
        if isinstance(raw_tool_ids, str):
            raw_tool_ids = [raw_tool_ids] if raw_tool_ids.strip() else []
        tool_server_ids = [str(s).strip() for s in raw_tool_ids if str(s).strip()]

        if not model_name:
            return JsonResponse({"error": "Missing model parameter"}, status=400)

        try:
            chat = Chat.objects.get(id=chat_id)
        except Chat.DoesNotExist:
            return JsonResponse({"error": "Chat not found"}, status=404)

        target_id = data.get("user_message_id")
        ordered = list(chat.messages.order_by("created_at", "id"))
        if target_id:
            user_message_record = next(
                (m for m in ordered if m.role == "user" and str(m.id) == str(target_id)),
                None,
            )
        else:
            user_message_record = next((m for m in reversed(ordered) if m.role == "user"), None)

        if user_message_record is None:
            return JsonResponse({"error": "No user message to regenerate"}, status=400)

        # Drop every message that came after the targeted user turn, including
        # the assistant reply we are about to replace.
        preserve_context_compression = bool(data.get("preserve_context_compression"))
        for message in ordered:
            if message.created_at > user_message_record.created_at or (
                message.created_at == user_message_record.created_at and message.id > user_message_record.id
            ):
                if preserve_context_compression and _message_has_context_compression_summary(message):
                    continue
                message.delete()

        _remember_active_model(engine, model_name)
        selected_tool_servers = _resolve_tool_servers(engine, model_name, tool_server_ids)
        model_info_payload = _build_model_info_payload(engine, model_name, allow_fallback=True)
        _sync_runtime_model_metadata(
            engine,
            model_name,
            model_info_payload,
            source="regeneration",
            route="/api/chat/regenerate/",
        )
        _validate_tool_server_support(
            engine,
            model_name,
            [server["id"] for server in selected_tool_servers],
            payload=model_info_payload,
        )
        sandbox_enabled = _selected_tools_include_sandbox(selected_tool_servers)

        import json as _json
        active_slug = _json.dumps([s["id"] for s in selected_tool_servers], ensure_ascii=False)
        if chat.active_tool_slug != active_slug:
            chat.active_tool_slug = active_slug
            chat.save(update_fields=["active_tool_slug", "updated_at"])

        think_value, think_level_value, clean_options = _split_generation_options(
            options,
            think_param_name=str(model_info_payload.get("think_param_name", "think") or "think"),
            think_level_param_name=str(model_info_payload.get("think_level_param_name", "think_level") or "think_level"),
        )
        instant_mode = is_instant_generation(think_value, think_level_value)
        system_prompt = _compose_system_prompt(
            data.get("system_prompt", ""),
            include_skills_baseline=False,
            instant_mode=instant_mode,
        )
        llm_messages, compression_event = _build_chat_history(
            chat,
            user_message_record,
            user_message_record.content,
            system_prompt,
            engine,
            model_name,
            model_info_payload,
            sandbox_enabled=sandbox_enabled,
        )

        generate_kwargs = _build_generate_kwargs(
            engine,
            model_name,
            llm_messages,
            think_value,
            think_level_value,
            clean_options,
            str(chat.id),
            selected_tool_servers,
            think_param_name=str(model_info_payload.get("think_param_name", "think") or "think"),
            think_level_param_name=str(model_info_payload.get("think_level_param_name", "think_level") or "think_level"),
            sync_operation_defaults=_build_lms_sync_operation_defaults(
                engine,
                model_info_payload,
                think_value,
                think_level_value,
            ),
            instant_mode=instant_mode,
        )

        assistant_message_record = Message.objects.create(
            chat=chat,
            role="assistant",
            content="",
            llm_transcript=[],
        )

        generation_id = uuid.uuid4().hex
        response = StreamingHttpResponse(
            _stream_chat_response(
                engine,
                generate_kwargs,
                generation_id,
                chat=chat,
                assistant_message_record=assistant_message_record,
                session_id=str(chat.id),
                compression_event=compression_event,
                model_info_payload=model_info_payload,
                system_prompt=system_prompt,
                current_user_message_id=user_message_record.id,
            ),
            content_type="text/plain; charset=utf-8",
        )
        response["X-Chat-ID"] = str(chat.id)
        response["X-LLM-Engine"] = engine
        response["X-User-Message-ID"] = str(user_message_record.id)
        response["X-Assistant-Message-ID"] = str(assistant_message_record.id)
        response["X-Generation-ID"] = generation_id
        return response

    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Unhandled exception in regenerate_chat_api")
        return JsonResponse({"error": str(exc)}, status=500)


# Rename a chat thread.
def rename_chat_api(request, chat_id):
    if request.method != "PATCH":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        title = str(data.get("title", "")).strip()
        if not title:
            return JsonResponse({"error": "Title is required"}, status=400)

        chat = Chat.objects.get(id=chat_id)
        chat.title = title
        chat.save(update_fields=["title"])
        return JsonResponse({"ok": True, "title": chat.title})
    except Chat.DoesNotExist:
        return JsonResponse({"error": "Chat not found"}, status=404)
    except Exception as exc:
        logger.exception("Failed to rename chat %s", chat_id)
        return JsonResponse({"error": str(exc)}, status=500)


# Delete a chat thread and all its messages.
def delete_chat_api(request, chat_id):
    if request.method != "DELETE":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        chat = Chat.objects.get(id=chat_id)
        chat.delete()
        return JsonResponse({"ok": True})
    except Chat.DoesNotExist:
        return JsonResponse({"error": "Chat not found"}, status=404)
    except Exception as exc:
        logger.exception("Failed to delete chat %s", chat_id)
        return JsonResponse({"error": str(exc)}, status=500)


# Download a lossless portable archive of one entire chat.
def export_chat_api(request, chat_id):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        chat = Chat.objects.get(id=chat_id)
        response = HttpResponse(build_chat_archive(chat), content_type="application/zip")
        response["Content-Disposition"] = f'attachment; filename="{archive_filename(chat)}"'
        return response
    except Chat.DoesNotExist:
        return JsonResponse({"error": "Chat not found"}, status=404)
    except Exception as exc:
        logger.exception("Failed to export chat %s", chat_id)
        return JsonResponse({"error": str(exc)}, status=500)


# Load persisted messages for a chat thread.
def load_chat_api(request, chat_id):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        chat = Chat.objects.get(id=chat_id)
        messages = chat.messages.order_by("created_at", "id").prefetch_related("attachments", "images")
        message_records = list(messages)
        payload = [_serialize_message(message, include_attachment_data=False) for message in message_records]
        payload_by_id = {item["id"]: item for item in payload}

        for branch in chat.outgoing_branches.select_related("child_chat").all():
            if branch.source_message_id in payload_by_id:
                payload_by_id[branch.source_message_id].setdefault("branch_links", []).append({
                    "direction": "to_branch",
                    "chat_id": str(branch.child_chat_id),
                    "title": branch.child_chat.title,
                })
        try:
            incoming_branch = chat.incoming_branch
        except ChatBranch.DoesNotExist:
            incoming_branch = None
        if incoming_branch and incoming_branch.child_message_id in payload_by_id:
            payload_by_id[incoming_branch.child_message_id].setdefault("branch_links", []).append({
                "direction": "to_origin",
                "chat_id": str(incoming_branch.source_chat_id),
                "title": incoming_branch.source_chat.title,
            })
        active_tool_server_ids = _parse_active_tool_slugs(chat.active_tool_slug)
        return JsonResponse({
            "chat_id": str(chat.id),
            "title": chat.title,
            "messages": payload,
            "active_tool_server_ids": active_tool_server_ids,
            "active_tool_server_id": active_tool_server_ids[0] if active_tool_server_ids else "",
        })
    except Chat.DoesNotExist:
        return JsonResponse({"error": "Chat not found"}, status=404)
    except Exception as exc:
        logger.exception("Failed to load chat %s", chat_id)
        return JsonResponse({"error": str(exc)}, status=500)


# Model and tool discovery APIs.


# Return discovered tool servers.
def mcp_config_api(request):
    mcp_json.ensure_default_mcp_json()

    if request.method == "GET":
        return JsonResponse({"content": mcp_json.load_raw_text()})

    if request.method not in {"PUT", "PATCH"}:
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        payload = _read_json_request_body(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    content = payload.get("content")
    if not isinstance(content, str):
        return JsonResponse({"error": "Expected JSON object with string field 'content'"}, status=400)

    try:
        mcp_json.save_raw_text(content)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    tool_registry.reset_cache()
    _clear_tool_server_cache()

    return JsonResponse({"ok": True, "content": mcp_json.load_raw_text()})


# Restart configured user MCP sessions and return freshly discovered tools.
def mcp_reload_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        payload = _read_json_request_body(request) if request.body else {}
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    engine, engine_error = _resolve_request_engine_or_response(request, payload)
    if engine_error is not None:
        return engine_error
    model_name = str(payload.get("model") or "").strip() or None

    tool_registry.reset_cache()
    _clear_tool_server_cache()
    servers = _list_tool_servers_cached(engine, model_name)
    return JsonResponse(
        {
            "ok": True,
            "tool_servers": servers,
            "servers": servers,
            "tools": servers,
        }
    )


# Return a normalized JSON error for skills APIs.
def _skills_error_response(exc: Exception) -> JsonResponse:
    if isinstance(exc, FileNotFoundError):
        return JsonResponse({"error": str(exc)}, status=404)
    if isinstance(exc, ValueError):
        return JsonResponse({"error": str(exc)}, status=400)
    logger.exception("Unhandled skills API error")
    return JsonResponse({"error": str(exc)}, status=500)


# List skills or create a new skill folder.
def skills_api(request):
    if request.method == "GET":
        try:
            return JsonResponse(skills_config.list_skills())
        except Exception as exc:
            return _skills_error_response(exc)

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        payload = _read_json_request_body(request)
        return JsonResponse(skills_config.create_skill_folder(str(payload.get("name") or "")))
    except Exception as exc:
        return _skills_error_response(exc)


# Rename or delete one skill folder.
def skills_folder_api(request):
    if request.method not in {"PATCH", "DELETE"}:
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        payload = _read_json_request_body(request)
        if request.method == "PATCH":
            return JsonResponse(
                skills_config.rename_skill_folder(
                    str(payload.get("old_name") or payload.get("name") or ""),
                    str(payload.get("new_name") or ""),
                )
            )
        return JsonResponse(skills_config.delete_skill_folder(str(payload.get("name") or "")))
    except Exception as exc:
        return _skills_error_response(exc)


# Read, write, or delete one skill file.
def skills_file_api(request):
    if request.method == "GET":
        try:
            return JsonResponse(
                skills_config.read_skill_file(
                    str(request.GET.get("folder") or ""),
                    str(request.GET.get("file") or ""),
                )
            )
        except Exception as exc:
            return _skills_error_response(exc)

    if request.method not in {"PUT", "DELETE"}:
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        payload = _read_json_request_body(request)
        folder = str(payload.get("folder") or "")
        file_path = str(payload.get("file") or "")
        if request.method == "PUT":
            return JsonResponse(skills_config.write_skill_file(folder, file_path, str(payload.get("content") or "")))
        return JsonResponse(skills_config.delete_skill_file(folder, file_path))
    except Exception as exc:
        return _skills_error_response(exc)


# Enable or disable one skill.
def skills_enabled_api(request):
    if request.method not in {"PATCH", "POST"}:
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        payload = _read_json_request_body(request)
        return JsonResponse(
            skills_config.set_skill_enabled(
                str(payload.get("folder") or payload.get("name") or ""),
                skills_config.parse_enabled_flag(payload.get("enabled")),
            )
        )
    except Exception as exc:
        return _skills_error_response(exc)


# Create or delete a subdirectory inside a skill folder.
def skills_directory_api(request):
    if request.method not in {"POST", "DELETE"}:
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        payload = _read_json_request_body(request)
        folder = str(payload.get("folder") or "")
        path = str(payload.get("path") or "")
        if request.method == "DELETE":
            return JsonResponse(skills_config.delete_skill_subdirectory(folder, path))
        return JsonResponse(skills_config.create_skill_subdirectory(folder, path))
    except Exception as exc:
        return _skills_error_response(exc)


# Import a skill folder from a list of {path, content} file entries.
def skills_import_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        payload = _read_json_request_body(request)
        skill_name = str(payload.get("name") or "")
        files = payload.get("files")
        if not isinstance(files, list):
            return JsonResponse({"error": "files must be a list"}, status=400)
        return JsonResponse(skills_config.import_skill_files(skill_name, files))
    except ValueError as exc:
        status = 409 if "already exists" in str(exc).lower() else 400
        return JsonResponse({"error": str(exc)}, status=status)
    except Exception as exc:
        return _skills_error_response(exc)


# Rename a file or directory inside a skill folder.
def skills_path_api(request):
    if request.method != "PATCH":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        payload = _read_json_request_body(request)
        return JsonResponse(
            skills_config.rename_skill_item(
                str(payload.get("folder") or ""),
                str(payload.get("old_path") or ""),
                str(payload.get("new_path") or ""),
                str(payload.get("kind") or "file"),
            )
        )
    except Exception as exc:
        return _skills_error_response(exc)


# Return locally discovered MCP-style tool servers for the requested engine/model.
def get_tools_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    engine, engine_error = _resolve_request_engine_or_response(request)
    if engine_error is not None:
        return engine_error
    model_name = str(request.GET.get("model", "") or "").strip() or None
    servers = _list_tool_servers_cached(engine, model_name)
    return JsonResponse({"tool_servers": servers, "servers": servers, "tools": servers})


# Re-probe Docker availability on demand and return the fresh result.
def docker_status_refresh_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    from Services import docker_status

    status = docker_status.refresh()
    return JsonResponse(
        {"available": status["available"], "checked_at": status["checked_at"]}
    )


# Resolve and proxy a stable favicon for a search result domain.
def favicon_api(request):
    # Build one inline image HTTP response for a byte range.
    def image_response(content_type: str, content: bytes) -> HttpResponse:
        response = HttpResponse(content, content_type=content_type)
        response["Cache-Control"] = "public, max-age=604800"
        response["X-Content-Type-Options"] = "nosniff"
        return response

    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    domain = _normalize_favicon_domain(request.GET.get("domain") or request.GET.get("url") or "")
    if not domain:
        return JsonResponse({"error": "Domain is required"}, status=400)

    now = time.time()
    with _favicon_cache_lock:
        cached = _favicon_cache.get(domain)
        if cached and cached[0] > now:
            content_type, content = cached[1], cached[2]
            if content:
                return image_response(content_type, content)
            return HttpResponse(status=404)

        disk_cached = _read_favicon_disk_cache(domain, now)
        if disk_cached:
            content_type, content = disk_cached
            _favicon_cache[domain] = (now + FAVICON_CACHE_TTL_SECONDS, content_type, content)
            return image_response(content_type, content)

    resolved = _resolve_favicon_content(domain)
    with _favicon_cache_lock:
        if resolved:
            content_type, content = resolved
            expires_at = now + FAVICON_CACHE_TTL_SECONDS
            _favicon_cache[domain] = (expires_at, content_type, content)
            _write_favicon_disk_cache(domain, content_type, content, expires_at)
            return image_response(content_type, content)

        _favicon_cache[domain] = (now + FAVICON_NEGATIVE_CACHE_TTL_SECONDS, "", b"")

    return HttpResponse(status=404)


# Preset APIs.

# Return Ollama preset metadata.
def get_ollama_presets_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    model_name = str(request.GET.get("model", "") or "").strip()
    if not model_name:
        return JsonResponse({"error": "Model parameter is required"}, status=400)

    try:
        return JsonResponse(get_ollama_preset_payload(model_name))
    except Exception as exc:
        logger.exception("Error getting Ollama presets for %s", model_name)
        return JsonResponse({"error": str(exc)}, status=500)


# Return preset metadata for the selected LM Studio model.
def get_lms_presets_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    model_name = str(request.GET.get("model", "") or "").strip()
    if not model_name:
        return JsonResponse({"error": "Model parameter is required"}, status=400)

    try:
        return JsonResponse(get_lms_preset_payload(model_name))
    except Exception as exc:
        logger.exception("Error getting LM Studio presets for %s", model_name)
        return JsonResponse({"error": str(exc)}, status=500)


# Return a JSON response after invalidating model metadata.
def _preset_mutation_response(payload: dict[str, Any]) -> JsonResponse:
    _clear_model_metadata_caches()
    return JsonResponse(payload)


# Sync the active Ollama preset.
def sync_ollama_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        config = data.get("config", {})
        if not model_name:
            return JsonResponse({"error": "Model parameter is required"}, status=400)
        return _preset_mutation_response(sync_active_ollama_preset(model_name, config))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Error syncing Ollama preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Persist UI changes to the active LM Studio preset.
def sync_lms_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        config = data.get("config", {})
        if not model_name:
            return JsonResponse({"error": "Model parameter is required"}, status=400)
        return _preset_mutation_response(sync_active_lms_preset(model_name, config))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Error syncing LM Studio preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Activate an Ollama preset.
def select_ollama_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        if not model_name or not preset_id:
            return JsonResponse({"error": "Model and preset_id are required"}, status=400)
        return _preset_mutation_response(activate_ollama_preset(model_name, preset_id))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except OllamaPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error selecting Ollama preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Set the active preset for an LM Studio model.
def select_lms_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        if not model_name or not preset_id:
            return JsonResponse({"error": "Model and preset_id are required"}, status=400)
        return _preset_mutation_response(activate_lms_preset(model_name, preset_id))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except LmsPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error selecting LM Studio preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Create an Ollama preset.
def create_ollama_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_name = str(data.get("name", "") or "").strip()
        config = data.get("config", {})
        if not model_name:
            return JsonResponse({"error": "Model parameter is required"}, status=400)
        return _preset_mutation_response(
            create_ollama_preset(
                model_name,
                name=preset_name or None,
                config=config if isinstance(config, dict) else {},
                activate=True,
            )
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Error creating Ollama preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Create a new LM Studio preset for the selected model.
def create_lms_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_name = str(data.get("name", "") or "").strip()
        config = data.get("config", {})
        if not model_name:
            return JsonResponse({"error": "Model parameter is required"}, status=400)
        return _preset_mutation_response(
            create_lms_preset(
                model_name,
                name=preset_name or None,
                config=config if isinstance(config, dict) else {},
                activate=True,
            )
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Error creating LM Studio preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Rename an Ollama preset.
def rename_ollama_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        preset_name = str(data.get("name", "") or "").strip()
        if not model_name or not preset_id or not preset_name:
            return JsonResponse({"error": "Model, preset_id and name are required"}, status=400)
        return _preset_mutation_response(rename_ollama_preset(model_name, preset_id, preset_name))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except OllamaPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error renaming Ollama preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Rename an existing custom LM Studio preset.
def rename_lms_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        preset_name = str(data.get("name", "") or "").strip()
        if not model_name or not preset_id or not preset_name:
            return JsonResponse({"error": "Model, preset_id and name are required"}, status=400)
        return _preset_mutation_response(rename_lms_preset(model_name, preset_id, preset_name))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except LmsPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error renaming LM Studio preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Delete an Ollama preset.
def delete_ollama_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        if not model_name or not preset_id:
            return JsonResponse({"error": "Model and preset_id are required"}, status=400)
        return _preset_mutation_response(delete_ollama_preset(model_name, preset_id))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except OllamaPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error deleting Ollama preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Delete an existing custom preset and fall back to the default one.
def delete_lms_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        if not model_name or not preset_id:
            return JsonResponse({"error": "Model and preset_id are required"}, status=400)
        return _preset_mutation_response(delete_lms_preset(model_name, preset_id))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except LmsPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error deleting LM Studio preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Resolve the endpoint URL that scopes OpenAI presets.
def _get_openai_preset_endpoint() -> str:
    return normalize_openai_endpoint_url(settings.get_engine_url("openai"))


# Return preset metadata for the selected OpenAI model and endpoint.
def get_openai_presets_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    model_name = str(request.GET.get("model", "") or "").strip()
    if not model_name:
        return JsonResponse({"error": "Model parameter is required"}, status=400)

    try:
        return JsonResponse(get_openai_preset_payload(_get_openai_preset_endpoint(), model_name))
    except Exception as exc:
        logger.exception("Error getting OpenAI presets for %s", model_name)
        return JsonResponse({"error": str(exc)}, status=500)


# Persist UI changes to the active OpenAI preset.
def sync_openai_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        config = data.get("config", {})
        if not model_name:
            return JsonResponse({"error": "Model parameter is required"}, status=400)
        return _preset_mutation_response(
            sync_active_openai_preset(_get_openai_preset_endpoint(), model_name, config)
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Error syncing OpenAI preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Set the active preset for an OpenAI model.
def select_openai_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        if not model_name or not preset_id:
            return JsonResponse({"error": "Model and preset_id are required"}, status=400)
        return _preset_mutation_response(
            activate_openai_preset(_get_openai_preset_endpoint(), model_name, preset_id)
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except OpenAiPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error selecting OpenAI preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Create a new OpenAI preset for the selected model.
def create_openai_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_name = str(data.get("name", "") or "").strip()
        config = data.get("config", {})
        if not model_name:
            return JsonResponse({"error": "Model parameter is required"}, status=400)
        return _preset_mutation_response(
            create_openai_preset(
                _get_openai_preset_endpoint(),
                model_name,
                name=preset_name or None,
                config=config if isinstance(config, dict) else {},
                activate=True,
            )
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Error creating OpenAI preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Rename an existing custom OpenAI preset.
def rename_openai_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        preset_name = str(data.get("name", "") or "").strip()
        if not model_name or not preset_id or not preset_name:
            return JsonResponse({"error": "Model, preset_id and name are required"}, status=400)
        return _preset_mutation_response(
            rename_openai_preset(_get_openai_preset_endpoint(), model_name, preset_id, preset_name)
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except OpenAiPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error renaming OpenAI preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Delete an existing custom OpenAI preset and fall back to the default one.
def delete_openai_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        if not model_name or not preset_id:
            return JsonResponse({"error": "Model and preset_id are required"}, status=400)
        return _preset_mutation_response(
            delete_openai_preset(_get_openai_preset_endpoint(), model_name, preset_id)
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except OpenAiPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error deleting OpenAI preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Return preset metadata for the selected Google GenAI model.
def get_google_genai_presets_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    model_name = str(request.GET.get("model", "") or "").strip()
    if not model_name:
        return JsonResponse({"error": "Model parameter is required"}, status=400)

    try:
        return JsonResponse(get_google_genai_preset_payload(model_name))
    except Exception as exc:
        logger.exception("Error getting Google GenAI presets for %s", model_name)
        return JsonResponse({"error": str(exc)}, status=500)


# Persist UI changes to the active Google GenAI preset.
def sync_google_genai_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        config = data.get("config", {})
        if not model_name:
            return JsonResponse({"error": "Model parameter is required"}, status=400)
        return _preset_mutation_response(sync_active_google_genai_preset(model_name, config))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Error syncing Google GenAI preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Set the active preset for a Google GenAI model.
def select_google_genai_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        if not model_name or not preset_id:
            return JsonResponse({"error": "Model and preset_id are required"}, status=400)
        return _preset_mutation_response(activate_google_genai_preset(model_name, preset_id))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except GoogleGenAiPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error selecting Google GenAI preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Create a new Google GenAI preset for the selected model.
def create_google_genai_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_name = str(data.get("name", "") or "").strip()
        config = data.get("config", {})
        if not model_name:
            return JsonResponse({"error": "Model parameter is required"}, status=400)
        return _preset_mutation_response(
            create_google_genai_preset(
                model_name,
                name=preset_name or None,
                config=config if isinstance(config, dict) else {},
                activate=True,
            )
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        logger.exception("Error creating Google GenAI preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Rename an existing custom Google GenAI preset.
def rename_google_genai_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        preset_name = str(data.get("name", "") or "").strip()
        if not model_name or not preset_id or not preset_name:
            return JsonResponse({"error": "Model, preset_id and name are required"}, status=400)
        return _preset_mutation_response(rename_google_genai_preset(model_name, preset_id, preset_name))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except GoogleGenAiPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error renaming Google GenAI preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Delete an existing custom Google GenAI preset and fall back to the default one.
def delete_google_genai_preset_api(request):
    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
        model_name = str(data.get("model", "") or "").strip()
        preset_id = str(data.get("preset_id", "") or "").strip()
        if not model_name or not preset_id:
            return JsonResponse({"error": "Model and preset_id are required"}, status=400)
        return _preset_mutation_response(delete_google_genai_preset(model_name, preset_id))
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except GoogleGenAiPreset.DoesNotExist as exc:
        return JsonResponse({"error": str(exc)}, status=404)
    except Exception as exc:
        logger.exception("Error deleting Google GenAI preset")
        return JsonResponse({"error": str(exc)}, status=500)


# Runtime settings API.

# Read or update runtime settings.
def runtime_settings_api(request):
    if request.method == "GET":
        return JsonResponse(_build_runtime_settings_payload())

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)

    allowed_keys = {
        "llm-engine",
        "lms_url",
        "openai_url",
        "openai_api_key",
        "google_genai_url",
        "google_genai_api_key",
    }

    previous_engine = settings.get_llm_engine()
    next_engine = previous_engine

    # Return whether the payload looks like a masked placeholder, not a real key.
    def _is_masked_api_key(value: str) -> bool:
        raw = str(value or "")
        stripped = raw.strip()
        if not stripped:
            return False

        # Password placeholders often arrive as repeated mask glyphs.
        if all(char in {"*", "\u2022", "\u25cf", "\u2219", "\u00b7", "\u25e6"} for char in stripped):
            return True

        lowered = stripped.lower()
        if lowered in {
            "stored api key",
            "stored api key.",
            "stored key",
            "use saved key",
        }:
            return True

        return False

    # Persist only known settings and normalize engine-specific values.
    for raw_key, raw_value in data.items():
        if raw_key not in allowed_keys:
            continue

        if raw_key == "llm-engine":
            value = settings.resolve_enabled_engine(raw_value)
            next_engine = value
        else:
            value = str(raw_value or "").strip()
            if raw_key in {"openai_api_key", "google_genai_api_key"} and _is_masked_api_key(value):
                # Ignore UI mask placeholders so they never overwrite real keys.
                continue

        settings.set(raw_key, value)

    # Apply runtime transitions after the new settings are saved.
    llm_api.handle_engine_transition(previous_engine, next_engine)
    _clear_model_metadata_caches()

    return JsonResponse(_build_runtime_settings_payload())


# Browser portal control APIs.

_frame404_burst_log_count = 0
_frame404_burst_log_last_ts = 0.0


# Browser portal roots.
def _browser_portal_roots() -> list[Path]:
    return [
        settings.BASE_DIR / "Data" / "runtime" / "browser_portal",
    ]


# Browser portal debug log path.
def _browser_portal_debug_log_path(root: Path | None = None) -> Path:
    return (root or _browser_portal_roots()[0]) / "debug.jsonl"


# Browser portal debug safe.
def _browser_portal_debug_safe(value: Any, *, depth: int = 0) -> Any:
    if depth > 4:
        return repr(value)[:4000]
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value[:4000]
    if isinstance(value, dict):
        return {
            str(key)[:200]: _browser_portal_debug_safe(child, depth=depth + 1)
            for key, child in value.items()
            if str(key) not in {"data_base64", "preview"}
        }
    if isinstance(value, (list, tuple, set)):
        return [_browser_portal_debug_safe(child, depth=depth + 1) for child in list(value)[:50]]
    return repr(value)[:4000]


# Compact portal POST body for debug.jsonl (typing floods the log otherwise).
def _browser_portal_http_event_body_for_log(data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    event_type = str(data.get("type") or "").strip().lower()
    slim: dict[str, Any] = {}
    for key in ("id", "created_at", "type", "version", "session_id"):
        if key in data:
            slim[key] = data[key]
    client_meta = data.get("client_meta")
    if isinstance(client_meta, dict):
        slim["client_meta_keys"] = list(client_meta.keys())[:40]
    if event_type == "type":
        text = data.get("text")
        slim["text_len"] = len(text) if isinstance(text, str) else 0
    elif event_type == "key":
        slim["key"] = str(data.get("key") or "")[:120]
    else:
        for key in ("x", "y", "delta_x", "delta_y", "viewport_width", "viewport_height"):
            if key in data:
                slim[key] = data[key]
    return slim


# Browser portal debug logging is intentionally disabled.
def _write_browser_portal_debug_event(root: Path | None, event: str, **fields: Any) -> None:
    return None


# Return browser portal state from.
def _read_browser_portal_state_from(root: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads((root / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


# Is active browser portal state.
def _is_active_browser_portal_state(payload: dict[str, Any]) -> bool:
    if str(payload.get("status") or "").lower() != "waiting":
        return False
    try:
        updated_at = float(payload.get("updated_at") or 0)
        timeout_seconds = float(payload.get("timeout_seconds") or 45)
        deadline_at = float(payload.get("deadline_at") or 0)
    except (TypeError, ValueError):
        return False
    if deadline_at > 0:
        return time.time() <= deadline_at + 10
    return updated_at > 0 and time.time() <= updated_at + timeout_seconds + 10


# Active browser portal root.
def _active_browser_portal_root() -> Path | None:
    candidates: list[tuple[float, Path]] = []
    for root in _browser_portal_roots():
        payload = _read_browser_portal_state_from(root)
        if payload and _is_active_browser_portal_state(payload):
            try:
                candidates.append((float(payload.get("updated_at") or 0), root))
            except (TypeError, ValueError):
                continue
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


# Browser portal state path.
def _browser_portal_state_path() -> Path:
    root = _active_browser_portal_root()
    if root is None:
        return _browser_portal_roots()[0] / "state.json"
    return root / "state.json"


# Browser portal events dir.
def _browser_portal_events_dir() -> Path:
    root = _active_browser_portal_root()
    if root is None:
        return _browser_portal_roots()[0] / "events"
    return root / "events"


# Return browser portal state.
def _read_browser_portal_state() -> dict[str, Any]:
    root = _active_browser_portal_root()
    if root is None:
        return {"ok": False, "error": "No active browser_wait_for_user session is available."}
    payload = _read_browser_portal_state_from(root)
    return payload if payload else {"ok": False, "error": "Invalid browser portal state."}


# Return the latest frame published by browser_wait_for_user.
def browser_portal_frame_api(request):
    global _frame404_burst_log_count, _frame404_burst_log_last_ts

    if request.method != "GET":
        _write_browser_portal_debug_event(None, "frame_rejected_method", method=request.method)
        return JsonResponse({"error": "Invalid request method"}, status=405)
    payload = _read_browser_portal_state()
    status = 200 if payload.get("ok") else 404
    if status == 200:
        _frame404_burst_log_count = 0
    if status != 200:
        should_log_frame = True
        if status == 404:
            now_ts = time.time()
            _frame404_burst_log_count += 1
            if _frame404_burst_log_count <= 8:
                should_log_frame = True
            elif now_ts - _frame404_burst_log_last_ts >= 5.0:
                _frame404_burst_log_last_ts = now_ts
                should_log_frame = True
            else:
                should_log_frame = False
        if should_log_frame:
            _write_browser_portal_debug_event(
                _active_browser_portal_root(),
                "frame_response",
                status_code=status,
                payload_status=payload.get("status"),
                session_id=payload.get("session_id"),
                ok=payload.get("ok"),
                error=payload.get("error"),
                has_frame=isinstance(payload.get("frame"), dict),
                url=payload.get("url"),
            )
    return JsonResponse(payload, status=status)


# Queue one human portal event for the active browser_wait_for_user loop.
def browser_portal_event_api(request):
    if request.method != "POST":
        _write_browser_portal_debug_event(None, "event_rejected_method", method=request.method)
        return JsonResponse({"error": "Invalid request method"}, status=405)

    try:
        data = _read_json_request_body(request)
    except ValueError as exc:
        _write_browser_portal_debug_event(None, "event_rejected_invalid_json", error=str(exc))
        return JsonResponse({"error": str(exc)}, status=400)

    event_type = str(data.get("type") or "").strip().lower()
    _write_browser_portal_debug_event(
        _active_browser_portal_root(),
        "event_received",
        event_type=event_type,
        payload=_browser_portal_http_event_body_for_log(data),
        request_path=request.path,
    )
    if event_type not in {"click", "scroll", "key", "type", "finish", "click_ref"}:
        _write_browser_portal_debug_event(None, "event_rejected_unsupported_type", event_type=event_type, payload=_browser_portal_http_event_body_for_log(data))
        return JsonResponse({"error": "Unsupported browser portal event type."}, status=400)

    active_root = _active_browser_portal_root()
    if active_root is None:
        _write_browser_portal_debug_event(None, "event_rejected_no_active_session", event_type=event_type, payload=_browser_portal_http_event_body_for_log(data))
        return JsonResponse({"error": "No active browser_wait_for_user session is available."}, status=409)

    active_state = _read_browser_portal_state_from(active_root) or {}
    active_session_id = str(active_state.get("session_id") or "").strip()
    requested_session_id = str(data.get("session_id") or "").strip()
    if requested_session_id and active_session_id and requested_session_id != active_session_id:
        _write_browser_portal_debug_event(
            active_root,
            "event_rejected_session_mismatch",
            event_type=event_type,
            requested_session_id=requested_session_id,
            active_session_id=active_session_id,
            payload=_browser_portal_http_event_body_for_log(data),
        )
        return JsonResponse({"error": "This browser_wait_for_user session is no longer active."}, status=409)

    events_dir = active_root / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    event_id = uuid.uuid4().hex
    event_payload = {
        "id": event_id,
        "created_at": time.time(),
        **data,
        "type": event_type,
    }
    if active_session_id:
        event_payload["session_id"] = active_session_id
    event_path = events_dir / f"event_{int(time.time() * 1000)}_{event_id}.json"
    event_path.write_text(json.dumps(event_payload, ensure_ascii=False), encoding="utf-8")
    _write_browser_portal_debug_event(
        active_root,
        "event_written",
        event_id=event_id,
        event_type=event_type,
        event_path=str(event_path),
        active_session_id=active_session_id,
        requested_session_id=requested_session_id,
        payload=_browser_portal_http_event_body_for_log(event_payload),
    )

    if event_type == "finish":
        payload = {
            **active_state,
            "ok": True,
            "status": "done",
            "updated_at": time.time(),
            "version": int(time.time() * 1000),
        }
    else:
        payload = _read_browser_portal_state()
    if not payload.get("ok"):
        payload = {"ok": True, "event_id": event_id, "queued": True}
    else:
        payload = {**payload, "event_id": event_id, "queued": True}
    _write_browser_portal_debug_event(
        active_root,
        "event_response",
        event_id=event_id,
        event_type=event_type,
        queued=True,
        response_status=payload.get("status"),
        response_session_id=payload.get("session_id"),
    )
    return JsonResponse(payload)


# Render the preloaded chat page.
class ChatView(TemplateView):
    template_name = "main/main.html"

    # Build the preloaded chat page context.
    def get_context_data(self, **kwargs: Any) -> dict[str, Any]:
        context = super().get_context_data(**kwargs)
        context.update(_build_base_context())
        context["preload_chat_id"] = str(kwargs.get("chat_id", ""))
        return context


# Resolve an inline [[images: ...]] token from an assistant message into a photo row.
def image_search_api(request):
    """Run the mcp-web-search image vertical for one query.

    Deliberately NOT gated on the active model's vision capability: the gallery is
    user-facing chat content — history written by a multimodal model must keep
    rendering after switching to a text-only model. The multimodality gate lives
    where it belongs, on the tool offered to the model (build_ollama_tools).
    """

    if request.method != "POST":
        return JsonResponse({"error": "Invalid request method"}, status=405)

    data = _read_json_request_body(request)
    query = " ".join(str(data.get("query") or "").split())
    if not query:
        return JsonResponse({"error": "Missing query."}, status=400)

    engine = str(data.get("engine") or "").strip() or None
    model_name = str(data.get("model") or "").strip() or None

    try:
        limit = max(1, min(24, int(data.get("limit") or 12)))
    except (TypeError, ValueError):
        limit = 12

    try:
        _tools, tool_lookup = tool_registry.build_ollama_tools(
            ["web_search"], engine=engine, model_name=model_name, model_vision=True
        )
        if "web_search__image_search" not in tool_lookup:
            return JsonResponse({"ok": False, "error": "Image search tool is unavailable."}, status=503)
        result = tool_registry.call_ollama_tool(
            tool_lookup,
            "web_search__image_search",
            {"query": query, "limit": limit},
        )
    except Exception as exc:  # noqa: BLE001 — one token must never break the chat UI
        logger.warning("Inline image search failed for %r: %s", query[:120], exc)
        return JsonResponse({"ok": False, "error": "Image search failed."}, status=502)

    # call_ollama_tool hands back either a serialized JSON string (plain results)
    # or the {"_tool_result_content", "_tool_result_structured"} envelope (rich
    # results) — unwrap both before reading the gallery payload.
    payload = result
    if isinstance(payload, str) and payload.strip().startswith("{"):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = None
    if isinstance(payload, dict) and isinstance(payload.get("_tool_result_structured"), dict):
        structured = payload["_tool_result_structured"]
        ui = structured.get("ui") if isinstance(structured.get("ui"), dict) else structured
    else:
        ui = payload.get("ui") if isinstance(payload, dict) else None
    images = (ui or {}).get("images") if isinstance(ui, dict) else None
    return JsonResponse({
        "ok": True,
        "query": query,
        "images": images or [],
        "status": (ui or {}).get("status") if isinstance(ui, dict) else "empty",
    })
