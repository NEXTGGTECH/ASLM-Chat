# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import ipaddress
import os
import socket
from urllib.parse import parse_qsl, urljoin, urlparse, urlunparse, urlencode

# Query parameters that only carry tracking or referral data and are dropped by normalize_url.
_TRACKING_PARAMS = frozenset({
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "utm_id", "fbclid", "gclid", "msclkid", "mc_cid", "mc_eid",
    "ref", "referrer", "source", "_ga", "yclid", "ysclid",
})


# Strip tracking query params and normalize scheme, host, and path.
def normalize_url(url: str) -> str:
    try:
        p = urlparse(url)
        clean_qs = urlencode([(k, v) for k, v in parse_qsl(p.query) if k.lower() not in _TRACKING_PARAMS])
        return urlunparse((
            p.scheme.lower(), p.netloc.lower(),
            p.path.rstrip("/") or "/", p.params, clean_qs, "",
        ))
    except Exception:
        return url


# File extensions of binary or media assets that are never read as text, grouped by kind.
_NON_TEXT_EXTS = frozenset({
    # Images
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".avif", ".bmp", ".tif", ".tiff",
    ".ico", ".svg", ".heic", ".heif",
    # Video
    ".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".mpeg", ".mpg",
    # Audio
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".oga", ".opus", ".wma",
    # Archives / compressed blobs
    ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz", ".tgz", ".zst",
    # Office / ebook / database / binaries. PDF is intentionally excluded:
    # the research pipeline routes PDFs to a text extractor.
    ".csv", ".tsv", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".odp",
    ".rtf", ".epub", ".sqlite", ".db", ".exe", ".dmg", ".msi", ".bin", ".dll", ".so",
})

# Content-type families (prefix match) and exact content types that never enter text extraction.
_NON_TEXT_CONTENT_PREFIXES = (
    "image/",
    "video/",
    "audio/",
    "font/",
)
_NON_TEXT_CONTENT_TYPES = frozenset({
    "application/zip",
    "application/x-zip-compressed",
    "application/x-rar-compressed",
    "application/x-7z-compressed",
    "application/x-tar",
    "application/gzip",
    "application/x-gzip",
    "application/x-bzip2",
    "application/x-xz",
    "application/zstd",
    "application/octet-stream",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/epub+zip",
    "application/x-sqlite3",
})


# Return True when URL path points to a non-text binary/media asset.
def has_non_text_extension(url: str) -> bool:
    try:
        path = urlparse(url).path.lower()
    except Exception:
        path = url.lower().split("?", 1)[0]
    return any(path.endswith(ext) for ext in _NON_TEXT_EXTS)


# Return True for content types that should not enter text extraction.
def is_non_text_content_type(content_type: str) -> bool:
    ctype = (content_type or "").split(";", 1)[0].strip().lower()
    if not ctype:
        return False
    if ctype == "application/pdf":
        return False
    if any(ctype.startswith(prefix) for prefix in _NON_TEXT_CONTENT_PREFIXES):
        return True
    return ctype in _NON_TEXT_CONTENT_TYPES


# SSRF guard settings: the only schemes a fetch may use, hostnames and name suffixes that
# denote internal targets, and the maximum number of redirect hops per fetch.
_FETCH_SCHEMES = frozenset({"http", "https"})
_INTERNAL_HOSTS = frozenset({
    "localhost",
    "localhost.localdomain",
    "metadata",
    "metadata.google.internal",
})
_INTERNAL_SUFFIXES = (
    ".localhost",
    ".local",
    ".internal",
    ".lan",
    ".home",
    ".corp",
    ".intranet",
)
_MAX_REDIRECTS = 5


# Error raised when a URL points at an SSRF-sensitive target.
class UnsafeFetchUrl(ValueError):
    pass


# Return True when ASLM_WEB_ALLOW_PRIVATE_NET permits RFC1918 / link-local targets.
def _private_fetch_allowed() -> bool:
    return os.environ.get("ASLM_WEB_ALLOW_PRIVATE_NET", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Return True when host is a literal IP address (v4 or v6).
def _host_is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]"))
        return True
    except ValueError:
        return False


# Return True when the IP is not globally routable (private, loopback, link-local, etc.);
# unparsable text counts as blocked.
def _is_blocked_ip(ip_text: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_text)
    except ValueError:
        return True
    return not ip.is_global


# Resolve host to A/AAAA addresses for SSRF checks.
def _resolve_host_ips(host: str) -> set[str]:
    try:
        return {
            info[4][0]
            for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
            if info and info[4]
        }
    except OSError as exc:
        raise UnsafeFetchUrl(f"could not resolve host {host!r}: {exc}") from exc


# Validate a URL before an LLM-controlled web fetch (public HTTP/S only by default) and return
# it unchanged. Raises UnsafeFetchUrl for other schemes, internal or single-label hosts, and
# hosts that resolve to non-public addresses; allow_private skips the host checks.
def validate_public_fetch_url(url: str, *, allow_private: bool | None = None) -> str:
    allow_private = _private_fetch_allowed() if allow_private is None else allow_private
    parsed = urlparse((url or "").strip())
    if parsed.scheme.lower() not in _FETCH_SCHEMES:
        raise UnsafeFetchUrl("only http and https URLs are allowed")

    host = (parsed.hostname or "").strip().lower().rstrip(".")
    if not host:
        raise UnsafeFetchUrl("URL host is empty")

    if allow_private:
        return url

    # Reject well-known internal names and dotless hosts before any DNS lookup.
    if host in _INTERNAL_HOSTS or any(host.endswith(suffix) for suffix in _INTERNAL_SUFFIXES):
        raise UnsafeFetchUrl(f"blocked internal host {host!r}")

    if not _host_is_ip_literal(host) and "." not in host:
        raise UnsafeFetchUrl(f"blocked single-label host {host!r}")

    # Check every address the host maps to (or the literal IP itself) against the non-global ranges.
    ips = {host.strip("[]")} if _host_is_ip_literal(host) else _resolve_host_ips(host)
    blocked = sorted(ip for ip in ips if _is_blocked_ip(ip))
    if blocked:
        raise UnsafeFetchUrl(f"blocked non-public address for {host!r}: {', '.join(blocked[:3])}")

    return url


# Resolve one redirect Location header against the current URL and validate the result.
def validate_redirect_target(current_url: str, location: str, *, allow_private: bool | None = None) -> str:
    if not location:
        raise UnsafeFetchUrl("redirect without Location header")
    next_url = urljoin(current_url, location)
    return validate_public_fetch_url(next_url, allow_private=allow_private)


# Maximum redirect hops allowed per fetch.
def max_safe_redirects() -> int:
    return _MAX_REDIRECTS
