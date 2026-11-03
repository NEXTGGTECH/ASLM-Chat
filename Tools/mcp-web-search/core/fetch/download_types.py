# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlparse


# File-type policy used by read_page to explain when a URL points to a file
# rather than to a normal HTML page.
ALLOWED_EXTENSIONS: dict[str, list[str]] = {
    "text": [".pdf", ".docx", ".xlsx", ".csv", ".txt", ".md", ".json", ".xml", ".html"],
    "media": [".mp3", ".mp4", ".wav", ".webm", ".mkv", ".jpg", ".jpeg", ".png", ".gif", ".webp"],
    "archive": [".zip", ".tar", ".gz", ".tar.gz", ".7z"],
    "data": [".sqlite", ".db"],
    "code": [
        ".py", ".js", ".mjs", ".cjs", ".ts", ".jsx", ".tsx",
        ".css", ".scss", ".sass", ".rb", ".php", ".lua",
    ],
}

# Executable and native-library extensions that are never reported as downloadable files.
BLOCKED_EXTENSIONS: list[str] = [
    ".exe",
    ".dll",
    ".bat",
    ".sh",
    ".ps1",
    ".msi",
    ".bin",
    ".vbs",
    ".cmd",
    ".so",
    ".dylib",
]


# Return the lowercase extension, including compound archive suffixes.
def _extract_extension(filename: str) -> str:
    name = filename.lower()
    for compound in (".tar.gz", ".tar.bz2", ".tar.xz"):
        if name.endswith(compound):
            return compound
    return Path(name).suffix


# Return (extension, category) if url points to a known downloadable file.
def get_download_info(url: str) -> tuple[str, str] | None:
    # Take the file name from the decoded URL path; an unparsable URL is not a download.
    try:
        path = unquote(urlparse(url).path)
        filename = PurePosixPath(path).name or ""
    except Exception:
        return None

    # Blocked or unknown extensions are not reported; otherwise look up the category.
    ext = _extract_extension(filename)
    if not ext or ext in BLOCKED_EXTENSIONS:
        return None

    for category, exts in ALLOWED_EXTENSIONS.items():
        if ext in exts:
            return ext, category

    return None


__all__ = ["ALLOWED_EXTENSIONS", "BLOCKED_EXTENSIONS", "get_download_info"]
