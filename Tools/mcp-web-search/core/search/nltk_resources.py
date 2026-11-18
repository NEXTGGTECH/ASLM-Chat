# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import nltk

# NLTK data packages used by the search tool (ported from ASLM-Search
# core/search/nltk_resources.py; same URLs and checksums). Missing data is fetched once in
# a background thread on first use; callers fall back to data-free NLTK code meanwhile.
# ASLM_NLTK_DATA overrides the location, ASLM_NLTK_DOWNLOAD=0 disables downloading.

logger = logging.getLogger("mcp.web_search.nltk")


# One NLTK data package: its name, path inside the data directory, download URL and checksum.
@dataclass(frozen=True, slots=True)
class NltkResource:
    package: str
    path: str
    url: str
    sha256: str


RESOURCES: tuple[NltkResource, ...] = (
    NltkResource(
        "punkt_tab",
        "tokenizers/punkt_tab",
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/tokenizers/punkt_tab.zip",
        "e57f64187974277726a3417ca6f181ec5403676c717672eef6a748a7b20e0106",
    ),
    NltkResource(
        "stopwords",
        "corpora/stopwords",
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/stopwords.zip",
        "48c0e52d8b52546e827f53761fb30300c0ab94f70660d28bd65ba0a86270946b",
    ),
    # BCP 47 tags: nltk.langnames maps ISO codes ("tr", "zh") to the English names the
    # stemmer / punkt / stopwords resources are filed under.
    NltkResource(
        "bcp47",
        "corpora/bcp47",
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/bcp47.zip",
        "435d986fd9de0ae540a34e0978dbbaf5d1db7576b2bc7571da71cf6a01c8dfaa",
    ),
)
_BY_PACKAGE = {resource.package: resource for resource in RESOURCES}
_RETRY_AFTER_SECONDS = 600.0
_LOCK = threading.Lock()
_downloading: set[str] = set()
_failed_at: dict[str, float] = {}


# Directory for NLTK data: ASLM_NLTK_DATA when set, else the tool's cache.
def nltk_data_path() -> Path:
    configured = os.environ.get("ASLM_NLTK_DATA", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2] / "_cache" / "nltk_data"


# Make NLTK look in the tool's data directory first and return it.
def configure_nltk_data_path() -> Path:
    path = nltk_data_path()
    if str(path) not in nltk.data.path:
        nltk.data.path.insert(0, str(path))
    return path


# Whether a data package is present on disk.
def installed(package: str) -> bool:
    resource = _BY_PACKAGE[package]
    configure_nltk_data_path()
    for candidate in (resource.path, f"{resource.path}.zip"):
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


# Download a data package into the data directory and verify its checksum.
def _download(resource: NltkResource) -> None:
    import httpx

    destination = configure_nltk_data_path() / f"{resource.path}.zip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    try:
        digest = hashlib.sha256()
        with httpx.stream("GET", resource.url, follow_redirects=False, timeout=60.0) as response:
            response.raise_for_status()
            with temporary.open("wb") as output:
                for chunk in response.iter_bytes(64 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
        if digest.hexdigest() != resource.sha256:
            raise RuntimeError("download checksum mismatch")
        temporary.replace(destination)
        logger.info("nltk data installed package=%s path=%s", resource.package, destination)
    except Exception as exc:  # noqa: BLE001 - a failed download only keeps the fallback
        temporary.unlink(missing_ok=True)
        with _LOCK:
            _failed_at[resource.package] = time.monotonic()
        logger.warning("nltk data download failed package=%s: %s", resource.package, exc)
    finally:
        with _LOCK:
            _downloading.discard(resource.package)


# True when the package is usable now; otherwise starts a background download (at most
# one per package, retried after a pause on failure) and returns False.
def ensure_resource(package: str) -> bool:
    if installed(package):
        return True
    if os.environ.get("ASLM_NLTK_DOWNLOAD", "1").strip() == "0":
        return False
    with _LOCK:
        failed = _failed_at.get(package)
        if package in _downloading or (failed and time.monotonic() - failed < _RETRY_AFTER_SECONDS):
            return False
        _downloading.add(package)
    threading.Thread(
        target=_download, args=(_BY_PACKAGE[package],), name=f"nltk-download-{package}", daemon=True
    ).start()
    return False


# Start every missing download now (first tool call) so the data is ready early.
def prefetch_resources() -> None:
    for resource in RESOURCES:
        ensure_resource(resource.package)


configure_nltk_data_path()
