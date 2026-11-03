# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import html as html_lib
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, unquote, urljoin, urlparse

from core.fetch.arxiv_api import ARXIV_API_ENDPOINT, arxiv_api_slot

from custom_domains.base import SCOPE_READ_PAGE, FetchContext, PageResult

# Download size caps for the Atom API response and the HTML full text, and the maximum number of
# bibliography entries rendered into the page.
_API_MAX_BYTES = 2 * 1024 * 1024
_HTML_MAX_BYTES = 16 * 1024 * 1024
_MAX_REFERENCES = 80

# User-Agent sent to arXiv.
_UA = "ASLM-Chat/1.0 (https://github.com/NGGTLightKeeper/ASLM-Chat)"

# Id formats: new style (2401.01234v2), old style (hep-th/9901001v1), the trailing version suffix,
# the "[undef...]" label arXiv HTML puts on unresolved bibliography items, and runs of whitespace.
_NEW_ID_RE = re.compile(r"^\d{4}\.\d{4,5}(?:v\d+)?$", re.IGNORECASE)
_OLD_ID_RE = re.compile(r"^[a-z][a-z0-9.-]*/\d{7}(?:v\d+)?$", re.IGNORECASE)
_VERSION_RE = re.compile(r"v(\d+)$", re.IGNORECASE)
_UNDEFINED_REFERENCE_LABEL_RE = re.compile(r"^\[undef[a-z]*\]\s*", re.IGNORECASE)
_SPACE_RE = re.compile(r"\s+")

# XML namespaces of the Atom feed and of arXiv's extension elements.
_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"


# One author of an arXiv article with the affiliations the API lists for them.
@dataclass(slots=True)
class ArxivAuthor:
    name: str
    affiliations: list[str] = field(default_factory=list)


# Metadata of one arXiv article as returned by the Atom API; urls are normalised to https.
@dataclass(slots=True)
class ArxivRecord:
    arxiv_id: str
    title: str
    abstract: str
    authors: list[ArxivAuthor]
    published: str = ""
    updated: str = ""
    categories: list[str] = field(default_factory=list)
    primary_category: str = ""
    abs_url: str = ""
    pdf_url: str = ""
    doi: str = ""
    doi_url: str = ""
    journal_ref: str = ""
    comment: str = ""
    license_url: str = ""


# Parsing

# Unescape HTML entities and collapse whitespace; None becomes an empty string.
def _clean(value: Any) -> str:
    return _SPACE_RE.sub(" ", html_lib.unescape(str(value or ""))).strip()


# Upgrade a plain http:// URL to https://; other values are returned trimmed.
def _https_url(value: str) -> str:
    url = str(value or "").strip()
    return "https://" + url.removeprefix("http://") if url.startswith("http://") else url


# Return a validated arXiv id for an abstract (/abs/) URL, else an empty string. Direct /pdf/ URLs
# deliberately do not match, so they stay on the generic PDF path and read_page extracts the paper itself.
def arxiv_id_from_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").lower().rstrip(".")
    if host not in {"arxiv.org", "www.arxiv.org"} or not parsed.path.startswith("/abs/"):
        return ""
    identifier = unquote(parsed.path.removeprefix("/abs/")).strip("/")
    return identifier if (_NEW_ID_RE.fullmatch(identifier) or _OLD_ID_RE.fullmatch(identifier)) else ""


# Return the cleaned text of a direct child element, or an empty string when it is missing.
def _entry_text(entry: ET.Element, tag: str) -> str:
    node = entry.find(tag)
    return _clean(node.text) if node is not None else ""


# Parse one article from an arXiv Atom API response. Return None for malformed XML, an empty
# feed or the error entry the API returns for unknown ids.
def parse_arxiv_atom(xml_text: str, *, requested_id: str) -> ArxivRecord | None:
    # Locate the entry and reject the API's error entries.
    try:
        root = ET.fromstring(xml_text)
    except (ET.ParseError, TypeError, ValueError):
        return None
    entry = root.find(f"{_ATOM}entry")
    if entry is None:
        return None
    entry_id = _entry_text(entry, f"{_ATOM}id")
    title = _entry_text(entry, f"{_ATOM}title")
    if not title or title.casefold() == "error" or "/api/errors#" in entry_id:
        return None

    # Authors with their affiliations.
    authors: list[ArxivAuthor] = []
    for author_node in entry.findall(f"{_ATOM}author"):
        name = _entry_text(author_node, f"{_ATOM}name")
        affiliations = [
            _clean(node.text)
            for node in author_node.findall(f"{_ARXIV}affiliation")
            if _clean(node.text)
        ]
        if name:
            authors.append(ArxivAuthor(name=name, affiliations=affiliations))

    # Links: the first alternate link is the abstract page; PDF and DOI links are told apart by
    # their title or MIME type.
    abs_url = ""
    pdf_url = ""
    doi_url = ""
    for link in entry.findall(f"{_ATOM}link"):
        href = _https_url(link.get("href") or "")
        rel = str(link.get("rel") or "").lower()
        title_attr = str(link.get("title") or "").lower()
        mime = str(link.get("type") or "").lower()
        if rel == "alternate" and not abs_url:
            abs_url = href
        if title_attr == "pdf" or mime == "application/pdf":
            pdf_url = href
        elif title_attr == "doi":
            doi_url = href

    # Canonical id from the abstract link, with fallbacks built from it for the DOI, abstract and PDF links.
    canonical_id = requested_id
    if abs_url and "/abs/" in abs_url:
        canonical_id = abs_url.split("/abs/", 1)[1].strip("/") or requested_id
    doi = _entry_text(entry, f"{_ARXIV}doi")
    if doi and not doi_url:
        doi_url = f"https://doi.org/{quote(doi, safe='/():;.-_')}"
    if not abs_url:
        abs_url = f"https://arxiv.org/abs/{canonical_id}"
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{canonical_id}"

    # Categories (deduplicated while building the record) and the license link.
    categories = [
        _clean(node.get("term"))
        for node in entry.findall(f"{_ATOM}category")
        if _clean(node.get("term"))
    ]
    primary_node = entry.find(f"{_ARXIV}primary_category")
    primary_category = _clean(primary_node.get("term")) if primary_node is not None else ""
    license_node = entry.find(f"{_ARXIV}license")
    license_url = ""
    if license_node is not None:
        license_url = _https_url(license_node.get("href") or _clean(license_node.text))

    return ArxivRecord(
        arxiv_id=canonical_id,
        title=title,
        abstract=_entry_text(entry, f"{_ATOM}summary"),
        authors=authors,
        published=_entry_text(entry, f"{_ATOM}published"),
        updated=_entry_text(entry, f"{_ATOM}updated"),
        categories=list(dict.fromkeys(categories)),
        primary_category=primary_category,
        abs_url=abs_url,
        pdf_url=pdf_url,
        doi=doi,
        doi_url=doi_url,
        journal_ref=_entry_text(entry, f"{_ARXIV}journal_ref"),
        comment=_entry_text(entry, f"{_ARXIV}comment"),
        license_url=license_url,
    )


# Fetching

# GET a URL and return its decoded body, failing with ValueError once the response (declared or
# streamed) exceeds max_bytes. The body is streamed so an oversized page is never fully downloaded.
async def _bounded_get(url: str, *, timeout: float, max_bytes: int, params: dict[str, str] | None = None) -> str:
    import httpx

    headers = {"User-Agent": _UA, "Accept": "application/atom+xml,text/html;q=0.9,*/*;q=0.1"}
    chunks: list[bytes] = []
    total = 0
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=headers) as client:
        async with client.stream("GET", url, params=params) as response:
            response.raise_for_status()
            declared = int(response.headers.get("content-length") or 0)
            if declared > max_bytes:
                raise ValueError(f"arXiv response exceeds {max_bytes} bytes")
            async for chunk in response.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError(f"arXiv response exceeds {max_bytes} bytes")
                chunks.append(chunk)
            return b"".join(chunks).decode(response.encoding or "utf-8", errors="replace")


# Query the Atom API for one article inside the shared arXiv API rate-limit slot.
async def _fetch_atom(arxiv_id: str, timeout: float) -> ArxivRecord | None:
    async with arxiv_api_slot():
        xml_text = await _bounded_get(
            ARXIV_API_ENDPOINT,
            timeout=timeout,
            max_bytes=_API_MAX_BYTES,
            params={"id_list": arxiv_id, "max_results": "1"},
        )
    return parse_arxiv_atom(xml_text, requested_id=arxiv_id)


# References

# Flatten one bibliography item into a single line, keeping its label and turning anchors into
# markdown links so DOI, arXiv and publisher links stay actionable.
def _reference_text(item: Any, base_url: str) -> str:
    clone = item.__copy__()
    label_node = clone.select_one(".ltx_tag_bibitem")
    label = _clean(label_node.get_text(" ", strip=True)) if label_node is not None else ""
    if label_node is not None:
        label_node.decompose()
    for anchor in clone.select("a[href]"):
        label_text = _clean(anchor.get_text(" ", strip=True)) or "link"
        href = urljoin(base_url, str(anchor.get("href") or ""))
        anchor.replace_with(f"[{label_text}]({href})" if href else label_text)
    text = _clean(clone.get_text(" ", strip=True))
    return f"{label} {text}".strip()


# Extract bibliography entries from the official arXiv HTML full text. Return the first
# _MAX_REFERENCES rendered entries and the total number found; the result is empty when
# BeautifulSoup is unavailable.
def parse_arxiv_html_references(raw_html: str, *, base_url: str) -> tuple[list[str], int]:
    try:
        from bs4 import BeautifulSoup
    except Exception:
        # The HTML parser is optional; without it the page just has no reference list.
        return [], 0
    soup = BeautifulSoup(raw_html or "", "lxml")
    items = list(soup.select("section.ltx_bibliography li.ltx_bibitem"))
    if not items:
        items = list(soup.select(".ltx_bibliography .ltx_bibitem"))
    references: list[str] = []
    for index, item in enumerate(items, 1):
        text = _reference_text(item, base_url)
        if text:
            references.append(_UNDEFINED_REFERENCE_LABEL_RE.sub(f"[{index}] ", text))
    return references[:_MAX_REFERENCES], len(references)


# Download the official HTML full text and return its references, their total count and the HTML URL.
# The references are empty when the HTML conversion does not exist or cannot be fetched.
async def _fetch_html_references(arxiv_id: str, timeout: float) -> tuple[list[str], int, str]:
    html_url = f"https://arxiv.org/html/{arxiv_id}"
    try:
        raw_html = await _bounded_get(html_url, timeout=timeout, max_bytes=_HTML_MAX_BYTES)
    except Exception:
        # Not every paper has an HTML version; the caller renders a fallback note instead.
        return [], 0, html_url
    references, total = parse_arxiv_html_references(raw_html, base_url=html_url)
    return references, total, html_url


# Markdown output

# Join the authors into one line, appending each author's affiliations in parentheses.
def _format_authors(authors: list[ArxivAuthor]) -> str:
    rendered: list[str] = []
    for author in authors:
        suffix = f" ({'; '.join(author.affiliations)})" if author.affiliations else ""
        rendered.append(f"{author.name}{suffix}")
    return ", ".join(rendered)


# Render the record as markdown: a header with links and metadata, then authors, abstract and
# references, truncated to max_chars when a budget is set.
def _record_to_markdown(
    record: ArxivRecord,
    *,
    references: list[str],
    reference_total: int,
    html_url: str,
    api_url: str,
    max_chars: int,
) -> str:
    # Title and the fixed link block.
    version_match = _VERSION_RE.search(record.arxiv_id)
    source_url = f"https://arxiv.org/src/{record.arxiv_id}"
    lines = [
        f"# {record.title}",
        "",
        "**Source:** arXiv",
        f"**arXiv ID:** {record.arxiv_id}",
        f"**Abstract page:** {record.abs_url}",
        f"**PDF:** {record.pdf_url}",
        f"**HTML full text:** {html_url}",
        f"**Source archive:** {source_url}",
        f"**Metadata API:** {api_url}",
    ]

    # Optional metadata lines, only for the fields the API returned.
    if version_match:
        lines.append(f"**Version:** v{version_match.group(1)}")
    if record.published:
        lines.append(f"**First submitted:** {record.published}")
    if record.updated:
        lines.append(f"**Last updated:** {record.updated}")
    if record.primary_category:
        lines.append(f"**Primary category:** {record.primary_category}")
    if record.categories:
        lines.append(f"**Categories:** {', '.join(record.categories)}")
    if record.doi:
        doi_value = f"[{record.doi}]({record.doi_url})" if record.doi_url else record.doi
        lines.append(f"**DOI:** {doi_value}")
    if record.journal_ref:
        lines.append(f"**Journal reference:** {record.journal_ref}")
    if record.comment:
        lines.append(f"**Author comment:** {record.comment}")
    if record.license_url:
        lines.append(f"**License:** {record.license_url}")

    # Authors, abstract and the reference list (or a note that no bibliography was available).
    if record.authors:
        lines.extend(["", "## Authors", "", _format_authors(record.authors)])
    lines.extend(["", "## Abstract", "", record.abstract or "*No abstract returned by the API.*"])
    lines.extend(["", "## References", ""])
    if references:
        shown = len(references)
        lines.append(
            f"Extracted from the official arXiv HTML full text ({shown} of {reference_total} entries)."
        )
        lines.append("")
        lines.extend(f"- {reference}" for reference in references)
    else:
        lines.append(
            "The arXiv metadata API does not include the bibliography, and an official HTML "
            f"bibliography was not available for this article. Read the references in the [PDF]({record.pdf_url})."
        )

    # Join and truncate to the character budget.
    markdown = "\n".join(lines).strip()
    if max_chars and len(markdown) > max_chars:
        from core.extract.content_processor import _truncate_markdown_to_budget

        markdown = _truncate_markdown_to_budget(markdown, max_chars)
    return markdown


# Page reading and handler

# Fetch an arXiv abstract page as markdown, or an "Error: ..." string on failure. The Atom API is
# authoritative for metadata and stable links; it does not expose the bibliography, so references
# are supplemented from arXiv's official HTML full text when that conversion exists.
async def fetch_arxiv_page(url: str, *, timeout: float = 20.0, max_chars: int = 20_000) -> str:
    arxiv_id = arxiv_id_from_url(url)
    if not arxiv_id:
        return f"Error: Unsupported arXiv abstract URL: {url}"

    # Query the API and the HTML references concurrently; only the API result is mandatory.
    atom_task = asyncio.create_task(_fetch_atom(arxiv_id, timeout))
    refs_task = asyncio.create_task(_fetch_html_references(arxiv_id, timeout))
    record_result, refs_result = await asyncio.gather(atom_task, refs_task, return_exceptions=True)
    if isinstance(record_result, Exception) or record_result is None:
        detail = str(record_result) if isinstance(record_result, Exception) else "empty API response"
        return f"Error: arXiv API fetch failed for {url}: {detail}"

    # A failed HTML fetch degrades to an empty reference list.
    references, reference_total, html_url = (
        refs_result if not isinstance(refs_result, Exception) else ([], 0, f"https://arxiv.org/html/{arxiv_id}")
    )
    api_url = f"{ARXIV_API_ENDPOINT}?id_list={quote(arxiv_id, safe='/')}&max_results=1"
    return _record_to_markdown(
        record_result,
        references=references,
        reference_total=reference_total,
        html_url=html_url,
        api_url=api_url,
        max_chars=max_chars,
    )


# Handler for arXiv abstract pages (read_page only, falls back to the generic pipeline on errors).
class ArxivHandler:
    name = "arxiv"
    scope = SCOPE_READ_PAGE
    fallback_to_generic = True

    # Return True for arXiv abstract URLs.
    def matches(self, url: str) -> bool:
        return bool(arxiv_id_from_url(url))

    # Fetch the abstract page; an "Error:" result is reported as a failed PageResult.
    async def read(self, url: str, ctx: FetchContext) -> PageResult:
        markdown = await fetch_arxiv_page(url, timeout=ctx.timeout, max_chars=ctx.max_chars)
        ok = bool(markdown) and not markdown.lstrip().lower().startswith("error:")
        return PageResult(
            markdown=markdown,
            ok=ok,
            method="arxiv_api",
            apply_budget=False,
            error="" if ok else markdown,
        )


# Handler instance registered in custom_domains/__init__.py.
HANDLER = ArxivHandler()

__all__ = [
    "HANDLER",
    "ArxivAuthor",
    "ArxivHandler",
    "ArxivRecord",
    "arxiv_id_from_url",
    "fetch_arxiv_page",
    "parse_arxiv_atom",
    "parse_arxiv_html_references",
]
