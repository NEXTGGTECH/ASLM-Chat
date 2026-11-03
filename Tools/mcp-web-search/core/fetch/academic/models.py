# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# One scholarly work normalized from any academic provider. Mirrors ShoppingProduct in
# spirit: a flat, citable record the search layer can adapt into a source dict.
@dataclass(slots=True)
class AcademicPaper:
    id: str
    title: str
    url: str
    # Provider name (openalex/crossref/...).
    source: str
    source_domain: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    year: int | None = None
    # Journal, conference or repository name.
    venue: str = ""
    doi: str = ""
    pdf_url: str = ""
    open_access: bool = False
    citations: int | None = None
    confidence: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


# Telemetry for one provider call (mirrors ShoppingProviderAttempt).
@dataclass(slots=True)
class AcademicProviderAttempt:
    provider: str
    method: str
    url: str
    ok: bool
    elapsed_ms: int
    status_code: int | None = None
    papers: int = 0
    error: str = ""


# Outcome of one academic search: the ranked papers, the per-provider attempts, timings and
# whether the hard deadline cut it short (`partial`, with the reason in `partial_reason`).
@dataclass(slots=True)
class AcademicSearchResult:
    query: str
    effort: str
    papers: list[AcademicPaper]
    attempts: list[AcademicProviderAttempt]
    timings: dict[str, Any] = field(default_factory=dict)
    partial: bool = False
    partial_reason: str = ""
