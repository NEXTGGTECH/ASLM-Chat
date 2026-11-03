# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# Handler scope: where a domain's bespoke handler is allowed to run.
#   both       — usable by web_search inline parsing AND the read_page tool (default);
#   read_page  — read_page tool only; web_search keeps the source as a snippet and never
#                parses it inline (handlers that need a browser or a slow API: reddit, x,
#                ebay, youtube). Cheap handlers (github/Wikipedia/stackexchange APIs) stay both.
SCOPE_BOTH = "both"
SCOPE_READ_PAGE = "read_page"


# Final outcome of reading one URL through a handler or the generic pipeline. `method` is the
# winning fetch method (e.g. "github_api", "browser"); `apply_budget` asks read_page to run its
# BM25 compression budget on the markdown.
@dataclass(slots=True)
class PageResult:
    markdown: str = ""
    ok: bool = False
    method: str = ""
    blocked: bool = False
    apply_budget: bool = False
    error: str = ""


# Parameters that steer read_page's generic fetch+normalise pipeline. Strategy handlers
# build one of these and hand it to ctx.generic_read.
@dataclass(slots=True)
class GenericRequest:
    url: str
    browser_first: bool = False
    prefer_rsc: bool = False
    url_variants: list[str] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)


# One recorded fetch attempt, for trace output and runtime-profile accounting.
@dataclass(slots=True)
class PageAttempt:
    url: str
    method: str = ""
    variant: str = ""
    user_agent: str = ""
    status: int = 0
    fetch_ms: float = 0.0
    parse_ms: float = 0.0
    markdown_length: int = 0
    weak: bool = True
    blocked: bool = False
    winner: bool = False


# Capabilities injected by read_page so handlers can fetch without importing read_page
# (avoids an import cycle). `generic_read` is the generic pipeline; `attempts` collects
# trace records appended by both handlers and the generic pipeline.
@dataclass(slots=True)
class FetchContext:
    timeout: float
    max_chars: int
    focus: str
    cfg: Any
    cache: Any
    generic_read: Callable[["GenericRequest"], Awaitable[PageResult]]
    collect_attempts: bool = False
    attempts: list[PageAttempt] = field(default_factory=list)


# A custom-domain handler: the unified contract of the pass-through layer, which keeps read_page
# thin (it asks `match(url)` for a handler and calls `handler.read(url, ctx)`). `matches` is a
# cheap host/path test; `read` does the work. Two handler shapes share this one API:
#   - terminal handlers (github, Wikipedia, reddit, x, stackexchange, amazon, ebay, youtube)
#     produce their own final markdown;
#   - strategy handlers (retail dns-shop/citilink, twitch, cursor, ...) reuse read_page's
#     generic pipeline via `ctx.generic_read(...)` with per-domain parameters, so domain
#     quirks live here and never bloat read_page.
# `fallback_to_generic` lets terminal handlers (amazon/ebay) defer to the generic
# pipeline when their specialised fetch comes back empty/blocked.
@runtime_checkable
class DomainHandler(Protocol):
    name: str
    fallback_to_generic: bool
    # Optional; consumers read it via getattr(handler, "scope", SCOPE_BOTH).
    scope: str

    # Return True when this handler is responsible for the URL.
    def matches(self, url: str) -> bool: ...

    # Fetch and normalise the URL into a PageResult.
    async def read(self, url: str, ctx: FetchContext) -> PageResult: ...
