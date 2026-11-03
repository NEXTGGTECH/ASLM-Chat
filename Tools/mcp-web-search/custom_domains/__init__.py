# Copyright NEXTGGTECH. Elastic License 2.0.

from custom_domains.base import (
    SCOPE_BOTH,
    SCOPE_READ_PAGE,
    DomainHandler,
    FetchContext,
    GenericRequest,
    PageAttempt,
    PageResult,
)
from custom_domains.amazon import HANDLER as _amazon
from custom_domains.arxiv import HANDLER as _arxiv
from custom_domains.dns_shop import HANDLER as _dns_shop
from custom_domains.ebay import HANDLER as _ebay
from custom_domains.github import HANDLER as _github
from custom_domains.hackernews import HANDLER as _hackernews
from custom_domains.reddit import HANDLER as _reddit
from custom_domains.stackexchange import HANDLER as _stackexchange
from custom_domains.telegram import HANDLER as _telegram
from custom_domains.wikipedia import HANDLER as _wikipedia
from custom_domains.x import HANDLER as _x
from custom_domains.youtube import HANDLER as _youtube

# Registry of the pass-through custom-domain handlers (single API, see base.py): read_page asks
# match(url) for a handler and calls handler.read(url, ctx). The first matching handler wins; the
# host tests of the handlers below do not overlap.
# Domains needing only a forced fetch method (twitch/flashscore/sofascore -> browser,
# cursor -> nextjs_rsc, citilink -> browser) are NOT handlers: read_page's generic pipeline
# picks that up from core.profiles.known_domains. Only domains with bespoke fetching
# (terminal APIs) or URL-variant logic (dns-shop) get a handler here.
HANDLERS: list[DomainHandler] = [
    _arxiv,
    _github,
    _wikipedia,
    _telegram,
    _reddit,
    _x,
    _stackexchange,
    _hackernews,
    _youtube,
    _amazon,
    _ebay,
    _dns_shop,
]


# Return the first custom-domain handler matching the URL, or None for generic handling.
def match(url: str) -> DomainHandler | None:
    for handler in HANDLERS:
        try:
            if handler.matches(url):
                return handler
        except Exception:
            # A faulty matches() must not break dispatch; skip to the next handler.
            continue
    return None


# Return True when the URL's handler is read_page-only, so web_search must keep it snippet-only
# rather than parse it inline. Domains without a handler (or scope=both) are not affected.
def is_read_page_only(url: str) -> bool:
    handler = match(url)
    return handler is not None and getattr(handler, "scope", SCOPE_BOTH) == SCOPE_READ_PAGE


__all__ = [
    "SCOPE_BOTH",
    "SCOPE_READ_PAGE",
    "DomainHandler",
    "FetchContext",
    "GenericRequest",
    "HANDLERS",
    "PageAttempt",
    "PageResult",
    "is_read_page_only",
    "match",
]
