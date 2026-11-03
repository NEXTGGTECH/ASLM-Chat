# Copyright NEXTGGTECH. Elastic License 2.0.

from core.fetch.hackernews_fetcher import fetch_hackernews_item, is_hackernews_item_url

from custom_domains.base import FetchContext, PageResult

__all__ = [
    "HANDLER",
    "HackerNewsHandler",
    "fetch_hackernews_item",
    "is_hackernews_item_url",
]


# Terminal handler: HN item pages via the Algolia API (the site itself 429s by IP,
# browser included, so there is no generic fallback worth taking).
class HackerNewsHandler:
    name = "hackernews"
    fallback_to_generic = False

    # Return True for news.ycombinator.com/item?id=NNN URLs.
    def matches(self, url: str) -> bool:
        return is_hackernews_item_url(url)

    # Fetch and format the story/comment thread as markdown.
    async def read(self, url: str, ctx: FetchContext) -> PageResult:
        markdown = await fetch_hackernews_item(url, timeout=ctx.timeout)
        ok = bool(markdown) and not markdown.lstrip().lower().startswith("error:")
        return PageResult(
            markdown=markdown or f"Error: Hacker News fetch failed for {url}",
            ok=ok,
            method="hackernews_api",
            error="" if ok else (markdown or "hackernews fetch failed"),
        )


# Handler instance registered in custom_domains/__init__.py.
HANDLER = HackerNewsHandler()
