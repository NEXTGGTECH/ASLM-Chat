# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

# Make the web search root importable when this file runs as a standalone script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from custom_domains.common import extract_with_preferred_pipeline, looks_blocked, trim


# Fetch an eBay listing via the warm cloakbrowser and return an extraction snapshot dict.
async def fetch_ebay_snapshot(url: str, timeout: float = 20.0, wait: float = 4.0) -> dict[str, Any]:
    started = time.perf_counter()
    candidates: list[dict[str, Any]] = []
    attempts_per_engine = 2

    # Warm cloakbrowser — ~1.4s warm with full HTML on eBay. Routed through the shared
    # browser client (warm daemon, autostarted on first call).
    from core.fetch.browser.client import browser_fetch

    # Retry the browser fetch until a non-blocked page with HTML comes back, then extract it.
    for attempt in range(1, attempts_per_engine + 1):
        result = await browser_fetch(url, nav_timeout=timeout, wait_sec=wait)
        html = result.html or ""
        title = trim(result.title, 180)
        blocked = result.blocked or looks_blocked(title, html)
        if html and not blocked:
            parsed = extract_with_preferred_pipeline(url, html, prefer_trafilatura=True)
            candidates.append(
                {
                    "source": "ebay",
                    "url": url,
                    "engine": "cloakbrowser",
                    "attempt": attempt,
                    "status": 200 if result.ok else 0,
                    "final_url": url,
                    "content_type": "",
                    "title": title,
                    "raw_html_chars": len(html),
                    "blocked_like": blocked,
                    **parsed,
                }
            )
            break

    # Return the richest candidate, or an empty "blocked" snapshot when every attempt failed.
    if candidates:
        best = max(candidates, key=lambda item: (int(item["markdown_chars"]), int(item["raw_html_chars"])))
        best["duration_sec"] = round(time.perf_counter() - started, 3)
        return best

    return {
        "source": "ebay",
        "url": url,
        "engine": "none",
        "status": 0,
        "final_url": url,
        "content_type": "",
        "title": "",
        "raw_html_chars": 0,
        "blocked_like": True,
        "duration_sec": round(time.perf_counter() - started, 3),
        "strategy": "none",
        "markdown": "",
        "markdown_chars": 0,
        "markdown_preview": "",
    }


# CLI argument parser for standalone ebay.py runs.
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the custom eBay snapshot fetcher.")
    parser.add_argument("url")
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--wait", type=float, default=4.0)
    return parser.parse_args()


# CLI entry: fetch snapshot and print JSON to stdout.
def main() -> None:
    args = _parse_args()
    result = asyncio.run(fetch_ebay_snapshot(args.url, timeout=args.timeout, wait=args.wait))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


from urllib.parse import urlparse

from custom_domains.base import FetchContext, PageResult

# eBay marketplace hosts (without the www./m. prefix) that this handler serves.
_EBAY_HOSTS = frozenset({
    "ebay.com", "ebay.co.uk", "ebay.de", "ebay.fr", "ebay.it",
})


# Unified handler: fetch an eBay listing snapshot via the warm cloakbrowser.
class EbayHandler:
    name = "ebay"
    fallback_to_generic = True
    # The retail page is heavy and behind anti-bot checks, so web_search keeps it snippet-only.
    scope = "read_page"

    # Return True for known eBay marketplace hosts.
    def matches(self, url: str) -> bool:
        host = urlparse(url).netloc.lower().removeprefix("www.").removeprefix("m.")
        return host in _EBAY_HOSTS

    # Fetch a snapshot; empty/blocked results defer to read_page's generic pipeline.
    async def read(self, url: str, ctx: FetchContext) -> PageResult:
        try:
            snapshot = await fetch_ebay_snapshot(url, timeout=ctx.timeout)
        except Exception:
            return PageResult(ok=False, method="ebay_custom", error="ebay snapshot failed")
        markdown = str(snapshot.get("markdown") or "").strip()
        blocked = bool(snapshot.get("blocked_like"))
        ok = bool(markdown) and not blocked
        return PageResult(
            markdown=markdown,
            ok=ok,
            method=str(snapshot.get("engine") or "ebay_custom"),
            blocked=blocked,
        )


# Handler instance registered in custom_domains/__init__.py.
HANDLER = EbayHandler()
