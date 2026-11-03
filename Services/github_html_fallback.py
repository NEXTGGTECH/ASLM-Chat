# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlencode, urljoin, urlparse


# Map from GitHub MCP search tool name to the "type" value of the public github.com/search page.
GITHUB_SEARCH_TYPE_BY_TOOL = {
    "search_code": "code",
    "search_commits": "commits",
    "search_issues": "issues",
    "search_pull_requests": "pullrequests",
    "search_repositories": "repositories",
    "search_users": "users",
}

# Lowercase phrases in a GitHub MCP result that identify an API rate-limit failure.
_RATE_LIMIT_MARKERS = (
    "api rate limit exceeded",
    "403 rate limit exceeded",
    "secondary rate limit",
)

# First path segments of github.com site pages, which are never an owner, so a link to them is
# not a user or repository search result.
_RESERVED_SINGLE_SEGMENTS = {
    "about", "account", "apps", "collections", "contact", "customer-stories",
    "enterprise", "events", "features", "issues", "login", "marketplace", "mcp",
    "new", "notifications", "open-source", "organizations", "orgs", "pricing",
    "pulls", "search", "security", "settings", "signup", "site", "sponsors",
    "topics", "trending", "users", "why-github",
}


# Return whether a GitHub MCP result represents an API rate-limit failure.
def is_github_rate_limit_result(value: Any) -> bool:
    text = str(value or "").lower()
    return any(marker in text for marker in _RATE_LIMIT_MARKERS) or (
        "rate limit" in text and "docs.github.com/rest" in text
    )


# Build the search query string from the tool arguments, scoping it to owner/repo when given and
# adding the is:issue / is:pr qualifier the issue and pull request tools imply.
def _search_query(tool_name: str, arguments: dict[str, Any]) -> str:
    query = re.sub(r"\s+", " ", str(arguments.get("query") or "")).strip()
    owner = re.sub(r"\s+", "", str(arguments.get("owner") or "")).strip()
    repo = re.sub(r"\s+", "", str(arguments.get("repo") or "")).strip()
    if owner and repo and f"repo:{owner}/{repo}".lower() not in query.lower():
        query = f"{query} repo:{owner}/{repo}".strip()
    if tool_name == "search_issues" and "is:issue" not in query.lower():
        query = f"{query} is:issue".strip()
    if tool_name == "search_pull_requests" and "is:pr" not in query.lower():
        query = f"{query} is:pr".strip()
    return query


# Build the public github.com/search URL corresponding to one MCP search call.
# Return an empty string when the tool has no HTML equivalent or the query is empty.
def build_github_html_search_url(tool_name: str, arguments: dict[str, Any]) -> str:
    search_type = GITHUB_SEARCH_TYPE_BY_TOOL.get(str(tool_name or ""))
    if not search_type:
        return ""
    query = _search_query(tool_name, arguments)
    if not query:
        return ""
    params: dict[str, Any] = {"q": query, "type": search_type}
    page = arguments.get("page")
    try:
        if int(page or 1) > 1:
            params["p"] = int(page)
    except (TypeError, ValueError):
        # A malformed page number is ignored, which leaves the first page.
        pass
    sort = str(arguments.get("sort") or "").strip()
    order = str(arguments.get("order") or "").strip().lower()
    if sort:
        params["s"] = sort
    if order in {"asc", "desc"}:
        params["o"] = order
    return "https://github.com/search?" + urlencode(params)


# Return whether a link points to a result of the given search type, judged by the shape of its
# github.com path (owner/repo, /commit/, /blob/, /issues/N or /pull/N).
def _candidate_href(href: str, search_type: str) -> bool:
    parsed = urlparse(urljoin("https://github.com", str(href or "")))
    if parsed.netloc.lower() not in {"github.com", "www.github.com"}:
        return False
    parts = [part for part in parsed.path.split("/") if part]
    if search_type == "repositories":
        return len(parts) == 2 and parts[0].lower() not in _RESERVED_SINGLE_SEGMENTS
    if search_type == "users":
        return len(parts) == 1 and parts[0].lower() not in _RESERVED_SINGLE_SEGMENTS
    if search_type == "commits":
        return len(parts) >= 4 and parts[2] == "commit"
    if search_type == "code":
        return len(parts) >= 5 and parts[2] == "blob"
    if search_type == "issues":
        return len(parts) >= 4 and parts[2] == "issues" and parts[3].isdigit()
    if search_type == "pullrequests":
        return len(parts) >= 4 and parts[2] == "pull" and parts[3].isdigit()
    return False


# Extract compact result links and snippets from a GitHub HTML search page.
# Return at most limit results (clamped to 1..20) as dicts with title, url and snippet.
def parse_github_html_search(
    html: str,
    *,
    search_type: str,
    limit: int = 10,
) -> list[dict[str, str]]:
    from bs4 import BeautifulSoup

    # Locate the results list and the per-result containers (class prefix of the page's CSS module).
    soup = BeautifulSoup(str(html or ""), "html.parser")
    root = soup.select_one('[data-testid="results-list"]') or soup
    containers = []
    for node in root.find_all("div"):
        classes = [str(value) for value in node.get("class", [])]
        if any(value.startswith("Result-module__Result__") for value in classes):
            containers.append(node)

    # Take the first matching link of each container (or of the whole page when no container was
    # found), skip duplicates, and strip the repeated title from the snippet.
    results: list[dict[str, str]] = []
    seen: set[str] = set()
    search_nodes = containers or [root]
    for container in search_nodes:
        candidate = next(
            (
                anchor
                for anchor in container.find_all("a", href=True)
                if _candidate_href(anchor.get("href", ""), search_type)
            ),
            None,
        )
        if candidate is None:
            continue
        url = urljoin("https://github.com", candidate.get("href", ""))
        if url in seen:
            continue
        seen.add(url)
        title = re.sub(r"\s+", " ", candidate.get_text(" ", strip=True)).strip() or url
        snippet = re.sub(r"\s+", " ", container.get_text(" ", strip=True)).strip()
        if snippet.lower().startswith(title.lower()):
            snippet = snippet[len(title):].strip(" -·")
        results.append({"title": title, "url": url, "snippet": snippet[:700]})
        if len(results) >= max(1, min(int(limit or 10), 20)):
            break
    return results


# Execute a rate-limit fallback through GitHub's public HTML search page.
# Return a Markdown-style result text, a GITHUB_HTML_FALLBACK_NO_RESULTS message, or an empty
# string when the tool has no HTML equivalent.
def search_github_html(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    timeout: float = 20.0,
) -> str:
    import httpx

    search_type = GITHUB_SEARCH_TYPE_BY_TOOL.get(str(tool_name or ""))
    url = build_github_html_search_url(tool_name, arguments)
    if not search_type or not url:
        return ""

    # Download the public search page.
    response = httpx.get(
        url,
        headers={
            "Accept": "text/html,application/xhtml+xml",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
            ),
        },
        follow_redirects=True,
        timeout=max(3.0, float(timeout)),
    )
    response.raise_for_status()

    # Parse the results, honoring the perPage argument as the result limit.
    try:
        requested_limit = int(arguments.get("perPage") or 10)
    except (TypeError, ValueError):
        requested_limit = 10
    results = parse_github_html_search(
        response.text,
        search_type=search_type,
        limit=requested_limit,
    )
    if not results:
        # Public code search needs a signed-in session, so explain that instead of a bare "no results".
        page_text = response.text.lower()
        if search_type == "code" and ("sign in" in page_text or "code search" in page_text):
            return (
                "GITHUB_HTML_FALLBACK_NO_RESULTS: public HTML code search requires a signed-in "
                f"GitHub session. Search URL: {url}"
            )
        return f"GITHUB_HTML_FALLBACK_NO_RESULTS: no public results parsed. Search URL: {url}"

    # Format the results as a numbered Markdown list under a fallback notice.
    lines = [
        "GITHUB_API_RATE_LIMIT_FALLBACK: GitHub REST API was rate-limited; public HTML search succeeded.",
        f"Search URL: {url}",
        "Results:",
    ]
    for index, result in enumerate(results, start=1):
        lines.append(f"{index}. [{result['title']}]({result['url']})")
        if result["snippet"]:
            lines.append(f"   {result['snippet']}")
    return "\n".join(lines)
