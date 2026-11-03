# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import html
import re
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from core.fetch.thread_pool import io_pool as _io_pool

# Pattern capturing the numeric question id from a /questions/{id} path, and the Stack Exchange
# API site parameter of the stand-alone domains (network sites are derived from the subdomain).
_QUESTION_RE = re.compile(r"/questions/(\d+)(?:/|$)")
_HOST_TO_SITE = {
    "stackoverflow.com": "stackoverflow",
    "superuser.com": "superuser",
    "serverfault.com": "serverfault",
    "askubuntu.com": "askubuntu",
}


# Return the normalized host of a URL: lowercase, without a leading www.
def _host(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


# Map a Stack Exchange hostname to its API site parameter, or None for other hosts.
def _site_from_host(host: str) -> str | None:
    if host in _HOST_TO_SITE:
        return _HOST_TO_SITE[host]
    if host.endswith(".stackexchange.com"):
        return host.split(".", 1)[0]
    return None


# Extract the numeric question id from a /questions/{id}/ URL path, or None.
def stackexchange_question_id(url: str) -> str | None:
    m = _QUESTION_RE.search(urlparse(url).path)
    return m.group(1) if m else None


# Return True when url is a supported Stack Exchange question page.
def is_stackexchange_question_url(url: str) -> bool:
    return _site_from_host(_host(url)) is not None and stackexchange_question_id(url) is not None


# Convert an API HTML fragment to plain text (code blocks preserved as fenced).
def _strip_html_fragment(fragment: str) -> str:
    # Preferred path: parse with BeautifulSoup, dropping scripts and styles.
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(fragment or "", "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        for pre in soup.find_all("pre"):
            code_text = pre.get_text("\n", strip=False)
            pre.replace_with("\n```text\n" + code_text.strip() + "\n```\n")
        text = soup.get_text("\n", strip=True)
        return html.unescape(text).strip()
    except Exception:
        # Fallback when BeautifulSoup is unavailable or fails: crude regex tag stripping.
        text = re.sub(r"<br\s*/?>", "\n", fragment or "", flags=re.IGNORECASE)
        text = re.sub(r"</p\s*>", "\n\n", text, flags=re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "", text)
        return html.unescape(text).strip()


# Format a Unix epoch as ISO-8601 UTC with a Z suffix; an invalid value gives "".
def _format_timestamp(value: object) -> str:
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return ""


# Fetch a Stack Exchange API URL (blocking) and return the parsed JSON; HTTP errors raise.
def _stackexchange_api_get(url: str, timeout: int) -> dict[str, Any]:
    from curl_cffi import requests as _r

    resp = _r.get(
        url,
        impersonate="chrome124",
        timeout=timeout,
        headers={
            "Accept": "application/json,text/plain,*/*",
            "Accept-Language": "en-US,en;q=0.9",
            "User-Agent": "Mozilla/5.0",
        },
    )
    resp.raise_for_status()
    return resp.json()


# Return the display name from the API owner object, or "".
def _owner_name(item: dict[str, Any]) -> str:
    owner = item.get("owner") if isinstance(item.get("owner"), dict) else {}
    return str(owner.get("display_name") or "").strip()


# Normalize one comment item from API JSON.
def _normalize_comment(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "comment_id": int(item.get("comment_id") or 0),
        "score": int(item.get("score") or 0),
        "created_at": _format_timestamp(item.get("creation_date")),
        "author": _owner_name(item),
        "body_text": _strip_html_fragment(str(item.get("body") or "")),
    }


# Normalize one answer item plus its comment list.
def _normalize_answer(item: dict[str, Any], comments: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "answer_id": int(item.get("answer_id") or 0),
        "score": int(item.get("score") or 0),
        "is_accepted": bool(item.get("is_accepted")),
        "created_at": _format_timestamp(item.get("creation_date")),
        "last_activity_at": _format_timestamp(item.get("last_activity_date")),
        "author": _owner_name(item),
        "body_markdown": _strip_html_fragment(str(item.get("body") or "")),
        "comment_count": len(comments),
        "comments": comments,
    }


# Group comment items by their post_id (or other parent id field), normalizing each comment;
# items without a valid parent id are skipped.
def _comments_by_post_id(items: list[dict[str, Any]], id_field: str) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for item in items:
        try:
            key = int(item.get(id_field) or 0)
        except Exception:
            # A malformed id is treated as missing and skipped below.
            key = 0
        if not key:
            continue
        grouped.setdefault(key, []).append(_normalize_comment(item))
    return grouped


# Fetch the question, its top answers and their comments via the Stack Exchange API (blocking)
# and return them as a dict with "ok": True, or "ok": False plus an "error" message when the URL
# is unsupported or the API returns no question. HTTP failures of the API calls raise.
def fetch_stackexchange_question_data_sync(
    url: str,
    timeout: float = 20.0,
    answer_limit: int = 3,
    comment_limit: int = 5,
) -> dict[str, Any]:
    # Identify the site and question id from the URL.
    host = _host(url)
    site = _site_from_host(host)
    qid = stackexchange_question_id(url)
    if not site or not qid:
        return {
            "ok": False,
            "error": f"Unsupported Stack Exchange question URL: {url}",
            "url": url,
        }

    # API endpoints for the question, its answers (by votes) and its comments, with bodies.
    timeout_int = max(10, int(timeout))
    question_url = f"https://api.stackexchange.com/2.3/questions/{qid}?site={site}&filter=withbody"
    answers_url = (
        f"https://api.stackexchange.com/2.3/questions/{qid}/answers"
        f"?site={site}&filter=withbody&sort=votes"
    )
    question_comments_url = (
        f"https://api.stackexchange.com/2.3/questions/{qid}/comments"
        f"?site={site}&filter=withbody&sort=votes"
    )

    # Fetch the question itself; without it there is nothing to return.
    question_payload = _stackexchange_api_get(question_url, timeout_int)
    question_items = question_payload.get("items") if isinstance(question_payload, dict) else []
    if not isinstance(question_items, list) or not question_items:
        return {
            "ok": False,
            "error": f"Stack Exchange API returned no question data for: {url}",
            "url": url,
        }

    question_item = question_items[0]

    # Fetch the answers and keep only the top answer_limit by votes.
    answers_payload = _stackexchange_api_get(answers_url, timeout_int)
    answer_items = answers_payload.get("items") if isinstance(answers_payload, dict) else []
    if not isinstance(answer_items, list):
        answer_items = []
    answer_items = answer_items[: max(0, int(answer_limit))]

    # Fetch the question's own comments, limited to comment_limit.
    question_comments_payload = _stackexchange_api_get(question_comments_url, timeout_int)
    question_comment_items = question_comments_payload.get("items") if isinstance(question_comments_payload, dict) else []
    if not isinstance(question_comment_items, list):
        question_comment_items = []
    question_comments = [_normalize_comment(item) for item in question_comment_items[: max(0, int(comment_limit))]]

    # Fetch the comments of all kept answers in one batched call, then cap them per answer.
    answer_comments_by_id: dict[int, list[dict[str, Any]]] = {}
    answer_ids = [int(item.get("answer_id") or 0) for item in answer_items if int(item.get("answer_id") or 0)]
    if answer_ids:
        joined_ids = ";".join(str(item) for item in answer_ids)
        answer_comments_url = (
            f"https://api.stackexchange.com/2.3/answers/{joined_ids}/comments"
            f"?site={site}&filter=withbody&sort=votes"
        )
        answer_comments_payload = _stackexchange_api_get(answer_comments_url, timeout_int)
        answer_comment_items = answer_comments_payload.get("items") if isinstance(answer_comments_payload, dict) else []
        if not isinstance(answer_comment_items, list):
            answer_comment_items = []
        limited_comments: list[dict[str, Any]] = []
        per_answer_counts: dict[int, int] = {}
        for item in answer_comment_items:
            try:
                post_id = int(item.get("post_id") or 0)
            except Exception:
                # A malformed post id is treated as missing and skipped below.
                post_id = 0
            if not post_id:
                continue
            if per_answer_counts.get(post_id, 0) >= max(0, int(comment_limit)):
                continue
            limited_comments.append(item)
            per_answer_counts[post_id] = per_answer_counts.get(post_id, 0) + 1
        answer_comments_by_id = _comments_by_post_id(limited_comments, "post_id")

    # Assemble the normalized answers and question.
    answers = [
        _normalize_answer(item, answer_comments_by_id.get(int(item.get("answer_id") or 0), []))
        for item in answer_items
    ]

    question = {
        "question_id": int(question_item.get("question_id") or 0),
        "title": str(question_item.get("title") or "").strip(),
        "author": _owner_name(question_item),
        "created_at": _format_timestamp(question_item.get("creation_date")),
        "last_activity_at": _format_timestamp(question_item.get("last_activity_date")),
        "score": int(question_item.get("score") or 0),
        "view_count": int(question_item.get("view_count") or 0),
        "answer_count": int(question_item.get("answer_count") or 0),
        "is_answered": bool(question_item.get("is_answered")),
        "accepted_answer_id": int(question_item.get("accepted_answer_id") or 0) or None,
        "tags": [str(tag) for tag in (question_item.get("tags") or []) if str(tag).strip()],
        "body_markdown": _strip_html_fragment(str(question_item.get("body") or "")),
        "comment_count": len(question_comments),
        "comments": question_comments,
    }

    # Success payload; quota_remaining reports the API's remaining daily request quota.
    return {
        "ok": True,
        "source": "stackexchange_api",
        "site": site,
        "host": host,
        "url": url,
        "quota_remaining": int(question_payload.get("quota_remaining") or 0),
        "question": question,
        "answers": answers,
    }


# Render the API payload as markdown for read_page / preview: a metadata header, the question
# body, its comments and the top answers with their comments. A failed payload becomes an
# "Error: ..." string.
def render_stackexchange_question_markdown(data: dict[str, Any]) -> str:
    if not data.get("ok"):
        return f"Error: {data.get('error') or 'unknown Stack Exchange fetch failure'}"

    # Read the question and answers defensively; missing parts default to empty values.
    question = data.get("question") if isinstance(data.get("question"), dict) else {}
    answers = data.get("answers") if isinstance(data.get("answers"), list) else []

    title = str(question.get("title") or "").strip() or "Stack Exchange Question"
    body = str(question.get("body_markdown") or "").strip()

    # Metadata header; optional fields appear only when present.
    lines = [f"# {title}", "", f"**Site:** {data.get('host') or data.get('site') or ''}", f"**URL:** {data.get('url') or ''}"]
    if question.get("author"):
        lines.append(f"**Author:** {question['author']}")
    if question.get("created_at"):
        lines.append(f"**Created:** {question['created_at']}")
    if question.get("last_activity_at"):
        lines.append(f"**Last Activity:** {question['last_activity_at']}")
    lines.append(f"**Score:** {int(question.get('score') or 0)}")
    lines.append(f"**Views:** {int(question.get('view_count') or 0)}")
    lines.append(f"**Answer Count:** {int(question.get('answer_count') or 0)}")
    lines.append(f"**Is Answered:** {bool(question.get('is_answered'))}")
    if question.get("accepted_answer_id"):
        lines.append(f"**Accepted Answer ID:** {question['accepted_answer_id']}")
    tags = question.get("tags") if isinstance(question.get("tags"), list) else []
    if tags:
        lines.append(f"**Tags:** {', '.join(str(tag) for tag in tags if str(tag).strip())}")
    lines.extend(["", "---", ""])

    # Question body, then its comments.
    if body:
        lines.append(body)

    question_comments = question.get("comments") if isinstance(question.get("comments"), list) else []
    if question_comments:
        lines.extend(["", "## Question Comments", ""])
        for idx, comment in enumerate(question_comments, 1):
            meta = [f"Comment {idx}"]
            if comment.get("author"):
                meta.append(f"Author: {comment['author']}")
            if comment.get("created_at"):
                meta.append(f"Created: {comment['created_at']}")
            meta.append(f"Score: {int(comment.get('score') or 0)}")
            lines.append(" | ".join(meta))
            if comment.get("body_text"):
                lines.append(str(comment["body_text"]))
            lines.append("")

    # Top answers, each with a header (marking the accepted one), metadata, body and comments.
    if answers:
        lines.extend(["", "## Top Answers", ""])
        for idx, answer in enumerate(answers, 1):
            header = f"### Answer {idx}"
            if answer.get("is_accepted"):
                header += " [accepted]"
            lines.extend([header, ""])
            meta: list[str] = []
            if answer.get("author"):
                meta.append(f"Author: {answer['author']}")
            if answer.get("created_at"):
                meta.append(f"Created: {answer['created_at']}")
            meta.append(f"Score: {int(answer.get('score') or 0)}")
            if meta:
                lines.append(" | ".join(meta))
                lines.append("")
            if answer.get("body_markdown"):
                lines.append(str(answer["body_markdown"]))
                lines.append("")

            answer_comments = answer.get("comments") if isinstance(answer.get("comments"), list) else []
            if answer_comments:
                lines.append("Comments:")
                for comment in answer_comments:
                    meta = []
                    if comment.get("author"):
                        meta.append(str(comment["author"]))
                    if comment.get("created_at"):
                        meta.append(str(comment["created_at"]))
                    meta.append(f"score={int(comment.get('score') or 0)}")
                    prefix = " | ".join(meta)
                    lines.append(f"- {prefix}: {str(comment.get('body_text') or '').strip()}")
                lines.append("")

    return "\n".join(lines).strip()


# Fetch a question (blocking) and return it rendered as markdown.
def fetch_stackexchange_question_sync(url: str, timeout: float = 20.0, answer_limit: int = 3) -> str:
    data = fetch_stackexchange_question_data_sync(url, timeout=timeout, answer_limit=answer_limit)
    return render_stackexchange_question_markdown(data)


# Run fetch_stackexchange_question_data_sync in the shared I/O pool.
async def fetch_stackexchange_question_data(
    url: str,
    timeout: float = 20.0,
    answer_limit: int = 3,
    comment_limit: int = 5,
) -> dict[str, Any]:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        _io_pool,
        lambda: fetch_stackexchange_question_data_sync(
            url,
            timeout=timeout,
            answer_limit=answer_limit,
            comment_limit=comment_limit,
        ),
    )


# Fetch a question without blocking the event loop and return it as markdown for read_page / preview.
async def fetch_stackexchange_question(url: str, timeout: float = 20.0, answer_limit: int = 3) -> str:
    data = await fetch_stackexchange_question_data(url, timeout=timeout, answer_limit=answer_limit)
    return render_stackexchange_question_markdown(data)
