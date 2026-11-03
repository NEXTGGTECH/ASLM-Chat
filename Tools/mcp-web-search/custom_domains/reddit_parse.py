# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from dataclasses import dataclass, field

from selectolax.lexbor import LexborHTMLParser, LexborNode

# Base URL used to turn root-relative old.reddit links into absolute ones.
_OLD_REDDIT = "https://old.reddit.com"


# Public types

# A Reddit post parsed from old.reddit HTML: author, score, self-vs-link and the self-text body.
@dataclass(slots=True)
class RedditPost:
    id: str | None
    title: str
    author: str
    subreddit: str | None
    score: int
    body: str | None
    num_comments: int
    permalink: str
    url: str | None
    is_self: bool
    flair: str | None
    created_utc: str | None


# One comment with its nesting depth and replies.
@dataclass(slots=True)
class RedditComment:
    id: str | None
    author: str
    body: str
    # None when Reddit hides the score (fresh/deleted comments). Distinct from a
    # genuine net-zero score of 0.
    score: int | None
    depth: int
    is_op: bool
    created_utc: str | None
    replies: list["RedditComment"] = field(default_factory=list)


# A parsed thread: the post (None when it could not be read) and its top-level comments.
@dataclass(slots=True)
class RedditThread:
    source_url: str
    post: RedditPost | None
    comments: list[RedditComment]


# Public API

# Parse a Reddit thread from old.reddit.com HTML. Return None when the page has
# no recognisable Reddit structure (a listing page, an antibot wall, an error).
# old.reddit.com serves fully server-rendered HTML with stable class names and `data-*`
# attributes (no JS, no API key, no `.json` trick), so the structure is read directly, keeping what
# an `inner_text` scrape throws away: each comment's exact score, nesting depth and author, and
# self-vs-link on the post. Ported from the reference `webclaw-core/src/reddit.rs`; the DOM
# contract (which class holds the score, why `.entry` must be scoped to avoid a reply's body, how
# `morechildren` stubs and hidden scores appear) is documented inline where it bites.
def parse_thread(html: str, url: str) -> RedditThread | None:
    # Only thread pages (URLs with /comments/) are parsed.
    if "/comments/" not in url:
        return None
    doc = LexborHTMLParser(html)
    post = _parse_post(doc)
    op = post.author if post else ""
    comments = _parse_comments(doc, op)
    if post is None and not comments:
        return None
    return RedditThread(source_url=url, post=post, comments=comments)


# Parse a thread and render it to markdown, or None when the HTML isn't a thread.
def try_extract_markdown(html: str, url: str) -> str | None:
    thread = parse_thread(html, url)
    if thread is None:
        return None
    md = thread_to_markdown(thread)
    return md or None


# Markdown rendering

# Render a parsed thread as markdown: the post header and body, then the comment tree.
def thread_to_markdown(thread: RedditThread) -> str:
    out: list[str] = []
    p = thread.post

    # Post: title, author line (subreddit, score, comment count), body and external link.
    if p is not None:
        out.append(f"# {p.title}\n\n")
        pts = _pt_label(p.score)
        if p.num_comments == 0:
            cmt = ""
        elif p.num_comments == 1:
            cmt = " · 1 comment"
        else:
            cmt = f" · {p.num_comments} comments"
        sub = p.subreddit or "?"
        out.append(f"**u/{p.author}** · r/{sub} · {pts}{cmt}\n\n")
        if p.body:
            out.append(p.body)
            out.append("\n\n")
        if p.url and not p.is_self:
            out.append(f"[Link]({p.url})\n\n")
        out.append("---\n\n")

    # Comments, depth-first.
    if thread.comments:
        out.append("## Comments\n\n")
        for c in thread.comments:
            _render_comment(c, out)
    return _collapse_blank_lines("".join(out).rstrip())


# Render one comment and its replies. Nesting is expressed with blockquote depth
# (`> ` per level), NOT leading spaces: a 4+ space indent turns ordinary text and
# ``` fences into CommonMark indented code blocks, corrupting any comment depth ≥ 2.
def _render_comment(c: RedditComment, out: list[str]) -> None:
    q = "> " * c.depth
    blank = ">" * c.depth
    author = f"**u/{c.author} [OP]**" if c.is_op else f"**u/{c.author}**"
    out.append(f"{q}{author} · {_pt_label(c.score)}\n")
    for line in c.body.split("\n"):
        if line == "":
            out.append(blank)
            out.append("\n")
        else:
            out.append(q)
            out.append(line)
            out.append("\n")
    out.append("\n")
    for reply in c.replies:
        _render_comment(reply, out)


# Return the score label: "score hidden" for None, otherwise "N pt(s)".
def _pt_label(n: int | None) -> str:
    if n is None:
        return "score hidden"
    if n == 1:
        return "1 pt"
    if n == -1:
        return "-1 pt"
    return f"{n} pts"


# Collapse runs of 3+ newlines to a single blank-line separator so blockquote
# prefixes and <pre> spacing don't leave large gaps.
def _collapse_blank_lines(s: str) -> str:
    out: list[str] = []
    newlines = 0
    for ch in s:
        if ch == "\n":
            newlines += 1
            if newlines <= 2:
                out.append(ch)
        else:
            newlines = 0
            out.append(ch)
    return "".join(out)


# Post parsing

# Parse the post from the first link `.thing` of the site table. Return None when it is missing or
# has no title.
def _parse_post(doc: LexborHTMLParser) -> RedditPost | None:
    thing = doc.css_first("#siteTable .thing.link")
    if thing is None:
        return None
    attrs = thing.attributes

    # Identity and counters come from the `data-*` attributes of the post element.
    fullname = attrs.get("data-fullname")
    post_id = fullname[3:] if fullname and fullname.startswith("t3_") else fullname
    author = attrs.get("data-author") or "[deleted]"
    subreddit = attrs.get("data-subreddit")
    score = _int_attr(attrs.get("data-score"), 0)
    num_comments = _int_attr(attrs.get("data-comments-count"), 0)
    permalink = _OLD_REDDIT + (attrs.get("data-permalink") or "")

    # Self-posts carry the `self` class and a `self.<sub>` domain; their data-url
    # points back at the permalink rather than an external site.
    domain = attrs.get("data-domain") or ""
    is_self = _has_class(thing, "self") or domain.startswith("self.")
    link_url = attrs.get("data-url")
    url = None if is_self else link_url

    # Title (a post without one is not usable), flair, body and timestamp.
    title_node = thing.css_first(".title a.title")
    title = _node_text(title_node) if title_node is not None else ""
    if not title:
        return None

    flair_node = thing.css_first(".linkflairlabel")
    flair = _node_text(flair_node) if flair_node is not None else ""

    entry = _direct_child(thing, "entry")
    body = _extract_md(entry) if entry is not None else None
    created_utc = _created_utc(thing)

    # Assemble the post; an empty flair becomes None.
    return RedditPost(
        id=post_id,
        title=title,
        author=author,
        subreddit=subreddit,
        score=score,
        body=body,
        num_comments=num_comments,
        permalink=permalink,
        url=url,
        is_self=is_self,
        flair=flair or None,
        created_utc=created_utc,
    )


# Comment parsing

# Parse the top-level comments of the thread. old.reddit.com nests comments structurally, not via
# a depth attribute:
#
#   .commentarea
#     .sitetable.nestedlisting
#       .comment.thing                          <- root comment
#         .entry -> form -> .usertext-body -> .md  <- its own body
#         .child
#           .sitetable.listing
#             .comment.thing                    <- reply (recurse)
#
# `data-depth`/`data-replies` are absent or "0" in the logged-out HTML, so we
# walk the tree by recursing into each comment's `.child`.
def _parse_comments(doc: LexborHTMLParser, op: str) -> list[RedditComment]:
    # Root listing is `.sitetable.nestedlisting` inside `.commentarea` (a class,
    # not an id). Fall back to the first `.nestedlisting` anywhere for
    # comment-permalink pages.
    listing = doc.css_first(".commentarea .sitetable.nestedlisting")
    if listing is None:
        listing = doc.css_first(".sitetable.nestedlisting")
    if listing is None:
        return []
    return _walk_comment_level(listing, op, 0)


# Parse the direct-child `.comment.thing` elements of a comment listing.
def _walk_comment_level(listing: LexborNode, op: str, depth: int) -> list[RedditComment]:
    out: list[RedditComment] = []
    for node in listing.iter():
        classes = _classes(node)
        if "comment" in classes and "thing" in classes:
            comment = _parse_one_comment(node, op, depth)
            if comment is not None:
                out.append(comment)
    return out


# Parse one `.comment.thing` node and, recursively, its replies. Return None for "load more" stubs.
def _parse_one_comment(node: LexborNode, op: str, depth: int) -> RedditComment | None:
    attrs = node.attributes
    classes = _classes(node)

    # "load more comments" placeholders are `.thing` with type=morechildren. They
    # carry a t1_ fullname but no real content — skip them (a real .comment.thing
    # never carries the morechildren class, but guard anyway).
    if attrs.get("data-type") == "morechildren" or "morechildren" in classes:
        return None

    is_deleted = "deleted" in classes
    fullname = attrs.get("data-fullname")
    cid = fullname[3:] if fullname and fullname.startswith("t1_") else fullname
    author = attrs.get("data-author") or "[deleted]"

    # Own body lives in `.entry > form > .usertext-body > .md`. `.child` (nested
    # replies) is a sibling of `.entry`, so scoping to the direct-child entry
    # never crosses into a reply's body.
    entry = _direct_child(node, "entry")
    body = _extract_md(entry) if entry is not None else None
    if not body:
        body = "[removed]" if is_deleted else ""

    # Displayed score is `.score.unvoted`, whose `title` holds the exact integer
    # (the sibling likes/dislikes spans are ±1). Hidden-score comments have no
    # `.score.unvoted` span → None, kept distinct from a genuine 0.
    score = _comment_score(entry) if entry is not None else None
    created_utc = _created_utc(entry) if entry is not None else None
    is_op = (not is_deleted) and author != "[deleted]" and author == op

    # Replies: `.comment > .child > .sitetable > .comment`.
    replies: list[RedditComment] = []
    child = _direct_child(node, "child")
    if child is not None:
        sitetable = _direct_child(child, "sitetable")
        if sitetable is not None:
            replies = _walk_comment_level(sitetable, op, depth + 1)

    return RedditComment(
        id=cid,
        author=author,
        body=body,
        score=score,
        depth=depth,
        is_op=is_op,
        created_utc=created_utc,
        replies=replies,
    )


# Read a comment's score from the `.score.unvoted` span inside `.entry`. Prefers
# the `title` attribute (exact integer); fall back to the text ("10 points").
# Return None when Reddit hides the score (no `.score.unvoted` span).
def _comment_score(entry: LexborNode) -> int | None:
    span = entry.css_first("span.score.unvoted")
    if span is None:
        return None
    title = (span.attributes.get("title") or "").strip()
    if title:
        try:
            return int(title)
        except ValueError:
            # Non-numeric title: fall through to parsing the visible text.
            pass
    return _parse_score(_node_text(span))


# Return the datetime attribute of the first <time> element under the node, or None.
def _created_utc(node: LexborNode | None) -> str | None:
    if node is None:
        return None
    time_node = node.css_first("time[datetime]")
    if time_node is None:
        return None
    return time_node.attributes.get("datetime")


# Body conversion

# Render a reddit `.md` div (server-rendered markdown→HTML) back to markdown.
def _extract_md(entry: LexborNode | None) -> str | None:
    if entry is None:
        return None
    body = entry.css_first(".usertext-body")
    if body is None:
        return None
    md = body.css_first(".md")
    if md is None:
        return None
    rendered = _md_to_markdown(md)
    return rendered or None


# Convert the children of an `.md` element to a markdown string.
def _md_to_markdown(el: LexborNode) -> str:
    out: list[str] = []
    _render_children(el, out)
    return "".join(out).strip()


# Append the markdown of every child of the element: text nodes verbatim, elements via _render_node.
def _render_children(el: LexborNode, out: list[str]) -> None:
    for child in el.iter(include_text=True):
        if child.tag == "-text":
            out.append(child.text(deep=False) or "")
        else:
            _render_node(child, out)


# Append the markdown of one HTML element chosen by its tag: paragraphs, inline emphasis and code,
# links, quotes, lists, headings and rules; unknown containers just render their children.
def _render_node(el: LexborNode, out: list[str]) -> None:
    tag = el.tag
    if tag in ("p", "div"):
        inner: list[str] = []
        _render_children(el, inner)
        text = "".join(inner).strip()
        if text:
            out.append(text)
            out.append("\n\n")
    elif tag == "br":
        out.append("\n")
    elif tag in ("strong", "b"):
        text = _node_text(el)
        if text:
            out.append(f"**{text}**")
    elif tag in ("em", "i"):
        text = _node_text(el)
        if text:
            out.append(f"*{text}*")
    elif tag in ("del", "s", "strike"):
        text = _node_text(el)
        if text:
            out.append(f"~~{text}~~")
    elif tag == "code":
        out.append("`")
        out.append(_node_text(el))
        out.append("`")
    elif tag == "pre":
        text = el.text(deep=True, separator="")
        out.append("```\n")
        out.append(text.rstrip("\n"))
        out.append("\n```\n\n")
    elif tag == "a":
        text = _node_text(el)
        if text:
            # Preserve the destination. Resolve root-relative reddit hrefs
            # (/r/, /user/, /wiki/, ...) and drop non-navigational ones
            # (javascript:, #fragment, mailto:).
            href = el.attributes.get("href") or ""
            if href.startswith("http://") or href.startswith("https://"):
                out.append(f"[{text}]({href})")
            elif href.startswith("/"):
                out.append(f"[{text}]({_OLD_REDDIT}{href})")
            else:
                out.append(text)
    elif tag == "blockquote":
        inner = []
        _render_children(el, inner)
        trimmed = "".join(inner).strip()
        for line in trimmed.split("\n"):
            if line:
                out.append("> ")
                out.append(line)
                out.append("\n")
            else:
                out.append(">\n")
        out.append("\n")
    elif tag == "ul":
        _render_list(el, False, 0, out)
    elif tag == "ol":
        _render_list(el, True, 0, out)
    elif tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
        level = int(tag[1])
        text = _node_text(el)
        if text:
            out.append("#" * level)
            out.append(" ")
            out.append(text)
            out.append("\n\n")
    elif tag == "hr":
        out.append("---\n\n")
    elif tag == "sup":
        out.append(_node_text(el))
    else:
        # Unknown / generic containers: recurse.
        _render_children(el, out)


# Render a <ul>/<ol>, indenting nested lists by two spaces per level so child
# items keep their own line instead of being glued to the parent.
def _render_list(list_el: LexborNode, ordered: bool, indent: int, out: list[str]) -> None:
    pad = "  " * indent
    n = 0
    for li in list_el.iter():
        if li.tag != "li":
            continue
        n += 1

        # Inline content of this <li>, excluding nested lists (rendered after).
        inline: list[str] = []
        for child in li.iter(include_text=True):
            if child.tag == "-text":
                inline.append(child.text(deep=False) or "")
            elif child.tag in ("ul", "ol"):
                continue
            else:
                _render_node(child, inline)
        marker = f"{n}. " if ordered else "- "
        out.append(f"{pad}{marker}{''.join(inline).strip()}\n")
        for child in li.iter():
            if child.tag == "ul":
                _render_list(child, False, indent + 1, out)
            elif child.tag == "ol":
                _render_list(child, True, indent + 1, out)
    if indent == 0:
        out.append("\n")


# DOM and value helpers

# Return the class names of the node.
def _classes(node: LexborNode) -> list[str]:
    return (node.attributes.get("class") or "").split()


# Return True when the node's class list contains the given class.
def _has_class(node: LexborNode, name: str) -> bool:
    return name in _classes(node)


# Return the first direct child element whose class list includes `cls`, or None.
def _direct_child(node: LexborNode, cls: str) -> LexborNode | None:
    for child in node.iter():
        if _has_class(child, cls):
            return child
    return None


# Return the stripped text of the node and its descendants, or an empty string for None.
def _node_text(node: LexborNode | None) -> str:
    if node is None:
        return ""
    return node.text(deep=True, separator="").strip()


# Parse an attribute value as an integer, falling back to the default when it is missing or malformed.
def _int_attr(value: str | None, default: int) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


# Parse the leading token of a score text such as "10 points" as an integer, or None if it is not a number.
def _parse_score(text: str) -> int | None:
    parts = text.split()
    if not parts:
        return None

    # Replace the unicode minus sign with an ASCII one so int() accepts negative scores.
    token = parts[0].replace("−", "-")
    try:
        return int(token)
    except ValueError:
        return None
