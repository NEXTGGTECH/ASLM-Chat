# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from .fact_signals import fact_signal_score, is_fact_like_text

# Import Tag for type hints only; bs4 itself is imported lazily where it is used.
if TYPE_CHECKING:
    from bs4 import Tag

# Fastest available BeautifulSoup backend (lxml C parser when installed, else stdlib).
try:
    import lxml  # noqa: F401

    _BS_PARSER = "lxml"
except ImportError:  # pragma: no cover — lxml is normally installed
    _BS_PARSER = "html.parser"

# Blocks with ui_score >= this are hard-rejected (unless protected).
NAV_HARD_REJECT: float = 0.52

# Shortest text (in characters) that counts as a content block.
_MIN_BLOCK_CHARS: int = 40

# Tags a user can click, and the subset that are form controls.
_CLICKABLE_TAGS = frozenset({"a", "button", "input", "select", "textarea"})
_CONTROL_TAGS = frozenset({"button", "input", "select", "textarea"})

# CSS class/id fragments (language-neutral DOM hints)
_NAV_ATTR_FRAGMENTS: frozenset[str] = frozenset({
    "menu", "nav", "navigation", "breadcrumb", "breadcrumbs",
    "sidebar", "widget", "footer", "header", "topbar", "toolbar",
    "cookie", "banner", "popup", "overlay", "modal",
    "social", "share", "pagination", "pager", "tabs-nav", "tab-nav",
    "mega", "dropdown", "submenu", "subnav", "related", "recommend",
})

# ARIA roles that mark navigation or other page chrome.
_NAV_ROLES: frozenset[str] = frozenset({
    "navigation", "menubar", "menu", "toolbar", "banner",
    "complementary", "contentinfo", "search",
})

# A menu-style separator character, and a run of non-whitespace characters (one token).
_SEPARATOR_RE = re.compile(r"[|›»·•/\\]")
_TOKEN_RE = re.compile(r"\S+")

# Optional debug-only lexicon (weight ≤ 0.02 in scoring; never sole reject reason).
_NAV_WORDS_RE = re.compile(
    r"\b(back|login|sign in|sign up|cart|checkout|home|next|previous|"
    r"назад|войти|корзина|каталог|главная|zurück|warenkorb|retour|panier|"
    r"atrás|carrito|voltar|carrinho|geri|sepet|wstecz|koszyk)\b",
    re.IGNORECASE | re.UNICODE,
)

# Per-domain DOM-path template stats: domain -> {path -> {seen, total_len, total_link}}
_domain_stats: dict[str, dict[str, Any]] = {}


# Record one more sampled page for per-domain template frequency.
def observe_domain_page(domain: str) -> None:
    if not domain:
        return
    bucket = _domain_stats.setdefault(domain, {})
    bucket["__pages__"] = int(bucket.get("__pages__", 0)) + 1


# Add one observation (text length and link density) of a DOM path to the stats of a domain.
def _record_path_observation(
    domain: str,
    dom_path: str,
    text_len: int,
    link_density: float,
) -> None:
    if not domain or not dom_path:
        return
    bucket = _domain_stats.setdefault(domain, {})
    entry = bucket.setdefault(dom_path, {"seen": 0, "total_len": 0, "total_link": 0.0})
    entry["seen"] = int(entry.get("seen", 0)) + 1
    entry["total_len"] = int(entry.get("total_len", 0)) + text_len
    entry["total_link"] = float(entry.get("total_link", 0.0)) + link_density


# Return a score in [0, 1]: how often this DOM path appeared on sampled pages for the domain.
def template_frequency_score(domain: str, dom_path: str) -> float:
    bucket = _domain_stats.get(domain or "", {})
    entry = bucket.get(dom_path)
    if not entry or not isinstance(entry, dict):
        return 0.0
    sampled = max(1, int(bucket.get("__pages__", 1)))
    return min(int(entry.get("seen", 0)) / sampled, 1.0)


# Return the lowercase hostname of the URL with a leading "www." stripped, or "" when unavailable.
def _domain_from_url(url: str | None) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url).netloc.lower()
        return host[4:] if host.startswith("www.") else host
    except Exception:
        # An unparsable URL simply has no domain.
        return ""


# Return the slash-separated tag path from the html root down to the element.
def _dom_path(tag: "Tag") -> str:
    parts: list[str] = []
    node = tag
    while node is not None and getattr(node, "name", None):
        name = node.name
        if name in {"html", "[document]"}:
            break
        parts.append(name)
        node = node.parent
    return "/" + "/".join(reversed(parts))


# Return a nav/UI score in [0, 1] from the id, class, role and aria attributes: 0.20 per nav
# fragment found plus 0.25 when a nav role is present.
def _attr_nav_score(tag: "Tag") -> float:
    attrs = tag.attrs or {}
    combined = " ".join(
        str(v) if not isinstance(v, list) else " ".join(str(x) for x in v)
        for k, v in attrs.items()
        if k in {"id", "class", "role", "aria-label", "data-testid", "data-component"}
    ).lower()
    if not combined:
        return 0.0
    hits = sum(1 for f in _NAV_ATTR_FRAGMENTS if f in combined)
    role_hit = any(r in combined for r in _NAV_ROLES)
    return min(hits * 0.20 + (0.25 if role_hit else 0.0), 1.0)


# Return the fraction of block text inside anchor tags.
def _anchor_density(tag: "Tag", text: str) -> float:
    if not text:
        return 0.0
    anchor_chars = sum(len(a.get_text()) for a in tag.find_all("a", recursive=True))
    return min(anchor_chars / max(len(text), 1), 1.0)


# Return the fraction of descendants that are clickable elements.
def _clickable_density(tag: "Tag") -> float:
    descendants = tag.find_all(True, recursive=True)
    if not descendants:
        return 0.0
    clickable = tag.find_all(list(_CLICKABLE_TAGS), recursive=True)
    return min(len(clickable) / max(len(descendants), 1), 1.0)


# Return the fraction of descendants that are form controls.
def _control_density(tag: "Tag") -> float:
    descendants = tag.find_all(True, recursive=True)
    if not descendants:
        return 0.0
    controls = tag.find_all(list(_CONTROL_TAGS), recursive=True)
    return min(len(controls) / max(len(descendants), 1), 1.0)


# Return the density of menu-style separator characters in text, scaled so that one separator per
# twelve characters reaches 1.
def _separator_ratio(text: str) -> float:
    if not text:
        return 0.0
    seps = len(_SEPARATOR_RE.findall(text))
    return min(seps / max(len(text), 1) * 12, 1.0)


# Return the fraction of text nodes shorter than 30 characters (menu-label signal).
def _short_text_node_ratio(tag: "Tag") -> float:
    nodes = [s.strip() for s in tag.find_all(string=True) if s and str(s).strip()]
    if not nodes:
        return 0.0
    short = sum(1 for s in nodes if len(s) < 30)
    return short / len(nodes)


# Return the text length per descendant element, normalized so 120 characters per element reach 1
# (paragraph mass proxy).
def _text_density(tag: "Tag", text: str) -> float:
    descendants = tag.find_all(True, recursive=True)
    return min(len(text) / max(len(descendants), 1) / 120.0, 1.0)


# Return the fraction of clause splits that look like real sentences (8 to 90 words, not menu labels).
def _sentence_like_ratio(text: str) -> float:
    parts = re.split(r"[.!?;:。！？]+", text or "")
    parts = [p.strip() for p in parts if p.strip()]
    if not parts:
        return 0.0
    good = sum(1 for p in parts if 8 <= len(p.split()) <= 90)
    return good / len(parts)


# Return a score for long unstructured word lists without facts (mega-menus): texts of at least
# 200 characters and 28 words with few sentences and few digits, growing with the word count.
def _monolithic_list_score(text: str) -> float:
    if len(text) < 200:
        return 0.0
    words = text.split()
    if len(words) < 28:
        return 0.0
    sent = _sentence_like_ratio(text)
    digit_ratio = sum(ch.isdigit() for ch in text) / max(len(text), 1)
    if sent < 0.14 and digit_ratio < 0.04:
        return min(1.0, len(words) / 65.0)
    return 0.0


# Return the share of same-tag siblings whose text length is close to the average (menu cluster).
# Fewer than three siblings, or an average length under 20 characters, score 0.
def _menu_cluster_score(tag: "Tag") -> float:
    parent = tag.parent
    if parent is None:
        return 0.0
    siblings = [
        s for s in parent.find_all(recursive=False)
        if getattr(s, "name", None) == tag.name
    ]
    if len(siblings) < 3:
        return 0.0
    lengths = [len(s.get_text(strip=True)) for s in siblings]
    if not lengths:
        return 0.0
    avg = sum(lengths) / len(lengths)
    if avg < 20:
        return 0.0
    similar = sum(1 for ln in lengths if abs(ln - avg) <= max(25, avg * 0.45))
    return min(1.0, similar / len(siblings))


# Return 0.7 when the element or one of its three nearest ancestors has a menu/nav-related id,
# class or role, otherwise 0.
def _ancestor_menu_hint(tag: "Tag") -> float:
    node = tag
    for _ in range(4):
        if node is None:
            break
        attrs = " ".join(
            str(v) if not isinstance(v, list) else " ".join(str(x) for x in v)
            for k, v in (node.attrs or {}).items()
            if k in {"id", "class", "role"}
        ).lower()
        if any(f in attrs for f in ("menu", "nav", "breadcrumb", "mega")):
            return 0.7
        node = node.parent
    return 0.0


# Return the similarity of tag, text length and link density among the siblings of an element
# (uniform siblings suggest a menu or list). Fewer than two siblings score 0.
def _sibling_uniformity(tag: "Tag") -> float:
    parent = tag.parent
    if parent is None:
        return 0.0
    siblings = [
        s for s in parent.find_all(recursive=False)
        if getattr(s, "name", None) and s is not tag
    ]
    if len(siblings) < 2:
        return 0.0

    # Share of siblings with the same tag, and share with a similar text length.
    same_tag = sum(1 for s in siblings if s.name == tag.name) / len(siblings)
    my_len = len(tag.get_text(strip=True))
    tol = max(20, my_len * 0.35)
    similar_len = sum(
        1 for s in siblings
        if abs(len(s.get_text(strip=True)) - my_len) <= tol
    ) / len(siblings)

    # Link density of the siblings against the element's own; closer means more uniform.
    sib_link = []
    for s in siblings:
        st = s.get_text(separator=" ", strip=True)
        sib_link.append(_anchor_density(s, st) if st else 0.0)
    my_link = _anchor_density(tag, tag.get_text(separator=" ", strip=True))
    if sib_link:
        avg_link = sum(sib_link) / len(sib_link)
        link_sim = 1.0 - min(abs(my_link - avg_link), 1.0)
    else:
        link_sim = 0.0
    return min(1.0, 0.40 * same_tag + 0.35 * similar_len + 0.25 * link_sim)


# Return the debug-only density of nav words (login, cart, home, ...) among the text tokens.
def _nav_word_density_debug(text: str) -> float:
    if not text:
        return 0.0
    matches = _NAV_WORDS_RE.findall(text)
    tokens = _TOKEN_RE.findall(text)
    return min(len(matches) / max(len(tokens), 1), 1.0)


# Return a language-agnostic UI/nav score in [0, 1]; higher = more likely boilerplate. It is a
# weighted sum of structural signals (link, click and control density, short text nodes, sibling
# uniformity, template frequency, separators, nav attributes, menu shapes), reduced by
# prose-like signals (sentence shape, text density).
def structure_ui_score(
    tag: "Tag",
    *,
    domain: str = "",
    debug_lexicon: bool = False,
) -> float:
    # Compute each signal once; the template score uses the per-domain path statistics.
    text = tag.get_text(separator=" ", strip=True)
    link_s = _anchor_density(tag, text)
    click_s = _clickable_density(tag)
    control_s = _control_density(tag)
    short_s = _short_text_node_ratio(tag)
    sep_s = _separator_ratio(text)
    sib_s = _sibling_uniformity(tag)
    template_s = template_frequency_score(domain, _dom_path(tag))
    sentence_s = _sentence_like_ratio(text)
    text_s = _text_density(tag, text)
    attr_s = _attr_nav_score(tag)
    mono_s = _monolithic_list_score(text)
    cluster_s = _menu_cluster_score(tag)
    ancestor_s = _ancestor_menu_hint(tag)

    # Combine the signals: boilerplate signals add, prose signals subtract.
    ui = (
        0.20 * link_s
        + 0.16 * click_s
        + 0.14 * control_s
        + 0.14 * short_s
        + 0.10 * sib_s
        + 0.08 * template_s
        + 0.04 * sep_s
        + 0.06 * attr_s
        + 0.14 * mono_s
        + 0.10 * cluster_s
        + 0.08 * ancestor_s
        - 0.10 * sentence_s
        - 0.08 * text_s
    )
    structural_hint = max(attr_s, ancestor_s, cluster_s, mono_s)

    # A structural menu-shape hint (uniform siblings, list shape) must not override clear
    # prose: a real nav/menu cluster carries links or controls. Sentence-like text with no
    # link/click/control density is content, however uniform its siblings look — otherwise
    # article paragraph runs get mis-scored as menus and dropped.
    is_clear_prose = sentence_s >= 0.60 and link_s < 0.20 and click_s < 0.20 and control_s < 0.20
    if structural_hint >= 0.45 and not is_clear_prose:
        ui = max(ui, structural_hint * 0.82)
    if debug_lexicon:
        ui += 0.02 * _nav_word_density_debug(text)
    return max(0.0, min(round(ui, 4), 1.0))


# Backward-compatible wrapper around structure_ui_score without the debug lexicon.
def nav_score(tag: "Tag", *, domain: str = "") -> float:
    return structure_ui_score(tag, domain=domain)


# Return a content signal in [0, 1] from sentence shape, text mass, low link density and a low
# structural UI score.
def _content_score(tag: "Tag", text: str) -> float:
    sentence_s = _sentence_like_ratio(text)
    text_s = _text_density(tag, text)
    link_s = _anchor_density(tag, text)
    low_link = max(0.0, 1.0 - link_s * 1.4)
    paragraph_mass = min(len(text) / 600.0, 1.0)
    return min(
        1.0,
        0.26 * text_s
        + 0.22 * sentence_s
        + 0.18 * low_link
        + 0.14 * paragraph_mass
        + 0.20 * max(0.0, 1.0 - structure_ui_score(tag)),
    )


# Return the final keep score of a block: content minus UI, plus optional query relevance and a
# bonus for factual text.
def block_keep_score(
    tag: "Tag",
    text: str,
    *,
    domain: str = "",
    query_relevance: float = 0.0,
) -> float:
    ui = structure_ui_score(tag, domain=domain)
    content = _content_score(tag, text)
    fact_s = fact_signal_score(text)
    return (
        0.60 * content
        - 0.45 * ui
        + 0.20 * query_relevance
        + 0.10 * fact_s
    )


# Return True for non-empty table cells/rows and fact-like text blocks, which are kept even when
# their UI score says navigation.
def _is_protected(tag: "Tag", text: str) -> bool:
    if tag.name in {"td", "th", "tr", "table"}:
        return bool(text.strip())
    return is_fact_like_text(text)


# Tags that hold a content block, and the generic containers that count as a block only when
# they contain none of those tags.
_CANDIDATE_TAGS = frozenset({
    "p", "li", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre", "code",
})
_CONTAINER_TAGS = frozenset({"article", "section", "main", "div"})


# Collect leaf-level block tags with their whitespace-normalized text. Blocks shorter than
# _MIN_BLOCK_CHARS are skipped, and a block whose first 60 characters repeat an earlier block
# is treated as a duplicate.
def _collect_leaf_blocks(soup) -> list[tuple["Tag", str]]:
    results: list[tuple["Tag", str]] = []
    seen_texts: set[str] = set()

    for tag in soup.find_all(_CANDIDATE_TAGS | _CONTAINER_TAGS):
        # A container with candidate descendants is not a leaf; its children are collected instead.
        if tag.name in _CONTAINER_TAGS and tag.find(list(_CANDIDATE_TAGS)):
            continue
        text = re.sub(r"[ \t\r\f\v]+", " ", tag.get_text(separator=" ", strip=True)).strip()
        if len(text) < _MIN_BLOCK_CHARS:
            continue
        sig = text[:60].lower()
        if sig in seen_texts:
            continue
        seen_texts.add(sig)
        results.append((tag, text))
    return results


# Extract content blocks from cleaned HTML, filtering structural UI/nav noise. A block whose UI
# score reaches nav_reject_threshold is dropped unless it is protected (table cell or factual
# text). Return the kept block texts and a stats dict with the candidate, nav-rejected and
# protected-kept counts and an extraction_status of "ok", "nav_heavy", "no_blocks" or "empty".
def extract_dom_blocks(
    cleaned_html: str,
    *,
    domain: str | None = None,
    url: str | None = None,
    min_block_chars: int = _MIN_BLOCK_CHARS,
    nav_reject_threshold: float = NAV_HARD_REJECT,
    debug_lexicon: bool = False,
) -> tuple[list[str], dict[str, int | str]]:
    empty_stats: dict[str, int | str] = {
        "total_candidates": 0,
        "nav_rejected": 0,
        "protected_kept": 0,
        "extraction_status": "empty",
    }
    if not cleaned_html:
        return [], empty_stats

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return [], empty_stats

    # Parse the page and collect the leaf block candidates.
    resolved_domain = (domain or _domain_from_url(url)).lower()

    soup = BeautifulSoup(cleaned_html, _BS_PARSER)
    candidates = _collect_leaf_blocks(soup)
    total = len(candidates)
    rejected = 0
    protected_kept = 0
    blocks: list[str] = []

    # Pass 1: score each block using the template statistics gathered from earlier pages.
    path_observations: list[tuple[str, int, float]] = []
    for tag, text in candidates:
        if len(text) < min_block_chars:
            continue
        dom_path = _dom_path(tag)
        link_d = _anchor_density(tag, text)
        if resolved_domain:
            path_observations.append((dom_path, len(text), link_d))

        ui = structure_ui_score(tag, domain=resolved_domain, debug_lexicon=debug_lexicon)
        if ui >= nav_reject_threshold:
            if _is_protected(tag, text):
                protected_kept += 1
                blocks.append(text)
            else:
                rejected += 1
        else:
            blocks.append(text)

    # Pass 2: update domain stats after scoring, so this page does not influence its own scores.
    # Each DOM path is recorded once per page.
    if resolved_domain:
        observe_domain_page(resolved_domain)
        seen_paths: set[str] = set()
        for dom_path, text_len, link_d in path_observations:
            if dom_path not in seen_paths:
                seen_paths.add(dom_path)
                _record_path_observation(resolved_domain, dom_path, text_len, link_d)

    # Flag pages where most candidates were rejected as navigation and almost nothing is left.
    status = "ok"
    if total > 0 and rejected / total > 0.5 and len(blocks) < 2:
        status = "nav_heavy"
    elif not blocks:
        status = "no_blocks"

    stats: dict[str, int | str] = {
        "total_candidates": total,
        "nav_rejected": rejected,
        "protected_kept": protected_kept,
        "extraction_status": status,
    }
    return blocks, stats
