# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import html as html_lib
import logging
import math
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

logger = logging.getLogger("core.extract.content_processor")

# Fastest available BeautifulSoup backend, chosen once: lxml's C parser when the
# lxml package is present (it is — trafilatura depends on it), else the stdlib
# html.parser. ~1.6x faster parsing with the same bs4 API, no new dependency.
try:
    import lxml  # noqa: F401

    _BS_PARSER = "lxml"
except ImportError:  # pragma: no cover — lxml is normally installed
    _BS_PARSER = "html.parser"


# LaTeX processing for BM25 index text and LLM text

# Patterns for a LaTeX command name such as \alpha, and for a LaTeX span: $$...$$, \[...\],
# \(...\), inline $...$ without a newline, or \cmd{...}.
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+")
_LATEX_SPAN_RE = re.compile(
    r"\$\$.*?\$\$"
    r"|\\\[.*?\\\]"
    r"|\\\(.*?\\\)"
    r"|\$[^$\n]+?\$"
    r"|\\[a-zA-Z]+\{[^}]*\}",
    re.DOTALL,
)

# Greek letters → Unicode
_GREEK: dict[str, str] = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "zeta": "ζ", "eta": "η", "theta": "θ",
    "iota": "ι", "kappa": "κ", "lambda": "λ", "mu": "μ",
    "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "chi": "χ", "psi": "ψ", "omega": "ω",
    "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ", "Lambda": "Λ",
    "Pi": "Π", "Sigma": "Σ", "Phi": "Φ", "Psi": "Ψ", "Omega": "Ω",
}

# Common symbols → Unicode
_SYMBOLS: dict[str, str] = {
    "cdot": "·", "times": "×", "div": "÷",
    "leq": "≤", "geq": "≥", "neq": "≠", "approx": "≈",
    "infty": "∞", "in": "∈", "notin": "∉",
    "sum": "Σ", "prod": "Π", "int": "∫",
    "pm": "±", "mp": "∓",
    "rightarrow": "→", "leftarrow": "←",
    "Rightarrow": "⇒", "Leftarrow": "⇐",
    "to": "→", "mapsto": "↦",
    "ldots": "...", "cdots": "···", "vdots": "⋮",
    "sqrt": "√",
    # delimiters — stripped
    "left": "", "right": "",
    "big": "", "Big": "", "bigg": "", "Bigg": "",
    # spacing — stripped
    "quad": " ", "qquad": "  ", ",": " ", ";": " ", "!": "",
}


# Macros already handled by _node_to_text or pylatexenc's default context.
# Unknown macros not in this set will be registered as 1-arg pass-through so
# their content is preserved instead of silently dropped.
_KNOWN_MACROS: frozenset[str] = (
    frozenset(_GREEK)
    | frozenset(_SYMBOLS)
    | frozenset({
        "text", "textbf", "textit", "texttt", "textrm",
        "mathrm", "mathbf", "mathit", "mathtt", "mathcal",
        "mathbb", "emph", "operatorname",
        "frac", "sqrt", "sum", "prod", "int",
        "left", "right", "big", "Big", "bigg", "Bigg",
        "quad", "qquad",
        # common LaTeX2e structural macros (no content to preserve)
        "begin", "end", "label", "ref", "cite", "bibitem",
        "footnote", "item", "newline", "noindent", "centering",
        "hline", "cline", "multicolumn", "multirow",
        # math accents / decorators handled by walker pass-through already
        "hat", "bar", "tilde", "vec", "dot", "ddot", "overline",
        "underline", "widehat", "widetilde",
    })
)


# Return the sorted list of \cmd names in text that are not in _KNOWN_MACROS.
def _unknown_macros(text: str) -> list[str]:
    found = {m[1:] for m in _LATEX_CMD_RE.findall(text)}
    return sorted(found - _KNOWN_MACROS)


# Build a pylatexenc LatexWalker context with unknown macros as 1-arg pass-through. Return None
# when there are no unknown macros or pylatexenc cannot build the context.
def _make_walker_context(unknown: list[str]):
    if not unknown:
        return None
    try:
        from pylatexenc.latexwalker import get_default_latex_context_db
        from pylatexenc.macrospec import MacroSpec
        db = get_default_latex_context_db()
        db.add_context_category(
            "custom_macros",
            macros=[MacroSpec(name, args_parser="{") for name in unknown],
        )
        return db
    except Exception as exc:
        logger.debug("Failed to build LatexWalker context: %s", exc)
        return None


# Build a pylatexenc latex2text context with unknown macros as 1-arg pass-through. Return None
# when there are no unknown macros or pylatexenc cannot build the context.
def _make_l2t_context(unknown: list[str]):
    if not unknown:
        return None
    try:
        from pylatexenc.latex2text import get_default_latex_context_db, MacroTextSpec
        db = get_default_latex_context_db()
        db.add_context_category(
            "custom_macros",
            macros=[MacroTextSpec(name, simplify_repl="%s") for name in unknown],
        )
        return db
    except Exception as exc:
        logger.debug("Failed to build latex2text context: %s", exc)
        return None


# Return True when text likely contains LaTeX markup: display or paren math delimiters, inline
# math with a command, or at least three LaTeX commands.
def _has_latex(text: str) -> bool:
    if not text:
        return False
    if "$$" in text or "\\[" in text or "\\(" in text:
        return True

    # Inline math: $...\cmd...$ — must contain a backslash command inside the
    # dollar signs to avoid false positives on plain dollar amounts like "$5".
    if re.search(r'\$[^$\n]*\\[a-zA-Z]+[^$\n]*\$', text):
        return True
    return len(_LATEX_CMD_RE.findall(text)) >= 3


# Convert LaTeX markup to plain text suitable for BM25 tokenisation, using pylatexenc with a
# regex fallback when it is missing or fails.
def _clean_latex_for_index(text: str) -> str:
    if not _has_latex(text):
        return text
    try:
        from pylatexenc.latex2text import LatexNodes2Text
        ctx = _make_l2t_context(_unknown_macros(text))
        l2t = LatexNodes2Text(math_mode="text", latex_context=ctx) if ctx else LatexNodes2Text(math_mode="text")
        return l2t.latex_to_text(text)
    except Exception as exc:
        logger.debug("Latex-to-text conversion failed, using regex fallback: %s", exc)
        pass

    # Regex fallback: unwrap \cmd{content} → content, strip bare \cmd
    cleaned = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", text)
    cleaned = re.sub(r"\\[a-zA-Z]+", " ", cleaned)
    return re.sub(r"[\$\{\}]", " ", cleaned)


# Recursively transpile a pylatexenc node to readable text for the LLM: Greek letters and
# symbols become Unicode, text wrappers keep their content, \frac becomes (a)/(b), \sqrt becomes
# √, comments and unknown bare commands vanish.
def _node_to_text(node) -> str:  # noqa: ANN001
    from pylatexenc.latexwalker import (
        LatexCharsNode, LatexMacroNode, LatexGroupNode,
        LatexMathNode, LatexEnvironmentNode, LatexCommentNode,
    )

    if node is None:
        return ""
    if isinstance(node, LatexCharsNode):
        return node.chars

    if isinstance(node, LatexGroupNode):
        return "".join(_node_to_text(n) for n in (node.nodelist or []))

    if isinstance(node, LatexCommentNode):
        return ""

    if isinstance(node, LatexMathNode):
        # Recurse into the math content
        return "".join(_node_to_text(n) for n in (node.nodelist or []))

    if isinstance(node, LatexMacroNode):
        name = node.macroname
        args = node.nodeargd.argnlist if (node.nodeargd and node.nodeargd.argnlist) else []
        rendered_args = [_node_to_text(a) for a in args if a is not None]

        # Greek & symbols
        if name in _GREEK:
            return _GREEK[name]
        if name in _SYMBOLS:
            sym = _SYMBOLS[name]

            # Append rendered args if any (e.g. \sqrt{x} → √x)
            return sym + "".join(rendered_args)

        # Text/formatting wrappers — return inner content only
        if name in {"text", "textbf", "textit", "texttt", "textrm",
                    "mathrm", "mathbf", "mathit", "mathtt", "mathcal",
                    "mathbb", "emph", "operatorname"}:
            return "".join(rendered_args)

        # \frac{a}{b} → (a)/(b)
        if name == "frac" and len(rendered_args) >= 2:
            return f"({rendered_args[0]})/({rendered_args[1]})"

        # \sqrt[n]{x} or \sqrt{x}
        if name == "sqrt":
            # With two arguments the first (optional) one is the root index.
            if len(rendered_args) == 2:
                return f"{rendered_args[0]}√{rendered_args[1]}"
            if rendered_args:
                return f"√{rendered_args[0]}"
            return "√"

        # \sum / \prod / \int with optional limits
        if name in {"sum", "prod", "int"}:
            sym = {"sum": "Σ", "prod": "Π", "int": "∫"}[name]
            return sym + "".join(rendered_args)

        # Superscript/subscript are handled at LatexCharsNode level
        # Bare \cmd with args — render args
        if rendered_args:
            return "".join(rendered_args)

        # Bare \cmd no args
        return ""

    if isinstance(node, LatexEnvironmentNode):
        return "".join(_node_to_text(n) for n in (node.nodelist or []))

    # Fallback: stringify
    return str(getattr(node, "chars", "") or "")


# Transpile LaTeX markup to human-readable notation for the LLM. Falls back to the original text
# when the parse yields nothing, and to the index cleanup when pylatexenc fails.
def _render_latex_for_llm(text: str) -> str:
    if not _has_latex(text):
        return text

    try:
        from pylatexenc.latexwalker import LatexWalker
        ctx = _make_walker_context(_unknown_macros(text))
        walker = LatexWalker(text, latex_context=ctx) if ctx else LatexWalker(text)
        nodes, _, _ = walker.get_latex_nodes()
        result = "".join(_node_to_text(n) for n in (nodes or [])).strip()
        return result if result else text
    except Exception:
        # pylatexenc is missing or could not parse the markup; fall back to the regex cleanup.
        return _clean_latex_for_index(text)


# BM25 helpers (no external deps; used for paragraph compression)

# A word token; the BM25 term-frequency saturation (k1) and length normalization (b)
# parameters; and the token length that tokens must exceed to be indexed.
_TOKEN_RE = re.compile(r"\b\w+\b")
_BM25_K1 = 1.5
_BM25_B = 0.75
_BM25_MIN_TOKEN_LEN = 2


# Tokenize text for BM25 (words longer than _BM25_MIN_TOKEN_LEN).
def _bm25_tokenize(text: str) -> list[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text or "") if len(t) > _BM25_MIN_TOKEN_LEN]


# Return the BM25 score of each paragraph against query_terms, treating the paragraphs as the
# document collection (Okapi BM25 with the idf smoothed by +1, so it is never negative).
def _bm25_score_paragraphs(paragraphs: list[str], query_terms: list[str]) -> list[float]:
    if not paragraphs or not query_terms:
        return [0.0] * len(paragraphs)

    # Collection statistics: average paragraph length in tokens and the number of paragraphs
    # that contain each distinct query term.
    para_tokens = [_bm25_tokenize(p) for p in paragraphs]
    avg_dl = sum(len(t) for t in para_tokens) / max(1, len(para_tokens))
    N = len(paragraphs)

    df: dict[str, int] = {}
    for term in set(query_terms):
        df[term] = sum(1 for tokens in para_tokens if term in tokens)

    # Score each paragraph by summing idf * normalized term frequency over the query terms.
    scores: list[float] = []
    for tokens in para_tokens:
        dl = len(tokens)
        tf_map: dict[str, int] = {}
        for t in tokens:
            tf_map[t] = tf_map.get(t, 0) + 1
        score = 0.0
        for term in query_terms:
            tf = tf_map.get(term, 0)
            if tf == 0:
                continue
            idf = math.log((N - df.get(term, 0) + 0.5) / (df.get(term, 0) + 0.5) + 1)
            tf_norm = (tf * (_BM25_K1 + 1)) / (tf + _BM25_K1 * (1 - _BM25_B + _BM25_B * dl / max(1, avg_dl)))
            score += idf * tf_norm
        scores.append(score)
    return scores


# Pin fenced code verbatim, then spend the remaining character budget on BM25 prose.
# Protected code may make the result exceed max_chars. Text that already fits is returned as is.
def compress_to_budget(text: str, query: str, max_chars: int) -> str:
    if not text or len(text) <= max_chars:
        return text

    # Without paragraph structure there is nothing to rank, so fall back to plain truncation.
    paragraphs = _split_blocks(text)
    if not paragraphs:
        return _truncate_markdown_to_budget(text, max_chars, preserve_fenced_code=True)

    # Pin fenced code blocks and reserve their cost (plus the blank-line separator) up front.
    pinned = {idx for idx, paragraph in enumerate(paragraphs) if _is_fenced_code_block(paragraph)}
    pinned_cost = sum(len(paragraphs[idx]) + 2 for idx in pinned)
    prose_budget = max(0, max_chars - pinned_cost)

    query_terms = _bm25_tokenize(query)
    if not query_terms:
        # No query terms — greedily take paragraphs from the top
        selected = set(pinned)
        budget = prose_budget
        for idx, p in enumerate(paragraphs):
            if idx in pinned:
                continue
            cost = len(p) + 2
            if cost <= budget:
                selected.add(idx)
                budget -= cost
        return "\n\n".join(paragraphs[idx] for idx in sorted(selected))

    # Score against LaTeX-clean versions so math notation doesn't confuse
    # tokenisation, but keep original paragraph text for the output.
    if _has_latex(text):
        score_paras = [_clean_latex_for_index(p) for p in paragraphs]
    else:
        score_paras = paragraphs

    # Fill the remaining budget with the best-scoring paragraphs; one that does not fit is
    # skipped so smaller ones can still be taken, and filling stops once the budget is used up.
    scores = _bm25_score_paragraphs(score_paras, query_terms)
    ranked = sorted(range(len(paragraphs)), key=lambda i: -scores[i])

    selected: set[int] = set(pinned)
    budget = prose_budget
    for idx in ranked:
        if idx in pinned:
            continue
        cost = len(paragraphs[idx]) + 2
        if cost <= budget:
            selected.add(idx)
            budget -= cost
        if budget <= 0:
            break

    # Reconstruct in original order; plain truncation when nothing was selected.
    result = [paragraphs[i] for i in sorted(selected)]
    return "\n\n".join(result) if result else _truncate_markdown_to_budget(
        text, max_chars, preserve_fenced_code=True
    )


# Build a fallback BM25 query from the last five meaningful URL path segments (generic words
# and numbers dropped) and the page title taken from a "# " heading or a "**Title:**" line.
def derive_read_page_focus(url: str, markdown: str) -> str:
    from urllib.parse import unquote, urlparse

    # Path words: split the decoded path on separators and drop host/route filler.
    parts: list[str] = []
    if url:
        parsed = urlparse(url)
        path = unquote(parsed.path or "")
        stop = frozenset({
            "www", "blob", "tree", "raw", "main", "master", "head", "tag", "refs",
            "pull", "issues", "wiki", "html", "htm", "php", "asp", "aspx",
        })
        segments = [
            seg
            for seg in re.split(r"[/._\-]+", path)
            if seg and seg.lower() not in stop and not seg.isdigit()
        ]
        if segments:
            parts.append(" ".join(segments[-5:]))

    # Title: the first heading-style match wins.
    for pattern in (r"^#\s+(.+)$", r"^\*\*Title:\*\*\s*(.+)$"):
        match = re.search(pattern, markdown or "", re.MULTILINE | re.IGNORECASE)
        if match:
            title = match.group(1).strip()
            if title:
                parts.append(title)
                break

    return " ".join(parts).strip()


# Prefer explicit focus; else derive from URL/title; else empty.
def _resolve_read_page_compress_query(focus: str, url: str, markdown: str) -> str:
    explicit = (focus or "").strip()
    if explicit:
        return explicit
    try:
        return derive_read_page_focus(url, markdown)
    except Exception:
        # Deriving a focus is best-effort; an empty query just disables BM25 ranking.
        logger.debug("derive_read_page_focus failed for url=%r", url, exc_info=True)
        return ""


# Shrink long read_page output with BM25 relevance. Fenced code is outside the
# prose budget, so max_chars becomes a soft ceiling when protected code alone is larger.
# Compression applies only above compress_threshold and aims at compress_target (or max_chars
# when that is not positive), and a final hard truncation enforces max_chars.
def compress_read_page_markdown(
    markdown: str,
    *,
    url: str = "",
    focus: str = "",
    max_chars: int,
    compress_threshold: int,
    compress_target: int,
    enable_compress: bool = True,
) -> str:
    text = _repair_unclosed_fence(markdown or "")
    if not text:
        return text

    # Stage 1: reduce long text to the budget. With a focus (search preview) use the chunk
    # compaction engine; without one (standalone read_page) use the BM25 budget fill.
    if enable_compress and compress_threshold > 0 and len(text) > compress_threshold:
        budget = compress_target if compress_target > 0 else max_chars
        budget = min(budget, max_chars)
        query = _resolve_read_page_compress_query(focus, url, text)
        if focus.strip():
            # Search preview: a real query is present — select the most relevant chunks
            # and reject SEO-stuffed blocks (entity/BM25 + SEO penalty engine).
            from core.extract.chunk_compaction import compress_chunks

            compacted = compress_chunks(text, focus, char_budget=budget)
            text = compacted or text
        else:
            # Standalone read_page (no query): gentle BM25 budget fill over a derived focus.
            text = compress_to_budget(text, query, budget)

    # Stage 2: clause/sentence-level micro-prune: drops SEO-stuffed and off-query micro-chunks
    # within otherwise-kept blocks. Only runs with a real search query (focus); a no-op
    # for the standalone read_page tool, which has no query to prune against.
    if enable_compress and focus.strip() and text.strip():
        from core.extract.micro_chunk_worker import prune_micro_chunks

        # Prune block by block; fenced code is kept verbatim and fully pruned blocks vanish.
        protected: list[str] = []
        for block in _split_blocks(text):
            if _is_fenced_code_block(block):
                protected.append(block)
                continue
            pruned, _micro = prune_micro_chunks(block, focus)
            if pruned.strip():
                protected.append(pruned)
        if protected:
            text = "\n\n".join(protected)

    # Stage 3: hard limit, still keeping fenced code intact.
    if len(text) > max_chars:
        text = _truncate_markdown_to_budget(text, max_chars, preserve_fenced_code=True)
    return text


# Tag and noise constants

# Tags removed outright from the page because they never hold article text.
_NOISE_TAGS = {
    "script", "style", "nav", "header", "footer", "aside", "form",
    "iframe", "svg", "picture", "img", "video", "source", "canvas",
    "button", "noscript", "figure", "figcaption",
}

# Tags treated as text blocks, and the leaf subset that holds the text directly.
_BLOCK_TAGS = {
    "article", "section", "main", "div", "p", "li", "pre", "code",
    "blockquote", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6",
}
_LEAF_BLOCK_TAGS = {
    "p", "li", "pre", "code", "blockquote", "td", "th",
    "h1", "h2", "h3", "h4", "h5", "h6",
}

# Containers that are never removed as noise, even when their attributes or text look like it.
_PROTECTED_CONTAINER_TAGS = {"html", "body", "main", "article"}

# Lowercase phrases in attributes or short text that mark a node as cookie, newsletter, ad,
# social or legal boilerplate.
_NOISE_MARKERS = (
    "cookie", "consent", "gdpr", "newsletter", "subscribe", "sign up",
    "sign-up", "advert", "sponsored", "promo", "share", "follow us",
    "all rights reserved", "accept all", "privacy policy", "terms of service",
)

# Runs of horizontal whitespace (newlines excluded), and an HTML tag.
_WHITESPACE_RE = re.compile(r"[ \t\r\f\v]+")
_TAG_RE = re.compile(r"<[^>]+>")


# HTML cleaning helpers

# Collapse whitespace and unescape HTML entities in text; edge spaces, dashes and pipes are trimmed.
def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(
        " ", html_lib.unescape(text or "").replace("\u00a0", " ")
    ).strip(" -|\t\r\n")


# Join paragraph blocks into a single pipe-separated line.
def _single_line(text: str) -> str:
    blocks = [_normalize_text(piece) for piece in re.split(r"\n\s*\n", text or "")]
    return " | ".join(block for block in blocks if block)


# Whitespace that follows sentence-ending punctuation (. ! ?), i.e. a sentence boundary.
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


# Truncate text to max_chars, preferring a sentence boundary when it falls in the last 40% of
# the window; otherwise cut at max_chars and strip trailing punctuation.
def _truncate_at_sentence(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]

    # Find the last sentence-end position inside the window; it is the position right after
    # the closing punctuation.
    best = -1
    for m in _SENTENCE_END_RE.finditer(window):
        best = m.start()

    min_acceptable = int(max_chars * 0.60)
    if best >= min_acceptable:
        return window[:best].rstrip()

    return window.rstrip(" ,;:|-")


# Strip scripts, styles, inline event-handler attributes and tags with regexes; return normalized
# plain text. Last-resort conversion that needs no HTML parser.
def _regex_html_to_text(raw_html: str) -> str:
    no_js = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw_html or "",
                   flags=re.IGNORECASE | re.DOTALL)
    no_js = re.sub(r'\s+on\w+="[^"]*"', "", no_js)
    return _normalize_text(_TAG_RE.sub(" ", no_js))


# Remove noise tags and noise-marker nodes from HTML. Return the cleaned HTML, or the input
# unchanged when BeautifulSoup is unavailable.
def _preclean_html(raw_html: str) -> str:
    if not raw_html:
        return ""
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        logger.debug("BeautifulSoup unavailable during preclean, returning raw HTML: %s", exc)
        return raw_html

    soup = BeautifulSoup(raw_html, _BS_PARSER)

    # Pass 1: remove every tag that never holds article text.
    for tag_name in _NOISE_TAGS:
        for node in soup.find_all(tag_name):
            node.decompose()

    # Pass 2: remove nodes whose identifying attributes, or short text, contain a noise marker.
    # Protected containers are never removed, and article/main are removed only when their
    # text is short.
    for node in soup.find_all(True):
        attrs_map = node.attrs or {}
        attrs = " ".join(
            str(value)
            for key, value in attrs_map.items()
            if key in {"id", "class", "role", "aria-label", "data-testid"}
        ).lower()
        text_preview = _normalize_text(node.get_text(" ", strip=True))[:240].lower()
        attr_hit = any(marker in attrs for marker in _NOISE_MARKERS)
        text_hit = len(text_preview) <= 120 and any(marker in text_preview for marker in _NOISE_MARKERS)
        if attr_hit or text_hit:
            if node.name in _PROTECTED_CONTAINER_TAGS:
                continue
            if len(text_preview) < 300 or node.name not in {"article", "main"}:
                node.decompose()

    return str(soup)


# Re-split over-merged trafilatura output when possible: text without blank lines is split
# before each numbered item ("1. ", "2. ", ...) into paragraphs.
def _split_trafilatura_sections(text: str) -> str:
    if not text or "\n\n" in text:
        return text
    parts = re.split(r"\n(?=\d+\.\s)", text)
    if len(parts) > 1:
        return "\n\n".join(p.strip() for p in parts if p.strip())
    return text


# Return a structural flat-text nav score in [0, 1] (short lines, separators, low sentence
# density); higher means the text looks more like flattened navigation.
def _flat_nav_structural_ratio(text: str) -> float:
    if not text or len(text) < 40:
        return 0.0
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if not lines:
        return 0.0

    short_line_ratio = sum(1 for l in lines if len(l.split()) <= 4) / len(lines)
    seps = len(re.findall(r"[|›»·•]", text))
    sep_density = min(seps / max(len(text), 1) * 15, 1.0)

    parts = re.split(r"[.!?;:。！？]+", text)
    parts = [p.strip() for p in parts if p.strip()]
    sentence_ratio = (
        sum(1 for p in parts if 8 <= len(p.split()) <= 90) / len(parts)
        if parts else 0.0
    )

    score = (
        0.55 * short_line_ratio
        + 0.30 * sep_density
        - 0.35 * sentence_ratio
    )
    return max(0.0, min(score, 1.0))


# Pick trafilatura or DOM blocks, whichever yields cleaner structure. Return the chosen text, its
# source ("trafilatura" or "dom_blocks") and the DOM extraction stats.
def _choose_extraction(
    cleaned_html: str,
    *,
    url: str = "",
    min_clean_chars: int = 220,
) -> tuple[str, str, dict[str, int | str]]:
    # Run both extractors on the same cleaned HTML.
    dom_stats: dict[str, int | str] = {}
    traf = _extract_text_with_trafilatura(cleaned_html)
    traf = _split_trafilatura_sections(traf)
    dom_text, dom_stats = _extract_text_with_dom_blocks(cleaned_html, url=url)

    # Compare them: text length, paragraph count, table and navigation signals, and how many
    # DOM candidates were rejected as navigation.
    traf_n = len(_normalize_text(traf))
    dom_n = len(_normalize_text(dom_text))
    traf_paras = len(_split_blocks(traf))
    dom_paras = len(_split_blocks(dom_text)) if dom_text else 0

    traf_has_table = "|" in traf and traf.count("\n") >= 4
    traf_nav_heavy = _flat_nav_structural_ratio(traf) > 0.18
    total_dom_cand = int(dom_stats.get("total_candidates", 0))
    dom_nav_rej = int(dom_stats.get("nav_rejected", 0))
    dom_cleaned_heavily = (
        total_dom_cand > 0 and dom_nav_rej / total_dom_cand >= 0.35 and dom_paras >= 1
    )

    # Decide whether the DOM result is preferable: it wins when navigation filtering was heavy,
    # unless trafilatura already produced a clean table; otherwise it wins when trafilatura is
    # thin, unstructured or nav-heavy while the DOM output is substantial.
    prefer_dom = False
    if dom_stats.get("extraction_status") == "nav_heavy":
        prefer_dom = dom_n > traf_n
    elif dom_cleaned_heavily:
        prefer_dom = True
    elif traf_has_table and not traf_nav_heavy and traf_n >= min_clean_chars:
        prefer_dom = False
    elif dom_n >= min_clean_chars:
        if traf_n < min_clean_chars:
            prefer_dom = True
        elif traf_paras <= 1 and dom_paras >= 2:
            prefer_dom = True
        elif dom_paras >= traf_paras + 2:
            prefer_dom = True
        elif traf_nav_heavy:
            prefer_dom = True

    # Pick the result in order of confidence, ending with whichever text is longer.
    structural_min = min(200, min_clean_chars)

    if prefer_dom and dom_n >= structural_min:
        return dom_text, "dom_blocks", dom_stats
    if dom_cleaned_heavily and dom_n >= structural_min:
        return dom_text, "dom_blocks", dom_stats
    if traf_n >= min_clean_chars:
        return traf, "trafilatura", dom_stats
    if dom_n >= min_clean_chars:
        return dom_text, "dom_blocks", dom_stats
    if dom_cleaned_heavily and dom_n > 0:
        return dom_text, "dom_blocks", dom_stats
    if traf_n > dom_n:
        return traf, "trafilatura", dom_stats
    return dom_text or traf, "dom_blocks" if dom_text else "trafilatura", dom_stats


# DOM-aware block extraction with navigation rejection. Return the blocks joined by blank lines
# and the extraction stats; the text is empty when nothing was extracted or the page was
# dominated by navigation.
def _extract_text_with_dom_blocks(
    cleaned_html: str,
    *,
    url: str = "",
) -> tuple[str, dict[str, int | str]]:
    from core.extract.dom_block_extractor import extract_dom_blocks

    blocks, stats = extract_dom_blocks(cleaned_html, url=url)
    if stats.get("extraction_status") == "nav_heavy":
        return "", stats
    if not blocks:
        return "", stats
    return "\n\n".join(blocks), stats


# Extract leaf block tags via BeautifulSoup: block texts of at least 40 characters joined by blank
# lines, or the whole page text when no block qualifies.
def _extract_text_with_bs4(cleaned_html: str) -> str:
    if not cleaned_html:
        return ""
    try:
        from bs4 import BeautifulSoup
    except Exception as exc:
        logger.debug("trafilatura unavailable, skipping extraction: %s", exc)
        return ""

    soup = BeautifulSoup(cleaned_html, _BS_PARSER)
    blocks: list[str] = []
    for node in soup.find_all(_BLOCK_TAGS):
        # A generic container that holds leaf blocks is skipped; its children are collected.
        if node.name not in _LEAF_BLOCK_TAGS and node.find(_LEAF_BLOCK_TAGS):
            continue
        text = _normalize_text(node.get_text(separator=" ", strip=True))
        if len(text) >= 40:
            blocks.append(text)
    if not blocks:
        return soup.get_text(separator="\n", strip=True) or ""
    return "\n\n".join(blocks)


# Full-body strip set — ONLY non-content machinery. Deliberately narrower than
# _NOISE_TAGS: nav/header/footer/form/button are kept, because on landing pages, doc
# indexes and download pages that "chrome" IS the information.
_FULLBODY_STRIP_TAGS = ("script", "style", "noscript", "template", "svg", "iframe")


# Return the whole readable <body> as trimmed text lines, keeping nav/header/footer chrome.
# This is the thin-extraction rescue path, never the primary extractor. Tables are rendered as
# markdown and <pre> blocks as fenced code. Return "" when bs4 is unavailable.
def extract_full_body_text(raw_html: str) -> str:
    if not raw_html:
        return ""
    try:
        from bs4 import BeautifulSoup
    except Exception:  # noqa: BLE001 — rescue path must degrade to a no-op
        return ""

    # Drop the machinery tags and keep the rest of the body.
    soup = BeautifulSoup(raw_html, _BS_PARSER)
    for tag_name in _FULLBODY_STRIP_TAGS:
        for node in soup.find_all(tag_name):
            node.decompose()
    body = soup.body or soup

    # Preserve tables as markdown instead of flattening them to a text wall: swap each
    # <table> for a positional marker, take the body text, then restore the markdown.
    from core.extract.markdown_code import restore_pre_markers, wrap_pre_with_markers
    from core.extract.markdown_tables import swap_tables_for_markers

    code_md = wrap_pre_with_markers(soup)
    table_md = swap_tables_for_markers(soup)

    # Trim every body line (collapsing inner whitespace, except for table marker lines), drop
    # empty lines, then put the table markdown and the code blocks back.
    lines = (
        line.strip() if line.strip().startswith("\x00TBL") else _WHITESPACE_RE.sub(" ", line).strip()
        for line in body.get_text(separator="\n").splitlines()
    )
    text = "\n".join(line for line in lines if line)
    for marker, md in table_md.items():
        text = text.replace(marker, f"\n{md}\n")
    return restore_pre_markers(text, code_md)


# Return a callable that returns True for boilerplate text blocks.
def _get_boilerplate_filter():
    # Return True when a block is short (up to 120 characters) and contains a noise marker. A
    # noise word is meaningful only on a short UI-like block. Long prose can use the same word
    # legitimately, and short headings/version notes are content by default.
    def _fallback(text: str) -> bool:
        compact = _normalize_text(text).lower()
        return len(compact) <= 120 and any(marker in compact for marker in _NOISE_MARKERS)
    return _fallback


# Deduplicate blocks by exact normalized text while preserving their original layout.
def _dedupe_blocks(blocks: Iterable[str]) -> list[str]:
    deduped: list[str] = []
    seen_exact: set[str] = set()
    for block in blocks:
        normalized = _normalize_text(block)
        if not normalized:
            continue
        exact_key = normalized.lower()
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)
        deduped.append(block.strip())
    return deduped


# Opening Markdown fence: up to three leading spaces, then backticks or tildes.
_MD_FENCE_OPEN_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(?:[^`~].*)?$")


# Return the fence marker when a line opens a fenced block.
def _fence_marker(line: str) -> str:
    match = _MD_FENCE_OPEN_RE.match(line.rstrip())
    return match.group(1) if match else ""


# Return True when a line closes the given fence marker (same character, at least as long, and
# nothing after it).
def _closes_fence(line: str, marker: str) -> bool:
    stripped = line.strip()
    if not marker or not stripped.startswith(marker[0] * len(marker)):
        return False
    return not stripped[len(marker):].strip(marker[0]).strip()


# Remove a final unmatched opener while retaining all following text as ordinary prose.
# A malformed source fence must not turn the remainder of a document into one huge atom.
def _repair_unclosed_fence(text: str) -> str:
    lines = (text or "").splitlines()
    marker = ""
    opener_index = -1

    # Track fence state line by line; an opener stays pending until its closing fence.
    for index, line in enumerate(lines):
        if marker:
            if _closes_fence(line, marker):
                marker = ""
                opener_index = -1
            continue
        opened = _fence_marker(line)
        if opened:
            marker = opened
            opener_index = index

    # A fence is still open at the end: delete the opener line.
    if marker and opener_index >= 0:
        del lines[opener_index]
        return "\n".join(lines)
    return text


# Split text into paragraph blocks while keeping every fenced code block atomic.
def _split_blocks(text: str) -> list[str]:
    # Finished blocks, the pending prose lines, the lines of the fence being read and its marker.
    blocks: list[str] = []
    prose: list[str] = []
    fenced: list[str] = []
    marker = ""

    # Split the pending prose lines into paragraphs and append them to blocks. A paragraph ends
    # at a blank line or after a period followed by two or more whitespace characters.
    def flush_prose() -> None:
        if not prose:
            return
        segment = "\n".join(prose)
        prose.clear()
        for piece in re.split(r"\n\s*\n|(?<=\.)\s{2,}", segment):
            stripped = piece.strip()
            if stripped:
                blocks.append(stripped)

    # Collect fenced lines until the closing fence, and emit prose before each new fence.
    for line in (text or "").splitlines():
        if marker:
            fenced.append(line)
            if _closes_fence(line, marker):
                blocks.append("\n".join(fenced).strip("\n"))
                fenced.clear()
                marker = ""
            continue

        opened = _fence_marker(line)
        if opened:
            flush_prose()
            marker = opened
            fenced.append(line)
        else:
            prose.append(line)

    if fenced:
        # Degrade an unterminated source fence to prose instead of swallowing the whole
        # tail into an oversized atomic block. The unmatched opener itself is discarded.
        prose.extend(fenced[1:])
    flush_prose()
    return blocks


# Return True when the block starts with a Markdown fence.
def _is_fenced_code_block(block: str) -> bool:
    first_line = (block or "").lstrip().split("\n", 1)[0]
    return bool(_fence_marker(first_line))


# Truncate Markdown at block boundaries and end it with a "[...truncated]" marker. With
# preserve_fenced_code, every fenced block is pinned verbatim and the result may exceed max_chars.
# Text that already fits is returned as is, and a non-positive budget gives "".
def _truncate_markdown_to_budget(
    text: str,
    max_chars: int,
    *,
    preserve_fenced_code: bool = False,
) -> str:
    text = _repair_unclosed_fence(text)
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text

    suffix = "[...truncated]"
    blocks = _split_blocks(text)

    # Pinned-code mode: keep all fenced blocks, then fill the rest of the budget with prose
    # blocks in order, cutting the first block that does not fit at a sentence boundary.
    if preserve_fenced_code:
        pinned = {idx for idx, block in enumerate(blocks) if _is_fenced_code_block(block)}
        pinned_cost = sum(len(blocks[idx]) + 2 for idx in pinned)
        prose_budget = max(0, max_chars - pinned_cost - len(suffix) - 2)
        selected: dict[int, str] = {idx: blocks[idx] for idx in pinned}
        used = 0
        for idx, block in enumerate(blocks):
            if idx in pinned:
                continue
            separator = 2 if selected else 0
            cost = separator + len(block)
            if used + cost <= prose_budget:
                selected[idx] = block
                used += cost
                continue
            remaining = prose_budget - used - separator
            if remaining > 0:
                fragment = _truncate_at_sentence(block, remaining)
                if fragment:
                    selected[idx] = fragment
            break
        prefix = "\n\n".join(selected[idx] for idx in sorted(selected)).rstrip()
        return f"{prefix}\n\n{suffix}" if prefix else suffix

    # Plain mode: take blocks in order within the budget. An oversized code block is skipped
    # whole, while an oversized prose block is cut at a sentence boundary and ends the loop.
    content_budget = max(0, max_chars - len(suffix) - 2)
    selected: list[str] = []
    used = 0
    for block in blocks:
        separator = 2 if selected else 0
        cost = separator + len(block)
        if used + cost <= content_budget:
            selected.append(block)
            used += cost
            continue

        if _is_fenced_code_block(block):
            # Omit this block atomically, but keep scanning: later prose or smaller code
            # can still use the remaining budget instead of being discarded with it.
            continue

        remaining = content_budget - used - separator
        if remaining > 0:
            fragment = _truncate_at_sentence(block, remaining)
            if fragment:
                selected.append(fragment)
        break

    prefix = "\n\n".join(selected).rstrip()
    return f"{prefix}\n\n{suffix}" if prefix else suffix


# Quality scoring and trafilatura extraction

# Extract main text via trafilatura (tables on, comments off, precision favored). Return "" when
# trafilatura is missing, fails, or the input is XML.
def _extract_text_with_trafilatura(cleaned_html: str) -> str:
    if not cleaned_html:
        return ""
    try:
        import trafilatura
    except Exception as exc:
        logger.debug("trafilatura.extract failed: %s", exc)
        return ""
    html_value = cleaned_html.lstrip()
    if html_value[:6].lower().startswith("<?xml"):
        return ""
    try:
        text = trafilatura.extract(
            cleaned_html,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
            deduplicate=False,
            output_format="txt",
        )
    except Exception:
        # Extraction is best-effort; the caller falls back to other extractors on empty text.
        return ""
    return text or ""


# Filter boilerplate blocks and dedupe; return joined text and block count.
def _clean_extracted_text(text: str) -> tuple[str, int]:
    is_boilerplate = _get_boilerplate_filter()
    blocks = _split_blocks(text)
    filtered = [block for block in blocks if not is_boilerplate(block)]
    deduped = _dedupe_blocks(filtered)
    return "\n\n".join(deduped), len(deduped)


# Return a heuristic quality score in [0, 1]: rewards length, block count and alphabetic text,
# and penalizes digit-heavy text and noise markers.
def _estimate_quality(text: str, block_count: int) -> float:
    compact = _normalize_text(text)
    if not compact:
        return 0.0
    chars = len(compact)
    alpha_ratio = sum(1 for ch in compact if ch.isalpha() or ch.isspace()) / max(1, chars)
    digit_ratio = sum(1 for ch in compact if ch.isdigit()) / max(1, chars)
    marker_penalty = sum(1 for marker in _NOISE_MARKERS if marker in compact.lower())
    score = 0.0
    score += min(chars / 1_200.0, 1.0) * 0.50
    score += min(block_count / 6.0, 1.0) * 0.20
    score += max(0.0, min(alpha_ratio, 1.0)) * 0.25
    score -= min(digit_ratio, 0.20) * 0.20
    score -= min(marker_penalty * 0.08, 0.25)
    return max(0.0, min(round(score, 4), 1.0))
