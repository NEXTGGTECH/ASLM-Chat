# Copyright NEXTGGTECH. Elastic License 2.0.


from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from core.search.nltk_languages import sentence_spans
from core.search.query_locales import analyze_query
from core.extract.content_processor import (
    _NOISE_MARKERS,
    _bm25_score_paragraphs,
    _bm25_tokenize,
    _clean_latex_for_index,
    _has_latex,
    _is_fenced_code_block,
    _repair_unclosed_fence,
    _split_blocks,
    _truncate_markdown_to_budget,
    derive_read_page_focus,
)

# Curve parameters are clamped to these bounds so a bad config cannot invert the curve.
_MIN_RATIO_AT_DOUBLE = 0.01
_MAX_RATIO_CEILING = 0.97
_MAX_EXPONENT = 0.95  # k < 1 keeps the kept-size curve monotonically increasing

# Blocks at least this long are shortened (by sentences / list items) rather than dropped.
_TRIM_MIN_CHARS = 700
# Absolute ceiling for "junk" (dropped outright before any trimming).
_JUNK_SCORE = 0.15

_HEADING_RE = re.compile(r"^#{1,6}\s+\S")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
_RULE_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_MATH_OPEN_RE = re.compile(r"^\s*(?:\$\$|\\\[|\\begin\{)")
_KEEP_OPEN_RE = re.compile(r"<!--\s*(?:aslm:)?keep\s*-->", re.IGNORECASE)
_KEEP_CLOSE_RE = re.compile(r"<!--\s*/\s*(?:aslm:)?keep\s*-->", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_MD_LINK_RE = re.compile(r"!?\[[^\]]*\]\([^)]*\)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_WS_RE = re.compile(r"\s+")

OMISSION_TEMPLATE = "[… omitted ~{size} …]"
# Gaps smaller than this are removed without a marker: a marker would cost about as much
# as the text it replaces, and such runs are short labels, captions and link lines.
_SILENT_GAP_CHARS = 200


# Shape of the size-adaptive compression: the soft limit, the share cut at twice the limit, the ceiling on
# that share and the hard cap on output size.
@dataclass(frozen=True, slots=True)
class CompressionCurve:
    soft_limit: int = 60_000
    ratio_at_double: float = 0.30
    max_ratio: float = 0.85
    hard_cap: int = 150_000

    # Exponent of the saturating power curve, solved from ratio(2) == ratio_at_double.
    @property
    def exponent(self) -> float:
        ceiling = min(_MAX_RATIO_CEILING, max(0.05, self.max_ratio))
        at_double = min(ceiling * 0.95, max(_MIN_RATIO_AT_DOUBLE, self.ratio_at_double))
        return min(_MAX_EXPONENT, -math.log2(1.0 - at_double / ceiling))

    # Fraction of the text to remove for a page of `length` characters (0 below the limit).
    def ratio(self, length: int) -> float:
        if self.soft_limit <= 0 or length <= self.soft_limit:
            return 0.0
        ceiling = min(_MAX_RATIO_CEILING, max(0.05, self.max_ratio))
        x = length / self.soft_limit
        return ceiling * (1.0 - x ** (-self.exponent))

    # Characters to keep for a page of `length` characters, honouring the hard cap.
    def target(self, length: int) -> int:
        kept = int(round(length * (1.0 - self.ratio(length))))
        if self.hard_cap > 0:
            kept = min(kept, self.hard_cap)
        return max(0, kept)


# What a compression pass did: original, target and output sizes, protected size, how many blocks were
# dropped or trimmed, whether the hard cap applied and the block kinds seen.
@dataclass(slots=True)
class CompressionReport:
    original_chars: int
    target_chars: int
    output_chars: int = 0
    protected_chars: int = 0
    dropped_blocks: int = 0
    trimmed_blocks: int = 0
    hard_capped: bool = False
    kinds: dict[str, int] = field(default_factory=dict)

    # Whether the output is shorter than the original.
    @property
    def compressed(self) -> bool:
        return self.output_chars < self.original_chars


# One markdown block with its kind (prose or a protected kind), importance score and whether it opens a
# section.
@dataclass(slots=True)
class _Block:
    text: str
    kind: str  # "prose" or a protected kind
    score: float = 0.0
    lead: bool = False

    # Whether the block is protected markup (headings, code, tables, math, keep markers) that is never
    # trimmed.
    @property
    def protected(self) -> bool:
        return self.kind != "prose"


# Merge split blocks back together while a multi-block region (keep sentinel, $$ math)
# is open, so the region is classified, and protected, as one atom.
def _merge_regions(blocks: list[str]) -> list[str]:
    merged: list[str] = []
    buffer: list[str] = []
    closer: re.Pattern[str] | None = None
    dollar_open = False

    for block in blocks:
        if buffer:
            buffer.append(block)
            if closer is not None and closer.search(block):
                closer = None
            if dollar_open and block.count("$$") % 2 == 1:
                dollar_open = False
            if closer is None and not dollar_open:
                merged.append("\n\n".join(buffer))
                buffer = []
            continue
        if _is_fenced_code_block(block):
            merged.append(block)
            continue
        opens_keep = bool(_KEEP_OPEN_RE.search(block))
        closes_keep = bool(_KEEP_CLOSE_RE.search(block))
        if opens_keep and not closes_keep:
            closer = _KEEP_CLOSE_RE
        if block.lstrip().startswith("$$") and block.count("$$") % 2 == 1:
            dollar_open = True
        if closer is not None or dollar_open:
            buffer.append(block)
        else:
            merged.append(block)
    if buffer:
        merged.append("\n\n".join(buffer))
    return merged


# Classify one block by the structural markers it carries.
def _classify(block: str) -> str:
    if _is_fenced_code_block(block):
        return "code"
    if _KEEP_OPEN_RE.search(block):
        return "keep"
    lines = [line for line in block.splitlines() if line.strip()]
    if not lines:
        return "prose"
    first = lines[0]
    if len(lines) == 1 and _RULE_RE.match(first):
        return "rule"
    if _HEADING_RE.match(first) and len(lines) <= 2 and len(block) <= 300:
        return "heading"
    if _MATH_OPEN_RE.match(first):
        return "math"
    table_rows = sum(1 for line in lines if _TABLE_ROW_RE.match(line))
    if table_rows >= 2 and (any(_TABLE_SEP_RE.match(line) for line in lines) or table_rows == len(lines)):
        return "table"
    return "prose"


# Number of leading blocks that form the normalizer's metadata header ("# Title" … "---").
def _meta_prefix_len(blocks: list[str]) -> int:
    for idx, block in enumerate(blocks[:8]):
        if _RULE_RE.match(block.strip()):
            return idx + 1
    return 1 if blocks and _HEADING_RE.match(blocks[0]) else 0


# Scale scores to 0..1 relative to the highest one; all zeros when nothing scores.
def _normalize_scores(values: list[float]) -> list[float]:
    peak = max(values, default=0.0)
    if peak <= 0:
        return [0.0] * len(values)
    return [min(1.0, max(0.0, v / peak)) for v in values]


# Cheap factual-density proxy (0..1): digits, unique-token ratio, sentence shape
# (sentences per NLTK Punkt for the page language).
def _density(text: str, language: str) -> float:
    tokens = _bm25_tokenize(text)
    if not tokens:
        return 0.0
    unique_ratio = len(set(tokens)) / len(tokens)
    digits = sum(1 for ch in text if ch.isdigit())
    sentences = len(sentence_spans(text, language))
    sentence_shape = min(1.0, sentences / max(1.0, len(text) / 200))
    score = min(0.45, unique_ratio * 0.6)
    score += 0.2 if digits >= 3 else 0.08 if digits else 0.0
    score += 0.35 * sentence_shape
    return min(1.0, score)


# Fraction of the block's characters spent on Markdown link syntax.
def _link_density(text: str) -> float:
    if not text:
        return 0.0
    linked = sum(len(match.group(0)) for match in _MD_LINK_RE.finditer(text))
    return linked / len(text)


# Score every prose block in place (protected blocks keep score 1.0).
def _score_blocks(blocks: list[_Block], focus: str, language: str) -> None:
    prose_idx = [i for i, block in enumerate(blocks) if not block.protected]
    if not prose_idx:
        return
    texts = [blocks[i].text for i in prose_idx]
    index_texts = [_clean_latex_for_index(t) for t in texts] if _has_latex("\n\n".join(texts)) else texts
    terms = _bm25_tokenize(focus)
    relevance = _normalize_scores(_bm25_score_paragraphs(index_texts, terms)) if terms else [0.0] * len(texts)

    seen: set[str] = set()
    total = len(blocks)
    for rel, idx in zip(relevance, prose_idx):
        block = blocks[idx]
        text = block.text
        lowered = text.lower()
        score = 0.35 * rel + 0.35 * _density(text, language)
        score += 0.10 * (1.0 - idx / max(1, total))  # mild preference for earlier content
        if block.lead:
            score += 0.15
        if _INLINE_CODE_RE.search(text):
            score += 0.10
        link_share = _link_density(text)
        if link_share > 0.5:
            score -= 0.35
        elif link_share > 0.3:
            score -= 0.15
        if len(text) < 300 and any(marker in lowered for marker in _NOISE_MARKERS):
            score -= 0.30
        if len(text) < 40 and not re.search(r"[.!?…:]$", text.strip()):
            score -= 0.10
        key = _WS_RE.sub(" ", lowered).strip()
        if len(key) > 20 and key in seen:
            score = 0.0  # verbatim repeat of an earlier block
        seen.add(key)
        block.score = max(0.0, min(1.0, score))


# Mark the first prose block of the document and of every heading's section.
def _mark_leads(blocks: list[_Block]) -> None:
    expect_lead = True
    for block in blocks:
        if block.kind == "heading":
            expect_lead = True
        elif not block.protected and expect_lead:
            block.lead = True
            expect_lead = False


# Shorten a block to roughly `keep_chars` by whole sentences (NLTK Punkt) or list items.
def _trim_block(text: str, keep_chars: int, language: str) -> str | None:
    lines = text.splitlines()
    if sum(1 for line in lines if _LIST_ITEM_RE.match(line)) >= 3:
        units, joiner = lines, "\n"
    else:
        units, joiner = [text[s:e].strip() for s, e in sentence_spans(text, language)], " "
    if len(units) < 2:
        return None
    kept: list[str] = []
    used = 0
    for unit in units:
        if kept and used + len(unit) > keep_chars:
            break
        kept.append(unit)
        used += len(unit) + len(joiner)
    if len(kept) == len(units):
        return None
    dropped = len(text) - used
    return joiner.join(kept).rstrip() + joiner + OMISSION_TEMPLATE.format(size=_human(dropped))


# Format a character count for the compression note ("12.3k chars").
def _human(chars: int) -> str:
    return f"{chars / 1000:.1f}k chars" if chars >= 1000 else f"{chars} chars"


# Compress markdown to the curve's target. Returns (text, report); text is unchanged
# when the page is under the soft limit.
def compress_adaptive(
    markdown: str,
    curve: CompressionCurve,
    *,
    url: str = "",
    focus: str = "",
) -> tuple[str, CompressionReport]:
    text = _repair_unclosed_fence(markdown or "")
    length = len(text)
    target = curve.target(length)
    report = CompressionReport(original_chars=length, target_chars=target, output_chars=length)
    if not text or target >= length:
        return text, report

    raw_blocks = _merge_regions(_split_blocks(text))
    meta_len = _meta_prefix_len(raw_blocks)
    blocks = [
        _Block(text=block, kind="meta" if idx < meta_len else _classify(block))
        for idx, block in enumerate(raw_blocks)
    ]
    for block in blocks:
        if block.protected:
            block.score = 1.0
            report.protected_chars += len(block.text) + 2
            report.kinds[block.kind] = report.kinds.get(block.kind, 0) + 1
    _mark_leads(blocks)
    try:
        topic = (focus or "").strip() or derive_read_page_focus(url, text)
    except Exception:  # noqa: BLE001 - scoring must degrade, never fail the read
        topic = ""
    # Page language picks the Punkt sentence model (sampled from the opening text).
    language = analyze_query(text[:6000]).language
    _score_blocks(blocks, topic, language)

    current = sum(len(block.text) + 2 for block in blocks)
    need = current - target
    dropped: set[int] = set()
    first_prose = next((i for i, block in enumerate(blocks) if not block.protected), -1)
    order = sorted((i for i, block in enumerate(blocks) if not block.protected), key=lambda i: blocks[i].score)
    # Scores are relative within a page (with no query few blocks clear an absolute bar),
    # so "junk" is the page's own bottom quartile, capped at an absolute ceiling.
    quartile = blocks[order[len(order) // 4]].score if order else 0.0
    # A zero quartile (a page full of repeats) still drops the zero-scored repeats.
    junk_bar = min(_JUNK_SCORE, quartile) or 1e-9
    rank = {idx: pos / max(1, len(order) - 1) for pos, idx in enumerate(order)}

    # Pass 1: drop junk only - duplicates, link farms, boilerplate (lowest score first).
    for idx in order:
        if need <= 0 or blocks[idx].score >= junk_bar:
            break
        if idx == first_prose:
            continue
        dropped.add(idx)
        need -= len(blocks[idx].text) + 2

    # Pass 2: shorten the remaining long prose proportionally - the weaker a block ranks,
    # the less of it stays (30 % … 85 %) - so the cut is spread over the page instead of
    # deleting whole sections.
    for idx in order:
        if need <= 0:
            break
        block = blocks[idx]
        if idx in dropped or len(block.text) < _TRIM_MIN_CHARS:
            continue
        keep_ratio = 0.30 + 0.55 * rank[idx]
        keep_chars = max(int(len(block.text) * keep_ratio), len(block.text) - need)
        trimmed = _trim_block(block.text, keep_chars, language)
        if trimmed is not None and len(trimmed) < len(block.text):
            need -= len(block.text) - len(trimmed)
            block.text = trimmed
            report.trimmed_blocks += 1

    # Pass 3: still over target - drop whole blocks, weakest first, always keeping the
    # document's opening paragraph. Protected markup is never touched; if it alone exceeds
    # the hard cap, the final cut below handles it.
    for idx in order:
        if need <= 0:
            break
        if idx in dropped or idx == first_prose:
            continue
        dropped.add(idx)
        need -= len(blocks[idx].text) + 2

    pieces: list[str] = []
    gap = 0
    for idx, block in enumerate(blocks):
        if idx in dropped:
            gap += len(block.text) + 2
            continue
        if gap >= _SILENT_GAP_CHARS:
            pieces.append(OMISSION_TEMPLATE.format(size=_human(gap)))
        gap = 0
        pieces.append(block.text)
    if gap >= _SILENT_GAP_CHARS:
        pieces.append(OMISSION_TEMPLATE.format(size=_human(gap)))
    output = "\n\n".join(pieces)

    if curve.hard_cap > 0 and len(output) > curve.hard_cap:
        # Only reachable when protected markup alone exceeds the hard cap.
        output = _truncate_markdown_to_budget(output, curve.hard_cap)
        report.hard_capped = True

    report.dropped_blocks = len(dropped)
    report.output_chars = len(output)
    return output, report


# One-line note placed after the metadata header so the model knows text was removed.
def compression_note(report: CompressionReport) -> str:
    if not report.compressed:
        return ""
    pct = round(100 * (1 - report.output_chars / max(1, report.original_chars)))
    return (
        f"> read_page: long page compressed {_human(report.original_chars)} → "
        f"{_human(report.output_chars)} (−{pct}%). Gaps are marked “[… omitted …]”; code, "
        "tables, formulas and headings are kept verbatim. To locate exact text anywhere in "
        "the full page, call read_page again with `find`."
    )


# Insert the note right after the metadata header ("---"), or at the top.
def insert_note(markdown: str, note: str) -> str:
    if not note:
        return markdown
    head = markdown[:4000]
    match = re.search(r"^\s*---\s*$", head, re.MULTILINE)
    if match:
        cut = match.end()
        return f"{markdown[:cut]}\n\n{note}\n{markdown[cut:]}"
    return f"{note}\n\n{markdown}"
