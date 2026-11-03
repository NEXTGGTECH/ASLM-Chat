# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from core.extract.adaptive_compression import _classify
from core.extract.content_processor import (
    _fence_marker,
    _is_fenced_code_block,
    _repair_unclosed_fence,
    _split_blocks,
)
from core.extract.stemming import TOKEN_RE, stem_tokens, stems
from core.search.nltk_languages import DEFAULT_LANGUAGE, script_of, sentence_spans, stopwords_for

# read_page `find`: locate a word or phrase in the FULL page text and return whole,
# self-contained passages around each hit.
# A passage is built from the page's own blocks, never from character windows: the
# matched paragraph / list / table / code block, plus the lead-in that introduces it
# ("…as follows:") or the block it introduces, plus neighbours while the passage is
# too short to stand alone - always inside one section. Long blocks are reduced by whole
# units: sentences for prose, items for lists, rows (with the header) for tables, blank-
# line segments for code. Passages are ranked by query coverage; the best ones that fit
# the budget are returned in page order with their section path.

DEFAULT_MAX_HITS = 12
DEFAULT_MAX_CHARS = 20_000

_PASSAGE_MIN_CHARS = 450      # extend with neighbours while shorter than this
_PASSAGE_MAX_CHARS = 3_500    # stop extending past this
_LONG_PROSE_CHARS = 1_400     # longer paragraphs are cut to the hit sentences
_LONG_LIST_CHARS = 1_800
_LONG_TABLE_CHARS = 3_000
_LONG_CODE_CHARS = 4_000
_CONTEXT_CHARS = 700          # context (non-hit) blocks are cut to about this
_MAX_TOKEN_GAP = 4            # max non-word chars between words of a phrase

_REGEX_FORM_RE = re.compile(r"^/(.+)/([a-z]*)$", re.DOTALL)
_ALT_SPLIT_RE = re.compile(r"\s+\|\s+")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+•]|\d+[.)])\s+")
_RULE_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_GROUP_ROWS = 8               # rows a matched group-label row brings along
_HEADER_HIT_ROWS = 5          # rows shown when only the column header matched
_MIN_RELATIVE_SCORE = 0.5     # passages scoring under half of the best one are not shown
_MIN_RELATIVE_SCORE_WORDS = 0.7  # stricter without an exact phrase: common words match everywhere
_BACK_MATTER_WEIGHT = 0.2     # notes / references / bibliography rank after the body
_BACKLINKS_PER_LIST = 3
_READABLE_SHARE = 0.6         # word characters in plain prose's visible text, roughly
_LINK_LIST_SHARE = 0.6        # visible text above this share of link labels is a link list
_BACKLINK_RE = re.compile(r"#cite[_-]ref")
_SUBROW_RE = re.compile(r"^\s*\|\s*[•·◦▪\-–—]")

# Output tidying, applied to passages only (the page itself is untouched):
#   * MediaWiki-style footnote markers "<sup>[\[70\]](…#cite_note-…)</sup>" - anchors into
#     the reference list that only cost the model tokens;
#   * percent-encoded UTF-8 in link targets, decoded so a Cyrillic / CJK URL is readable
#     and 3-9x shorter (ASCII escapes such as %20 or %29 stay encoded).
_FOOTNOTE_RE = re.compile(
    r"(?:<sup>)?\[\\?\[[^\]\n]{1,40}?\\?\]\]\(<?[^)\s>]*#cite[_-]note[^)\s>]*>?\)(?:</sup>)?"
)
_PERCENT_RUN_RE = re.compile(r"(?:%[0-9A-Fa-f]{2})+")


# One page block with its kind, section path, section id and stemmed tokens.
@dataclass(slots=True)
class _Unit:
    index: int
    text: str
    kind: str          # heading / code / table / list / math / keep / rule / meta / prose
    section: str       # heading path, e.g. "FTS5 › Ranking"
    section_id: int    # index of the governing heading (-1 before the first one)
    tokens: list[tuple[frozenset[str], int, int]] = field(default_factory=list)
    language: str = "en"  # picks the Punkt sentence model


# A run of neighbouring blocks around one or more hits, with its match count and score.
@dataclass(slots=True)
class _Passage:
    lo: int
    hi: int
    hit_units: dict[int, list[tuple[int, int]]]
    score: float = 0.0
    matches: int = 0


# A parsed `find` query: alternatives, the stemmed terms of each, regex and literal patterns.
@dataclass(slots=True)
class _Query:
    described: str
    regex: re.Pattern[str] | None = None
    phrases: list[list[frozenset[str]]] = field(default_factory=list)  # per alternative
    literals: list[re.Pattern[str]] = field(default_factory=list)
    terms: list[frozenset[str]] = field(default_factory=list)          # distinct words


# Literal pattern for an alternative that carries punctuation; word edges are bounded so
# "io" cannot match inside "function" (not for CJK, which has no word edges).
def _literal_pattern(alt: str) -> re.Pattern[str]:
    body = r"\s+".join(re.escape(part) for part in alt.split())

    # Whether a character is a word character that needs a word boundary in a literal pattern (not CJK).
    def bounded(ch: str) -> bool:
        return ch.isalnum() and script_of(ch) != "cjk"

    lead = r"(?<!\w)" if bounded(alt[:1]) else ""
    tail = r"(?!\w)" if bounded(alt[-1:]) else ""
    return re.compile(lead + body + tail, re.IGNORECASE)


# Language of the find text; it picks the Snowball stemmer for query AND page tokens.
def _query_language(find: str) -> str:
    from core.search.query_locales import analyze_query

    return analyze_query(_REGEX_FORM_RE.sub("", find or "") or find or "").language


# Parse the `find` text into a query (alternatives split by `|`, a /regex/ form, literals); None when it is
# empty.
def parse_query(find: str, language: str = "") -> _Query | None:
    raw = (find or "").strip()
    if not raw:
        return None
    regex_form = _REGEX_FORM_RE.match(raw)
    if regex_form:
        try:
            pattern = re.compile(regex_form.group(1), re.IGNORECASE | re.MULTILINE)
            return _Query(described=f"regex {raw}", regex=pattern)
        except re.error:
            raw = regex_form.group(1)
    query = _Query(described=f"“{raw}”")
    # Function words ("how", "the", "для") count for phrases but not as fallback terms.
    stopwords = stopwords_for(language) | stopwords_for(DEFAULT_LANGUAGE)
    seen_terms: list[frozenset[str]] = []
    all_terms: list[frozenset[str]] = []
    for alt in [a.strip() for a in _ALT_SPLIT_RE.split(raw) if a.strip()] or [raw]:
        raw_words = [m.group(0) for m in TOKEN_RE.finditer(alt)]
        words = [stems(word, language) for word in raw_words]
        if words:
            query.phrases.append(words)
            for raw_word, word in zip(raw_words, words):
                if not any(word & seen for seen in all_terms):
                    all_terms.append(word)
                    if raw_word.casefold() not in stopwords:
                        seen_terms.append(word)
        if re.search(r"[^\w\s]", alt) or not words:
            query.literals.append(_literal_pattern(alt))
    query.terms = seen_terms or all_terms
    return query


# Spans of whole-phrase hits (any inflection, consecutive words) in one unit.
def _phrase_spans(unit: _Unit, query: _Query) -> list[tuple[int, int]]:
    if query.regex is not None:
        return [(m.start(), m.end()) for m in query.regex.finditer(unit.text) if m.end() > m.start()]
    spans: list[tuple[int, int]] = []
    tokens = unit.tokens
    text = unit.text
    for phrase in query.phrases:
        n = len(phrase)
        for i in range(len(tokens) - n + 1):
            if not all(tokens[i + j][0] & phrase[j] for j in range(n)):
                continue
            if any(tokens[i + j + 1][1] - tokens[i + j][2] > _MAX_TOKEN_GAP for j in range(n - 1)):
                continue
            if any(re.search(r"\n\s*\n", text[tokens[i + j][2]:tokens[i + j + 1][1]]) for j in range(n - 1)):
                continue
            spans.append((tokens[i][1], tokens[i + n - 1][2]))
    for pattern in query.literals:
        spans.extend((m.start(), m.end()) for m in pattern.finditer(text) if m.end() > m.start())
    return sorted(set(spans))


# Distinct query words present in a unit, with their spans.
def _term_spans(unit: _Unit, query: _Query) -> dict[int, list[tuple[int, int]]]:
    found: dict[int, list[tuple[int, int]]] = {}
    for token_stems, start, end in unit.tokens:
        for idx, term in enumerate(query.terms):
            if token_stems & term:
                found.setdefault(idx, []).append((start, end))
    return found


# Heading text for the section path: link text only, no "¶" anchors, bounded length.
def _clean_heading(title: str) -> str:
    title = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", title)
    title = re.sub(r"[¶#]+", "", title).strip(" *_`")
    return title if len(title) <= 80 else title[:77].rstrip() + "…"


# Split the page into blocks with their kinds and the heading path each one sits under.
def _units(markdown: str, language: str = "") -> list[_Unit]:
    blocks = _split_blocks(markdown)
    meta_end = next((i + 1 for i, b in enumerate(blocks[:8]) if _RULE_RE.match(b.strip())), 0)
    stack: list[tuple[int, str, int]] = []
    units: list[_Unit] = []
    for idx, block in enumerate(blocks):
        kind = "meta" if idx < meta_end else _classify(block)
        if kind == "prose":
            lines = [line for line in block.splitlines() if line.strip()]
            if len(lines) >= 2 and sum(1 for line in lines if _LIST_ITEM_RE.match(line)) >= max(2, len(lines) // 2):
                kind = "list"
        if kind == "heading":
            match = _HEADING_RE.match(block.strip().split("\n", 1)[0])
            if match:
                level = len(match.group(1))
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, _clean_heading(match.group(2)), idx))
        section = " › ".join(title for _level, title, _i in stack[-3:])
        section_id = stack[-1][2] if stack else -1
        units.append(_Unit(idx, block, kind, section, section_id, stem_tokens(block, language), language or "en"))
    return units


# Whether two blocks sit under the same heading.
def _same_section(a: _Unit, b: _Unit) -> bool:
    return a.section_id == b.section_id


# Block range [lo, hi] that makes a hit in `units[i]` a self-contained passage.
def _expand(units: list[_Unit], i: int) -> tuple[int, int]:
    unit = units[i]
    lo = hi = i
    size = len(unit.text)

    if unit.kind == "heading":
        # A matched heading stands for its section: take the section's opening blocks.
        j = i + 1
        while j < len(units) and units[j].kind != "heading" and size < _PASSAGE_MIN_CHARS * 2:
            size += len(units[j].text)
            hi = j
            j += 1
        return lo, hi
    if unit.kind == "meta":
        return lo, hi

    prev = units[i - 1] if i > 0 else None
    nxt = units[i + 1] if i + 1 < len(units) else None
    # A code/table/list/math block needs the sentence that introduces it.
    if unit.kind in {"code", "table", "list", "math"} and prev is not None and prev.kind == "prose" \
            and _same_section(prev, unit) and (prev.text.rstrip().endswith(":") or len(prev.text) < 400):
        lo = i - 1
        size += len(prev.text)
    # A paragraph that announces something ("…as follows:") needs what it announces.
    if unit.kind == "prose" and unit.text.rstrip().endswith(":") and nxt is not None \
            and nxt.kind != "heading" and _same_section(nxt, unit):
        hi = i + 1
        size += len(nxt.text)
    # Too short to stand alone: grow inside the section, next block first.
    for _round in range(2):
        if size >= _PASSAGE_MIN_CHARS:
            break
        grown = False
        for j in (hi + 1, lo - 1):
            if 0 <= j < len(units) and units[j].kind not in {"heading", "meta", "rule"} \
                    and _same_section(units[j], unit) and size + len(units[j].text) <= _PASSAGE_MAX_CHARS:
                lo, hi = min(lo, j), max(hi, j)
                size += len(units[j].text)
                grown = True
                if size >= _PASSAGE_MIN_CHARS:
                    break
        if not grown:
            break
    return lo, hi


# Keep the sentences (NLTK Punkt) that contain a hit plus one neighbour each side.
# Chinese / Japanese sentences are joined without the space Latin text needs.
def _render_prose(text: str, spans: list[tuple[int, int]], language: str) -> str:
    sentences = sentence_spans(text, language)
    hit = {i for i, (s, e) in enumerate(sentences) for (hs, he) in spans if hs < e and he > s}
    keep = sorted({j for i in hit for j in (i - 1, i, i + 1) if 0 <= j < len(sentences)})
    return _join_kept([text[s:e].strip() for s, e in sentences], keep, "" if script_of(text) == "cjk" else " ")


# Join the kept parts in order, marking every skipped run (including head/tail) with "…".
def _join_kept(parts: list[str], keep: list[int], joiner: str) -> str:
    out: list[str] = []
    if keep and keep[0] > 0:
        out.append("…")
    for pos, idx in enumerate(keep):
        if pos and idx != keep[pos - 1] + 1:
            out.append("…")
        out.append(parts[idx])
    if keep and keep[-1] < len(parts) - 1:
        out.append("…")
    return joiner.join(out)


# Character spans of the lines of a text.
def _line_spans(text: str) -> list[tuple[int, int]]:
    out, start = [], 0
    for line in text.split("\n"):
        out.append((start, start + len(line)))
        start += len(line) + 1
    return out


# Indexes of the lines that overlap any hit span.
def _hit_lines(text: str, spans: list[tuple[int, int]]) -> set[int]:
    return {i for i, (s, e) in enumerate(_line_spans(text)) for (hs, he) in spans if hs <= e and he > s}


# List: a non-item lead line, the items with a hit (with their continuation lines).
def _render_list(text: str, spans: list[tuple[int, int]]) -> str:
    lines = text.split("\n")
    items: list[list[int]] = []
    for i, line in enumerate(lines):
        if _LIST_ITEM_RE.match(line) or not items:
            items.append([i])
        else:
            items[-1].append(i)
    hit_lines = _hit_lines(text, spans)
    keep = [k for k, item in enumerate(items) if hit_lines & set(item)]
    if items and not _LIST_ITEM_RE.match(lines[items[0][0]]) and 0 not in keep:
        keep = [0] + keep
    rendered = ["\n".join(lines[i] for i in item) for item in items]
    return _join_kept(rendered, keep, "\n")


# Cell texts of one markdown table row.
def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


# A row that only labels a group ("| Population |  |") - its values are in the rows below.
def _is_group_row(row: str) -> bool:
    cells = _cells(row)
    return len(cells) >= 2 and bool(cells[0]) and not any(cells[1:])


# Table: header + separator + matching rows. A matched group-label row brings the rows
# it labels; a match in the column header alone brings the first rows.
def _render_table(text: str, spans: list[tuple[int, int]]) -> str:
    lines = text.split("\n")
    head = 2 if len(lines) > 1 and re.match(r"^\s*\|?\s*:?-{3,}", lines[1]) else 1
    hit_lines = sorted(_hit_lines(text, spans))
    rows: set[int] = set()
    for i in hit_lines:
        if i < head:
            continue
        rows.add(i)
        if _is_group_row(lines[i]):
            # Rows of the group: bulleted sub-rows when the group uses them, else up to
            # the next group label.
            bulleted = i + 1 < len(lines) and bool(_SUBROW_RE.match(lines[i + 1]))
            j = i + 1
            while j < len(lines) and j - i <= _GROUP_ROWS and not _is_group_row(lines[j]) \
                    and (not bulleted or _SUBROW_RE.match(lines[j])):
                rows.add(j)
                j += 1
    if not rows and hit_lines:
        rows = set(range(head, min(len(lines), head + _HEADER_HIT_ROWS)))
    omitted = len(lines) - head - len(rows)
    out = lines[:head] + [lines[i] for i in sorted(rows)]
    if omitted > 0:
        out.append(f"… ({omitted} other rows omitted)")
    return "\n".join(out)


# Code: blank-line separated segments (functions, statements groups) that contain a hit.
def _render_code(text: str, spans: list[tuple[int, int]]) -> str:
    lines = text.split("\n")
    fence_open, fence_close = lines[0], (lines[-1] if len(lines) > 1 else "```")
    if not _fence_marker(fence_open):
        fence_open, fence_close = "```", "```"
    body_lines = lines[1:-1]
    segments: list[list[int]] = [[]]
    for i, line in enumerate(body_lines, start=1):
        if not line.strip() and segments[-1]:
            segments.append([])
        elif line.strip():
            segments[-1].append(i)
    segments = [seg for seg in segments if seg]
    hit_lines = _hit_lines(text, spans)
    keep = [k for k, seg in enumerate(segments) if hit_lines & set(seg)]
    rendered = ["\n".join(lines[i] for i in seg) for seg in segments]
    return f"{fence_open}\n{_join_kept(rendered, keep, chr(10))}\n{fence_close}"


# Context block (no hit) that is long: keep the side facing the hit.
def _render_context(unit: _Unit, before_hit: bool) -> str:
    text = unit.text
    if len(text) <= _CONTEXT_CHARS or unit.kind in {"code", "table", "math", "keep"}:
        return text
    sentences = sentence_spans(text, unit.language)
    parts = [text[s:e].strip() for s, e in sentences]
    if len(parts) < 2:
        return text
    picked: list[int] = []
    size = 0
    order = range(len(parts) - 1, -1, -1) if before_hit else range(len(parts))
    for i in order:
        if picked and size + len(parts[i]) > _CONTEXT_CHARS:
            break
        picked.append(i)
        size += len(parts[i])
    return _join_kept(parts, sorted(picked), " ")


# Render one block of a passage: a block with hits is cut to the hit parts when it is long, a context block
# to the side facing the hit.
def _render_unit(unit: _Unit, spans: list[tuple[int, int]] | None, before_hit: bool) -> str:
    if not spans:
        return _render_context(unit, before_hit)
    text = unit.text
    if unit.kind == "prose" and len(text) > _LONG_PROSE_CHARS:
        return _render_prose(text, spans, unit.language)
    if unit.kind == "list" and len(text) > _LONG_LIST_CHARS:
        return _render_list(text, spans)
    if unit.kind == "table" and len(text) > _LONG_TABLE_CHARS:
        return _render_table(text, spans)
    if unit.kind == "code" and len(text) > _LONG_CODE_CHARS and _is_fenced_code_block(text):
        return _render_code(text, spans)
    return text


# Decode a run of percent-escaped UTF-8 in a link target into readable characters; ASCII escapes and invalid
# bytes stay encoded.
def _decode_percent_run(match: re.Match[str]) -> str:
    raw = bytes.fromhex(match.group(0).replace("%", ""))
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError:
        return match.group(0)
    return "".join(
        ch if ord(ch) >= 0x80 and ch.isprintable() and not ch.isspace() else f"%{ord(ch):02X}"
        for ch in decoded
    )


# Remove footnote markers and decode percent-encoded link targets in a rendered passage.
def _tidy(text: str) -> str:
    # A removed marker takes the space before it, so "5642,7 [2] м" reads "5642,7 м".
    text = re.sub(r" ?(?:\x00)+( ?)", r"\1", _FOOTNOTE_RE.sub("\x00", text))
    return _PERCENT_RUN_RE.sub(_decode_percent_run, text)


# Render a passage as text from its blocks.
def _render_passage(units: list[_Unit], passage: _Passage) -> str:
    first_hit = min(passage.hit_units)
    pieces = []
    for idx in range(passage.lo, passage.hi + 1):
        pieces.append(_render_unit(units[idx], passage.hit_units.get(idx), before_hit=idx < first_hit))
    return _tidy("\n\n".join(piece for piece in pieces if piece.strip()))


# 0.25 … 1.0: share of the passage that is readable text once link targets and
# citation markers are stripped (code passages are always readable).
def _readability(text: str) -> float:
    if "```" in text:
        return 1.0
    visible = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    visible = re.sub(r"\[\[?\d+\]?\]", "", visible)
    # Letters and digits both carry content: a row of figures is data, not debris.
    words = re.findall(r"[^\W_]{2,}", visible)
    letters = sum(len(word) for word in words)
    if letters < 60:
        return 0.25
    # Measured on the visible text: a link reads as its label, so a well-linked article
    # or infobox is not punished for its URLs. Text that is almost all link labels is a
    # link list (nav, "see also") and ranks low.
    share = min(1.0, letters / max(1, len(visible)) / _READABLE_SHARE)
    labels = sum(len(m.group(1)) for m in re.finditer(r"!?\[([^\]]*)\]\([^)]*\)", text))
    link_share = labels / max(1, len(visible))
    if link_share > _LINK_LIST_SHARE:
        share *= (1.0 - link_share) / (1.0 - _LINK_LIST_SHARE)
    return max(0.25, share)


# Index of the first unit of the page's back matter: the section holding the first
# reference list - a block with several "#cite_ref" back-links, which MediaWiki writes in
# every language - and everything after it (notes, bibliography, external links).
# len(units) when the page has none.
def _back_matter_start(units: list[_Unit]) -> int:
    for unit in units:
        if len(_BACKLINK_RE.findall(unit.text)) >= _BACKLINKS_PER_LIST:
            j = unit.index
            while j > 0 and units[j].kind != "heading":
                j -= 1
            return j if units[j].kind == "heading" else unit.index
    return len(units)


# Group hit blocks into passages: neighbouring hits in one section share a passage while it stays under the
# size cap.
def _build_passages(
    units: list[_Unit], hits: dict[int, list[tuple[int, int]]]
) -> list[_Passage]:
    passages: list[_Passage] = []
    for i in sorted(hits):
        lo, hi = _expand(units, i)
        last = passages[-1] if passages else None
        if last is not None and lo <= last.hi + 1 and units[lo].section_id == units[last.lo].section_id \
                and sum(len(units[j].text) for j in range(last.lo, max(hi, last.hi) + 1)) <= _PASSAGE_MAX_CHARS:
            last.hi = max(last.hi, hi)
            last.hit_units[i] = hits[i]
        else:
            passages.append(_Passage(lo, hi, {i: hits[i]}))
    return passages


# Find a word or phrase in the full page text and return the best passages around the hits, each with its
# section path, within the hit and size budgets.
def find_in_page(
    markdown: str,
    find: str,
    *,
    url: str = "",
    max_hits: int = DEFAULT_MAX_HITS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str:
    text = _repair_unclosed_fence(markdown or "")
    language = _query_language(find)
    query = parse_query(find, language)
    if query is None:
        return text
    where = f" in {url}" if url else ""
    units = _units(text, language)

    hits = {u.index: spans for u in units if (spans := _phrase_spans(u, query))}
    fallback = False
    passages = _build_passages(units, hits) if hits else []
    for passage in passages:
        covered = {t for j in range(passage.lo, passage.hi + 1) for t in _term_spans(units[j], query)}
        passage.matches = sum(len(s) for s in passage.hit_units.values())
        passage.score = 2.0 * len(covered) / max(1, len(query.terms)) + min(passage.matches, 6) * 0.3

    if not passages and len(query.terms) >= 2:
        # No phrase anywhere: passages holding most of the words, by stem. Coverage is
        # checked per hit block and its own context BEFORE passages merge, so a word that
        # occurs everywhere ("column") cannot chain unrelated blocks into one passage.
        term_hits = {u.index: found for u in units if (found := _term_spans(u, query))}
        required = min(len(query.terms), max(2, math.ceil(len(query.terms) * 0.6)))
        qualified: dict[int, list[tuple[int, int]]] = {}
        for i, found in term_hits.items():
            lo, hi = _expand(units, i)
            covered = {t for j in range(lo, hi + 1) for t in term_hits.get(j, {})}
            if len(covered) >= required:
                qualified[i] = sorted(s for spans in found.values() for s in spans)
        for passage in _build_passages(units, qualified):
            covered = {t for j in range(passage.lo, passage.hi + 1) for t in term_hits.get(j, {})}
            passage.matches = sum(len(s) for s in passage.hit_units.values())
            passage.score = 2.0 * len(covered) / len(query.terms) + min(passage.matches, 6) * 0.1
            passages.append(passage)
        fallback = bool(passages)

    if not passages:
        return (
            f"No matches for {query.described}{where} (page text: {len(text):,} chars). "
            "Any inflection of the words is already matched; try other words, fewer words, "
            "alternatives like `a | b`, or omit `find` to read the page."
        )

    # Best passages that fit the budget, presented in page order. Passages that are mostly
    # link markup (nav, link lists) rank low, the back matter (notes, bibliography) last,
    # and passages far below the best one are only counted, not shown.
    rendered = {id(p): _render_passage(units, p) for p in passages}
    back_matter = _back_matter_start(units)
    for passage in passages:
        passage.score *= _readability(rendered[id(passage)])
        if passage.lo >= back_matter:
            passage.score *= _BACK_MATTER_WEIGHT
    floor = max(p.score for p in passages) * (_MIN_RELATIVE_SCORE_WORDS if fallback else _MIN_RELATIVE_SCORE)
    chosen: list[_Passage] = []
    used = 0
    for passage in sorted(passages, key=lambda p: (-p.score, p.lo)):
        cost = len(rendered[id(passage)]) + len(units[passage.lo].section) + 40
        if len(chosen) >= max_hits or (chosen and (used + cost > max_chars or passage.score < floor)):
            continue
        chosen.append(passage)
        used += cost
    chosen.sort(key=lambda p: p.lo)

    total = sum(p.matches for p in passages)
    header = (
        f"Find {query.described}{where}: {total} match(es) in {len(passages)} passage(s)"
        + (" — no exact phrase; passages containing most of the words" if fallback else "")
        + (f"; showing the {len(chosen)} most relevant, in page order" if len(chosen) < len(passages) else "")
        + f". Page text: {len(text):,} chars."
    )
    lines = [header, ""]
    for number, passage in enumerate(chosen, 1):
        section = units[passage.hit_units and min(passage.hit_units)].section
        lines.append(f"--- passage {number}/{len(chosen)}" + (f" · § {section}" if section else "") + " ---")
        body = rendered[id(passage)]
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + " …"
        lines.append(body)
        lines.append("")
    if len(chosen) < len(passages):
        lines.append(f"[{len(passages) - len(chosen)} more passage(s) not shown; narrow `find` to see them]")
    return "\n".join(lines).rstrip()
