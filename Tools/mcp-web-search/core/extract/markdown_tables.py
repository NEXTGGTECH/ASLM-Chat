# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re

# A markdown table separator row: optional pipes, only spaces, colons, pipes and dashes, and at
# least one dash (for example "| --- | :---: |").
_SEP_LINE_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")


# Escape a cell so its content can't break the markdown table grid: whitespace runs collapse to
# one space and pipes are backslash-escaped.
def _cell(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).replace("|", "\\|").strip()


# Render one bs4 <table> node to a GitHub-flavoured markdown table, or "" when it has no usable
# rows. Fully blank rows are skipped, and a table without <th> cells gets its first row as the
# header (header promotion). Short rows are padded to the widest row. Used by the full-body and
# data-island rescue paths, which would otherwise flatten a table into plain text and lose the
# column structure.
def html_table_to_markdown(table) -> str:
    rows: list[list[str]] = []
    header_from_th = False
    for tr in table.find_all("tr"):
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        values = [_cell(c.get_text(separator=" ", strip=True)) for c in cells]

        # Skip fully blank rows.
        if not any(values):
            continue
        if not rows and any(c.name == "th" for c in cells):
            header_from_th = True
        rows.append(values)
    if not rows:
        return ""

    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]

    # Header promotion: a <th>-less table still gets a header, its first row. Either way the
    # header is row 0.
    header = rows[0]
    body = rows[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
    ]
    for r in body:
        lines.append("| " + " | ".join(r) + " |")

    # Both header sources land in row 0; the flag is kept for clarity and debugging.
    _ = header_from_th
    return "\n".join(lines)


# Replace every <table> in a bs4 tree (mutating it) with a positional marker and return the
# {marker: markdown} map. A table that renders to nothing is removed. The caller substitutes
# the markers back after get_text(), so tables keep their place in the surrounding text instead
# of collapsing into it.
def swap_tables_for_markers(soup, tag_prefix: str = "\x00TBL") -> dict[str, str]:
    markers: dict[str, str] = {}
    for i, table in enumerate(soup.find_all("table")):
        md = html_table_to_markdown(table)
        marker = f"{tag_prefix}{i}\x00"
        markers[marker] = md
        table.replace_with(marker if md else "")
    return markers


# Turn one markdown table block (list of raw lines) into repaired GFM lines. Ensure a leading and
# trailing `|` on every row, insert a separator after the header when missing, and drop
# all-blank rows. trafilatura's formatted output emits real tables but drops the leading `|` on
# the header row and can omit the separator; this restores both so every renderer parses the table.
def _repair_table_block(block: list[str]) -> list[str]:
    # Split one row into its stripped cells, ignoring one outer pipe on each side.
    def cells(line: str) -> list[str]:
        s = line.strip()
        s = s.removeprefix("|").removesuffix("|")
        return [c.strip() for c in s.split("|")]

    # Collect the non-blank data rows; existing separator rows are dropped and rebuilt below.
    data_rows: list[list[str]] = []
    has_sep = False
    for line in block:
        if _SEP_LINE_RE.match(line):
            has_sep = True
            continue
        row = cells(line)
        if any(row):
            data_rows.append(row)

    # Nothing recognisable: leave the block as it is.
    if not data_rows:
        return block

    # Pad every row to the widest one, then emit the header, a fresh separator and the body.
    width = max(len(r) for r in data_rows)
    data_rows = [r + [""] * (width - len(r)) for r in data_rows]
    out = ["| " + " | ".join(data_rows[0]) + " |"]
    out.append("| " + " | ".join(["---"] * width) + " |")
    for r in data_rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    _ = has_sep
    return out


# Repair every markdown table in a document. A table block is a run of ≥2 consecutive
# non-blank lines that all contain `|` AND include a dashed separator row — so prose lines
# with an incidental pipe are never touched. Lines inside code fences are copied unchanged.
def normalize_markdown_tables(markdown: str) -> str:
    from core.extract.markdown_code import closes_markdown_fence, markdown_fence_marker

    if "|" not in (markdown or ""):
        return markdown
    lines = markdown.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)
    fence = ""
    while i < n:
        current = markdown_fence_marker(lines[i])

        # Inside a code fence: copy the line; a valid closing fence ends the region.
        if fence:
            if closes_markdown_fence(lines[i], fence):
                fence = ""
            out.append(lines[i])
            i += 1
            continue

        # A fence opener starts a protected region.
        if current:
            fence = current
            out.append(lines[i])
            i += 1
            continue

        # Gather a maximal run of consecutive pipe-bearing non-blank lines.
        j = i
        while j < n and lines[j].strip() and "|" in lines[j]:
            j += 1
        run = lines[i:j]
        if len(run) >= 2 and any(_SEP_LINE_RE.match(ln) for ln in run):
            out.extend(_repair_table_block(run))
            i = j
        else:
            out.append(lines[i])
            i += 1
    return "\n".join(out)
