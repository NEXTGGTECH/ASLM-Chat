# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from bs4 import BeautifulSoup

from core.extract.content_processor import extract_full_body_text
from core.extract.markdown_tables import (
    html_table_to_markdown,
    normalize_markdown_tables,
)
from core.extract.page_normalizer import normalize_page


# Parse the HTML and return its first table element.
def _table(html: str):
    return BeautifulSoup(html, "html.parser").find("table")


# Verify that an HTML table with a thead becomes a GFM table with the header row, the separator
# row and one line per body row.
def test_html_table_with_thead():
    md = html_table_to_markdown(_table(
        "<table><thead><tr><th>Model</th><th>RAM</th></tr></thead>"
        "<tbody><tr><td>A1</td><td>8GB</td></tr><tr><td>B2</td><td>16GB</td></tr></tbody></table>"
    ))
    assert md.splitlines()[0] == "| Model | RAM |"
    assert md.splitlines()[1] == "| --- | --- |"
    assert "| A1 | 8GB |" in md
    assert "| B2 | 16GB |" in md


# Verify that a table without th cells promotes its first row to the GFM header.
def test_html_table_header_promotion_without_th():
    # No <th> → first row is promoted to the header.
    md = html_table_to_markdown(_table(
        "<table><tr><td>Name</td><td>Score</td></tr><tr><td>x</td><td>9</td></tr></table>"
    ))
    lines = md.splitlines()
    assert lines[0] == "| Name | Score |"
    assert lines[1] == "| --- | --- |"
    assert lines[2] == "| x | 9 |"


# Verify that a blank table row is dropped and a literal pipe in a cell is escaped.
def test_html_table_skips_blank_rows_and_escapes_pipes():
    md = html_table_to_markdown(_table(
        "<table><tr><th>A</th><th>B</th></tr>"
        # The blank row is dropped.
        "<tr><td></td><td></td></tr>"
        # The literal pipe is escaped.
        "<tr><td>a|b</td><td>c</td></tr></table>"
    ))

    # There is no blank data row.
    assert "|  |  |" not in md
    assert r"a\|b" in md


# Verify that a row shorter than the header is padded with empty cells to the full table width.
def test_html_table_ragged_rows_padded():
    md = html_table_to_markdown(_table(
        "<table><tr><th>A</th><th>B</th><th>C</th></tr><tr><td>1</td><td>2</td></tr></table>"
    ))

    # The short row is padded to width 3.
    assert "| 1 | 2 |  |" in md


# Verify that the normalizer adds the missing leading pipe to a table whose header row lacks it.
def test_normalize_fixes_missing_leading_pipe():
    # trafilatura shape: header row lacks the leading `|`.
    raw = "Model | RAM | Price |\n|---|---|---|\n| A1 | 8GB | $100 |"
    fixed = normalize_markdown_tables(raw)
    for line in fixed.splitlines():
        assert line.startswith("| ")
    assert "| Model | RAM | Price |" in fixed


# Verify that a pipe character in ordinary prose is left untouched.
def test_normalize_leaves_prose_pipes_untouched():
    prose = "Use the pipe operator a | b to combine.\n\nAnother line."
    assert normalize_markdown_tables(prose) == prose


# Verify that text without any table is returned unchanged.
def test_normalize_no_table_is_noop():
    text = "# Title\n\nJust a paragraph.\n"
    assert normalize_markdown_tables(text) == text


# Verify that table-shaped lines inside a fenced code block are returned unchanged.
def test_normalize_ignores_table_shaped_code():
    code = "```text\nleft | right\n--- | ---\na | b\n```"

    assert normalize_markdown_tables(code) == code


# Verify that the full-body extraction renders a table as GFM markdown instead of flattening it,
# and still keeps the nav text.
def test_full_body_preserves_table_as_markdown():
    # A landing page rescued by full-body must keep its table, not flatten it.
    html = (
        "<html><body><nav><a href='/'>Home</a></nav>"
        "<table><tr><th>Plan</th><th>Price</th></tr>"
        "<tr><td>Pro</td><td>$20</td></tr></table>"
        "<footer>Contact</footer></body></html>"
    )
    text = extract_full_body_text(html)
    assert "| Plan | Price |" in text
    assert "| Pro | $20 |" in text

    # The nav chrome is still kept.
    assert "Home" in text


# Verify that normalizing a full page yields valid GFM table rows (each starting with a pipe),
# including a repaired header row, as trafilatura emits that row without the leading pipe.
def test_normalize_page_emits_valid_gfm_table():
    # End-to-end: when trafilatura's formatted path keeps the table, the header row it emits
    # WITHOUT a leading pipe must be repaired into valid GFM by the normalizer.
    html = (
        "<html><head><title>Specs</title></head><body><article>"
        "<h2>Comparison of models</h2>"
        "<table><thead><tr><th>Model</th><th>RAM</th><th>Price</th></tr></thead>"
        "<tbody><tr><td>A1</td><td>8GB</td><td>100 USD</td></tr>"
        "<tr><td>B2</td><td>16GB</td><td>200 USD</td></tr></tbody></table>"
        "<p>A paragraph of real article text, long enough to survive cleaning and keep "
        "the page off the thin-content rescue path here for sure.</p>"
        "</article></body></html>"
    )
    md = normalize_page("https://x.com/specs", html)
    table_lines = [ln for ln in md.splitlines() if "|" in ln]
    assert table_lines, "a table should be present"
    for ln in table_lines:
        # Every row is valid GFM.
        assert ln.startswith("| ")
    assert "| Model | RAM | Price |" in md
