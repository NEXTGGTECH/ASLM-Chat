# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from pathlib import Path

from core.extract.content_processor import compress_read_page_markdown
from core.extract.dom_block_extractor import extract_dom_blocks
from core.extract.page_normalizer import normalize_page

# Real HTML fixture that imitates navigation, UI and SEO junk around a short article.
_FIXTURE = Path(__file__).parent / "fixtures" / "nav_seo_junk.html"


# Return the HTML of the junk fixture.
def _html() -> str:
    return _FIXTURE.read_text(encoding="utf-8")


# Distinctive real-content phrases vs. junk labels that must be filtered out.
_CONTENT_MARKERS = ("4.2 volts", "constant-current", "18650")
_JUNK_MARKERS = ("Shopping Cart", "Wishlist", "Accept All Cookies", "RSS Feed", "Buy Now Discount")


# Verify that the DOM block extractor rejects at least three navigation link clusters of the
# fixture while the real content survives and the nav and footer link farms are gone.
def test_dom_block_extractor_rejects_nav_clusters():
    from core.extract.content_processor import _preclean_html

    blocks, stats = extract_dom_blocks(_preclean_html(_html()), url="https://example.com/guide")
    joined = "\n\n".join(blocks)

    # The link clusters are rejected.
    assert int(stats["nav_rejected"]) >= 3

    # The real content survives.
    assert "4.2 volts" in joined

    # The navigation cluster is gone.
    assert "Shopping Cart" not in joined

    # The footer link farm is gone.
    assert "RSS Feed" not in joined


# Verify that normalizing the fixture page keeps every real-content marker and drops every junk
# marker.
def test_normalize_page_drops_junk_keeps_content():
    md = normalize_page("https://example.com/guide", raw_html=_html())

    for marker in _CONTENT_MARKERS:
        assert marker in md, f"content dropped: {marker!r}"
    for junk in _JUNK_MARKERS:
        assert junk not in md, f"junk survived: {junk!r}"


# Verify that normalizing a semantic article keeps its heading, link and code fence and drops the
# navigation outside the article.
def test_formatted_primary_keeps_structure_and_links():
    html = """<html><body>
    <main>
      <nav><a href='/menu'>Menu</a></nav>
      <article>
        <h2>Structured reference</h2>
        <p>This semantic article contains enough substantial prose for extraction and
        links to the <a href='/guide'>complete guide</a> without relying on host rules.</p>
        <pre><code>first line\n    indented line</code></pre>
        <p>A second substantial paragraph keeps the article above the extraction floor
        while the surrounding navigation remains outside the nested article.</p>
      </article>
    </main>
    </body></html>"""

    md = normalize_page("https://example.com/reference", raw_html=html)

    assert "# Structured reference" in md
    assert "[complete guide](https://example.com/guide)" in md
    assert "```" in md
    assert "Menu" not in md


# Verify that the formatted trafilatura extraction is called with deduplicate=False, so its
# process-global dedupe cache cannot drop repeated content.
def test_formatted_extraction_disables_process_global_dedupe(monkeypatch):
    import core.extract.page_normalizer as normalizer

    captured = {}

    # Record the extraction keyword arguments and return a fixed output.
    def fake_extract(_html, **kwargs):
        captured.update(kwargs)
        return "Substantial formatted extraction output that safely exceeds no limits."

    monkeypatch.setattr(normalizer.trafilatura, "extract", fake_extract)
    normalizer._extract_with_trafilatura_formatted("<article>content</article>")

    assert captured["deduplicate"] is False


# Verify that the content router returns a complete formatted extraction as is and does not run
# the DOM block fallback.
def test_content_router_keeps_complete_formatted_result(monkeypatch):
    import core.extract.page_normalizer as normalizer

    formatted = (
        "Formatted article with [a live link](https://example.com/guide).\n\n"
        "```python\nprint('structure survives')\n```\n\n"
        + "Substantial reference prose. " * 12
    )
    monkeypatch.setattr(normalizer, "_extract_with_trafilatura_formatted", lambda *_args: formatted)

    # Fail the test if the DOM fallback is reached.
    def unexpected_dom(*_args):
        raise AssertionError("DOM fallback ran after a complete formatted extraction")

    monkeypatch.setattr(normalizer, "_extract_with_dom_blocks", unexpected_dom)

    result = normalizer._extract_content(
        "<article>source</article>", None, "https://example.com"
    )

    assert result == formatted


# Verify that strict cleaning keeps a short factual line and a long explanatory paragraph that
# contains a word typical of social controls.
def test_cleaning_keeps_short_facts_and_long_prose_with_common_words():
    from core.extract.page_normalizer import _clean_content

    text = (
        "Added in version 3.11.\n\n"
        "Processes share memory through an implementation-specific mechanism, and this "
        "long explanatory paragraph must remain content even though it contains a word "
        "that can also occur in a compact social control."
    )

    cleaned = _clean_content(text, strict=True)

    assert "Added in version 3.11." in cleaned
    assert "Processes share memory" in cleaned


# Verify that block deduplication keeps two blocks that share a long prefix but end in different
# facts.
def test_dedupe_does_not_merge_common_prefixes():
    from core.extract.content_processor import _dedupe_blocks

    prefix = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen"
    blocks = [f"{prefix} alpha fact.", f"{prefix} beta fact."]

    assert _dedupe_blocks(blocks) == blocks


# Verify that a character budget smaller than the code block keeps the fenced code whole (even
# though the output exceeds the budget) and appends the truncation marker.
def test_budget_never_cuts_through_fenced_code():
    code = "```python\n" + "\n".join(f"    value_{i} = {i}" for i in range(30)) + "\n```"
    text = "Introductory prose before the example.\n\n" + code + "\n\nTrailing prose."

    out = compress_read_page_markdown(
        text,
        max_chars=220,
        compress_threshold=10**9,
        compress_target=0,
        enable_compress=False,
    )

    assert out.count("```") % 2 == 0
    assert code in out
    assert len(out) > 220
    assert "[...truncated]" in out


# Verify that query-aware micro pruning leaves a fenced code block verbatim.
def test_micro_prune_preserves_fenced_code_verbatim():
    code = "```python\nasync def main():\n    await important_query_call()\n```"
    text = "Useful prose about the important query.\n\n" + code

    out = compress_read_page_markdown(
        text,
        focus="important query",
        max_chars=5_000,
        compress_threshold=10**9,
        compress_target=0,
        enable_compress=True,
    )

    assert code in out
    assert out.count("```") == 2


# Verify that an unclosed code fence is treated as prose: the fence marker is dropped and the text
# after it stays in the output instead of being swallowed as code.
def test_unclosed_fence_degrades_to_prose_instead_of_swallowing_tail():
    text = (
        "Introductory paragraph.\n\n"
        "```python\n"
        "useful_call()\n\n"
        "Important explanatory tail that must remain eligible for compaction."
    )

    out = compress_read_page_markdown(
        text,
        max_chars=5_000,
        compress_threshold=10**9,
        compress_target=0,
        enable_compress=True,
    )

    assert "```" not in out
    assert "useful_call()" in out
    assert "Important explanatory tail" in out


# Verify that micro pruning keeps the factual clause of a sentence and drops the keyword-stuffed,
# fact-poor clause.
def test_micro_prune_drops_keyword_stuffed_clause():
    # One sentence: a factual clause (kept) + a query-keyword-stuffed, fact-poor clause
    # (an "SEO tumor" — what micro_chunk_worker targets).
    text = (
        "Lithium-ion cells should be charged to 4.2 volts per cell, "
        "buy lithium battery charging lithium battery voltage lithium battery "
        "best lithium battery deals online."
    )
    out = compress_read_page_markdown(
        text,
        focus="lithium battery charging voltage",
        max_chars=5_000,
        # The huge threshold skips budget compaction, so only the micro prune is exercised.
        compress_threshold=10**9,
        compress_target=0,
        enable_compress=True,
    )

    # The factual clause is kept.
    assert "4.2 volts" in out

    # The keyword-stuffed clause is dropped.
    assert "best lithium battery deals" not in out


# Verify that prune_micro_chunks reports at least one dropped clause in its debug info and keeps
# the factual clause.
def test_micro_prune_debug_reports_drop():
    from core.extract.micro_chunk_worker import prune_micro_chunks

    text = (
        "Lithium-ion cells should be charged to 4.2 volts per cell, "
        "buy lithium battery charging lithium battery voltage lithium battery "
        "best lithium battery deals online."
    )
    pruned, dbg = prune_micro_chunks(text, "lithium battery charging voltage")
    assert dbg.clauses_dropped >= 1
    assert "4.2 volts" in pruned


# Verify that without a focus query the micro pruning removes nothing.
def test_micro_prune_noop_without_query():
    # Standalone read_page (no focus) must not prune anything.
    text = "Click here to subscribe to our newsletter for exclusive deals and discounts today."
    out = compress_read_page_markdown(
        text, focus="", max_chars=5_000, compress_threshold=10**9, compress_target=0,
    )
    assert out == text
