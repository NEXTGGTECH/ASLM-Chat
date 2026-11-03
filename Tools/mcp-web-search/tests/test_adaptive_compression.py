# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio

import pytest

from core.extract.adaptive_compression import (
    CompressionCurve,
    compress_adaptive,
    compression_note,
    insert_note,
)

pytestmark = pytest.mark.unit


# Build one prose paragraph of about `size` characters that mentions its index.
def _prose(index: int, size: int = 900) -> str:
    sentence = (
        f"Paragraph {index} discusses measured value {index * 7} in detail and explains why it "
        "matters for the overall result of the experiment. "
    )
    return (sentence * (size // len(sentence) + 1))[:size].rstrip() + "."


# Build a markdown page with a header and `blocks` paragraphs.
def _page(blocks: int = 120) -> str:
    parts = ["# Example Page", "**URL:** https://example.com/page", "---"]
    for i in range(blocks):
        if i % 10 == 0:
            parts.append(f"## Section {i // 10}")
        parts.append(_prose(i))
    return "\n\n".join(parts)



# Verify that curve is identity below the soft limit.
def test_curve_is_identity_below_the_soft_limit():
    curve = CompressionCurve(soft_limit=60_000, hard_cap=0)
    for length in (0, 1, 30_000, 60_000):
        assert curve.target(length) == length


# Verify that curve hits the configured ratio at double the limit.
def test_curve_hits_the_configured_ratio_at_double_the_limit():
    curve = CompressionCurve(soft_limit=60_000, ratio_at_double=0.30, max_ratio=0.85, hard_cap=0)
    assert curve.target(120_000) == pytest.approx(84_000, rel=0.001)


# Verify that curve ratio grows nonlinearly and saturates.
def test_curve_ratio_grows_nonlinearly_and_saturates():
    curve = CompressionCurve(soft_limit=60_000, ratio_at_double=0.30, max_ratio=0.85, hard_cap=0)
    ratios = [curve.ratio(60_000 * 2**n) for n in range(1, 8)]
    assert ratios == sorted(ratios)
    assert all(r < 0.85 for r in ratios)
    assert curve.ratio(10**9) < 0.85
    # Each doubling adds less than the previous one: concave, not linear.
    steps = [b - a for a, b in zip(ratios, ratios[1:])]
    assert steps == sorted(steps, reverse=True)


# Verify that kept size never shrinks when the page grows.
def test_kept_size_never_shrinks_when_the_page_grows():
    curve = CompressionCurve(soft_limit=60_000, ratio_at_double=0.30, max_ratio=0.85, hard_cap=0)
    kept = [curve.target(n) for n in range(60_000, 2_000_001, 20_000)]
    assert kept == sorted(kept)


# Verify that extreme config cannot invert the curve.
def test_extreme_config_cannot_invert_the_curve():
    curve = CompressionCurve(soft_limit=10_000, ratio_at_double=0.99, max_ratio=0.99, hard_cap=0)
    kept = [curve.target(n) for n in range(10_000, 1_000_001, 10_000)]
    assert kept == sorted(kept)


# Verify that hard cap bounds the target.
def test_hard_cap_bounds_the_target():
    curve = CompressionCurve(soft_limit=60_000, hard_cap=150_000)
    assert curve.target(5_000_000) == 150_000



# Verify that under limit pages are returned verbatim.
def test_under_limit_pages_are_returned_verbatim():
    page = _page(10)
    out, report = compress_adaptive(page, CompressionCurve(soft_limit=len(page) + 1))
    assert out == page
    assert not report.compressed


# Verify that output tracks the curve target.
def test_output_tracks_the_curve_target():
    page = _page(160)
    curve = CompressionCurve(soft_limit=len(page) // 2, hard_cap=0)
    out, report = compress_adaptive(page, curve, url="https://example.com/page")
    assert report.compressed
    assert len(out) == pytest.approx(curve.target(len(page)), rel=0.08)


# Verify that structural markup survives verbatim.
def test_structural_markup_survives_verbatim():
    code = "```python\ndef keep_me(x):\n    return x * 2\n```"
    table = "| Option | Value |\n| --- | --- |\n| alpha | 1 |\n| beta | 2 |"
    math = "$$\nE = mc^2\n\n\\int_0^1 x\\,dx\n$$"
    kept_region = "<!-- keep -->\nSpecial handler text that must stay.\n\nSecond line.\n<!-- /keep -->"
    parts = ["# Title", "**URL:** https://example.com/x", "---"]
    for i in range(150):
        parts.append(_prose(i))
        if i == 40:
            parts.extend([code, "### Deep heading", table])
        if i == 90:
            parts.extend([math, kept_region])
    page = "\n\n".join(parts)

    out, report = compress_adaptive(page, CompressionCurve(soft_limit=20_000, hard_cap=0))

    assert report.compressed and len(out) < len(page) * 0.6
    for fragment in (code, table, math, kept_region, "### Deep heading", "# Title",
                     "**URL:** https://example.com/x"):
        assert fragment in out
    assert out.count("```") % 2 == 0


# Verify that link farm and boilerplate go before real prose.
def test_link_farm_and_boilerplate_go_before_real_prose():
    nav = "\n".join(f"- [Nav item {i}](https://example.com/n{i})" for i in range(12))
    cookie = "We use cookies. Accept all cookies to continue. Privacy policy."
    parts = ["# Article", "---"]
    for i in range(60):
        parts.append(_prose(i))
        parts.extend([nav, cookie])
    page = "\n\n".join(parts)

    out, _report = compress_adaptive(page, CompressionCurve(soft_limit=len(page) // 2, hard_cap=0))

    # Repeated nav/cookie blocks cover the whole cut, so no real paragraph is touched.
    assert out.count("Nav item 0") < 15
    assert out.count("Accept all cookies") < 15
    assert all(_prose(i) in out for i in range(60))


# Verify that long mid blocks are trimmed not only dropped.
def test_long_mid_blocks_are_trimmed_not_only_dropped():
    page = _page(140)
    _out, report = compress_adaptive(page, CompressionCurve(soft_limit=len(page) // 2, hard_cap=0))
    assert report.trimmed_blocks > 0


# Verify that gaps are marked and note is inserted after header.
def test_gaps_are_marked_and_note_is_inserted_after_header():
    page = _page(160)
    out, report = compress_adaptive(page, CompressionCurve(soft_limit=len(page) // 3, hard_cap=0))
    assert "[… omitted ~" in out
    noted = insert_note(out, compression_note(report))
    header, _sep, rest = noted.partition("---")
    assert "# Example Page" in header
    assert rest.lstrip().startswith("> read_page: long page compressed")
    assert "`find`" in noted


# Verify that hard cap is enforced even when protected markup dominates.
def test_hard_cap_is_enforced_even_when_protected_markup_dominates():
    code = "```\n" + "\n".join(f"line_{i} = {i}" for i in range(4000)) + "\n```"
    page = "# T\n\n---\n\n" + _prose(1) + "\n\n" + code
    out, report = compress_adaptive(page, CompressionCurve(soft_limit=5_000, hard_cap=20_000))
    assert len(out) <= 20_000
    assert report.hard_capped


# Verify that read_page uses the adaptive curve when no max_chars is given.
def test_read_service_uses_adaptive_mode_without_max_chars(monkeypatch):
    from core.read import service as read_service
    from core.read.service import PageResult, ReadPageOptions, ReadPageService

    page = _page(200)

    async def fake_generic_read(self, req):
        return PageResult(markdown=page, ok=True, method="httpx", apply_budget=True)

    monkeypatch.setattr(ReadPageService, "_generic_read", fake_generic_read)
    monkeypatch.setattr(read_service, "validate_public_fetch_url", lambda url: None)
    monkeypatch.setattr(read_service.custom_domains, "match", lambda url: None)

    tool_out = asyncio.run(read_service.run_read_page("https://example.com/page", allow_browser=False))
    assert "> read_page: long page compressed" in tool_out
    assert len(tool_out) > 20_000  # no longer the old fixed 20k cut

    budget_service = ReadPageService(options=ReadPageOptions(max_chars=8_000, focus="measured value"))
    budget_out = asyncio.run(budget_service.read("https://example.com/page"))
    assert "> read_page: long page compressed" not in budget_out
    assert len(budget_out) <= 9_000
