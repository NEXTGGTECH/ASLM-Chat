# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
from pathlib import Path

from trafilatura.meta import reset_caches

from core.extract.content_processor import (
    _truncate_markdown_to_budget,
    compress_read_page_markdown,
)
from core.extract.page_normalizer import normalize_page


_FIXTURE = Path(__file__).parent / "fixtures" / "atomic_code_budget.html"
_URL = "https://fixture.test/atomic-budget"
_MARKERS = ("SMALL", "BOUNDARY", "OVERSIZED")


def _markdown() -> str:
    reset_caches()
    return normalize_page(_URL, _FIXTURE.read_text(encoding="utf-8"))


def _paired(output: str, marker: str) -> bool:
    return (f"SYNTH_{marker}_BEGIN" in output) == (f"SYNTH_{marker}_END" in output)


def _fences_balanced(output: str) -> bool:
    return len(re.findall(r"^[ \t]{0,3}(?:`{3,}|~{3,})", output, re.MULTILINE)) % 2 == 0


def _first_complete_budget(markdown: str, marker: str) -> int:
    needle = f"SYNTH_{marker}_BEGIN"
    low, high = 1, len(markdown) + 100
    while low < high:
        middle = (low + high) // 2
        if needle in _truncate_markdown_to_budget(markdown, middle):
            high = middle
        else:
            low = middle + 1
    return low


def test_atomic_thresholds_switch_without_partial_markers():
    markdown = _markdown()

    for marker in _MARKERS:
        threshold = _first_complete_budget(markdown, marker)
        below = _truncate_markdown_to_budget(markdown, threshold - 1)
        exact = _truncate_markdown_to_budget(markdown, threshold)

        assert _paired(below, marker)
        assert f"SYNTH_{marker}_BEGIN" not in below
        assert _paired(exact, marker)
        assert f"SYNTH_{marker}_BEGIN" in exact
        assert len(exact) <= threshold
        assert _fences_balanced(below)
        assert _fences_balanced(exact)


def test_skipped_code_does_not_discard_following_prose():
    markdown = _markdown()
    small_threshold = _first_complete_budget(markdown, "SMALL")

    output = _truncate_markdown_to_budget(markdown, small_threshold - 1)

    assert "SYNTH_SMALL_BEGIN" not in output
    # Extractors may escape underscores in prose without changing the visible text.
    assert "SYNTH_AFTER_SMALL" in output.replace("\\_", "_")
    assert _fences_balanced(output)


def test_production_compactor_never_emits_partial_synthetic_blocks():
    markdown = _markdown()
    source_code = re.findall(
        r"^[ \t]{0,3}(?:`{3,}|~{3,})[^\n]*\n.*?^[ \t]{0,3}(?:`{3,}|~{3,})[ \t]*$",
        markdown,
        re.MULTILINE | re.DOTALL,
    )

    for budget in (500, 1_000, 1_243, 1_244, 4_000, 6_546, 6_547, 10_000, 20_000):
        output = compress_read_page_markdown(
            markdown,
            url=_URL,
            focus="",
            max_chars=budget,
            compress_threshold=1,
            compress_target=budget,
            enable_compress=True,
        )

        assert all(block in output for block in source_code)
        assert _fences_balanced(output)
        assert all(f"SYNTH_{marker}_BEGIN" in output for marker in _MARKERS)
        assert all(f"SYNTH_{marker}_END" in output for marker in _MARKERS)

    assert len(
        compress_read_page_markdown(
            markdown,
            url=_URL,
            focus="",
            max_chars=500,
            compress_threshold=1,
            compress_target=500,
            enable_compress=True,
        )
    ) > 500


def test_query_compactor_pins_all_code_even_when_it_exceeds_budget():
    markdown = _markdown()

    boundary = compress_read_page_markdown(
        markdown,
        url=_URL,
        focus="boundary_value compute_boundary",
        max_chars=6_000,
        compress_threshold=1,
        compress_target=6_000,
        enable_compress=True,
    )
    oversized = compress_read_page_markdown(
        markdown,
        url=_URL,
        focus="oversized_value compute_oversized",
        max_chars=14_000,
        compress_threshold=1,
        compress_target=14_000,
        enable_compress=True,
    )

    for output in (boundary, oversized):
        assert all(f"SYNTH_{marker}_BEGIN" in output for marker in _MARKERS)
        assert all(f"SYNTH_{marker}_END" in output for marker in _MARKERS)
        assert _fences_balanced(output)
    assert len(boundary) > 6_000
    assert len(oversized) > 14_000
