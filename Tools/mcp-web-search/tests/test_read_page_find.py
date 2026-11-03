# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio

import pytest

from core.extract.page_find import find_in_page, parse_query
from core.extract.stemming import stems

pytestmark = pytest.mark.unit

_PAGE = "\n\n".join([
    "# FTS5 Reference",
    "**URL:** https://sqlite.org/fts5.html",
    "---",
    "## Overview",
    "FTS5 is a virtual table module that provides full-text search for applications.",
    "## Ranking",
    "### The bm25() function[¶](https://sqlite.org/fts5.html#bm25)",
    "The built-in bm25() function returns a real value. Each column may be assigned a weight. "
    "Weights are passed positionally, one per column.",
    "For example, the following query ranks title matches higher:",
    "```sql\nSELECT * FROM email WHERE email MATCH 'fts5' ORDER BY bm25(email, 10.0, 5.0);\n```",
    "## Tokenizers",
    "Индексы токенов хранятся отдельно, длина токена ограничена 32768 байтами.",
    "Поддерживаемые токенизаторы:",
    "- unicode61 — по умолчанию\n- ascii — только ASCII\n- trigram — поиск по подстрокам",
])


# Split a find output into the text of its passages.
def _passages(out: str) -> list[str]:
    return [chunk.split("---\n", 1)[1] for chunk in out.split("--- passage ")[1:]]



# Verify that inflected forms share a stem.
@pytest.mark.parametrize("forms", [
    ["индекс", "индекса", "индексы", "индексами", "индексов"],
    ["токен", "токены", "токенов", "токенами"],
    ["высота", "высоты", "высоте", "высотой"],
    ["весовой", "весовые", "весовых"],
    ["хранить", "хранятся", "хранится"],
    ["weight", "weights", "weighted", "weighting"],
    ["rank", "ranks", "ranked", "ranking"],
    ["index", "indexes", "indexed"],
    ["use", "uses", "used", "using"],
    ["study", "studies", "studied"],
])
def test_inflections_share_a_stem(forms):
    sets = [stems(word) for word in forms]
    assert all(a & b for a in sets for b in sets)


# Verify that different words do not match.
@pytest.mark.parametrize("a,b", [("токен", "токенизатор"), ("io", "function"), ("bm25", "bm2")])
def test_different_words_do_not_match(a, b):
    assert not stems(a) & stems(b)



# Verify that an empty `find` returns the page text unchanged.
def test_empty_find_returns_the_page_unchanged():
    assert find_in_page(_PAGE, "") == _PAGE
    assert find_in_page(_PAGE, "   ") == _PAGE


# Verify that any inflection matches in both languages.
def test_any_inflection_matches_both_languages():
    assert "Индексы токенов" in find_in_page(_PAGE, "индексами")
    assert "Индексы токенов" in find_in_page(_PAGE, "токен")
    assert "Each column may be assigned a weight" in find_in_page(_PAGE, "weights")


# Verify that short words match whole tokens only.
def test_short_words_match_whole_tokens_only():
    assert find_in_page(_PAGE, "IO").startswith("No matches")


# Verify that a phrase matches consecutive words in any inflection.
def test_phrase_matches_consecutive_words_in_any_inflection():
    out = find_in_page(_PAGE, "ranking title")
    assert "no exact phrase" not in out
    assert "ranks title matches" in out


# Verify that the query falls back to passages holding most of the words when no phrase matches.
def test_words_fallback_when_no_phrase():
    out = find_in_page(_PAGE, "column weight bm25")
    assert "no exact phrase" in out
    assert "assigned a weight" in out


# Verify that alternatives (`a | b`), /regex/ and punctuation queries are supported.
def test_alternatives_regex_and_literal_punctuation():
    assert "2 match(es)" in find_in_page(_PAGE, "trigram | 32768")
    assert "bm25(email, 10.0, 5.0)" in find_in_page(_PAGE, r"/bm25\(\w+,\s*[\d.]+/")
    assert "bm25(email, 10.0, 5.0)" in find_in_page(_PAGE, "bm25(email")
    query = parse_query("/bm25(/")  # invalid regex -> literal
    assert query is not None and query.regex is None and query.literals


# Verify that a query with no match returns a message explaining how to retry instead of an empty result.
def test_no_match_message():
    out = find_in_page(_PAGE, "zzqx nothing", url="https://x.test/")
    assert out.startswith("No matches for “zzqx nothing” in https://x.test/")
    assert "omit `find`" in out



# Verify that a passage joins the lead-in with the block it introduces.
def test_passage_joins_lead_in_with_the_block_it_introduces():
    (passage,) = _passages(find_in_page(_PAGE, "ORDER BY bm25"))
    assert passage.index("the following query ranks title") < passage.index("```sql")
    assert passage.count("```") == 2


# Verify that a lead-in announcing a list brings the list.
def test_lead_in_announcing_a_list_brings_the_list():
    (passage,) = _passages(find_in_page(_PAGE, "поддерживаемые токенизаторы"))
    assert "- trigram — поиск по подстрокам" in passage


# Verify that the section path is clean.
def test_section_path_is_clean():
    out = find_in_page(_PAGE, "weights")
    # The page title sits in the metadata header, so sections start below it.
    assert "§ Ranking › The bm25() function ---" in out
    assert "¶" not in out.split("\n")[2]


# Verify that passages never cross sections.
def test_passages_never_cross_sections():
    out = find_in_page(_PAGE, "трigram | fts5 | токенов")
    for passage in _passages(out):
        assert not ("Индексы токенов" in passage and "virtual table module" in passage)


# Verify that a long paragraph is cut to whole sentences around the hit.
def test_long_paragraph_is_cut_to_whole_sentences_around_the_hit():
    sentences = [f"Sentence number {i} talks about filler topic {i}." for i in range(80)]
    sentences[40] = "Here the needle phrase appears exactly once."
    page = "# T\n\n---\n\n## S\n\n" + " ".join(sentences)
    (passage,) = _passages(find_in_page(page, "needle phrase"))
    assert passage.startswith("… Sentence number 39 talks about filler topic 39.")
    assert "Here the needle phrase appears exactly once." in passage
    assert passage.rstrip().endswith("Sentence number 41 talks about filler topic 41. …")


# Verify that a long table returns header and matching rows.
def test_long_table_returns_header_and_matching_rows():
    rows = "\n".join(f"| opt_{i} | value {i} with a longer description of the option |" for i in range(120))
    table = "| Option | Usage |\n| --- | --- |\n" + rows
    page = "# T\n\n---\n\n## Options\n\n" + table
    (passage,) = _passages(find_in_page(page, "opt_77"))
    lines = passage.splitlines()
    assert lines[:2] == ["| Option | Usage |", "| --- | --- |"]
    assert any("opt_77" in line for line in lines)
    assert "other rows omitted" in passage


# Verify that a long code block returns matching segments inside a fence.
def test_long_code_returns_matching_segments_inside_a_fence():
    funcs = "\n\n".join(f"def func_{i}(x):\n    return x + {i}" for i in range(400))
    page = "# T\n\n---\n\n## Code\n\n```python\n" + funcs + "\n```"
    (passage,) = _passages(find_in_page(page, "func_123"))
    assert passage.startswith("```python")
    assert passage.rstrip().endswith("```")
    assert "def func_123(x):\n    return x + 123" in passage
    assert "func_300" not in passage


# Verify that the output is bounded and reports hidden passages.
def test_output_is_bounded_and_reports_hidden_passages():
    page = "# T\n\n---\n\n" + "\n\n".join(
        f"## Part {i}\n\nThe needle sits in part {i} with a sentence of context around it." for i in range(60)
    )
    out = find_in_page(page, "needle", max_hits=5, max_chars=4_000)
    assert len(_passages(out)) == 5
    assert "more passage(s) not shown" in out


# Verify that link debris ranks below real prose.
def test_link_debris_ranks_below_real_prose():
    debris = "".join(f"[x{i}](https://e.test/{i})" for i in range(30)) + " needle"
    prose = "The needle is explained here in a full sentence with enough readable words to matter."
    page = "# T\n\n---\n\n## A\n\n" + debris + "\n\n## B\n\n" + prose
    out = find_in_page(page, "needle", max_hits=1)
    assert prose in out and "e.test" not in out



# Verify that `find` searches the full extracted page, not the compressed text.
def test_read_service_find_searches_full_text_not_compressed(monkeypatch):
    from core.read import service as read_service
    from core.read.service import PageResult, ReadPageService

    body = "\n\n".join(f"Filler paragraph {i} " + "words " * 150 for i in range(400))
    page = f"# Huge\n\n---\n\n{body}\n\nThe hidden fact is 4242 widgets."

    async def fake_generic_read(self, req):
        return PageResult(markdown=page, ok=True, method="httpx", apply_budget=True)

    monkeypatch.setattr(ReadPageService, "_generic_read", fake_generic_read)
    monkeypatch.setattr(read_service, "validate_public_fetch_url", lambda url: None)
    monkeypatch.setattr(read_service.custom_domains, "match", lambda url: None)

    out = asyncio.run(read_service.run_read_page("https://example.com/huge", find="4242 widget"))
    assert out.startswith("Find “4242 widget”")
    assert "The hidden fact is 4242 widgets." in out


# Verify that the read_page tool schema has `find` and passes it to every URL read.
def test_mcp_server_passes_find_through(monkeypatch):
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "mcp_server_under_test", Path(__file__).resolve().parents[1] / "mcp-server.py"
    )
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)

    calls: list[tuple[str, str]] = []

    async def fake_read(url, find=""):
        calls.append((url, find))
        return f"text for {url}"

    monkeypatch.setattr(server, "run_read_page", fake_read)
    monkeypatch.setattr(server, "write_search_io_event", lambda event: None)
    schema = next(tool for tool in server.TOOLS if tool["id"] == "read_page")["parameters"]
    assert "find" in schema["properties"] and schema["required"] == ["url"]

    payload = asyncio.run(server._call_read_page({"url": ["https://a.test/x", "https://b.test/y"], "find": " bm25 "}))
    assert calls == [("https://a.test/x", "bm25"), ("https://b.test/y", "bm25")]
    assert payload["find"] == "bm25"

    calls.clear()
    asyncio.run(server._call_read_page({"url": "https://a.test/x"}))
    assert calls == [("https://a.test/x", "")]



# Build a Wikipedia-shaped page: the given body blocks followed by a notes section with a reference list.
def _wiki_page(body: list[str], notes: int = 4) -> str:
    refs = "\n".join(
        f"{n}. **[↑](https://xx.wikipedia.org#cite_ref-{n})** Author, A. Population studies {n}. ISBN 978-0-00-00000{n}-0."
        for n in range(1, notes + 1)
    )
    return "\n\n".join(["# Country", "**URL:** https://xx.wikipedia.org/wiki/Country", "---", *body, "## Notes", refs])


# Verify that footnote markers are stripped and decodes link targets.
def test_find_strips_footnote_markers_and_decodes_link_targets():
    page = _wiki_page([
        "## Geography",
        r"The summit is 5642,7 <sup>[\[2\]](https://ru.wikipedia.org#cite_note-2)</sup> m high and the "
        "[Эльбрус](https://ru.wikipedia.org/wiki/%D0%AD%D0%BB%D1%8C%D0%B1%D1%80%D1%83%D1%81_(%D0%B3%D0%BE%D1%80%D0%B0)) "
        r"volcano is dormant.[\[3\]](https://ru.wikipedia.org#cite_note-x-3) A [space](https://a.test/a%20b) stays.",
    ], notes=0)
    out = find_in_page(page, "summit")
    assert "cite_note" not in out and "5642,7 m high" in out
    assert "(https://ru.wikipedia.org/wiki/Эльбрус_(гора))" in out
    assert "dormant. A" in out and "a%20b" in out


# Verify that a matched group-label table row brings the rows it labels.
def test_find_group_label_row_brings_its_values():
    rows = "\n".join(f"| Field {i} | value {i} |" for i in range(200))
    table = "\n".join([
        "| Country |  |", "| --- | --- |", rows,
        "| [Population](https://xx.wikipedia.org/wiki/Demographics) |  |",
        "| • 2025 census | 86.092.168 |", "| • Density | 112/km2 |",
        "| GDP (PPP) | 2026 estimate |", "| • Total | $4 trillion |",
    ])
    out = find_in_page(_wiki_page([table, "## History", "Founded long ago."]), "population")
    assert "86.092.168" in out and "112/km2" in out
    assert "$4 trillion" not in out and "Field 3" not in out


# Verify that a hit in a table's column header shows the first rows.
def test_find_column_header_hit_shows_first_rows():
    body = "\n".join(f"| City {i} | {i * 1000} |" for i in range(200))
    table = f"| City | Population |\n| --- | --- |\n{body}"
    out = find_in_page(_wiki_page(["## Cities", table]), "population")
    assert "| City 0 | 0 |" in out and "| City 4 | 4000 |" in out and "City 30" not in out


# Verify that back matter (notes, references) ranks after the article.
def test_find_ranks_back_matter_after_the_article():
    body = ["## Demographics", "The population grew to 86 million people according to the latest census."]
    out = find_in_page(_wiki_page(body, notes=6), "population")
    assert "86 million" in out
    assert "Population studies" not in out and "more passage" in out


# Verify that trimmed Chinese sentences are joined without spaces.
def test_find_joins_trimmed_cjk_sentences_without_spaces():
    filler = "这是一个很长的句子，用来让段落超过长度限制。" * 60
    page = f"# 地球\n\n---\n\n## 大气\n\n{filler}地球的大气层没有明确边界。离地表越远空气越稀薄。{filler}"
    out = find_in_page(page, "大气层")
    assert "限制。地球的大气层没有明确边界。离地表越远空气越稀薄。" in out
