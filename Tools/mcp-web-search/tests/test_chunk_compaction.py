# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from core.extract.chunk_compaction import compress_chunks


# Build a 40-paragraph document where every fourth paragraph is about transformer attention and
# the rest is unrelated filler about cooking and travel.
def _doc() -> str:
    paras = []
    for i in range(40):
        if i % 4 == 0:
            paras.append(
                f"Transformer attention scales with sequence length via softmax over "
                f"query-key dot products; multi-head attention section {i}."
            )
        else:
            paras.append(f"Unrelated filler paragraph {i} about cooking and travel, low relevance.")
    return "\n\n".join(paras)


# Verify that compression shortens the document (not a passthrough), keeps the relevant chunks and
# does not keep more filler than relevant text.
def test_compress_selects_relevant_and_compacts():
    doc = _doc()
    out = compress_chunks(doc, "transformer attention mechanism", char_budget=1_400)

    # The output is compacted, not a passthrough.
    assert 0 < len(out) < len(doc)

    # The relevant chunks survive.
    assert "attention" in out.lower()
    assert out.lower().count("attention") >= out.lower().count("cooking")


# Verify that a smaller character budget never yields a longer result than a larger one.
def test_compress_respects_budget_scaling():
    doc = _doc()
    small = compress_chunks(doc, "transformer attention mechanism", char_budget=600)
    large = compress_chunks(doc, "transformer attention mechanism", char_budget=3_000)
    assert len(small) <= len(large)


# Verify that a keyword-stuffed SEO block is rejected while the genuine explanation of the same
# topic is kept.
def test_compress_rejects_seo_stuffed_block():
    spam = "buy attention attention attention cheap attention deals attention shop attention now " * 3
    real = (
        "The transformer uses scaled dot-product attention to weight tokens by relevance, "
        "computed as softmax of query-key products divided by the square root of the dimension."
    )
    out = compress_chunks(f"{spam}\n\n{real}", "transformer attention", char_budget=1_400)

    # The genuine content is kept.
    assert "softmax" in out

    # The SEO-stuffed block is rejected.
    assert "cheap attention deals" not in out


# Verify that compression with an empty query returns a string instead of raising.
def test_compress_empty_query_does_not_crash():
    doc = _doc()
    out = compress_chunks(doc, "", char_budget=1_400)
    assert isinstance(out, str)


# Verify that a selected fenced code block is kept whole, with its opening and closing fence.
def test_compress_chunks_keeps_selected_fence_atomic():
    code = (
        "```python\n"
        "# Transformer attention query computes 42 token weights.\n"
        "async def attention_query():\n"
        "    return transformer_attention_result\n"
        "```"
    )
    filler = "Unrelated cooking and travel paragraph with enough words to be scored."
    doc = "\n\n".join([filler] * 8 + [code] + [filler] * 8)

    out = compress_chunks(doc, "transformer attention query", char_budget=600)

    assert code in out
    assert out.count("```") == 2


# Verify that a code block unrelated to the query is still kept verbatim even when it makes the
# output longer than the character budget.
def test_compress_chunks_pins_irrelevant_code_verbatim_over_budget():
    code = (
        "```python\n"
        + "\n".join(f"preserved_value_{i} = {i}" for i in range(80))
        + "\n```"
    )
    relevant = (
        "Transformer attention uses query-key products to compute token relevance."
    )

    out = compress_chunks(f"{relevant}\n\n{code}", "transformer attention", char_budget=300)

    assert code in out
    assert len(out) > 300
