# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio

import custom_domains
import custom_domains.arxiv as arxiv
import core.fetch.arxiv_api as arxiv_api
from custom_domains.base import SCOPE_READ_PAGE


# Atom feed of the arXiv API with one entry carrying every metadata field the parser maps.
ATOM_SAMPLE = """\
<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2401.04088v2</id>
    <updated>2024-01-10T12:00:00Z</updated>
    <published>2024-01-08T12:00:00Z</published>
    <title>  Mixtral of Experts  </title>
    <summary>We introduce a sparse mixture of experts model.</summary>
    <author><name>Albert Q. Jiang</name><arxiv:affiliation>Mistral AI</arxiv:affiliation></author>
    <author><name>Alexandre Sablayrolles</name></author>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:primary_category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
    <arxiv:comment>v2: corrected benchmarks</arxiv:comment>
    <arxiv:journal_ref>Example Journal 1 (2024)</arxiv:journal_ref>
    <arxiv:doi>10.1000/example</arxiv:doi>
    <arxiv:license href="http://creativecommons.org/licenses/by/4.0/"/>
    <link href="http://arxiv.org/abs/2401.04088v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2401.04088v2" rel="related" type="application/pdf"/>
    <link title="doi" href="http://dx.doi.org/10.1000/example" rel="related"/>
  </entry>
</feed>
"""


# arXiv HTML page reduced to its bibliography: one reference with a DOI link and one with a
# relative arXiv link.
HTML_SAMPLE = """\
<!doctype html><html><body>
<section class="ltx_bibliography">
  <h2>References</h2>
  <ul class="ltx_biblist">
    <li id="bib.bib1" class="ltx_bibitem">
      <span class="ltx_tag ltx_tag_bibitem">[1]</span>
      A. Author. First paper. <a href="https://doi.org/10.1000/first">doi</a>
    </li>
    <li id="bib.bib2" class="ltx_bibitem">
      <span class="ltx_tag ltx_tag_bibitem">[undefa]</span>
      B. Author. <a href="/abs/2301.00001">Second paper</a>.
    </li>
  </ul>
</section>
</body></html>
"""


# Verify that arxiv_id_from_url extracts new-style, versioned and old-style ids from abstract URLs
# only, and returns an empty string for PDF, HTML and non-arXiv URLs.
def test_arxiv_id_from_url_accepts_abstract_ids_only():
    assert arxiv.arxiv_id_from_url("https://arxiv.org/abs/2401.04088") == "2401.04088"
    assert arxiv.arxiv_id_from_url("https://www.arxiv.org/abs/2401.04088v2") == "2401.04088v2"
    assert arxiv.arxiv_id_from_url("https://arxiv.org/abs/hep-th/9901001v3") == "hep-th/9901001v3"
    assert arxiv.arxiv_id_from_url("https://arxiv.org/pdf/2401.04088") == ""
    assert arxiv.arxiv_id_from_url("https://arxiv.org/html/2401.04088") == ""
    assert arxiv.arxiv_id_from_url("https://example.com/abs/2401.04088") == ""


# Verify that the Atom parser returns the full record: versioned id, trimmed title, authors with
# affiliations, categories, links, DOI, journal reference, comment and license.
def test_parse_arxiv_atom_returns_rich_metadata():
    record = arxiv.parse_arxiv_atom(ATOM_SAMPLE, requested_id="2401.04088")

    assert record is not None
    assert record.arxiv_id == "2401.04088v2"
    assert record.title == "Mixtral of Experts"
    assert record.abstract.startswith("We introduce")
    assert record.authors[0].name == "Albert Q. Jiang"
    assert record.authors[0].affiliations == ["Mistral AI"]
    assert record.categories == ["cs.LG", "cs.CL"]
    assert record.primary_category == "cs.LG"
    assert record.pdf_url == "https://arxiv.org/pdf/2401.04088v2"
    assert record.doi == "10.1000/example"
    assert record.doi_url == "https://dx.doi.org/10.1000/example"
    assert record.journal_ref == "Example Journal 1 (2024)"
    assert record.comment == "v2: corrected benchmarks"
    assert record.license_url == "https://creativecommons.org/licenses/by/4.0/"


# Verify that the Atom parser returns None for an error feed (entry titled "Error") and for
# input that is not XML.
def test_parse_arxiv_atom_rejects_error_feed():
    error = ATOM_SAMPLE.replace("<title>  Mixtral of Experts  </title>", "<title>Error</title>")
    assert arxiv.parse_arxiv_atom(error, requested_id="bad") is None
    assert arxiv.parse_arxiv_atom("not xml", requested_id="bad") is None


# Verify that the HTML reference parser returns the total count and formats each reference as
# numbered markdown, resolving relative links against the base URL.
def test_parse_arxiv_html_references_preserves_links():
    refs, total = arxiv.parse_arxiv_html_references(
        HTML_SAMPLE,
        base_url="https://arxiv.org/html/2401.04088",
    )

    assert total == 2
    assert refs[0] == "[1] A. Author. First paper. [doi](https://doi.org/10.1000/first)"
    assert refs[1] == "[2] B. Author. [Second paper](https://arxiv.org/abs/2301.00001) ."


# Verify that the fetched arXiv page markdown combines the API metadata (PDF, DOI, journal
# reference, version, authors with affiliations) with the HTML reference list.
def test_fetch_arxiv_page_combines_api_metadata_and_references(monkeypatch):
    record = arxiv.parse_arxiv_atom(ATOM_SAMPLE, requested_id="2401.04088")

    # Return the parsed sample record for the requested id.
    async def fake_atom(arxiv_id: str, timeout: float):
        assert arxiv_id == "2401.04088"
        return record

    # Return one linked reference and the HTML page URL.
    async def fake_refs(arxiv_id: str, timeout: float):
        return ["[1] Linked reference"], 1, f"https://arxiv.org/html/{arxiv_id}"

    monkeypatch.setattr(arxiv, "_fetch_atom", fake_atom)
    monkeypatch.setattr(arxiv, "_fetch_html_references", fake_refs)

    markdown = asyncio.run(arxiv.fetch_arxiv_page("https://arxiv.org/abs/2401.04088"))

    assert "**PDF:** https://arxiv.org/pdf/2401.04088v2" in markdown
    assert "**DOI:** [10.1000/example](https://dx.doi.org/10.1000/example)" in markdown
    assert "**Journal reference:** Example Journal 1 (2024)" in markdown
    assert "**Version:** v2" in markdown
    assert "Albert Q. Jiang (Mistral AI)" in markdown
    assert "## References" in markdown
    assert "- [1] Linked reference" in markdown


# Verify that the fetched page explains that the metadata API has no bibliography and links the PDF
# when the HTML page yields no references.
def test_fetch_arxiv_page_explains_reference_fallback(monkeypatch):
    record = arxiv.parse_arxiv_atom(ATOM_SAMPLE, requested_id="2401.04088")

    # Return the parsed sample record.
    async def fake_atom(_arxiv_id: str, _timeout: float):
        return record

    # Return no references, as when the HTML version is unavailable.
    async def fake_refs(arxiv_id: str, _timeout: float):
        return [], 0, f"https://arxiv.org/html/{arxiv_id}"

    monkeypatch.setattr(arxiv, "_fetch_atom", fake_atom)
    monkeypatch.setattr(arxiv, "_fetch_html_references", fake_refs)

    markdown = asyncio.run(arxiv.fetch_arxiv_page("https://arxiv.org/abs/2401.04088"))

    assert "metadata API does not include the bibliography" in markdown
    assert "[PDF](https://arxiv.org/pdf/2401.04088v2)" in markdown


# Verify that the arXiv handler is registered for abstract URLs with the read-page scope only and
# that PDF URLs are left to the PDF reader.
def test_arxiv_handler_is_registered_read_page_only_and_leaves_pdf_to_pdf_reader():
    handler = custom_domains.match("https://arxiv.org/abs/2401.04088")

    assert handler is arxiv.HANDLER
    assert handler.scope == SCOPE_READ_PAGE
    assert custom_domains.is_read_page_only("https://arxiv.org/abs/2401.04088") is True
    assert custom_domains.match("https://arxiv.org/pdf/2401.04088") is None


# Verify that the arXiv API gate waits only for the remaining part of the minimum spacing between
# two calls (2 s minus the 1 s that already passed).
def test_arxiv_api_gate_serializes_and_spaces_calls(monkeypatch):
    # A fake clock that only moves when the test or the fake sleep advances it.
    clock = [100.0]
    sleeps: list[float] = []

    # Return the fake clock value.
    def fake_monotonic():
        return clock[0]

    # Record the requested delay and advance the fake clock instead of sleeping.
    async def fake_sleep(delay: float):
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setattr(arxiv_api, "_LOCK", asyncio.Lock())
    monkeypatch.setattr(arxiv_api, "_last_started", 0.0)
    monkeypatch.setattr(arxiv_api.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(arxiv_api.asyncio, "sleep", fake_sleep)

    # Enter the gate twice, one fake second apart.
    async def exercise():
        async with arxiv_api.arxiv_api_slot():
            pass
        clock[0] += 1.0
        async with arxiv_api.arxiv_api_slot():
            pass

    asyncio.run(exercise())

    assert sleeps == [2.0]
