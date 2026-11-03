# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from core.extract.content_processor import extract_full_body_text
from core.extract.markdown_code import restore_pre_markers
from core.extract.page_normalizer import normalize_page


# Paragraph long enough to keep the synthetic articles on the primary formatted extraction path,
# so code preservation is tested independently of the fallbacks.
_PROSE = (
    "This substantial explanatory paragraph keeps the synthetic article on the primary "
    "formatted extraction path while code preservation is verified independently."
)


# Verify that a pre/code block with a language class is rendered as a fenced block that keeps its
# indentation and language, with no internal marker left behind.
def test_normalizer_preserves_pre_indentation_and_language():
    html = f"""<html><head><title>Code fixture</title></head><body><article>
    <p>{_PROSE}</p><p>{_PROSE}</p>
    <pre><code class="language-python">async def main():
    await task()
    return 42</code></pre>
    <p>{_PROSE}</p>
    </article></body></html>"""

    markdown = normalize_page("https://fixture.test/code", html)

    assert "```python\nasync def main():\n    await task()\n    return 42\n```" in markdown
    assert "ASLMCODE" not in markdown


# Verify that a code block rendered as one div per line (React sandbox markup) becomes a fenced
# block with the language from the class and one source line per div.
def test_normalizer_handles_react_div_line_markup():
    html = f"""<html><head><title>React code fixture</title></head><body><article>
    <p>{_PROSE}</p><p>{_PROSE}</p>
    <pre class="sp-cm sp-pristine sp-javascript"><code>
      <div class="cm-line"><span>function App() {{</span><br></div>
      <div class="cm-line"><span>  return value;</span><br></div>
      <div class="cm-line"><span>}}</span></div>
    </code></pre>
    <p>{_PROSE}</p>
    </article></body></html>"""

    markdown = normalize_page("https://fixture.test/react-code", html)

    assert "```javascript\nfunction App() {\n  return value;\n}\n```" in markdown


# Verify that code containing a ``` line gets a longer (four-backtick) fence, keeps its blank
# lines and that table-shaped lines inside it are not turned into a table.
def test_dynamic_fence_and_blank_lines_survive_postprocessing():
    html = f"""<html><head><title>Fence fixture</title></head><body><article>
    <p>{_PROSE}</p><p>{_PROSE}</p>
    <pre><code>alpha


```
left | right
--- | ---
omega</code></pre>
    <p>{_PROSE}</p>
    </article></body></html>"""

    markdown = normalize_page("https://fixture.test/fence", html)

    assert "````\nalpha\n\n\n```\nleft | right\n--- | ---\nomega\n````" in markdown
    assert "| left | right |" not in markdown


# Verify that the full-body fallback keeps a pre/code block as a fenced block with its language,
# between the surrounding header and footer text.
def test_full_body_fallback_restores_pre_in_place():
    html = """<html><body><header>Download</header>
    <pre><code class="language-shell">if ready; then
  run_command
fi</code></pre>
    <footer>Checksums</footer></body></html>"""

    text = extract_full_body_text(html)

    assert "Download" in text
    assert "```shell\nif ready; then\n  run_command\nfi\n```" in text
    assert "Checksums" in text


# Verify that restore_pre_markers replaces everything between the start and stop markers, even a
# code block the extractor generated itself, with the stored fenced code.
def test_marker_boundaries_replace_extractor_generated_code_range():
    start = "ASLMCODESTARTFIXTURE0000END"
    stop = "ASLMCODESTOPFIXTURE0000END"
    extracted = f"{start}\n```\nflattened code\n```\n{stop}"
    inner = "```javascript\nfunction App() {\n  return value;\n}\n```"

    restored = restore_pre_markers(extracted, [(start, stop, inner)])

    assert restored == inner
    assert restored.count("```") == 2


# Verify that the DOM block fallback extraction also keeps a pre/code block as a fenced block with
# its language and indentation, with no internal marker left behind.
def test_dom_fallback_preserves_preformatted_code():
    from core.extract.page_normalizer import _extract_with_dom_blocks

    html = f"""<html><body><article>
    <p>{_PROSE}</p>
    <pre><code class="language-python">async def fallback():
    await task()</code></pre>
    <p>{_PROSE}</p>
    </article></body></html>"""

    text, _stats = _extract_with_dom_blocks(html, "https://fixture.test/dom-fallback")

    assert "```python\nasync def fallback():\n    await task()\n```" in text
    assert "ASLMCODE" not in text
