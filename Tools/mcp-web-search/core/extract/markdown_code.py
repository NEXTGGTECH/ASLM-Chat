# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import hashlib
import re
from copy import copy


# Language class names such as language-python or lang-c++ (group 1 is the language); a fence
# info string that contains only safe characters; and a Markdown code fence opener of three or
# more backticks or tildes, indented by at most three spaces or tabs (group 1 is the marker).
_LANGUAGE_RE = re.compile(r"^(?:language|lang)-([A-Za-z0-9_.+-]+)$", re.IGNORECASE)
_SAFE_LANGUAGE_RE = re.compile(r"^[A-Za-z0-9_.+-]+$")
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


# Return the Markdown fence marker (the run of backticks or tildes) that opens the line, or an
# empty string when the line is not a fence.
def markdown_fence_marker(line: str) -> str:
    match = _FENCE_RE.match(line or "")
    return match.group(1) if match else ""


# Return True only for a valid closing fence: the same character, at least as long as the
# opening marker, with no info string after the delimiter.
def closes_markdown_fence(line: str, marker: str) -> bool:
    stripped = (line or "").strip()
    if not marker or not stripped.startswith(marker[0] * len(marker)):
        return False
    return not stripped[len(marker):].strip(marker[0]).strip()


# Extract preformatted text from a <pre> node without flattening indentation or React-style
# div.cm-line rows. The node is copied first, so the original DOM stays intact.
def pre_to_text(pre) -> str:
    from bs4 import NavigableString

    clone = copy(pre)

    # Turn <br> into newlines and drop whitespace-only text between block divs, so editor markup
    # that wraps every code line in a div does not produce extra blank lines.
    for br in clone.find_all("br"):
        br.replace_with("\n")
    for container in (clone, *clone.find_all("code")):
        if container.find("div", recursive=False):
            for child in list(container.children):
                if isinstance(child, NavigableString) and not str(child).strip():
                    child.extract()

    # End every div with a newline (innermost first) so each div becomes one line of code.
    for div in reversed(clone.find_all("div")):
        if not div.get_text("", strip=False).endswith("\n"):
            div.append("\n")

    # Normalize line endings and trim blank lines at both ends, keeping inner indentation.
    lines = clone.get_text("", strip=False).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


# Infer the fence info string (language) of a <pre> from structural attributes only: data-language,
# data-lang, language-*/lang-* classes on <code> or <pre>, and the sp-* class of Sandpack. Return an
# empty string when none is found or the value contains unsafe characters.
def _language(pre) -> str:
    code = pre.find("code")
    nodes = (code, pre) if code is not None else (pre,)
    for node in nodes:
        explicit = node.get("data-language") or node.get("data-lang")
        if explicit and _SAFE_LANGUAGE_RE.fullmatch(str(explicit)):
            return str(explicit)
        for token in node.get("class") or ():
            match = _LANGUAGE_RE.fullmatch(str(token))
            if match:
                return match.group(1)

    # Sandpack/CodeMirror markup used by React docs exposes the language as the last
    # ``sp-*`` class on <pre>; choosing the last structural token avoids host rules.
    sandpack = [str(token)[3:] for token in pre.get("class") or () if str(token).startswith("sp-")]
    if sandpack and _SAFE_LANGUAGE_RE.fullmatch(sandpack[-1]):
        return sandpack[-1]
    return ""


# Wrap code in a fence longer than every backtick run contained in the code itself.
def fenced_code(text: str, language: str = "") -> str:
    longest = max((len(run) for run in re.findall(r"`+", text or "")), default=0)
    fence = "`" * max(3, longest + 1)
    info = language if _SAFE_LANGUAGE_RE.fullmatch(language or "") else ""
    return f"{fence}{info}\n{text}\n{fence}"


# Protect <pre> code blocks while generic extractors process the surrounding prose: put a start
# and a stop marker paragraph around every non-empty <pre> and return (start, stop, fenced
# markdown) triples for restore_pre_markers. The <pre> itself stays in the DOM. The markers carry
# a fingerprint of the whole document, so they cannot collide with page text.
def wrap_pre_with_markers(soup) -> list[tuple[str, str, str]]:
    source_fingerprint = hashlib.sha256(str(soup).encode("utf-8", errors="replace")).hexdigest()[:24].upper()
    markers: list[tuple[str, str, str]] = []
    for index, pre in enumerate(list(soup.find_all("pre"))):
        text = pre_to_text(pre)
        language = _language(pre)
        if not text:
            continue

        # Add the marker paragraphs before and after the <pre> and remember its fenced markdown.
        start = f"ASLMCODESTART{source_fingerprint}{index:04d}END"
        stop = f"ASLMCODESTOP{source_fingerprint}{index:04d}END"
        start_node = soup.new_tag("p")
        stop_node = soup.new_tag("p")
        start_node.string = start
        stop_node.string = stop
        pre.insert_before(start_node)
        pre.insert_after(stop_node)
        markers.append((start, stop, fenced_code(text, language)))
    return markers


# Replace each extracted START..STOP range (whatever text the extractor made of the code between
# the markers) with the exact fenced Markdown recorded by wrap_pre_with_markers.
def restore_pre_markers(text: str, markers: list[tuple[str, str, str]]) -> str:
    restored = text or ""
    for start, stop, markdown in markers:
        start_position = restored.find(start)
        stop_position = restored.find(stop, start_position + len(start)) if start_position >= 0 else -1
        if start_position >= 0 and stop_position >= 0:
            before = restored[:start_position]
            after = restored[stop_position + len(stop):]
            replacement = markdown
            if before and not before.endswith("\n"):
                replacement = "\n" + replacement
            if after and not after.startswith("\n"):
                replacement += "\n"
            restored = before + replacement + after
        else:
            # Never expose internal markers when an extractor keeps only one boundary.
            restored = restored.replace(start, "").replace(stop, "")
    return restored


# Collapse excessive prose whitespace (at most two consecutive blank lines) without modifying
# blank lines inside code fences.
def collapse_blank_lines_preserving_fences(markdown: str) -> str:
    lines = (markdown or "").split("\n")
    output: list[str] = []
    marker = ""
    blank_run = 0
    for line in lines:
        current = markdown_fence_marker(line)

        # Inside a fence: copy the line unchanged; a valid closing fence ends the protected region.
        if marker:
            if closes_markdown_fence(line, marker):
                marker = ""
            output.append(line)
            blank_run = 0
            continue

        # A fence opener starts a protected region.
        if current:
            marker = current
            output.append(line)
            blank_run = 0
            continue

        # Prose: keep non-blank lines and at most two blank lines in a row.
        if line.strip():
            output.append(line)
            blank_run = 0
        else:
            blank_run += 1
            if blank_run <= 2:
                output.append(line)
    return "\n".join(output)
