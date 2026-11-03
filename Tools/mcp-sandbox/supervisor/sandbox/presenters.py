# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import re

from sandbox.config import MAX_FILE_MAP_SYMBOLS


# Matches a declaration line (class, def, fn, struct, ...) with up to four leading spaces and optional
# pub/export/async/static/abstract modifiers. Groups: indentation, declaration keyword, name.
_STRUCTURE_RE = re.compile(
    r"^(\s{0,4})"
    r"(?:(?:pub(?:\(crate\))?\s+)?)"
    r"(?:(?:export\s+(?:default\s+)?)?)"
    r"(?:async\s+)?"
    r"(?:static\s+)?"
    r"(?:abstract\s+)?"
    r"(class|def|function|func|fn|struct|enum|trait|impl|"
    r"interface|module|type|namespace|mod)"
    r"\s+(\w+)"
)

# Matches a markdown heading. Groups: the hashes (heading level) and the heading text.
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# File extensions previewed as source code.
_CODE_EXTENSIONS = frozenset({
    ".c", ".cc", ".cpp", ".cs", ".go", ".java", ".js", ".jsx", ".kt",
    ".mjs", ".php", ".py", ".rs", ".scala", ".swift", ".ts", ".tsx",
})

# File extensions previewed as prose or configuration, even if their head looks like code.
_TEXT_EXTENSIONS = frozenset({
    ".cfg", ".conf", ".csv", ".css", ".html", ".ini", ".json", ".less",
    ".log", ".markdown", ".md", ".rst", ".scss", ".sql", ".toml", ".tsv",
    ".txt", ".xml", ".yaml", ".yml",
})


# Format byte count as B, KB, or MB for display.
def _human_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


# Landmark extraction

# Scan head lines for symbols (class, def, fn, etc.) indented at most four spaces; at most
# MAX_FILE_MAP_SYMBOLS are returned, each with its 1-based line, indentation, kind and name.
def _extract_code_structure(lines: list[str]) -> list[dict[str, object]]:
    symbols: list[dict[str, object]] = []
    for index, line in enumerate(lines, 1):
        match = _STRUCTURE_RE.match(line)
        if not match:
            continue
        symbols.append(
            {
                "line": index,
                "indent": len(match.group(1)),
                "kind": match.group(2),
                "name": match.group(3),
            }
        )
        if len(symbols) >= MAX_FILE_MAP_SYMBOLS:
            break
    return symbols


# Collect markdown heading landmarks from head lines.
def _extract_markdown_headings(lines: list[str]) -> list[dict[str, object]]:
    headings: list[dict[str, object]] = []
    for index, line in enumerate(lines, 1):
        match = _HEADING_RE.match(line)
        if match:
            headings.append(
                {
                    "line": index,
                    "level": len(match.group(1)),
                    "text": match.group(2).strip(),
                }
            )
    return headings


# Sample evenly spaced non-empty lines as landmarks for plain text files: always the first and last
# line, and up to max_landmarks entries in total, with each text cut to 80 characters.
def _extract_text_landmarks(lines: list[str], max_landmarks: int = 12) -> list[dict[str, object]]:
    if not lines:
        return []
    landmarks: list[dict[str, object]] = [{"line": 1, "text": lines[0].rstrip()[:80]}]
    if len(lines) > 2:
        step = max(1, len(lines) // max_landmarks)
        for index in range(step, len(lines) - 1, step):
            text = lines[index].rstrip()[:80]
            if text.strip():
                landmarks.append({"line": index + 1, "text": text})
            if len(landmarks) >= max_landmarks - 1:
                break
    if len(lines) > 1:
        landmarks.append({"line": len(lines), "text": lines[-1].rstrip()[:80]})
    return landmarks


# Previews

# Structured preview for source files: symbol map, head/tail slices, next-step hints.
def _present_code_preview(
    path: str,
    head_lines: list[str],
    total_lines: int,
    size_bytes: int,
    tail_lines: list[str] | None,
    tail_start_line: int,
) -> str:
    # Header line, then the symbol map found in the head lines.
    parts = [f"-- {path} ({total_lines} lines, {_human_size(size_bytes)}) --"]
    symbols = _extract_code_structure(head_lines)
    if symbols:
        parts.append("\n[structure]")
        for symbol in symbols:
            pad = " " * (int(symbol["indent"]) // 2)
            parts.append(f"  L{symbol['line']:<6} {pad}{symbol['kind']} {symbol['name']}")

    # The first 30 lines, and the tail when one was read.
    head_end = min(30, len(head_lines), total_lines)
    if head_end:
        parts.append(f"\n[head: 1-{head_end}]")
        parts.append("\n".join(head_lines[:head_end]))

    if tail_lines and tail_start_line:
        parts.append(f"\n[tail: {tail_start_line}-{total_lines}]")
        parts.append("\n".join(tail_lines))

    # Suggest follow-up reads: the first three symbols, or the middle of the file when there are none.
    parts.append("\n[next]")
    if symbols:
        for symbol in symbols[:3]:
            start = int(symbol["line"])
            parts.append(f"  cat {path} | sed -n '{start},{min(start + 60, total_lines)}p'")
    else:
        mid = max(1, total_lines // 2)
        parts.append(f"  cat {path} | sed -n '{mid},{min(mid + 80, total_lines)}p'")
    return "\n".join(parts)


# Structured preview for prose/config: landmarks, head/tail slices, next-step hints.
def _present_text_preview(
    path: str,
    head_lines: list[str],
    total_lines: int,
    size_bytes: int,
    tail_lines: list[str] | None,
    tail_start_line: int,
) -> str:
    # Header line, then landmarks: headings for markup files, sampled lines for other text.
    parts = [f"-- {path} ({total_lines} lines, {_human_size(size_bytes)}) --"]
    ext = os.path.splitext(path)[1].lower()
    landmarks = (
        _extract_markdown_headings(head_lines)
        if ext in {".md", ".markdown", ".rst"}
        else _extract_text_landmarks(head_lines)
    )
    if landmarks:
        parts.append("\n[landmarks]")
        for landmark in landmarks[:12]:
            text = str(landmark.get("text", ""))
            parts.append(f"  L{landmark['line']:<6} {text}")

    # The first 30 lines, and the tail when one was read.
    head_end = min(30, len(head_lines), total_lines)
    if head_end:
        parts.append(f"\n[head: 1-{head_end}]")
        parts.append("\n".join(head_lines[:head_end]))

    if tail_lines and tail_start_line:
        parts.append(f"\n[tail: {tail_start_line}-{total_lines}]")
        parts.append("\n".join(tail_lines))

    # Suggest a follow-up read around the middle of the file.
    parts.append("\n[next]")
    mid = max(1, total_lines // 2)
    parts.append(f"  cat {path} | sed -n '{mid},{min(mid + 80, total_lines)}p'")
    return "\n".join(parts)


# Pick code vs text preview based on extension and detected structure in head lines. Non-text kinds and
# empty files get a one-line description instead.
def present_auto_preview(
    path: str,
    head_lines: list[str],
    total_lines: int,
    size_bytes: int,
    mime: str,
    kind: str = "text",
    tail_lines: list[str] | None = None,
    tail_start_line: int = 0,
) -> str:
    if kind != "text" or not head_lines:
        return f"[{kind} file: {mime}, {_human_size(size_bytes)}]"

    ext = os.path.splitext(path)[1].lower()
    if ext in _CODE_EXTENSIONS or (
        ext not in _TEXT_EXTENSIONS and _extract_code_structure(head_lines[:120])
    ):
        return _present_code_preview(
            path, head_lines, total_lines, size_bytes, tail_lines, tail_start_line
        )
    return _present_text_preview(
        path, head_lines, total_lines, size_bytes, tail_lines, tail_start_line
    )


# Controller responses

# Format a bounded read slice for legacy controller OPEN responses.
def present_read_slice(
    *,
    path: str,
    content: str,
    start_line: int | None,
    end_line: int | None,
    total_lines: int,
    size_bytes: int,
) -> str:
    header = f"-- {path} ({total_lines} lines, {_human_size(size_bytes)}) --"
    if start_line is not None and end_line is not None:
        header += f"\n[lines {start_line}-{end_line}]"
    body = content.rstrip("\n")
    return f"{header}\n{body}" if body else header


# Format grep matches for legacy controller LOCATE responses, listing at most the first 50 matches
# as path:line:text and counting the rest.
def present_grep_results(
    *,
    matches: list[dict[str, object]],
    pattern: str,
    path: str,
) -> str:
    lines = [f"-- grep {pattern!r} in {path} ({len(matches)} matches) --"]
    for match in matches[:50]:
        rel = str(match.get("path", "?"))
        line_no = match.get("line_number", "?")
        text = str(match.get("line", "")).rstrip()
        lines.append(f"{rel}:{line_no}:{text}")
    if len(matches) > 50:
        lines.append(f"... ({len(matches) - 50} more)")
    return "\n".join(lines)
