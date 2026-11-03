# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import logging
from typing import Any, Callable

from sandbox.config import (
    MAX_CAT_FILE_BYTES,
    MAX_CAT_LINE_THRESHOLD,
    MAX_FIND_RESULTS,
    MAX_GREP_RESULTS,
    MAX_LS_ENTRIES,
    MAX_READ_BYTES,
)
from sandbox.intent import Intent, NormalizedCommand, classify
from sandbox.presenters import (
    present_auto_preview,
    present_grep_results,
    present_read_slice,
)
from sandbox.session_state import ExplorationState
from sandbox.workspace import (
    describe,
    find,
    grep,
    ls,
    normalize_model_relative_path,
    read,
    resolve_model_path,
)

logger = logging.getLogger(__name__)


# Backend handlers

# OPEN intent: read file content with loop-breaking and preview fallbacks. Large files, long files and
# repeated reads of the same file get a structured preview instead of the raw text; a read that mostly
# overlaps earlier reads is redirected to the next unread window.
def _handle_open(
    nc: NormalizedCommand,
    state: ExplorationState,
    cwd: str = ".",
) -> dict[str, Any]:
    path = resolve_model_path(nc.target, cwd)
    norm_path = normalize_model_relative_path(path)

    # Check if we should serve an alternative representation due to loops.
    loop_broken = False
    if state.should_break_loop(norm_path, "open"):
        loop_broken = True

    # Get file metadata.
    try:
        meta = describe(path)
    except Exception as exc:
        return _not_found(str(exc))

    meta_inner = meta.get("result", {})
    size = int(meta_inner.get("size_bytes", 0))
    kind = meta_inner.get("kind", "text")

    # Binary / image — quick exit.
    if kind in ("binary", "image"):
        from sandbox.presenters import _human_size
        mime = meta_inner.get("mime", "?")
        return _ok(f"[{kind} file: {mime}, {_human_size(size)}]", norm_path)

    # Large file or read loop → structured preview instead of raw dump.
    if size > MAX_CAT_FILE_BYTES or loop_broken:
        preview = _build_preview(path, meta_inner, size)
        warnings = []
        if loop_broken:
            warnings.append(
                "Loop detected: switching to structured preview instead of raw text."
            )
        else:
            warnings.append(
                "Large file: showing structured preview with navigation aids."
            )
        record = state.record_touch
        record(norm_path, "open",
               representation="map" if loop_broken else "map")
        return _ok(preview, norm_path, warnings=warnings)

    # Normal read — with optional line range.
    start = nc.start_line
    end = nc.end_line

    # Tail support from compound pipeline normalization.
    tail_n = getattr(nc, "_tail_n", None)
    if tail_n is not None and start is None:
        full = read(path=path, max_bytes=MAX_READ_BYTES)
        total = full.get("result", {}).get("total_lines", 0)
        start = max(1, total - tail_n + 1)
        end = total

    # Read the requested slice (the whole file when no range was given) and unpack the result.
    result = read(
        path=path,
        start_line=start,
        end_line=end,
        max_bytes=MAX_READ_BYTES,
    )
    inner = result.get("result", {})
    content = inner.get("content", "")
    s_line = inner.get("start_line", 1)
    e_line = inner.get("end_line", 1)
    total_lines = inner.get("total_lines", 0)

    # Long file (small bytes, many lines) — switch to file_map preview.
    # Skip when the model explicitly requested a line range.
    if (
        start is None
        and end is None
        and total_lines > MAX_CAT_LINE_THRESHOLD
    ):
        preview = _build_preview(path, meta_inner, size)
        state.record_touch(norm_path, "open", representation="map")
        return _ok(
            preview, norm_path,
            warnings=[
                f"Long file ({total_lines} lines): showing structured preview with navigation aids."
            ],
        )

    # Overlap with prior reads → serve adjacent unread window when possible (more than 70% overlap).
    if not loop_broken and start is None:
        overlap = state.get_read_overlap(norm_path, s_line, e_line)
        if overlap > 0.7:
            unread = state.get_unread_windows(norm_path, total_lines)
            if unread:
                ns, ne = unread[0]
                result = read(path=path, start_line=ns, end_line=ne,
                              max_bytes=MAX_READ_BYTES)
                inner = result.get("result", {})
                content = inner.get("content", "")
                s_line, e_line = ns, ne
                warnings_list = [
                    f"Already read [{s_line}:{e_line}] — serving adjacent unread region."
                ]
            else:
                warnings_list = ["All regions of this file have been read."]
        else:
            warnings_list = list(result.get("warnings", []))
    else:
        warnings_list = list(result.get("warnings", []))

    # Render the slice for the model and remember it for loop and overlap detection.
    presented = present_read_slice(
        path=norm_path,
        content=content,
        start_line=s_line,
        end_line=e_line,
        total_lines=total_lines,
        size_bytes=size,
    )

    state.record_touch(norm_path, "open",
                       window=(s_line, e_line), representation="raw")
    return _ok(presented, norm_path, warnings=warnings_list)


# LOCATE intent: grep-style search via workspace.grep and presenter.
def _handle_locate(
    nc: NormalizedCommand,
    state: ExplorationState,
    cwd: str = ".",
) -> dict[str, Any]:
    if not nc.pattern:
        return _err("No search pattern provided.")

    path = resolve_model_path(nc.target, cwd)
    norm_path = normalize_model_relative_path(path)

    # Run the search with the parsed flags, capped at MAX_GREP_RESULTS matches.
    result = grep(
        pattern=nc.pattern,
        path=path,
        glob=nc.glob_pattern,
        case_sensitive=nc.case_sensitive,
        context_before=nc.context_before,
        context_after=nc.context_after,
        max_results=MAX_GREP_RESULTS,
    )

    # Format the matches and record the search for exploration tracking.
    matches = result.get("result", {}).get("matches", [])
    presented = present_grep_results(
        matches=matches,
        pattern=nc.pattern,
        path=norm_path,
    )

    state.record_search(nc.pattern, norm_path, len(matches))
    state.record_touch(norm_path, "locate")
    return _ok(presented, norm_path, warnings=result.get("warnings", []))


# SURVEY intent (ls/tree): directory listing with optional survey cache. The listing of the task root is
# cached so repeated root surveys are answered from the cache.
def _handle_survey(
    nc: NormalizedCommand,
    state: ExplorationState,
    cwd: str = ".",
) -> dict[str, Any]:
    path = resolve_model_path(nc.target, cwd)
    norm_path = normalize_model_relative_path(path)

    # Return cached survey for repeated root surveys.
    if norm_path in (".", "") and state.has_survey_cache():
        cached = state.repo_survey_cache
        return _ok(cached["output"], norm_path,
                   warnings=["Survey from cache — run `ls -a .` to force refresh."])

    result = ls(
        path=path,
        depth=nc.depth or 1,
        max_entries=MAX_LS_ENTRIES,
        include_hidden=nc.include_hidden,
    )

    # Render one indented line per entry (directories with a trailing slash, files with their size).
    entries = result.get("result", {}).get("entries", [])
    lines = []
    dirs_count = 0
    files_count = 0
    for e in entries:
        indent = "  " * e.get("depth", 1)
        name = e["name"]
        if e.get("type") == "directory":
            lines.append(f"{indent}{name}/")
            dirs_count += 1
        else:
            size = e.get("size_bytes")
            if size is not None:
                from sandbox.presenters import _human_size
                lines.append(f"{indent}{name}  ({_human_size(size)})")
            else:
                lines.append(f"{indent}{name}")
            files_count += 1

    # Add the header and totals; only the root listing is cached.
    output = "\n".join(lines) or "(empty)"
    output = f"{norm_path}/\n{output}\n\n{dirs_count} directories, {files_count} files"

    if norm_path in (".", ""):
        state.record_survey({"output": output})

    state.record_touch(norm_path, "survey")
    return _ok(output, norm_path, warnings=result.get("warnings", []))


# SURVEY intent (find/fd): find files by name/type via workspace.find.
def _handle_find(
    nc: NormalizedCommand,
    state: ExplorationState,
    cwd: str = ".",
) -> dict[str, Any]:
    path = resolve_model_path(nc.target, cwd)
    norm_path = normalize_model_relative_path(path)

    result = find(
        path=path,
        name_pattern=nc.name_pattern,
        type_filter=nc.find_type,
        max_depth=nc.depth or 8,
        max_results=MAX_FIND_RESULTS,
    )

    matches = result.get("result", {}).get("matches", [])
    output = "\n".join(m["path"] for m in matches) if matches else "(no matches)"

    state.record_touch(norm_path, "survey")
    return _ok(output, norm_path, warnings=result.get("warnings", []))


# Helpers

# Build structured head/tail preview for large or long files. The head comes from the start of the
# file; files over 45 lines also show their last 16 lines as a tail.
def _build_preview(path: str, meta: dict[str, Any], size: int) -> str:
    # Read the head of the file (limited by MAX_READ_BYTES).
    head_result = read(path=path, max_bytes=MAX_READ_BYTES)
    head_inner = head_result.get("result", {})
    content = head_inner.get("content", "")
    total_lines = head_inner.get("total_lines", 0)
    mime = head_inner.get("mime", meta.get("mime", "text/plain"))
    head_lines = content.split("\n") if content else []

    # Long files also get their last lines, read separately.
    tail_lines: list[str] = []
    tail_start_line = 0
    if total_lines > 45:
        tail_start = max(1, total_lines - 15)
        tail_result = read(
            path=path,
            start_line=tail_start,
            end_line=total_lines,
            max_bytes=MAX_READ_BYTES,
        )
        tail_content = tail_result.get("result", {}).get("content", "")
        tail_lines = tail_content.split("\n") if tail_content else []
        tail_start_line = tail_start

    # Format the preview for the model.
    norm = normalize_model_relative_path(path)
    return present_auto_preview(
        path=norm,
        head_lines=head_lines,
        total_lines=total_lines,
        size_bytes=size,
        mime=mime,
        kind="text",
        tail_lines=tail_lines or None,
        tail_start_line=tail_start_line,
    )


# Internal success shape before conversion to bash response.
def _ok(
    stdout: str,
    path: str = ".",
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "intent_routed": True,
        "stdout": stdout,
        "stderr": "",
        "warnings": warnings or [],
    }


# Internal error shape before conversion to bash response.
def _err(message: str) -> dict[str, Any]:
    return {
        "intent_routed": True,
        "stdout": "",
        "stderr": message,
        "error": message,
        "warnings": [],
    }


# Signal path errors so _try_supervise can fall back to real bash. Always raises FileNotFoundError.
def _not_found(message: str) -> None:
    raise FileNotFoundError(message)


# Main dispatch

# Command names routed to the find handler instead of the directory-listing handler.
_FIND_COMMANDS = frozenset({"find", "fd"})


# Classify command and dispatch; None means fall through to real bash. Path-resolution errors
# propagate to the caller; other failures are logged and also fall through to real bash.
def dispatch(
    command: str,
    cwd: str,
    state: ExplorationState,
    make_bash_success: Callable,
    make_bash_error: Callable,
) -> dict[str, Any] | None:
    nc = classify(command, cwd)
    if nc is None:
        return None

    # Real bash handles execution.
    if nc.intent == Intent.RUN:
        return None

    # Existing _ROUTE_MAP handlers cover mutations.
    if nc.intent == Intent.MUTATE:
        return None

    # Route to the handler of the intent.
    try:
        if nc.intent == Intent.OPEN:
            result = _handle_open(nc, state, cwd=cwd)

        elif nc.intent == Intent.LOCATE:
            result = _handle_locate(nc, state, cwd=cwd)

        elif nc.intent == Intent.SURVEY:
            # find/fd → _handle_find; ls/tree → _handle_survey
            raw_cmd = nc.raw_command.split()[0] if nc.raw_command else ""
            if raw_cmd in _FIND_COMMANDS:
                result = _handle_find(nc, state, cwd=cwd)
            else:
                result = _handle_survey(nc, state, cwd=cwd)

        else:
            return None

    except (FileNotFoundError, NotADirectoryError, ValueError):
        # Path resolution / not-found errors → fall through so _try_supervise
        # can apply _is_path_resolution_error and fall back to real bash.
        raise
    except Exception as exc:
        # Unknown errors fall through to real bash.
        logger.debug("Controller dispatch failed for %r: %s", command, exc)
        return None

    # Convert internal result to bash-shaped response.
    error = result.get("error")
    if error:
        return make_bash_error(
            error,
            stderr=result.get("stderr", ""),
            cwd=cwd,
        )

    return make_bash_success(
        result.get("stdout", ""),
        warnings=result.get("warnings"),
        cwd=cwd,
    )
