# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from sandbox.config import CONTAINER_WORKSPACE, DEFAULT_TASK_DIR, RG_TYPE_TO_GLOB

# Container task-root prefix (e.g. "/workspace/_sandbox").
_CONTAINER_TASK_PREFIX = f"{CONTAINER_WORKSPACE}/{DEFAULT_TASK_DIR}"


# What a shell command is meant to do, which decides how it is executed:
# OPEN reads file content (any form), LOCATE searches text inside files, SURVEY explores directory
# structure or finds by name, RUN executes code (build, test, install, git, network) and MUTATE changes
# the filesystem (mkdir, mv, cp, rm, touch).
class Intent(Enum):
    OPEN   = "open"
    LOCATE = "locate"
    SURVEY = "survey"
    RUN    = "run"
    MUTATE = "mutate"


# A shell command reduced to an intent plus parsed targets and options, so a handler can run the
# equivalent native operation instead of real bash.
@dataclass
class NormalizedCommand:
    intent: Intent
    raw_command: str

    # Primary target (file or directory path, unresolved).
    target: str = "."

    # Secondary targets (multi-file ops).
    extra_targets: list[str] = field(default_factory=list)

    # LOCATE: search pattern.
    pattern: str | None = None

    # OPEN: explicit line range request.
    start_line: int | None = None
    end_line: int | None = None

    # SURVEY: tree depth.
    depth: int | None = None

    # Modifiers.
    case_sensitive: bool = False
    recursive: bool = True
    include_hidden: bool = False
    glob_pattern: str | None = None

    # LOCATE: grep context lines.
    context_before: int = 0
    context_after: int = 0
    files_with_matches: bool = False

    # SURVEY/find: name pattern, type filter.
    name_pattern: str | None = None
    find_type: str | None = None

    # Set when the original was a compound pipeline that we normalized.
    was_compound: bool = False


# Intent tables

# Command names that read file content. Anything not in the intent tables below is treated as RUN and
# goes to real bash.
_OPEN_CMDS = frozenset({
    "cat", "less", "more", "tac",
    "head", "tail",
    "sed",
})

# Command names that search text inside files.
_LOCATE_CMDS = frozenset({
    "grep", "egrep", "fgrep", "rg", "ag",
})

# Command names that explore directory structure or find files by name.
_SURVEY_CMDS = frozenset({
    "ls", "ll", "la", "tree",
    "find", "fd",
    # `du` is intentionally NOT here: routing it as a directory listing
    # would silently misrepresent disk-usage output. Goes to real bash.
})

# Command names that change the filesystem.
_MUTATE_CMDS = frozenset({
    "mkdir", "touch",
    "mv", "cp", "rm",
    "chmod", "chown",
    "ln",
})

# Pipelines whose right-hand side merely transforms read output — the
# whole expression is still classified as OPEN or LOCATE.
_READ_ONLY_RHS_CMDS = frozenset({
    "head", "tail", "grep", "egrep", "rg",
    "less", "more", "cat",
    "sed",
    # Excluded: tee has side effects (writes to file), wc/cut/sort/uniq/tr/nl/awk
    # change output format — if their semantics aren't implemented, the pipeline
    # must go to real bash rather than silently returning wrong output.
})


# Helpers

# Shell-safe tokenization; fall back to whitespace split on parse errors (e.g. unbalanced quotes).
def _safe_split(s: str) -> list[str]:
    try:
        return shlex.split(s)
    except ValueError:
        return s.split()


# Return True for absolute paths outside task-space (e.g. /etc); task paths are routed.
def _is_container_path(path: str) -> bool:
    if not path.startswith("/"):
        return False

    # Inside task-space container mount → not a container path.
    if path == _CONTAINER_TASK_PREFIX or path.startswith(_CONTAINER_TASK_PREFIX + "/"):
        return False
    return True


# Pipeline parser

# Split on | only; None when &&, ||, ;, subshells, or redirections are present, or a stage is empty.
def _split_pipeline(command: str) -> list[str] | None:
    # Reject anything with complex shell constructs.
    if re.search(r"&&|\|\||;|`|\$\(|>|<", command):
        return None

    stages = [s.strip() for s in command.split("|")]
    if any(not s for s in stages):
        return None
    return stages


# Map a bare command name to Intent, or None if unknown (→ RUN).
def _classify_single_command(cmd: str) -> Intent | None:
    if cmd in _OPEN_CMDS:
        return Intent.OPEN
    if cmd in _LOCATE_CMDS:
        return Intent.LOCATE
    if cmd in _SURVEY_CMDS:
        return Intent.SURVEY
    if cmd in _MUTATE_CMDS:
        return Intent.MUTATE
    return None


# Per-intent argument parsers

# Extract target path, line range, and unsupported-flag marker from OPEN commands. The result also
# carries extra_targets for further files and tail_n for a tail request (the handler computes the
# actual range for it).
def _parse_open_args(cmd: str, args: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {"target": ".", "start_line": None, "end_line": None}
    files: list[str] = []
    i = 0

    # head/tail: only the -n forms are understood; the default is 10 lines.
    if cmd in ("head", "tail"):
        n_lines = 10
        while i < len(args):
            p = args[i]
            if p == "-n" and i + 1 < len(args):
                try:
                    n_lines = int(args[i + 1])
                except ValueError:
                    # Non-numeric count: keep the default.
                    pass
                i += 2
            elif re.match(r"^-n(\d+)$", p):
                n_lines = int(p[2:])
                i += 1
            elif re.match(r"^-(\d+)$", p):
                n_lines = int(p[1:])
                i += 1
            elif p.startswith("-"):
                # Any flag besides -n/-N is unsupported:
                # -f/--follow changes semantics entirely (streaming),
                # -c/--bytes, --pid, --retry, -q, -v, etc. are not implemented.
                result["unsupported"] = True
                i += 1
            else:
                files.append(p)
                i += 1
        if cmd == "head":
            result["start_line"] = 1
            result["end_line"] = n_lines
        else:
            # tail: use -n as a hint; actual end_line computed by handler
            result["tail_n"] = n_lines

    elif cmd == "sed":
        # Only the "-n 'N,Mp'" and "-n 'Np'" print forms are understood.
        while i < len(args):
            p = args[i]
            if p == "-n":
                i += 1
            elif re.match(r"^['\"]?\d+,\d+p['\"]?$", p):
                expr = p.strip("'\"")
                m = re.match(r"(\d+),(\d+)p", expr)
                if m:
                    result["start_line"] = int(m.group(1))
                    result["end_line"] = int(m.group(2))
                i += 1
            elif re.match(r"^['\"]?\d+p['\"]?$", p):
                expr = p.strip("'\"")
                m = re.match(r"(\d+)p", expr)
                if m:
                    result["start_line"] = int(m.group(1))
                    result["end_line"] = int(m.group(1))
                i += 1
            elif not p.startswith("-"):
                files.append(p)
                i += 1
            else:
                # Unknown flag → can't route safely
                result["unsupported"] = True
                return result

    else:
        # cat, less, more, tac — collect positional file paths.
        while i < len(args):
            p = args[i]
            if not p.startswith("-"):
                files.append(p)
            i += 1

    # The first file is the target; any others are reported as extra targets.
    if files:
        result["target"] = files[0]
        result["extra_targets"] = files[1:]
    return result


# Extract pattern, search path, glob, context, and flags from LOCATE commands. Flags the native
# search cannot reproduce set "unsupported" so the command falls back to real bash.
def _parse_locate_args(cmd: str, args: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "pattern": None,
        "target": ".",
        "case_sensitive": True,
        "glob_pattern": None,
        "context_before": 0,
        "context_after": 0,
        "files_with_matches": False,
        "unsupported": False,
    }

    _RG_TYPE_TO_GLOB = RG_TYPE_TO_GLOB

    i = 0
    positional: list[str] = []

    # Combined short-flag set we know how to interpret for grep/rg.
    _GREP_COMBINABLE = set("rRniIlLvV")

    # Walk the arguments: known flags update the result, everything else is positional
    # (the first positional is the pattern, the second the search path).
    while i < len(args):
        p = args[i]
        if p in ("--ignore-case",):
            result["case_sensitive"] = False
            i += 1
        elif p in ("--recursive",):
            i += 1
        elif (
            len(p) >= 2
            and p.startswith("-")
            and not p.startswith("--")
            and all(ch in _GREP_COMBINABLE for ch in p[1:])
        ):
            for ch in p[1:]:
                if ch in ("i", "I"):
                    result["case_sensitive"] = False
                elif ch in ("l", "L"):
                    result["files_with_matches"] = True
                elif ch in ("v", "V"):
                    result["unsupported"] = True

                # r/R/n: noop (recursive is default; line numbers always shown)
            i += 1
        elif p in ("-l", "--files-with-matches"):
            result["files_with_matches"] = True
            i += 1
        elif p in ("-n", "--line-number"):
            i += 1
        elif p == "-v":
            # Inverted match — semantics differ, fall back to real bash
            result["unsupported"] = True
            i += 1
        elif p in ("--include", "-g", "--glob") and i + 1 < len(args):
            result["glob_pattern"] = args[i + 1]
            i += 2
        elif p in ("-t", "--type") and i + 1 < len(args):
            t = args[i + 1].lower()
            result["glob_pattern"] = _RG_TYPE_TO_GLOB.get(t)
            i += 2
        elif p.startswith("--type="):
            t = p.split("=", 1)[1].lower()
            result["glob_pattern"] = _RG_TYPE_TO_GLOB.get(t)
            i += 1
        elif p in ("-C", "--context") and i + 1 < len(args):
            try:
                result["context_before"] = result["context_after"] = int(args[i + 1])
            except ValueError:
                # Non-numeric context size: keep the default of 0.
                pass
            i += 2
        elif p in ("-A", "--after-context") and i + 1 < len(args):
            try:
                result["context_after"] = int(args[i + 1])
            except ValueError:
                # Non-numeric context size: keep the default of 0.
                pass
            i += 2
        elif p in ("-B", "--before-context") and i + 1 < len(args):
            try:
                result["context_before"] = int(args[i + 1])
            except ValueError:
                # Non-numeric context size: keep the default of 0.
                pass
            i += 2
        elif p.startswith("-"):
            # Unknown flag → can't safely simulate; fall back to real bash.
            result["unsupported"] = True
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
        else:
            positional.append(p)
            i += 1

    if positional:
        result["pattern"] = positional[0]
    if len(positional) >= 2:
        result["target"] = positional[1]
    return result


# Extract path, depth, hidden/name/type flags from SURVEY (ls/tree/find) commands. Flags the native
# listing cannot reproduce set "unsupported" so the command falls back to real bash.
def _parse_survey_args(cmd: str, args: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "target": ".",
        "depth": 3 if cmd == "tree" else 1,
        "include_hidden": False,
        "name_pattern": None,
        "find_type": None,
        "unsupported": False,
    }

    is_find = cmd in ("find", "fd")

    # ls combinable short flags we understand: -a (hidden), -A (hidden no .), -l/-h/-F/-1 (display-only).
    _LS_COMBINABLE = set("aAlhF1RrSt")

    # Walk the arguments: known flags update the result, a bare word becomes the target path.
    i = 0
    while i < len(args):
        p = args[i]
        if p == "-L" and i + 1 < len(args) and not is_find:
            try:
                result["depth"] = int(args[i + 1])
            except ValueError:
                # Non-numeric depth: keep the default.
                pass
            i += 2
        elif p == "-a" or p == "-A" or p == "--all":
            result["include_hidden"] = True
            i += 1
        elif (
            not is_find
            and len(p) >= 2
            and p.startswith("-")
            and not p.startswith("--")
            and all(ch in _LS_COMBINABLE for ch in p[1:])
        ):
            # Combined ls short flags: e.g. -la, -lah, -Al
            if "a" in p or "A" in p:
                result["include_hidden"] = True
            i += 1
        elif p == "-name" and i + 1 < len(args):
            result["name_pattern"] = args[i + 1]
            i += 2
        elif p == "-iname" and i + 1 < len(args):
            # Case-insensitive name match — not supported by our find handler.
            result["unsupported"] = True
            i += 2
        elif p == "-type" and i + 1 < len(args):
            t = args[i + 1]
            if t == "f":
                result["find_type"] = "file"
            elif t == "d":
                result["find_type"] = "directory"
            else:
                # l/s/p/c/b — unsupported by our find handler.
                result["unsupported"] = True
            i += 2
        elif p == "-maxdepth" and i + 1 < len(args):
            try:
                result["depth"] = int(args[i + 1])
            except ValueError:
                # Non-numeric depth: keep the default.
                pass
            i += 2
        elif is_find and p in ("-mindepth", "-mtime", "-ctime", "-atime", "-size",
                                "-exec", "-execdir", "-delete", "-print0", "-prune",
                                "-newer", "-perm", "-user", "-group", "-not", "!", "-o"):
            # find features that change result semantics — fall back to real bash.
            result["unsupported"] = True
            i += 1
        elif p.startswith("-"):
            # Unknown flag → can't safely simulate.
            result["unsupported"] = True
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
        else:
            result["target"] = p
            i += 1

    return result


# Pipeline normalization

# Collapse read-only pipe chains into one NormalizedCommand; None if unsafe. The first stage must be
# an OPEN or SURVEY command and every later stage a read-only transform; a grep stage turns the whole
# pipeline into a LOCATE, head/tail/sed stages narrow the line range.
def _normalize_pipeline(stages: list[str], cwd: str) -> NormalizedCommand | None:
    if len(stages) < 2:
        return None

    # Parse the first stage (the source)
    first_parts = _safe_split(stages[0])
    if not first_parts:
        return None
    first_cmd = first_parts[0]
    first_args = first_parts[1:]

    primary_intent = _classify_single_command(first_cmd)
    if primary_intent not in (Intent.OPEN, Intent.SURVEY):
        # Only open/survey commands can be the source of a pipeline.
        return None

    # Every RHS stage must be a read-only transform (head, grep, sed, …).
    for stage in stages[1:]:
        stage_parts = _safe_split(stage)
        if not stage_parts:
            return None
        if stage_parts[0] not in _READ_ONLY_RHS_CMDS:
            # Side-effecting command → real bash.
            return None

    # Parse source arguments.
    open_args = _parse_open_args(first_cmd, first_args)
    target = open_args.get("target", ".")
    start_line = open_args.get("start_line")
    end_line = open_args.get("end_line")

    # Fold RHS stages into intent (OPEN vs LOCATE) and line/pattern parameters.
    final_intent = Intent.OPEN
    pattern: str | None = None
    rhs_start: int | None = None
    rhs_end: int | None = None
    tail_n: int | None = open_args.get("tail_n")

    for stage in stages[1:]:
        stage_parts = _safe_split(stage)
        rhs_cmd = stage_parts[0]
        rhs_args = stage_parts[1:]

        if rhs_cmd in _LOCATE_CMDS or rhs_cmd in ("grep", "egrep"):
            # The whole pipeline becomes a LOCATE
            final_intent = Intent.LOCATE
            loc = _parse_locate_args(rhs_cmd, rhs_args)
            pattern = loc.get("pattern")

        elif rhs_cmd == "head":
            n = 10
            if "-n" in rhs_args:
                idx = rhs_args.index("-n")
                if idx + 1 < len(rhs_args):
                    try:
                        n = int(rhs_args[idx + 1])
                    except ValueError:
                        # Non-numeric count: keep the default of 10.
                        pass
            rhs_start = start_line or 1
            rhs_end = (start_line or 1) + n - 1

        elif rhs_cmd == "tail":
            n = 10
            if "-n" in rhs_args:
                idx = rhs_args.index("-n")
                if idx + 1 < len(rhs_args):
                    try:
                        n = int(rhs_args[idx + 1])
                    except ValueError:
                        # Non-numeric count: keep the default of 10.
                        pass
            tail_n = n

        elif rhs_cmd in ("wc",) and "-l" in rhs_args:
            # Stays OPEN, handler will show wc metadata.
            pass

        elif rhs_cmd == "sed":
            loc_args = _parse_open_args("sed", rhs_args)
            if loc_args.get("start_line"):
                rhs_start = loc_args["start_line"]
                rhs_end = loc_args["end_line"]

    # Build the normalized command
    nc = NormalizedCommand(
        intent=final_intent,
        raw_command="|".join(stages),
        target=target,
        start_line=rhs_start or start_line,
        end_line=rhs_end or end_line,
        pattern=pattern,
        was_compound=True,
    )
    if tail_n is not None and nc.start_line is None:
        # Store for handler to compute actual tail range.
        nc._tail_n = tail_n  # type: ignore[attr-defined]

    return nc


# Main entry point

# Classify a shell command; None means fall through to real bash.
def classify(command: str, cwd: str = ".") -> NormalizedCommand | None:
    command = command.strip()
    if not command:
        return None

    # Strip optional leading `cd dir && ` prefix (model cd idiom)
    command = re.sub(r"^cd\s+\S+\s*&&\s*", "", command)

    # Check for complex shell constructs that can't be safely parsed
    # (&&, ||, ;, subshells, redirections) — real bash only
    if re.search(r"&&|\|\||;|`|\$\(|>>?|<", command):
        return None

    # Check if it's a pipeline
    if "|" in command:
        stages = _split_pipeline(command)
        if stages is None:
            return None
        if len(stages) == 1:
            # Single-stage "pipeline" — treat as normal
            command = stages[0]
        else:
            return _normalize_pipeline(stages, cwd)

    # Single command (no pipe, or pipe collapsed to one stage).
    parts = _safe_split(command)
    if not parts:
        return None

    cmd = parts[0]
    args = parts[1:]
    intent = _classify_single_command(cmd)

    if intent is None:
        # Unknown command → RUN (real bash)
        return NormalizedCommand(intent=Intent.RUN, raw_command=command, target=".")

    if intent == Intent.OPEN:
        parsed = _parse_open_args(cmd, args)
        if parsed.get("unsupported"):
            return None
        target = parsed.get("target", ".")
        if _is_container_path(target):
            return None

        # Multi-file: cat a.txt b.txt — let real bash concatenate correctly.
        if parsed.get("extra_targets"):
            return None
        return NormalizedCommand(
            intent=intent,
            raw_command=command,
            target=target,
            extra_targets=[],
            start_line=parsed.get("start_line"),
            end_line=parsed.get("end_line"),
        )

    if intent == Intent.LOCATE:
        parsed = _parse_locate_args(cmd, args)
        if parsed.get("unsupported"):
            return None

        # -l/-L change output format to filenames — not implemented in handler,
        # so let real bash (or rg) return the correct output.
        if parsed.get("files_with_matches"):
            return None
        target = parsed.get("target", ".")
        if _is_container_path(target):
            return None
        return NormalizedCommand(
            intent=intent,
            raw_command=command,
            target=target,
            pattern=parsed.get("pattern"),
            case_sensitive=parsed.get("case_sensitive", True),
            glob_pattern=parsed.get("glob_pattern"),
            context_before=parsed.get("context_before", 0),
            context_after=parsed.get("context_after", 0),
            files_with_matches=False,
        )

    if intent == Intent.SURVEY:
        parsed = _parse_survey_args(cmd, args)
        if parsed.get("unsupported"):
            return None
        target = parsed.get("target", ".")
        if _is_container_path(target):
            return None
        return NormalizedCommand(
            intent=intent,
            raw_command=command,
            target=target,
            depth=parsed.get("depth"),
            include_hidden=parsed.get("include_hidden", False),
            name_pattern=parsed.get("name_pattern"),
            find_type=parsed.get("find_type"),
        )

    # MUTATE — no special parsing needed here, pass through
    if intent == Intent.MUTATE:
        target = next((p for p in args if not p.startswith("-")), ".")
        return NormalizedCommand(
            intent=intent,
            raw_command=command,
            target=target,
        )

    return None
