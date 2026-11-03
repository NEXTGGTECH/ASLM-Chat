# Copyright NEXTGGTECH. Elastic License 2.0.

import os
import sys
from pathlib import Path

# Package root of mcp-sandbox; its sandbox source directory is put on sys.path and the host workspace is
# set at import time so the sandbox package can be imported.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["SANDBOX_HOST_WORKSPACE"] = str(ROOT)

from sandbox.intent import Intent, classify


# Verify that read-like commands (cat, head, tail, less, more, sed -n) classify as OPEN.
def test_open_variants():
    cases = [
        "cat file.py",
        "head -n 30 file.py",
        "tail -n 20 file.py",
        "less file.py",
        "more file.py",
        "sed -n '1,30p' file.py",
    ]
    for cmd in cases:
        nc = classify(cmd)
        assert nc is not None, f"Got None for: {cmd}"
        assert nc.intent == Intent.OPEN, f"Expected OPEN for: {cmd}, got {nc.intent}"


# Verify that head -n and sed -n 'a,bp' yield the start and end line of the requested range.
def test_open_line_ranges():
    nc = classify("head -n 30 file.py")
    assert nc.start_line == 1
    assert nc.end_line == 30, f"end_line={nc.end_line}"

    nc = classify("sed -n '10,50p' file.py")
    assert nc.start_line == 10, f"start_line={nc.start_line}"
    assert nc.end_line == 50, f"end_line={nc.end_line}"


# Verify that search commands (grep, rg, egrep) classify as LOCATE with the expected pattern.
def test_locate_variants():
    cases = [
        ("grep -r pattern .", "pattern"),
        ("grep -ri Pattern src/", "Pattern"),
        ("rg pattern src/", "pattern"),
        ("egrep pattern file.py", "pattern"),
    ]
    for cmd, expected_pattern in cases:
        nc = classify(cmd)
        assert nc is not None, f"Got None for: {cmd}"
        assert nc.intent == Intent.LOCATE, f"Expected LOCATE for: {cmd}, got {nc.intent}"
        assert nc.pattern == expected_pattern, f"Expected pattern={expected_pattern!r}, got {nc.pattern!r} for: {cmd}"


# Verify that ls, tree and find commands classify as SURVEY.
def test_survey_variants():
    cases = ["ls .", "ls -la src/", "tree -L 2", "find . -name '*.py'"]
    for cmd in cases:
        nc = classify(cmd)
        assert nc is not None, f"Got None for: {cmd}"
        assert nc.intent == Intent.SURVEY, f"Expected SURVEY for: {cmd}, got {nc.intent}"


# Verify that `cat file | head -n 20` classifies as a compound OPEN of file.py ending at line 20.
def test_compound_cat_head():
    nc = classify("cat file.py | head -n 20")
    assert nc is not None, "Got None for compound cat|head"
    assert nc.intent == Intent.OPEN, f"Expected OPEN, got {nc.intent}"
    assert nc.was_compound is True
    assert nc.end_line == 20, f"Expected end_line=20, got {nc.end_line}"
    assert nc.target == "file.py", f"Expected target=file.py, got {nc.target}"


# Verify that `cat file | grep pattern` classifies as a compound LOCATE of the pattern in file.py.
def test_compound_cat_grep():
    nc = classify("cat file.py | grep pattern")
    assert nc is not None, "Got None for compound cat|grep"
    assert nc.intent == Intent.LOCATE, f"Expected LOCATE, got {nc.intent}"
    assert nc.was_compound is True
    assert nc.pattern == "pattern", f"Expected pattern='pattern', got {nc.pattern!r}"
    assert nc.target == "file.py", f"Expected target=file.py, got {nc.target}"


# Verify that `head -n 50 file | grep pattern` classifies as a compound LOCATE.
def test_compound_head_grep():
    nc = classify("head -n 50 file.py | grep pattern")
    assert nc is not None
    assert nc.intent == Intent.LOCATE, f"Expected LOCATE, got {nc.intent}"
    assert nc.was_compound is True


# Verify that execution commands (python, pytest, npm, git, curl, make, pip) are not classified as
# OPEN/LOCATE/SURVEY: the result is None or RUN, so they reach real bash.
def test_run_commands_return_none():
    run_cases = [
        "python script.py",
        "pytest tests/",
        "npm install",
        "git status",
        "curl https://example.com",
        "make build",
        "pip install requests",
    ]
    for cmd in run_cases:
        nc = classify(cmd)

        # These should either return None or Intent.RUN
        if nc is not None:
            assert nc.intent == Intent.RUN, f"Expected RUN for: {cmd}, got {nc.intent}"


# Verify that commands with &&, ||, ;, subshells or redirections are not routed as OPEN/LOCATE/SURVEY, so
# they reach real bash.
def test_chains_return_none():
    chain_cases = [
        "echo hello && cat file.py",
        "cat file.py || echo failed",
        "cat file.py; cat other.py",
        "cat $(find . -name main.py)",
        "cat file.py > output.txt",
        "cat file.py >> output.txt",
    ]
    for cmd in chain_cases:
        nc = classify(cmd)

        # These must not be routed as OPEN/LOCATE/SURVEY
        if nc is not None:
            assert nc.intent not in (Intent.OPEN, Intent.LOCATE, Intent.SURVEY), (
                f"Should not route {cmd!r} as {nc.intent}"
            )


# Verify that grep -i produces a LOCATE with case_sensitive False.
def test_locate_case_sensitivity():
    nc = classify("grep -ri pattern src/")
    assert nc is not None
    assert nc.intent == Intent.LOCATE
    assert nc.case_sensitive is False, "Expected case_sensitive=False"


# Verify that `rg --type py` produces a LOCATE with the glob pattern *.py.
def test_locate_rg_type_flag():
    nc = classify("rg pattern --type py src/")
    assert nc is not None
    assert nc.intent == Intent.LOCATE
    assert nc.glob_pattern == "*.py", f"Expected glob=*.py, got {nc.glob_pattern}"


# Verify that find with -name, -type f and -maxdepth classifies as SURVEY carrying the name pattern, the
# file type and the depth.
def test_find_carries_name_and_type():
    nc = classify('find . -name "*.py" -type f -maxdepth 2')
    assert nc is not None, "find with supported flags should classify"
    assert nc.intent == Intent.SURVEY
    assert nc.name_pattern == "*.py", f"name_pattern={nc.name_pattern!r}"
    assert nc.find_type == "file", f"find_type={nc.find_type!r}"
    assert nc.depth == 2, f"depth={nc.depth!r}"


# Verify that grep -C sets both context sizes, and that -A and -B set the lines after and before separately.
def test_grep_context_carried():
    nc = classify("grep -C 2 pattern file.py")
    assert nc is not None
    assert nc.intent == Intent.LOCATE
    assert nc.context_before == 2 and nc.context_after == 2, (
        f"context=({nc.context_before},{nc.context_after})"
    )

    nc = classify("grep -A 3 -B 1 pattern file.py")
    assert nc is not None
    assert nc.context_before == 1 and nc.context_after == 3


# Verify that the ls flags -la, -al, -lah and -A all set include_hidden.
def test_ls_la_includes_hidden():
    for cmd in ("ls -la", "ls -al", "ls -lah .", "ls -A"):
        nc = classify(cmd)
        assert nc is not None, cmd
        assert nc.intent == Intent.SURVEY
        assert nc.include_hidden is True, f"include_hidden False for: {cmd}"


# Verify that du is not routed as a directory listing: the result is None or RUN, so it reaches real bash.
def test_du_falls_through_to_bash():
    nc = classify("du -sh .")

    # Either None (no routing) or RUN intent — must not be SURVEY.
    if nc is not None:
        assert nc.intent == Intent.RUN, f"du routed as {nc.intent}"


# Verify that find with unsupported flags (-exec, -mtime, -iname, -type l) is not routed: the result is
# None or RUN, so it falls back to real bash.
def test_find_unsupported_flags_fall_back():
    for cmd in (
        "find . -name '*.log' -exec rm {} ;",
        "find . -mtime -7",
        "find . -iname '*.PY'",
        "find . -type l",
    ):
        nc = classify(cmd)
        if nc is not None:
            assert nc.intent == Intent.RUN, f"{cmd!r} routed as {nc.intent}"


# Verify that grep -l, which changes the output to file names and is not handled by the router, is not
# routed: the result is None or RUN, so it falls back to real bash.
def test_grep_files_with_matches():
    nc = classify("grep -l pattern src/")

    # -l/-L are not handled by the router, so classifier returns None → real bash
    if nc is not None:
        assert nc.intent == Intent.RUN


# Verify that the inverted grep -v, which changes the match semantics, is not routed: the result is None
# or RUN, so it falls back to real bash.
def test_grep_inverted_falls_back():
    nc = classify("grep -v pattern file.py")
    if nc is not None:
        assert nc.intent == Intent.RUN
