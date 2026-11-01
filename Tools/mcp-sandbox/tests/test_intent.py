# Copyright NEXTGGTECH. Elastic License 2.0.

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["SANDBOX_HOST_WORKSPACE"] = str(ROOT)

from sandbox.intent import Intent, classify


# All read-like commands classify as OPEN.
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


# head/tail/sed extract line ranges.
def test_open_line_ranges():
    nc = classify("head -n 30 file.py")
    assert nc.start_line == 1
    assert nc.end_line == 30, f"end_line={nc.end_line}"

    nc = classify("sed -n '10,50p' file.py")
    assert nc.start_line == 10, f"start_line={nc.start_line}"
    assert nc.end_line == 50, f"end_line={nc.end_line}"



# All search commands classify as LOCATE.
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


# ls/tree/find classify as SURVEY.
def test_survey_variants():
    cases = ["ls .", "ls -la src/", "tree -L 2", "find . -name '*.py'"]
    for cmd in cases:
        nc = classify(cmd)
        assert nc is not None, f"Got None for: {cmd}"
        assert nc.intent == Intent.SURVEY, f"Expected SURVEY for: {cmd}, got {nc.intent}"


# cat file | head -n 20 → OPEN with line range.
def test_compound_cat_head():
    nc = classify("cat file.py | head -n 20")
    assert nc is not None, "Got None for compound cat|head"
    assert nc.intent == Intent.OPEN, f"Expected OPEN, got {nc.intent}"
    assert nc.was_compound is True
    assert nc.end_line == 20, f"Expected end_line=20, got {nc.end_line}"
    assert nc.target == "file.py", f"Expected target=file.py, got {nc.target}"


# cat file | grep pattern → LOCATE.
def test_compound_cat_grep():
    nc = classify("cat file.py | grep pattern")
    assert nc is not None, "Got None for compound cat|grep"
    assert nc.intent == Intent.LOCATE, f"Expected LOCATE, got {nc.intent}"
    assert nc.was_compound is True
    assert nc.pattern == "pattern", f"Expected pattern='pattern', got {nc.pattern!r}"
    assert nc.target == "file.py", f"Expected target=file.py, got {nc.target}"


# head -n 50 file | grep pattern → LOCATE.
def test_compound_head_grep():
    nc = classify("head -n 50 file.py | grep pattern")
    assert nc is not None
    assert nc.intent == Intent.LOCATE, f"Expected LOCATE, got {nc.intent}"
    assert nc.was_compound is True


# Execution commands return None (→ real bash).
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


# &&, ||, ;, subshells, redirections → None (real bash).
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


# grep -i → case_sensitive=False.
def test_locate_case_sensitivity():
    nc = classify("grep -ri pattern src/")
    assert nc is not None
    assert nc.intent == Intent.LOCATE
    assert nc.case_sensitive is False, "Expected case_sensitive=False"


# rg --type py → glob_pattern=*.py.
def test_locate_rg_type_flag():
    nc = classify("rg pattern --type py src/")
    assert nc is not None
    assert nc.intent == Intent.LOCATE
    assert nc.glob_pattern == "*.py", f"Expected glob=*.py, got {nc.glob_pattern}"


# find -name '*.py' -type f → NormalizedCommand with name_pattern/find_type.
def test_find_carries_name_and_type():
    nc = classify('find . -name "*.py" -type f -maxdepth 2')
    assert nc is not None, "find with supported flags should classify"
    assert nc.intent == Intent.SURVEY
    assert nc.name_pattern == "*.py", f"name_pattern={nc.name_pattern!r}"
    assert nc.find_type == "file", f"find_type={nc.find_type!r}"
    assert nc.depth == 2, f"depth={nc.depth!r}"


# grep -C 2 pattern file → context_before/after = 2.
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


# ls -la and combined short flags must enable include_hidden.
def test_ls_la_includes_hidden():
    for cmd in ("ls -la", "ls -al", "ls -lah .", "ls -A"):
        nc = classify(cmd)
        assert nc is not None, cmd
        assert nc.intent == Intent.SURVEY
        assert nc.include_hidden is True, f"include_hidden False for: {cmd}"


# du is NOT routed as a directory listing — must reach real bash.
def test_du_falls_through_to_bash():
    nc = classify("du -sh .")
    # Either None (no routing) or RUN intent — must not be SURVEY.
    if nc is not None:
        assert nc.intent == Intent.RUN, f"du routed as {nc.intent}"


# find with -exec / -mtime / etc. must fall back to real bash.
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


# grep -l changes output format to filenames — not implemented, fall back to bash.
def test_grep_files_with_matches():
    nc = classify("grep -l pattern src/")
    # -l/-L are not handled by the router, so classifier returns None → real bash
    if nc is not None:
        assert nc.intent == Intent.RUN


# grep -v changes semantics — fall back to real bash.
def test_grep_inverted_falls_back():
    nc = classify("grep -v pattern file.py")
    if nc is not None:
        assert nc.intent == Intent.RUN


