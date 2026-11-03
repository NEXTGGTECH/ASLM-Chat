# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest

from browser_text import (
    _css_string,
    _infer_action,
    _next_text,
    _parse_line_range,
    _replace_match,
    replace_line_range,
)


# Verify that a normal range is a replace, a range past the end of the text raises an "outside text" error,
# and an end before the start (6:5 on a 5-line text) is an insert after the last line.
@pytest.mark.unit
def test_parse_line_range_replace_and_insert_modes() -> None:
    assert _parse_line_range("2:3", 5) == (2, 3, False)
    with pytest.raises(ValueError, match="outside text"):
        _parse_line_range("9:10", 5)
    assert _parse_line_range("6:5", 5) == (6, 5, True)


# Verify that replacing line 2 keeps the other lines, and that the range 3:2 on a two-line text appends
# the new line after the last one.
@pytest.mark.unit
def test_replace_line_range_replaces_and_inserts() -> None:
    current = "line1\nline2\nline3\n"
    replaced = replace_line_range(current, "2:2", "MIDDLE")
    assert replaced.splitlines() == ["line1", "MIDDLE", "line3"]

    inserted = replace_line_range("a\nb\n", "3:2", "NEW")
    assert inserted.splitlines() == ["a", "b", "NEW"]


# Verify that the action comes from the explicit action argument, or else from the shape of the arguments:
# old_text/new_text means replace, text means set, and a range means delete.
@pytest.mark.unit
@pytest.mark.parametrize(
    ("args", "expected"),
    [
        ({"action": "read"}, "read"),
        ({"old_text": "x", "new_text": "y"}, "replace"),
        ({"text": "hello"}, "set"),
        ({"range": "1:1"}, "delete"),
    ],
)
def test_infer_action(args: dict, expected: str) -> None:
    assert _infer_action(args) == expected


# Verify that a single match is replaced, several matches raise an error unless replace_all is set, and
# replace_all replaces every match.
@pytest.mark.unit
def test_replace_match_requires_unique_match_without_replace_all() -> None:
    assert _replace_match("only foo here", "foo", "bar", False) == "only bar here"
    with pytest.raises(ValueError, match="found 2 times"):
        _replace_match("foo foo", "foo", "bar", False)
    assert _replace_match("foo foo", "foo", "bar", True) == "bar bar"


# Verify that set returns the given text, replace swaps old_text for new_text, and delete with all=true
# clears the text.
@pytest.mark.unit
def test_next_text_set_replace_and_delete() -> None:
    assert _next_text("set", {"text": "NEW"}, "OLD") == "NEW"
    assert _next_text("replace", {"old_text": "OLD", "new_text": "NEW"}, "OLD") == "NEW"
    assert _next_text("delete", {"all": True}, "OLD") == ""


# Verify that the value is wrapped in double quotes and inner double quotes are backslash-escaped.
@pytest.mark.unit
def test_css_string_escapes_quotes() -> None:
    assert _css_string('say "hi"') == r'"say \"hi\""'
