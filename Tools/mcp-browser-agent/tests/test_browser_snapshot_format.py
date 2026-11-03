# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import pytest

from browser import (
    _build_parsed_state,
    _camoufox_launch_options,
    _filter_snapshot_controls,
    _format_control_line,
    _format_parsed_controls,
    _is_noise_element,
    is_browser_closed_error,
)


# Verify that the Camoufox launch options exclude the optional UBO default addon.
@pytest.mark.unit
def test_camoufox_launch_excludes_optional_default_addon() -> None:
    options = _camoufox_launch_options()

    assert options["exclude_addons"]
    assert options["exclude_addons"][0].name == "UBO"


# Verify that an unnamed link in the footer landmark counts as noise while a named button in main does not.
@pytest.mark.unit
def test_is_noise_element_filters_empty_links_outside_main() -> None:
    assert _is_noise_element({"role": "link", "name": "", "landmark": "contentinfo"})
    assert not _is_noise_element(
        {"role": "button", "name": "Submit", "landmark": "main"},
    )


# Verify that a text box line shows its ref, role, editable flag and region.
@pytest.mark.unit
def test_format_control_line_includes_editable_and_region() -> None:
    line = _format_control_line(
        {
            "ref": "e3",
            "role": "textbox",
            "name": "Search",
            "value": "cats",
            "landmark": "main",
            "interactive": True,
        }
    )
    assert "[e3]" in line
    assert "textbox" in line
    assert "editable" in line
    assert 'region=main' in line


# Verify that the compact filter drops the unnamed footer link and the full filter keeps both controls.
@pytest.mark.unit
def test_filter_snapshot_controls_hides_noise_in_compact_mode() -> None:
    elements = [
        {"ref": "e0", "role": "button", "name": "Go", "interactive": True, "landmark": "main"},
        {"ref": "e1", "role": "link", "name": "", "interactive": True, "landmark": "contentinfo"},
    ]
    compact = _filter_snapshot_controls(elements, full=False)
    refs = {item["ref"] for item in compact}
    assert refs == {"e0"}
    full = _filter_snapshot_controls(elements, full=True)
    assert len(full) == 2


# Verify that controls are listed under group headings and that a max_items cap of 2 leaves a "more
# controls hidden" note for the third control.
@pytest.mark.unit
def test_format_parsed_controls_groups_and_caps_items() -> None:
    elements = [
        {"ref": "e0", "role": "textbox", "name": "Query", "interactive": True, "landmark": "main"},
        {"ref": "e1", "role": "button", "name": "Search", "interactive": True, "landmark": "main"},
        {"ref": "e2", "role": "link", "name": "Docs", "interactive": True, "landmark": "main"},
    ]
    lines = _format_parsed_controls(elements, full=False, max_items=2)
    joined = "\n".join(lines)
    assert "Text inputs" in joined
    assert "more controls hidden" in joined


# Verify that the parsed state of a compact snapshot reports the "controls" mode and the control and
# text-input counts.
@pytest.mark.unit
def test_build_parsed_state_reports_counts() -> None:
    elements = [
        {"ref": "e0", "role": "textbox", "name": "Query", "interactive": True, "landmark": "main"},
        {"ref": "e1", "role": "button", "name": "Search", "interactive": True, "landmark": "main"},
    ]
    state = _build_parsed_state(elements, full=False, max_items=10)
    assert state["mode"] == "controls"
    assert state["counts"]["visible_controls"] == 2
    assert state["counts"]["text_inputs"] == 1


# Verify that exceptions with browser-closed or driver-connection-closed messages are recognized as a dead
# browser.
@pytest.mark.unit
@pytest.mark.parametrize(
    "message",
    [
        "Target page, context or browser has been closed",
        "Connection closed while reading from the driver",
    ],
)
def test_is_browser_closed_error(message: str) -> None:
    assert is_browser_closed_error(RuntimeError(message))
