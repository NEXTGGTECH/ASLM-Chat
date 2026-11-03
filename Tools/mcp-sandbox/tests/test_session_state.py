# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import os
import sys
from pathlib import Path

# Package root of mcp-sandbox with its supervisor and sandbox source directories, put on sys.path at import
# time; the host workspace is set there too so the sandbox package can be imported.
ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR = ROOT / "supervisor"
SRC = ROOT / "src"
for path in (SUPERVISOR, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
os.environ["SANDBOX_HOST_WORKSPACE"] = str(ROOT)

from sandbox.session_state import get_session_state, reset_session_state  # noqa: E402
import sandbox.session_state as session_state_mod  # noqa: E402


# Verify that repeated touches of the same path are counted.
def test_state_tracks_touches():
    reset_session_state()
    state = get_session_state()

    state.record_touch("track.py", "open")
    state.record_touch("track.py", "open")

    assert state.touched_paths.get("track.py", 0) == 2


# Verify that a recorded search is kept with its pattern and hit count.
def test_state_tracks_searches():
    reset_session_state()
    state = get_session_state()

    state.record_search("hello", ".", 2)

    assert state.last_searches
    assert state.last_searches[-1]["pattern"] == "hello"
    assert state.last_searches[-1]["hit_count"] == 2


# Verify that a path touched three times with the same action is flagged as a loop, while another path is
# not.
def test_loop_detection():
    reset_session_state()
    state = get_session_state()

    state.record_touch("loop.py", "open")
    state.record_touch("loop.py", "open")
    state.record_touch("loop.py", "open")

    assert state.should_break_loop("loop.py", "open")
    assert not state.should_break_loop("other.py", "open")


# Verify that the unread windows of a 200-line file include the gap after the first read window (starting
# at line 51) and the tail up to line 200.
def test_unread_windows():
    reset_session_state()
    state = get_session_state()

    state.record_touch("file.py", "open", window=(1, 50))
    state.record_touch("file.py", "open", window=(80, 120))

    unread = state.get_unread_windows("file.py", 200)
    assert any(start == 51 for start, _end in unread)
    assert any(end == 200 for _start, end in unread)


# Verify that the read overlap of a requested range with lines 1-100 is 1.0 for the same range, about half
# for a half-overlapping range and 0.0 for a disjoint range.
def test_read_overlap():
    reset_session_state()
    state = get_session_state()

    state.record_touch("ov.py", "open", window=(1, 100))

    assert state.get_read_overlap("ov.py", 1, 100) == 1.0
    assert 0.4 < state.get_read_overlap("ov.py", 50, 150) < 0.6
    assert state.get_read_overlap("ov.py", 101, 200) == 0.0


# Verify that the per-session memory is capped: the number of tracked paths, loop counters, read windows
# and served representations stay within their limits, and per-path windows and representations keep only
# the newest entries.
def test_state_caps_long_lived_path_memory(monkeypatch):
    # Arrange: lower the caps so a few touches exceed them.
    reset_session_state()
    monkeypatch.setattr(session_state_mod, "MAX_TRACKED_PATHS", 3)
    monkeypatch.setattr(session_state_mod, "MAX_LOOP_COUNTERS", 4)
    monkeypatch.setattr(session_state_mod, "MAX_READ_WINDOWS_PER_PATH", 2)
    monkeypatch.setattr(session_state_mod, "MAX_REPRESENTATIONS_PER_PATH", 2)
    state = get_session_state()

    # Touch ten different paths: the tracked paths must stay within the caps.
    for index in range(10):
        state.record_touch(
            f"file_{index}.py",
            "open",
            window=(index + 1, index + 1),
            representation="raw",
        )

    assert len(state.touched_paths) <= 3
    assert len(state.read_windows) <= 3
    assert len(state.loop_counters) <= 4
    assert len(state.representations_served) <= 3

    # Touch one path three times: only the two newest windows and representations remain.
    state.record_touch("hot.py", "open", window=(1, 1), representation="raw")
    state.record_touch("hot.py", "open", window=(2, 2), representation="map")
    state.record_touch("hot.py", "open", window=(3, 3), representation="outline")

    assert state.read_windows["hot.py"] == [(2, 2), (3, 3)]
    assert state.representations_served["hot.py"] == ["map", "outline"]
