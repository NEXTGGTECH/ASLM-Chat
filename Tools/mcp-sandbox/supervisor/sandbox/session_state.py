# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from sandbox.config import CONTAINER_WORKSPACE, DEFAULT_TASK_DIR, LOOP_BREAK_THRESHOLD

# Container-side task root (e.g. "/workspace/_sandbox").
_CONTAINER_TASK_ROOT = f"{CONTAINER_WORKSPACE}/{DEFAULT_TASK_DIR}"

# Caps that keep the per-session state bounded: tracked paths, read windows kept per path, loop
# counters, and representations remembered per path.
MAX_TRACKED_PATHS = 1000
MAX_READ_WINDOWS_PER_PATH = 100
MAX_LOOP_COUNTERS = 1000
MAX_REPRESENTATIONS_PER_PATH = 20


# Keep singleton state bounded during long-lived supervisor sessions: drop the oldest-inserted keys
# until at most max_items remain.
def _trim_dict_oldest(mapping: dict[Any, Any], max_items: int) -> None:
    overflow = len(mapping) - max_items
    if overflow <= 0:
        return
    for key in list(mapping.keys())[:overflow]:
        mapping.pop(key, None)


# Per-session exploration memory: which paths were touched and which line windows were read, loop
# counters, served representations, recent searches, a repo survey cache and the persisted task cwd.
@dataclass
class ExplorationState:
    # path → number of times touched by any read-like intent
    touched_paths: dict[str, int] = field(default_factory=dict)

    # path → [(start_line, end_line), ...] windows already read
    read_windows: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    # "intent:path" → consecutive hit count (resets when target changes)
    loop_counters: dict[str, int] = field(default_factory=dict)

    # path → list of representation types already served ("raw", "map", "outline", ...)
    representations_served: dict[str, list[str]] = field(default_factory=dict)

    # last N search queries (for dedup / context)
    last_searches: list[dict[str, Any]] = field(default_factory=list)

    # cached repo survey result (avoid re-scanning)
    repo_survey_cache: dict[str, Any] | None = None
    repo_survey_ts: float = 0.0

    # Persisted task working directory (relative to task_root, always inside it)
    task_cwd: str = "."

    # Recording

    # Record that a path was accessed with a given intent, optionally with the line window that was read
    # and the representation that was served. Every table is trimmed to its cap.
    def record_touch(
        self,
        path: str | None,
        intent: str,
        window: tuple[int, int] | None = None,
        representation: str | None = None,
    ) -> None:
        if path is None:
            return

        # Count the access.
        self.touched_paths[path] = self.touched_paths.get(path, 0) + 1
        _trim_dict_oldest(self.touched_paths, MAX_TRACKED_PATHS)

        # Remember the line window that was read.
        if window is not None:
            windows = self.read_windows.setdefault(path, [])
            windows.append(window)
            if len(windows) > MAX_READ_WINDOWS_PER_PATH:
                del windows[: len(windows) - MAX_READ_WINDOWS_PER_PATH]
            _trim_dict_oldest(self.read_windows, MAX_TRACKED_PATHS)

        # Count repeated hits of the same intent on the same path for loop detection.
        key = f"{intent}:{path}"
        self.loop_counters[key] = self.loop_counters.get(key, 0) + 1
        _trim_dict_oldest(self.loop_counters, MAX_LOOP_COUNTERS)

        # Remember which representation was served.
        if representation is not None:
            served = self.representations_served.setdefault(path, [])
            served.append(representation)
            if len(served) > MAX_REPRESENTATIONS_PER_PATH:
                del served[: len(served) - MAX_REPRESENTATIONS_PER_PATH]
            _trim_dict_oldest(self.representations_served, MAX_TRACKED_PATHS)

    # Record a search query.
    def record_search(self, pattern: str, path: str, hit_count: int) -> None:
        self.last_searches.append({
            "pattern": pattern,
            "path": path,
            "hit_count": hit_count,
            "ts": time.time(),
        })

        # Keep last 10.
        if len(self.last_searches) > 10:
            self.last_searches = self.last_searches[-10:]

    # Cache a repo survey result.
    def record_survey(self, data: dict[str, Any]) -> None:
        self.repo_survey_cache = data
        self.repo_survey_ts = time.time()

    # Invalidate the survey cache after a filesystem mutation.
    def invalidate_survey_cache(self) -> None:
        self.repo_survey_cache = None
        self.repo_survey_ts = 0.0

    # Loop detection

    # Return True when the same intent+path has been hit threshold or more times.
    def should_break_loop(
        self,
        path: str | None,
        intent: str,
        threshold: int = LOOP_BREAK_THRESHOLD,
    ) -> bool:
        if path is None:
            return False
        key = f"{intent}:{path}"
        return self.loop_counters.get(key, 0) >= threshold

    # Return fraction of [start, end] already covered by prior reads.
    def get_read_overlap(
        self,
        path: str,
        start_line: int,
        end_line: int,
    ) -> float:
        windows = self.read_windows.get(path, [])
        if not windows:
            return 0.0

        requested = set(range(start_line, end_line + 1))
        if not requested:
            return 0.0

        covered = set()
        for ws, we in windows:
            covered.update(range(ws, we + 1))

        overlap = requested & covered
        return len(overlap) / len(requested)

    # Representation selection

    # Suggest the next representation to serve for this path.
    # If the model has already seen "raw" multiple times, switch to "map".
    # If it has seen "map", switch to "outline".
    def get_best_representation(self, path: str, intent: str) -> str:
        served = self.representations_served.get(path, [])
        raw_count = served.count("raw")
        map_count = served.count("map")

        if raw_count >= 2 and map_count == 0:
            return "map"
        if raw_count >= 2 and map_count >= 1:
            return "outline"
        return "raw"

    # Return True if a survey is cached that is younger than max_age_s seconds.
    def has_survey_cache(self, max_age_s: float = 300.0) -> bool:
        if self.repo_survey_cache is None:
            return False
        return (time.time() - self.repo_survey_ts) < max_age_s

    # Unread regions

    # Return contiguous unread line ranges for a file, skipping gaps shorter than min_gap lines. A file
    # without recorded reads is one range covering all lines.
    def get_unread_windows(
        self,
        path: str,
        total_lines: int,
        min_gap: int = 20,
    ) -> list[tuple[int, int]]:
        windows = self.read_windows.get(path, [])
        if not windows:
            return [(1, total_lines)]

        covered = set()
        for ws, we in windows:
            covered.update(range(ws, we + 1))

        all_lines = set(range(1, total_lines + 1))
        unread = sorted(all_lines - covered)

        if not unread:
            return []

        # Merge into contiguous ranges.
        ranges: list[tuple[int, int]] = []
        start = unread[0]
        prev = unread[0]
        for line in unread[1:]:
            if line == prev + 1:
                prev = line
            else:
                if prev - start + 1 >= min_gap:
                    ranges.append((start, prev))
                start = line
                prev = line
        if prev - start + 1 >= min_gap:
            ranges.append((start, prev))

        return ranges

    # Compact context for injection

    # Build a compact exploration summary for injection into responses.
    # Returns None if the state is too sparse to be useful.
    def compact_context(self, max_items: int = 5) -> str | None:
        if not self.touched_paths:
            return None

        parts: list[str] = ["── exploration state ──"]

        # Hot files (most touched).
        hot = sorted(
            self.touched_paths.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:max_items]
        if hot:
            items = ", ".join(f"{p} ({c}x)" for p, c in hot)
            parts.append(f"Touched: {items}")

        # Read coverage.
        for path, windows in list(self.read_windows.items())[:max_items]:
            ranges = ", ".join(f"[{s}:{e}]" for s, e in windows[-3:])
            parts.append(f"Read: {path} {ranges}")

        # Recent searches.
        if self.last_searches:
            last = self.last_searches[-1]
            parts.append(
                f"Last search: \"{last['pattern']}\" in {last['path']}"
                f" ({last['hit_count']} hits)"
            )

        return "\n".join(parts) if len(parts) > 1 else None

    # Task CWD management

    # Update task_cwd if final_cwd is inside task_root.
    # final_cwd_container: absolute container path from bash wrapper
    # (e.g. '/workspace/_sandbox/src'). Returns True if task_cwd was updated,
    # False if it stayed the same.
    def update_task_cwd(self, final_cwd_container: str) -> bool:
        if not final_cwd_container:
            return False

        # Check if final_cwd is inside the container task root.
        if final_cwd_container == _CONTAINER_TASK_ROOT:
            self.task_cwd = "."
            return True

        if final_cwd_container.startswith(_CONTAINER_TASK_ROOT + "/"):
            rel = final_cwd_container[len(_CONTAINER_TASK_ROOT) + 1:]
            self.task_cwd = rel or "."
            return True

        # Outside task root — don't update task_cwd.
        return False

    # Resolve the effective cwd for a bash command.
    # Contract: cwd absent or None → use persisted task_cwd; cwd == "." → use
    # persisted task_cwd; cwd == explicit → one-shot override (does not mutate state).
    def get_effective_cwd(self, explicit_cwd: str | None) -> str:
        if explicit_cwd is None or explicit_cwd == "." or explicit_cwd == "":
            return self.task_cwd
        return explicit_cwd

    # Format execution_ended_in for display.
    # Returns None if the command stayed in task_cwd (no need to show).
    # Returns relative path if inside task-space.
    # Returns 'container:/path' if outside task-space.
    def format_execution_ended_in(
        self,
        final_cwd_container: str,
    ) -> str | None:
        if not final_cwd_container:
            return None

        # Inside task-root?
        if final_cwd_container == _CONTAINER_TASK_ROOT:
            rel = "."
        elif final_cwd_container.startswith(_CONTAINER_TASK_ROOT + "/"):
            rel = final_cwd_container[len(_CONTAINER_TASK_ROOT) + 1:] or "."
        else:
            # Outside task-root → container-space.
            return f"container:{final_cwd_container}"

        # If ended where we are, suppress.
        if rel == self.task_cwd:
            return None

        return rel


# Singleton per server process

# The lazily created session state shared by all tool calls of this process.
_session_state: ExplorationState | None = None


# Return the current session state, creating it if needed.
def get_session_state() -> ExplorationState:
    global _session_state
    if _session_state is None:
        _session_state = ExplorationState()
    return _session_state


# Reset the session state (for testing).
def reset_session_state() -> None:
    global _session_state
    _session_state = None
