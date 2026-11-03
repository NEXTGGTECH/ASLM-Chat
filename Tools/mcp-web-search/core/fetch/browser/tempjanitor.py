# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from pathlib import Path

logger = logging.getLogger("core.fetch.browser.tempjanitor")

# Temp-dir name prefixes created per chromium launch (playwright + chromium itself).
_LEAK_PREFIXES = (
    "playwright_chromiumdev_profile-",
    "playwright-artifacts-",
    "scoped_dir",
)

# Orphans older than this are reaped. Kept above the daemon idle-shutdown (1800s) so a
# live daemon's profile — fresh while serving, gone once idle-shut — is never matched.
_MAX_AGE_SEC = 3600.0


# Remove orphaned browser temp profiles older than max_age_sec and return the count reaped.
# Every chromium launch (cloakbrowser, playwright) creates a throwaway user-data-dir under the
# system temp dir. Playwright removes it on a clean close, but a force-killed or crashed daemon
# leaks the whole profile (~120+ files of stale disk cache), and over a long-lived deployment
# those orphans pile up. The daemon owns the browser lifecycle, so it sweeps on startup.
# Safety: only dirs whose mtime is older than max_age_sec are touched, and the default age is
# kept above the daemon's idle-shutdown window, so a live profile is never a candidate. Removal
# is best-effort per dir: a still-locked dir makes rmtree raise and is skipped rather than
# deleted out from under a running browser.
def sweep_stale_browser_temp(max_age_sec: float = _MAX_AGE_SEC) -> int:
    root = Path(tempfile.gettempdir())
    now = time.time()
    reaped = 0

    # List the temp dir; if it cannot be read there is nothing to sweep.
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        logger.debug("temp janitor: cannot list %s: %s", root, exc)
        return 0

    # Reap every leaked browser dir that is old enough.
    for entry in entries:
        name = entry.name
        if not entry.is_dir() or not name.startswith(_LEAK_PREFIXES):
            continue
        try:
            age = now - entry.stat().st_mtime
        except OSError:
            # The dir vanished or is unreadable; leave it alone.
            continue

        # Too fresh: it could belong to a live browser.
        if age < max_age_sec:
            continue

        # rmtree raises if any file is locked (a live profile), in which case the dir is skipped.
        try:
            shutil.rmtree(entry)
        except OSError as exc:
            logger.debug("temp janitor: skip locked/partial %s: %s", name, exc)
            continue
        reaped += 1
    if reaped:
        logger.info("temp janitor: reaped %d orphaned browser temp dir(s)", reaped)
    return reaped
