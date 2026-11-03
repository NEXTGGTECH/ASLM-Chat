# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

# arXiv's legacy metadata API endpoint and the minimum gap between request starts that its
# published terms allow (one request per three seconds).
ARXIV_API_ENDPOINT = "https://export.arxiv.org/api/query"
ARXIV_API_MIN_INTERVAL_SECONDS = 3.0

# Process-wide pacing state: the lock allows a single API connection at a time and
# _last_started is the monotonic time at which the previous call began.
_LOCK = asyncio.Lock()
_last_started = 0.0


# Serialize legacy API calls and keep their start times at least three seconds apart. Academic
# search and the read_page arXiv handler share this gate, as the arXiv API terms require.
@asynccontextmanager
async def arxiv_api_slot() -> AsyncIterator[None]:
    global _last_started
    async with _LOCK:
        # Sleep off the remainder of the interval since the previous call started.
        delay = ARXIV_API_MIN_INTERVAL_SECONDS - (time.monotonic() - _last_started)
        if delay > 0:
            await asyncio.sleep(delay)
        _last_started = time.monotonic()
        yield


__all__ = [
    "ARXIV_API_ENDPOINT",
    "ARXIV_API_MIN_INTERVAL_SECONDS",
    "arxiv_api_slot",
]
