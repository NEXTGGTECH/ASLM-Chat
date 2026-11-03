# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import threading
import time

# Statuses that mean "the provider actively refused us", weighted as antibot signals.
ANTIBOT_STATUS = frozenset({401, 403, 405, 406, 429, 503})

# Cooldown policy: the number of consecutive hard failures before the first cooldown, the
# first cooldown in seconds (5 min, doubled per failure past the threshold) and its cap (30 min).
_FAIL_THRESHOLD = 2
_COOLDOWN_BASE = 300.0
_COOLDOWN_MAX = 1800.0


# Persistent per-provider cooldown for the academic vertical.
# Unlike the shopping engine's intra-search ProviderState (rebuilt each call), this lives in a
# process-level singleton so a provider that keeps getting blocked (403/429/503) or erroring is
# benched across searches: repeated antibot walls back it off with an exponential cooldown
# instead of being re-hit every query. Mirrors the circuit-breaker spirit of
# `core/search/health.py` but is far simpler: scholarly REST APIs either answer or don't.
# A clean 200 (even with zero papers, a legitimately empty result) resets the streak; only hard
# failures (antibot status, transport error, parse blow-up) accrue toward a cooldown.
class ProviderHealth:
    # Create empty state: failure streaks, cooldown deadlines, last failure reasons and last
    # fire times, all keyed by provider name and guarded by one lock.
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._fails: dict[str, int] = {}
        self._cooldown_until: dict[str, float] = {}
        self._last_reason: dict[str, str] = {}
        self._last_fired: dict[str, float] = {}

    # Return True when the provider may be queried now: not in cooldown, and at least
    # `min_interval` seconds (its polite rps budget) since the last fire — preemptive
    # pacing so an antibot-prone source (Scholar) is skipped rather than blocked.
    def available(self, name: str, min_interval: float = 0.0) -> bool:
        now = time.time()
        if self._cooldown_until.get(name, 0.0) > now:
            return False
        if min_interval > 0.0 and (now - self._last_fired.get(name, 0.0)) < min_interval:
            return False
        return True

    # Record that a provider is about to be queried now (drives min-interval pacing).
    def note_fired(self, name: str) -> None:
        self._last_fired[name] = time.time()

    # Return the seconds remaining on a provider's cooldown (0.0 when it is not cooling down).
    def cooldown_remaining(self, name: str) -> float:
        return max(0.0, self._cooldown_until.get(name, 0.0) - time.time())

    # Record a provider call outcome and (de)escalate its cooldown.
    def record(
        self, name: str, *, ok: bool, status_code: int | None = None, error: str = ""
    ) -> None:
        antibot = status_code in ANTIBOT_STATUS
        hard = antibot or (not ok and not _is_empty_ok(status_code, error))
        with self._lock:
            # Not a hard failure: clear the streak and any active cooldown.
            if not hard:
                self._fails[name] = 0
                self._cooldown_until.pop(name, None)
                self._last_reason.pop(name, None)
                return

            # Hard failure: extend the streak and remember why.
            count = self._fails.get(name, 0) + 1
            self._fails[name] = count
            self._last_reason[name] = (
                f"antibot {status_code}" if antibot else (error or f"http {status_code}")
            )

            # Past the threshold, bench the provider with an exponentially growing cooldown.
            if count >= _FAIL_THRESHOLD:
                backoff = min(_COOLDOWN_MAX, _COOLDOWN_BASE * (2 ** (count - _FAIL_THRESHOLD)))
                self._cooldown_until[name] = time.time() + backoff

    # Return the providers currently cooling down with their failure count, remaining seconds and
    # last failure reason.
    def snapshot(self) -> dict[str, dict[str, object]]:
        now = time.time()
        return {
            name: {
                "fails": self._fails.get(name, 0),
                "cooldown_s": round(max(0.0, until - now), 1),
                "reason": self._last_reason.get(name, ""),
            }
            for name, until in self._cooldown_until.items()
            if until > now
        }

    # Test/maintenance hook: forget all cooldown state.
    def reset(self) -> None:
        with self._lock:
            self._fails.clear()
            self._cooldown_until.clear()
            self._last_reason.clear()


# A 200 that simply returned nothing is not a failure — never cools the provider down.
def _is_empty_ok(status_code: int | None, error: str) -> bool:
    return status_code == 200 and not error


# Process-wide health tracker shared by all academic searches.
_HEALTH = ProviderHealth()


# Return the shared process-wide ProviderHealth instance.
def get_provider_health() -> ProviderHealth:
    return _HEALTH
