# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import StrEnum

# Cooldowns in seconds for the two trigger classes: a short one after a degradation (success but thin
# results or a latency spike) and a long one after an error (HTTP block, timeout, blocked or changed
# parse). Config-level constants by design: tune here, not inline.
DEGRADATION_COOLDOWN = 30.0
ERROR_COOLDOWN = 300.0

# Stage C — polite per-engine min-interval between fires (seconds), applied on top of the
# breaker. Even a healthy engine is not hit faster than this, with jitter so the cadence is
# not a detectable metronome. Brave is the most rate-sensitive (429-prone, §1); tolerant
# engines (DDG) pace at 0. The gate is cross-search: it skips an engine that fired too
# recently in a prior search, and a substitute of the same tier is used instead.
_ENGINE_MIN_INTERVAL = {
    "brave": 6.0,
    "google": 2.0,
    "qwant": 2.0,
    "yandex": 1.5,
    "startpage": 1.5,
    "duckduckgo": 0.0,
    "legacy_yandex": 1.5,
}
# Jitter band multiplied into the base interval, so two fires are never exactly N seconds apart.
_PACE_JITTER = (0.8, 1.25)
# Backoff multiplier applied when a half-open probe fails again, and its cap.
BACKOFF_FACTOR = 2.0
MAX_COOLDOWN = 1800.0
# A half-open probe is admitted but its outcome (record()) can be lost — e.g. the
# search deadline cancels the producer before the engine's status event is consumed.
# An in-flight probe older than this is treated as abandoned and a fresh one is
# admitted, so a single lost outcome cannot lock the engine out for the process life.
PROBE_TIMEOUT = 60.0

# EWMA smoothing for fetch latency.
_EWMA_ALPHA = 0.30
# A successful call this much slower than the engine's EWMA counts as degradation.
_LATENCY_SPIKE_FACTOR = 3.0
# Successful parses with fewer results than this count as degradation.
_THIN_RESULTS_FLOOR = 1

# Parse statuses that count as hard errors for the breaker.
_ERROR_STATUSES = frozenset({"blocked", "timeout", "error", "changed"})


# Circuit breaker states: CLOSED (healthy), OPEN (cooling down) and HALF_OPEN (one probe allowed).
class BreakerState(StrEnum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


# Mutable health record of one engine: breaker state, cooldown timing, counters and pacing gates.
# open_until is the clock time the OPEN state ends, cooldown is the current cooldown length (it
# grows with backoff) and ewma_fetch_ms is the smoothed fetch latency in milliseconds.
@dataclass(slots=True)
class EngineHealth:
    state: BreakerState = BreakerState.CLOSED
    open_until: float = 0.0
    cooldown: float = 0.0
    ewma_fetch_ms: float = 0.0
    successes: int = 0
    errors: int = 0
    degradations: int = 0
    last_status: str = ""
    # True while a half-open probe is in flight (only one probe at a time).
    probe_inflight: bool = field(default=False, repr=False)
    # When the in-flight probe was admitted, so an abandoned probe can be expired.
    probe_started: float = field(default=0.0, repr=False)
    # Earliest time this engine may be fired again (Stage C jittered pacing).
    paced_until: float = field(default=0.0, repr=False)


# In-memory, per-process health registry and circuit breaker for SERP engines.
# The breaker goes CLOSED -> OPEN (cooling down) -> HALF_OPEN (one probe allowed) -> CLOSED on
# success, or back to OPEN with exponential backoff on failure. Errors (HTTP 4xx/5xx block,
# timeout, BLOCKED/CHANGED parse) get the long cooldown; degradation (success but thin results or
# a latency spike) gets the short one. Latency is tracked as an EWMA with a fixed alpha, which is
# cheap and good enough to spot a p95-style spike without storing samples.
class EngineHealthTracker:
    # Create an empty registry; the clock is injectable for tests.
    def __init__(self, *, clock=time.monotonic) -> None:
        self._clock = clock
        self._engines: dict[str, EngineHealth] = {}

    # Get (or create) the health record for an engine.
    def _health(self, engine: str) -> EngineHealth:
        health = self._engines.get(engine)
        if health is None:
            health = EngineHealth()
            self._engines[engine] = health
        return health

    # Return True when the engine may be called right now. Layers Stage C pacing over the
    # breaker: an engine fired too recently (within its jittered min-interval) is held
    # back even when healthy. A HALF_OPEN engine allows exactly one in-flight probe.
    # Callers that get False should use a substitute of the same tier.
    def allow(self, engine: str) -> bool:
        health = self._health(engine)
        now = self._clock()
        if now < health.paced_until:
            return False
        if not self._breaker_admits(health, now):
            return False
        self._note_fired(engine, health, now)
        return True

    # Return True only when the engine's breaker is solidly CLOSED (healthy). Pure read — no
    # pacing, no probe consumption, no state transition — so selection code can ask "is
    # this engine trustworthy right now?" without side effects. An OPEN breaker (cooling)
    # OR a HALF_OPEN probe (recovery unproven) both read False: in either case a same-family
    # substitute should ride along so a failed/blocked slot doesn't cost the search its
    # family result. A never-seen engine is CLOSED, hence healthy.
    def is_healthy(self, engine: str) -> bool:
        health = self._engines.get(engine)
        if health is None:
            return True
        if health.state == BreakerState.OPEN and self._clock() >= health.open_until:
            # Cooldown elapsed but not yet probed — the next allow() will probe, so it is
            # not solidly healthy. Treat as unhealthy for substitution purposes.
            return False
        return health.state == BreakerState.CLOSED

    # Apply the circuit-breaker half of allow(): CLOSED passes, OPEN waits then probes once.
    def _breaker_admits(self, health: EngineHealth, now: float) -> bool:
        if health.state == BreakerState.CLOSED:
            return True
        if health.state == BreakerState.OPEN:
            if now < health.open_until:
                return False
            health.state = BreakerState.HALF_OPEN
            health.probe_inflight = False

        # HALF_OPEN: admit a single probe. An in-flight probe whose outcome was
        # never recorded (deadline lost the engine event) is expired so it cannot
        # wedge the engine shut forever.
        if health.probe_inflight and (now - health.probe_started) <= PROBE_TIMEOUT:
            return False
        health.probe_inflight = True
        health.probe_started = now
        return True

    # Record that an engine is being fired now and set its next jittered pace gate.
    def _note_fired(self, engine: str, health: EngineHealth, now: float) -> None:
        base = _ENGINE_MIN_INTERVAL.get(engine, 0.0)
        if base > 0.0:
            health.paced_until = now + base * random.uniform(*_PACE_JITTER)

    # Record the outcome of one engine call (its serp_api payload fields).
    def record(self, engine: str, *, status: str, fetch_ms: float, results: int) -> None:
        health = self._health(engine)
        health.last_status = status
        is_error = status in _ERROR_STATUSES

        # Classify the call: a latency spike against the EWMA or a thin result set is a degradation.
        spike = (
            not is_error
            and health.ewma_fetch_ms > 0
            and fetch_ms > health.ewma_fetch_ms * _LATENCY_SPIKE_FACTOR
        )
        thin = not is_error and results < _THIN_RESULTS_FLOOR
        is_degraded = spike or thin

        # Fold the latency of successful calls into the EWMA (the first sample seeds it).
        if not is_error and fetch_ms > 0:
            if health.ewma_fetch_ms <= 0:
                health.ewma_fetch_ms = fetch_ms
            else:
                health.ewma_fetch_ms += _EWMA_ALPHA * (fetch_ms - health.ewma_fetch_ms)

        # Trip the breaker with the cooldown of the trigger class, or close it on a clean success.
        if is_error:
            health.errors += 1
            self._trip(health, ERROR_COOLDOWN)
        elif is_degraded:
            health.degradations += 1
            self._trip(health, DEGRADATION_COOLDOWN)
        else:
            health.successes += 1

            # Success closes the breaker and resets backoff.
            health.state = BreakerState.CLOSED
            health.cooldown = 0.0
            health.probe_inflight = False

    # Open the breaker; a failed half-open probe grows the cooldown exponentially.
    def _trip(self, health: EngineHealth, base_cooldown: float) -> None:
        if health.state == BreakerState.HALF_OPEN and health.cooldown > 0:
            cooldown = min(MAX_COOLDOWN, health.cooldown * BACKOFF_FACTOR)
        else:
            cooldown = base_cooldown
        health.cooldown = cooldown
        health.open_until = self._clock() + cooldown
        health.state = BreakerState.OPEN
        health.probe_inflight = False

    # Return a per-engine snapshot of breaker state, remaining cooldowns and counters for diagnostics payloads.
    def snapshot(self) -> dict[str, dict[str, object]]:
        now = self._clock()
        out: dict[str, dict[str, object]] = {}
        for engine, health in self._engines.items():
            out[engine] = {
                "state": health.state.value,
                "cooldown_remaining_s": round(max(0.0, health.open_until - now), 1),
                "paced_remaining_s": round(max(0.0, health.paced_until - now), 1),
                "ewma_fetch_ms": round(health.ewma_fetch_ms, 1),
                "successes": health.successes,
                "errors": health.errors,
                "degradations": health.degradations,
                "last_status": health.last_status,
            }
        return out


# Process-wide tracker shared across searches (engine health is global state).
_shared_tracker: EngineHealthTracker | None = None


# Return the process-wide tracker, creating it on first use.
def get_health_tracker() -> EngineHealthTracker:
    global _shared_tracker
    if _shared_tracker is None:
        _shared_tracker = EngineHealthTracker()
    return _shared_tracker
