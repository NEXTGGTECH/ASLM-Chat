# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from dataclasses import dataclass

# Canonical fetch-method identifiers recorded in runtime profiles. Generic read_page
# uses the HTTP methods plus the warm browser; terminal custom-domain handlers record
# their own names.
METHOD_HTTPX = "httpx"
METHOD_CURL_CFFI = "curl_cffi"
# Warm persistent stealth-browser (cloakbrowser daemon) — the only browser backend.
# The cost-aware method selector ranks it last (above nothing) as the heavy fallback.
METHOD_BROWSER = "browser"


# Outcome of one fetch attempt against a URL, recorded into the runtime profile store.
# quality is the extracted-markdown length used as a coarse text-quality signal; it is
# 0 until normalisation runs (a fetch that returns HTML but extracts nothing is "empty").
@dataclass(slots=True)
class FetchAttempt:
    method: str
    user_agent: str = ""
    status: int = 0
    fetch_ms: float = 0.0
    parse_ms: float = 0.0
    quality: int = 0
    success: bool = False
    blocked: bool = False
    timed_out: bool = False
    empty: bool = False


# Learned recommendation for how to fetch a domain. avoid marks a domain whose cheapest
# working method is still expensive/unreliable, so a fast path should skip it entirely.
@dataclass(slots=True)
class ProfileHint:
    method: str
    user_agent: str = ""
    expected_fetch_ms: float = 0.0
    expected_quality: float = 0.0
    confidence: float = 0.0
    avoid: bool = False


# Hard, hand-curated knowledge for a domain that must not be discovered at runtime
# (e.g. "this host only yields content through a real browser"). Seeds runtime profiles
# and overrides them while confidence is still low.
@dataclass(frozen=True, slots=True)
class DomainOverride:
    # Fetch method forced for the domain (e.g. METHOD_BROWSER).
    required_method: str = ""
    # Special page parsing mode (e.g. "nextjs_rsc").
    parsing_mode: str = ""
    note: str = ""


# Point-in-time view of learned domain trust, loaded once per search so triage scoring
# stays I/O-free. penalties holds POSITIVE magnitudes (triage subtracts them, already
# capped when the snapshot is built); proven lists domains with enough recent successful parses to be
# exempt from the unproven-TLD strict parse bar.
@dataclass(frozen=True, slots=True)
class ReputationSnapshot:
    penalties: dict[str, float]
    proven: frozenset[str]
