# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

# Registry seed (`profiles/seeds/academic_registry.json`): the source of truth for which
# scholarly aggregators exist and how to reach each.
_SEED_PATH = Path(__file__).resolve().parents[2] / "profiles" / "seeds" / "academic_registry.json"


# One registry row: a scholarly domain plus how to reach it (method = json_api / http / browser,
# rps/burst pacing, the open REST endpoint in `json_api_hint`, and field notes).
@dataclass(frozen=True, slots=True)
class AcademicDomain:
    pattern: str
    # Politeness tier: friendly / moderate / hardened.
    tier: str
    # How the domain is reached: json_api / http / browser.
    method: str
    rps: float
    burst: int
    json_api_hint: str
    text_search_capable: bool
    aliases: tuple[str, ...]
    notes: str

    # The registrable host the records of this domain live under (used as source_domain).
    @property
    def host(self) -> str:
        return self.pattern


# Build an AcademicDomain from a raw seed entry, normalizing text and applying defaults.
def _coerce(entry: dict[str, Any]) -> AcademicDomain:
    return AcademicDomain(
        pattern=str(entry.get("pattern") or "").strip().lower(),
        tier=str(entry.get("tier") or "moderate").strip().lower(),
        method=str(entry.get("method") or "http").strip().lower(),
        rps=float(entry.get("rps") or 1.0),
        burst=int(entry.get("burst") or 1),
        json_api_hint=str(entry.get("json_api_hint") or "").strip(),
        text_search_capable=bool(entry.get("text_search_capable", True)),
        aliases=tuple(str(a).strip().lower() for a in entry.get("aliases", []) if a),
        notes=str(entry.get("notes") or "").strip(),
    )


# Load and cache every registry row from the seed; an unreadable or invalid seed gives an
# empty registry. The engine builds its provider set from the `json_api` rows (open, keyless
# REST APIs); `http`/`browser` tiers (arxiv html, pubmed, semanticscholar) are described in the
# seed but reached through their own clients or deferred to the browser path.
@lru_cache(maxsize=1)
def load_registry() -> tuple[AcademicDomain, ...]:
    try:
        raw = json.loads(_SEED_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # A missing or malformed seed leaves the registry empty rather than failing import.
        return ()
    return tuple(_coerce(e) for e in raw.get("domains", []) if isinstance(e, dict) and e.get("pattern"))


# Registry rows reachable by a keyless REST endpoint and usable for free-text search.
def json_api_domains() -> tuple[AcademicDomain, ...]:
    return tuple(
        d for d in load_registry()
        if d.method == "json_api" and d.json_api_hint and d.text_search_capable
    )


# Look up a registry row by host (or alias) with an exact match on pattern/alias, ignoring a
# leading "www.".
def domain_for(host: str) -> AcademicDomain | None:
    h = (host or "").strip().lower().removeprefix("www.")
    for d in load_registry():
        if h == d.pattern or h in d.aliases:
            return d
    return None
