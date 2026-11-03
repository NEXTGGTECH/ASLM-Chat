# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass


# A complete browser identity used to build convincing request metadata.
#
# primp_target uses a bare family alias ("chrome" / "edge" / "firefox") rather than
# a pinned version. Pinned versions are fragile: an unknown "chrome_NNN" silently
# falls back to a random fingerprint, and primp's Rust-side warning cannot be
# captured reliably to detect it. A bare alias always resolves to primp's bundled
# latest fingerprint for that family, so every request gets a coherent, real browser
# TLS fingerprint.
#
# family classifies the engine behind the UA (chrome, edge or firefox). It drives header
# coherence: only Chromium families (chrome/edge) emit the Sec-CH-UA client-hint headers —
# Firefox never sends them, so a Firefox UA paired with Sec-CH-UA is an instant bot tell.
# The sec_ch_ua fields are empty strings for non-Chromium families.
@dataclass(frozen=True)
class BrowserProfile:
    name: str
    family: str
    user_agent: str
    sec_ch_ua: str
    sec_ch_ua_platform: str
    sec_ch_ua_mobile: str
    accept_language: str
    primp_target: str
    primp_os: str
    accept: str = (
        "text/html,application/xhtml+xml,application/xml;"
        "q=0.9,image/avif,image/webp,image/apng,*/*;"
        "q=0.8,application/signed-exchange;v=b3;q=0.7"
    )

    # Return True for the Chromium families (chrome and edge), which send client-hint headers.
    @property
    def is_chromium(self) -> bool:
        return self.family in ("chrome", "edge")


# Firefox's navigation Accept header differs from Chromium's (no apng / signed-exchange).
_FIREFOX_ACCEPT = (
    "text/html,application/xhtml+xml,application/xml;q=0.9,"
    "image/avif,image/webp,image/png,image/svg+xml,*/*;q=0.8"
)


# All known realistic browser profiles. for_engine() assigns one deterministically
# per engine; pick() draws at random (kept for the benchmark baseline only).
_PROFILES: list[BrowserProfile] = [
    BrowserProfile(
        name="chrome_131_win",
        family="chrome",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.6778.86 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        sec_ch_ua_platform='"Windows"',
        sec_ch_ua_mobile="?0",
        accept_language="en-US,en;q=0.9",
        primp_target="chrome",
        primp_os="windows",
    ),
    BrowserProfile(
        name="chrome_130_win",
        family="chrome",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.6723.117 Safari/537.36"
        ),
        sec_ch_ua='"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
        sec_ch_ua_platform='"Windows"',
        sec_ch_ua_mobile="?0",
        accept_language="en-US,en;q=0.9",
        primp_target="chrome",
        primp_os="windows",
    ),
    BrowserProfile(
        name="chrome_131_mac",
        family="chrome",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.6778.86 Safari/537.36"
        ),
        sec_ch_ua='"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        sec_ch_ua_platform='"macOS"',
        sec_ch_ua_mobile="?0",
        accept_language="en-US,en;q=0.9",
        primp_target="chrome",
        primp_os="macos",
    ),
    BrowserProfile(
        name="chrome_130_mac",
        family="chrome",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.6723.117 Safari/537.36"
        ),
        sec_ch_ua='"Chromium";v="130", "Google Chrome";v="130", "Not?A_Brand";v="99"',
        sec_ch_ua_platform='"macOS"',
        sec_ch_ua_mobile="?0",
        accept_language="en-US,en;q=0.9",
        primp_target="chrome",
        primp_os="macos",
    ),
    BrowserProfile(
        name="edge_131_win",
        family="edge",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36 Edg/131.0.2903.51"
        ),
        sec_ch_ua='"Microsoft Edge";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
        sec_ch_ua_platform='"Windows"',
        sec_ch_ua_mobile="?0",
        accept_language="en-US,en;q=0.9",
        primp_target="edge",
        primp_os="windows",
    ),
    # Firefox family. No Sec-CH-UA: Firefox does not implement client hints. build_nav_headers
    # omits the Sec-CH-UA* trio for these so the header set stays coherent with the UA.
    BrowserProfile(
        name="firefox_133_win",
        family="firefox",
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:133.0) "
            "Gecko/20100101 Firefox/133.0"
        ),
        sec_ch_ua="",
        sec_ch_ua_platform="",
        sec_ch_ua_mobile="",
        accept_language="en-US,en;q=0.5",
        primp_target="firefox",
        primp_os="windows",
        accept=_FIREFOX_ACCEPT,
    ),
    BrowserProfile(
        name="firefox_133_mac",
        family="firefox",
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:133.0) "
            "Gecko/20100101 Firefox/133.0"
        ),
        sec_ch_ua="",
        sec_ch_ua_platform="",
        sec_ch_ua_mobile="",
        accept_language="en-US,en;q=0.5",
        primp_target="firefox",
        primp_os="macos",
        accept=_FIREFOX_ACCEPT,
    ),
]


# Pick a random browser profile from the pool.
#
# Kept for the benchmark baseline only. Production engines use for_engine() so each
# engine keeps one stable identity (cookie continuity, no rotate-under-pressure tell).
def pick() -> BrowserProfile:
    return random.choice(_PROFILES)


# Deterministically map an engine key to a fixed profile.
#
# The same key always yields the same identity, so an engine looks like one returning
# browser rather than a fleet of rotating ones. generation lets a burned identity be
# rotated wholesale to a different fixed profile (the captcha-burn recycle policy) —
# bumping it deterministically re-seeds to another stable profile, never to per-request
# randomness.
def for_engine(key: str, *, generation: int = 0) -> BrowserProfile:
    digest = hashlib.sha1(f"{key}#{generation}".encode()).hexdigest()
    return _PROFILES[int(digest, 16) % len(_PROFILES)]


# Default country subtag per language, for building a realistic region tag when the
# caller has no explicit country. Covers the scripts search.quality detects; uppercasing
# the language (ru->ru-RU) is wrong for most of them (ja-JA / ar-AR are not real tags).
_LANG_COUNTRY = {
    "ru": "RU", "de": "DE", "ar": "SA", "he": "IL", "ja": "JP",
    "zh": "CN", "ko": "KR", "th": "TH", "hi": "IN", "el": "GR",
    "fr": "FR", "es": "ES", "it": "IT", "pt": "BR", "uk": "UA",
}


# Build an Accept-Language header coherent with the query's language.
#
# A real user searching Cyrillic/CJK/etc. text has that language in Accept-Language;
# sending "en-US" for a Russian query is an anti-signal. The query language is already
# inferred upstream (search.quality.infer_query_language drives region routing), so the
# engine only needs to turn its region's language into a header. English returns "" so
# callers keep the profile's own family-styled default (Firefox uses q=0.5, not q=0.9).
def accept_language_for(language: str, country: str = "") -> str:
    lang = (language or "en").lower()
    if lang == "en":
        return ""
    region = (country or _LANG_COUNTRY.get(lang, lang)).upper()
    return f"{lang}-{region},{lang};q=0.9,en-US;q=0.8,en;q=0.7"


# Build the full header set for a navigation request using the given profile.
# sec_fetch_site should be "same-origin" when the referer matches the host.
def build_nav_headers(
    profile: BrowserProfile,
    *,
    referer: str | None = None,
    sec_fetch_site: str = "none",
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    headers: dict[str, str] = {
        "User-Agent": profile.user_agent,
        "Accept": profile.accept,
        "Accept-Language": profile.accept_language,
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": sec_fetch_site,
        "Sec-Fetch-User": "?1",
        "Upgrade-Insecure-Requests": "1",
        "Cache-Control": "max-age=0",
    }

    # Client hints are Chromium-only; emitting them under a Firefox UA is a bot tell.
    if profile.is_chromium:
        headers["Sec-CH-UA"] = profile.sec_ch_ua
        headers["Sec-CH-UA-Mobile"] = profile.sec_ch_ua_mobile
        headers["Sec-CH-UA-Platform"] = profile.sec_ch_ua_platform

    # Optional referer and caller overrides go last, so extra wins over the defaults.
    if referer:
        headers["Referer"] = referer
    if extra:
        headers.update(extra)
    return headers
