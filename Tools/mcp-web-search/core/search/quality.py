# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import datetime as _dt
import re
from urllib.parse import urlparse

# Model-free SERP quality signals: everything in this module is deterministic, allocation-light and
# budgeted for sub-millisecond per-source evaluation, so triage can run inline with the live result
# stream. There is deliberately NO lexical term-matching scorer here. The ranking is consensus +
# position + earned signals only: flat term matching paid SEO titles for filler words and was
# removed after a measured ablation (2026-07-05) showed removal beat every weighting tried. Chunk
# selection still matches terms, but that lives in core.extract.scoring and shapes model_context,
# never source order.
# The three sets below mark hub/garbage pages: URL path segments and title phrases of listing pages,
# and title patterns of non-content pages (login walls, error pages).
_HUB_URL_SEGMENTS = frozenset({
    "category", "categories", "tag", "tags", "topic", "topics",
    "theme", "themes", "rubric", "rubrics", "section", "sections",
    "label", "labels", "archive", "archives", "feed", "rss",
    "search", "results", "page", "index", "catalog",
})
_HUB_TITLE_PHRASES = (
    "all news", "all articles", "all posts", "news feed", "tag page",
    "category page", "topic page", "browse", "archive",
    "все новости", "последние новости", "все статьи",
)
_SKIP_TITLE_PATTERNS = frozenset({
    "login", "log in", "sign up", "signup", "sign in", "register",
    "create account", "subscribe", "404", "403", "not found",
    "access denied", "page not found", "permission denied",
})


# Hub and garbage detection

# Return a penalty in [0,1] for hub/listing pages that parse into link soup, not content.
def hub_penalty(url: str, title: str, snippet: str) -> float:
    penalty = 0.0
    path = (urlparse(url).path or "").lower().strip("/")
    if set(path.split("/")) & _HUB_URL_SEGMENTS:
        penalty += 0.5
    if not path or path in {"index", "index.html", "index.php"}:
        penalty += 0.3
    title_l = (title or "").lower()
    if any(phrase in title_l for phrase in _HUB_TITLE_PHRASES):
        penalty += 0.4
    snippet = snippet or ""
    if snippet.count(" | ") >= 4 or snippet.count(" · ") >= 3 or snippet.count(" • ") >= 3:
        penalty += 0.25
    return min(penalty, 1.0)


# Return True for titles that scream non-content (login walls, error pages).
def is_skip_title(title: str) -> bool:
    title_l = (title or "").lower()
    return any(pattern in title_l for pattern in _SKIP_TITLE_PATTERNS)


# SEO-slug penalty (identity-blind)

# No domain favouritism: the ranker must not carry a list of "good" hosts — that would
# reintroduce the curated trust registry consensus was built to replace, and bias the
# system toward whoever the author hand-picked. The only domain-shaped signal we allow
# is a penalty on the *structure* of a URL, never on its identity: year-stuffed,
# heavily hyphenated slugs are a reliable SEO-content-farm tell regardless of who owns
# the domain. Authority itself comes from cross-engine consensus (see triage), which is
# earned from the live result stream, not declared here.
# _SEO_SLUG_RE finds a standalone four-digit year (1900-2099) inside a slug.
_SEO_SLUG_RE = re.compile(r"\b(19|20)\d{2}\b")


# Return a non-positive penalty for SEO-farm-shaped slugs. Identity-blind: keys only off the URL
# slug's shape (hyphen density + a year), so it generalises to any host and plays no
# favourites. Deliberately gentle — it nudges, never filters.
def seo_slug_penalty(url: str) -> float:
    last = (urlparse(url).path or "").rstrip("/").rsplit("/", 1)[-1].lower()
    if last.count("-") >= 6 and _SEO_SLUG_RE.search(last):
        return -0.06
    return 0.0


# TLD tiers and transport security (identity-blind, structural)

# IMPORTANT DISTINGUISHING SIGNAL — the TLD tier.
# The TLD a domain sits under is a cheap, identity-blind proxy for one hard-to-fake fact:
# was this zone reachable BEFORE the era when a throwaway domain cost a dollar? The line
# is drawn at ICANN's 2012 new-gTLD program, not at anyone's taste. Everything a
# registrant could obtain before that expansion — the pre-2012 gTLDs below, plus every
# ccTLD (two-letter, or an IDN ccTLD in xn--* punycode) — is "established": it earns a
# NEUTRAL prior, never a bonus. Everything the expansion minted (.dev/.xyz/.top/.online/…)
# is "unproven": also never penalised by name — that would rebuild the curated trust
# registry consensus was built to delete — but held to a slightly stricter parse bar
# until the domain earns positive history in the runtime reputation store.
#
# Why this is the right axis and not a blocklist: content farms and phishing kits cluster
# on the cheapest, most permissive new gTLDs precisely because identity there is
# disposable; a .com/.co.uk carries no such prior either way, so we neither reward nor
# punish it. The tier only decides how much proof we ask of a stranger, never whether it
# is "good" or "bad" — that verdict is always earned downstream (consensus + reputation).
#
# Completeness (verified 2026-07 against ICANN/Wikipedia): this is the FULL pre-2012 set —
# 1985 RFC-920 originals (com/edu/gov/mil/org/net) + int (1988) + arpa (infrastructure),
# the 2000 round (aero/biz/coop/info/museum/name/pro), the 2003–04 sponsored round
# (asia/cat/jobs/mobi/tel/travel), xxx (2011) and post (UPU, 2012). Anything not here
# is expansion-era by construction, so the frozenset never needs growing — new gTLDs are
# supposed to fall through to the unproven tier.
_LEGACY_GTLDS = frozenset({
    "com", "net", "org", "edu", "gov", "mil", "int", "arpa",
    "info", "biz", "name", "pro", "aero", "asia", "cat", "coop",
    "jobs", "mobi", "museum", "post", "tel", "travel", "xxx",
})


# Return True when a host sits under an established TLD (legacy gTLD or any ccTLD). ccTLDs pass
# on the two-letter rule, so multi-label suffixes (.co.uk, .com.br) qualify via their
# real TLD (.uk/.br) with no separate suffix list to maintain.
def is_established_tld(host_or_domain: str) -> bool:
    tld = (host_or_domain or "").lower().rstrip(".").rsplit(".", 1)[-1]
    return len(tld) == 2 or tld.startswith("xn--") or tld in _LEGACY_GTLDS


# Return a non-positive penalty for plain-http SERP URLs — the only transport-security signal
# available with zero I/O at triage time. Gentle: legitimate old mirrors still exist.
def insecure_scheme_penalty(url: str) -> float:
    return -0.08 if (url or "").lower().startswith("http://") else 0.0


# Bad-site tells (identity-blind, zero-I/O)

# Structural marks that phishing/farm sites wear and legitimate sites rarely do. Same
# contract as seo_slug_penalty: shape only, never names; each tell is gentle and the
# pack is capped so stacked tells demote a source, never execute it — SKIP still comes
# only from irrelevance, and consensus votes can outweigh the whole pack.
# The cap is the largest total penalty of the pack; the patterns match an IPv4 literal host and a
# four-digit year (1900-2099) inside a domain label.
_SUSPICIOUS_PACK_CAP = 0.20
_IP_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_DOMAIN_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


# Return the combined non-positive penalty for structural unsafe-site tells in a SERP URL.
def suspicious_url_penalty(url: str) -> float:
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        port = parsed.port
    except ValueError:
        # An unparseable URL is its own tell.
        return -_SUSPICIOUS_PACK_CAP
    if not host:
        return 0.0
    penalty = 0.0

    # IP-literal host: organic results virtually never point at bare addresses.
    if _IP_HOST_RE.fullmatch(host) or host.startswith("["):
        penalty += 0.15

    # Explicit non-default port — SERP-worthy content does not live on :8080.
    if port not in (None, 80, 443):
        penalty += 0.12

    labels = host.split(".")

    # A gTLD-shaped label buried in the subdomain chain ("github.com.evil.xyz") is a
    # phishing costume. labels[:-2] keeps ccTLD second-level registries (example.com.br,
    # example.co.uk) out of the blast radius.
    if any(label in _LEGACY_GTLDS for label in labels[:-2]):
        penalty += 0.15

    # Hyphen-dense or year-stamped domain labels: the registered-name cousins of the
    # SEO slug tell ("best-ai-coding-tools-2026.com", "top10vpn2024.net"). Deliberately
    # feather-light: name shape correlates with farms but convicts nobody, and stacking
    # heavier name penalties would just monopolise the SERP for incumbent big domains.
    # Safety tells above (IP/port/costume) stay stronger — those are about deception,
    # not taste in naming.
    if any(label.count("-") >= 3 for label in labels[:-1]):
        penalty += 0.04
    if any(_DOMAIN_YEAR_RE.search(label) for label in labels[:-1]):
        penalty += 0.03

    return -min(penalty, _SUSPICIOUS_PACK_CAP)


# Return a non-positive penalty for the exact-match-domain tell (Google EMD-update heritage): a
# registered name that IS the query ("claudecode.pro" for "claude code") is usually a squatter
# riding the phrase, not the subject itself. Deliberately the gentlest tell — legitimate exact-match
# domains exist, and they can win the margin back with one consensus vote. Kept at a
# whisper (-0.03) by explicit policy: name-shape penalties must nudge the margin, not
# gate the market for established domains.
def emd_penalty(url: str, terms: tuple[str, ...] | list[str]) -> float:
    if len(terms) < 2:
        return 0.0
    try:
        host = (urlparse(url).hostname or "").lower()
    except ValueError:
        return 0.0
    labels = host.removeprefix("www.").split(".")
    if len(labels) < 2:
        return 0.0
    name = labels[0].replace("-", "")
    if len(name) < 6:
        return 0.0

    # The registered name must be fully spelled by ≥2 consecutive query terms.
    for start in range(len(terms) - 1):
        candidate = terms[start]
        for term in terms[start + 1:]:
            candidate += term
            if len(candidate) > len(name):
                break
            if candidate == name:
                return -0.03
    return 0.0


# Dates (policy: soft signal only, never a hard filter)

# The date signal keys off an ACTUAL date the page carries — an engine-emitted snippet
# date ("Jul 3, 2026 —", "3 июл. 2026 г. —", "03.07.2026", "2 days ago") or, post-parse,
# the page's own published_time. A bare year token in a title never counts: rewarding
# "…Pricing 2026 - Costs & Providers" for containing "2026" is exactly the SEO-farm
# title-stuffing move, and paying for it is how farms outranked primary sources. A real
# date needs a month or a day next to the year, which titles do not stuff.
# _YEAR_RE finds a standalone four-digit year (1900-2099), used for years written in a query.
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")

# Month-name prefixes across the languages the engines actually localise snippets
# into. This is DATA, not a registry: month names are closed, stable facts of a
# language, and the whitelist is what keeps the date anchor ungameable — any-word
# patterns would let "Top 10 Tools 2026" read as a date. Prefixes are matched
# longest-first with a letters-only tail, so inflected forms (июля, juillet,
# października, липня, července) resolve through the same key.
_MONTH_BY_PREFIX: dict[str, int] = {
    # en + de/nl/sv/da/no shared Latin abbreviations
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
    "mär": 3, "maa": 3, "mei": 5, "maj": 5, "mai": 5, "okt": 10, "dez": 12, "des": 12,
    # fr (juin/juillet need 4 chars to split)
    "fév": 2, "fev": 2, "avr": 4, "juin": 6, "juil": 7, "aoû": 8, "aou": 8, "déc": 12,
    # es
    "ene": 1, "abr": 4, "ago": 8, "dic": 12,
    # it
    "gen": 1, "mag": 5, "giu": 6, "lug": 7, "set": 9, "ott": 10,
    # pt
    "out": 10,
    # pl
    "sty": 1, "lut": 2, "kwi": 4, "cze": 6, "lip": 7, "sie": 8,
    "wrz": 9, "paź": 10, "paz": 10, "lis": 11, "gru": 12,
    # tr
    "oca": 1, "şub": 2, "sub": 2, "nis": 4, "haz": 6, "tem": 7,
    "ağu": 8, "agu": 8, "eyl": 9, "eki": 10, "kas": 11, "ara": 12,
    # cs — "led" (leden) and "pro" (prosinec) are deliberately absent: with a digit
    # nearby they collide with product-name English ("Top 10 LED 2026", "17 Pro 2026").
    "úno": 2, "uno": 2, "bře": 3, "bre": 3, "dub": 4, "kvě": 5, "kve": 5,
    "čer": 6, "cer": 6, "srp": 8, "zář": 9, "zar": 9, "říj": 10, "rij": 10,
    # ru
    "янв": 1, "фев": 2, "мар": 3, "апр": 4, "май": 5, "мая": 5, "июн": 6,
    "июл": 7, "авг": 8, "сен": 9, "окт": 10, "ноя": 11, "дек": 12,
    # uk
    "січ": 1, "лют": 2, "бер": 3, "кві": 4, "тра": 5, "чер": 6,
    "лип": 7, "сер": 8, "вер": 9, "жов": 10, "лис": 11, "гру": 12,
    # el (Ιούνιος/Ιούλιος need 4 chars to split)
    "ιαν": 1, "φεβ": 2, "μάρ": 3, "μαρ": 3, "απρ": 4, "μαΐ": 5, "μαι": 5,
    "ιούν": 6, "ιουν": 6, "ιούλ": 7, "ιουλ": 7, "αύγ": 8, "αυγ": 8,
    "σεπ": 9, "οκτ": 10, "νοέ": 11, "νοε": 11, "δεκ": 12,
    # ar (Gregorian set; Levantine kanun-style names are not covered)
    "ينا": 1, "فبر": 2, "مار": 3, "أبر": 4, "ابر": 4, "ماي": 5, "يون": 6,
    "يول": 7, "أغس": 8, "اغس": 8, "سبت": 9, "أكت": 10, "اكت": 10, "نوف": 11, "ديس": 12,
    # he
    "ינו": 1, "פבר": 2, "מרץ": 3, "אפר": 4, "מאי": 5, "יונ": 6,
    "יול": 7, "אוג": 8, "ספט": 9, "אוק": 10, "נוב": 11, "דצמ": 12,
    # hi
    "जनव": 1, "फ़र": 2, "फरव": 2, "मार्च": 3, "अप्र": 4, "मई": 5, "जून": 6,
    "जुल": 7, "अगस": 8, "सित": 9, "अक्ट": 10, "नवं": 11, "दिस": 12,
}
_MONTH_ALT = "|".join(sorted((re.escape(k) for k in _MONTH_BY_PREFIX), key=len, reverse=True))

# Bare month-year ("July 2026", "июль 2026") is the loosest form, so the month must be
# a COMPLETE month word — prefix+tail would let "Marketing 2026" read as March. Kept to
# en/ru where engines actually emit this shape; other languages require a day number.
_BARE_MONTH_WORDS = frozenset({
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "sept",
    "oct", "nov", "dec",
    "january", "february", "march", "april", "june", "july", "august",
    "september", "october", "november", "december",
    "янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек",
    "январь", "января", "февраль", "февраля", "март", "марта", "апрель", "апреля",
    "мая", "июнь", "июня", "июль", "июля", "август", "августа",
    "сентябрь", "сентября", "октябрь", "октября", "ноябрь", "ноября",
    "декабрь", "декабря",
})

# "2026-07-03" / "2026-07"
_ISO_DATE_RE = re.compile(r"\b((?:19|20)\d{2})-(\d{1,2})(?:-(\d{1,2}))?\b")
# "03.07.2026" (numeric day-first: ru/de/tr and most of Europe)
_NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.((?:19|20)\d{2})\b")
# "03/07/2026" / "7/3/2026" — day/month order is ambiguous; resolved below. A wrong
# guess is off by weeks at most, which the freshness half-life doesn't feel.
_SLASH_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/((?:19|20)\d{2})\b")
# "2026年7月3日" (ja/zh) / "2026년 7월 3일" (ko) — no month names needed at all.
_CJK_DATE_RE = re.compile(
    r"((?:19|20)\d{2})\s*[年년]\s*(\d{1,2})\s*[月월](?:\s*(\d{1,2})\s*[日일])?"
)
# "3 июл. 2026" / "Jul 3, 2026" / "3. Juli 2026" / "3 de julio de 2026" / "July 2026"
_TEXT_DATE_RE = re.compile(
    rf"(?:\b(\d{{1,2}})\.?(?:\s+de)?\s+)?\b({_MONTH_ALT})([^\W\d_]*)\.?,?(?:\s+de)?\s*"
    rf"(?:(\d{{1,2}})(?:st|nd|rd|th)?,?\s+)?((?:19|20)\d{{2}})\b",
    re.IGNORECASE,
)

# Relative dates. Suffix form: "2 days ago" / "5 часов назад"; prefix form:
# "vor 3 Tagen" / "il y a 2 jours" / "hace 2 días"; CJK: "3天前" / "3日前" / "3일 전".
_REL_UNIT_DAYS = {
    "min": 0.0, "hour": 0.04, "day": 1.0, "week": 7.0, "month": 30.0, "year": 365.0,
    "мин": 0.0, "час": 0.04, "дн": 1.0, "недел": 7.0, "месяц": 30.0, "год": 365.0, "лет": 365.0,
    "stunde": 0.04, "tag": 1.0, "woche": 7.0, "monat": 30.0, "jahr": 365.0,
    "heure": 0.04, "jour": 1.0, "semaine": 7.0, "mois": 30.0, "an": 365.0,
    "hora": 0.04, "día": 1.0, "dia": 1.0, "semana": 7.0, "mes": 30.0, "año": 365.0, "ano": 365.0,
}
_REL_UNIT_ALT = "|".join(sorted(_REL_UNIT_DAYS, key=len, reverse=True))
_REL_SUFFIX_RE = re.compile(
    rf"\b(\d+)\s*({_REL_UNIT_ALT})[^\W\d_]*\s+(?:ago|назад)", re.IGNORECASE
)
_REL_PREFIX_RE = re.compile(
    rf"\b(?:vor|hace|il y a)\s+(\d+)\s*({_REL_UNIT_ALT})[^\W\d_]*", re.IGNORECASE
)
_REL_CJK_RE = re.compile(
    r"(\d+)\s*(分鐘|分钟|分|時間|小時|小时|時|天|日|週間|週|周|个月|ヶ月|月|年|시간|분|일|주|개월|년)\s*(?:前|전)"
)
_REL_CJK_DAYS = {
    "分鐘": 0.0, "分钟": 0.0, "分": 0.0, "時間": 0.04, "小時": 0.04, "小时": 0.04, "時": 0.04,
    "天": 1.0, "日": 1.0, "週間": 7.0, "週": 7.0, "周": 7.0,
    "个月": 30.0, "ヶ月": 30.0, "月": 30.0, "年": 365.0,
    "시간": 0.04, "분": 0.0, "일": 1.0, "주": 7.0, "개월": 30.0, "년": 365.0,
}

# Freshness half-life: a fresh page earns the full date reward, ~9 months halves it.
_FRESHNESS_HALFLIFE_DAYS = 270.0


# Return the years the caller explicitly wrote into the query. Only these may influence
# scoring — hard date limits are forbidden by policy (they cut good content).
def query_years(query: str) -> list[str]:
    return _YEAR_RE.findall(query or "")


# Return the first real date found in free text as (year, month, day) with day defaulting to 1;
# None when the text carries no date-shaped marker (a bare year is NOT a date). Absolute formats
# are tried first (ISO, numeric, slash, CJK, month names), then relative ones ("2 days ago"),
# which are resolved against `now` (today by default).
def extract_date(text: str, *, now: _dt.date | None = None) -> tuple[int, int, int] | None:
    text = text or ""

    # ISO form: 2026-07-03 or 2026-07.
    if m := _ISO_DATE_RE.search(text):
        year, month = int(m.group(1)), int(m.group(2))
        day = int(m.group(3) or 1)
        if 1 <= month <= 12 and 1 <= day <= 31:
            return (year, month, day)

    # Numeric day-first form: 03.07.2026.
    if m := _NUMERIC_DATE_RE.search(text):
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return (year, month, day)

    # Slash form: 03/07/2026 or 7/3/2026.
    if m := _SLASH_DATE_RE.search(text):
        first, second, year = int(m.group(1)), int(m.group(2)), int(m.group(3))

        # >12 disambiguates; otherwise assume day-first (most of the world writes it so).
        day, month = (first, second) if first > 12 or second <= 12 else (second, first)
        if 1 <= month <= 12 and 1 <= day <= 31:
            return (year, month, day)

    # CJK form: 2026年7月3日 (ja/zh) or 2026년 7월 3일 (ko).
    if m := _CJK_DATE_RE.search(text):
        year, month = int(m.group(1)), int(m.group(2))
        day = int(m.group(3) or 1)
        if 1 <= month <= 12 and 1 <= day <= 31:
            return (year, month, day)

    # Month-name forms in many languages ("Jul 3, 2026", "3 июл. 2026"); needs a day or a bare
    # month-year with a complete month word.
    if m := _TEXT_DATE_RE.search(text):
        month = _MONTH_BY_PREFIX.get(m.group(2).lower(), 0)
        day_str = m.group(1) or m.group(4)
        year = int(m.group(5))

        # Bare month-year: only a complete en/ru month word qualifies (see above).
        bare_ok = f"{m.group(2)}{m.group(3)}".lower() in _BARE_MONTH_WORDS
        if month and (day_str or bare_ok):
            day = int(day_str or 1)
            if 1 <= day <= 31:
                return (year, month, day)

    # Relative forms ("2 days ago", "vor 3 Tagen", "3天前"): subtract the age from today.
    rel_days: float | None = None
    if m := _REL_SUFFIX_RE.search(text) or _REL_PREFIX_RE.search(text):
        unit = m.group(2).lower()
        per_unit = next((d for p, d in _REL_UNIT_DAYS.items() if unit.startswith(p)), None)
        if per_unit is not None:
            rel_days = int(m.group(1)) * per_unit
    elif m := _REL_CJK_RE.search(text):
        rel_days = int(m.group(1)) * _REL_CJK_DAYS[m.group(2)]
    if rel_days is not None:
        then = (now or _dt.date.today()) - _dt.timedelta(days=rel_days)
        return (then.year, then.month, then.day)
    return None


# Return the date reward in [-0.3, 1.0] for an extracted page date.
# Query anchors a year → alignment: the page's real date in a queried year is a full
# match, a different year a small negative (old year_match_score policy, now applied
# to an actual date instead of any year token in title/snippet). No year in the query →
# pure freshness with half-life decay. No date at all → 0: unknown is never punished.
def page_date_score(
    date: tuple[int, int, int] | None,
    years: list[str],
    *,
    now: _dt.date | None = None,
) -> float:
    if date is None:
        return 0.0
    year, month, day = date
    if years:
        return 1.0 if str(year) in years else -0.3
    today = now or _dt.date.today()
    try:
        age_days = max(0.0, (today - _dt.date(year, month, day)).days)
    except ValueError:
        return 0.0
    return 0.5 ** (age_days / _FRESHNESS_HALFLIFE_DAYS)


# Matches the "**Date:** …" line that page_normalizer emits in the markdown head when the page
# declares published_time.
_MD_DATE_LINE_RE = re.compile(r"^\*\*Date:\*\*\s*(.+)$", re.MULTILINE)


# Return the parsed page's own date from a normalized markdown head, or None. This is the
# strongest date evidence available — it replaces the snippet-derived estimate after a parse.
#
# Dates on or after the fetch day are discarded: sites whose templates render the
# CURRENT date into a <time>/meta tag (fossil, dashboards, some CMS footers) would
# otherwise read as eternally-fresh and collect the full freshness reward on every
# search. A genuinely just-published page loses at most one day of bonus — engine
# snippet dates still cover breaking news — while the dynamic-date noise is cut
# entirely. Future dates are junk by definition.
def markdown_meta_date(markdown: str, *, now: _dt.date | None = None) -> tuple[int, int, int] | None:
    m = _MD_DATE_LINE_RE.search((markdown or "")[:600])
    if not m:
        return None
    date = extract_date(m.group(1), now=now)
    if date is None:
        return None
    try:
        if _dt.date(*date) >= (now or _dt.date.today()):
            return None
    except ValueError:
        return None
    return date


# Query language

# Unicode script ranges as (language code, first code point, last code point): ru = Cyrillic,
# ar = Arabic, he = Hebrew, ja = Hiragana/Katakana (unambiguous Japanese), zh = CJK Unified (zh
# unless Japanese kana is present), ko = Hangul, th = Thai, hi = Devanagari, el = Greek. At least
# 15% of the characters in one script qualifies the language.
_SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("ru", 0x0400, 0x04FF),
    ("ar", 0x0600, 0x06FF),
    ("he", 0x0590, 0x05FF),
    ("ja", 0x3040, 0x30FF),
    ("zh", 0x4E00, 0x9FFF),
    ("ko", 0xAC00, 0xD7AF),
    ("th", 0x0E00, 0x0E7F),
    ("hi", 0x0900, 0x097F),
    ("el", 0x0370, 0x03FF),
)


# Detect the dominant script of a query → 2-letter language code (default "en"). Ported from the
# legacy services/web_search.py::infer_query_language.
# Used to route region/language per engine (e.g. Yandex for Cyrillic queries).
def infer_query_language(query: str) -> str:
    text = str(query or "")
    if not text:
        return "en"
    threshold = max(1, len(text)) * 0.15

    # Count the characters that fall into each script range.
    counts = {lang: 0 for lang, _, _ in _SCRIPT_RANGES}
    for ch in text:
        cp = ord(ch)
        for lang, lo, hi in _SCRIPT_RANGES:
            if lo <= cp <= hi:
                counts[lang] += 1
                break

    # Kana present: the CJK block is Japanese, not Chinese.
    if counts["ja"] > 0:
        counts["zh"] = 0

    # The first script (in table order) that reaches the threshold wins.
    for lang, _, _ in _SCRIPT_RANGES:
        if counts[lang] >= threshold:
            return lang
    return "en"
