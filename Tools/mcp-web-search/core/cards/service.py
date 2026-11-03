# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Awaitable, Callable
from urllib.parse import parse_qs, quote, unquote, urlsplit

import aiohttp

# Child of "mcp.server" so card events land in logs/mcp_trace.log (logging_setup).
logger = logging.getLogger("mcp.server.answer_cards")

# Largest provider response (in bytes) that is read before the lookup gives up.
_MAX_RESPONSE_BYTES = 1_048_576

# Wikimedia's API User-Agent policy requires a descriptive agent with contact
# info; bare product names and raw browser UAs both draw 403s on the REST API.
_USER_AGENT = "ASLM-Chat-WebSearch/0.2 (https://github.com/Di120078/ASLM-Chat)"
_WIKIDATA_API = "https://www.wikidata.org/w/api.php"

# Hostname -> Stack Exchange API site name for the sites that are not <name>.stackexchange.com.
_STACK_EXCHANGE_SITES = {
    "stackoverflow.com": "stackoverflow",
    "superuser.com": "superuser",
    "serverfault.com": "serverfault",
    "askubuntu.com": "askubuntu",
    "mathoverflow.net": "mathoverflow.net",
}

# URL path of a Stack Exchange question (captures the question id), and the first path segments on
# github.com that are site pages rather than repository owners.
_QUESTION_RE = re.compile(r"/(?:questions|q)/(\d+)(?:/|$)")
_GITHUB_RESERVED = frozenset(
    {
        "about", "apps", "collections", "customer-stories", "enterprise", "events",
        "explore", "features", "issues", "login", "marketplace", "new", "notifications",
        "orgs", "pricing", "pulls", "search", "settings", "site", "sponsors", "topics",
        "trending", "users",
    }
)

# URL paths of a PyPI project and an npm package (they capture the package name).
_PYPI_RE = re.compile(r"^/project/([A-Za-z0-9._-]+)/?")
_NPM_RE = re.compile(r"^/package/((?:@[A-Za-z0-9._-]+/)?[A-Za-z0-9._-]+)/?")

# Character limits of the text in a card: general card text and a Stack Exchange answer.
_CARD_TEXT_LIMIT = 900
_ANSWER_TEXT_LIMIT = 1200

# A card line whose words already appear in the source's own parsed content at or
# above this share is dropped: search already extracts StackExchange answers and
# GitHub/PyPI page text, so those cards would only repeat it.
_DUPLICATE_COVERAGE = 0.8

# Words of a text, as used to measure how much of a card line a source already contains.
_WORD_RE = re.compile(r"\w+", re.UNICODE)


# Helpers

# HTML parser that collects the non-blank text nodes of a markup fragment into parts.
class _PlainTextParser(HTMLParser):
    # Create the parser with character references converted and an empty part list.
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    # Keep every text node that is not only whitespace.
    def handle_data(self, data: str) -> None:
        if data.strip():
            self.parts.append(data)


# Strip markup from a value and return its text with collapsed whitespace, clipped to maximum
# characters. Malformed markup gives an empty string.
def _plain_text(value: object, *, maximum: int) -> str:
    if value is None:
        return ""
    parser = _PlainTextParser()
    try:
        parser.feed(str(value))
        parser.close()
    except Exception:  # noqa: BLE001 — malformed markup must never break a card
        return ""
    text = " ".join(" ".join(parser.parts).split())
    return _clip(text, maximum)


# Trim the text to at most maximum characters, cutting at a word boundary and adding an ellipsis.
def _clip(text: str, maximum: int) -> str:
    text = text.strip()
    if len(text) <= maximum:
        return text
    cut = text[:maximum].rsplit(" ", 1)[0].rstrip(" ,;:")
    return f"{cut}…"


# HTML parser that turns a StackExchange answer body into compact text that keeps its code: code
# blocks become fenced blocks, inline code gets backticks, and paragraphs, list items and headings
# start on their own lines.
class _AnswerHtmlParser(HTMLParser):
    # Create the parser with empty output and zero open pre/code elements.
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._pre = 0
        self._code = 0

    # Open code fences, inline code and line breaks for the supported tags.
    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "pre":
            self._pre += 1
            self.parts.append("\n```\n")
        elif tag == "code" and not self._pre:
            self._code += 1
            self.parts.append("`")
        elif tag in {"p", "li", "br", "h1", "h2", "h3", "blockquote"}:
            self.parts.append("\n")
            if tag == "li":
                self.parts.append("- ")

    # Close the code fence or inline code opened by the matching start tag.
    def handle_endtag(self, tag: str) -> None:
        if tag == "pre" and self._pre:
            self._pre -= 1
            self.parts.append("\n```\n")
        elif tag == "code" and self._code:
            self._code -= 1
            self.parts.append("`")

    # Add a text node: kept verbatim inside code blocks, otherwise with whitespace collapsed.
    def handle_data(self, data: str) -> None:
        self.parts.append(data if self._pre else re.sub(r"\s+", " ", data))


# Convert a StackExchange answer body (HTML) to compact text that keeps its code, clipped to
# maximum characters. Malformed markup gives an empty string.
def _answer_text(body: object, *, maximum: int) -> str:
    # Parse the markup.
    parser = _AnswerHtmlParser()
    try:
        parser.feed(str(body or ""))
        parser.close()
    except Exception:  # noqa: BLE001
        # Malformed markup must never break a card.
        return ""

    # Tidy the whitespace, then clip to the limit.
    text = "".join(parser.parts)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if len(text) <= maximum:
        return text
    clipped = text[:maximum]

    # Never leave a code fence open after clipping.
    if clipped.count("```") % 2:
        clipped += "\n```"
    return clipped.rstrip() + "\n…"


# Return the lowercase hostname of a URL without a leading "www." (empty when it cannot be parsed).
def _host(url: str) -> str:
    try:
        return (urlsplit(url).hostname or "").lower().removeprefix("www.")
    except ValueError:
        return ""


# Format a Unix timestamp as a UTC YYYY-MM-DD date (empty when it is not a valid timestamp).
def _iso_date(timestamp: object) -> str:
    try:
        return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


# Return the leading YYYY-MM-DD part of an ISO date or timestamp string (empty when it has none).
def _short_date(value: object) -> str:
    text = str(value or "")
    return text[:10] if re.match(r"\d{4}-\d{2}-\d{2}", text) else ""


# Tiny in-memory TTL cache; cards are cheap to rebuild after a restart. When full, the entry that
# expires first is evicted.
class _TtlCache:
    # Create an empty cache holding at most max_entries entries.
    def __init__(self, max_entries: int = 512) -> None:
        self._entries: dict[str, tuple[Any, float]] = {}
        self._max_entries = max_entries

    # Return the cached value, or None when the key is missing or expired (expired entries are dropped).
    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        value, expires = entry
        if expires <= time.monotonic():
            self._entries.pop(key, None)
            return None
        return value

    # Store a value that expires after ttl seconds (at least one second), evicting the entry that
    # expires first when the cache is full.
    def set(self, key: str, value: Any, *, ttl: float) -> None:
        if len(self._entries) >= self._max_entries:
            oldest = min(self._entries.items(), key=lambda item: item[1][1])[0]
            self._entries.pop(oldest, None)
        self._entries[key] = (value, time.monotonic() + max(1.0, ttl))


# Build the cache key of a card lookup (kind, language and target), versioned so a format change
# can invalidate old entries.
def _cache_key(kind: str, target: str, language: str = "") -> str:
    digest = hashlib.sha256(f"{kind}|{language}|{target}".encode("utf-8")).hexdigest()
    return f"answer-card:v2:{digest}"


# GET a JSON object from a provider. Return an empty dict on a non-200 status, a response over the
# size limit, invalid JSON, or a JSON value that is not an object.
async def _fetch_json(
    session: aiohttp.ClientSession,
    url: str,
    *,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    async with session.get(url, params=params, headers=headers, allow_redirects=True) as response:
        if response.status != 200:
            return {}
        chunks: list[bytes] = []
        received = 0

        # Read the body in chunks so an oversized response is abandoned early.
        async for chunk in response.content.iter_chunked(65_536):
            received += len(chunk)
            if received > _MAX_RESPONSE_BYTES:
                return {}
            chunks.append(chunk)
        try:
            value = json.loads(b"".join(chunks).decode(response.charset or "utf-8"))
        except (LookupError, UnicodeDecodeError, ValueError):
            return {}
        return value if isinstance(value, dict) else {}


# Target detection

# Return (language, article title) for a Wikipedia article URL, or None for other URLs, namespaced
# pages (titles with a colon) and hosts whose subdomain is not a language code.
def _wikipedia_target(url: str) -> tuple[str, str] | None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower().removeprefix("www.").removeprefix("m.")
    if not host.endswith(".wikipedia.org") or "/wiki/" not in parsed.path:
        return None
    language = host.split(".", 1)[0]
    title = unquote(parsed.path.split("/wiki/", 1)[1]).strip()
    if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z]+)?", language) or not title or ":" in title:
        return None
    return language, title


# Return (Stack Exchange site name, question id) for a question URL, or None for other URLs.
def _stackexchange_target(url: str) -> tuple[str, str] | None:
    host = _host(url)
    site = _STACK_EXCHANGE_SITES.get(host)
    if site is None and host.endswith(".stackexchange.com") and host.count(".") == 2:
        site = host.split(".", 1)[0]
    match = _QUESTION_RE.search(urlsplit(url).path)
    if not site or not match:
        return None
    return site, match.group(1)


# Return (owner, repository) for a github.com repository URL, or None for other URLs and site pages.
def _github_target(url: str) -> tuple[str, str] | None:
    if _host(url) != "github.com":
        return None
    segments = [segment for segment in urlsplit(url).path.split("/") if segment]
    if len(segments) < 2 or segments[0].lower() in _GITHUB_RESERVED:
        return None
    owner, repo = segments[0], segments[1].removesuffix(".git")
    if not re.fullmatch(r"[A-Za-z0-9-]+", owner) or not re.fullmatch(r"[A-Za-z0-9._-]+", repo):
        return None
    return owner, repo


# Return (registry, package name) for a PyPI or npm package page, or None for other URLs.
def _package_target(url: str) -> tuple[str, str] | None:
    host = _host(url)
    path = urlsplit(url).path
    if host == "pypi.org":
        match = _PYPI_RE.match(path)
        return ("pypi", match.group(1)) if match else None
    if host == "npmjs.com":
        match = _NPM_RE.match(path)
        return ("npm", match.group(1)) if match else None
    return None


# Wikipedia

# Wikidata properties worth a line in a card, in display order: (property id, label, maximum number
# of values shown).
_WIKIDATA_PROPERTIES: tuple[tuple[str, str, int], ...] = (
    ("P31", "Instance of", 2),
    ("P569", "Born", 1),
    ("P19", "Birthplace", 1),
    ("P570", "Died", 1),
    ("P20", "Place of death", 1),
    ("P27", "Citizenship", 2),
    ("P106", "Occupation", 3),
    ("P166", "Awards", 4),
    ("P571", "Founded / inception", 1),
    ("P577", "Published", 1),
    ("P112", "Founded by", 2),
    ("P50", "Author", 2),
    ("P57", "Director", 2),
    ("P178", "Developer", 2),
    ("P136", "Genre", 2),
    ("P400", "Platform", 3),
    ("P277", "Written in", 2),
    ("P348", "Latest version", 1),
    ("P275", "License", 1),
    ("P17", "Country", 1),
    ("P131", "Located in", 1),
    ("P36", "Capital", 1),
    ("P35", "Head of state", 1),
    ("P6", "Head of government", 1),
    ("P38", "Currency", 1),
    ("P159", "Headquarters", 1),
    ("P1128", "Employees", 1),
    ("P1082", "Population", 1),
    ("P2046", "Area", 1),
    ("P2048", "Height", 1),
    ("P2043", "Length", 1),
    ("P2044", "Elevation", 1),
    ("P625", "Coordinates", 1),
    ("P856", "Website", 1),
)


# Return the preferred claims when there are any, otherwise all non-deprecated claims.
def _best_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    usable = [claim for claim in claims if claim.get("rank") != "deprecated"]
    preferred = [claim for claim in usable if claim.get("rank") == "preferred"]
    return preferred or usable


# Format a Wikidata time value according to its precision (day, month, year or decade), with a
# " BCE" suffix for negative years. Return an empty string when the time cannot be parsed.
def _wikidata_time(value: dict[str, Any]) -> str:
    raw = str(value.get("time") or "")
    match = re.match(r"([+-])(\d+)-(\d{2})-(\d{2})", raw)
    if not match:
        return ""
    sign, year, month, day = match.groups()
    year_number = int(year)
    precision = int(value.get("precision") or 11)
    era = " BCE" if sign == "-" else ""
    if precision >= 11:
        return f"{year_number:04d}-{month}-{day}{era}"
    if precision == 10:
        return f"{year_number:04d}-{month}{era}"
    if precision == 9:
        return f"{year_number}{era}"
    if precision == 8:
        return f"{year_number // 10 * 10}s{era}"
    return f"{year_number}{era}"


# Format a Wikidata quantity amount: integers get space-separated thousands, other numbers four
# significant digits; a value that is not a number is returned as is.
def _wikidata_number(amount: object) -> str:
    text = str(amount or "").lstrip("+")
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return f"{int(number):,}".replace(",", " ")
    return f"{number:,.4g}".replace(",", " ")


# Return the Wikidata entity id (for example Q42) held by an entity-valued datavalue, or "".
def _entity_id(datavalue: dict[str, Any]) -> str:
    value = datavalue.get("value") if isinstance(datavalue.get("value"), dict) else {}
    return str(value.get("id") or "")


# Common Wikidata units rendered as symbols instead of (localized) labels.
_UNIT_SYMBOLS = {
    "Q11573": "m",
    "Q828224": "km",
    "Q174728": "cm",
    "Q174789": "mm",
    "Q712226": "km²",
    "Q25343": "m²",
    "Q35852": "ha",
    "Q11570": "kg",
    "Q191118": "t",
    "Q41803": "g",
    "Q11579": "K",
    "Q25267": "°C",
    "Q3194304": "kbit/s",
    "Q4917": "USD",
    "Q4916": "EUR",
    "Q1104069": "CAD",
    "Q41044": "RUB",
}


# Turn a Wikidata entity's claims into (label, value text) fact lines in _WIKIDATA_PROPERTIES order.
# Entity-valued claims and quantity units without a symbol are resolved to labels in the wanted
# language (falling back to English) with batched wbgetentities calls.
async def _wikidata_facts(
    session: aiohttp.ClientSession, entity: dict[str, Any], language: str
) -> list[tuple[str, str]]:
    claims = entity.get("claims") if isinstance(entity.get("claims"), dict) else {}
    if not claims:
        return []

    # Pick the best claims of each wanted property and collect the entity ids that need a label.
    picked: list[tuple[str, list[dict[str, Any]]]] = []
    label_ids: set[str] = set()
    for prop, _label, limit in _WIKIDATA_PROPERTIES:
        values: list[dict[str, Any]] = []
        for claim in _best_claims(claims.get(prop) or []):
            snak = claim.get("mainsnak") if isinstance(claim.get("mainsnak"), dict) else {}
            datavalue = snak.get("datavalue") if isinstance(snak.get("datavalue"), dict) else {}
            if not datavalue:
                continue
            values.append(datavalue)
            if datavalue.get("type") == "wikibase-entityid":
                label_ids.add(_entity_id(datavalue))
            elif datavalue.get("type") == "quantity":
                unit = str((datavalue.get("value") or {}).get("unit") or "")
                unit_id = unit.rsplit("/", 1)[1] if "/entity/Q" in unit else ""
                if unit_id and unit_id not in _UNIT_SYMBOLS:
                    label_ids.add(unit_id)
            if len(values) >= limit:
                break
        if values:
            picked.append((prop, values))

    # Resolve the collected ids to labels, 50 per request (the API limit).
    labels: dict[str, str] = {}
    ids = sorted(label_id for label_id in label_ids if label_id)
    if ids:
        languages = "|".join(dict.fromkeys([language, "en"]))
        for start in range(0, len(ids), 50):
            label_payload = await _fetch_json(
                session,
                _WIKIDATA_API,
                params={
                    "action": "wbgetentities",
                    "ids": "|".join(ids[start : start + 50]),
                    "props": "labels",
                    "languages": languages,
                    "format": "json",
                },
            )
            for entity_id, item in (label_payload.get("entities") or {}).items():
                item_labels = item.get("labels") if isinstance(item, dict) else {}
                chosen = (item_labels or {}).get(language) or (item_labels or {}).get("en") or {}
                if chosen.get("value"):
                    labels[entity_id] = str(chosen["value"])

    # Render each value by its datatype and join the distinct ones per property.
    names = {prop: label for prop, label, _limit in _WIKIDATA_PROPERTIES}
    facts: list[tuple[str, str]] = []
    for prop, values in picked:
        rendered: list[str] = []
        for datavalue in values:
            kind = datavalue.get("type")
            value = datavalue.get("value")
            if kind == "wikibase-entityid":
                text = labels.get(_entity_id(datavalue), "")
            elif kind == "time" and isinstance(value, dict):
                text = _wikidata_time(value)
            elif kind == "quantity" and isinstance(value, dict):
                unit = str(value.get("unit") or "")
                unit_id = unit.rsplit("/", 1)[1] if "/entity/Q" in unit else ""
                unit_label = _UNIT_SYMBOLS.get(unit_id) or labels.get(unit_id, "")
                text = f"{_wikidata_number(value.get('amount'))} {unit_label}".strip()
            elif kind == "globecoordinate" and isinstance(value, dict):
                try:
                    text = f"{float(value['latitude']):.4f}, {float(value['longitude']):.4f}"
                except (KeyError, TypeError, ValueError):
                    text = ""
            elif kind == "string":
                text = str(value or "")
            elif kind == "monolingualtext" and isinstance(value, dict):
                text = str(value.get("text") or "")
            else:
                text = ""
            if text and text not in rendered:
                rendered.append(text)
        if rendered:
            facts.append((names[prop], ", ".join(rendered)))
    return facts


# Build a Wikipedia card: the article description, Wikidata key facts (labelled in the wanted
# language) and a short lead. Return an empty dict when the article is missing or has nothing to show.
async def _wikipedia_card(
    session: aiohttp.ClientSession, target: tuple[str, str], language: str
) -> dict[str, Any]:
    source_language, page_title = target
    wanted = language.lower().split("-", 1)[0]
    if not re.fullmatch(r"[a-z]{2,3}", wanted):
        wanted = source_language

    # Fetch the REST summary of the article.
    async def summary(lang: str, title: str) -> dict[str, Any]:
        return await _fetch_json(
            session,
            f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(title, safe='()_,-')}",
            params={"redirect": "true"},
        )

    # The article summary and its Wikidata entity (looked up by sitelink) load in
    # parallel; only the label lookup for item values waits on the entity.
    payload, entity_payload = await asyncio.gather(
        summary(source_language, page_title),
        _fetch_json(
            session,
            _WIKIDATA_API,
            params={
                "action": "wbgetentities",
                "sites": f"{source_language.replace('-', '_')}wiki",
                "titles": page_title.replace("_", " "),
                "props": "claims",
                "redirects": "yes",
                "format": "json",
            },
        ),
    )

    # Give up when the article does not exist.
    if not payload or str(payload.get("type") or "").endswith("not_found"):
        return {}

    # Take the Wikidata entity and its facts, then assemble the card lines.
    entities = entity_payload.get("entities") if isinstance(entity_payload.get("entities"), dict) else {}
    qid, entity = next(
        ((key, value) for key, value in entities.items() if key.startswith("Q") and isinstance(value, dict)),
        ("", {}),
    )
    facts = await _wikidata_facts(session, entity, wanted) if entity else []
    description = _plain_text(payload.get("description"), maximum=160)
    lead = _plain_text(payload.get("extract"), maximum=420)
    if not facts and not description and not lead:
        return {}
    lines: list[str] = []
    if description:
        lines.append(description)
    lines.extend(f"{label}: {value}" for label, value in facts)
    if lead:
        lines.append(f"Lead: {lead}")
    return {
        "kind": "wikipedia",
        "provider": "Wikipedia + Wikidata",
        "title": _plain_text(payload.get("title"), maximum=160) or page_title.replace("_", " "),
        "wikidata": qid,
        "lines": lines,
    }


# Any other MediaWiki wiki (Fandom, wiki.gg…)

# Article URLs in every standard MediaWiki $wgArticlePath layout. Nothing here is
# site-specific: Fandom, wiki.gg, Miraheze, Wikimedia sister projects (Wikivoyage,
# Wiktionary…), ArchWiki, PCGamingWiki and self-hosted mod wikis all match one of
# them. Whether a host really runs MediaWiki is learned from its API (see below).
# The layouts, in order: /wiki/T and /ru/wiki/T, /w/T, /title/T (ArchWiki style), /index.php/T.
_WIKI_ARTICLE_PATHS = (
    re.compile(r"^(?P<prefix>/[a-z]{2,3}(?:-[a-z]{2,4})?)?/wiki/(?P<title>.+)$"),
    re.compile(r"^/w/(?!(?:index|api|load)\.php)(?P<title>[^/].*)$"),
    re.compile(r"^/title/(?P<title>.+)$"),
    re.compile(r"^(?P<prefix>(?:/[^/]+)*?)/index\.php/(?P<title>.+)$"),
)

# Maximum number of infobox facts in a wiki card.
_WIKI_FACT_LIMIT = 15

# base URL -> working api.php; "" marks a host that turned out not to be MediaWiki,
# so a site that merely shares a URL shape costs one probe per process.
_wiki_api_by_base: dict[str, str] = {}


# Return (wiki base URL, page title) for a URL that looks like a MediaWiki article, or None.
# Wikipedia itself is excluded because it gets the richer Wikidata card.
def _wiki_target(url: str) -> tuple[str, str] | None:
    parsed = urlsplit(url)
    host = (parsed.hostname or "").lower()

    # Wikipedia gets the richer Wikidata card instead.
    if not host or host == "wikipedia.org" or host.endswith(".wikipedia.org"):
        return None
    root = f"{parsed.scheme or 'https'}://{host}"

    # Extract the title (and any language/path prefix) from the index.php query or a known layout.
    if parsed.path.endswith("/index.php"):
        # /index.php?title=T (and /w/index.php?title=T): the API sits next to index.php.
        raw_title = (parse_qs(parsed.query).get("title") or [""])[0]
        prefix = parsed.path.removesuffix("/index.php")
    else:
        for pattern in _WIKI_ARTICLE_PATHS:
            match = pattern.match(parsed.path)
            if match:
                raw_title = match.group("title")
                prefix = match.groupdict().get("prefix") or ""
                break
        else:
            return None
    title = unquote(raw_title).replace("_", " ").strip()
    if not title:
        return None
    return root + prefix, title


# Return the api.php URLs to probe for a wiki base URL, in order and without duplicates.
def _wiki_api_candidates(base: str) -> list[str]:
    parts = urlsplit(base)
    root = f"{parts.scheme}://{parts.hostname}"

    # The two layouts MediaWiki installs use: api.php beside the article prefix
    # (Fandom, wiki.gg, /ru/ language wikis) or under /w/ (Wikimedia, Miraheze).
    return list(dict.fromkeys([f"{base}/api.php", f"{root}/w/api.php"]))


# Return the text of an HTML node with whitespace collapsed and link-boundary spacing tightened.
def _node_text(node: Any) -> str:
    text = " ".join(node.text(separator=" ").split())

    # Link boundaries leave "word , word" and "( x )"; tighten them back.
    text = re.sub(r"\s+([,.;:!?)\]])", r"\1", text)
    text = re.sub(r"([(\[])\s+", r"\1", text)
    return text.strip(" ;")


# Extract up to _WIKI_FACT_LIMIT distinct (label, value) pairs from the infoboxes of a parsed wiki
# page: Fandom portable infoboxes and classic or stat tables inside an infobox.
def _wiki_infobox_facts(tree: Any) -> list[tuple[str, str]]:
    facts: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()

    # Normalize one label/value pair and record it unless it is empty, repeated or over the limit.
    def add(label: str, value: str) -> None:
        label = _clip(re.sub(r"\s*;\s*", " ", label).strip(" :;"), 40)
        value = re.sub(r"(?:\s*;\s*)+", "; ", value)
        value = re.sub(r":\s*;\s*", ": ", value).strip(" ;")

        # Item tooltips pack stats, set bonuses and resistances into one cell.
        value = _clip(value, 480)
        if not label or not value or value.casefold() == label.casefold():
            return
        if (label, value) not in seen and len(facts) < _WIKI_FACT_LIMIT:
            seen.add((label, value))
            facts.append((label, value))

    # Fandom portable infoboxes: <h3 class="pi-data-label"> + <div class="pi-data-value">.
    for item in tree.css("aside.portable-infobox .pi-data"):
        label, value = item.css_first(".pi-data-label"), item.css_first(".pi-data-value")
        if label is not None and value is not None:
            add(_node_text(label), _node_text(value))

    # Classic and stat tables: rows of one header cell and one value cell, plus
    # full-width section headers followed by a full-width text row (item tooltips).
    for table in tree.css("table"):
        classes = str(table.attributes.get("class") or "")
        inside_infobox = "infobox" in classes or any(
            "infobox" in str(parent.attributes.get("class") or "")
            for parent in _ancestors(table)
        )
        if not inside_infobox:
            continue
        pending_header = ""
        for row in table.css("tr"):
            headers, cells = row.css("th"), row.css("td")
            if len(headers) == 1 and len(cells) == 1:
                add(_node_text(headers[0]), _node_text(cells[0]))
                pending_header = ""
            elif len(headers) == 1 and not cells:
                pending_header = _node_text(headers[0])
            elif pending_header and len(cells) == 1 and not headers:
                add(pending_header, _node_text(cells[0]))
                pending_header = ""
            elif pending_header and len(cells) == 2 and not headers:
                # Sectioned rows: "Release dates" > "Windows | May 16, 2011".
                qualifier = _node_text(cells[0])
                label = f"{pending_header} ({qualifier})" if qualifier else pending_header
                add(label, _node_text(cells[1]))
    return facts


# Return up to 40 ancestors of an HTML node, nearest first.
def _ancestors(node: Any) -> list[Any]:
    found: list[Any] = []
    parent = node.parent
    while parent is not None and len(found) < 40:
        found.append(parent)
        parent = parent.parent
    return found


# Return the first prose paragraph of a wiki page (up to 420 characters), skipping image captions
# and paragraphs inside infoboxes, tables, navigation or message boxes. Empty when there is none.
def _wiki_lead(tree: Any) -> str:
    skip_classes = (
        "infobox", "message-box", "hatnote", "navbox", "toc", "notice", "ambox",
        "gallery", "thumb", "caption", "video",
    )
    for paragraph in tree.css("p"):
        text = _node_text(paragraph)

        # A lead is prose: image captions ("X attacking target dummies.") are short.
        if len(text) < 30 or (len(text) < 80 and text.count(" ") < 8):
            continue
        if any(
            parent.tag in {"table", "aside", "figure", "figcaption", "ul", "ol"}
            or any(name in str(parent.attributes.get("class") or "") for name in skip_classes)
            for parent in _ancestors(paragraph)
        ):
            continue
        return _clip(text, 420)
    return ""


# Build a card for any other MediaWiki page: infobox fields and the lead, read from the site's own
# parse API, which is discovered from the URL alone. The language is unused. Return an empty dict
# when the host is not MediaWiki or the page has nothing to show.
async def _wiki_card(
    session: aiohttp.ClientSession, target: tuple[str, str], language: str
) -> dict[str, Any]:
    del language
    from selectolax.parser import HTMLParser

    # Hosts already known not to be MediaWiki are skipped.
    base, title = target
    known = _wiki_api_by_base.get(base)
    if known == "":
        return {}

    # Request the intro section (section 0) of the page as rendered HTML.
    params = {
        "action": "parse",
        "page": title,
        "prop": "text",
        "section": "0",
        "redirects": "1",
        "disablelimitreport": "1",
        "format": "json",
        "formatversion": "2",
    }

    # Query one api.php candidate; a connection or TLS failure means no MediaWiki API there.
    async def probe(api: str) -> tuple[str, dict[str, Any]]:
        try:
            return api, await _fetch_json(session, api, params=params)
        except aiohttp.ClientError:
            return api, {}

    # Return True when the payload proves the endpoint is MediaWiki.
    def is_mediawiki(payload: dict[str, Any]) -> bool:
        # A page error ("missingtitle") still proves the endpoint is MediaWiki.
        return isinstance(payload.get("parse"), dict) or isinstance(payload.get("error"), dict)

    # Unknown hosts try both API layouts at once and keep whichever answers.
    tasks = [asyncio.ensure_future(probe(api)) for api in ([known] if known else _wiki_api_candidates(base))]
    payload: dict[str, Any] = {}
    try:
        for finished in asyncio.as_completed(tasks):
            api, candidate = await finished
            if is_mediawiki(candidate):
                _wiki_api_by_base[base] = api
                payload = candidate
                break
        else:
            _wiki_api_by_base[base] = ""
            return {}
    finally:
        for task in tasks:
            task.cancel()

    # Parse the page HTML and drop scripts, references, edit links and line breaks that would
    # pollute the extracted text.
    parse = payload.get("parse") if isinstance(payload.get("parse"), dict) else {}
    html = str(parse.get("text") or "")
    if not html:
        return {}
    tree = HTMLParser(html)
    for node in tree.css("style, script, sup.reference, .mw-editsection, audio"):
        node.decompose()
    for node in tree.css("br"):
        node.replace_with("; ")

    # Assemble the card lines from the infobox facts and the lead.
    facts = _wiki_infobox_facts(tree)
    lead = _wiki_lead(tree)
    if not facts and not lead:
        return {}
    provider = (urlsplit(base).hostname or "").lower().removeprefix("www.")
    lines = [f"{label}: {value}" for label, value in facts]
    if lead:
        lines.append(f"Lead: {lead}")
    return {
        "kind": "wiki",
        "provider": provider,
        "title": _clip(str(parse.get("title") or title), 160),
        "lines": lines,
    }


# Stack Exchange

# Build a card with the accepted (or top-voted) answer of a question: its code-preserving text, and
# a title line with the score, answer date and edit date. The language is unused. Return an empty
# dict when the question has no usable answer.
async def _stackexchange_card(
    session: aiohttp.ClientSession, target: tuple[str, str], language: str
) -> dict[str, Any]:
    del language
    site, question_id = target

    # Fetch the question's top answers with their bodies, highest score first.
    payload = await _fetch_json(
        session,
        f"https://api.stackexchange.com/2.3/questions/{question_id}/answers",
        params={
            "site": site,
            "sort": "votes",
            "order": "desc",
            "pagesize": "5",
            "filter": "withbody",
        },
    )

    # Prefer the accepted answer; otherwise the top-voted one.
    answers = [item for item in payload.get("items") or [] if isinstance(item, dict)]
    if not answers:
        return {}
    accepted = next((item for item in answers if item.get("is_accepted")), None)
    answer = accepted or answers[0]
    text = _answer_text(answer.get("body"), maximum=_ANSWER_TEXT_LIMIT)
    if not text:
        return {}

    # Summarize the answer's provenance in the card title.
    meta = [
        "accepted answer" if answer.get("is_accepted") else "top-voted answer",
        f"score {answer.get('score', 0)}",
    ]
    answered = _iso_date(answer.get("creation_date"))
    if answered:
        meta.append(f"answered {answered}")
    edited = _iso_date(answer.get("last_edit_date") or answer.get("last_activity_date"))
    if edited and edited != answered:
        meta.append(f"last edited {edited}")
    if accepted is not None and answers[0] is not accepted:
        meta.append(f"a higher-voted answer exists (score {answers[0].get('score', 0)})")
    return {
        "kind": "stackexchange",
        "provider": "Stack Overflow" if site == "stackoverflow" else f"StackExchange ({site})",
        "title": "; ".join(meta),
        "lines": [text],
    }


# GitHub

# Build a repository health card: description, status (stars, forks, open issues, archived), last
# push, latest release and license. The language is unused. Return an empty dict for an unknown
# repository.
async def _github_card(
    session: aiohttp.ClientSession, target: tuple[str, str], language: str
) -> dict[str, Any]:
    del language
    owner, repo = target

    # Fetch the repository and its latest release in parallel.
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    base = f"https://api.github.com/repos/{owner}/{repo}"
    repo_payload, release_payload = await asyncio.gather(
        _fetch_json(session, base, headers=headers),
        _fetch_json(session, f"{base}/releases/latest", headers=headers),
    )

    # An unknown repository gives an empty payload, so there is no card.
    if not repo_payload.get("full_name"):
        return {}

    # Description, then the status line (an archived repository is flagged first).
    lines: list[str] = []
    description = _plain_text(repo_payload.get("description"), maximum=200)
    if description:
        lines.append(description)
    status = [
        f"stars {repo_payload.get('stargazers_count', 0)}",
        f"forks {repo_payload.get('forks_count', 0)}",
        f"open issues {repo_payload.get('open_issues_count', 0)}",
    ]
    if repo_payload.get("archived"):
        status.insert(0, "ARCHIVED (read-only, unmaintained)")
    lines.append(", ".join(status))

    # Last push, latest release, then license and language when known.
    pushed = _short_date(repo_payload.get("pushed_at"))
    if pushed:
        lines.append(f"Last push: {pushed}")
    if release_payload.get("tag_name"):
        published = _short_date(release_payload.get("published_at"))
        lines.append(
            f"Latest release: {release_payload['tag_name']}" + (f" ({published})" if published else "")
        )
    license_info = repo_payload.get("license") if isinstance(repo_payload.get("license"), dict) else {}
    extras = [
        f"license {license_info.get('spdx_id')}" if license_info.get("spdx_id") not in (None, "", "NOASSERTION") else "",
        f"language {repo_payload.get('language')}" if repo_payload.get("language") else "",
    ]
    extras = [item for item in extras if item]
    if extras:
        lines.append(", ".join(extras))
    return {
        "kind": "github",
        "provider": "GitHub",
        "title": str(repo_payload.get("full_name")),
        "lines": lines,
    }


# Package

# Build a card for the latest version of a PyPI or npm package: description, version (with release
# date for PyPI), required Python or Node version, and license. The language is unused. Return an
# empty dict when the registry has no version for the package.
async def _package_card(
    session: aiohttp.ClientSession, target: tuple[str, str], language: str
) -> dict[str, Any]:
    del language
    registry, name = target
    lines: list[str] = []

    # PyPI: the JSON API of the project.
    if registry == "pypi":
        payload = await _fetch_json(session, f"https://pypi.org/pypi/{quote(name)}/json")
        info = payload.get("info") if isinstance(payload.get("info"), dict) else {}
        version = str(info.get("version") or "")
        if not version:
            return {}
        uploads = payload.get("urls") if isinstance(payload.get("urls"), list) else []
        released = _short_date((uploads[0] or {}).get("upload_time_iso_8601")) if uploads else ""
        lines.append(f"Latest version: {version}" + (f" (released {released})" if released else ""))
        if info.get("summary"):
            lines.insert(0, _plain_text(info.get("summary"), maximum=200))
        if info.get("requires_python"):
            lines.append(f"Requires Python: {info['requires_python']}")
        license_name = str(info.get("license_expression") or info.get("license") or "").strip()
        if license_name and len(license_name) <= 60:
            lines.append(f"License: {license_name}")
        title = str(info.get("name") or name)
        provider = "PyPI"
    else:
        # npm: the registry document of the latest version.
        payload = await _fetch_json(session, f"https://registry.npmjs.org/{quote(name, safe='@')}/latest")
        version = str(payload.get("version") or "")
        if not version:
            return {}
        if payload.get("description"):
            lines.append(_plain_text(payload.get("description"), maximum=200))
        lines.append(f"Latest version: {version}")
        engines = payload.get("engines") if isinstance(payload.get("engines"), dict) else {}
        if engines.get("node"):
            lines.append(f"Requires Node: {engines['node']}")
        if isinstance(payload.get("license"), str):
            lines.append(f"License: {payload['license']}")
        title = str(payload.get("name") or name)
        provider = "npm"
    return {"kind": "package", "provider": provider, "title": title, "lines": lines}


# Deduplication

# Return the lowercase (case-folded) words of a text.
def _words(text: str) -> list[str]:
    return [word.casefold() for word in _WORD_RE.findall(text)]


# Return the part of a card line that is compared with the source text: the value of a short
# "Label: value" line, or the whole line otherwise.
def _line_value(line: str) -> str:
    # "Label: value" lines are judged by their value; the label is card vocabulary.
    label, separator, value = line.partition(":")
    return value if separator and len(label) <= 30 and "\n" not in label else line


# Keep only card lines that add information beyond the source's own content: a line is dropped when
# at least _DUPLICATE_COVERAGE of its words already occur in source_text.
def _new_lines(lines: list[str], source_text: str) -> list[str]:
    known = set(_words(source_text))
    if not known:
        return list(lines)
    kept: list[str] = []
    for line in lines:
        words = _words(_line_value(line))
        if not words:
            continue
        coverage = sum(word in known for word in words) / len(words)
        if coverage < _DUPLICATE_COVERAGE:
            kept.append(line)
    return kept


# Join the title, snippet, markdown and content of a search result into the text a card is compared with.
def _source_text(result: dict[str, Any]) -> str:
    return " ".join(
        str(result.get(field) or "") for field in ("title", "snippet", "markdown", "content")
    )


# Service

# Signature of a card builder: (HTTP session, target, language) -> card dict (empty = no card).
_CardBuilder = Callable[[aiohttp.ClientSession, tuple[str, str], str], Awaitable[dict[str, Any]]]

# Card kinds in priority order: (kind, URL -> target detector, card builder). A result URL is
# matched against the kinds in this order and the first match wins.
_PROVIDERS: tuple[tuple[str, Callable[[str], tuple[str, str] | None], _CardBuilder], ...] = (
    ("wikipedia", _wikipedia_target, _wikipedia_card),
    ("stackexchange", _stackexchange_target, _stackexchange_card),
    ("wiki", _wiki_target, _wiki_card),
    ("github", _github_target, _github_card),
    ("package", _package_target, _package_card),
)

# Cards per kind in one search. Game/mod questions often surface a Fandom wiki and
# the project's own wiki side by side, and both carry different infobox data.
_KIND_LIMITS = {"wiki": 2}

# Fandom's API routinely needs 1.5–3 s for a parse; other providers answer < 1 s.
# Cards resolve in parallel, so this only bounds the slowest wiki card.
_KIND_TIMEOUT_FLOOR = {"wiki": 3.5}


# Model-facing answer cards for the top web_search results, built and cached per result.
# A card adds structured data that search snippets do not carry, for a source that is already in
# the results, and is rendered as a compact text block inside the model context (no UI). Every card
# names the citation handle of the source it enriches, so the model cites the card's facts exactly
# like the source itself.
# Card kinds (one per kind - two for wiki - in result order, at most max_cards):
# - wikipedia: Wikidata key facts (dates, sizes, population, places, website…) plus a one-line
#   description and a short lead.
# - wiki: any other MediaWiki site (Fandom, wiki.gg, Miraheze, Wikimedia sister projects, ArchWiki,
#   PCGamingWiki, self-hosted mod wikis…): infobox fields and the lead, via the site's own API,
#   discovered from the URL alone.
# - stackexchange: the accepted (or top-voted) answer with its code, score, date.
# - github: repository health - stars, last push, latest release, license, archived.
# - package: PyPI / npm latest version, release date, Python requirement, license.
# Everything fails open under a short timeout: a provider error or timeout simply yields no card and
# the search result is untouched. Results are cached in memory (positive and negative TTLs).
class RichCardService:
    # Cache shared by all service instances, so cards survive across searches.
    _shared_cache = _TtlCache()

    # Attach the shared cache.
    def __init__(self) -> None:
        self.cache = self._shared_cache

    # Build the cards for the top results, in result order. config provides the switches and limits
    # (enabled, max_cards, scan_results, per-kind switches, timeout_seconds, ttl_seconds,
    # negative_ttl_seconds); language is the preferred language of the labels. Each card dict carries
    # the source's citation handle and URL and only the lines that add information beyond the source's
    # own content. Return an empty list when cards are disabled or nothing matches.
    async def build(
        self,
        results: list[dict[str, Any]],
        config: Any,
        *,
        language: str = "en",
    ) -> list[dict[str, Any]]:
        if not getattr(config, "enabled", False) or not results:
            return []
        max_cards = max(0, int(getattr(config, "max_cards", 3)))
        scan_limit = max(1, int(getattr(config, "scan_results", 8)))

        # Select candidates: scan the top results and match each URL to the first enabled card kind
        # that is below its per-kind limit and not already targeted, until max_cards is reached.
        candidates: list[tuple[int, str, tuple[str, str], _CardBuilder, dict[str, Any]]] = []
        kind_counts: dict[str, int] = {}
        seen_targets: set[tuple[str, str]] = set()
        for index, result in enumerate(results[:scan_limit]):
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "")
            for kind, detect, builder in _PROVIDERS:
                if kind_counts.get(kind, 0) >= _KIND_LIMITS.get(kind, 1):
                    continue
                if not getattr(config, kind, True):
                    continue
                target = detect(url)
                if target is None or target in seen_targets:
                    continue
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
                seen_targets.add(target)
                candidates.append((index, kind, target, builder, result))
                break
            if len(candidates) >= max_cards:
                break
        if not candidates:
            return []

        # The session timeout covers the slowest candidate kind (see _KIND_TIMEOUT_FLOOR).
        timeout = float(getattr(config, "timeout_seconds", 2.0))
        kinds = {candidate[1] for candidate in candidates}
        session_timeout = max([timeout, *(_KIND_TIMEOUT_FLOOR.get(kind, 0.0) for kind in kinds)])
        client_timeout = aiohttp.ClientTimeout(total=session_timeout)
        headers = {"Accept": "application/json", "User-Agent": _USER_AGENT}

        async with aiohttp.ClientSession(timeout=client_timeout, headers=headers) as session:

            # Resolve one candidate to (result index, card): serve it from the cache or build it
            # under its own timeout, cache the outcome (a failed lookup gets the shorter negative
            # TTL), and strip the lines the source already contains. An empty dict means no card.
            async def resolve(
                index: int,
                kind: str,
                target: tuple[str, str],
                builder: _CardBuilder,
                result: dict[str, Any],
            ) -> tuple[int, dict[str, Any]]:
                key = _cache_key(kind, "/".join(target), language.lower())
                cached = self.cache.get(key)
                if cached is not None:
                    card = cached.get("card") if isinstance(cached.get("card"), dict) else {}
                    logger.debug("answer card cache hit kind=%s", kind)
                else:
                    started = time.perf_counter()
                    try:
                        async with asyncio.timeout(max(timeout, _KIND_TIMEOUT_FLOOR.get(kind, 0.0))):
                            card = await builder(session, target, language)
                    except (aiohttp.ClientError, TimeoutError, ValueError, TypeError, KeyError) as exc:
                        logger.debug("answer card failed kind=%s error=%s", kind, exc)
                        card = {}
                    logger.info(
                        "answer card resolved kind=%s target=%r hit=%s in %.0fms",
                        kind, "/".join(target)[:80], bool(card), (time.perf_counter() - started) * 1000,
                    )
                    ttl = float(getattr(config, "ttl_seconds", 86400)) if card else float(
                        getattr(config, "negative_ttl_seconds", 900)
                    )
                    self.cache.set(key, {"card": card}, ttl=ttl)
                if not card:
                    return index, {}

                # The cache holds the full card; deduplication is per result because
                # the same page can be excerpted differently for another query.
                lines = _new_lines(list(card.get("lines") or []), _source_text(result))
                if not lines:
                    logger.info("answer card dropped as duplicate kind=%s", kind)
                    return index, {}
                return index, {
                    **card,
                    "lines": lines,
                    "dropped_lines": len(card.get("lines") or []) - len(lines),
                    "handle": str(result.get("id") or ""),
                    "url": str(result.get("url") or ""),
                }

            # Resolve all candidates in parallel and keep the cards in result order.
            resolved = await asyncio.gather(*(resolve(*candidate) for candidate in candidates))
        resolved.sort(key=lambda item: item[0])
        return [card for _index, card in resolved if card]

    # Render cards as a compact, citable text block for the model context (empty string when there
    # is nothing to show). Each card is a header with the source handle, provider and title followed
    # by its indented lines, trimmed to the card text limits.
    @staticmethod
    def render(cards: list[dict[str, Any]]) -> str:
        blocks: list[str] = []
        for card in cards:
            lines = [str(line) for line in card.get("lines") or [] if str(line).strip()]
            if not lines:
                continue
            handle = str(card.get("handle") or "")
            header = f"{card.get('provider')}: {card.get('title')}"
            if handle:
                header = f"[{handle}] {header}"
            body = "\n".join(lines)
            if len(body) > _ANSWER_TEXT_LIMIT + 200:
                body = body[: _ANSWER_TEXT_LIMIT + 200].rstrip() + "\n…"
            elif card.get("kind") != "stackexchange" and len(body) > _CARD_TEXT_LIMIT:
                body = body[:_CARD_TEXT_LIMIT].rstrip() + "…"
            indented = "\n".join(f"  {line}" if line else "" for line in body.split("\n"))
            blocks.append(f"{header}\n{indented}")
        if not blocks:
            return ""
        return (
            "Answer cards (structured data for sources in these results, showing only "
            "what their excerpts below lack; cite the handle shown; if a value "
            "disagrees with page text, report both):\n"
            + "\n\n".join(blocks)
        )


# Place the cards right before the first source list (Google/Brave style); without a source list
# they are appended to the model context.
def insert_cards_context(model_context: str, cards_text: str) -> str:
    if not cards_text:
        return model_context
    marker = "\nSources:\n"
    position = model_context.find(marker)
    if position < 0:
        return f"{model_context.rstrip()}\n\n{cards_text}" if model_context.strip() else cards_text
    return f"{model_context[:position].rstrip()}\n\n{cards_text}\n{model_context[position:]}"


__all__ = ["RichCardService", "insert_cards_context"]
