# Copyright NEXTGGTECH. Elastic License 2.0.

"""Model-facing MCP contract for the search tools (ported from the legacy adapter).

The model sees a compact query interface; advanced calls compile into the existing
mixed-vertical execution plan. Older vertical-field arguments remain readable for stored
conversations. Region routing, safe-search, and engine selection remain internal.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.query.search_plan import (
    PlanValidationError,
    VERTICAL_QUERY_LIMITS,
    prepare_advanced_search,
)

_SPACE_RE = re.compile(r"\s+")
_EFFORT_VALUES = ("low", "medium", "high")
_EFFORT_ALIASES = {"": "medium", "normal": "medium", "default": "medium", "standard": "medium"}
SEARCH_BATCH_LIMIT = VERTICAL_QUERY_LIMITS["web"]
INSTANT_SEARCH_BATCH_LIMIT = 3
LEGACY_BATCH_LIMIT = max(
    VERTICAL_QUERY_LIMITS[vertical] for vertical in ("web", "shopping", "academic")
)

MCP_SERVER_DESCRIPTION = "Search and page-reading tools."


LEGACY_WEB_SEARCH_TOOL_DESCRIPTION = """Search the web for facts, sources, and documentation.
Pass query as one focused string or up to three independent queries. Use shopping=true for
prices/offers or academic=true for papers. Effort defaults to medium. Optional search
operators (site:, quoted phrases, after:) belong in query. Cite returned source handles
and use read_page for full text when needed."""


ADVANCED_WEB_SEARCH_TOOL_DESCRIPTION = """Search the web for facts, sources, documentation, and research.
Pass query as a focused string or up to three independent query strings. Optional vertical
selects web (default), academic (papers), shopping (prices/offers), or onion when available.
Optional description is a short UI label; effort defaults to medium. Search operators such
as site:, quoted phrases, and after: can be included in query when needed.
Inspect the returned evidence, cite its exact source handles, and use read_page for full text.
Example: {"query":"M2 high speed steel thermal conductivity","vertical":"academic"}."""


# Compatibility export for callers that do not use the config-aware builder.
WEB_SEARCH_TOOL_DESCRIPTION = ADVANCED_WEB_SEARCH_TOOL_DESCRIPTION


IMAGE_SEARCH_TOOL_DESCRIPTION = (
    "Search the web for images matching a query. Returns normalized image results: "
    "full image URL, thumbnail, source page, dimensions. The query's own language "
    "and region are auto-detected, so write it in the user's language. Offered only "
    "to multimodal (vision-capable) models: you can pass any returned image_url to "
    "your image input to actually look at it.\n\n"
    "When this tool returns images, you MUST include the display token "
    "[[images: search terms]] in your final answer, replacing search terms with "
    "the search query (2-140 characters). Put the token on its own line in plain "
    "text, outside code blocks, inline code, quotes, links, and lists. The chat UI "
    "replaces this service token with a native image gallery, just as citation "
    "handles render search citations. URLs or img IDs alone do not display the "
    "gallery. Do not emit an image token when no images were found.\n\n"
    "Example after searching for red pandas:\n[[images: red pandas]]\n\n"
    "You can also use this token without calling the tool: the chat UI runs the "
    "image search automatically."
)


READ_PAGE_TOOL_DESCRIPTION = """\
Open one or more exact URLs and extract their readable text as markdown.

The `url` input must contain an exact, non-empty URL. Never call this with an empty value or
a search topic. If no exact URL is available yet, call web_search first.

Use it when you need:
- the full content of an article, documentation page, post, or thread;
- cleaner text after search surfaced promising URLs;
- a small batch read of several shortlisted pages.

It works best as the second step after search, once discovery is done and you want to
read the sources. PDFs, YouTube transcripts, Reddit threads, GitHub pages, StackExchange
questions, and X/Twitter posts are handled automatically from their URL."""


SEARCH_QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "query": {
            "oneOf": [
                {"type": "string", "minLength": 1, "maxLength": 220},
                {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": LEGACY_BATCH_LIMIT,
                    "items": {"type": "string", "minLength": 1, "maxLength": 220},
                },
            ],
            "description": "One focused search query or up to three independent queries.",
        },
        "effort": {
            "type": "string",
            "enum": list(_EFFORT_VALUES),
            "default": "medium",
            "description": (
                "Search depth: low for quick discovery, medium by default, high for a "
                "remaining difficult gap. High is the right tier for rare or hard-to-find "
                "content that default depths do not surface: old archives, scans, "
                "textbooks, documents, downloads, and lost media."
            ),
        },
        "shopping": {
            "type": "boolean",
            "default": False,
            "description": (
                "Required routing for product discovery, budgets, prices, sellers, stock, "
                "availability, and purchase options. Set true whenever the evidence gap "
                "contains any of those needs; keep the query to the product, model, spec, "
                "SKU, or product phrase. Use a separate ordinary web search for reviews."
            ),
        },
        "academic": {
            "type": "boolean",
            "default": False,
            "description": (
                "Required routing for papers, authors, citations, DOI records, preprints, "
                "peer-reviewed support, scholarly consensus, and primary scientific "
                "literature. Set true whenever the evidence gap contains any of those "
                "needs; query by topic, title, author, or DOI."
            ),
        },
    },
    "required": ["query"],
}


# The onion opt-in property — added to the schema ONLY when the tor path is enabled in
# config, so the model never sees an argument it cannot use.
_ONION_PROPERTY: dict[str, Any] = {
    "type": "boolean",
    "default": False,
    "description": (
        "Enable censorship-resistant onion sources over Tor (vetted allowlist: news "
        "SecureDrop/onion mirrors, rights orgs, privacy services). Set true only when the "
        "user explicitly needs Tor/onion access or is working around blocking; it is slow "
        "(Tor latency) and off-topic for ordinary searches. Leave false otherwise."
    ),
}


# Build the search tool's input schema for the current config. The onion opt-in is exposed
# only while tor.enabled, so the advertised arguments track the actual capability.
def search_schema_mode() -> str:
    try:
        from core.config import load_search_config

        mode = str(load_search_config().query.schema_mode or "advanced").strip().lower()
    except Exception:  # noqa: BLE001
        mode = "advanced"
    return mode if mode in {"legacy", "advanced"} else "advanced"


def build_search_description() -> str:
    return (
        LEGACY_WEB_SEARCH_TOOL_DESCRIPTION
        if search_schema_mode() == "legacy"
        else ADVANCED_WEB_SEARCH_TOOL_DESCRIPTION
    )


def build_search_schema() -> dict[str, Any]:
    import copy

    try:
        from core.config import load_search_config

        tor_enabled = bool(load_search_config().tor.enabled)
    except Exception:  # noqa: BLE001
        tor_enabled = False

    if search_schema_mode() == "advanced":
        verticals = ["web", "academic", "shopping", *(["onion"] if tor_enabled else [])]
        query_string = {"type": "string", "minLength": 1, "maxLength": 512}
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "query": {
                    "oneOf": [query_string, {
                        "type": "array", "minItems": 1, "maxItems": SEARCH_BATCH_LIMIT,
                        "items": dict(query_string),
                    }],
                    "description": "One focused query or up to three independent queries.",
                },
                "vertical": {"type": "string", "enum": verticals, "default": "web",
                             "description": "web: general; academic: papers; shopping: prices/offers."},
                "effort": {"type": "string", "enum": list(_EFFORT_VALUES), "default": "medium"},
                "description": {"type": "string", "minLength": 1, "maxLength": 80,
                                "description": "Optional short UI action label in the user's language."},
            },
            "required": ["query"],
        }

    schema = copy.deepcopy(SEARCH_QUERY_SCHEMA)
    if tor_enabled:
        schema["properties"]["onion"] = dict(_ONION_PROPERTY)
    return schema


# Best-effort JSON parse for string tool arguments that look like objects.
def _try_parse_json(value: str) -> Any:
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return None
    try:
        return json.loads(stripped)
    except Exception:  # noqa: BLE001
        return None


# Collapse whitespace and cap a free-text query at 220 characters.
def sanitize_query(query: str) -> str:
    text = str(query or "").strip().replace("\r", " ").replace("\n", " ")
    return _SPACE_RE.sub(" ", text).strip()[:220].strip()


# Convert the public query argument into provider-ready search strings up to the supplied limit.
def coerce_search_queries(value: Any, *, limit: int = SEARCH_BATCH_LIMIT) -> list[str]:
    limit = max(1, int(limit or SEARCH_BATCH_LIMIT))
    if isinstance(value, dict):
        plan = value.get("query", value)
        if isinstance(plan, dict):
            for key in ("raw_query", "q", "text"):
                if key in plan:
                    return coerce_search_queries(plan[key], limit=limit)
            return []
        return coerce_search_queries(plan, limit=limit)
    if isinstance(value, (list, tuple)):
        queries: list[str] = []
        for item in value:
            query = sanitize_query(item)
            if query:
                queries.append(query)
            if len(queries) >= limit:
                break
        return queries
    if isinstance(value, str):
        parsed = _try_parse_json(value)
        if isinstance(parsed, (dict, list)):
            nested = coerce_search_queries(parsed, limit=limit)
            if nested:
                return nested
        query = sanitize_query(value)
        return [query] if query else []
    query = sanitize_query(str(value or ""))
    return [query] if query else []


def _search_query_item_count(value: Any) -> int:
    if isinstance(value, dict):
        plan = value.get("query", value)
        if isinstance(plan, dict):
            for key in ("raw_query", "q", "text"):
                if key in plan:
                    return _search_query_item_count(plan[key])
            return 0
        return _search_query_item_count(plan)
    if isinstance(value, (list, tuple)):
        return len(value)
    if isinstance(value, str):
        parsed = _try_parse_json(value)
        if isinstance(parsed, (dict, list)):
            return _search_query_item_count(parsed)
        return 1 if value.strip() else 0
    return 1 if value is not None and str(value).strip() else 0


# Backwards-compatible single-query helper for older callers.
def coerce_search_query(value: Any) -> str:
    queries = coerce_search_queries(value, limit=1)
    return queries[0] if queries else ""


# Normalize the public effort argument to one of low/medium/high (default medium).
def coerce_search_effort(value: Any = None) -> str:
    if isinstance(value, dict):
        value = value.get("effort")
    elif isinstance(value, str):
        parsed = _try_parse_json(value)
        if isinstance(parsed, dict):
            value = parsed.get("effort")
    effort = _EFFORT_ALIASES.get(str(value or "").strip().lower(), str(value or "").strip().lower())
    return effort if effort in _EFFORT_VALUES else "medium"


# Normalize an explicit boolean opt-in argument (shopping / academic) to a bool.
def _coerce_bool_opt(value: Any, key: str) -> bool:
    if isinstance(value, dict):
        value = value.get(key, False)
    elif isinstance(value, str):
        parsed = _try_parse_json(value)
        if isinstance(parsed, dict):
            value = parsed.get(key, False)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value) if isinstance(value, (int, float)) else False


# Normalize the explicit shopping opt-in argument to a bool.
def coerce_search_shopping(value: Any = None) -> bool:
    return _coerce_bool_opt(value, "shopping")


# Normalize the explicit academic opt-in argument to a bool.
def coerce_search_academic(value: Any = None) -> bool:
    return _coerce_bool_opt(value, "academic")


# Normalize the onion opt-in, AND-gated on the tor capability: even if a caller passes
# onion=true, it is honored only when tor.enabled — so the intent flag can never activate
# the path while the capability is off (double-guards a stale/cached schema).
def coerce_search_onion(value: Any = None) -> bool:
    if not _coerce_bool_opt(value, "onion"):
        return False
    try:
        from core.config import load_search_config

        return bool(load_search_config().tor.enabled)
    except Exception:  # noqa: BLE001
        return False


def _public_search_request(request: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in request.items() if key != "queries"}
    public["queries"] = [
        {key: value for key, value in query.items() if key != "timelimit"}
        for query in request.get("queries", [])
        if isinstance(query, dict)
    ]
    return public


def _recover_public_query_wrappers(arguments: Any) -> Any:
    """Recover the common model hallucination ``{"item": ...}`` without losing a batch."""

    if not isinstance(arguments, dict):
        return arguments
    normalized = dict(arguments)
    for vertical in ("web", "academic", "shopping", "onion"):
        value = normalized.get(vertical)
        if not isinstance(value, dict):
            continue
        for wrapper_key in ("item", "items"):
            if set(value) == {wrapper_key} and isinstance(value.get(wrapper_key), (str, list)):
                normalized[vertical] = value[wrapper_key]
                break
    return normalized


def invalid_search_plan_result(
    issues: list[dict[str, str]],
    *,
    description: str = "",
    received_arguments: Any = None,
) -> dict[str, Any]:
    details = "; ".join(
        f"{issue.get('path', '$')}: {issue.get('message', 'invalid value')}"
        for issue in issues
    )
    try:
        received = json.dumps(
            received_arguments,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )[:1600]
    except Exception:  # noqa: BLE001
        received = str(received_arguments)[:1600]
    message = (
        "INVALID_SEARCH_PLAN: web_search rejected this call before running any search.\n"
        f"Failures: {details}\n"
        "Required shape: query as one non-empty string or up to three strings; optional "
        "vertical (web/academic/shopping), effort (low/medium/high), description (UI label).\n"
        'Valid examples: {"query":"project release notes"} or '
        '{"query":"steel thermal conductivity","vertical":"academic"}\n'
        f"Received arguments: {received}\n"
        "Correct the exact failures above before the next call. Never repeat the rejected shape."
    )
    return {
        "error": {"code": "INVALID_SEARCH_PLAN", "issues": issues},
        "sources": [],
        "model_context": message,
        "ui": {
            "kind": "web_search",
            "status": "rejected",
            "description": sanitize_query(description)[:80],
            "result_count": 0,
            "query_count": 0,
            "error": {"code": "INVALID_SEARCH_PLAN", "issues": issues},
        },
    }


def prepare_instant_search_arguments(arguments: Any) -> dict[str, Any]:
    """Validate the compact instant shape and compile it into a low-effort web plan."""

    args = dict(arguments) if isinstance(arguments, dict) else {}
    args.pop("_instant_mode", None)
    issues: list[dict[str, str]] = []
    for key in sorted(set(args) - {"query", "description", "operators"}):
        issues.append({"path": f"$.{key}", "message": "is not allowed in instant mode"})

    raw_description = args.get("description", "")
    description = sanitize_query(raw_description)[:80] if isinstance(raw_description, str) else ""
    if not description:
        issues.append({"path": "$.description", "message": "must be a non-empty string"})

    raw_query = args.get("query")
    raw_queries = list(raw_query) if isinstance(raw_query, list) else [raw_query]
    if not raw_queries or raw_queries == [None]:
        issues.append({"path": "$.query", "message": "must contain at least one query"})
    if len(raw_queries) > INSTANT_SEARCH_BATCH_LIMIT:
        issues.append({
            "path": "$.query",
            "message": f"instant mode permits at most {INSTANT_SEARCH_BATCH_LIMIT} queries",
        })

    operators = args.get("operators", {})
    if operators in (None, ""):
        operators = {}
    if not isinstance(operators, dict):
        issues.append({"path": "$.operators", "message": "must be an object"})
        operators = {}

    if issues:
        error_result = invalid_search_plan_result(
            issues,
            description=description,
            received_arguments=args,
        )
        details = "; ".join(
            f"{issue.get('path', '$')}: {issue.get('message', 'invalid value')}"
            for issue in issues
        )
        error_result["model_context"] = (
            "INVALID_SEARCH_PLAN: instant web_search rejected this call before running. "
            f"Failures: {details}. Required shape: query (one string or up to three strings), "
            "description (short UI label), and optional operators. Do not send effort or verticals."
        )
        error_result["ui"]["instant_mode"] = True
        return {
            "ok": False,
            "arguments": {},
            "tool_ui": error_result["ui"],
            "error_result": error_result,
        }

    # Keep plain instant queries plain when no shared operator object was supplied.
    # Models commonly put a quoted phrase directly in a query string; the advanced
    # validator accepts that public string form, while its structured form correctly
    # reserves operators for the separate ``operators`` object.
    structured_queries = [
        {"text": query, "operators": dict(operators)} if operators else query
        for query in raw_queries
    ]
    advanced_arguments = {
        "call_description": description,
        "web": structured_queries[0] if len(structured_queries) == 1 else structured_queries,
        "effort": "low",
    }
    try:
        from core.config import load_search_config

        prepared = prepare_advanced_search(
            advanced_arguments,
            query_config=load_search_config().query,
            tor_enabled=False,
            allow_structured_queries=True,
            enforce_word_limits=False,
            allow_multiple_queries=True,
            max_queries=INSTANT_SEARCH_BATCH_LIMIT,
        )
    except PlanValidationError as exc:
        error_result = invalid_search_plan_result(
            exc.issues,
            description=description,
            received_arguments=args,
        )
        details = "; ".join(
            f"{issue.get('path', '$')}: {issue.get('message', 'invalid value')}"
            for issue in exc.issues
        )
        error_result["model_context"] = (
            "INVALID_SEARCH_PLAN: instant web_search rejected this call before running. "
            f"Failures: {details}. Correct query, description, or operators without adding "
            "effort or vertical fields."
        )
        error_result["ui"]["instant_mode"] = True
        return {
            "ok": False,
            "arguments": {},
            "tool_ui": error_result["ui"],
            "error_result": error_result,
        }

    search_request = prepared["search_request"]
    search_request["schema_mode"] = "instant"
    public_request = _public_search_request(search_request)
    canonical_query: str | list[str] = (
        str(raw_queries[0]) if len(raw_queries) == 1 else [str(query) for query in raw_queries]
    )
    canonical_arguments: dict[str, Any] = {
        "query": canonical_query,
        "description": description,
    }
    if operators:
        canonical_arguments["operators"] = dict(operators)
    return {
        "ok": True,
        "arguments": canonical_arguments,
        "search_request": search_request,
        "warnings": list(prepared.get("warnings") or []),
        "tool_ui": {
            "kind": "web_search",
            "status": "pending",
            "instant_mode": True,
            "description": description,
            "query_count": len(raw_queries),
            "search_request": public_request,
        },
    }


def prepare_search_arguments(arguments: Any, *, instant_mode: bool = False) -> dict[str, Any]:
    """Return canonical arguments and normalized UI data before network work."""

    if instant_mode or (isinstance(arguments, dict) and arguments.get("_instant_mode") is True):
        return prepare_instant_search_arguments(arguments)

    if search_schema_mode() == "advanced":
        normalized_arguments = _recover_public_query_wrappers(arguments)
        simple_shape = isinstance(normalized_arguments, dict) and "query" in normalized_arguments
        if simple_shape:
            # Keep the existing plan/execution pipeline behind a compact public interface.
            args = normalized_arguments
            vertical = args.get("vertical", "web")
            allowed_verticals = {"web", "academic", "shopping", "onion"}
            issues = []
            for key in sorted(set(args) - {"query", "vertical", "effort", "description"}):
                issues.append({"path": f"$.{key}", "message": "is not allowed"})
            if not isinstance(vertical, str) or vertical not in allowed_verticals:
                issues.append({"path": "$.vertical", "message": "must be web, academic, shopping, or enabled onion"})
            if issues:
                error = invalid_search_plan_result(issues, received_arguments=arguments)
                return {"ok": False, "arguments": {}, "tool_ui": error["ui"], "error_result": error}
            normalized_arguments = {
                "call_description": args.get("description", "Search sources"),
                vertical: args.get("query"),
                "effort": args.get("effort", "medium"),
            }
        try:
            from core.config import load_search_config

            cfg = load_search_config()
            prepared = prepare_advanced_search(
                normalized_arguments,
                query_config=cfg.query,
                tor_enabled=bool(cfg.tor.enabled),
                allow_structured_queries=False,
                enforce_word_limits=False,
                allow_multiple_queries=True,
            )
        except PlanValidationError as exc:
            issues = exc.issues
            if simple_shape:
                issues = [dict(issue) for issue in issues]
                for issue in issues:
                    path = issue["path"]
                    if path == "$.call_description":
                        issue["path"] = "$.description"
                    elif path == f"$.{vertical}" or path.startswith(f"$.{vertical}["):
                        issue["path"] = "$.query" + path[len(vertical) + 2:]
            raw_description = (
                arguments.get("description", arguments.get("call_description", ""))
                if isinstance(arguments, dict) else ""
            )
            error_result = invalid_search_plan_result(
                issues,
                description=raw_description if isinstance(raw_description, str) else "",
                received_arguments=arguments,
            )
            return {
                "ok": False,
                "arguments": {},
                "tool_ui": error_result["ui"],
                "error_result": error_result,
            }
        request = _public_search_request(prepared["search_request"])
        warnings = list(prepared.get("warnings") or [])
        return {
            "ok": True,
            "arguments": (
                {"query": prepared["canonical_arguments"][vertical], "vertical": vertical,
                 "description": request["description"], "effort": request["effort"]}
                if simple_shape else prepared["canonical_arguments"]
            ),
            "search_request": prepared["search_request"],
            "warnings": warnings,
            "tool_ui": {
                "kind": "web_search",
                "status": "pending",
                "description": request["description"],
                "query_count": len(request["queries"]),
                "search_request": request,
                **({"warnings": warnings} if warnings else {}),
            },
        }

    args = arguments if isinstance(arguments, dict) else {}
    raw_query = args.get("query", "")
    shopping = coerce_search_shopping(args)
    academic = coerce_search_academic(args)
    onion = coerce_search_onion(args)
    vertical = "shopping" if shopping else ("academic" if academic else ("onion" if onion else "web"))
    query_limit = VERTICAL_QUERY_LIMITS[vertical]
    query_count = _search_query_item_count(raw_query)
    if query_count > query_limit:
        error_result = invalid_search_plan_result(
            [
                {
                    "path": "$.query",
                    "message": f"{vertical} permits at most {query_limit} queries per call",
                }
            ]
        )
        return {
            "ok": False,
            "arguments": {},
            "tool_ui": error_result["ui"],
            "error_result": error_result,
        }
    queries = coerce_search_queries(raw_query, limit=query_limit)
    canonical_query: str | list[str] = queries[0] if len(queries) == 1 else queries
    effort = coerce_search_effort(args)
    canonical: dict[str, Any] = {
        "query": canonical_query,
        "effort": effort,
        "shopping": shopping,
        "academic": academic,
    }
    if onion:
        canonical["onion"] = True
    request = {
        "schema_mode": "legacy",
        "effort": effort,
        "queries": [
            {
                "vertical": vertical,
                "compiled_query": query,
                "operators": {},
            }
            for query in queries
        ],
    }
    return {
        "ok": True,
        "arguments": canonical,
        "search_request": request,
        "tool_ui": {
            "kind": "web_search",
            "status": "pending",
            "query_count": len(queries),
            "search_request": request,
        },
    }
