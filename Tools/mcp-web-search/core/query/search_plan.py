# Copyright NEXTGGTECH. Elastic License 2.0.

"""Strict, deterministic preparation of advanced web-search plans."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any
from urllib.parse import urlsplit

from core.query.operators import (
    SEARCH_OPERATOR_BY_KEY,
    SEARCH_OPERATOR_SPECS,
    SearchOperatorSpec,
    has_search_operators,
)
from core.search.query_dates import resolve_query_dates


VERTICAL_QUERY_LIMITS = {
    "web": 3,
    "shopping": 3,
    "academic": 3,
    "onion": 3,
}
VERTICAL_WORD_LIMITS = {
    "web": 10,
    "shopping": 4,
    "academic": 8,
    "onion": 7,
}
ADVANCED_BATCH_LIMIT = 3
COMPILED_QUERY_LIMIT = 512
DESCRIPTION_LIMIT = 80
_EFFORTS = ("low", "medium", "high")
_BASE_VERTICALS = ("web", "academic", "shopping")
_OPERATOR_KEYS = tuple(SEARCH_OPERATOR_BY_KEY)
_LIST_SPECS = tuple(spec for spec in SEARCH_OPERATOR_SPECS if spec.value_kind == "list")
_GROUP_SPECS = tuple(spec for spec in SEARCH_OPERATOR_SPECS if spec.value_kind == "groups")
_SPACE_RE = re.compile(r"\s+")
_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$",
    re.IGNORECASE,
)
_FILETYPE_RE = re.compile(r"^[a-z0-9]{1,12}$", re.IGNORECASE)

class PlanValidationError(ValueError):
    """One or more structural problems in an advanced search plan."""

    def __init__(self, issues: list[dict[str, str]]):
        self.issues = issues
        super().__init__("INVALID_SEARCH_PLAN")


def _clean_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").replace("\r", " ").replace("\n", " ")).strip()


def _string_schema(
    description: str,
    *,
    max_length: int,
    max_words: int | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {
        "type": "string",
        "minLength": 1,
        "maxLength": max_length,
        "description": description,
    }
    if max_words is not None:
        # Keep the word ceiling in the actual JSON Schema as well as prose. Providers
        # that support constrained tool decoding can reject an overlong value before
        # it reaches preflight; the preparer remains the authoritative fallback.
        schema["pattern"] = rf"^\s*\S+(?:\s+\S+){{0,{max_words - 1}}}\s*$"
    return schema


def build_advanced_search_schema(*, tor_enabled: bool = False) -> dict[str, Any]:
    verticals = [*_BASE_VERTICALS, *(("onion",) if tor_enabled else ())]
    vertical_descriptions = {
        "web": (
            "Official, independent, community, reporting, measurement, and general web evidence. "
            "MUST NOT be used for products, prices, sellers, stock, or availability; use shopping. "
            "MUST NOT be used for scholarly literature; use academic."
        ),
        "academic": (
            "MUST be used for papers, citations, DOI records, preprints, peer-reviewed evidence, "
            "and primary scientific literature."
        ),
        "shopping": (
            "MUST be used for products, budgets, prices, sellers, stock, availability, delivery, "
            "and purchase options."
        ),
        "onion": "Censorship-resistant onion sources over Tor when explicitly required.",
    }
    vertical_properties: dict[str, Any] = {}
    for vertical in verticals:
        word_limit = VERTICAL_WORD_LIMITS[vertical]
        query_string = _string_schema(
            f"HARD LIMIT: {word_limit} WORDS TOTAL; {word_limit + 1} WORDS IS INVALID. "
            f"Count every whitespace-separated token before calling. "
            f"{vertical_descriptions[vertical]} Each query must be a plain string.",
            max_length=COMPILED_QUERY_LIMIT,
            max_words=word_limit,
        )
        vertical_properties[vertical] = {
            "oneOf": [
                query_string,
                {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": ADVANCED_BATCH_LIMIT,
                    "items": query_string,
                },
            ],
            "description": (
                f"{vertical_descriptions[vertical]} Pass one query string normally. For two "
                "independent evidence gaps in this vertical, pass an array of exactly two "
                f"strings. The {word_limit}-word limit applies separately to each string."
            ),
        }
    return {
        "type": "object",
        "additionalProperties": False,
        # call_description is mandatory, so two properties are the smallest object that
        # can also carry the required vertical query. The preparer enforces which second
        # property is a vertical because draft-07 cannot express that without combinators.
        "minProperties": 2,
        "properties": {
            "call_description": _string_schema(
                "Required UI-only description of this tool call, not a search query. Write "
                "a short action phrase in the user's language describing what this call is "
                "checking. This field never replaces a non-empty vertical query field.",
                max_length=DESCRIPTION_LIMIT,
            ),
            **vertical_properties,
            "effort": {
                "type": "string",
                "enum": list(_EFFORTS),
                "default": "medium",
                "description": (
                    "Search effort for this single query. Start with medium. Use high only "
                    "after a lower-effort search leaves a specific high-stakes gap."
                ),
            },
        },
        "required": ["call_description"],
    }


def _issue(issues: list[dict[str, str]], path: str, message: str) -> None:
    issues.append({"path": path, "message": message})


def _normalize_domain(value: str) -> str:
    raw = _clean_text(value).lower()
    if "://" in raw:
        parsed = urlsplit(raw)
        raw = parsed.hostname or ""
    else:
        raw = raw.split("/", 1)[0].split(":", 1)[0]
    raw = raw.removeprefix("www.").rstrip(".")
    try:
        raw = raw.encode("idna").decode("ascii")
    except UnicodeError:
        return ""
    return raw if _DOMAIN_RE.fullmatch(raw) else ""


def _normalize_list(
    operators: dict[str, Any], spec: SearchOperatorSpec, path: str,
    issues: list[dict[str, str]],
) -> list[str]:
    value = operators.get(spec.key, [])
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        _issue(issues, path, "must be an array")
        return []
    if len(value) > spec.max_items:
        _issue(issues, path, f"must contain at most {spec.max_items} items")
    output: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value[:spec.max_items]):
        item_path = f"{path}[{index}]"
        if not isinstance(item, str):
            _issue(issues, item_path, "must be a string")
            continue
        text = _clean_text(item)
        if not text:
            _issue(issues, item_path, "must not be empty")
            continue
        if len(text) > spec.max_length:
            _issue(issues, item_path, f"must be at most {spec.max_length} characters")
            continue
        if spec.normalizer == "domain":
            text = _normalize_domain(text)
            if not text:
                _issue(issues, item_path, "must be a fully qualified domain")
                continue
        elif spec.normalizer == "file_type":
            text = text.lower().lstrip(".")
            if not _FILETYPE_RE.fullmatch(text):
                _issue(issues, item_path, "must be a file extension containing only letters or digits")
                continue
        marker = text.casefold()
        if marker not in seen:
            seen.add(marker)
            output.append(text)
    return output


def _normalize_date(value: Any, path: str, issues: list[dict[str, str]]) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str):
        _issue(issues, path, "must be an ISO date string")
        return ""
    text = value.strip()
    try:
        parsed = dt.date.fromisoformat(text)
    except ValueError:
        _issue(issues, path, "must use YYYY-MM-DD")
        return ""
    return parsed.isoformat()


def _normalize_groups(
    operators: dict[str, Any], spec: SearchOperatorSpec, path: str,
    issues: list[dict[str, str]],
) -> list[list[str]]:
    value = operators.get(spec.key, [])
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        _issue(issues, path, "must be an array of arrays")
        return []
    if len(value) > spec.max_items:
        _issue(issues, path, f"must contain at most {spec.max_items} groups")
    output: list[list[str]] = []
    seen_groups: set[tuple[str, ...]] = set()
    for group_index, raw_group in enumerate(value[:spec.max_items]):
        group_path = f"{path}[{group_index}]"
        if not isinstance(raw_group, list):
            _issue(issues, group_path, "must be an array")
            continue
        if len(raw_group) < 2:
            _issue(issues, group_path, "must contain at least two alternatives")
        if len(raw_group) > spec.group_max_items:
            _issue(
                issues,
                group_path,
                f"must contain at most {spec.group_max_items} alternatives",
            )
        group: list[str] = []
        seen: set[str] = set()
        for item_index, item in enumerate(raw_group[:spec.group_max_items]):
            item_path = f"{group_path}[{item_index}]"
            if not isinstance(item, str):
                _issue(issues, item_path, "must be a string")
                continue
            text = _clean_text(item)
            if not text:
                _issue(issues, item_path, "must not be empty")
                continue
            if len(text) > spec.max_length:
                _issue(issues, item_path, f"must be at most {spec.max_length} characters")
                continue
            marker = text.casefold()
            if marker not in seen:
                seen.add(marker)
                group.append(text)
        if len(group) < 2:
            _issue(issues, group_path, "must contain two distinct alternatives")
            continue
        group_marker = tuple(item.casefold() for item in group)
        if group_marker not in seen_groups:
            seen_groups.add(group_marker)
            output.append(group)
    return output


def _quoted(value: str) -> str:
    return f'"{_clean_text(value).replace(chr(34), "")}"'


def _operator_value(value: str) -> str:
    clean = _clean_text(value).replace('"', "")
    return _quoted(clean) if " " in clean else clean


def _or_group(values: list[str], render) -> str:
    rendered = [render(value) for value in values]
    if not rendered:
        return ""
    return rendered[0] if len(rendered) == 1 else f"({' OR '.join(rendered)})"


def _compile_operator(spec: SearchOperatorSpec, value: Any) -> str | list[str]:
    if spec.value_kind == "date":
        return f"{spec.prefix}{value}" if value else ""
    values = list(value or [])
    if spec.compile_kind == "quoted":
        return [_quoted(item) for item in values]
    if spec.compile_kind == "or":
        return _or_group(values, _operator_value)
    if spec.compile_kind == "or_groups":
        return [_or_group(group, _operator_value) for group in values]
    if spec.compile_kind == "exclude":
        return [f"{spec.prefix}{_operator_value(item)}" for item in values]
    if spec.compile_kind == "or_prefix":
        return _or_group(values, lambda item: f"{spec.prefix}{item}")
    if spec.compile_kind == "prefix":
        return [f"{spec.prefix}{_operator_value(item)}" for item in values]
    return ""


def _compile_query(text: str, operators: dict[str, Any], qcfg: object) -> tuple[str, str | None]:
    clean_text, timelimit = resolve_query_dates(text, qcfg)
    parts = [clean_text]
    for spec in SEARCH_OPERATOR_SPECS:
        compiled = _compile_operator(spec, operators.get(spec.key))
        if isinstance(compiled, list):
            parts.extend(compiled)
        elif compiled:
            parts.append(compiled)
    return " ".join(part for part in parts if part), timelimit


def prepare_advanced_search(
    arguments: Any,
    *,
    query_config: object,
    tor_enabled: bool = False,
    allow_structured_queries: bool = True,
    enforce_word_limits: bool = False,
    allow_multiple_queries: bool = True,
    max_queries: int = ADVANCED_BATCH_LIMIT,
) -> dict[str, Any]:
    """Validate and normalize one advanced plan, or raise PlanValidationError."""

    issues: list[dict[str, str]] = []
    if not isinstance(arguments, dict):
        raise PlanValidationError([{"path": "$", "message": "must be an object"}])
    allowed_verticals = [*_BASE_VERTICALS, *(("onion",) if tor_enabled else ())]
    allowed_root_keys = {"call_description", "effort", *allowed_verticals}
    unknown_root = sorted(set(arguments) - allowed_root_keys)
    for key in unknown_root:
        _issue(issues, f"$.{key}", "is not allowed in advanced mode")

    raw_description = arguments.get("call_description")
    if not isinstance(raw_description, str):
        _issue(issues, "$.call_description", "must be a string")
    description = _clean_text(raw_description) if isinstance(raw_description, str) else ""
    if not description:
        _issue(issues, "$.call_description", "must be a non-empty string")
    elif len(description) > DESCRIPTION_LIMIT:
        _issue(
            issues,
            "$.call_description",
            f"must be at most {DESCRIPTION_LIMIT} characters",
        )

    effort_value = arguments.get("effort", "medium")
    if not isinstance(effort_value, str):
        _issue(issues, "$.effort", "must be a string")
    effort = str(effort_value or "").strip().lower()
    if effort not in _EFFORTS:
        _issue(issues, "$.effort", "must be one of low, medium, high")
        effort = "medium"

    raw_queries: list[tuple[str, Any, str]] = []
    saw_vertical = False
    for key, raw_value in arguments.items():
        if key not in allowed_verticals:
            continue
        saw_vertical = True
        if isinstance(raw_value, list):
            if not allow_multiple_queries:
                _issue(
                    issues,
                    f"$.{key}",
                    "must be one plain query string; arrays are not allowed",
                )
                continue
            if not raw_value:
                _issue(issues, f"$.{key}", "must contain at least one query")
                continue
            raw_queries.extend(
                (key, raw_query, f"$.{key}[{index}]")
                for index, raw_query in enumerate(raw_value)
            )
        else:
            raw_queries.append((key, raw_value, f"$.{key}"))

    if not raw_queries and not saw_vertical:
        vertical_names = ", ".join(allowed_verticals)
        _issue(issues, "$", f"must include a query in one of: {vertical_names}")

    warnings: list[dict[str, str]] = []
    if not allow_multiple_queries and len(raw_queries) > 1:
        _issue(
            issues,
            "$",
            "must include exactly one vertical query; submit additional queries as separate calls",
        )
    elif effort == "high" and len(raw_queries) > 1:
        warnings.append(
            {
                "code": "HIGH_EFFORT_BATCH_TRUNCATED",
                "message": (
                    "High effort does not allow batching. Only the first query was executed; "
                    "submit any remaining queries separately with medium or low effort."
                ),
            }
        )
        raw_queries = raw_queries[:1]
    elif len(raw_queries) > max(1, int(max_queries)):
        batch_message = f"batch permits at most {max(1, int(max_queries))} queries total"
        _issue(
            issues,
            "$",
            batch_message,
        )

    prepared_queries: list[dict[str, Any]] = []
    for vertical, raw_query, base_path in raw_queries:
        structured_query = isinstance(raw_query, dict)
        if isinstance(raw_query, str):
            raw_text = raw_query
            raw_operators: Any = {}
        elif structured_query:
            if not allow_structured_queries:
                _issue(
                    issues,
                    base_path,
                    "must be a plain query string; object wrappers are not allowed",
                )
                continue
            for key in sorted(set(raw_query) - {"text", "operators"}):
                _issue(issues, f"{base_path}.{key}", "is not allowed")
            raw_text = raw_query.get("text")
            raw_operators = raw_query.get("operators", {})
        else:
            _issue(issues, base_path, "must be a string")
            continue

        if not isinstance(raw_text, str):
            _issue(issues, f"{base_path}.text", "must be a string")
        text = _clean_text(raw_text) if isinstance(raw_text, str) else ""
        if not text:
            path = f"{base_path}.text" if structured_query else base_path
            _issue(issues, path, "must be a non-empty string")
        elif len(text) > COMPILED_QUERY_LIMIT:
            path = f"{base_path}.text" if structured_query else base_path
            _issue(issues, path, f"must be at most {COMPILED_QUERY_LIMIT} characters")
        elif structured_query and has_search_operators(text):
            _issue(issues, f"{base_path}.text", "must not contain recognized search operators")

        if enforce_word_limits and text:
            words = text.split()
            word_count = len(words)
            word_limit = VERTICAL_WORD_LIMITS[vertical]
            if word_count > word_limit:
                excess = word_count - word_limit
                numbered_words = ", ".join(
                    f"{index}={word}" for index, word in enumerate(words, start=1)
                )
                _issue(
                    issues,
                    base_path,
                    (
                        f"contains {word_count} whitespace-separated words; {vertical} allows at "
                        f"most {word_limit}. Delete at least {excess} word"
                        f"{'s' if excess != 1 else ''} before retrying. Counted words: "
                        f"{numbered_words}"
                    ),
                )

        if raw_operators in (None, ""):
            raw_operators = {}
        if not isinstance(raw_operators, dict):
            _issue(issues, f"{base_path}.operators", "must be an object")
            raw_operators = {}
        for key in sorted(set(raw_operators) - set(_OPERATOR_KEYS)):
            _issue(issues, f"{base_path}.operators.{key}", "is not a supported operator")

        normalized_operators: dict[str, Any] = {}
        for spec in _LIST_SPECS:
            values = _normalize_list(
                raw_operators, spec, f"{base_path}.operators.{spec.key}", issues
            )
            if values:
                normalized_operators[spec.key] = values
        for spec in _GROUP_SPECS:
            groups = _normalize_groups(
                raw_operators, spec, f"{base_path}.operators.{spec.key}", issues
            )
            if groups:
                normalized_operators[spec.key] = groups
        after = _normalize_date(raw_operators.get("after"), f"{base_path}.operators.after", issues)
        before = _normalize_date(raw_operators.get("before"), f"{base_path}.operators.before", issues)
        if after:
            normalized_operators["after"] = after
        if before:
            normalized_operators["before"] = before
        if after and before and after >= before:
            _issue(issues, f"{base_path}.operators", "after must be earlier than before")

        compiled_query, timelimit = _compile_query(text, normalized_operators, query_config)
        if len(compiled_query) > COMPILED_QUERY_LIMIT:
            _issue(
                issues,
                base_path,
                f"compiled query must be at most {COMPILED_QUERY_LIMIT} characters",
            )

        canonical_item: dict[str, Any] = {
            "vertical": vertical,
            "text": text,
        }
        if normalized_operators:
            canonical_item["operators"] = normalized_operators
        prepared_queries.append(
            {
                **canonical_item,
                "operators": normalized_operators,
                "compiled_query": compiled_query,
                "timelimit": timelimit,
            }
        )

    if issues:
        raise PlanValidationError(issues)
    canonical_arguments: dict[str, Any] = {"call_description": description}
    for query in prepared_queries:
        vertical = str(query.get("vertical") or "")
        model_query = str(query.get("compiled_query") or "")
        existing = canonical_arguments.get(vertical)
        if existing is None:
            canonical_arguments[vertical] = model_query
        elif isinstance(existing, list):
            existing.append(model_query)
        else:
            canonical_arguments[vertical] = [existing, model_query]
    canonical_arguments["effort"] = effort

    search_request: dict[str, Any] = {
        "schema_mode": "advanced",
        "description": description,
        "effort": effort,
        "queries": prepared_queries,
    }
    if warnings:
        search_request["warnings"] = warnings
    return {
        "canonical_arguments": canonical_arguments,
        "search_request": search_request,
        "warnings": warnings,
    }
