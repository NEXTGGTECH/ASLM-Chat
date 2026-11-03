# Copyright NEXTGGTECH. Elastic License 2.0.

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


# Maximum number of queries and maximum words per query, by vertical.
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

# Limits of an advanced plan: queries per batch, characters of a compiled query, characters of the
# UI-only call description.
ADVANCED_BATCH_LIMIT = 3
COMPILED_QUERY_LIMIT = 512
DESCRIPTION_LIMIT = 80

# Allowed effort levels, and the verticals that are always available (onion needs Tor enabled).
_EFFORTS = ("low", "medium", "high")
_BASE_VERTICALS = ("web", "academic", "shopping")

# Operator keys, and the operator specs that take a plain list or a list of alternative groups.
_OPERATOR_KEYS = tuple(SEARCH_OPERATOR_BY_KEY)
_LIST_SPECS = tuple(spec for spec in SEARCH_OPERATOR_SPECS if spec.value_kind == "list")
_GROUP_SPECS = tuple(spec for spec in SEARCH_OPERATOR_SPECS if spec.value_kind == "groups")

# Patterns: whitespace runs, a fully qualified domain name (at most 253 characters, labels of at most
# 63), and a file extension of 1-12 letters or digits.
_SPACE_RE = re.compile(r"\s+")
_DOMAIN_RE = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$",
    re.IGNORECASE,
)
_FILETYPE_RE = re.compile(r"^[a-z0-9]{1,12}$", re.IGNORECASE)


# Error raised for one or more structural problems in an advanced search plan; issues holds the
# {path, message} entries.
class PlanValidationError(ValueError):
    # Keep the issue list and raise with the fixed INVALID_SEARCH_PLAN code.
    def __init__(self, issues: list[dict[str, str]]):
        self.issues = issues
        super().__init__("INVALID_SEARCH_PLAN")


# Collapse all whitespace (including newlines) to single spaces and trim the text.
def _clean_text(value: Any) -> str:
    return _SPACE_RE.sub(" ", str(value or "").replace("\r", " ").replace("\n", " ")).strip()


# Build the JSON Schema of a non-empty string with a length limit and an optional word limit.
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


# Build the JSON Schema of the advanced search tool arguments: a required call_description, one
# property per vertical (a query string or an array of them, with per-vertical word limits) and an
# effort level. The onion vertical is included only when Tor is enabled.
def build_advanced_search_schema(*, tor_enabled: bool = False) -> dict[str, Any]:
    verticals = [*_BASE_VERTICALS, *(("onion",) if tor_enabled else ())]

    # Model-facing guidance on when each vertical must (or must not) be used.
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

    # One schema property per vertical, each accepting a single query string or an array of them.
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

    # Assemble the object schema around the call description, the verticals and the effort.
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


# Record one validation problem at a JSON path.
def _issue(issues: list[dict[str, str]], path: str, message: str) -> None:
    issues.append({"path": path, "message": message})


# Reduce a URL or host to a bare IDNA-encoded domain without "www." and any port or path. Return an
# empty string when the result is not a fully qualified domain.
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


# Validate and normalize one list-valued operator: check the item count, text and length limits,
# apply the spec's normalizer (domain or file extension) and drop case-insensitive duplicates.
# Problems are appended to issues and the offending items are skipped.
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

    # Check every item up to the limit; skip invalid ones and duplicates.
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

        # Apply the spec's normalizer.
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

        # Keep the first occurrence of each case-insensitive value.
        marker = text.casefold()
        if marker not in seen:
            seen.add(marker)
            output.append(text)
    return output


# Validate a YYYY-MM-DD date value and return it in ISO form; an empty value gives an empty
# string, and an invalid one records an issue and gives an empty string.
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


# Validate and normalize a group-valued operator (a list of alternative groups): each group needs
# two to group_max_items distinct non-empty alternatives, and duplicate groups are dropped.
# Problems are appended to issues and the offending items or groups are skipped.
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

    # Check every group up to the limit.
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

        # Collect the distinct, valid alternatives of this group.
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

        # A group needs two distinct alternatives; skip groups already seen.
        if len(group) < 2:
            _issue(issues, group_path, "must contain two distinct alternatives")
            continue
        group_marker = tuple(item.casefold() for item in group)
        if group_marker not in seen_groups:
            seen_groups.add(group_marker)
            output.append(group)
    return output


# Return the cleaned text wrapped in double quotes, with embedded quotes removed.
def _quoted(value: str) -> str:
    return f'"{_clean_text(value).replace(chr(34), "")}"'


# Return the cleaned text, quoted when it contains a space so it stays one search term.
def _operator_value(value: str) -> str:
    clean = _clean_text(value).replace('"', "")
    return _quoted(clean) if " " in clean else clean


# Join the rendered values with OR inside parentheses (a single value is returned bare, no values
# give an empty string).
def _or_group(values: list[str], render) -> str:
    rendered = [render(value) for value in values]
    if not rendered:
        return ""
    return rendered[0] if len(rendered) == 1 else f"({' OR '.join(rendered)})"


# Compile one normalized operator value into its Google-style query text: a string for single-part
# operators, a list of strings for operators that contribute several parts, or an empty string.
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


# Compile the query text (with its year hint resolved by the query config) and the operators into
# the canonical query string; return it with the time limit derived from the text, if any.
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


# Validate and deterministically normalize one advanced search plan, or raise PlanValidationError
# with every problem found. Return the canonical arguments (what the model effectively asked for),
# the search request (description, effort and the prepared queries with their operators, compiled
# query and time limit) and any warnings.
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
    issues: list[dict[str, str]] = []
    if not isinstance(arguments, dict):
        raise PlanValidationError([{"path": "$", "message": "must be an object"}])

    # Reject unknown top-level keys.
    allowed_verticals = [*_BASE_VERTICALS, *(("onion",) if tor_enabled else ())]
    allowed_root_keys = {"call_description", "effort", *allowed_verticals}
    unknown_root = sorted(set(arguments) - allowed_root_keys)
    for key in unknown_root:
        _issue(issues, f"$.{key}", "is not allowed in advanced mode")

    # Validate the UI-only call description.
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

    # Validate the effort level; an invalid one is reported and replaced by medium.
    effort_value = arguments.get("effort", "medium")
    if not isinstance(effort_value, str):
        _issue(issues, "$.effort", "must be a string")
    effort = str(effort_value or "").strip().lower()
    if effort not in _EFFORTS:
        _issue(issues, "$.effort", "must be one of low, medium, high")
        effort = "medium"

    # Flatten the vertical properties into (vertical, raw query, JSON path) entries, expanding arrays.
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

    # Enforce the batch rules: a single query when batching is off, one query at high effort (extra
    # queries are dropped with a warning), and the maximum batch size otherwise.
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

    # Prepare each query: unwrap its text and operators, validate them, and compile the final query.
    prepared_queries: list[dict[str, Any]] = []
    for vertical, raw_query, base_path in raw_queries:
        # A query is a plain string, or (when allowed) an object with text and operators.
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

        # Validate the query text: non-empty, within the length limit and, for structured queries,
        # free of operators (those belong in the operators object).
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

        # Enforce the per-vertical word limit, listing the numbered words so the model can trim them.
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

        # Validate the operators object and normalize each operator by its kind.
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

        # Date bounds must be valid and ordered.
        after = _normalize_date(raw_operators.get("after"), f"{base_path}.operators.after", issues)
        before = _normalize_date(raw_operators.get("before"), f"{base_path}.operators.before", issues)
        if after:
            normalized_operators["after"] = after
        if before:
            normalized_operators["before"] = before
        if after and before and after >= before:
            _issue(issues, f"{base_path}.operators", "after must be earlier than before")

        # Compile the final query and keep it within the length limit.
        compiled_query, timelimit = _compile_query(text, normalized_operators, query_config)
        if len(compiled_query) > COMPILED_QUERY_LIMIT:
            _issue(
                issues,
                base_path,
                f"compiled query must be at most {COMPILED_QUERY_LIMIT} characters",
            )

        # Record the prepared query.
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

    # Rebuild the canonical arguments from the compiled queries: one string per vertical, or an
    # array when a vertical has several queries.
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

    # Build the search request handed to the search pipeline.
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
