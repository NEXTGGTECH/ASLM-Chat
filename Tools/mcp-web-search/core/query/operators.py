# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal


# Schema, validation, compilation and scoring metadata for one operator field. The specs below are
# the single source of truth for every supported search operator. The fields are the operator key,
# the kind of value it takes, a description, how it compiles into the query string, its query
# prefix (if any), the item and length limits, the normalizer applied to values, and whether its
# values count as content terms.
@dataclass(frozen=True)
class SearchOperatorSpec:
    key: str
    value_kind: Literal["list", "groups", "date"]
    description: str
    compile_kind: Literal[
        "quoted", "or", "or_groups", "exclude", "prefix", "or_prefix", "date"
    ]
    prefix: str = ""
    max_items: int = 0
    max_length: int = 0
    group_max_items: int = 0
    normalizer: Literal["text", "domain", "file_type"] = "text"
    content_value: bool = False


# All supported operators, in the order they are documented and compiled.
SEARCH_OPERATOR_SPECS: tuple[SearchOperatorSpec, ...] = (
    SearchOperatorSpec(
        "exact_phrases", "list",
        "Fixed names, titles, errors, or quotations that need verbatim matching.",
        "quoted", max_items=4, max_length=120, content_value=True,
    ),
    SearchOperatorSpec(
        "or_terms", "list",
        "Interchangeable terms compiled into one explicit OR group, keeping one intent together.",
        "or", max_items=8, max_length=80, content_value=True,
    ),
    SearchOperatorSpec(
        "or_groups", "groups",
        "Independent alternative groups; each inner list becomes (term OR term), and groups are ANDed.",
        "or_groups", max_items=4, group_max_items=6, max_length=80, content_value=True,
    ),
    SearchOperatorSpec(
        "exclude_terms", "list",
        "Known ambiguities or noisy meanings excluded with a leading minus.",
        "exclude", prefix="-", max_items=6, max_length=80,
    ),
    SearchOperatorSpec(
        "site_include", "list",
        "Fully qualified domains selected by the research plan and restricted with site:.",
        "or_prefix", prefix="site:", max_items=4, max_length=253, normalizer="domain",
    ),
    SearchOperatorSpec(
        "site_exclude", "list",
        "Known irrelevant fully qualified domains excluded with -site:.",
        "prefix", prefix="-site:", max_items=4, max_length=253, normalizer="domain",
    ),
    SearchOperatorSpec(
        "file_types", "list",
        "Required document or dataset formats restricted with filetype:.",
        "or_prefix", prefix="filetype:", max_items=3, max_length=12, normalizer="file_type",
    ),
    SearchOperatorSpec(
        "title_terms", "list",
        "Signals for the planned page class required in its title with intitle:.",
        "prefix", prefix="intitle:", max_items=4, max_length=80, content_value=True,
    ),
    SearchOperatorSpec(
        "url_terms", "list",
        "Signals for a planned documentation, issue, or section path required with inurl:.",
        "prefix", prefix="inurl:", max_items=4, max_length=80, content_value=True,
    ),
    SearchOperatorSpec(
        "after", "date",
        "Lower publication-date bound, used when the requested answer materially depends on recency.",
        "date", prefix="after:"
    ),
    SearchOperatorSpec(
        "before", "date",
        "Upper publication-date bound, used when the requested answer needs a real time boundary.",
        "date", prefix="before:"
    ),
)

# Lookup of the spec by operator key.
SEARCH_OPERATOR_BY_KEY = {spec.key: spec for spec in SEARCH_OPERATOR_SPECS}

# Query prefixes grouped by role: operators that do not carry content terms (the bare "-" exclusion
# is left out), operators whose values are content terms, and the date bounds.
NON_CONTENT_OPERATOR_PREFIXES = tuple(
    spec.prefix
    for spec in SEARCH_OPERATOR_SPECS
    if spec.prefix and not spec.content_value and spec.prefix != "-"
)
CONTENT_OPERATOR_PREFIXES = tuple(
    spec.prefix for spec in SEARCH_OPERATOR_SPECS if spec.prefix and spec.content_value
)
DATE_OPERATOR_PREFIXES = tuple(
    spec.prefix for spec in SEARCH_OPERATOR_SPECS if spec.value_kind == "date"
)

# Regex that detects an operator in a raw query (case-insensitive): a known prefix (optionally
# negated), a quote, an OR keyword, or a minus-prefixed word, each at a word boundary.
_RECOGNIZED_PREFIX_PATTERN = "|".join(
    re.escape(spec.prefix.lstrip("-"))
    for spec in SEARCH_OPERATOR_SPECS
    if spec.prefix not in {"", "-"}
)
SEARCH_OPERATOR_RE = re.compile(
    rf'(?:^|\s)(?:-?(?:{_RECOGNIZED_PREFIX_PATTERN})|\")'
    r'|(?:^|\s)OR(?:\s|$)|(?:^|\s)-\w',
    re.IGNORECASE,
)


# Return True when the query already contains any search operator (quotes, OR, minus, site:, ...).
def has_search_operators(query: str) -> bool:
    return bool(SEARCH_OPERATOR_RE.search(str(query or "")))
