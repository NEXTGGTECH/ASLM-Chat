# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.config as config_module
import core.search.web_search as search_module
from core.mcp_contract import (
    build_search_description,
    build_search_schema,
    prepare_instant_search_arguments,
    prepare_search_arguments,
)
from core.query.search_plan import PlanValidationError, prepare_advanced_search
from core.search.query_dates import resolve_query_dates


# Build a minimal search config stub with the given schema mode and the Tor switch.
def _config(*, mode: str = "advanced", tor: bool = False):
    return SimpleNamespace(
        query=SimpleNamespace(
            schema_mode=mode,
            year_hint_mode="timelimit",
            year_hint_current="m",
            year_hint_prev="y",
            year_hint_older=None,
        ),
        tor=SimpleNamespace(enabled=tor),
    )


# Build a valid advanced search plan that uses every web operator.
def _plan():
    return {
        "call_description": "Convert reference currency",
        "web": {
            "text": "10000 RUB USD exchange rate",
            "operators": {
                "exact_phrases": ["central bank rate"],
                "or_terms": ["official", "reference"],
                "or_groups": [["daily", "monthly"], ["published", "released"]],
                "exclude_terms": ["archive copy"],
                "site_include": ["www.example.com", "docs.example.org"],
                "site_exclude": ["old.example.com"],
                "file_types": [".PDF", "csv"],
                "title_terms": ["exchange rates"],
                "url_terms": ["rates"],
                "after": "2026-07-01",
                "before": "2026-07-22",
            }
        },
        "effort": "medium",
    }


# Verify that instant search arguments are prepared as a low-effort web batch of three queries and
# flagged as instant mode in the tool UI.
def test_instant_arguments_force_low_web_batch(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())

    prepared = prepare_instant_search_arguments({
        "query": ["alpha release", "beta release", "gamma release"],
        "description": "Checking release status",
        "operators": {"site_include": ["example.com"]},
    })

    assert prepared["ok"] is True
    assert prepared["search_request"]["schema_mode"] == "instant"
    assert prepared["search_request"]["effort"] == "low"
    assert len(prepared["search_request"]["queries"]) == 3
    assert prepared["tool_ui"]["instant_mode"] is True


# Verify that instant search arguments with four queries are rejected, still flagged as instant mode.
def test_instant_arguments_reject_more_than_three_queries(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())

    prepared = prepare_instant_search_arguments({
        "query": ["one", "two", "three", "four"],
        "description": "Checking candidates",
    })

    assert prepared["ok"] is False
    assert prepared["tool_ui"]["instant_mode"] is True


# Verify that plain instant queries keep their inline quotes and compile unchanged when no shared
# operators are given.
def test_instant_plain_queries_accept_inline_quotes_without_shared_operators(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())

    prepared = prepare_instant_search_arguments({
        "query": [
            'TerraFirmaGreg "owl beans" spawn',
            'TFC "owl bean" climate',
            'TerraFirmaGreg "owl beans" crop',
        ],
        "description": "Checking owl bean locations",
    })

    assert prepared["ok"] is True
    assert len(prepared["search_request"]["queries"]) == 3
    assert prepared["search_request"]["queries"][0]["compiled_query"] == (
        'TerraFirmaGreg "owl beans" spawn'
    )


# Verify that the schema mode defaults to advanced for a missing file, falls back to advanced with
# a logged warning for an invalid value and accepts the legacy value.
def test_config_schema_mode_defaults_and_invalid_values_fall_back_to_advanced(tmp_path, caplog):
    from core.config.settings import load_search_config

    # A missing config file yields the default mode.
    missing = tmp_path / "missing.json"
    assert load_search_config(missing).query.schema_mode == "advanced"

    # An unknown mode falls back to the default and logs the problem.
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"query": {"schema_mode": "unknown"}}), encoding="utf-8")
    assert load_search_config(invalid).query.schema_mode == "advanced"
    assert "invalid value" in caplog.text

    # The legacy mode is a valid value and is kept.
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"query": {"schema_mode": "legacy"}}), encoding="utf-8")
    assert load_search_config(legacy).query.schema_mode == "legacy"


# Verify that the compact schema is the default (query, vertical, effort, description; a string or
# a batch of up to three queries) and that the onion vertical appears only when Tor is enabled.
def test_compact_schema_is_default_and_onion_is_capability_gated(monkeypatch):
    # Tor disabled: the onion vertical is not offered.
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config(tor=False))
    schema = build_search_schema()
    assert schema["required"] == ["query"]
    assert set(schema["properties"]) == {"query", "vertical", "effort", "description"}
    string_schema, batch_schema = schema["properties"]["query"]["oneOf"]
    assert string_schema["type"] == "string"
    assert "pattern" not in string_schema
    assert batch_schema["maxItems"] == 3
    assert "onion" not in schema["properties"]["vertical"]["enum"]
    assert "query" in build_search_description()

    # Tor enabled: the onion vertical becomes available.
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config(tor=True))
    assert "onion" in build_search_schema()["properties"]["vertical"]["enum"]


# Verify that a compact search compiles into a request for the chosen vertical with medium effort
# and that preparing the resulting arguments again gives the same arguments.
@pytest.mark.parametrize("vertical", ["web", "academic", "shopping"])
def test_compact_search_compiles_and_round_trips(monkeypatch, vertical):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())
    prepared = prepare_search_arguments({"query": "M2 high speed steel properties", "vertical": vertical})
    assert prepared["ok"] is True
    assert prepared["search_request"]["queries"][0]["vertical"] == vertical
    assert prepared["search_request"]["effort"] == "medium"
    assert prepare_search_arguments(prepared["arguments"])["arguments"] == prepared["arguments"]


# Verify that invalid compact arguments (empty or oversized query, unknown or gated vertical or
# effort, non-string description, ambiguous extra field) are rejected with an empty error result
# before any search is executed.
@pytest.mark.parametrize("arguments", [
    {"query": ""}, {"query": ["one", "two", "three", "four"]},
    {"query": "steel", "vertical": "invented"},
    {"query": "steel", "vertical": "onion"},
    {"query": "steel", "effort": "invented"},
    {"query": "steel", "description": 42},
    {"query": "steel", "web": "ambiguous"},
])
def test_compact_search_rejects_invalid_inputs_before_execution(monkeypatch, arguments):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())
    prepared = prepare_search_arguments(arguments)
    assert prepared["ok"] is False
    assert prepared["error_result"]["sources"] == []


# Verify that the advanced compiler turns every operator of the plan into the expected query text
# (normalizing file types and site hosts) and passes the description through.
def test_advanced_compiler_covers_every_operator_and_normalizes_stably():
    prepared = prepare_advanced_search(_plan(), query_config=_config().query)
    query = prepared["search_request"]["queries"][0]["compiled_query"]
    assert query == (
        '10000 RUB USD exchange rate "central bank rate" (official OR reference) '
        '(daily OR monthly) (published OR released) '
        '-"archive copy" (site:example.com OR site:docs.example.org) '
        '-site:old.example.com (filetype:pdf OR filetype:csv) '
        'intitle:"exchange rates" inurl:rates after:2026-07-01 before:2026-07-22'
    )
    assert prepared["canonical_arguments"]["web"] == query
    assert prepared["canonical_arguments"]["call_description"] == "Convert reference currency"
    assert prepared["search_request"]["description"] == "Convert reference currency"


# Verify that a compact query defaults to the web vertical and that an advanced plan with an
# invalid second web entry (a site operator inside the text) is rejected as a whole, with the
# issue reported on the text path.
def test_advanced_plan_rejects_legacy_shape_and_is_atomic(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())
    compact = prepare_search_arguments({"query": "compact query"})
    assert compact["ok"] is True
    assert compact["arguments"]["vertical"] == "web"

    # One valid and one invalid web entry: nothing is accepted.
    plan = _plan()
    plan["web"] = [plan["web"], {"text": "query site:example.com"}]
    with pytest.raises(PlanValidationError) as exc:
        prepare_advanced_search(plan, query_config=_config().query)
    assert any(issue["path"].endswith(".text") for issue in exc.value.issues)


# Verify that preflight keeps the call description in the arguments, the tool UI and the search
# request.
def test_preflight_advertises_normalized_activity_description(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())

    prepared = prepare_search_arguments({
        "call_description": "Convert reference currency",
        "web": "RUB USD exchange rate",
        "effort": "medium",
    })

    assert prepared["arguments"]["call_description"] == "Convert reference currency"
    assert prepared["tool_ui"]["description"] == "Convert reference currency"
    assert prepared["tool_ui"]["search_request"]["description"] == "Convert reference currency"


# Verify that public preflight accepts queries longer than the former per-vertical word limits and
# returns the query unchanged.
@pytest.mark.parametrize(
    ("vertical", "query", "expected_count", "limit"),
    [
        ("web", "one two three four five six seven eight nine ten eleven", 11, 10),
        ("shopping", "one two three four five", 5, 4),
        ("academic", "one two three four five six seven eight nine", 9, 8),
    ],
)
def test_public_preflight_accepts_queries_beyond_former_word_limits(
    monkeypatch, vertical, query, expected_count, limit
):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())
    prepared = prepare_search_arguments({"query": query, "vertical": vertical})
    assert prepared["ok"] is True
    assert prepared["arguments"]["query"] == query


# Verify that a web value wrapped in an "item" object is recovered as a batch of two queries.
def test_public_preflight_recovers_item_wrapper_as_batch(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())
    arguments = {
        "call_description": "Check repositories",
        "web": {"item": ["first query", "second query"]},
        "effort": "medium",
    }

    prepared = prepare_search_arguments(arguments)

    assert prepared["ok"] is True
    assert prepared["arguments"]["web"] == ["first query", "second query"]
    assert prepared["tool_ui"]["query_count"] == 2


# Verify that public preflight accepts both two-query batch shapes: an array in one vertical and
# one query in each of two verticals.
def test_public_preflight_accepts_two_query_batch_shapes(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())

    # Two queries in the web vertical.
    array_result = prepare_search_arguments({
        "call_description": "Check repositories",
        "web": ["first query", "second query"],
        "effort": "medium",
    })
    assert array_result["ok"] is True
    assert array_result["arguments"]["web"] == ["first query", "second query"]

    # One query in each of two verticals.
    multi_result = prepare_search_arguments({
        "call_description": "Check two verticals",
        "web": "project documentation",
        "academic": "project evaluation paper",
        "effort": "medium",
    })
    assert multi_result["ok"] is True
    assert multi_result["arguments"]["web"] == "project documentation"
    assert multi_result["arguments"]["academic"] == "project evaluation paper"


# Verify that a batch of long queries is accepted without any word counting.
def test_public_preflight_accepts_batch_without_word_counting(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())
    result = prepare_search_arguments({
        "call_description": "Compare repositories",
        "web": [
            "one two three four five six seven eight nine ten",
            "one two three four five six seven eight nine ten eleven",
        ],
        "effort": "medium",
    })

    assert result["ok"] is True
    assert len(result["search_request"]["queries"]) == 2


# Verify that a plan without call_description and with a single-term OR group is rejected, with
# both problems reported by path.
def test_advanced_plan_requires_call_description_and_rejects_invalid_or_group():
    plan = _plan()
    del plan["call_description"]
    plan["web"]["operators"]["or_groups"] = [["only one"]]

    with pytest.raises(PlanValidationError) as exc:
        prepare_advanced_search(plan, query_config=_config().query)

    paths = {issue["path"] for issue in exc.value.issues}
    assert "$.call_description" in paths
    assert "$.web.operators.or_groups[0]" in paths


# Verify that the old "description" field is rejected as ambiguous, with issues reported for both
# description and call_description.
def test_advanced_plan_rejects_ambiguous_legacy_description_field():
    with pytest.raises(PlanValidationError) as exc:
        prepare_advanced_search(
            {
                "description": "Ambiguous old field",
                "web": "cve.global project capabilities",
                "effort": "medium",
            },
            query_config=_config().query,
        )

    assert {issue["path"] for issue in exc.value.issues} == {
        "$.description",
        "$.call_description",
    }


# Verify that a plan with four queries in total is rejected as a whole with a single batch-limit
# issue.
def test_advanced_plan_enforces_three_query_total_atomically():
    plan = _plan()
    plan["web"] = [plan["web"], {"text": "second web intent"}]
    plan["academic"] = [{"text": "third academic intent"}, {"text": "fourth academic intent"}]

    with pytest.raises(PlanValidationError) as exc:
        prepare_advanced_search(plan, query_config=_config().query)

    assert exc.value.issues == [{
        "path": "$",
        "message": (
            "batch permits at most 3 queries total"
        ),
    }]


# Verify that a plan with two queries in one vertical compiles into two web queries and keeps the
# list form in the canonical arguments.
def test_advanced_plan_accepts_two_queries_in_one_vertical():
    plan = {
        "call_description": "Compare web evidence",
        "web": ["web evidence one", "web evidence two"],
        "effort": "medium",
    }

    prepared = prepare_advanced_search(plan, query_config=_config().query)

    assert [q["vertical"] for q in prepared["search_request"]["queries"]] == ["web", "web"]
    assert isinstance(prepared["canonical_arguments"]["web"], list)


# Verify that a plan with one query in each of two verticals compiles into one shopping and one
# web query, in that order.
def test_advanced_plan_accepts_one_query_in_each_of_two_verticals():
    plan = {
        "call_description": "Compare prices and tests",
        "shopping": "Steam Deck OLED price",
        "web": "Steam Deck OLED battery test",
        "effort": "medium",
    }

    prepared = prepare_advanced_search(plan, query_config=_config().query)

    assert [q["vertical"] for q in prepared["search_request"]["queries"]] == [
        "shopping",
        "web",
    ]


# Verify that a high-effort batch is cut down to its first query and a truncation warning is
# returned.
def test_public_high_effort_executes_first_query_and_warns(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())
    plan = {
        "call_description": "Check critical evidence",
        "academic": ["first critical study", "second critical study"],
        "effort": "high",
    }

    prepared = prepare_search_arguments(plan)

    assert prepared["ok"] is True
    assert prepared["arguments"]["academic"] == "first critical study"
    assert prepared["warnings"][0]["code"] == "HIGH_EFFORT_BATCH_TRUNCATED"


# Verify that explicit after: and before: operators stay in the query text when the legacy year
# hint processing resolves the query dates.
def test_explicit_date_operators_survive_legacy_year_hint_processing():
    query = "latest pricing after:2026-01-01 before:2026-07-01"
    clean, _timelimit = resolve_query_dates(query, _config().query)
    assert "after:2026-01-01" in clean
    assert "before:2026-07-01" in clean


# Verify that a web plus shopping plan runs both queries concurrently, passes the per-vertical
# flags and operators to the search function, and keeps the description, query indexes, verticals
# and unique source ids in the merged result.
def test_mixed_vertical_plan_runs_concurrently_and_preserves_metadata(monkeypatch):
    # Arrange: a fake search that tracks how many calls overlap and records its arguments.
    active = 0
    max_active = 0
    calls: list[tuple[str, dict]] = []

    # Record the call, wait briefly so concurrent calls overlap, and return one fake source.
    async def fake_search(query: str, **kwargs):
        nonlocal active, max_active
        calls.append((query, kwargs))
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return {
            "search_id": f"id-{len(calls)}",
            "language": "en",
            "region": "us-en",
            "engines_used": ["fake"],
            "cached": False,
            "sources": [{"url": f"https://example.com/{len(calls)}", "host": "example.com", "title": query, "snippet": query}],
        }

    monkeypatch.setattr(search_module, "run_web_search", fake_search)
    request = {
        "schema_mode": "advanced",
        "description": "Compare evidence sources",
        "effort": "medium",
        "queries": [
            {
                "vertical": "web", "text": "alpha", "compiled_query": "alpha after:2026-07-01",
                "operators": {"after": "2026-07-01"}, "timelimit": None,
            },
            {
                "vertical": "shopping", "text": "beta", "compiled_query": "beta",
                "operators": {}, "timelimit": None,
            },
        ],
    }

    # Act: run the plan.
    result = asyncio.run(search_module.run_web_search_plan(request))

    # Assert: both queries overlapped, got their vertical flags and operators, and the result keeps
    # the per-query metadata.
    assert max_active == 2
    assert calls[0][1]["shopping"] is False
    assert calls[1][1]["shopping"] is True
    assert calls[0][1]["vertical_only"] is False
    assert calls[1][1]["vertical_only"] is True
    assert calls[0][1]["query_text"] == "alpha"
    assert calls[0][1]["operators"] == {"after": "2026-07-01"}
    assert result["ui"]["description"] == "Compare evidence sources"
    assert result["ui"]["search_request"]["description"] == "Compare evidence sources"
    assert result["query_results"][1]["index"] == 2
    assert result["query_results"][1]["vertical"] == "shopping"
    assert result["sources"][0]["query_index"] == 1
    assert result["sources"][0]["vertical"] == "web"
    assert len({source["id"] for source in result["sources"]}) == 2


# Verify that a compact three-query academic call to the MCP server is turned into one plan request
# with three academic queries at medium effort and returns the plan result.
def test_compact_search_executes_through_the_server_plan_pipeline(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())

    # Load the server script (its file name contains a dash) as a throwaway module.
    path = Path(__file__).resolve().parents[1] / "mcp-server.py"
    spec = importlib.util.spec_from_file_location("compact_search_server_test", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    captured = []

    # Capture the plan request instead of running a real search.
    async def fake_plan(request):
        captured.append(request)
        return {"sources": [{"url": "https://example.com/paper"}], "model_context": "Evidence"}

    monkeypatch.setattr(server, "run_web_search_plan", fake_plan)
    monkeypatch.setattr(server, "write_search_io_event", lambda *_: None)
    result = asyncio.run(server._call_web_search({
        "query": ["M2 high speed steel density", "M2 high speed steel thermal conductivity", "M2 high speed steel heat capacity"],
        "vertical": "academic",
    }))
    assert result["model_context"] == "Evidence"
    assert len(captured) == 1
    assert [query["vertical"] for query in captured[0]["queries"]] == ["academic", "academic", "academic"]
    assert captured[0]["effort"] == "medium"
