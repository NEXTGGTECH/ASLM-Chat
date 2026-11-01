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


def test_instant_arguments_reject_more_than_three_queries(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())

    prepared = prepare_instant_search_arguments({
        "query": ["one", "two", "three", "four"],
        "description": "Checking candidates",
    })

    assert prepared["ok"] is False
    assert prepared["tool_ui"]["instant_mode"] is True


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


def test_config_schema_mode_defaults_and_invalid_values_fall_back_to_advanced(tmp_path, caplog):
    from core.config.settings import load_search_config

    missing = tmp_path / "missing.json"
    assert load_search_config(missing).query.schema_mode == "advanced"

    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps({"query": {"schema_mode": "unknown"}}), encoding="utf-8")
    assert load_search_config(invalid).query.schema_mode == "advanced"
    assert "invalid value" in caplog.text

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"query": {"schema_mode": "legacy"}}), encoding="utf-8")
    assert load_search_config(legacy).query.schema_mode == "legacy"

def test_compact_schema_is_default_and_onion_is_capability_gated(monkeypatch):
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
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config(tor=True))
    assert "onion" in build_search_schema()["properties"]["vertical"]["enum"]


@pytest.mark.parametrize("vertical", ["web", "academic", "shopping"])
def test_compact_search_compiles_and_round_trips(monkeypatch, vertical):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())
    prepared = prepare_search_arguments({"query": "M2 high speed steel properties", "vertical": vertical})
    assert prepared["ok"] is True
    assert prepared["search_request"]["queries"][0]["vertical"] == vertical
    assert prepared["search_request"]["effort"] == "medium"
    assert prepare_search_arguments(prepared["arguments"])["arguments"] == prepared["arguments"]


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


def test_advanced_plan_rejects_legacy_shape_and_is_atomic(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())
    compact = prepare_search_arguments({"query": "compact query"})
    assert compact["ok"] is True
    assert compact["arguments"]["vertical"] == "web"

    plan = _plan()
    plan["web"] = [plan["web"], {"text": "query site:example.com"}]
    with pytest.raises(PlanValidationError) as exc:
        prepare_advanced_search(plan, query_config=_config().query)
    assert any(issue["path"].endswith(".text") for issue in exc.value.issues)


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


def test_public_preflight_accepts_two_query_batch_shapes(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())

    array_result = prepare_search_arguments({
        "call_description": "Check repositories",
        "web": ["first query", "second query"],
        "effort": "medium",
    })
    assert array_result["ok"] is True
    assert array_result["arguments"]["web"] == ["first query", "second query"]

    multi_result = prepare_search_arguments({
        "call_description": "Check two verticals",
        "web": "project documentation",
        "academic": "project evaluation paper",
        "effort": "medium",
    })
    assert multi_result["ok"] is True
    assert multi_result["arguments"]["web"] == "project documentation"
    assert multi_result["arguments"]["academic"] == "project evaluation paper"


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


def test_advanced_plan_requires_call_description_and_rejects_invalid_or_group():
    plan = _plan()
    del plan["call_description"]
    plan["web"]["operators"]["or_groups"] = [["only one"]]

    with pytest.raises(PlanValidationError) as exc:
        prepare_advanced_search(plan, query_config=_config().query)

    paths = {issue["path"] for issue in exc.value.issues}
    assert "$.call_description" in paths
    assert "$.web.operators.or_groups[0]" in paths


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


def test_advanced_plan_accepts_two_queries_in_one_vertical():
    plan = {
        "call_description": "Compare web evidence",
        "web": ["web evidence one", "web evidence two"],
        "effort": "medium",
    }

    prepared = prepare_advanced_search(plan, query_config=_config().query)

    assert [q["vertical"] for q in prepared["search_request"]["queries"]] == ["web", "web"]
    assert isinstance(prepared["canonical_arguments"]["web"], list)


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


def test_explicit_date_operators_survive_legacy_year_hint_processing():
    query = "latest pricing after:2026-01-01 before:2026-07-01"
    clean, _timelimit = resolve_query_dates(query, _config().query)
    assert "after:2026-01-01" in clean
    assert "before:2026-07-01" in clean


def test_mixed_vertical_plan_runs_concurrently_and_preserves_metadata(monkeypatch):
    active = 0
    max_active = 0
    calls: list[tuple[str, dict]] = []

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
    result = asyncio.run(search_module.run_web_search_plan(request))

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


def test_compact_search_executes_through_the_server_plan_pipeline(monkeypatch):
    monkeypatch.setattr(config_module, "load_search_config", lambda: _config())
    path = Path(__file__).resolve().parents[1] / "mcp-server.py"
    spec = importlib.util.spec_from_file_location("compact_search_server_test", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    captured = []

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
