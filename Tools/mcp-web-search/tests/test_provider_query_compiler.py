# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import asyncio
import datetime as dt

from core.engines.models import EngineParseResult, EngineRequest, ParseStatus
from core.query.provider_compiler import compile_provider_query
from core.search.serp_api import SerpApi


_OPERATORS = {
    "exact_phrases": ["release notes"],
    "or_terms": ["changelog", "announcement"],
    "or_groups": [["Codex CLI", "coding agent"]],
    "exclude_terms": ["old model"],
    "site_include": ["openai.com", "developers.openai.com"],
    "site_exclude": ["forum.example.com"],
    "file_types": ["pdf", "csv"],
    "title_terms": ["release notes"],
    "url_terms": ["changelog"],
    "after": "2026-07-01",
}


def test_google_keeps_full_operator_dialect():
    compiled = compile_provider_query(
        "OpenAI Codex features",
        _OPERATORS,
        "google",
        today=dt.date(2026, 7, 24),
    )

    assert '"release notes"' in compiled.query
    assert "(changelog OR announcement)" in compiled.query
    assert "(site:openai.com OR site:developers.openai.com)" in compiled.query
    assert "intitle:\"release notes\"" in compiled.query
    assert "inurl:changelog" in compiled.query
    assert "after:2026-07-01" in compiled.query
    assert compiled.timelimit is None
    assert compiled.omitted_operators == ()


def test_rolling_date_engines_translate_after_to_native_timelimit():
    for provider in ("duckduckgo", "startpage", "brave"):
        compiled = compile_provider_query(
            "OpenAI Codex features",
            _OPERATORS,
            provider,
            today=dt.date(2026, 7, 24),
        )
        assert "after:" not in compiled.query
        assert compiled.timelimit == "m"
        assert compiled.omitted_operators == ()


def test_yandex_translates_structural_operators_and_exact_date():
    compiled = compile_provider_query(
        "OpenAI Codex features",
        {**_OPERATORS, "before": "2026-07-24"},
        "yandex",
        today=dt.date(2026, 7, 24),
    )

    assert "(changelog | announcement)" in compiled.query
    assert "(mime:pdf | mime:csv)" in compiled.query
    assert "title:" not in compiled.query
    assert '"release notes"' in compiled.query
    assert "url:changelog" in compiled.query
    assert "after:" not in compiled.query
    assert "date:>20260701" in compiled.query
    assert "date:<20260724" in compiled.query
    assert compiled.timelimit is None
    assert compiled.omitted_operators == ()


def test_conservative_providers_do_not_receive_unknown_title_url_or_date_syntax():
    for provider in ("qwant", "legacy_yandex", "tavily", "firecrawl"):
        compiled = compile_provider_query(
            "OpenAI Codex features",
            _OPERATORS,
            provider,
            today=dt.date(2026, 7, 24),
        )
        assert "intitle:" not in compiled.query
        assert "inurl:" not in compiled.query
        assert "after:" not in compiled.query
        assert "release notes" in compiled.query
        assert "changelog" in compiled.query
        assert compiled.omitted_operators == ("after",)


def test_brave_keeps_supported_title_but_not_unadvertised_url_operator():
    compiled = compile_provider_query(
        "OpenAI Codex features",
        _OPERATORS,
        "brave",
        today=dt.date(2026, 7, 24),
    )
    assert 'intitle:"release notes"' in compiled.query
    assert "inurl:" not in compiled.query
    assert "changelog" in compiled.query


class _ParserOne:
    name = "one"
    provider_family = "one"

    @staticmethod
    def build_request(query: str, **_kwargs) -> EngineRequest:
        return EngineRequest(method="GET", url="https://one.example/search", params={"q": query})

    @staticmethod
    def parse(_document: str) -> EngineParseResult:
        return EngineParseResult(engine="one", status=ParseStatus.EMPTY)


class _ParserTwo(_ParserOne):
    name = "two"
    provider_family = "two"

    @staticmethod
    def build_request(query: str, **_kwargs) -> EngineRequest:
        return EngineRequest(method="GET", url="https://two.example/search", params={"q": query})

    @staticmethod
    def parse(_document: str) -> EngineParseResult:
        return EngineParseResult(engine="two", status=ParseStatus.EMPTY)


class _Response:
    status = 200
    body = b"empty"
    transport = "fake"

    @staticmethod
    def text() -> str:
        return "empty"


class _Transport:
    def __init__(self) -> None:
        self.requests: list[EngineRequest] = []

    async def fetch(self, request: EngineRequest):
        self.requests.append(request)
        return _Response()

    async def close(self) -> None:
        return None


def test_serp_api_executes_and_reports_per_engine_queries():
    transport = _Transport()
    api = SerpApi(transport=transport, engines=(_ParserOne, _ParserTwo))

    async def collect():
        return [event async for event in api.search_stream(
            "canonical",
            engine_queries={"one": "query one", "two": "query two"},
            engine_timelimits={"one": "m", "two": None},
            engine_omitted_operators={"two": ("after",)},
        )]

    events = asyncio.run(collect())
    assert {request.params["q"] for request in transport.requests} == {"query one", "query two"}
    payloads = {event["engine"]: event["payload"] for event in events if event["type"] == "engine"}
    assert payloads["one"]["query"] == "query one"
    assert payloads["one"]["timelimit"] == "m"
    assert payloads["two"]["query"] == "query two"
    assert payloads["two"]["omitted_operators"] == ["after"]
