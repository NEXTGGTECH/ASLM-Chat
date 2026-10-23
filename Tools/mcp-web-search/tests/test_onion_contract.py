# Copyright NEXTGGTECH. Elastic License 2.0.

"""The onion opt-in is advertised in the tool schema and honored ONLY when tor.enabled —
the capability config gates the model-facing intent flag."""

from __future__ import annotations

import core.config as cfgmod
from core.config.settings import QuerySection, SearchConfig, TorSection
from core.mcp_contract import build_search_schema, coerce_search_onion


def _tor(monkeypatch, enabled: bool, *, mode: str = "advanced"):
    monkeypatch.setattr(cfgmod, "load_search_config",
                        lambda *a, **k: SearchConfig(
                            query=QuerySection(schema_mode=mode),
                            tor=TorSection(enabled=enabled),
                        ))


def test_schema_hides_onion_when_disabled(monkeypatch):
    _tor(monkeypatch, False)
    assert "onion" not in build_search_schema()["properties"]["vertical"]["enum"]
    assert coerce_search_onion({"onion": True}) is False   # AND-gate: capability off


def test_schema_shows_onion_when_enabled(monkeypatch):
    _tor(monkeypatch, True)
    assert "onion" in build_search_schema()["properties"]["vertical"]["enum"]
    assert coerce_search_onion({"onion": True}) is True
    assert coerce_search_onion({"onion": False}) is False


def test_base_schema_unaffected(monkeypatch):
    _tor(monkeypatch, True, mode="legacy")
    assert {"query", "effort", "shopping", "academic"} <= set(build_search_schema()["properties"])
    assert build_search_schema()["properties"]["onion"]["type"] == "boolean"
