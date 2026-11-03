# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json

from core.config import api_keys


# Verify that all four hosted provider keys (Tavily, Firecrawl, Brave, SerpAPI) are loaded from the
# generated JSON document.
def test_load_api_keys_reads_generated_json(tmp_path) -> None:
    path = tmp_path / "api_keys.json"
    path.write_text(
        json.dumps(
            {
                "search": {
                    "hosted_api": {
                        "tavily_api_key": "tavily",
                        "firecrawl_api_key": "firecrawl",
                        "brave_api_key": "brave",
                        "serpapi_api_key": "serpapi",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = api_keys.load_api_keys(path)

    assert loaded.search.hosted_api.tavily_api_key == "tavily"
    assert loaded.search.hosted_api.firecrawl_api_key == "firecrawl"
    assert loaded.search.hosted_api.brave_api_key == "brave"
    assert loaded.search.hosted_api.serpapi_api_key == "serpapi"


# Verify that a missing JSON file yields an empty key set and that no replacement file is created.
def test_load_api_keys_does_not_bootstrap_missing_file(tmp_path) -> None:
    path = tmp_path / "api_keys.json"

    loaded = api_keys.load_api_keys(path)

    assert loaded.search.hosted_api.tavily_api_key is None
    assert not path.exists()


# Verify that a blank provider value is normalized to None, which disables that hosted integration.
def test_load_api_keys_normalizes_blank_values(tmp_path) -> None:
    path = tmp_path / "api_keys.json"
    path.write_text(
        json.dumps({"search": {"hosted_api": {"tavily_api_key": "   "}}}),
        encoding="utf-8",
    )

    loaded = api_keys.load_api_keys(path)

    assert loaded.search.hosted_api.tavily_api_key is None
