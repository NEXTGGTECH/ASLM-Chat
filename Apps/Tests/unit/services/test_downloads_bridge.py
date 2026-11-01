# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import io
import json
import sys

import pytest

from Services import downloads_bridge


pytestmark = [pytest.mark.unit]


SEARCH_PAGE_HTML = """
<html><body>
<ul role="list">
  <li>
    <a href="/library/llama3.2">
      <div title="Llama 3.2 card">
        <h2>Llama 3.2</h2>
        <p>Llama 3.2 is a small instruction-tuned model.</p>
      </div>
    </a>
    <div class="flex flex-wrap">
      <span>3B</span><span>1B</span><span>tools</span><span>vision</span>
    </div>
    <span class="flex items-center">263.3K Pulls</span>
    <span class="flex items-center">14 Tags</span>
    <span class="flex items-center">Updated 5 weeks ago</span>
  </li>
  <li>
    <a href="/library/llama3.2:1b"><h2>Tag deep link</h2></a>
  </li>
  <li>
    <a href="/blog/notes"><h2>Non-library link</h2></a>
  </li>
</ul>
</body></html>
"""


DETAIL_PAGE_HTML = """
<html><head><title>Llama 3.2 - Ollama</title></head><body>
<textarea id="summary-textarea">Llama 3.2 summary text</textarea>
<span class="flex items-center">18.7M Downloads</span>
<span class="flex items-center">Updated 2 weeks ago</span>
<a class="sm:hidden" href="/library/llama3.2:1b">
  <p>1B</p><p class="text-neutral-500">1.3GB | text | 1B params</p>
</a>
<div class="group sm:grid">
  <a href="/library/llama3.2:3b">3b</a>
  <p class="col-span-2">2.0GB</p>
  <p class="col-span-2">instructions</p>
  <p class="col-span-2">3B params</p>
</div>
<div class="group sm:grid">
  <a href="/library/llama3.2:1b">1b</a>
  <p class="col-span-2"></p>
  <p class="col-span-2"></p>
</div>
<textarea id="editor"># Model card

Llama 3.2 README body.</textarea>
</body></html>
"""


# Parse markup with the bridge's own soup helper.
def _parse(html: str):
    return downloads_bridge._make_soup(html)


# Build one minimal variant record.
def _variant(key: str, title: str) -> dict:
    return {"resourceKey": key, "title": title}


# Query normalization.


# Mixed filter keys split into sort and capability groups.
def test_normalize_search_query_splits_sort_and_capability_filters():
    query = downloads_bridge._normalize_search_query(
        "  code models  ",
        ["capability:vision", "sort:popular", "capability:tools", "capability:vision", "junk"],
    )

    assert query.query_text == "code models"
    assert query.sort_key == "popular"
    assert query.capabilities == ("tools", "vision")


# An empty query normalizes to the default sort.
def test_normalize_search_query_defaults_to_recent_sort():
    query = downloads_bridge._normalize_search_query("", [])

    assert query.query_text == ""
    assert query.sort_key == downloads_bridge.DEFAULT_SORT_KEY
    assert query.capabilities == ()


# Query parameters encode text, capabilities, and non-default sorts.
def test_build_search_params_encodes_query_filters_and_sort():
    params = downloads_bridge._build_search_params(
        downloads_bridge._normalize_search_query(
            "code", ["sort:newest", "capability:tools", "capability:vision"]
        )
    )

    assert params == [
        ("q", "code"),
        ("c", "tools"),
        ("c", "vision"),
        ("o", "newest"),
    ]

    # The default sort is not sent explicitly.
    default_params = downloads_bridge._build_search_params(
        downloads_bridge._normalize_search_query("code", [])
    )
    assert ("o", "popular") not in default_params


# Resource keys.


# Variant resource keys build and split symmetrically.
def test_variant_resource_key_roundtrip():
    assert downloads_bridge._variant_resource_key("llama3.2", "3b") == "ollama:llama3.2:3b"
    assert downloads_bridge._variant_resource_key("llama3.2", "") == "ollama:llama3.2"
    assert downloads_bridge._resource_key_to_slug("ollama:llama3.2:3b") == "llama3.2:3b"
    assert downloads_bridge._resource_key_to_slug("plain") == "plain"


# Size parsing.


# Size text parses across units and rejects junk.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2.0GB", 2_000_000_000),
        ("500MB", 500_000_000),
        ("1.5GiB", 1_610_612_736),
        ("800B", 800),
        ("junk", None),
        ("", None),
    ],
)
def test_parse_size_bytes(raw, expected):
    assert downloads_bridge._parse_size_bytes(raw) == expected


# Platform variant filtering.


# MLX detection covers the common tag shapes.
def test_mlx_detection_covers_common_tag_shapes():
    assert downloads_bridge._is_mlx_model_ref("ollama:gemma4:12b-mlx") is True
    assert downloads_bridge._is_mlx_model_ref("mlx") is True
    assert downloads_bridge._is_mlx_model_ref("foo-mlx-bar") is True
    assert downloads_bridge._is_mlx_model_ref("12b") is False
    assert downloads_bridge._is_mlx_model_ref("") is False
    assert downloads_bridge._variant_is_mlx(_variant("ollama:m:12b-mlx", "12b-mlx")) is True
    assert downloads_bridge._variant_is_mlx(_variant("ollama:m:12b", "12b")) is False
    assert downloads_bridge._variant_is_mlx(None) is False


# Host platform filtering keeps only matching variants.
def test_filter_variants_for_host_keeps_matching_platform(monkeypatch):
    variants = [
        _variant("ollama:m:12b", "12b"),
        _variant("ollama:m:12b-mlx", "12b-mlx"),
        _variant("ollama:m:latest", "latest"),
    ]

    monkeypatch.setattr(downloads_bridge, "_host_wants_mlx_variants", lambda: False)
    classic = downloads_bridge._filter_variants_for_host(variants)
    assert [item["title"] for item in classic] == ["12b", "latest"]
    assert [item["sortOrder"] for item in classic] == [0, 1]

    monkeypatch.setattr(downloads_bridge, "_host_wants_mlx_variants", lambda: True)
    mlx = downloads_bridge._filter_variants_for_host(variants)
    assert [item["title"] for item in mlx] == ["12b-mlx"]
    assert mlx[0]["sortOrder"] == 0


# The default variant prefers the "latest" tag.
def test_select_default_variant_prefers_latest():
    variants = [_variant("ollama:m:12b", "12b"), _variant("ollama:m:latest", "latest")]
    assert downloads_bridge._select_default_variant_key(variants) == "ollama:m:latest"

    assert downloads_bridge._select_default_variant_key([_variant("ollama:m:12b", "12b")]) == "ollama:m:12b"
    assert downloads_bridge._select_default_variant_key([]) == ""


# Search page parsing.


# The current card markup parses with meta counters and capability chips.
def test_parse_search_items_reads_current_markup():
    items = downloads_bridge._parse_search_items(_parse(SEARCH_PAGE_HTML))

    # Tag deep-links and non-library links are skipped.
    assert len(items) == 1
    item = items[0]
    assert item["resourceKey"] == "ollama:llama3.2"
    assert item["title"] == "Llama 3.2"
    assert item["summary"] == "Llama 3.2 is a small instruction-tuned model."
    assert item["homepageUrl"] == "https://ollama.com/library/llama3.2"
    assert [tag["text"] for tag in item["tags"]] == ["3B", "1B", "tools", "vision"]
    assert item["variantCount"] == 14
    assert item["defaultVariantResourceKey"] == "ollama:llama3.2:latest"

    details_text = json.dumps(item["details"])
    assert "263.3K" in details_text
    assert "5 weeks ago" in details_text


# Foreign markup yields no items instead of raising.
def test_parse_search_items_returns_empty_for_foreign_markup():
    assert downloads_bridge._parse_search_items(_parse("<html><body><p>nothing</p></body></html>")) == []


# Model page parsing.


# The model page yields meta, variants (with mobile fallback), and readme.
def test_parse_item_detail_extracts_meta_variants_and_readme():
    detail = downloads_bridge._parse_item_detail("llama3.2", DETAIL_PAGE_HTML)

    assert detail["resourceKey"] == "ollama:llama3.2"
    assert detail["title"] == "Llama 3.2 - Ollama"
    assert detail["summary"] == "Llama 3.2 summary text"
    assert detail["homepageUrl"] == "https://ollama.com/library/llama3.2"

    variants = detail["variants"]
    assert [variant["title"] for variant in variants] == ["3b", "1b"]
    by_title = {variant["title"]: variant for variant in variants}
    assert by_title["3b"]["size"] == 2_000_000_000
    assert by_title["3b"]["summary"] == "3B params"
    # Missing desktop columns are backfilled from the mobile row metadata.
    assert by_title["1b"]["size"] == 1_300_000_000
    assert by_title["1b"]["summary"] == "1B params"

    # No "latest" tag exists on the fixture page, so the first variant wins.
    assert detail["defaultVariantResourceKey"] == "ollama:llama3.2:3b"

    details_text = json.dumps(detail["details"])
    assert "18.7M" in details_text
    assert "2 weeks ago" in details_text

    readme_block = next(block for block in detail["blocks"] if block["id"] == "readme")
    assert readme_block["format"] == "markdown"
    assert "Llama 3.2 README body." in readme_block["content"]


# Legacy quick-start blocks are dropped from cached details.
def test_sanitize_detail_payload_drops_legacy_quick_start_block():
    detail = {
        "blocks": [
            {"id": "readme", "title": "Readme"},
            {"id": "quick-start", "title": "Quick start"},
        ]
    }

    sanitized = downloads_bridge._sanitize_detail_payload(detail)

    assert [block["id"] for block in sanitized["blocks"]] == ["readme"]


# Install manifests.


# Install manifests target an ollama pull action.
def test_install_manifest_targets_ollama_pull():
    manifest = downloads_bridge._build_install_manifest("ollama:llama3.2:3b")

    assert manifest["resourceKey"] == "ollama:llama3.2:3b"
    action = manifest["actions"][0]
    assert action["type"] == "ollama_pull"
    assert action["model"] == "llama3.2:3b"


# Uninstall manifests target an ollama remove action.
def test_uninstall_manifest_targets_ollama_remove():
    manifest = downloads_bridge._build_uninstall_manifest("ollama:llama3.2")

    assert manifest["actions"][0]["type"] == "ollama_remove"
    assert manifest["actions"][0]["model"] == "llama3.2"


# MLX builds are rejected on classic-only hosts.
def test_install_rejects_mlx_build_on_classic_host(monkeypatch):
    monkeypatch.setattr(downloads_bridge, "_host_wants_mlx_variants", lambda: False)

    with pytest.raises(ValueError, match="MLX"):
        downloads_bridge._build_install_manifest("ollama:gemma4:12b-mlx")

    # Family-level keys are always installable; the UI picks a variant later.
    assert downloads_bridge._build_install_manifest("ollama:gemma4")["title"] == "gemma4"


# An empty resource key fails with a clear error.
def test_build_manifest_requires_resource_key():
    with pytest.raises(ValueError, match="resourceKey"):
        downloads_bridge._build_install_manifest("")


# Dispatch contract.


# list_categories answers with the static Ollama category.
def test_dispatch_lists_categories():
    response = downloads_bridge.dispatch({"operation": "list_categories"})

    assert response["success"] is True
    categories = response["categories"]
    assert len(categories) == 1
    assert categories[0]["id"] == downloads_bridge.OLLAMA_CATEGORY_ID


# Unknown operations and categories fail with explicit errors.
def test_dispatch_rejects_unknown_operations_and_categories():
    response = downloads_bridge.dispatch({"operation": "teleport"})
    assert response["success"] is False
    assert "Unsupported downloads bridge operation" in response["error"]

    response = downloads_bridge.dispatch({"operation": "list_items", "categoryId": "npm"})
    assert response["success"] is False
    assert "Unsupported categoryId" in response["error"]


# list_items delegates to the search loader with normalized flags.
def test_dispatch_list_items_delegates_to_search_loader(monkeypatch):
    monkeypatch.setattr(
        downloads_bridge,
        "_load_search_payload",
        lambda query_text, filters, prefer_cached, force_refresh: (
            [{"resourceKey": "ollama:llama3.2"}],
            [],
            [],
        ),
    )

    response = downloads_bridge.dispatch(
        {
            "operation": "list_items",
            "categoryId": downloads_bridge.OLLAMA_CATEGORY_ID,
            "queryText": "llama",
            "preferCached": True,
        }
    )

    assert response["success"] is True
    assert response["items"][0]["resourceKey"] == "ollama:llama3.2"


# describe_item requires a resource key and delegates to the detail loader.
def test_dispatch_describe_item_requires_resource_key(monkeypatch):
    response = downloads_bridge.dispatch(
        {"operation": "describe_item", "categoryId": downloads_bridge.OLLAMA_CATEGORY_ID}
    )
    assert response["success"] is False
    assert "Missing resourceKey" in response["error"]

    monkeypatch.setattr(
        downloads_bridge,
        "_load_item_detail",
        lambda slug, prefer_cached, force_refresh: ({"resourceKey": f"ollama:{slug}"}, []),
    )
    response = downloads_bridge.dispatch(
        {
            "operation": "describe_item",
            "categoryId": downloads_bridge.OLLAMA_CATEGORY_ID,
            "resourceKey": "ollama:llama3.2:3b",
        }
    )
    assert response["success"] is True
    assert response["itemDetail"]["resourceKey"] == "ollama:llama3.2"


# resolve_install and resolve_uninstall answer with their manifests.
def test_dispatch_resolve_operations_produce_manifests():
    install = downloads_bridge.dispatch(
        {
            "operation": "resolve_install",
            "categoryId": downloads_bridge.OLLAMA_CATEGORY_ID,
            "resourceKey": "ollama:llama3.2:3b",
        }
    )
    assert install["installManifest"]["actions"][0]["type"] == "ollama_pull"

    uninstall = downloads_bridge.dispatch(
        {
            "operation": "resolve_uninstall",
            "categoryId": downloads_bridge.OLLAMA_CATEGORY_ID,
            "resourceKey": "ollama:llama3.2",
        }
    )
    assert uninstall["uninstallManifest"]["actions"][0]["type"] == "ollama_remove"


# CLI contract.


# run_cli answers one stdin request with one JSON response line.
def test_run_cli_answers_single_request(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"operation": "list_categories"}) + "\n")
    )

    exit_code = downloads_bridge.run_cli()

    assert exit_code == 0
    response = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert response["success"] is True
    assert response["categories"][0]["id"] == downloads_bridge.OLLAMA_CATEGORY_ID


# Errors surface as failed responses, not as crashes.
def test_run_cli_reports_errors_as_failed_response(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"operation": "nope"}) + "\n"))

    exit_code = downloads_bridge.run_cli()

    assert exit_code == 0
    response = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert response["success"] is False
    assert "Unsupported downloads bridge operation" in response["error"]


# Empty, broken, and valid stdin map to {}, {}, and the parsed document.
def test_read_request_tolerates_empty_and_broken_input(monkeypatch):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert downloads_bridge._read_request() == {}

    monkeypatch.setattr(sys, "stdin", io.StringIO("{broken"))
    assert downloads_bridge._read_request() == {}

    monkeypatch.setattr(sys, "stdin", io.StringIO('{"operation": "list_categories"}'))
    assert downloads_bridge._read_request()["operation"] == "list_categories"
