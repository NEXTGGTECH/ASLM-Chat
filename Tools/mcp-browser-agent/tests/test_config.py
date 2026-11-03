# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json

import config


# Verify that every browser-agent setting is read from a generated JSON config file.
def test_load_browser_config_reads_generated_values(tmp_path) -> None:
    # Arrange: a config file that sets every supported key to a non-default value.
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps(
            {
                "browser_width": 1440,
                "browser_height": 900,
                "browser_headless": True,
                "max_a11y_depth": 20,
                "max_elements": 300,
                "max_main_interactive": 80,
                "auto_text_preview_length": 2400,
            }
        ),
        encoding="utf-8",
    )

    # Act: load the file.
    loaded = config._load_browser_config(path)

    # Assert: each value comes from the file, not from the defaults.
    assert loaded["browser_width"] == 1440
    assert loaded["browser_height"] == 900
    assert loaded["browser_headless"] is True
    assert loaded["max_a11y_depth"] == 20
    assert loaded["max_elements"] == 300
    assert loaded["max_main_interactive"] == 80
    assert loaded["auto_text_preview_length"] == 2400


# Verify that settings missing from the config file keep their built-in defaults while present ones win.
def test_load_browser_config_merges_defaults(tmp_path) -> None:
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"browser_width": 1600}), encoding="utf-8")

    loaded = config._load_browser_config(path)

    assert loaded["browser_width"] == 1600
    assert loaded["browser_height"] == config.DEFAULT_CONFIG["browser_height"]
    assert loaded["max_elements"] == config.DEFAULT_CONFIG["max_elements"]


# Verify that a zero or non-numeric value falls back to the default (1280 for browser_width).
def test_positive_int_uses_default_for_invalid_values() -> None:
    assert config._positive_int({"browser_width": 0}, "browser_width") == 1280
    assert config._positive_int({"browser_width": "bad"}, "browser_width") == 1280
