# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest
from django.test import SimpleTestCase

from Settings import settings as project_settings


pytestmark = [pytest.mark.unit]


# The module file name contains a hyphen and needs importlib.
def _ollama_service():
    return importlib.import_module("Services.ollama-service")


@pytest.fixture
def isolated_pid_file(tmp_path, monkeypatch):
    module = _ollama_service()
    pid_file = tmp_path / "ollama-service.pid"
    monkeypatch.setattr(module, "PID_FILE", pid_file)
    return pid_file


# Desired-state policy (migrated from the original suite).


# Verify Ollama desired-state policy.
class OllamaDesiredStateTests(SimpleTestCase):
    @patch("Settings.settings.get_llm_engine", return_value="lms")
    @patch("Settings.settings.get", return_value=True)
    def test_desired_state_runs_when_enabled_even_if_active_engine_differs(self, _mock_get, _mock_active):
        ollama_service = _ollama_service()
        state = ollama_service._get_desired_state("ollama-service")
        self.assertTrue(state.should_run)

    def test_service_never_persists_runtime_output_in_module_settings(self):
        ollama_service = _ollama_service()
        source = Path(ollama_service.__file__).read_text(encoding="utf-8")

        self.assertFalse(hasattr(ollama_service, "LOG_FILE"))
        self.assertNotIn("ollama-service.log", source)
        self.assertIn("stdout=subprocess.PIPE", source)


# PID bookkeeping.


# The PID file round-trips through write, read, and clear.
def test_pid_file_roundtrip(isolated_pid_file):
    module = _ollama_service()

    assert module._read_pid() is None

    module._write_pid(4242)
    assert module._read_pid() == 4242

    module._clear_pid()
    assert module._read_pid() is None
    # Clearing a missing file is a no-op.
    module._clear_pid()


# A corrupt PID file reads as "no managed process".
def test_corrupt_pid_file_reads_as_none(isolated_pid_file):
    module = _ollama_service()
    isolated_pid_file.write_text("not-a-number", encoding="utf-8")

    assert module._read_pid() is None


# Liveness checks distinguish running processes from absent PIDs.
def test_liveness_check_distinguishes_running_processes():
    module = _ollama_service()

    assert module._is_pid_running(os.getpid()) is True
    assert module._is_pid_running(None) is False
    assert module._is_pid_running(0) is False


# Launch environment.


# The launch environment binds the configured port and models directory.
def test_service_environment_binds_configured_port_and_models(isolated_settings):
    module = _ollama_service()
    project_settings.save_settings(
        {"ollama-service_port": 30100, "ollama-service_models": "D:/models"},
        sync_tool_configs=False,
    )

    env, port = module._build_service_environment()

    assert port == 30100
    assert env["OLLAMA_HOST"] == "127.0.0.1:30100"
    assert env["OLLAMA_MODELS"] == "D:/models"


# An unconfigured models directory is dropped from the environment.
def test_service_environment_drops_models_setting_when_unconfigured(isolated_settings):
    module = _ollama_service()

    env, _port = module._build_service_environment()

    assert "OLLAMA_MODELS" not in env


# Console log formatting.


# ANSI escapes and trailing line breaks are stripped for the ASLM console.
def test_sanitize_console_line_strips_ansi_and_newlines():
    module = _ollama_service()

    assert module._sanitize_console_line("\x1b[32mgreen text\x1b[0m\r\n") == "green text"


# Long lines truncate to the debug-level limit with an ellipsis.
def test_truncate_console_line_applies_debug_limit(isolated_settings):
    module = _ollama_service()
    long_line = "x" * 1000

    truncated = module._truncate_console_line(long_line)

    # The default console_log_level is debug, which allows 420 characters.
    assert len(truncated) == 420
    assert truncated.endswith("...")
    assert module._truncate_console_line("short") == "short"


# Trace logging extends the truncation limit.
def test_truncate_console_line_extends_limit_for_trace(isolated_settings):
    module = _ollama_service()
    project_settings.save_settings({"console_log_level": "trace"}, sync_tool_configs=False)

    truncated = module._truncate_console_line("x" * 1000)

    assert len(truncated) == 900


# Structured key=value log lines parse into a dictionary.
def test_parse_structured_log_fields_extracts_key_values():
    module = _ollama_service()

    fields = module._parse_structured_log_fields('ts=123 message="hello world" tokens=42')

    assert fields == {"ts": "123", "message": "hello world", "tokens": "42"}


# Unparsable lines yield no fields instead of raising.
def test_parse_structured_log_fields_tolerates_unparsable_lines():
    module = _ollama_service()

    assert module._parse_structured_log_fields('"unterminated') == {}


# Desired state policy.


# The desired state follows the engine enablement flag.
def test_desired_state_follows_engine_enablement(isolated_settings):
    module = _ollama_service()

    state = module._get_desired_state()
    assert state.is_enabled is False
    assert state.should_run is False

    project_settings.save_settings({"ollama-service": True}, sync_tool_configs=False)

    state = module._get_desired_state()
    assert state.is_enabled is True
    assert state.should_run is True
    assert state.active_engine == "ollama-service"


# Requested engine aliases normalize to canonical ids.
def test_desired_state_normalizes_requested_engine_alias(isolated_settings):
    module = _ollama_service()

    state = module._get_desired_state("ollama")

    assert state.requested_engine == "ollama-service"
