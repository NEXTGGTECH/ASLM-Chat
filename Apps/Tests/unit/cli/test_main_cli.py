# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import importlib.util
import io
import json
import os
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


pytestmark = [pytest.mark.unit]


# Repository root of ASLM-Chat, four levels above this test file (main.py lives there).
REPO_ROOT = Path(__file__).resolve().parents[4]


# Load main.py under a dedicated module name so the CLI functions can be called directly.
def _main_module() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("aslm_chat_main_cli", REPO_ROOT / "main.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Provide the loaded main.py module, shared by all tests of this file.
@pytest.fixture(scope="module")
def main_module():
    return _main_module()


# Venv re-exec gating

# Verify that a command outside the server-venv command set never spawns a subprocess.
def test_non_venv_commands_run_in_process(main_module, monkeypatch):
    # Fail the test if the code tries to start a process.
    def fail_popen(*args, **kwargs):
        raise AssertionError("unexpected subprocess spawn")

    monkeypatch.setattr(main_module.subprocess, "Popen", fail_popen)
    monkeypatch.setattr(main_module.subprocess, "run", fail_popen)

    main_module._maybe_reexec_in_server_venv("get_setting")


# Verify that the ASLM_CHAT_ACTIVE_VENV marker makes a server command run in-process without a re-exec.
def test_active_venv_marker_skips_reexec(main_module, monkeypatch):
    # Fail the test if the code tries to start a process.
    def fail_popen(*args, **kwargs):
        raise AssertionError("unexpected subprocess spawn")

    monkeypatch.setenv("ASLM_CHAT_ACTIVE_VENV", "server")
    monkeypatch.setattr(main_module.subprocess, "Popen", fail_popen)

    main_module._maybe_reexec_in_server_venv("runserver")


# Verify that a server command re-runs main.py with the server venv interpreter, marks the child environment as
# the active venv, drops PYTHONHOME and exits with the child's status.
def test_server_commands_reexec_into_venv(main_module, monkeypatch):
    spawned = {}

    # Child process stand-in that exits successfully at once.
    class _FakeProcess:
        # Return the successful exit status immediately.
        def wait(self, timeout=None):
            return 0

        # Report the process as already finished with a successful status.
        def poll(self):
            return 0

    # Record the command line and environment instead of starting a process.
    def fake_popen(args, env=None, **kwargs):
        spawned["args"] = list(args)
        spawned["env"] = dict(env or {})
        return _FakeProcess()

    # Arrange: no active-venv marker. The suite itself runs inside the server venv; pretend to be the host
    # engine Python so the re-exec branch is taken.
    monkeypatch.delenv("ASLM_CHAT_ACTIVE_VENV", raising=False)
    monkeypatch.setattr(sys, "executable", "C:/Engines/Python/python.exe")
    monkeypatch.setattr(sys, "argv", ["main.py", "migrate"])
    monkeypatch.setattr(main_module.subprocess, "Popen", fake_popen)

    # Act: the re-exec ends the process with the child's exit status.
    with pytest.raises(SystemExit) as exc_info:
        main_module._maybe_reexec_in_server_venv("migrate")

    # Assert: the venv interpreter was launched with the original arguments and a clean environment.
    assert exc_info.value.code == 0
    scripts_dir = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    venv_python = REPO_ROOT / "Data" / "venvs" / "server" / scripts_dir / executable
    assert spawned["args"] == [str(venv_python), "-u", str(REPO_ROOT / "main.py"), "migrate"]
    assert spawned["env"]["ASLM_CHAT_ACTIVE_VENV"] == "server"
    assert spawned["env"].get("PYTHONHOME") is None


# Verify that the downloads_bridge re-exec forwards the request on stdin to the child, relays the child's stdout
# and exits with its status.
def test_downloads_bridge_reexec_pipes_stdin(main_module, monkeypatch, capsysbinary):
    captured = {}

    # Record the command line and stdin input and return a successful result with a JSON stdout.
    def fake_run(args, env=None, input=None, **kwargs):
        captured["args"] = list(args)
        captured["input"] = input
        return SimpleNamespace(returncode=0, stdout=b'{"success": true}', stderr=b"")

    monkeypatch.delenv("ASLM_CHAT_ACTIVE_VENV", raising=False)
    monkeypatch.setattr(sys, "executable", "C:/Engines/Python/python.exe")
    monkeypatch.setattr(sys, "argv", ["main.py", "downloads_bridge"])
    monkeypatch.setattr(main_module.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(b'{"operation": "list_categories"}')))

    with pytest.raises(SystemExit) as exc_info:
        main_module._maybe_reexec_in_server_venv("downloads_bridge")

    assert exc_info.value.code == 0
    assert captured["input"] == b'{"operation": "list_categories"}'
    assert capsysbinary.readouterr().out == b'{"success": true}'


# Verify that a command exits with status 1 when the missing server venv cannot be created.
def test_missing_venv_aborts_when_bootstrap_fails(main_module, monkeypatch):
    monkeypatch.delenv("ASLM_CHAT_ACTIVE_VENV", raising=False)
    monkeypatch.setattr("Services.venv_manager.ensure_venv", lambda venv_id, log=True: False)

    # Point the venv lookup at a directory that cannot exist.
    monkeypatch.setattr(main_module, "BASE_DIR", str(REPO_ROOT / "no" / "such" / "dir"))

    with pytest.raises(SystemExit) as exc_info:
        main_module._maybe_reexec_in_server_venv("runserver")

    assert exc_info.value.code == 1


# Runserver port resolution

# Verify that an explicitly requested non-default port is used as given.
def test_explicit_port_wins(main_module):
    assert main_module._resolve_runserver_port(30123) == 30123


# Verify that the ASLM_UI_PORT environment variable replaces the default port 30000.
def test_env_port_used_for_default_request(main_module, monkeypatch, isolated_settings):
    monkeypatch.setenv("ASLM_UI_PORT", "30777")

    assert main_module._resolve_runserver_port(30000) == 30777


# Verify that the persisted ui-port setting is used when the default port is requested and no environment port is
# set.
def test_settings_port_used_when_no_env(main_module, monkeypatch, isolated_settings):
    from Settings import settings as project_settings

    monkeypatch.delenv("ASLM_UI_PORT", raising=False)
    project_settings.save_settings({"ui-port": 30500}, sync_tool_configs=False)

    assert main_module._resolve_runserver_port(30000) == 30500


# Setting commands

# Verify that get_setting prints the stored value, or an empty line for an unknown key.
def test_get_setting_prints_value_or_empty(main_module, capsys, isolated_settings):
    from Settings import settings as project_settings

    project_settings.save_settings({"ui-port": 30400}, sync_tool_configs=False)

    main_module.cmd_get_setting("ui-port")
    assert capsys.readouterr().out.strip() == "30400"

    main_module.cmd_get_setting("unknown-key")
    assert capsys.readouterr().out == "\n"


# Verify that set_setting prints the new value, persists it as a normalized integer and mirrors it into the module
# manifest.
def test_set_setting_persists_normalized_value(main_module, capsys, isolated_settings, isolated_manifest):
    main_module.cmd_set_setting("ui-port", "30600")

    captured = capsys.readouterr().out
    assert "30600" in captured
    from Settings import settings as project_settings

    assert project_settings.get("ui-port") == 30600
    manifest = json.loads(isolated_manifest.path.read_text(encoding="utf-8"))
    mirrored = {item["key"]: item.get("value") for item in manifest["settings"]}
    assert mirrored["ui-port"] == 30600


# Verify that set_setting does not start or sync the engine runtimes, because the short-lived CLI process must
# never own one.
def test_set_setting_does_not_trigger_runtime_sync(main_module, isolated_settings, isolated_manifest):
    with patch("API.llm_api.sync_enabled_engine_runtimes") as sync:
        main_module.cmd_set_setting("llm-engine", "openai")

    sync.assert_not_called()


# Host snapshot commands

# Verify that apply_aslm_host_theme stores the theme snapshot read from the host temp file and reports it.
def test_apply_aslm_host_theme_writes_snapshot(main_module, tmp_path, monkeypatch, capsys):
    from Settings import host_theme

    theme_file = tmp_path / "aslm_theme.json"
    theme_file.write_text(json.dumps({"theme": "dark", "colors": {"BackgroundPrimary": "#111111"}}), encoding="utf-8")
    monkeypatch.setattr(host_theme, "HOST_THEME_FILE", tmp_path / "host_theme.json")

    main_module.cmd_apply_aslm_host_theme(str(theme_file))

    assert host_theme.load_host_theme()["theme"] == "dark"
    assert "Host theme snapshot" in capsys.readouterr().out


# Verify that apply_aslm_locale stores the locale snapshot, patches the manifest for its language and reports it.
def test_apply_aslm_locale_writes_snapshot_and_localizes_manifest(
    main_module, tmp_path, monkeypatch, capsys
):
    from Settings import host_locale

    locale_file = tmp_path / "aslm_locale.json"
    locale_file.write_text(json.dumps({"language": "de", "displayName": "Deutsch"}), encoding="utf-8")
    monkeypatch.setattr(host_locale, "HOST_LOCALE_FILE", tmp_path / "host_locale.json")

    localized = {}
    monkeypatch.setattr(
        "Settings.module_manifest_locale.apply_manifest_locale",
        lambda language: localized.setdefault("language", language),
    )

    main_module.cmd_apply_aslm_locale(str(locale_file))

    assert host_locale.get_language() == "de"
    assert localized["language"] == "de"
    assert "Host locale snapshot" in capsys.readouterr().out


# Verify that apply_aslm_locale exits when the snapshot file is not valid JSON.
def test_apply_aslm_locale_rejects_broken_payload(main_module, tmp_path, monkeypatch):
    from Settings import host_locale

    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    monkeypatch.setattr(host_locale, "HOST_LOCALE_FILE", tmp_path / "host_locale.json")

    with pytest.raises(SystemExit):
        main_module.cmd_apply_aslm_locale(str(broken))


# Banner gating

# Verify that a machine-facing command such as get_setting does not print the startup banner.
def test_banner_skipped_for_quiet_commands(main_module, monkeypatch):
    # Fail the test if the banner is created.
    def fail_banner():
        raise AssertionError("unexpected banner")

    monkeypatch.setattr("Settings.console.PrintTechData", fail_banner)

    main_module._maybe_print_banner("get_setting")


# Verify that an interactive command such as runserver prints the startup banner.
def test_banner_printed_for_interactive_commands(main_module, monkeypatch):
    printed = []

    # Banner stand-in that records that it was printed.
    class _FakeBanner:
        # Record the print call.
        def PTD_Print(self):
            printed.append(True)

    monkeypatch.setattr("Settings.console.PrintTechData", _FakeBanner)

    main_module._maybe_print_banner("runserver")

    assert printed == [True]


# Full CLI dispatch

# Verify that main() parses the arguments and runs get_setting end to end, printing the stored value.
def test_cli_dispatch_get_setting(main_module, monkeypatch, capsys, isolated_settings):
    from Settings import settings as project_settings

    project_settings.save_settings({"ui-port": 30300}, sync_tool_configs=False)
    monkeypatch.setattr(sys, "argv", ["main.py", "get_setting", "--key", "ui-port"])

    main_module.main()

    assert capsys.readouterr().out.strip() == "30300"


# Verify that an unknown command prints an "Unknown command" message and exits with status 1.
def test_cli_dispatch_unknown_command(main_module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "teleport"])

    with pytest.raises(SystemExit) as exc_info:
        main_module.main()

    assert exc_info.value.code == 1
    assert "Unknown command" in capsys.readouterr().out


# Verify that the parser keeps the given command and fills the default port 30000 and a disabled log flag.
def test_cli_parser_defaults(main_module):
    parser = main_module._build_parser()
    args = parser.parse_args(["set_setting", "--key", "ui-port", "--value", "1"])

    assert args.command == "set_setting"
    assert args.port == 30000
    assert args.log is False
