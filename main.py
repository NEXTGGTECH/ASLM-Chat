# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import argparse
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path


# Prepare project imports
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Select the Django settings module for every command that imports Django.
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ASLM.settings")

from Settings.proxy_policy import apply_loopback_proxy_bypass

# Commands that need Django or other server dependencies and therefore must run inside the server venv.
SERVER_VENV_COMMANDS = {
    "runserver",
    "migrate",
    "makemigrations",
    "collectstatic",
    "downloads_bridge",
}


# Server venv

# Delegate the current command to ASLM-Chat's server venv when it needs Django/server dependencies.
# The child process runs main.py again with the venv interpreter (the venv is created first when
# missing) and this process exits with its status. For downloads_bridge the request is passed
# through stdin and the child's stdout/stderr are relayed, so the bridge protocol stays unchanged.
def _maybe_reexec_in_server_venv(command: str) -> None:
    # Nothing to do for other commands, or when this process already runs in the server venv.
    if command not in SERVER_VENV_COMMANDS:
        return
    if os.environ.get("ASLM_CHAT_ACTIVE_VENV") == "server":
        return

    # Locate the venv interpreter; running with it already means the venv is active.
    venv_path = Path(BASE_DIR) / "Data" / "venvs" / "server"
    scripts_path = venv_path / ("Scripts" if os.name == "nt" else "bin")
    python_path = scripts_path / ("python.exe" if os.name == "nt" else "python")
    current_python = os.path.normcase(os.path.abspath(sys.executable))
    target_python = os.path.normcase(os.path.abspath(str(python_path)))
    if current_python == target_python:
        os.environ["ASLM_CHAT_ACTIVE_VENV"] = "server"
        return

    # Create the venv when it is missing; downloads_bridge stays quiet so its output is not polluted.
    venv_ready = python_path.exists()
    if not venv_ready:
        from Services import venv_manager

        quiet = command == "downloads_bridge"
        if not venv_manager.ensure_venv("server", log=not quiet):
            print("[ASLM-Chat] Error: server venv could not be prepared.")
            sys.exit(1)
        python_path = venv_manager.get_venv_python("server")
        venv_path = venv_manager.get_venv_path("server")
        scripts_path = python_path.parent

    # Build the child environment: mark the venv as active and put its scripts first on PATH.
    env = os.environ.copy()
    apply_loopback_proxy_bypass(env)
    env["ASLM_CHAT_ACTIVE_VENV"] = "server"
    env["VIRTUAL_ENV"] = str(venv_path)
    env["PATH"] = str(scripts_path) + os.pathsep + env.get("PATH", "")
    env.pop("PYTHONHOME", None)
    args = [str(python_path), "-u", __file__, *sys.argv[1:]]

    # On Windows the child must not open a console window.
    process_kwargs: dict[str, object] = {}
    if os.name == "nt":
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
        startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
        process_kwargs = {
            "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
            "startupinfo": startupinfo,
        }

    # downloads_bridge: forward stdin to the child, relay its output and exit with its status.
    if command == "downloads_bridge":
        request_payload = sys.stdin.buffer.read()
        completed = subprocess.run(
            args,
            env=env,
            input=request_payload,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            **process_kwargs,
        )
        sys.stdout.buffer.write(completed.stdout)
        sys.stdout.buffer.flush()
        sys.stderr.buffer.write(completed.stderr)
        sys.stderr.buffer.flush()
        sys.exit(completed.returncode)

    # Explicitly pass ASLM's pipes to the hidden child on Windows.
    if os.name == "nt":
        process_kwargs.update(stdout=sys.stdout, stderr=sys.stderr)

    # Wait for the child and stop it when the parent handles an interrupt.
    process = subprocess.Popen(args, env=env, **process_kwargs)
    try:
        sys.exit(process.wait())
    except KeyboardInterrupt:
        if process.poll() is None:
            process.terminate()
        raise
    finally:
        if process.poll() is None:
            process.terminate()


# Django server

# Execute a Django management command in this process, as if run through manage.py.
def run_django_command(*args: str, log: bool = False) -> None:
    from django.core.management import execute_from_command_line

    argv = ["manage.py", *args]
    if log:
        print(f"[ASLM-Chat] Running: {' '.join(argv)}")

    execute_from_command_line(argv)


# Prepare Django, then serve the UI on the requested loopback port with a threaded WSGI server.
def cmd_runserver(port: int, log: bool) -> None:
    if log:
        print(f"[ASLM-Chat] Starting server on port {port}...")

    import socket
    from socketserver import ThreadingMixIn
    from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

    # Large file uploads (video, audio) require a much longer socket timeout than
    # the default 30 s used by http.server.BaseHTTPRequestHandler.  Without this,
    # the connection is torn down mid-transfer and the browser shows "Upload failed".
    _UPLOAD_SOCKET_TIMEOUT_SECONDS = 3600

    # WSGI server that serves local UI requests concurrently, one daemon thread per request.
    class ThreadedWSGIServer(ThreadingMixIn, WSGIServer):
        daemon_threads = True

        # Accept a connection and apply a generous timeout for large uploads.
        def get_request(self):
            conn, addr = self.socket.accept()
            conn.settimeout(_UPLOAD_SOCKET_TIMEOUT_SECONDS)
            return conn, addr

    # WSGI handler with request logging for the local server.
    class RequestHandler(WSGIRequestHandler):
        # Match the server timeout so request handling also allows large uploads.
        timeout = _UPLOAD_SOCKET_TIMEOUT_SECONDS

        # Log completed requests without including query parameters.
        def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
            path = getattr(self, "path", "").partition("?")[0]
            self.log_message('"%s %s %s" %s %s', self.command or "-", path or "-", self.request_version, code, size)

    # Import Django before opening its listening socket.
    from ASLM.wsgi import application

    app = application
    with make_server(
        "127.0.0.1",
        port,
        app,
        server_class=ThreadedWSGIServer,
        handler_class=RequestHandler,
    ) as httpd:
        _probe_docker_status_async(log=log)
        print(f"[ASLM-Chat] UI server listening at http://127.0.0.1:{httpd.server_port}/", flush=True)
        httpd.serve_forever()


# Management commands

# Apply all pending database migrations.
def cmd_migrate(log: bool) -> None:
    if log:
        print("[ASLM-Chat] Applying migrations...")

    run_django_command("migrate", "--noinput", log=log)


# Create migration files for changed models, optionally for one app only.
def cmd_makemigrations(app: str | None, log: bool) -> None:
    args = ["makemigrations"]
    if app:
        args.append(app)

    run_django_command(*args, log=log)


# Collect static files into STATIC_ROOT.
def cmd_collectstatic(log: bool) -> None:
    run_django_command("collectstatic", "--noinput", log=log)


# Run first-time setup: generate the settings and tool environments, then apply the initial
# migrations inside the server venv.
def cmd_first_run(
    log: bool = True,
    ui_port: int = 30000,
) -> None:
    from Settings.first_run import run as first_run

    print("[ASLM-Chat] Running first-run setup...")
    first_run(log=log, ui_port=ui_port)

    # Migrate with the server venv interpreter, because the Django dependencies live there.
    from Services import venv_manager

    migrate_args = [os.path.abspath(__file__), "migrate"]
    if log:
        migrate_args.append("--log")

    if not venv_manager.run_venv_python("server", migrate_args, log=log):
        print("[ASLM-Chat] Database migration failed inside server venv.")
        sys.exit(1)


# Print a single setting value for ASLM integration hooks; an unset value prints an empty line.
def cmd_get_setting(key: str) -> None:
    from Settings.settings import get

    value = get(key)
    print(value if value is not None else "")


# Update a single setting from string input, converting it to the matching value type first. Host
# account keys are not echoed.
def cmd_set_setting(key: str, value: str) -> None:
    from Settings.settings import HOST_KEY_SETTING_KEYS, normalize_setting_value, set

    parsed_value = normalize_setting_value(value)

    # ASLM invokes this command in a short-lived process before the module's
    # run commands exist. Persisting a setting must not make that helper own a
    # long-running engine runtime (or wait for the dedicated runtime command).
    set(key, parsed_value, sync_runtime=False)
    if key in HOST_KEY_SETTING_KEYS:
        print(f"[ASLM-Chat] Setting '{key}' updated.")
    else:
        print(f"[ASLM-Chat] Setting '{key}' updated to {parsed_value}")


# Apply a JSON theme snapshot written by ASLM (temp file path in --file) by storing it for the
# module UI; exit with an error when the file is missing, unreadable or not a JSON object.
def cmd_apply_aslm_host_theme(theme_file: str) -> None:
    from pathlib import Path

    from Settings.host_theme import save_host_theme_payload

    # Read the snapshot file.
    path = Path(theme_file)
    if not path.is_file():
        print(f"Error: theme file not found: {theme_file}")
        sys.exit(1)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: could not read theme file: {exc}")
        sys.exit(1)

    # .NET may write UTF-8 with BOM; json.loads rejects a leading U+FEFF unless stripped.
    raw = raw.lstrip("\ufeff").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in theme file: {exc}")
        sys.exit(1)
    if not isinstance(data, dict):
        print("Error: host theme JSON must be an object.")
        sys.exit(1)

    # Store the snapshot.
    save_host_theme_payload(data)
    print("[ASLM-Chat] Host theme snapshot updated.")


# Apply a JSON locale snapshot written by ASLM (temp file path in --file): store it and patch the
# module manifest with the strings of the snapshot's language. Exits with an error when the file
# is missing, unreadable or not a JSON object.
def cmd_apply_aslm_locale(locale_file: str) -> None:
    from pathlib import Path

    from Settings.host_locale import save_host_locale_payload

    # Read the snapshot file.
    path = Path(locale_file)
    if not path.is_file():
        print(f"Error: locale file not found: {locale_file}")
        sys.exit(1)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Error: could not read locale file: {exc}")
        sys.exit(1)

    # Strip the .NET BOM, then require a JSON object.
    raw = raw.lstrip("\ufeff").strip()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"Error: invalid JSON in locale file: {exc}")
        sys.exit(1)
    if not isinstance(data, dict):
        print("Error: host locale JSON must be an object.")
        sys.exit(1)

    # Store the snapshot, then localize the module manifest to its language.
    save_host_locale_payload(data)
    from Settings.module_manifest_locale import apply_manifest_locale

    language = str(data.get("language", "en"))
    apply_manifest_locale(language)
    print("[ASLM-Chat] Host locale snapshot updated.")


# Probe Docker availability in the background so sandbox tool availability stays fresh. The probe
# is started at server startup and never blocks it.
def _probe_docker_status_async(log: bool) -> None:
    try:
        from Services import docker_status

        docker_status.refresh_async()
    except Exception as exc:  # noqa: BLE001 - never let the probe block startup
        if log:
            print(f"[ASLM-Chat] Warning: Docker status probe could not start. {exc}")


# Run the dedicated managed Ollama runtime as its own tracked process and exit with its exit code.
def cmd_ollama_runtime(log: bool) -> None:
    try:
        ollama_service = importlib.import_module("Services.ollama-service")
    except ImportError as exc:
        print(f"[ASLM-Chat] Error: Services.ollama-service could not be loaded. {exc}")
        sys.exit(1)

    exit_code = ollama_service.run_ollama_runtime(log=log)
    if exit_code:
        sys.exit(exit_code)


# Run the downloads bridge request dispatcher: read one JSON request from stdin and print the JSON
# response.
def cmd_downloads_bridge() -> None:
    from Services.downloads_bridge import run_cli

    raise SystemExit(run_cli())


# Command line

# Build the command-line parser for the project entry point.
def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ASLM-Chat management entry point",
    )
    parser.add_argument("command", type=str, help="Command to execute")
    parser.add_argument("--port", type=int, default=30000, help="Port for runserver (default: 30000)")
    parser.add_argument("--app", type=str, default=None, help="App name for makemigrations")
    parser.add_argument("--key", type=str, default=None, help="Setting key for get_setting/set_setting")
    parser.add_argument("--value", type=str, default=None, help="Setting value for set_setting")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Path to JSON payload for apply_aslm_host_theme or apply_aslm_locale",
    )
    parser.add_argument("--log", action="store_true", help="Enable verbose output")
    return parser


# Print technical module data once for interactive commands. Commands whose output ASLM reads
# (settings, bridge and snapshot commands) and re-launched child processes (RUN_MAIN set) stay silent.
def _maybe_print_banner(command: str) -> None:
    if not os.environ.get("RUN_MAIN") and command not in {
        "get_setting",
        "set_setting",
        "downloads_bridge",
        "apply_aslm_host_theme",
        "apply_aslm_locale",
    }:
        from Settings.console import PrintTechData

        PrintTechData().PTD_Print()


# Return the effective UI port for runserver: an explicit --port other than the default 30000 wins,
# then the ASLM_UI_PORT environment variable, then the ui-port setting.
def _resolve_runserver_port(requested_port: int) -> int:
    from Settings.settings import load_settings

    if requested_port != 30000:
        return requested_port

    env_port = os.environ.get("ASLM_UI_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            # A non-numeric ASLM_UI_PORT is ignored; the stored setting decides.
            pass

    runtime_settings = load_settings()
    return int(runtime_settings.get("ui-port", 30000))


# Parse the command line and dispatch the requested command: runserver, ollama_runtime, migrate,
# makemigrations, collectstatic, first_run, get_setting, set_setting, apply_aslm_host_theme,
# apply_aslm_locale, downloads_bridge or help. Commands that need Django first re-run themselves
# in the server venv.
def main() -> None:
    # Install the localhost bypass before a venv re-exec or any network client
    # can inherit the host application's system proxy configuration.
    apply_loopback_proxy_bypass()

    # Parse the arguments; a server-venv command is handed over to the venv and does not return.
    parser = _build_parser()
    args = parser.parse_args()

    _maybe_reexec_in_server_venv(args.command)
    _maybe_print_banner(args.command)

    # Dispatch to the handler of the command; settings and snapshot commands validate their arguments first.
    match args.command:
        case "runserver":
            port = _resolve_runserver_port(args.port)
            cmd_runserver(port, log=args.log)

        case "ollama_runtime":
            cmd_ollama_runtime(args.log)

        case "migrate":
            cmd_migrate(args.log)

        case "makemigrations":
            cmd_makemigrations(args.app, args.log)

        case "collectstatic":
            cmd_collectstatic(args.log)

        case "first_run":
            cmd_first_run(log=True, ui_port=args.port)

        case "get_setting":
            if not args.key:
                print("Error: --key argument is required.")
                sys.exit(1)
            cmd_get_setting(args.key)

        case "set_setting":
            if not args.key or args.value is None:
                print("Error: --key and --value arguments are required.")
                sys.exit(1)
            cmd_set_setting(args.key, args.value)

        case "apply_aslm_host_theme":
            if not args.file:
                print("Error: --file argument is required.")
                sys.exit(1)
            cmd_apply_aslm_host_theme(args.file)

        case "apply_aslm_locale":
            if not args.file:
                print("Error: --file argument is required.")
                sys.exit(1)
            cmd_apply_aslm_locale(args.file)

        case "downloads_bridge":
            cmd_downloads_bridge()

        case "help":
            parser.print_help()

        case _:
            print(f"[ASLM-Chat] Unknown command: '{args.command}'")
            print("Run 'python main.py help' for usage.")
            sys.exit(1)


# Run: python main.py <command> [--port N] [--app NAME] [--key KEY] [--value VALUE] [--file PATH] [--log]
if __name__ == "__main__":
    main()
