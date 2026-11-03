# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import logging
import os
import re
import shlex
import signal
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

from Settings import settings
from Settings.proxy_policy import apply_loopback_proxy_bypass, urlopen_direct

logger = logging.getLogger(__name__)

# The Ollama child process this service started in the current process, the thread that forwards
# its output to the console (guarded by the lock), and the file that stores the managed PID so
# a later run or a separate supervisor can stop the same process.
_ollama_process: subprocess.Popen | None = None
_log_stream_thread: threading.Thread | None = None
_log_stream_lock = threading.Lock()
PID_FILE = Path(__file__).resolve().parent.parent / "Settings" / "ollama-service.pid"

# Pattern that matches ANSI escape sequences in Ollama output, and the base length (in characters)
# at which a console line is cut (raised in debug and trace console modes).
ANSI_ESCAPE_RE = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
MAX_CONSOLE_LINE_LENGTH = 220

# Environment variables of Ollama's "server config" log line that are worth showing in the console.
SERVER_CONFIG_ENV_KEYS = (
    "OLLAMA_HOST",
    "OLLAMA_MODELS",
    "OLLAMA_KEEP_ALIVE",
    "OLLAMA_LOAD_TIMEOUT",
    "OLLAMA_NUM_PARALLEL",
    "OLLAMA_MAX_QUEUE",
    "OLLAMA_CONTEXT_LENGTH",
)


# Runtime state model

# Flags derived from settings that describe whether Ollama should run: the requested and the
# active LLM engine, whether the ollama-service setting is on, and the resulting run decision.
@dataclass(frozen=True)
class OllamaDesiredState:
    requested_engine: str
    active_engine: str
    is_enabled: bool
    should_run: bool


# Process tracking

# Read the managed Ollama PID from disk.
def _read_pid() -> int | None:
    try:
        raw_value = PID_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        return None

    if not raw_value:
        return None

    try:
        return int(raw_value)
    except ValueError:
        return None


# Persist the managed Ollama PID on disk.
def _write_pid(pid: int) -> None:
    PID_FILE.parent.mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(pid), encoding="utf-8")


# Remove the saved Ollama PID file.
def _clear_pid() -> None:
    try:
        PID_FILE.unlink()
    except OSError:
        # The file is already gone (or locked); a stale PID is harmless because _is_pid_running re-checks it.
        pass


# Return whether the given PID still points to a live process.
def _is_pid_running(pid: int | None) -> bool:
    if not pid:
        return False

    try:
        os.kill(pid, 0)
    except (OSError, SystemError):
        return False

    return True


# Runtime policy

# Resolve whether the managed Ollama runtime should currently be running.
def _get_desired_state(requested_engine: str | None = None) -> OllamaDesiredState:
    active_engine = settings.get_llm_engine()
    resolved_engine = settings.normalize_engine_name(requested_engine or active_engine)
    is_enabled = bool(settings.get("ollama-service", False))
    should_run = is_enabled
    return OllamaDesiredState(
        requested_engine=resolved_engine,
        active_engine=active_engine,
        is_enabled=is_enabled,
        should_run=should_run,
    )


# Build environment variables used to launch the managed Ollama service.
def _build_service_environment() -> tuple[dict[str, str], int]:
    ollama_models = settings.get("ollama-service_models")
    ollama_port = int(settings.get("ollama-service_port", 30002))

    # Start from the current process environment and overlay managed values.
    env = os.environ.copy()
    apply_loopback_proxy_bypass(env)
    env["OLLAMA_HOST"] = f"127.0.0.1:{ollama_port}"

    if ollama_models:
        env["OLLAMA_MODELS"] = str(ollama_models)
    else:
        env.pop("OLLAMA_MODELS", None)

    return env, ollama_port


# Log formatting

# Strip ANSI escapes and trailing line breaks for ASLM console rendering.
def _sanitize_console_line(message: str) -> str:
    cleaned = ANSI_ESCAPE_RE.sub("", message)
    return cleaned.rstrip("\r\n")


# Trim long console lines so ASLM stays readable.
def _truncate_console_line(
    message: str,
    limit: int = MAX_CONSOLE_LINE_LENGTH,
) -> str:
    if settings.is_console_trace_enabled():
        limit = max(limit, 900)
    elif settings.is_console_debug_enabled():
        limit = max(limit, 420)

    if len(message) <= limit:
        return message

    return f"{message[:max(0, limit - 3)].rstrip()}..."


# Parse Ollama key=value log lines into a dictionary.
def _parse_structured_log_fields(message: str) -> dict[str, str]:
    try:
        tokens = shlex.split(message, posix=True)
    except ValueError:
        return {}

    fields: dict[str, str] = {}
    for token in tokens:
        if "=" not in token:
            continue

        key, value = token.split("=", 1)
        normalized_key = key.strip().strip('"')
        if not normalized_key:
            continue

        fields[normalized_key] = value

    return fields


# Extract one environment variable from Ollama's server-config log dump.
def _extract_env_value(env_blob: str, env_key: str) -> str:
    match = re.search(rf"{re.escape(env_key)}:([^ ]+)", env_blob)
    if not match:
        return ""

    return match.group(1).strip()


# Format extra structured-log fields as key=value details.
def _collect_remaining_field_details(
    fields: dict[str, str],
    *,
    excluded_keys: set[str] | None = None,
    max_items: int | None = None,
) -> list[str]:
    excluded = {"time", "level", "source", "msg"}
    if excluded_keys:
        excluded |= set(excluded_keys)

    details: list[str] = []
    for key in sorted(fields, key=str.casefold):
        if key in excluded:
            continue

        value = str(fields.get(key, "") or "").strip()
        if not value:
            continue

        details.append(f"{key}={value}")
        if max_items is not None and len(details) >= max_items:
            break

    return details


# Extract useful runner load-request fields from Ollama trace lines.
def _summarize_load_request(request_blob: str) -> list[str]:
    details: list[str] = []
    patterns = {
        "operation": r"Operation:([A-Za-z0-9_]+)",
        "parallel": r"Parallel:([0-9]+)",
        "batch": r"BatchSize:([0-9]+)",
        "flash": r"FlashAttention:([A-Za-z]+)",
        "kv": r"KvSize:([0-9]+)",
        "threads": r"NumThreads:([0-9]+)",
        "gpu_layers": r"GPULayers:([^ ]+)",
    }

    for label, pattern in patterns.items():
        match = re.search(pattern, request_blob)
        if match:
            details.append(f"{label}={match.group(1)}")

    return details


# Convert GIN access logs into a compact summary.
def _format_gin_line(message: str) -> str | None:
    match = re.search(
        r'\[GIN\]\s+\S+\s+-\s+\S+\s+\|\s+(\d+)\s+\|\s+([^|]+?)\s+\|\s+([^|]+?)\s+\|\s+([A-Z]+)\s+"([^"]+)"',
        message,
    )
    if not match:
        return None

    status_code, duration, client_ip, method, path = (
        match.group(1),
        match.group(2).strip(),
        match.group(3).strip(),
        match.group(4).strip(),
        match.group(5).strip(),
    )
    return _truncate_console_line(
        f"[Ollama][HTTP] {method} {path} -> {status_code} in {duration} from {client_ip}"
    )


# Convert backend loader logs into a shorter summary.
def _format_backend_line(message: str) -> str:
    backend_path = message.partition(" from ")[2].strip()
    if backend_path:
        backend_name = Path(backend_path).name
        backend_summary = message.partition(": ")[2].split(" from ")[0]
        return _truncate_console_line(
            f"[Ollama][Backend] {backend_summary} ({backend_name})"
        )

    return _truncate_console_line(f"[Ollama][Backend] {message}")


# Convert Ollama structured logs into concise console lines.
def _format_structured_ollama_line(message: str) -> str | None:
    # Parse key=value pairs from Ollama's structured logger output; plain text lines are not handled here.
    fields = _parse_structured_log_fields(message)
    if not fields:
        return None

    level = fields.get("level", "INFO").upper()
    source = fields.get("source", "")
    raw_message = fields.get("msg", "").strip()
    trace_enabled = settings.is_console_trace_enabled()

    # Recover readable summaries for trace lines without a direct message.
    if not raw_message and not trace_enabled:
        source_key = source.casefold()
        if "cpu_windows.go" in source_key:
            raw_message = "CPU topology"
        elif "ggml.go:136" in source_key:
            raw_message = "Model metadata"
        elif "ggml.go:104" in source_key:
            raw_message = "System capabilities"
        elif "runner.go:1284" in source_key:
            raw_message = "Load request"
        elif "types.go:42" in source_key:
            raw_message = "Inference compute"
        else:
            return None
    elif not raw_message:
        raw_message = "Trace event"

    summary = raw_message
    details: list[str] = []

    # Collapse known high-signal log messages into compact summaries.
    if raw_message == "server config":
        env_blob = fields.get("env", "")
        for env_key in SERVER_CONFIG_ENV_KEYS:
            env_value = _extract_env_value(env_blob, env_key)
            if env_value:
                details.append(f"{env_key.removeprefix('OLLAMA_').lower()}={env_value}")
        summary = "Server config"
    elif raw_message == "starting runner":
        # Show the runner port and model file name taken from its command line.
        command = fields.get("cmd", "")
        port_match = re.search(r"--port\s+(\d+)", command)
        model_match = re.search(r"--model\s+([^\s]+)", command)
        if port_match:
            details.append(f"port={port_match.group(1)}")
        if model_match:
            details.append(f"model={Path(model_match.group(1)).name}")
        elif command:
            details.append("runner=bootstrap")
        summary = "Starting runner"
    elif raw_message == "discovering available GPUs...":
        summary = "Discovering available GPUs"
    elif raw_message == "inference compute":
        gpu_name = fields.get("name") or fields.get("description")
        if gpu_name:
            details.append(f"gpu={gpu_name}")
        if fields.get("library"):
            details.append(f"backend={fields['library']}")
        if fields.get("available"):
            details.append(f"vram_free={fields['available']}")
        if fields.get("total"):
            details.append(f"vram_total={fields['total']}")
        summary = "Inference compute"
    elif raw_message == "vram-based default context":
        if fields.get("default_num_ctx"):
            details.append(f"default_ctx={fields['default_num_ctx']}")
        if fields.get("total_vram"):
            details.append(f"total_vram={fields['total_vram']}")
        summary = "VRAM-based default context"
    elif raw_message == "gpu memory":
        for key, label in (
            ("available", "available"),
            ("free", "free"),
            ("minimum", "minimum"),
            ("id", "gpu"),
        ):
            value = fields.get(key)
            if value:
                details.append(f"{label}={value}")
        summary = "GPU memory"
    elif raw_message == "system memory":
        for key in ("total", "free", "free_swap"):
            value = fields.get(key)
            if value:
                details.append(f"{key}={value}")
        summary = "System memory"
    elif raw_message == "loading model":
        if fields.get("model layers"):
            details.append(f"layers={fields['model layers']}")
        if fields.get("requested"):
            details.append(f"requested={fields['requested']}")
        summary = "Loading model"
    elif raw_message == "Load request" or (
        source == "runner.go:1284" and fields.get("request")
    ):
        # Trace lines without a message are recognized by their source file and line.
        details.extend(_summarize_load_request(fields.get("request", "")))
        summary = "Load request"
    elif raw_message == "Model metadata" or source == "ggml.go:136":
        for key in ("architecture", "file_type", "num_tensors", "num_key_values"):
            value = fields.get(key)
            if value:
                details.append(f"{key}={value}")
        summary = "Model metadata"
    elif raw_message == "System capabilities" or source == "ggml.go:104":
        compiler = fields.get("compiler")
        if compiler:
            details.append(f"compiler={compiler}")

        cpu_flags: list[str] = []
        for candidate in ("CPU.0.AVX2", "CPU.0.AVX512", "CPU.0.FMA", "CPU.0.F16C"):
            if fields.get(candidate) == "1":
                cpu_flags.append(candidate.split(".")[-1])
        if cpu_flags:
            details.append(f"cpu_flags={','.join(cpu_flags)}")

        cuda_archs = fields.get("CUDA.0.ARCHS")
        if cuda_archs:
            details.append(f"cuda_archs={cuda_archs}")
        summary = "System capabilities"
    elif raw_message == "CPU topology" or source == "cpu_windows.go:195":
        for key in ("package", "cores", "efficiency", "threads"):
            value = fields.get(key)
            if value:
                details.append(f"{key}={value}")
        summary = "CPU topology"
    elif raw_message == "waiting for server to become available":
        if fields.get("status"):
            details.append(f"status={fields['status']}")
        summary = "Waiting for model runner"
    elif raw_message == "waiting for llama runner to start responding":
        summary = "Waiting for llama runner"
    elif raw_message == "loaded runners":
        if fields.get("count"):
            details.append(f"count={fields['count']}")
        summary = "Loaded runners"
    elif raw_message == "invalid option provided":
        if fields.get("option"):
            details.append(f"option={fields['option']}")
        summary = "Invalid option ignored"
    elif raw_message == "Listening on 127.0.0.1:30002 (version 0.18.3)":
        summary = raw_message
    else:
        # Unknown message: keep its text and show the common status and memory fields if present.
        for key, label in (
            ("status", "status"),
            ("count", "count"),
            ("size", "size"),
            ("device", "device"),
            ("available", "available"),
            ("total", "total"),
        ):
            value = fields.get(key)
            if value:
                details.append(f"{label}={value}")

    # Append extra trace-only fields when detailed console output is enabled.
    if trace_enabled:
        trace_details = _collect_remaining_field_details(
            fields,
            excluded_keys={"env"} | {detail.split("=", 1)[0] for detail in details if "=" in detail},
            max_items=10,
        )
        for item in trace_details:
            if item not in details:
                details.append(item)

    # Render the final console line in ASLM's compact format.
    source_suffix = f"[{source}]" if source else ""
    rendered = f"[Ollama][{level}]{source_suffix} {summary}"
    if details:
        rendered = f"{rendered} | {', '.join(details)}"

    return _truncate_console_line(rendered)


# Convert raw Ollama output into a readable console line.
def _format_console_log_line(message: str) -> str | None:
    if not message:
        return None

    if message.startswith("[GIN]"):
        return _format_gin_line(message) or _truncate_console_line(f"[Ollama] {message}")
    if message.startswith("load_backend:"):
        return _format_backend_line(message)
    if message.startswith("ggml_cuda_init:"):
        return _truncate_console_line(f"[Ollama][CUDA] {message.partition(': ')[2] or message}")
    if message.startswith("Device "):
        return _truncate_console_line(f"[Ollama][CUDA] {message}")

    formatted = _format_structured_ollama_line(message)
    if formatted:
        return formatted

    return _truncate_console_line(f"[Ollama] {message}")


# Log streaming

# Emit one watcher status line to stdout.
def _print_status(message: str) -> None:
    print(f"[ASLM-Chat] {message}", flush=True)


# Forward managed Ollama output directly to the current process stdout.
def _stream_process_output(process: subprocess.Popen) -> None:
    output = process.stdout
    if output is None:
        return

    _print_status(f"Streaming Ollama output for PID {process.pid}...")
    try:
        for raw_line in output:
            line = _sanitize_console_line(raw_line)
            rendered_line = _format_console_log_line(line)
            if rendered_line:
                print(rendered_line, flush=True)
    except (OSError, ValueError):
        # The pipe was closed or the process exited mid-read; end of output is the normal way out.
        pass
    finally:
        try:
            output.close()
        except OSError:
            # The pipe is already closed; nothing is left to release.
            pass
        _print_status(f"Managed Ollama output closed (PID: {process.pid}).")


# Start one background thread that forwards the current managed process output.
def _ensure_log_streaming(process: subprocess.Popen) -> None:
    global _log_stream_thread

    # Only one forwarding thread runs at a time; keep the existing one while it is alive.
    with _log_stream_lock:
        if _log_stream_thread is not None and _log_stream_thread.is_alive():
            return

        _log_stream_thread = threading.Thread(
            target=_stream_process_output,
            args=(process,),
            name="aslm-chat-ollama-log-stream",
            daemon=True,
        )
        _log_stream_thread.start()


# Readiness checks

# Wait until the local Ollama HTTP endpoint starts responding.
def _wait_until_ready(
    timeout_seconds: float = 15.0,
    process: subprocess.Popen | None = None,
) -> bool:
    deadline = time.time() + timeout_seconds
    host = settings.get_engine_url("ollama-service")
    version_url = f"{host.rstrip('/')}/api/version"

    # Poll /api/version until it answers, giving up early when the launched process has exited.
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            return False

        remaining = max(deadline - time.time(), 0.0)
        try:
            with urlopen_direct(
                version_url,
                timeout=max(0.1, min(1.5, remaining)),
            ) as response:
                if response.status < 500:
                    return True
        except (OSError, URLError):
            # The endpoint is not accepting connections yet; wait briefly and try again.
            time.sleep(min(0.25, max(deadline - time.time(), 0.0)))

    return False


# Give a separately launched Ollama runtime a short chance to appear.
def _wait_for_existing_runtime(timeout_seconds: float = 2.0) -> bool:
    deadline = time.time() + max(0.0, timeout_seconds)
    while time.time() < deadline:
        if _wait_until_ready(timeout_seconds=0.25):
            return True

        tracked_pid = _read_pid()
        if tracked_pid and _is_pid_running(tracked_pid):
            time.sleep(0.1)
            continue

        time.sleep(0.1)

    return False


# Return whether ASLM module infrastructure launched the current process.
def _is_running_inside_aslm() -> bool:
    return bool(os.environ.get("ASLM_MODULE_ID") or os.environ.get("ASLM_MODULE_DIR"))


# Public lifecycle

# Build Windows process settings that never allocate or show a console window.
def _hidden_subprocess_kwargs() -> dict[str, object]:
    if os.name != "nt":
        return {}

    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= getattr(subprocess, "STARTF_USESHOWWINDOW", 0)
    startupinfo.wShowWindow = getattr(subprocess, "SW_HIDE", 0)
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startupinfo,
    }


# Spawn Ollama as a supervised child whose output remains attached to ASLM.
def _spawn_ollama_process(ollama_path: str, env: dict[str, str]) -> subprocess.Popen:
    return subprocess.Popen(
        [ollama_path, "serve"],
        env=env,
        # ASLM intentionally closes the module root's redirected stdin. Newer
        # Ollama builds treat that inherited EOF as a clean shutdown signal, so
        # give the server its own pipe and keep the writer alive in this supervisor.
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        **_hidden_subprocess_kwargs(),
    )


# Terminate one managed Ollama process together with its descendants.
def _terminate_process_tree(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **_hidden_subprocess_kwargs(),
        )
        return

    os.kill(pid, signal.SIGTERM)


# Start the local Ollama service when the active engine requires it.
def start_ollama(engine: str | None = None) -> bool:
    global _ollama_process

    # Check whether the current engine selection should manage Ollama at all.
    desired_state = _get_desired_state(engine)
    if not desired_state.should_run:
        return False

    # Reuse an already available runtime instead of starting a new one.
    if _wait_for_existing_runtime(timeout_seconds=1.5):
        tracked_pid = _read_pid()
        if tracked_pid and _is_pid_running(tracked_pid):
            logger.info("Ollama service is already running (PID: %s)", tracked_pid)
        else:
            logger.info("Ollama service is already reachable on the configured port.")
        return True

    # Give the dedicated ASLM runtime a chance to come up before falling back.
    if _is_running_inside_aslm():
        logger.info(
            "Waiting for dedicated Ollama runtime process to become ready inside ASLM."
        )
        if _wait_until_ready(timeout_seconds=8.0):
            logger.info("Dedicated Ollama runtime became reachable on the configured port.")
            return True

        _print_status(
            "Dedicated Ollama runtime is not ready yet. "
            "Starting fallback local Ollama process..."
        )

    # Avoid duplicate local launches when the PID file becomes valid meanwhile.
    tracked_pid = _read_pid()
    if tracked_pid and _is_pid_running(tracked_pid):
        logger.info("Ollama service is already running (PID: %s)", tracked_pid)
        return True

    # Validate the configured executable path before spawning the runtime.
    ollama_path = settings.get("ollama-service_path")
    if not ollama_path or not os.path.exists(ollama_path):
        _print_status(f"Ollama service is enabled but not found at: {ollama_path}")
        return False

    # Prepare the launch environment. Runtime output stays in the current
    # process stream and is never persisted inside the updatable module.
    env, ollama_port = _build_service_environment()

    _print_status(f"Starting local Ollama service on port {ollama_port}...")
    _print_status(
        "Ollama runtime configured: "
        f"host={env.get('OLLAMA_HOST', f'127.0.0.1:{ollama_port}')}, "
        f"models={env.get('OLLAMA_MODELS', '(default)')}"
    )

    # Spawn a hidden local runtime and consume its output through a pipe.
    process: subprocess.Popen | None = None
    try:
        process = _spawn_ollama_process(str(ollama_path), env)
        _ollama_process = process
        _ensure_log_streaming(process)
        _write_pid(process.pid)

        # Report startup status based on HTTP readiness, not process spawn alone.
        if not _wait_until_ready(process=process):
            if process.poll() is not None:
                _clear_pid()
                _ollama_process = None
                _print_status(
                    f"Ollama service exited during startup with code {process.returncode}."
                )
                return False

            logger.warning(
                "Ollama process started but the HTTP endpoint did not become ready in time."
            )
            _print_status(
                f"Ollama service started (PID: {process.pid}) "
                "but did not become ready in time."
            )
            return False
        else:
            _print_status(f"Ollama service started successfully (PID: {process.pid})")

        return True
    except Exception as exc:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process.pid)
        _ollama_process = None
        _clear_pid()
        _print_status(f"Failed to start Ollama service: {exc}")
        return False


# Supervise ollama serve for the dedicated ASLM runtime command.
def run_ollama_runtime(log: bool = False) -> int:
    global _ollama_process

    # Skip the command when the managed runtime is disabled.
    desired_state = _get_desired_state("ollama-service")
    if not desired_state.is_enabled:
        _print_status("Ollama runtime command skipped: ollama-service is not enabled.")
        return 0

    # Validate the configured runtime executable.
    ollama_path = settings.get("ollama-service_path")
    if not ollama_path or not os.path.exists(ollama_path):
        _print_status(f"Ollama runtime command failed: executable not found at {ollama_path}")
        return 1

    # Print the launch configuration, and skip the launch when another Ollama already answers.
    env, ollama_port = _build_service_environment()

    if log:
        _print_status(f"Launching dedicated Ollama runtime on port {ollama_port}...")
    _print_status(
        "Ollama runtime configured: "
        f"host={env.get('OLLAMA_HOST', f'127.0.0.1:{ollama_port}')}, "
        f"models={env.get('OLLAMA_MODELS', '(default)')}"
    )

    if _wait_until_ready(timeout_seconds=0.25):
        _print_status("Ollama runtime is already reachable; dedicated launch is not required.")
        return 0

    # Spawn the runtime and record its PID; on any failure kill a half-started process and report it.
    process: subprocess.Popen | None = None
    try:
        process = _spawn_ollama_process(str(ollama_path), env)
        _ollama_process = process
        _write_pid(process.pid)
        _ensure_log_streaming(process)
    except Exception as exc:
        if process is not None and process.poll() is None:
            _terminate_process_tree(process.pid)
        _ollama_process = None
        _clear_pid()
        _print_status(f"Failed to launch dedicated Ollama runtime: {exc}")
        return 1

    # Report readiness, then block until the runtime exits so ASLM supervises its lifetime. Ctrl+C
    # stops the whole process tree, and the finally block releases the stdin pipe and the PID file.
    try:
        if _wait_until_ready(timeout_seconds=15.0, process=process):
            _print_status(f"Dedicated Ollama runtime is ready (PID: {process.pid}).")
        else:
            exit_code = process.poll()
            if exit_code is None:
                _print_status(
                    f"Dedicated Ollama runtime is still running but its endpoint is not ready "
                    f"(PID: {process.pid})."
                )
            else:
                _print_status(
                    f"Dedicated Ollama runtime exited before becoming ready "
                    f"(PID: {process.pid}, code: {exit_code})."
                )

        return process.wait()
    except KeyboardInterrupt:
        if process.poll() is None:
            _terminate_process_tree(process.pid)
        return process.wait()
    finally:
        if process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                # The pipe is already closed because the process is gone.
                pass
        _ollama_process = None
        _clear_pid()


# Stop the managed Ollama service when a tracked PID exists.
def stop_ollama() -> None:
    global _ollama_process

    # Prefer the live in-process handle during the small window before its PID
    # file is persisted; otherwise stop the separately supervised ASLM child.
    current_process = _ollama_process
    pid = _read_pid()
    if not pid and current_process is not None and current_process.poll() is None:
        pid = current_process.pid

    # Exit quietly when there is no tracked runtime.
    if not pid:
        _ollama_process = None
        return

    try:
        _terminate_process_tree(pid)
    except OSError:
        logger.info("Managed Ollama process %s was already stopped.", pid)
    finally:
        _ollama_process = None
        _clear_pid()


# Explain where managed Ollama output is available without creating log files.
def run_ollama_console(log: bool = False) -> None:
    # The log argument is reserved for future verbosity controls.
    del log
    _print_status(
        "Persistent Ollama logs are disabled. Live output is streamed by the "
        "ASLM process that launches the managed runtime."
    )
