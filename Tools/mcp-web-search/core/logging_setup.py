# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path

# Directory that holds the rotating log files (the logs folder next to the core package).
_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"

# Logger name -> log file name. Covers the web_search and read_page services plus the MCP adapter;
# serp_api/triage/health/quality log under the `core` namespace and land in core.log via
# propagation.
_SERVICE_LOGS: dict[str, str] = {
    "services.web_search": "web_search.log",
    "trace.web_search": "web_search_trace.log",
    "services.read_page": "read_page.log",
    "trace.read_page": "read_page_trace.log",
    "mcp.server": "mcp_trace.log",
    # FastMCP stdio connector. Its own file + propagate=False keeps its lines off the
    # root console handler — a stdio server that logs to stderr can WEDGE if the client
    # stops draining the pipe mid-generation (the write is synchronous and blocks the
    # event loop, so even asyncio.timeout can't fire). See adapters.mcp.server hardening.
    "adapters.mcp.server": "mcp_adapter.log",
    # Warm-browser layer (daemon + client + identity store). The daemon runs as its own
    # windowless process, so file logging is the only way to see its activity.
    "core.fetch.browser": "browser_daemon.log",
    "core": "core.log",
}

# Line and timestamp format shared by every log file.
_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


# Attach a rotating file handler to every logger in _SERVICE_LOGS (idempotent: a logger that already
# has one is left alone). Each logger stops propagating, so its lines do not reach the root console
# handler.
def setup_logging(
    log_dir: Path | None = None,
    level: int = logging.INFO,
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> None:
    target_dir = log_dir or _LOG_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    for logger_name, filename in _SERVICE_LOGS.items():
        log = logging.getLogger(logger_name)
        log.setLevel(level)

        # Skip loggers that were already configured by an earlier call.
        if any(isinstance(handler, logging.handlers.RotatingFileHandler) for handler in log.handlers):
            continue

        # Write this logger to its own size-rotated file and keep it off the root handlers.
        handler = logging.handlers.RotatingFileHandler(
            target_dir / filename,
            encoding="utf-8",
            maxBytes=max_bytes,
            backupCount=backup_count,
        )
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.propagate = False
