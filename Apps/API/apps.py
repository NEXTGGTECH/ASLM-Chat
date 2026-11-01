# Copyright NEXTGGTECH. Elastic License 2.0.

import logging
import os
import threading

from django.apps import AppConfig

logger = logging.getLogger(__name__)


# Configure the API application.
class ApiConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "Apps.API"

    def ready(self) -> None:
        """Prepare enabled engine runtimes without blocking Django startup."""

        # ASLM owns long-running engine processes through dedicated manifest
        # run commands. Starting the standalone reconciliation path here would
        # only import every enabled adapter in parallel with the first UI page,
        # competing for the Python import lock and delaying WebView startup.
        if os.environ.get("ASLM_MODULE_ID") or os.environ.get("ASLM_MODULE_DIR"):
            logger.debug("ASLM owns engine runtime startup; skipping Django runtime sync.")
            return

        def _sync_enabled_engine_runtimes() -> None:
            try:
                from API import llm_api

                llm_api.sync_enabled_engine_runtimes()
            except Exception as exc:
                logger.warning("Failed to sync enabled engine runtimes on startup: %s", exc)

        thread = threading.Thread(
            target=_sync_enabled_engine_runtimes,
            name="aslm-chat-engine-runtime-sync",
            daemon=True,
        )
        thread.start()
