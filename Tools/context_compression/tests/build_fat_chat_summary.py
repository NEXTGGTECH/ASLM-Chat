# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make the Tools directory importable so the script can run directly as a standalone module.
TOOLS_ROOT = Path(__file__).resolve().parents[2]
if str(TOOLS_ROOT) not in sys.path:
    sys.path.insert(0, str(TOOLS_ROOT))

from context_compression.history_compressor import build_structured_history_summary
from context_compression.tests.cache_chat_utils import (
    REPORTS_DIR,
    collect_chat_entries,
    connect_cache_db,
    load_fattest_chat,
    resolve_cache_db_path,
)

# JSON report written by main().
OUT_PATH = REPORTS_DIR / "fat_chat_summary.json"


# Build a raw-only structured summary for the largest cached chat and write JSON output.
def main() -> None:
    # Locate the cache database and pick the chat with the most content.
    db_path = resolve_cache_db_path()
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    conn = connect_cache_db(db_path)
    fat = load_fattest_chat(conn)
    if fat is None:
        raise SystemExit(f"No chats found in cache database: {db_path}")
    entries, recent_user_messages = collect_chat_entries(conn, fat["id"])

    # Build the summary without a model callback, so it contains only the raw harvest.
    summary_text, payload = build_structured_history_summary(
        overflow_entries=entries,
        recent_user_messages=recent_user_messages,
        direct_user_directives=[],
        summarize_with_model=None,
        max_overflow_entries=120,
    )

    # Write the chat statistics and the summary payload as the JSON report.
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "fat_chat": {
                    "id": fat["id"],
                    "title": fat["title"],
                    "messages": fat["messages"],
                    "chars": fat["chars"],
                },
                "summary_chars": len(summary_text),
                "summary_payload": payload,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # Print a short console report.
    print(f"Wrote: {OUT_PATH}")
    print(f"Chat: {fat['id']} ({fat['chars']} chars)")
    print(f"Summary chars: {len(summary_text)}")


# Run from the Tools directory: python -m context_compression.tests.build_fat_chat_summary
if __name__ == "__main__":
    main()
