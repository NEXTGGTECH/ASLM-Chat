# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

from typing import Any, Mapping

from . import control


# Return the runner module, importing it lazily to avoid loading model and sandbox integrations
# during ordinary chat startup.
def _implementation():
    from . import runner

    return runner


# Create the durable session before the long-running response starts. Return the tool arguments
# for the later run (with the clamped round count and a fresh session id) and the initial
# "planning" tool UI payload. Raise ValueError when the topic is empty.
def prepare_research(topic: str, *, instructions: str = "", max_rounds: int = 6) -> dict[str, Any]:
    # Validate the topic and clamp the round count to 2..25 (6 when it is not a number).
    clean_topic = str(topic or "").strip()
    if not clean_topic:
        raise ValueError("Deep research requires a non-empty topic.")
    try:
        rounds = min(25, max(2, int(max_rounds or 6)))
    except (TypeError, ValueError):
        # A non-numeric round count falls back to the default.
        rounds = 6

    # Arguments handed to the research run once the response starts.
    session_id = control.new_session_id()
    arguments = {
        "topic": clean_topic,
        "instructions": str(instructions or "").strip(),
        "max_rounds": rounds,
        "session_id": session_id,
        "approval_timeout_s": 900,
        "timeout_s": 3600,
    }

    # Initial UI payload: planning phase, no plan yet, the query budget is two queries per round.
    tool_ui = {
        "kind": "deep_research",
        "status": "planning",
        "topic": clean_topic,
        "session_id": session_id,
        "plan_version": 0,
        "checklist": [],
        "queries_used": 0,
        "query_budget": rounds * 2,
        "can_approve": False,
        "can_edit": False,
        "can_stop": True,
    }

    # Persist the session so approve, edit and stop commands can find it from the start.
    control.create_session(
        session_id,
        topic=clean_topic,
        extra={
            "status": "planning",
            "phase": "planning",
            "query_budget": rounds * 2,
            "can_approve": False,
            "can_edit": False,
            "can_stop": True,
        },
    )
    return {"arguments": arguments, "tool_ui": tool_ui}


# Run the existing deterministic orchestrator through the app service contract.
def run_research(arguments: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    return _implementation().run_deep_research(arguments, context)
