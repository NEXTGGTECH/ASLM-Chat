# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import re
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

from API import llm_api, mcp as tool_registry
from . import control


# Search tool aliases a session may use, and limits: queries per iteration, checklist items,
# query candidates per model stage, and public events kept in the event tail.
ALLOWED_SEARCH_ALIASES = (
    "web_search__web_search",
    "web_search__read_page",
)
MAX_QUERIES_PER_ITERATION = 2
MAX_PLAN_ITEMS = 7
MAX_QUERY_CANDIDATES = 6
MAX_PUBLIC_EVENTS = 160

# Medium web search returns up to ten results for each of the two queries in a
# batch. Keep that complete normal result set in the public event so the UI's
# own 3 + More overflow control can expose every source that was found.
MAX_PUBLIC_SEARCH_SOURCES = 20

# Character limits for the evidence given to the model: the whole evidence packet and a single
# evidence entry.
MODEL_EVIDENCE_CHARS = 48_000
MODEL_ENTRY_CHARS = 9_000

# System prompt of the planning stage: return a small JSON plan with checklist steps and candidate
# searches.
PLAN_PROMPT = """You plan a web research task. Do not answer it and do not search.
Think carefully about the deliverable and evidence needed. Return one small JSON object:
{"summary":"short approach","steps":[{"id":"s1","title":"verifiable evidence goal"}],"candidates":[{"text":"plain search terms","vertical":"web|academic|shopping","purpose":"gap unlocked"}]}
Use 3-6 steps and 3-6 distinct candidate queries. Keep query text under 140 characters.
The plan will be shown to the user for approval. Return JSON only."""

# System prompt of the reflection stage: audit the evidence against the checklist and return JSON
# with an assessment, gaps, checklist updates, a completion flag and new candidate searches.
REFLECTION_PROMPT = """You are the deliberate reflection stage of a research system.
No tools are available and no search happens in this turn. Spend real effort auditing the
evidence against every checklist item. Find contradictions, missing source classes, and the
highest-value unresolved gap. Then design several genuinely different candidate searches.
Return one small JSON object only:
{"assessment":"public concise evidence audit","gaps":["..."],"updates":[{"id":"s1","status":"pending|active|done|blocked","note":"public reason"}],"complete":false,"candidates":[{"text":"plain search terms","vertical":"web|academic|shopping","purpose":"why this query has high marginal value"}]}
Create 4-6 candidates when research is incomplete. Do not repeat a query from history. Query
text must be plain search terms, not a question or explanation, and stay under 140 characters."""

# System prompt of the query-budget gate: pick the one or two most valuable candidate queries as JSON.
QUERY_SELECTION_PROMPT = """You are the query-budget gate. Search slots are scarce.
Compare the proposed candidates against the evidence gap and query history. Select only the
one or two queries with the highest expected information gain, source quality, and novelty.
Prefer different source classes when two are selected. Return JSON only:
{"queries":[{"text":"plain search terms","vertical":"web|academic|shopping","purpose":"short selection rationale"}]}
Never select more than two and never repeat an earlier query."""

# System prompt of the final report stage: a cited Markdown report built only from the evidence.
REPORT_PROMPT = """Write the final self-contained research report in the user's language.
Use only the supplied evidence. Put the exact opaque citation handle (for example [c0001-2])
immediately after each supported claim; never invent or alter handles. Address every requested
deliverable, distinguish evidence from inference, and state unresolved gaps or disagreements.
Return the report itself, not JSON and not a diary of the research process.

Format the report as structured Markdown. Use LaTeX for equations or quantitative reasoning when
it improves precision. Use valid fenced Mermaid diagrams for processes, architectures, timelines,
or relationships when a diagram materially improves understanding. Keep diagrams evidence-based
and syntactically valid; do not add them merely as decoration."""


# Errors and events

# Cooperative cancellation requested by the user.
class ResearchCancelled(Exception):
    pass


# Error raised when the selected model failed before completing a required model stage.
class ResearchModelUnavailable(RuntimeError):
    pass


# Compact, ordered, public research events of one session. Each event gets a sequence number that
# continues from the stored state, is appended to the session's JSONL log, kept in a bounded
# in-memory tail and passed to the optional callback.
class SemanticEventStream:
    schema_version = 2

    # Resume the sequence and the event tail from the stored session state and open the JSONL log
    # (append mode, line-buffered).
    def __init__(
        self,
        session_id: str,
        *,
        callback: Callable[[dict[str, Any]], None] | None,
        logs_dir: Path,
    ) -> None:
        self.session_id = session_id
        self.callback = callback if callable(callback) else None

        # Continue numbering after the last stored event and keep only the bounded stored tail.
        existing_state = control.read_state(session_id)
        self.sequence = max(0, int(existing_state.get("last_sequence") or 0))
        self.events = [
            dict(event)
            for event in existing_state.get("event_tail") or []
            if isinstance(event, Mapping)
        ][-MAX_PUBLIC_EVENTS:]

        # The reentrant lock serializes emitters; the log file is named after the session key.
        self._lock = threading.RLock()
        logs_dir.mkdir(parents=True, exist_ok=True)
        key = control.session_key(session_id)
        self.path = (logs_dir / f"deep-research-{key}.jsonl").resolve()
        self._handle = self.path.open("a", encoding="utf-8", buffering=1)

    # Append one public event to the log and the in-memory tail (trimmed to MAX_PUBLIC_EVENTS) and
    # pass it to the callback. Return the event record.
    def emit(
        self,
        event_type: str,
        *,
        phase: str,
        data: Mapping[str, Any] | None = None,
        iteration: int = 0,
        plan_version: int = 0,
    ) -> dict[str, Any]:
        # Number, write and store the record under the lock so the log stays ordered.
        with self._lock:
            self.sequence += 1
            record = {
                "schema_version": self.schema_version,
                "session_id": self.session_id,
                "sequence": self.sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": str(phase),
                "iteration": max(0, int(iteration or 0)),
                "plan_version": max(0, int(plan_version or 0)),
                "type": str(event_type),
                "data": dict(data or {}),
            }
            self._handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._handle.flush()
            self.events.append(record)
            if len(self.events) > MAX_PUBLIC_EVENTS:
                del self.events[: len(self.events) - MAX_PUBLIC_EVENTS]

        # Notify the observer outside the lock.
        if self.callback is not None:
            try:
                self.callback(record)
            except Exception:
                # Observers must never interrupt research.
                pass
        return record

    # Close the log file; calling it again is harmless.
    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.close()


# Model output parsing

# Convert a dict, a mapping or a model object with model_dump into a plain dict. Anything else
# yields an empty dict.
def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, dict) else {}
    return {}


# Normalize a generation result to an iterator of chunks. None yields nothing, a mapping or model
# object is a single chunk, a string is wrapped as one assistant message chunk and any other
# iterable is streamed as it is.
def _iter_generation(result: Any) -> Iterator[Any]:
    if result is None:
        return
    if isinstance(result, (dict, Mapping)) or callable(getattr(result, "model_dump", None)):
        yield result
        return
    if isinstance(result, (str, bytes)):
        yield {"message": {"role": "assistant", "content": str(result)}}
        return
    if isinstance(result, Iterable):
        yield from result
        return
    yield result


# Return the dict stored under key in a chunk, or an empty dict when it is missing.
def _message_from_chunk(chunk: Any, key: str) -> dict[str, Any]:
    return _as_dict(_as_dict(chunk).get(key))


# Collapse whitespace in a search query and cut it to 160 characters, trimming stray separators.
def _normalize_query_text(value: Any) -> str:
    text = re.sub(r"\s+", " ", str(value or "").replace("\r", " ").replace("\n", " ")).strip()
    return text[:160].strip(" ,;|")


# Return a lower-cased signature of the query's letters and digits (Latin and Cyrillic), used to
# detect repeated queries regardless of punctuation and spacing.
def _query_signature(value: Any) -> str:
    return re.sub(r"[^a-z0-9а-яё]+", " ", str(value or "").casefold(), flags=re.I).strip()


# Return the search vertical (web, academic or shopping), defaulting to web for anything else.
def _safe_vertical(value: Any) -> str:
    vertical = str(value or "web").strip().lower()
    return vertical if vertical in {"web", "academic", "shopping"} else "web"


# Interpret a model-provided flag strictly: a real boolean, the number 1, or the text true, yes or 1
# is True; everything else is False.
def _strict_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value == 1
    return str(value or "").strip().casefold() in {"true", "yes", "1"}


# Extract the first JSON object from model output: the whole text, the contents of a fenced code
# block, or an object starting at any brace inside either. Return an empty dict when there is none.
def _extract_json_object(text: Any) -> dict[str, Any]:
    source = str(text or "").strip()
    if not source:
        return {}

    # Candidate texts: the raw output first, then the contents of fenced code blocks.
    candidates = [source]
    candidates.extend(
        match.group(1).strip()
        for match in re.finditer(r"```(?:json)?\s*(.*?)```", source, flags=re.I | re.S)
    )
    decoder = json.JSONDecoder()
    for candidate in candidates:
        # Try the whole candidate, then an object embedded at any opening brace.
        try:
            parsed = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                parsed, _end = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                # Not a JSON object at this brace; try the next one.
                continue
            if isinstance(parsed, dict):
                return parsed
    return {}


# Plan and checklist

# Turn a plain-text plan into checklist titles: use the bullet, numbered or checkbox lines when
# there are any, otherwise every non-empty line. Lines shorter than 4 characters or starting with a
# brace are dropped, titles are cut to 220 characters and the list stops at MAX_PLAN_ITEMS.
def _plain_plan_lines(text: Any) -> list[str]:
    raw_lines = [line for line in str(text or "").splitlines() if line.strip()]
    marker = r"^\s*(?:(?:[-*+]|\d+[.)])\s+(?:\[[ xX]\]\s*)?|\[[ xX]\]\s*)"
    marked_lines = [
        line
        for line in raw_lines
        if re.match(marker, line)
    ]
    output: list[str] = []
    for line in marked_lines or raw_lines:
        clean = re.sub(marker, "", line).strip()
        if len(clean) >= 4 and not clean.startswith(("{", "}")):
            output.append(clean[:220])
        if len(output) >= MAX_PLAN_ITEMS:
            break
    return output


# Normalize raw plan steps (dicts or plain strings) into checklist items with a safe id, a title of
# up to 220 characters, a valid status and an optional note. When no step is usable, build the
# checklist from the lines of fallback_text instead.
def _normalize_checklist(raw_steps: Any, fallback_text: str = "") -> list[dict[str, Any]]:
    steps = raw_steps if isinstance(raw_steps, list) else []
    normalized: list[dict[str, Any]] = []
    for index, raw_step in enumerate(steps[:MAX_PLAN_ITEMS], start=1):
        # Dict steps may name the title title, step or goal; anything else is a plain title.
        if isinstance(raw_step, dict):
            title = str(raw_step.get("title") or raw_step.get("step") or raw_step.get("goal") or "").strip()
            raw_id = str(raw_step.get("id") or f"s{index}").strip()
            status = str(raw_step.get("status") or "pending").strip().lower()
            note = str(raw_step.get("note") or "").strip()
        else:
            title = str(raw_step or "").strip()
            raw_id = f"s{index}"
            status = "pending"
            note = ""
        if not title:
            continue

        # Sanitize the id and clamp the field sizes; unknown statuses become pending.
        item = {
            "id": re.sub(r"[^a-zA-Z0-9_-]+", "-", raw_id).strip("-")[:40] or f"s{index}",
            "title": title[:220],
            "status": status if status in {"pending", "active", "done", "blocked", "skipped"} else "pending",
        }
        if note:
            item["note"] = note[:300]
        normalized.append(item)
    if normalized:
        return normalized
    return [
        {"id": f"s{index}", "title": title, "status": "pending"}
        for index, title in enumerate(_plain_plan_lines(fallback_text), start=1)
    ]


# Return the fixed five-step checklist used when no usable plan exists.
def _fallback_checklist(topic: str) -> list[dict[str, Any]]:
    return [
        {"id": "s1", "title": "Define the answer-critical claims and evidence classes", "status": "pending"},
        {"id": "s2", "title": f"Find authoritative sources for {topic[:120]}", "status": "pending"},
        {"id": "s3", "title": "Check important claims against independent evidence", "status": "pending"},
        {"id": "s4", "title": "Resolve contradictions and document remaining gaps", "status": "pending"},
        {"id": "s5", "title": "Synthesize a cited answer", "status": "pending"},
    ]


# Normalize raw query candidates (dicts or plain strings) into unique text, vertical and purpose
# entries, up to limit. Duplicates are detected by query signature.
def _normalize_candidates(raw: Any, *, limit: int = MAX_QUERY_CANDIDATES) -> list[dict[str, str]]:
    items = raw if isinstance(raw, list) else []
    output: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in items:
        # Dict candidates may use alias keys for the text and the purpose; strings are web queries.
        if isinstance(item, dict):
            text = _normalize_query_text(item.get("text") or item.get("query") or item.get("q"))
            vertical = _safe_vertical(item.get("vertical"))
            purpose = str(item.get("purpose") or item.get("rationale") or item.get("why") or "").strip()
        else:
            text = _normalize_query_text(item)
            vertical = "web"
            purpose = ""

        # Skip empty and repeated queries.
        signature = _query_signature(text)
        if not signature or signature in seen:
            continue
        seen.add(signature)
        output.append({"text": text, "vertical": vertical, "purpose": purpose[:240]})
        if len(output) >= max(1, int(limit or 1)):
            break
    return output


# Render the plan as a summary line followed by an unchecked Markdown task list of the checklist
# titles.
def _plan_text(summary: str, checklist: list[dict[str, Any]]) -> str:
    header = str(summary or "Research the request through verifiable evidence.").strip()
    lines = [header, "", *[f"- [ ] {item['title']}" for item in checklist]]
    return "\n".join(lines).strip()


# Commands

# UI commands of one session, delivered once each. The file names already returned are remembered
# and acknowledged to control.read_commands, which then removes those files.
class CommandInbox:
    # Start with no processed commands.
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.processed: set[str] = set()

    # Return the commands that arrived since the previous poll, oldest first.
    def poll(self) -> list[dict[str, Any]]:
        commands: list[dict[str, Any]] = []
        for filename, command in control.read_commands(self.session_id, processed=self.processed):
            self.processed.add(filename)
            commands.append(command)
        return commands


# Return True when the user has asked to cancel the session.
def _cancel_requested(session_id: str) -> bool:
    return control.latest_cancel_requested(session_id)


# Model stages

# Consume a model generation and return its final text: the last transcript message without tool
# calls, otherwise the concatenated visible content. Check for a cancel command about every 0.2
# seconds; on cancel abort the engine's generation and raise ResearchCancelled.
def _collect_model_text(result: Any, *, engine: str, session_id: str) -> str:
    visible: list[str] = []
    final_transcript = ""
    last_cancel_check = 0.0
    for raw_chunk in _iter_generation(result):
        # Throttled cancellation check.
        now = time.monotonic()
        if now - last_cancel_check >= 0.2:
            last_cancel_check = now
            if _cancel_requested(session_id):
                try:
                    llm_api.abort_generation(engine)
                finally:
                    raise ResearchCancelled()

        # Collect visible content; a transcript message without tool calls is the finished answer.
        message = _message_from_chunk(raw_chunk, "message")
        content = message.get("content")
        if isinstance(content, str) and content:
            visible.append(content)
        transcript = _message_from_chunk(raw_chunk, "transcript_message")
        transcript_content = transcript.get("content")
        if isinstance(transcript_content, str) and transcript_content.strip() and not transcript.get("tool_calls"):
            final_transcript = transcript_content.strip()
    return final_transcript or "".join(visible).strip()


# Run one model stage (planning, reflection, ...) with a system and a user prompt and return the
# model's text. Image attachments of the original request are forwarded. Stage started, completed
# and failed events are emitted. Cancellation propagates unchanged; any other failure is raised
# as ResearchModelUnavailable so research never continues without a model answer.
def _run_model_stage(
    *,
    engine: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    generation_options: Mapping[str, Any],
    session_id: str,
    events: SemanticEventStream,
    phase: str,
    iteration: int,
    plan_version: int,
    request_message: Mapping[str, Any] | None = None,
) -> str:
    # Announce the stage, then build the user message with the request's attachments.
    events.emit(
        "model_stage_started",
        phase=phase,
        iteration=iteration,
        plan_version=plan_version,
        data={"stage": phase},
    )
    try:
        user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
        if isinstance(request_message, Mapping):
            for key in ("images", "image_mime_types", "media"):
                value = request_message.get(key)
                if isinstance(value, list) and value:
                    user_message[key] = list(value)

        # Generate and collect the stage output.
        result = llm_api.generate(
            engine,
            model,
            [
                {"role": "system", "content": system_prompt},
                user_message,
            ],
            **dict(generation_options),
        )
        text = _collect_model_text(result, engine=engine, session_id=session_id)
    except ResearchCancelled:
        # A user cancellation is not a model failure.
        raise
    except Exception as exc:
        events.emit(
            "model_stage_failed",
            phase=phase,
            iteration=iteration,
            plan_version=plan_version,
            data={"stage": phase, "error_type": type(exc).__name__, "message": str(exc)[:400]},
        )

        # Provider adapters use different exception classes for connection,
        # timeout, authentication, missing-model and lazy stream failures. Any
        # exception here means the selected model did not return a response;
        # never manufacture a plan or query and continue into web search.
        raise ResearchModelUnavailable(
            f"The selected model '{model}' failed during {phase}: "
            f"{type(exc).__name__}: {str(exc)[:300]}"
        ) from exc

    # Report the successful stage.
    events.emit(
        "model_stage_completed",
        phase=phase,
        iteration=iteration,
        plan_version=plan_version,
        data={"stage": phase, "output_chars": len(text)},
    )
    return text


# Checklist progress

# Apply the model's checklist updates (id, status and note) to a copy of the checklist and return
# it. A finished (done or skipped) item is never moved back to an open status, and its note is not
# overwritten by the rejected update. Only one item stays active.
def _apply_updates(checklist: list[dict[str, Any]], raw_updates: Any) -> list[dict[str, Any]]:
    updates = raw_updates if isinstance(raw_updates, list) else []
    by_id = {str(item.get("id")): dict(item) for item in checklist}
    preferred_active_id = ""
    rejected_regressions: set[str] = set()
    for update in updates:
        if not isinstance(update, dict):
            continue
        item_id = str(update.get("id") or "").strip()
        if item_id not in by_id:
            continue
        status = str(update.get("status") or "").strip().lower()
        if status in {"pending", "active", "done", "blocked", "skipped"}:
            previous_status = str(by_id[item_id].get("status") or "pending").strip().lower()

            # Evidence already accepted for a checklist item must not disappear
            # because a later reflection only considered the newest batch or
            # emitted a stale `pending`/`active` status.  A plan revision can
            # still replace the item by changing its title/identity.
            if previous_status not in {"done", "skipped"} or status in {"done", "skipped"}:
                by_id[item_id]["status"] = status
                rejected_regressions.discard(item_id)
            else:
                rejected_regressions.add(item_id)
            if status == "active" and previous_status not in {"done", "skipped"}:
                preferred_active_id = item_id
        note = str(update.get("note") or "").strip()
        if note and item_id not in rejected_regressions:
            by_id[item_id]["note"] = note[:300]
    merged = [by_id[str(item.get("id"))] for item in checklist]
    return _cohere_checklist_progress(merged, preferred_active_id=preferred_active_id)


# Return a whitespace-normalized, case-folded title used to match checklist items across plan versions.
def _checklist_title_key(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip()).casefold()


# Keep one current item while leaving evidence completion non-sequential. Checklist entries are
# evidence goals, not a wizard. A later goal may be proven before an earlier one, so this
# intentionally does not auto-complete preceding rows. It only makes the transient `active` state
# unambiguous: the preferred item (else the first active one) stays active and the others return
# to pending.
def _cohere_checklist_progress(
    checklist: list[dict[str, Any]],
    *,
    preferred_active_id: str = "",
) -> list[dict[str, Any]]:
    normalized = [dict(item) for item in checklist]
    active_ids = [
        str(item.get("id") or "")
        for item in normalized
        if str(item.get("status") or "pending").strip().lower() == "active"
    ]
    keep_active = (
        preferred_active_id
        if preferred_active_id and preferred_active_id in active_ids
        else (active_ids[0] if active_ids else "")
    )
    for item in normalized:
        item_id = str(item.get("id") or "")
        status = str(item.get("status") or "pending").strip().lower()
        if status == "active" and item_id != keep_active:
            item["status"] = "pending"
    return normalized


# Carry done and skipped progress from the previous checklist onto the matching items of a reloaded
# or revised one. Items match by id when the title is unchanged, otherwise by unique title; a
# finished item keeps its status and, when the new item has none, its note.
def _merge_checklist_progress(
    previous: Any,
    current: Any,
) -> list[dict[str, Any]]:
    previous_items = previous if isinstance(previous, list) else []
    current_items = current if isinstance(current, list) else []

    # Index the previous items by id and by title; a title that occurs more than once is ambiguous
    # and is not used for matching.
    previous_by_id = {
        str(item.get("id") or ""): item
        for item in previous_items
        if isinstance(item, dict) and str(item.get("id") or "")
    }
    previous_by_title: dict[str, dict[str, Any]] = {}
    duplicate_titles: set[str] = set()
    for item in previous_items:
        if not isinstance(item, dict):
            continue
        title_key = _checklist_title_key(item.get("title"))
        if not title_key:
            continue
        if title_key in previous_by_title:
            duplicate_titles.add(title_key)
        else:
            previous_by_title[title_key] = item
    for title_key in duplicate_titles:
        previous_by_title.pop(title_key, None)

    # Find the previous version of each current item and restore finished progress.
    output: list[dict[str, Any]] = []
    for raw_item in current_items:
        if not isinstance(raw_item, dict):
            continue
        item = dict(raw_item)
        item_id = str(item.get("id") or "")
        title_key = _checklist_title_key(item.get("title"))
        prior = previous_by_id.get(item_id)
        if prior is not None and _checklist_title_key(prior.get("title")) != title_key:
            # Generated ids such as s1 are positional and can be reused for a
            # genuinely different revised goal.  Do not inherit completion.
            prior = None
        if prior is None and title_key:
            prior = previous_by_title.get(title_key)
        prior_status = str((prior or {}).get("status") or "pending").strip().lower()
        current_status = str(item.get("status") or "pending").strip().lower()
        if prior_status in {"done", "skipped"} and current_status not in {"done", "skipped"}:
            item["status"] = prior_status
            if (
                not str(item.get("note") or "").strip()
                and str((prior or {}).get("note") or "").strip()
            ):
                item["note"] = str(prior.get("note"))[:300]
        output.append(item)
    return _cohere_checklist_progress(output)


# Return the id of the active item, else of the first unfinished item, else an empty string.
def _current_checklist_item_id(checklist: list[dict[str, Any]]) -> str:
    for item in checklist:
        if str(item.get("status") or "pending").strip().lower() == "active":
            return str(item.get("id") or "")
    for item in checklist:
        if str(item.get("status") or "pending").strip().lower() not in {"done", "skipped"}:
            return str(item.get("id") or "")
    return ""


# Plan approval

# Normalize a plan edited by the user: its lines become the checklist, the default checklist is used
# when it has no usable lines, and finished progress of the previous checklist is carried over.
# Return the plan text and the checklist.
def _normalize_user_plan(
    plan: Any,
    topic: str,
    previous_checklist: Any = None,
) -> tuple[str, list[dict[str, Any]]]:
    plan_text = str(plan or "").strip()
    checklist = _normalize_checklist([], plan_text)
    if not checklist:
        checklist = _fallback_checklist(topic)
        plan_text = _plan_text("User-adjusted research plan", checklist)
    checklist = _merge_checklist_progress(previous_checklist, checklist)
    return plan_text, checklist


# Process the commands waiting in the inbox, oldest first. A cancel command raises
# ResearchCancelled. Approve and revise commands issued against another plan version are rejected
# with an event. A command that carries plan text replaces the plan (bumping the plan version,
# emitting plan_updated and persisting the state); an approve command marks the plan approved.
# Return the plan, checklist, plan version and whether the plan was approved.
def _handle_commands(
    inbox: CommandInbox,
    *,
    topic: str,
    plan: str,
    checklist: list[dict[str, Any]],
    plan_version: int,
    events: SemanticEventStream,
    iteration: int,
) -> tuple[str, list[dict[str, Any]], int, bool]:
    approved = False
    for command in inbox.poll():
        action = str(command.get("action") or "").strip().lower()
        payload = command.get("payload") if isinstance(command.get("payload"), dict) else {}
        if action == "cancel":
            raise ResearchCancelled()

        # Reject approve and revise commands that were issued against a different plan version.
        if action in {"revise", "approve"}:
            try:
                command_version = int(command.get("expected_plan_version"))
            except (TypeError, ValueError):
                command_version = -1
            if command_version != plan_version:
                events.emit(
                    "command_rejected",
                    phase="approval" if iteration == 0 else "research",
                    iteration=iteration,
                    plan_version=plan_version,
                    data={
                        "action": action,
                        "reason": "stale_plan_version",
                        "expected_plan_version": command_version,
                    },
                )
                continue

        # A command with plan text revises the plan: renormalize it, bump the version, announce
        # the change and persist it.
        if action in {"revise", "approve"} and str(payload.get("plan") or "").strip():
            plan, checklist = _normalize_user_plan(
                payload.get("plan"),
                topic,
                previous_checklist=checklist,
            )
            plan_version += 1
            events.emit(
                "plan_updated",
                phase="approval" if iteration == 0 else "research",
                iteration=iteration,
                plan_version=plan_version,
                data={"plan": plan, "checklist": checklist},
            )
            control.update_state(
                events.session_id,
                plan=plan,
                checklist=checklist,
                plan_version=plan_version,
                latest_action="Research plan updated",
                last_sequence=events.sequence,
                event_tail=list(events.events[-80:]),
            )
        if action == "approve":
            approved = True
    return plan, checklist, plan_version, approved


# Publish the plan for approval and wait for the user's approve command, applying plan revisions
# meanwhile. With auto_approve the plan is approved at once. Return the approved plan, checklist
# and plan version; raise TimeoutError when no approval arrives within timeout_s.
def _wait_for_approval(
    *,
    session_id: str,
    topic: str,
    plan: str,
    checklist: list[dict[str, Any]],
    plan_version: int,
    inbox: CommandInbox,
    events: SemanticEventStream,
    timeout_s: float,
    auto_approve: bool,
) -> tuple[str, list[dict[str, Any]], int]:
    # Ask for approval: emit the event and persist the awaiting_approval state with the plan
    # controls enabled.
    events.emit(
        "approval_required",
        phase="approval",
        plan_version=plan_version,
        data={"plan": plan, "checklist": checklist},
    )
    control.update_state(
        session_id,
        status="awaiting_approval",
        phase="approval",
        plan=plan,
        checklist=checklist,
        plan_version=plan_version,
        latest_action="Waiting for plan approval",
        can_approve=True,
        can_edit=True,
        can_stop=True,
        last_sequence=events.sequence,
        event_tail=list(events.events[-80:]),
    )

    # Automatic approval skips the wait.
    if auto_approve:
        events.emit(
            "approval_granted",
            phase="approval",
            plan_version=plan_version,
            data={"automatic": True},
        )
        return plan, checklist, plan_version

    # Poll the inbox until the plan is approved (a cancel command raises) or the deadline passes.
    deadline = time.monotonic() + max(1.0, float(timeout_s or 900.0))
    while time.monotonic() < deadline:
        plan, checklist, plan_version, approved = _handle_commands(
            inbox,
            topic=topic,
            plan=plan,
            checklist=checklist,
            plan_version=plan_version,
            events=events,
            iteration=0,
        )
        if approved:
            events.emit(
                "approval_granted",
                phase="approval",
                plan_version=plan_version,
                data={"automatic": False},
            )
            return plan, checklist, plan_version
        time.sleep(0.15)
    raise TimeoutError("Deep research plan approval expired.")


# Sources and citations

# Normalize a tool result to a dict with model_context and sources: use its structured form when
# present, parse JSON text, and wrap anything else as plain model context.
def _structured_result(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        structured = result.get("_tool_result_structured")
        if isinstance(structured, dict):
            return structured
        if isinstance(result.get("model_context"), str):
            return result
    if isinstance(result, str) and result.lstrip().startswith("{"):
        try:
            decoded = json.loads(result)
        except json.JSONDecodeError:
            return {"model_context": result, "sources": []}
        return decoded if isinstance(decoded, dict) else {"model_context": result, "sources": []}
    return {"model_context": str(result or ""), "sources": []}


# Return a canonical form of an http(s) URL for deduplication: lower-cased scheme, case-folded host
# without a leading www., no trailing slash on the path and no fragment. Return an empty string
# for anything else.
def _canonical_source_url(value: Any) -> str:
    raw = str(value or "").strip()
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return ""
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    path = parsed.path.rstrip("/") or "/"
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.casefold().removeprefix("www."), path, parsed.query, "")
    )


# Return the identity of a source: its canonical URL, else its id.
def _source_key(source: Mapping[str, Any]) -> str:
    return _canonical_source_url(source.get("url")) or str(source.get("id") or "").strip()


# Citation handle pattern (such as c0001-2) and its compiled word-bounded form.
_CITATION_HANDLE = r"c[a-z0-9]{3,}-\d+"
_CITATION_HANDLE_RE = re.compile(rf"\b({_CITATION_HANDLE})\b", flags=re.I)


# Assigns durable, session-scoped handles to otherwise process-local search IDs. The web-search
# worker intentionally uses a cheap process-local counter. A restarted worker can therefore return
# c0000-1 again for a completely different URL. Deep Research must outlive that worker, so it
# translates every tool response into one monotonically increasing namespace derived from the
# durable research session.
class SessionCitationRegistry:
    # Derive the handle namespace from the session key and continue numbering at next_ordinal (at
    # least 1).
    def __init__(self, session_id: str, *, next_ordinal: Any = 1) -> None:
        self.namespace = control.session_key(session_id)[:8]
        try:
            self.next_ordinal = max(1, int(next_ordinal or 1))
        except (TypeError, ValueError):
            self.next_ordinal = 1

        # Durable handle per source identity key, and every handle already in use.
        self._handles_by_key: dict[str, str] = {}
        self._used_handles: set[str] = set()

    # Record the handle of a source key and, when the handle belongs to this session's namespace,
    # move the ordinal counter past it.
    def _remember(self, key: str, handle: str) -> None:
        normalized_handle = str(handle or "").strip().lower()
        if key:
            self._handles_by_key[key] = normalized_handle
        if normalized_handle:
            self._used_handles.add(normalized_handle)
        prefix = f"c{self.namespace}-"
        if normalized_handle.startswith(prefix):
            try:
                ordinal = int(normalized_handle[len(prefix):])
            except (TypeError, ValueError):
                # The suffix is not a number, so the counter is left alone.
                return
            self.next_ordinal = max(self.next_ordinal, ordinal + 1)

    # Allocate the next unused handle of the session namespace for a source key.
    def _allocate(self, key: str) -> str:
        while True:
            handle = f"c{self.namespace}-{self.next_ordinal}"
            self.next_ordinal += 1
            if handle.lower() not in self._used_handles:
                self._remember(key, handle)
                return handle

    # Rewrite the raw sources of one tool response to durable handles. Return the canonical sources
    # and a response-local map from every raw handle or alias (case-folded) to the durable handle.
    # A source keeps a handle assigned earlier, else keeps its own handle when it already belongs to
    # this namespace and is free, else gets a newly allocated one.
    def normalize_sources(
        self,
        raw_sources: Any,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        normalized_sources: list[dict[str, Any]] = []
        handle_map: dict[str, str] = {}
        for raw_source in raw_sources if isinstance(raw_sources, list) else []:
            if not isinstance(raw_source, Mapping):
                continue

            # Copy the source and collect its raw handle and aliases.
            source = dict(raw_source)
            raw_handle = str(source.get("id") or "").strip()
            raw_aliases = [
                str(alias or "").strip()
                for alias in source.get("citation_aliases") or []
                if str(alias or "").strip()
            ]

            # Identity key of the source: the canonical URL, else a combination of handle, title and URL.
            key = _canonical_source_url(source.get("url"))
            if not key:
                # ID is only a last-resort identity for non-URL records.  Include the
                # title so two restarted workers reusing c0000-1 do not collapse them.
                key = "|".join(
                    part
                    for part in (
                        raw_handle.casefold(),
                        str(source.get("title") or "").strip().casefold(),
                        str(source.get("url") or "").strip(),
                    )
                    if part
                )
            if not key:
                continue

            # Choose the durable handle for this source.
            stable_handle = self._handles_by_key.get(key)
            if not stable_handle:
                candidate = raw_handle.lower()
                own_prefix = f"c{self.namespace}-"
                if candidate.startswith(own_prefix) and candidate not in self._used_handles:
                    stable_handle = candidate
                    self._remember(key, stable_handle)
                else:
                    stable_handle = self._allocate(key)

            # Keep every other raw handle as an alias and map all of them to the durable handle.
            aliases: list[str] = []
            for alias in (raw_handle, *raw_aliases):
                if alias and alias.casefold() != stable_handle.casefold() and alias not in aliases:
                    aliases.append(alias)
                if alias:
                    handle_map[alias.casefold()] = stable_handle
            handle_map[stable_handle.casefold()] = stable_handle

            # Store the durable handle, the aliases and the canonical URL on the source.
            source["id"] = stable_handle
            if aliases:
                source["citation_aliases"] = aliases
            else:
                source.pop("citation_aliases", None)
            canonical_url = _canonical_source_url(source.get("url"))
            if canonical_url:
                source["canonical_url"] = canonical_url
            normalized_sources.append(source)
        return normalized_sources, handle_map


# Translate the handles in one tool response with the raw-to-durable map before the text is added to
# the model evidence. Handles missing from the map stay unchanged.
def _remap_citation_handles(text: str, handle_map: Mapping[str, str]) -> str:
    normalized_map = {
        str(key or "").strip().casefold(): str(value or "").strip()
        for key, value in handle_map.items()
        if str(key or "").strip() and str(value or "").strip()
    }
    if not normalized_map:
        return str(text or "")

    # Replace one matched handle by its durable counterpart when it is known.
    def replace(match: re.Match[str]) -> str:
        raw_handle = match.group(1)
        return normalized_map.get(raw_handle.casefold(), raw_handle)

    return _CITATION_HANDLE_RE.sub(replace, str(text or ""))


# Merge raw sources into target, keyed by source identity, and return how many were new. A source
# that is already present keeps its entry and gains the incoming handles as aliases and any field it
# was missing; a new source without an id gets a cpage-N id.
def _merge_sources(target: dict[str, dict[str, Any]], raw_sources: Any) -> int:
    added = 0
    for raw_source in raw_sources if isinstance(raw_sources, list) else []:
        if not isinstance(raw_source, dict):
            continue
        source = dict(raw_source)
        key = _source_key(source)
        if not key:
            continue

        # Known source: record the incoming handles as aliases and fill in missing fields.
        if key in target:
            existing = target[key]
            aliases = existing.get("citation_aliases")
            if not isinstance(aliases, list):
                aliases = []
            incoming_handles = [source.get("id"), *(source.get("citation_aliases") or [])]
            for raw_handle in incoming_handles:
                handle = str(raw_handle or "").strip()
                if handle and handle != str(existing.get("id") or "").strip() and handle not in aliases:
                    aliases.append(handle)
            if aliases:
                existing["citation_aliases"] = aliases
            for field, value in source.items():
                if value not in (None, "", [], {}) and existing.get(field) in (None, "", [], {}):
                    existing[field] = value
            continue

        # New source: make sure it has an id before storing it.
        if not str(source.get("id") or "").strip():
            source["id"] = f"cpage-{len(target) + 1}"
        target[key] = source
        added += 1
    return added


# Build the list that maps the citation handle of each read URL to the URL, for the full-page
# evidence. Return an empty string when no URL has a handle.
def _read_citation_map(
    urls: list[str],
    sources_by_key: Mapping[str, Mapping[str, Any]],
) -> str:
    lines: list[str] = []
    for url in urls:
        source = sources_by_key.get(_canonical_source_url(url)) or {}
        handle = str(source.get("id") or "").strip()
        if handle:
            lines.append(f"- [{handle}] {url}")
    if not lines:
        return ""
    return "Citation mapping for the full pages below:\n" + "\n".join(lines)


# Search calls

# Build one simple-shape web_search call per vertical. The web-search contract accepts
# {"query", "vertical", "effort", "description"} where query is one plain string or a short list of
# strings, and search operators (site:, quoted phrases, -term) stay inline in the text. A call
# carries a single vertical, so a mixed web/academic/shopping batch splits into several calls.
def _tool_search_calls(queries: list[dict[str, str]], iteration: int) -> list[dict[str, Any]]:
    # The description is the first stated purpose, shortened to 76 characters.
    description = next(
        (str(query.get("purpose") or "").strip() for query in queries if query.get("purpose")),
        f"Investigating evidence gap {iteration}",
    )
    description = re.sub(r"\s+", " ", description).strip()[:76] or "Investigating evidence gap"

    # Group the query texts by vertical, keeping at most MAX_QUERIES_PER_ITERATION queries.
    grouped: dict[str, list[str]] = {}
    for query in queries[:MAX_QUERIES_PER_ITERATION]:
        text = _normalize_query_text(query.get("text"))
        if text:
            grouped.setdefault(_safe_vertical(query.get("vertical")), []).append(text)
    return [
        {
            "query": texts if len(texts) > 1 else texts[0],
            "vertical": vertical,
            "effort": "medium",
            "description": description,
        }
        for vertical, texts in grouped.items()
    ]


# Return the public preview of a source: only its identity, title, URL, domain and rank fields that
# have a value.
def _safe_source_preview(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: source.get(key)
        for key in (
            "id",
            "citation_aliases",
            "title",
            "url",
            "canonical_url",
            "domain",
            "display_domain",
            "rank",
        )
        if source.get(key) not in (None, "")
    }


# Pick up to limit unseen http(s) URLs to read in full and add them to seen_urls. A second URL of an
# already picked domain is skipped unless it is needed to fill the limit.
def _select_read_urls(sources: list[dict[str, Any]], seen_urls: set[str], limit: int = 2) -> list[str]:
    output: list[str] = []
    seen_domains: set[str] = set()
    for source in sources:
        url = str(source.get("url") or "").strip()
        try:
            parsed = urlsplit(url)
        except ValueError:
            # A malformed URL cannot be read, so it is not a candidate.
            continue
        if parsed.scheme not in {"http", "https"} or not parsed.netloc or url in seen_urls:
            continue
        domain = parsed.netloc.casefold().removeprefix("www.")
        if domain in seen_domains and len(output) + 1 < limit:
            continue
        seen_domains.add(domain)
        seen_urls.add(url)
        output.append(url)
        if len(output) >= limit:
            break
    return output


# Evidence and report helpers

# Build the evidence text for the model from the newest entries backwards, within max_chars (at
# least 4,000). Each entry is cut to MODEL_ENTRY_CHARS and preceded by an iteration header, and the
# result is returned in chronological order.
def _evidence_packet(entries: list[dict[str, Any]], max_chars: int = MODEL_EVIDENCE_CHARS) -> str:
    pieces: list[str] = []
    remaining = max(4_000, int(max_chars or MODEL_EVIDENCE_CHARS))
    for entry in reversed(entries):
        text = str(entry.get("model_context") or "").strip()
        if not text:
            continue
        header = f"\n## Iteration {entry.get('iteration', '?')} {entry.get('kind', 'evidence')}\n"
        allowance = min(MODEL_ENTRY_CHARS, max(0, remaining - len(header)))
        if allowance <= 0:
            break
        fragment = header + text[:allowance]
        pieces.append(fragment)
        remaining -= len(fragment)
        if remaining <= 0:
            break
    return "".join(reversed(pieces)).strip()


# Convert the sources that have a handle and an http(s) URL into citation records: id, title (up to
# 180 characters), URL, canonical URL and aliases.
def _citation_records(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for source in sources:
        source_id = str(source.get("id") or "").strip().lower()
        url = str(source.get("url") or "").strip()
        try:
            parsed = urlsplit(url)
        except ValueError:
            # A malformed URL cannot be cited, so the source gets no citation record.
            continue

        # Build the record: the title falls back to the display domain, the domain, then the host.
        if source_id and parsed.scheme in {"http", "https"} and parsed.netloc:
            title = re.sub(
                r"\s+",
                " ",
                str(
                    source.get("title")
                    or source.get("display_domain")
                    or source.get("domain")
                    or parsed.netloc
                ),
            ).strip()[:180]
            records.append(
                {
                    "id": str(source.get("id") or "").strip(),
                    "title": title or parsed.netloc,
                    "url": url,
                    "canonical_url": _canonical_source_url(url),
                    "aliases": [
                        str(alias or "").strip()
                        for alias in source.get("citation_aliases") or []
                        if str(alias or "").strip()
                    ],
                }
            )
    return records


# Build a Markdown link with a sanitized label and a URL whose spaces and parentheses are escaped.
def _markdown_link(label: str, url: str) -> str:
    safe_label = re.sub(r"[\[\]\r\n]+", " ", str(label or "Source"))
    safe_label = re.sub(r"\s+", " ", safe_label).strip()[:180] or "Source"
    safe_url = str(url or "").strip().replace(" ", "%20").replace("(", "%28").replace(")", "%29")
    return f"[{safe_label}]({safe_url})"


# Replace bracketed citation handles, single or comma-separated, with Markdown links to their
# sources. Handles without a known source stay as plain [handle].
def _citation_links(text: str, sources: list[dict[str, Any]]) -> str:
    # Index the records by handle, then by alias.
    citation_index: dict[str, dict[str, Any]] = {}
    for record in _citation_records(sources):
        source_id = str(record.get("id") or "").strip().casefold()
        if source_id:
            citation_index[source_id] = record

        # An old raw handle can legitimately be reused after a web-search worker
        # restart.  Keep the first mapping only; durable report prompts see the stable
        # IDs, while aliases exist solely to resolve pre-migration/checkpoint text.
        for alias in record.get("aliases") or []:
            normalized_alias = str(alias or "").strip().casefold()
            if normalized_alias:
                citation_index.setdefault(normalized_alias, record)

    # Match bracketed handle groups that are not already followed by a link target.
    pattern = re.compile(
        rf"\[\s*({_CITATION_HANDLE}(?:\s*,\s*{_CITATION_HANDLE})*)\s*\](?!\()",
        flags=re.I,
    )

    # Turn one matched handle group into one link per handle.
    def replace(match: re.Match[str]) -> str:
        links: list[str] = []
        for item in (part.strip() for part in match.group(1).split(",")):
            record = citation_index.get(item.casefold())
            links.append(
                _markdown_link(str(record.get("title") or item), str(record.get("url") or ""))
                if record
                else f"[{item}]"
            )
        return " ".join(links)

    return pattern.sub(replace, str(text or ""))


# Build the Markdown report used when the model produced none (stopped or failed research): a
# title, the topic, the latest assessment, up to 8 remaining gaps and up to 24 collected sources
# with their citation handles.
def _fallback_report(
    *,
    topic: str,
    assessment: str,
    gaps: list[str],
    sources: list[dict[str, Any]],
    cancelled: bool = False,
) -> str:
    title = "Research stopped by the user" if cancelled else "Partial research result"
    parts = [f"## {title}", "", f"Topic: {topic}"]
    if assessment:
        parts.extend(["", assessment])
    if gaps:
        parts.extend(["", "### Remaining evidence gaps", *[f"- {gap}" for gap in gaps[:8]]])
    if sources:
        parts.extend(["", "### Sources collected"])
        for source in sources[:24]:
            label = str(source.get("title") or source.get("display_domain") or source.get("url") or "Source")
            handle = str(source.get("id") or "").strip()
            parts.append(f"- {label} [{handle}]" if handle else f"- {label}")
    return "\n".join(parts).strip()


# Build the payload returned to the caller: the report with resolved citation links as model
# context and report, the plan, checklist, sources, citations and event log summary, and the final
# UI state with every control disabled because the run is over.
def _result_payload(
    *,
    session_id: str,
    status: str,
    topic: str,
    plan: str,
    plan_version: int,
    checklist: list[dict[str, Any]],
    report: str,
    sources: list[dict[str, Any]],
    engine: str,
    model: str,
    queries_used: int,
    query_budget: int,
    iteration: int,
    events: SemanticEventStream,
) -> dict[str, Any]:
    # Resolve citation handles in the report to links.
    final_status = str(status or "partial")
    resolved_report = _citation_links(report, sources)
    citations = _citation_records(sources)

    # State shown by the UI.
    ui = {
        "kind": "deep_research",
        "status": final_status,
        "topic": topic,
        "session_id": session_id,
        "plan": plan,
        "plan_version": plan_version,
        "checklist": checklist,
        "result_count": len(sources),
        "source_count": len(sources),
        "sources": sources,
        "citations": citations,
        "engine": engine,
        "model": model,
        "queries_used": queries_used,
        "query_budget": query_budget,
        "iteration": iteration,
        "can_approve": False,
        "can_edit": False,
        "can_stop": False,
        "events": list(events.events),
        "event_log": {
            "schema_version": events.schema_version,
            "path": str(events.path),
            "event_count": events.sequence,
        },
    }

    # Result of the tool call; the UI state travels under "ui".
    return {
        "model_context": resolved_report,
        "report": resolved_report,
        "plan": plan,
        "checklist": checklist,
        "sources": sources,
        "citations": citations,
        "event_log": ui["event_log"],
        "ui": ui,
    }


# Run orchestration

# Phases in which the query slots of an iteration are already charged and checkpointed, so the
# iteration may have reached an external service.
_COUNTED_ITERATION_PHASES = frozenset(
    {
        "search",
        "search_completed",
        "reading",
        "reading_completed",
        "checkpoint",
    }
)


# Choose the first safe iteration after loading a durable snapshot. Query slots are charged and
# checkpointed before the external search starts. Consequently, every phase from `search` through
# `checkpoint` represents a batch that may already have reached an external service. Replaying that
# same iteration would spend a second batch (up to two more queries) under the same iteration
# number, so recovery conservatively advances even when the worker died while the external call
# was still in flight. A snapshot that was not approved starts at iteration 1 and one in final
# reflection or synthesis skips the iteration loop; any other phase repeats the recovered iteration.
def _resume_start_iteration(
    *,
    resumed_approved: bool,
    recovered_phase: str,
    iteration: int,
    max_iterations: int,
) -> int:
    if not resumed_approved:
        return 1
    normalized_phase = str(recovered_phase or "").strip().lower()
    recovered_iteration = max(0, int(iteration or 0))
    if normalized_phase in _COUNTED_ITERATION_PHASES:
        return max(1, recovered_iteration + 1)
    if normalized_phase in {"final_reflection", "synthesis"}:
        return max_iterations + 1
    return max(1, recovered_iteration)


# Run the deterministic research state machine for one session and return the result payload. A
# session recovered from a durable snapshot is resumed (or, when finished, returned as it is).
# Otherwise the model drafts a plan that the user approves, then each iteration reflects on the
# evidence, selects up to two queries, searches and reads the best pages until the evidence is
# complete, the query budget is spent or no new evidence appears. A final audit and the report
# stage close the run. Every step is checkpointed to the control store, and cancellation, model
# failures, expired approvals and unexpected errors end in a stored partial, failed or cancelled
# result instead of an exception.
def run_deep_research_v2(
    arguments: Mapping[str, Any],
    context: Mapping[str, Any] | None,
    *,
    runtime: Mapping[str, Any],
    generation_options: Mapping[str, Any],
    logs_dir: Path,
) -> dict[str, Any]:
    # Validate the arguments. The round count is clamped to 2-25 (6 when invalid) and the query
    # budget is two queries per round.
    topic = str(arguments.get("topic") or "").strip()
    if not topic:
        raise ValueError("Deep research requires a non-empty topic.")
    instructions = str(arguments.get("instructions") or "").strip()
    try:
        max_iterations = min(25, max(2, int(arguments.get("max_rounds") or 6)))
    except (TypeError, ValueError):
        max_iterations = 6
    query_budget = max_iterations * MAX_QUERIES_PER_ITERATION

    # The selected model and the user's request text, with the extra instructions appended.
    engine = str(runtime["engine"])
    model = str(runtime["model"])
    caller = context if isinstance(context, Mapping) else {}
    request_message = caller.get("request_message") if isinstance(caller.get("request_message"), Mapping) else {}
    request_content = str(request_message.get("content") or "").strip()
    request_text = request_content or topic
    if instructions:
        request_text = f"{request_text}\n\nAdditional instructions:\n{instructions}"

    # How long to wait for plan approval, in seconds (default 900).
    try:
        approval_timeout_s = float(
            arguments.get("approval_timeout_s") or caller.get("approval_timeout_s") or 900
        )
    except (TypeError, ValueError):
        approval_timeout_s = 900.0

    # Evidence budget in characters: about 24% of the model's context window at four characters per
    # token, between 12,000 and MODEL_EVIDENCE_CHARS; the maximum when the window is unknown.
    runtime_limits = runtime.get("limits") if isinstance(runtime.get("limits"), Mapping) else {}
    try:
        runtime_context_tokens = int(
            runtime_limits.get("context_window")
            or runtime_limits.get("model_context_limit")
            or 0
        )
    except (TypeError, ValueError):
        runtime_context_tokens = 0
    evidence_char_budget = min(
        MODEL_EVIDENCE_CHARS,
        max(12_000, int(runtime_context_tokens * 4 * 0.24))
        if runtime_context_tokens
        else MODEL_EVIDENCE_CHARS,
    )

    # Create or load the durable session. The id is validated first (an unsafe id raises), and a
    # recovered snapshot lets an interrupted run resume.
    session_id = str(arguments.get("session_id") or "").strip() or control.new_session_id()
    control.session_key(session_id)
    recovered_state = control.create_session(
        session_id,
        topic=topic,
        extra={"query_budget": query_budget, "engine": engine, "model": model},
    )

    # Event stream and command inbox. Commands acknowledged before an interruption are not replayed.
    events = SemanticEventStream(
        session_id,
        callback=caller.get("event_callback"),
        logs_dir=logs_dir,
    )
    inbox = CommandInbox(session_id)
    inbox.processed.update(
        str(filename)
        for filename in recovered_state.get("processed_commands") or []
        if str(filename or "").strip()
    )

    # Recover the plan, plan version and checklist from the snapshot.
    recovered_checklist = _normalize_checklist(recovered_state.get("checklist"))
    plan = str(recovered_state.get("plan") or "")
    try:
        plan_version = max(0, int(recovered_state.get("plan_version") or 0))
    except (TypeError, ValueError):
        plan_version = 0

    # Read a non-negative integer field of the recovered snapshot, using the default when the field
    # is missing or invalid.
    def recovered_int(field: str, default: int = 0) -> int:
        try:
            return max(0, int(recovered_state.get(field) or default))
        except (TypeError, ValueError):
            return max(0, default)

    # Rebuild the evidence state: sources under durable citation handles, the last 64 evidence
    # entries with remapped handles, and the queries and URLs seen so far.
    checklist: list[dict[str, Any]] = recovered_checklist
    sources_by_key: dict[str, dict[str, Any]] = {}
    citation_registry = SessionCitationRegistry(
        session_id,
        next_ordinal=recovered_state.get("citation_next"),
    )
    recovered_sources, recovered_handle_map = citation_registry.normalize_sources(
        recovered_state.get("sources")
    )
    _merge_sources(sources_by_key, recovered_sources)
    evidence_entries = [
        {
            **dict(entry),
            "model_context": _remap_citation_handles(
                str(entry.get("model_context") or "")[:MODEL_ENTRY_CHARS],
                recovered_handle_map,
            ),
        }
        for entry in recovered_state.get("evidence_entries") or []
        if isinstance(entry, Mapping) and str(entry.get("model_context") or "").strip()
    ][-64:]
    seen_queries = {
        str(query or "").strip()
        for query in recovered_state.get("seen_queries") or []
        if str(query or "").strip()
    }
    seen_urls = {
        str(url or "").strip()
        for url in recovered_state.get("seen_urls") or []
        if str(url or "").strip()
    }

    # Progress state recovered from the snapshot: counters, the latest assessment and gaps, the
    # stored status, report and candidate queries.
    queries_used = recovered_int("queries_used")
    iteration = recovered_int("iteration")
    latest_assessment = str(recovered_state.get("latest_assessment") or "")[:1600]
    latest_gaps = [
        str(gap or "").strip()[:300]
        for gap in recovered_state.get("latest_gaps") or []
        if str(gap or "").strip()
    ][:8]
    status = str(recovered_state.get("status") or "partial")
    report = str(recovered_state.get("report") or "")
    no_new_evidence_rounds = recovered_int("no_new_evidence_rounds")
    late_revision_without_search = bool(recovered_state.get("late_revision_without_search"))
    initial_candidates = _normalize_candidates(recovered_state.get("initial_candidates"))
    recovered_phase = str(recovered_state.get("phase") or "").strip().lower()
    has_recovered_plan = bool(plan.strip() and checklist and plan_version > 0)
    query_budget = recovered_int("query_budget", query_budget) or query_budget

    # Context for the search tools: the session acts as the chat and only the search aliases are
    # allowed.
    tool_context = {
        "chat_id": session_id,
        "engine": engine,
        "model_name": model,
        "module_dir": str(Path(__file__).resolve().parents[2]),
        "project_dir": str(Path(__file__).resolve().parents[2]),
        "allowed_tool_aliases": list(ALLOWED_SEARCH_ALIASES),
        "deep_research": True,
    }

    # A finished session returns its stored result again instead of running anew.
    if control.is_terminal_status(status):
        terminal_report = report or str(
            recovered_state.get("model_context")
            or recovered_state.get("error")
            or recovered_state.get("latest_action")
            or "Deep Research ended without a report."
        )
        terminal_sources = list(sources_by_key.values())
        resolved_terminal_report = _citation_links(terminal_report, terminal_sources)

        # Legacy terminal snapshots can predate session-scoped citations.  Returning a
        # migrated payload is not enough because the reopened UI reads this file directly;
        # atomically upgrade the durable snapshot while preserving its semantic status/phase.
        try:
            control.update_state(
                session_id,
                report=resolved_terminal_report,
                model_context=resolved_terminal_report,
                sources=[_safe_source_preview(source) for source in terminal_sources],
                citations=_citation_records(terminal_sources),
                citation_next=citation_registry.next_ordinal,
                source_count=len(terminal_sources),
            )
            return _result_payload(
                session_id=session_id,
                status=status,
                topic=topic,
                plan=plan,
                plan_version=plan_version,
                checklist=checklist,
                report=resolved_terminal_report,
                sources=terminal_sources,
                engine=str(recovered_state.get("engine") or engine),
                model=str(recovered_state.get("model") or model),
                queries_used=queries_used,
                query_budget=query_budget,
                iteration=iteration,
                events=events,
            )
        finally:
            events.close()

    # Resolve the search tools. A failure is recorded and reported later instead of aborting the run.
    registry_error = ""
    try:
        _tool_specs, tool_lookup = tool_registry.build_ollama_tools(
            ["web_search"],
            allowed_tool_aliases=list(ALLOWED_SEARCH_ALIASES),
        )
    except Exception as exc:
        # Planning and synthesis can still produce a useful partial result when
        # the search backend is temporarily unavailable.  Keep this failure local
        # instead of dropping the complete MCP call before the approval card opens.
        tool_lookup = {}
        registry_error = f"{type(exc).__name__}: {exc}"[:400]

    # Persist the current run state (plan, counters, evidence, sources, seen queries and URLs,
    # processed commands) together with the given changes. Citation handles in a report or model
    # context are resolved to links first.
    def checkpoint(**changes: Any) -> None:
        # Store public source previews and citation records instead of the full sources.
        durable_sources = [
            _safe_source_preview(source)
            for source in sources_by_key.values()
        ]
        durable_changes = dict(changes)

        # Resolve citation handles in the report and model context to links.
        if "report" in durable_changes:
            durable_changes["report"] = _citation_links(
                str(durable_changes.get("report") or ""),
                durable_sources,
            )
        if "model_context" in durable_changes:
            durable_changes["model_context"] = _citation_links(
                str(durable_changes.get("model_context") or ""),
                durable_sources,
            )
        durable_changes["sources"] = durable_sources
        durable_changes["citations"] = _citation_records(durable_sources)

        # Write everything needed to resume the run together with the given changes.
        control.update_state(
            session_id,
            plan=plan,
            plan_version=plan_version,
            checklist=checklist,
            source_count=len(sources_by_key),
            queries_used=queries_used,
            query_budget=query_budget,
            iteration=iteration,
            current_item_id=_current_checklist_item_id(checklist),
            last_sequence=events.sequence,
            event_tail=list(events.events[-80:]),
            evidence_entries=list(evidence_entries[-64:]),
            seen_queries=sorted(seen_queries),
            seen_urls=sorted(seen_urls),
            latest_assessment=latest_assessment,
            latest_gaps=latest_gaps,
            no_new_evidence_rounds=no_new_evidence_rounds,
            late_revision_without_search=late_revision_without_search,
            citation_next=citation_registry.next_ordinal,
            initial_candidates=initial_candidates,
            processed_commands=sorted(inbox.processed),
            **durable_changes,
        )

    # Main run: announce the session (resumed when a plan was recovered or queries were already
    # used), then plan, wait for approval and iterate over reflection, search and reading.
    try:
        events.emit(
            "session_resumed" if has_recovered_plan or queries_used else "session_started",
            phase="session",
            iteration=iteration,
            plan_version=plan_version,
            data={
                "topic": topic,
                "engine": engine,
                "model": model,
                "max_iterations": max_iterations,
                "query_budget": query_budget,
                "queries_per_iteration": MAX_QUERIES_PER_ITERATION,
                "orchestrator": "deterministic-v2",
                "resumed": bool(has_recovered_plan or queries_used),
                "restored_source_count": len(sources_by_key),
                "restored_queries_used": queries_used,
            },
        )
        if registry_error:
            events.emit(
                "search_backend_unavailable",
                phase="session",
                data={"message": registry_error},
            )

        # A plan that was already approved (the run was running or synthesizing) is resumed
        # without asking for approval again.
        resumed_approved = (
            has_recovered_plan
            and status in {"running", "synthesizing"}
            and recovered_phase not in {"planning", "approval"}
        )

        # Planning stage: ask the model for a plan and first candidate queries. Skipped when a plan
        # was recovered from the snapshot.
        if not has_recovered_plan:
            checkpoint(status="planning", phase="planning", latest_action="Drafting the research plan")
            events.emit("planning_started", phase="planning")
            raw_plan = _run_model_stage(
                engine=engine,
                model=model,
                system_prompt=PLAN_PROMPT,
                user_prompt=request_text,
                generation_options=generation_options,
                session_id=session_id,
                events=events,
                phase="planning",
                iteration=0,
                plan_version=0,
                request_message=request_message,
            )

            # Parse the plan, keeping the finished progress of any recovered checklist and falling
            # back to a default checklist when the model gave no usable steps.
            parsed_plan = _extract_json_object(raw_plan)
            planned_checklist = _normalize_checklist(parsed_plan.get("steps"), raw_plan)
            checklist = _merge_checklist_progress(recovered_checklist, planned_checklist)
            if not checklist:
                checklist = _fallback_checklist(topic)
            summary = str(parsed_plan.get("summary") or "").strip()
            plan = _plan_text(summary, checklist)
            initial_candidates = _normalize_candidates(
                parsed_plan.get("candidates") or parsed_plan.get("queries")
            )

            # The first plan is version 1; a replan that changes the checklist titles bumps it.
            previous_titles = [
                _checklist_title_key(item.get("title"))
                for item in recovered_checklist
            ]
            planned_titles = [
                _checklist_title_key(item.get("title"))
                for item in checklist
            ]
            if plan_version <= 0:
                plan_version = 1
            elif previous_titles and previous_titles != planned_titles:
                plan_version += 1

            # Publish the plan and persist the state that waits for approval.
            events.emit(
                "plan_ready",
                phase="planning",
                plan_version=plan_version,
                data={
                    "plan": plan,
                    "checklist": checklist,
                    "candidate_count": len(initial_candidates),
                    "fallback_used": not bool(parsed_plan),
                },
            )
            checkpoint(
                status="awaiting_approval",
                phase="approval",
                latest_action="Waiting for plan approval",
            )

        # Approval: wait for the user (or auto approval), applying plan revisions meanwhile.
        if not resumed_approved:
            plan, checklist, plan_version = _wait_for_approval(
                session_id=session_id,
                topic=topic,
                plan=plan,
                checklist=checklist,
                plan_version=plan_version,
                inbox=inbox,
                events=events,
                timeout_s=approval_timeout_s,
                auto_approve=bool(caller.get("auto_approve") or arguments.get("auto_approve")),
            )
            if plan_version != 1:
                # The initial candidates were written for the first plan version only.
                initial_candidates = []

        # Research starts: persist the running state with approval off and editing and stopping on.
        checkpoint(
            status="running",
            phase="research",
            latest_action=(
                "Resuming approved research" if resumed_approved else "Beginning approved research"
            ),
            can_approve=False,
            can_edit=True,
            can_stop=True,
        )
        events.emit(
            "research_resumed" if resumed_approved else "research_started",
            phase="research",
            iteration=iteration,
            plan_version=plan_version,
            data={"plan": plan, "checklist": checklist},
        )

        # Iteration loop: every round reflects on the evidence, selects queries, searches and reads.
        # A resumed run continues after the last batch that may already have been sent.
        start_iteration = _resume_start_iteration(
            resumed_approved=resumed_approved,
            recovered_phase=recovered_phase,
            iteration=iteration,
            max_iterations=max_iterations,
        )

        for iteration in range(start_iteration, max_iterations + 1):
            # Apply user commands first. A plan revision that is not approved at once pauses the
            # run until the revised plan is approved.
            previous_plan_version = plan_version
            plan, checklist, plan_version, _approved = _handle_commands(
                inbox,
                topic=topic,
                plan=plan,
                checklist=checklist,
                plan_version=plan_version,
                events=events,
                iteration=iteration,
            )
            if plan_version != previous_plan_version and not _approved:
                initial_candidates = []
                plan, checklist, plan_version = _wait_for_approval(
                    session_id=session_id,
                    topic=topic,
                    plan=plan,
                    checklist=checklist,
                    plan_version=plan_version,
                    inbox=inbox,
                    events=events,
                    timeout_s=approval_timeout_s,
                    auto_approve=False,
                )
                checkpoint(
                    status="running",
                    phase="research",
                    latest_action="Continuing with the approved revised plan",
                    can_approve=False,
                    can_edit=True,
                    can_stop=True,
                )

            # Do not start another model stage once the user asked to stop.
            if _cancel_requested(session_id):
                raise ResearchCancelled()

            # Reflection stage: the model audits the evidence against the checklist and proposes
            # candidate queries.
            checkpoint(
                status="running",
                phase="reflection",
                latest_action=f"Reflecting before search iteration {iteration}",
            )
            events.emit(
                "reflection_started",
                phase="reflection",
                iteration=iteration,
                plan_version=plan_version,
                data={"queries_remaining": query_budget - queries_used},
            )
            evidence_packet = _evidence_packet(evidence_entries, evidence_char_budget)
            reflection_input = (
                f"Original request:\n{request_text}\n\nApproved plan:\n{plan}\n\n"
                f"Checklist JSON:\n{json.dumps(checklist, ensure_ascii=False)}\n\n"
                f"Queries already used:\n{json.dumps(sorted(seen_queries), ensure_ascii=False)}\n\n"
                f"Evidence gathered so far:\n{evidence_packet or '(none yet)'}"
            )
            raw_reflection = _run_model_stage(
                engine=engine,
                model=model,
                system_prompt=REFLECTION_PROMPT,
                user_prompt=reflection_input,
                generation_options=generation_options,
                session_id=session_id,
                events=events,
                phase="reflection",
                iteration=iteration,
                plan_version=plan_version,
                request_message=request_message,
            )

            # Apply the reflection: assessment, up to 8 gaps, checklist updates and candidate queries
            # (the plan's initial candidates in the first iteration when there are none). The
            # research counts as complete only when the model says so and a source exists.
            reflection = _extract_json_object(raw_reflection)
            latest_assessment = str(reflection.get("assessment") or "").strip()[:1600]
            latest_gaps = [
                str(gap).strip()[:300]
                for gap in reflection.get("gaps", []) if str(gap).strip()
            ][:8] if isinstance(reflection.get("gaps"), list) else []
            checklist = _apply_updates(checklist, reflection.get("updates"))
            candidates = _normalize_candidates(reflection.get("candidates"))
            if not candidates and iteration == 1:
                candidates = initial_candidates
            complete = _strict_bool(reflection.get("complete")) and bool(sources_by_key)
            events.emit(
                "reflection_completed",
                phase="reflection",
                iteration=iteration,
                plan_version=plan_version,
                data={
                    "summary": latest_assessment or "Evidence audit completed.",
                    "evidence_gaps": latest_gaps,
                    "checklist": checklist,
                    "current_item_id": _current_checklist_item_id(checklist),
                    "complete": complete,
                    "candidate_count": len(candidates),
                },
            )
            checkpoint(
                status="running",
                phase="reflection",
                latest_action=latest_assessment or "Evidence audit completed",
            )

            # Finish when the evidence is complete or the query budget is spent.
            if complete:
                break
            if queries_used >= query_budget:
                break

            # Query selection stage: the budget gate picks the one or two most valuable candidates.
            selection_input = (
                f"Unresolved gaps:\n{json.dumps(latest_gaps, ensure_ascii=False)}\n\n"
                f"Assessment:\n{latest_assessment}\n\n"
                f"Candidate queries:\n{json.dumps(candidates, ensure_ascii=False)}\n\n"
                f"Query history:\n{json.dumps(sorted(seen_queries), ensure_ascii=False)}\n\n"
                f"Slots remaining this iteration: {min(MAX_QUERIES_PER_ITERATION, query_budget - queries_used)}"
            )
            raw_selection = _run_model_stage(
                engine=engine,
                model=model,
                system_prompt=QUERY_SELECTION_PROMPT,
                user_prompt=selection_input,
                generation_options=generation_options,
                session_id=session_id,
                events=events,
                phase="query_selection",
                iteration=iteration,
                plan_version=plan_version,
                request_message=request_message,
            )

            # Use the first candidates when the gate selects none, then drop queries already used.
            parsed_selection = _extract_json_object(raw_selection)
            selected = _normalize_candidates(
                parsed_selection.get("queries") or parsed_selection.get("selected"),
                limit=min(MAX_QUERIES_PER_ITERATION, query_budget - queries_used),
            )
            if not selected:
                selected = candidates[: min(MAX_QUERIES_PER_ITERATION, query_budget - queries_used)]
            unique_selected: list[dict[str, str]] = []
            for query in selected:
                signature = _query_signature(query.get("text"))
                if signature and signature not in seen_queries:
                    seen_queries.add(signature)
                    unique_selected.append(query)
            if not unique_selected:
                # Nothing new to search; two such rounds in a row end the research.
                no_new_evidence_rounds += 1
                if no_new_evidence_rounds >= 2:
                    break
                continue

            # Charge the query slots and checkpoint before the external search starts, so a crash
            # cannot replay the same batch.
            events.emit(
                "queries_selected",
                phase="query_selection",
                iteration=iteration,
                plan_version=plan_version,
                data={"queries": unique_selected},
            )
            queries_used += len(unique_selected)
            checkpoint(
                status="running",
                phase="search",
                latest_action=f"Searching: {unique_selected[0]['text']}",
                active_queries=unique_selected,
            )
            events.emit(
                "search_started",
                phase="search",
                iteration=iteration,
                plan_version=plan_version,
                data={"queries": unique_selected},
            )

            # Search: one web_search call per vertical. A failing call becomes an error note in the
            # evidence instead of ending the session.
            search_sources: list[dict[str, Any]] = []
            search_context_parts: list[str] = []
            for search_args in _tool_search_calls(unique_selected, iteration):
                try:
                    search_result = tool_registry.call_ollama_tool(
                        tool_lookup,
                        "web_search__web_search",
                        search_args,
                        context=tool_context,
                    )
                    search_structured = _structured_result(search_result)
                except Exception as exc:
                    search_structured = {
                        "model_context": f"Search failed without ending the session: {type(exc).__name__}: {exc}",
                        "sources": [],
                    }

                # Citation handles restart at the same IDs in every worker call,
                # so each call is normalized to session handles before merging.
                call_sources, call_handle_map = citation_registry.normalize_sources(
                    [
                        dict(source)
                        for source in search_structured.get("sources", [])
                        if isinstance(source, dict)
                    ]
                )
                search_sources.extend(call_sources)
                call_context = _remap_citation_handles(
                    str(search_structured.get("model_context") or "").strip(),
                    call_handle_map,
                )
                if call_context:
                    search_context_parts.append(call_context)
                if _cancel_requested(session_id):
                    raise ResearchCancelled()

            # Merge the new sources and keep the search results as evidence.
            added_sources = _merge_sources(sources_by_key, search_sources)
            search_context = "\n\n".join(search_context_parts)
            if search_context:
                evidence_entries.append(
                    {
                        "iteration": iteration,
                        "kind": "search",
                        "queries": unique_selected,
                        "model_context": search_context[:MODEL_ENTRY_CHARS],
                    }
                )
            events.emit(
                "search_completed",
                phase="search",
                iteration=iteration,
                plan_version=plan_version,
                data={
                    "queries": unique_selected,
                    "new_source_count": added_sources,
                    "source_count": len(sources_by_key),
                    "sources": [
                        _safe_source_preview(source)
                        for source in search_sources[:MAX_PUBLIC_SEARCH_SOURCES]
                    ],
                },
            )

            # Persist immediately after the external tool returns.  A browser/LLM process
            # can die during the subsequent page read without losing IDs or evidence.
            checkpoint(
                status="running",
                phase="search_completed",
                latest_action=f"Search completed with {len(search_sources)} sources",
                active_queries=[],
            )

            # Reading: open the most promising new pages in full (at most two, preferably from
            # different domains).
            read_urls = _select_read_urls(search_sources, seen_urls, limit=2)
            if read_urls:
                checkpoint(
                    status="running",
                    phase="reading",
                    latest_action=f"Reading {len(read_urls)} promising source pages",
                )
                events.emit(
                    "reading_started",
                    phase="reading",
                    iteration=iteration,
                    plan_version=plan_version,
                    data={"tool_id": "read_page", "urls": read_urls},
                )
                try:
                    read_result = tool_registry.call_ollama_tool(
                        tool_lookup,
                        "web_search__read_page",
                        {"url": read_urls},
                        context=tool_context,
                    )
                    read_structured = _structured_result(read_result)
                except Exception as exc:
                    read_structured = {
                        "model_context": f"Page reading failed without ending the session: {type(exc).__name__}: {exc}",
                        "sources": [],
                    }
                if _cancel_requested(session_id):
                    raise ResearchCancelled()

                # Normalize the page sources to session handles, merge them and keep the page text
                # as evidence together with a citation map.
                raw_read_sources = [
                    dict(source)
                    for source in read_structured.get("sources", [])
                    if isinstance(source, dict)
                ]
                read_sources, read_handle_map = citation_registry.normalize_sources(raw_read_sources)
                added_sources += _merge_sources(sources_by_key, read_sources)
                read_context = _remap_citation_handles(
                    str(read_structured.get("model_context") or "").strip(),
                    read_handle_map,
                )
                if read_context:
                    citation_map = _read_citation_map(read_urls, sources_by_key)
                    if citation_map:
                        read_context = f"{citation_map}\n\n{read_context}"
                    evidence_entries.append(
                        {
                            "iteration": iteration,
                            "kind": "page reading",
                            "urls": read_urls,
                            "model_context": read_context[:MODEL_ENTRY_CHARS],
                        }
                    )
                events.emit(
                    "reading_completed",
                    phase="reading",
                    iteration=iteration,
                    plan_version=plan_version,
                    data={
                        "tool_id": "read_page",
                        "urls": read_urls,
                        "url_count": len(read_urls),
                        "new_source_count": len(read_sources),
                    },
                )
                checkpoint(
                    status="running",
                    phase="reading_completed",
                    latest_action=f"Read {len(read_urls)} source pages",
                )

            # Count consecutive rounds without new sources; two in a row end the research.
            if added_sources <= 0:
                no_new_evidence_rounds += 1
            else:
                no_new_evidence_rounds = 0
            events.emit(
                "checkpoint_saved",
                phase="checkpoint",
                iteration=iteration,
                plan_version=plan_version,
                data={
                    "checklist": checklist,
                    "current_item_id": _current_checklist_item_id(checklist),
                    "source_count": len(sources_by_key),
                    "queries_used": queries_used,
                    "query_budget": query_budget,
                    "no_new_evidence_rounds": no_new_evidence_rounds,
                },
            )
            checkpoint(
                status="running",
                phase="checkpoint",
                latest_action=f"Checkpoint {iteration} saved",
            )
            if no_new_evidence_rounds >= 2:
                break

        # Apply commands that arrived during the last iteration. A plan revision that is not
        # approved at once is a late revision: it is approved before the final audit, but it was
        # never searched, so the result can only be partial.
        previous_plan_version = plan_version
        plan, checklist, plan_version, _approved = _handle_commands(
            inbox,
            topic=topic,
            plan=plan,
            checklist=checklist,
            plan_version=plan_version,
            events=events,
            iteration=iteration,
        )
        if plan_version != previous_plan_version and not _approved:
            initial_candidates = []
            plan, checklist, plan_version = _wait_for_approval(
                session_id=session_id,
                topic=topic,
                plan=plan,
                checklist=checklist,
                plan_version=plan_version,
                inbox=inbox,
                events=events,
                timeout_s=approval_timeout_s,
                auto_approve=False,
            )
            late_revision_without_search = True
        if _cancel_requested(session_id):
            raise ResearchCancelled()

        # Final reflection: the model re-audits every checklist item against all evidence and must
        # not request another search.
        checkpoint(
            status="running",
            phase="final_reflection",
            latest_action="Performing the final evidence audit",
            can_approve=False,
            can_edit=False,
            can_stop=True,
        )
        events.emit(
            "final_reflection_started",
            phase="final_reflection",
            iteration=iteration,
            plan_version=plan_version,
            data={"source_count": len(sources_by_key), "queries_used": queries_used},
        )
        final_reflection_input = (
            "This is the mandatory final audit after the last search. Do not design or request "
            "another search. Re-evaluate every checklist item against all collected evidence and "
            "identify any remaining unsupported claim.\n\n"
            f"Original request:\n{request_text}\n\nApproved plan:\n{plan}\n\n"
            f"Checklist JSON:\n{json.dumps(checklist, ensure_ascii=False)}\n\n"
            f"Evidence gathered:\n{_evidence_packet(evidence_entries, evidence_char_budget) or '(none)'}"
        )
        raw_final_reflection = _run_model_stage(
            engine=engine,
            model=model,
            system_prompt=REFLECTION_PROMPT,
            user_prompt=final_reflection_input,
            generation_options=generation_options,
            session_id=session_id,
            events=events,
            phase="final_reflection",
            iteration=iteration,
            plan_version=plan_version,
            request_message=request_message,
        )

        # Keep the previous assessment and gaps when the final audit does not provide new ones.
        final_reflection = _extract_json_object(raw_final_reflection)
        final_assessment = str(final_reflection.get("assessment") or "").strip()[:1600]
        if final_assessment:
            latest_assessment = final_assessment
        if isinstance(final_reflection.get("gaps"), list):
            latest_gaps = [
                str(gap).strip()[:300]
                for gap in final_reflection.get("gaps", [])
                if str(gap).strip()
            ][:8]
        checklist = _apply_updates(checklist, final_reflection.get("updates"))
        events.emit(
            "final_reflection_completed",
            phase="final_reflection",
            iteration=iteration,
            plan_version=plan_version,
            data={
                "summary": latest_assessment or "Final evidence audit completed.",
                "evidence_gaps": latest_gaps,
                "checklist": checklist,
                "current_item_id": _current_checklist_item_id(checklist),
            },
        )

        # Synthesis: the model writes the cited report from the plan, checklist, final audit and
        # evidence.
        checkpoint(status="synthesizing", phase="synthesis", latest_action="Writing the cited report")
        events.emit(
            "synthesis_started",
            phase="synthesis",
            iteration=iteration,
            plan_version=plan_version,
            data={"source_count": len(sources_by_key)},
        )
        sources = list(sources_by_key.values())
        report_input = (
            f"Original request:\n{request_text}\n\nApproved plan:\n{plan}\n\n"
            f"Final checklist:\n{json.dumps(checklist, ensure_ascii=False)}\n\n"
            f"Latest evidence audit:\n{latest_assessment}\n\n"
            f"Remaining gaps:\n{json.dumps(latest_gaps, ensure_ascii=False)}\n\n"
            f"Evidence packet:\n{_evidence_packet(evidence_entries, evidence_char_budget)}"
        )
        report = _run_model_stage(
            engine=engine,
            model=model,
            system_prompt=REPORT_PROMPT,
            user_prompt=report_input,
            generation_options=generation_options,
            session_id=session_id,
            events=events,
            phase="synthesis",
            iteration=iteration,
            plan_version=plan_version,
            request_message=request_message,
        )

        # An empty model report is replaced by a fallback report built from the final audit.
        synthesis_succeeded = bool(str(report or "").strip())
        if not synthesis_succeeded:
            report = _fallback_report(
                topic=topic,
                assessment=latest_assessment,
                gaps=latest_gaps,
                sources=sources,
            )

        # The run is completed only with sources, a model report, no unsearched late revision, no
        # remaining gaps and a finished checklist; otherwise it is partial.
        unresolved_checklist = any(
            str(item.get("status") or "pending").strip().lower()
            not in {"done", "completed", "skipped"}
            for item in checklist
        )
        status = (
            "completed"
            if (
                sources
                and synthesis_succeeded
                and not late_revision_without_search
                and not latest_gaps
                and not unresolved_checklist
            )
            else "partial"
        )

        # Publish and persist the final state (all controls off) and return the result.
        events.emit(
            "session_completed",
            phase="session",
            iteration=iteration,
            plan_version=plan_version,
            data={
                "status": status,
                "source_count": len(sources),
                "queries_used": queries_used,
                "query_budget": query_budget,
            },
        )
        checkpoint(
            status=status,
            phase="completed",
            latest_action="Research completed" if status == "completed" else "Partial report completed",
            can_approve=False,
            can_edit=False,
            can_stop=False,
            report=report,
            model_context=report,
            sources=[_safe_source_preview(source) for source in sources[:50]],
        )
        return _result_payload(
            session_id=session_id,
            status=status,
            topic=topic,
            plan=plan,
            plan_version=plan_version,
            checklist=checklist,
            report=report,
            sources=sources,
            engine=engine,
            model=model,
            queries_used=queries_used,
            query_budget=query_budget,
            iteration=iteration,
            events=events,
        )
    except ResearchCancelled:
        # The user stopped the run: keep the evidence, build a fallback report and record the
        # cancelled state.
        sources = list(sources_by_key.values())
        report = _fallback_report(
            topic=topic,
            assessment=latest_assessment,
            gaps=latest_gaps,
            sources=sources,
            cancelled=True,
        )
        events.emit(
            "session_cancelled",
            phase="session",
            iteration=iteration,
            plan_version=plan_version,
            data={"status": "cancelled", "source_count": len(sources), "queries_used": queries_used},
        )
        checkpoint(
            status="cancelled",
            phase="cancelled",
            latest_action="Research stopped by the user",
            can_approve=False,
            can_edit=False,
            can_stop=False,
            report=report,
            model_context=report,
            sources=[_safe_source_preview(source) for source in sources[:50]],
        )
        return _result_payload(
            session_id=session_id,
            status="cancelled",
            topic=topic,
            plan=plan,
            plan_version=plan_version,
            checklist=checklist,
            report=report,
            sources=sources,
            engine=engine,
            model=model,
            queries_used=queries_used,
            query_budget=query_budget,
            iteration=iteration,
            events=events,
        )
    except ResearchModelUnavailable as exc:
        # The selected model failed: record the failure together with the evidence collected so far.
        sources = list(sources_by_key.values())
        report = f"Deep Research failed: {exc}"
        events.emit(
            "session_failed",
            phase="session",
            iteration=iteration,
            plan_version=plan_version,
            data={
                "status": "failed",
                "error_type": type(exc).__name__,
                "message": str(exc)[:400],
            },
        )
        checkpoint(
            status="failed",
            phase="failed",
            latest_action="The selected research model failed",
            can_approve=False,
            can_edit=False,
            can_stop=False,
            error=str(exc)[:400],
            report=report,
            model_context=report,
            sources=[_safe_source_preview(source) for source in sources[:50]],
        )
        return _result_payload(
            session_id=session_id,
            status="failed",
            topic=topic,
            plan=plan,
            plan_version=plan_version,
            checklist=checklist,
            report=report,
            sources=sources,
            engine=engine,
            model=model,
            queries_used=queries_used,
            query_budget=query_budget,
            iteration=iteration,
            events=events,
        )
    except TimeoutError as exc:
        # An approval window expired: the result is partial when evidence exists, otherwise expired.
        sources = list(sources_by_key.values())
        timeout_status = "partial" if sources or queries_used else "expired"
        events.emit(
            "session_expired",
            phase="session",
            iteration=iteration,
            plan_version=plan_version,
            data={"status": timeout_status, "message": str(exc)},
        )
        report = (
            _fallback_report(
                topic=topic,
                assessment=(
                    latest_assessment
                    or "The revised-plan approval window expired; evidence collected before that point was preserved."
                ),
                gaps=latest_gaps,
                sources=sources,
            )
            if timeout_status == "partial"
            else "Deep research did not start because the initial plan approval window expired."
        )
        checkpoint(
            status=timeout_status,
            phase="expired",
            latest_action=report,
            can_approve=False,
            can_edit=False,
            can_stop=False,
            error=str(exc)[:400],
            report=report,
            model_context=report,
            sources=[
                _safe_source_preview(source)
                for source in sources[:50]
            ],
        )
        return _result_payload(
            session_id=session_id,
            status=timeout_status,
            topic=topic,
            plan=plan,
            plan_version=plan_version,
            checklist=checklist,
            report=report,
            sources=sources,
            engine=engine,
            model=model,
            queries_used=queries_used,
            query_budget=query_budget,
            iteration=iteration,
            events=events,
        )
    except Exception as exc:
        # Any other error: preserve the evidence as a partial result (failed when there is none)
        # and return a payload instead of raising.
        sources = list(sources_by_key.values())
        failure_status = "partial" if sources else "failed"
        failure_summary = (
            latest_assessment
            or f"The run stopped safely after an internal {type(exc).__name__}; collected evidence was preserved."
        )
        report = _fallback_report(
            topic=topic,
            assessment=failure_summary,
            gaps=latest_gaps,
            sources=sources,
        )
        try:
            events.emit(
                "session_failed",
                phase="session",
                iteration=iteration,
                plan_version=plan_version,
                data={
                    "status": failure_status,
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:400],
                    "source_count": len(sources),
                },
            )
        except Exception:
            # Publishing the failure event is best-effort; the failure is recorded below anyway.
            pass
        try:
            checkpoint(
                status=failure_status,
                phase="failed",
                latest_action=(
                    "Partial evidence preserved after an internal failure"
                    if sources
                    else "Research stopped after an internal failure"
                ),
                can_approve=False,
                can_edit=False,
                can_stop=False,
                error=f"{type(exc).__name__}: {str(exc)[:300]}",
                report=report,
                model_context=report,
                sources=[_safe_source_preview(source) for source in sources[:50]],
            )
        except Exception:
            # Persisting the failure is best-effort; the result payload is still returned.
            pass
        return _result_payload(
            session_id=session_id,
            status=failure_status,
            topic=topic,
            plan=plan,
            plan_version=plan_version,
            checklist=checklist,
            report=report,
            sources=sources,
            engine=engine,
            model=model,
            queries_used=queries_used,
            query_budget=query_budget,
            iteration=iteration,
            events=events,
        )
    finally:
        # Always release the tool runtime scope and close the event log.
        try:
            tool_registry.clear_tool_runtime_scope(tool_context)
        except Exception:
            # Releasing the scope is best-effort cleanup and must not hide the run result.
            pass
        events.close()
