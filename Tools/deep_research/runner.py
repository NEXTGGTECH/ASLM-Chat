# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import json
import re
import sys
import threading
import uuid
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from API import llm_api, mcp as tool_registry
from . import control as research_control


# Repository root and the sandbox package location. The sandbox folders go on sys.path at import
# time so the sandbox import below resolves.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SANDBOX_ROOT = PROJECT_ROOT / "Tools" / "mcp-sandbox"
for _sandbox_path in (SANDBOX_ROOT / "supervisor", SANDBOX_ROOT / "src"):
    if str(_sandbox_path) not in sys.path:
        sys.path.insert(0, str(_sandbox_path))

from sandbox.temporal import temporal_sandbox
from .orchestrator import run_deep_research_v2


# Shared model runtime metadata file, directory of the per-session JSONL event logs, and the tool
# aliases a research session is meant to use (search, page reading and sandbox bash).
RUNTIME_METADATA_FILE = PROJECT_ROOT / "Tools" / "model_runtime_metadata.json"
LOGS_DIR = research_control.CONTROL_ROOT / "logs"
ALLOWED_TOOL_ALIASES = (
    "web_search__web_search",
    "web_search__read_page",
    "sandbox__bash",
)

# System prompt of the planning stage: ask for a research plan and the first search requests
# without answering the question.
PLANNER_SYSTEM_PROMPT = """You are the planning stage of an isolated deep-research session.
Do not answer the user's question yet and do not claim that you searched. Produce a concrete
research plan and the first one or two search requests. Start from the final deliverables, split
them into evidence gaps, name the source class needed for each gap, note dependencies and a clear
success condition. Select web, shopping, academic, or onion search deliberately. Use query
operators only for a stated purpose. The first search requests must be focused and must unlock the
largest number of later verification steps. Keep the plan compact enough to execute."""

# System prompt of the research agent: tool list, search discipline, citation rules and report format.
RESEARCH_SYSTEM_PROMPT = """You are an isolated deep-research agent. Execute the supplied plan,
inspect evidence, and deliver a self-contained report that answers the original request.

Available tools are exactly:
- web_search__web_search: discovery across web, shopping, academic, and onion verticals;
- web_search__read_page: inspect primary pages and promising results in depth;
- sandbox__bash: calculations, transformations, and temporary evidence analysis in a sandbox.

Research discipline:
- Work on the next unresolved evidence gap, normally with one focused medium-effort query.
- A search call may contain no more than two queries. Request no more than two web_search/read_page
  calls in one assistant turn. After search results, reason about what was established, what
  conflicts, which source classes are still missing, and the most informative next query.
- Bash does not consume the search-call budget. Consecutive bash calls are allowed when they are
  the efficient way to calculate, transform, or inspect sandbox data; no artificial reflection
  step is required between them.
- Measure coverage by answer-critical claims and source classes, not by link count. Prefer primary
  and authoritative evidence, verify important or disputed claims independently, and use read_page
  for details that snippets cannot establish.
- Do not repeat equivalent queries. Change anchors or source class when evidence is weak.
- Use the shopping vertical for prices, sellers, stock, and market candidates; academic for papers,
  authors, DOI records, and scholarly evidence. Web does not replace those verticals.
- Finish only when every answer-critical deliverable has adequate evidence or is explicitly marked
  unresolved. State material limitations and disagreements.

Citation discipline:
- Tool results provide opaque citation handles such as [c0000-1]. Copy those handles exactly next
  to the claims they support. Never invent, alter, or infer a handle.
- Cite only evidence actually returned during this isolated session.
- The final response must contain the researched answer, not a diary of tool calls or hidden chain
  of thought. Answer in the language of the original request. It may briefly summarize methodology
  or limitations when useful.

Report presentation:
- Write structured Markdown with clear sections and comparison tables when they improve scanning.
- Use valid LaTeX delimiters for equations, quantitative models, or scientific notation when useful.
- Use a valid fenced ```mermaid diagram for processes, architectures, timelines, or relationships
  that are materially clearer visually. Do not add decorative diagrams with no analytical value."""

# System prompt for compressing completed research history into durable Markdown memory.
RESEARCH_COMPRESSION_SYSTEM_PROMPT = """Compress completed deep-research history into durable
research memory. Return only these Markdown sections: Research Goal, Plan State, Verified Claims,
Contradictions, Open Evidence Gaps, Source Index, Sandbox Findings, Next Queries.

Preserve every opaque citation handle and its URL exactly. Never invent or renumber handles.
Keep claim-to-source relationships, source-class coverage, failed directions, unresolved
contradictions, and concrete next-query anchors. Summarize verbose snippets and terminal output,
but retain numerical results and assumptions needed to reproduce calculations. Do not answer the
original question and do not include hidden chain of thought."""


# Event logging

# Ordered event stream of one research session: every event gets a sequence number, is appended
# as a JSON line to the session's log file and is passed to the optional callback.
class ResearchEventLogger:
    schema_version = 1

    # Open the session's JSONL log (append mode, line-buffered) and set up the event counters.
    def __init__(
        self,
        session_id: str,
        *,
        callback=None,
        logs_dir: Path = LOGS_DIR,
    ) -> None:
        # Event sequence number, iteration counter per phase, and the phases whose iteration
        # advances on the next model output. The reentrant lock serializes concurrent emitters.
        self.session_id = str(session_id)
        self.callback = callback if callable(callback) else None
        self.sequence = 0
        self.phase_iterations: dict[str, int] = {}
        self.advance_on_next_model_chunk: set[str] = set()
        self.lock = threading.RLock()

        # The log file is named after the session id with unsafe characters replaced.
        logs_dir.mkdir(parents=True, exist_ok=True)
        safe_session_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", self.session_id).strip("-")
        self.path = (logs_dir / f"{safe_session_id}.jsonl").resolve()
        self.handle = self.path.open("a", encoding="utf-8", buffering=1)

    # Return the current iteration number of a phase, starting at 1.
    def current_iteration(self, phase: str) -> int:
        return self.phase_iterations.setdefault(str(phase), 1)

    # Append one event to the log and pass it to the callback. The iteration defaults to the
    # phase's current one. Return the event record.
    def emit(
        self,
        event_type: str,
        *,
        phase: str,
        data: Any = None,
        iteration: int | None = None,
    ) -> dict[str, Any]:
        # Number and write the record under the lock so the log stays ordered.
        with self.lock:
            self.sequence += 1
            record = {
                "schema_version": self.schema_version,
                "session_id": self.session_id,
                "sequence": self.sequence,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "phase": str(phase),
                "iteration": iteration if iteration is not None else self.current_iteration(phase),
                "type": str(event_type),
                "data": data if data is not None else {},
            }
            self.handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self.handle.flush()

        # Notify the observer outside the lock.
        if self.callback is not None:
            try:
                self.callback(record)
            except Exception:
                # Observers must never interrupt research.
                pass
        return record

    # Start the next iteration of a phase when a tool result was seen and the model has now
    # produced new output.
    def _advance_iteration_if_needed(self, phase: str) -> None:
        if phase not in self.advance_on_next_model_chunk:
            return
        self.phase_iterations[phase] = self.current_iteration(phase) + 1
        self.advance_on_next_model_chunk.discard(phase)
        self.emit("iteration_started", phase=phase)

    # Translate one raw streamed generation chunk into logged events: reasoning and content deltas,
    # the completed model turn, tool calls, tool activity and results, and token usage. Model
    # output after a tool result starts a new iteration of the phase.
    def observe_chunk(self, phase: str, raw_chunk: Any) -> None:
        # Normalize every part of the chunk to plain dicts.
        chunk = _as_dict(raw_chunk)
        message = _as_dict(chunk.get("message"))
        transcript = _as_dict(chunk.get("transcript_message"))
        tool_events = chunk.get("tool_events")
        tool_event = chunk.get("tool_event")
        tool_result = _as_dict(chunk.get("tool_result"))
        tool_progress = _as_dict(chunk.get("tool_progress"))

        # Any model output or tool event after a tool result begins the next iteration.
        if message or transcript or tool_events or tool_event:
            self._advance_iteration_if_needed(phase)

        # Model output: reasoning and visible content deltas, then the completed turn.
        thinking = message.get("thinking")
        if isinstance(thinking, str) and thinking:
            self.emit(
                "model_output_delta",
                phase=phase,
                data={"channel": "reasoning", "content": thinking},
            )
        content = message.get("content")
        if isinstance(content, str) and content:
            self.emit(
                "model_output_delta",
                phase=phase,
                data={"channel": "content", "content": content},
            )
        if transcript:
            self.emit("model_turn_completed", phase=phase, data=transcript)

        # Tool calls arrive as a list, a single event, or both.
        normalized_tool_events = tool_events if isinstance(tool_events, list) else []
        if isinstance(tool_event, dict):
            normalized_tool_events = [*normalized_tool_events, tool_event]
        for event in normalized_tool_events:
            if isinstance(event, dict):
                self.emit("tool_call", phase=phase, data=event)

        # Tool progress and results; a result arms the iteration advance for the next model output.
        if tool_progress:
            self.emit("tool_activity", phase=phase, data=tool_progress)
        if tool_result:
            self.emit("tool_result", phase=phase, data=tool_result)
            self.advance_on_next_model_chunk.add(phase)

        # Token usage: the chunk's usage object, or the provider's loose token count fields.
        usage = chunk.get("usage")
        if not isinstance(usage, dict):
            usage = {
                key: chunk[key]
                for key in (
                    "prompt_eval_count",
                    "eval_count",
                    "prompt_tokens",
                    "completion_tokens",
                    "total_tokens",
                )
                if chunk.get(key) is not None
            }
        if usage:
            self.emit("usage", phase=phase, data=usage)

    # Close the log file; calling it again is harmless.
    def close(self) -> None:
        with self.lock:
            if not self.handle.closed:
                self.handle.close()


# Runtime model

# Read the shared runtime model metadata JSON object. Raise RuntimeError when the file is
# unreadable, malformed or not a JSON object.
def _read_runtime_metadata(metadata_path: Path = RUNTIME_METADATA_FILE) -> dict[str, Any]:
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not read runtime model metadata: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Runtime model metadata must contain a JSON object.")
    return payload


# Resolve the caller's current model and enrich it from the shared runtime JSON. The caller's
# engine and model win over the active ones in the metadata. Return the engine, model,
# capabilities, limits and where the choice came from. Raise RuntimeError when no model is
# available or the model does not support tool calling.
def resolve_runtime_model(
    context: Mapping[str, Any] | None,
    *,
    metadata_path: Path = RUNTIME_METADATA_FILE,
) -> dict[str, Any]:
    # Choose the engine and model: caller context first, then the active runtime model.
    metadata = _read_runtime_metadata(metadata_path)
    active = metadata.get("active") if isinstance(metadata.get("active"), dict) else {}
    caller = context if isinstance(context, Mapping) else {}
    engine = str(caller.get("engine") or active.get("engine") or "").strip()
    model = str(caller.get("model_name") or caller.get("model") or active.get("model") or "").strip()
    if not engine or not model:
        raise RuntimeError("No active engine/model is available for deep research.")

    # Find the model record by its engine:model key, else by scanning the records.
    models = metadata.get("models") if isinstance(metadata.get("models"), dict) else {}
    record = models.get(f"{engine}:{model}")
    if not isinstance(record, dict):
        record = next(
            (
                value
                for value in models.values()
                if isinstance(value, dict)
                and str(value.get("engine") or "").strip() == engine
                and str(value.get("model") or "").strip() == model
            ),
            {},
        )

    # Refuse a model that is known not to support tool calling.
    capabilities = record.get("capabilities") if isinstance(record.get("capabilities"), dict) else {}
    if capabilities.get("tools") is False:
        raise RuntimeError(f"The selected model '{model}' does not support tool calling.")

    return {
        "engine": engine,
        "model": model,
        "capabilities": dict(capabilities),
        "limits": dict(record.get("limits") or {}) if isinstance(record.get("limits"), dict) else {},
        "metadata_source": (
            "caller_context"
            if caller.get("engine") and (caller.get("model_name") or caller.get("model"))
            else "runtime_active"
        ),
    }


# Chunk parsing

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
    chunk_payload = _as_dict(chunk)
    return _as_dict(chunk_payload.get(key))


# Return the structured payload of a chunk's tool result: its structured_content (or
# structuredContent) dict, else its content when that is a dict or a JSON object string.
# Return an empty dict when there is none.
def _structured_tool_result(chunk: Any) -> dict[str, Any]:
    tool_message = _message_from_chunk(chunk, "tool_result")
    structured = tool_message.get("structured_content") or tool_message.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    content = tool_message.get("content")
    if isinstance(content, dict):
        return content
    if isinstance(content, str):
        try:
            decoded = json.loads(content)
        except json.JSONDecodeError:
            return {}
        return decoded if isinstance(decoded, dict) else {}
    return {}


# Report assembly

# Replace bracketed citation handles such as [c0000-1], alone or comma-separated, with Markdown
# links to the matching source URLs, so internal handles are resolved before the report reaches
# the parent model. Handles without a known http(s) source stay as plain [handle]; handles that
# are already followed by a link target are left untouched.
def replace_citation_handles_with_markdown_links(
    text: str,
    sources: list[dict[str, Any]],
) -> str:
    # Map each lower-cased source id to its URL, keeping only valid http(s) URLs.
    source_urls: dict[str, str] = {}
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_id = str(source.get("id") or "").strip().lower()
        source_url = str(source.get("url") or "").strip()
        try:
            parsed_url = urlsplit(source_url)
        except ValueError:
            # A malformed URL cannot be linked, so the handle stays plain.
            continue
        if source_id and parsed_url.scheme in {"http", "https"} and parsed_url.netloc:
            source_urls[source_id] = source_url

    # Match one or more comma-separated handles inside brackets that are not already a link.
    handle = r"c[a-z0-9]{3,}-\d+"
    group_pattern = re.compile(
        rf"\[\s*({handle}(?:\s*,\s*{handle})*)\s*\](?!\()",
        flags=re.I,
    )

    # Turn one bracketed handle group into one link per handle (plain [handle] when unknown).
    def replace_group(match: re.Match[str]) -> str:
        links: list[str] = []
        for raw_id in match.group(1).split(","):
            citation_id = raw_id.strip()
            source_url = source_urls.get(citation_id.lower())
            links.append(
                f"[{citation_id}]({source_url})"
                if source_url
                else f"[{citation_id}]"
            )
        return " ".join(links)

    return group_pattern.sub(replace_group, str(text or ""))


# Consume a generation result and return the final text with the unique sources found in its tool
# results. The text is the last transcript message without tool calls, falling back to the
# concatenated visible content. Every chunk is also reported to the event logger when one is given.
def _collect_generation(
    result: Any,
    *,
    event_logger: ResearchEventLogger | None = None,
    phase: str = "generation",
) -> tuple[str, list[dict[str, Any]]]:
    visible_fragments: list[str] = []
    final_transcript = ""
    sources_by_key: dict[str, dict[str, Any]] = {}

    for raw_chunk in _iter_generation(result):
        # Log the chunk, then collect the visible content fragments.
        if event_logger is not None:
            event_logger.observe_chunk(phase, raw_chunk)
        message = _message_from_chunk(raw_chunk, "message")
        content = message.get("content")
        if isinstance(content, str) and content:
            visible_fragments.append(content)

        # A transcript message without tool calls is a finished answer; the latest one wins.
        transcript = _message_from_chunk(raw_chunk, "transcript_message")
        transcript_content = transcript.get("content")
        if (
            isinstance(transcript_content, str)
            and transcript_content.strip()
            and not transcript.get("tool_calls")
        ):
            final_transcript = transcript_content.strip()

        # Sources from tool results, deduplicated by id (or URL); the first occurrence wins.
        structured = _structured_tool_result(raw_chunk)
        raw_sources = structured.get("sources") if isinstance(structured, dict) else None
        if not isinstance(raw_sources, list):
            continue
        for raw_source in raw_sources:
            if not isinstance(raw_source, dict):
                continue
            source = dict(raw_source)
            source_id = str(source.get("id") or "").strip()
            source_url = str(source.get("url") or "").strip()
            key = source_id or source_url
            if key and key not in sources_by_key:
                sources_by_key[key] = source

    text = final_transcript or "".join(visible_fragments).strip()
    return text, list(sources_by_key.values())


# Run settings

# Build the generation options: always streaming, plus high-level thinking when the model supports it.
def _generation_options(runtime: dict[str, Any]) -> dict[str, Any]:
    options: dict[str, Any] = {"stream": True}
    capabilities = runtime.get("capabilities") if isinstance(runtime.get("capabilities"), dict) else {}
    if capabilities.get("thinking") is True:
        options.update({"think": True, "think_level": "high"})
    return options


# Coerce the requested round limit to an integer from 4 to 24; a missing or invalid value gives 12.
def _coerce_max_rounds(value: Any) -> int:
    try:
        return min(24, max(4, int(value or 12)))
    except (TypeError, ValueError):
        return 12


# Return the history size in characters at which compression starts: 45% of the model's context
# window at about 4 characters per token, clamped to 24,000-120,000, or 60,000 when the window is
# unknown.
def _compression_trigger_chars(runtime: dict[str, Any]) -> int:
    limits = runtime.get("limits") if isinstance(runtime.get("limits"), dict) else {}
    raw_tokens = limits.get("context_window") or limits.get("model_context_limit") or 0
    try:
        context_tokens = max(0, int(raw_tokens))
    except (TypeError, ValueError):
        context_tokens = 0
    if not context_tokens:
        return 60_000
    return min(120_000, max(24_000, int(context_tokens * 4 * 0.45)))


# History compression

# Build a research memory without the model, used when the model returns no summary: the standard
# section headings, every citation handle and URL found in the history, and its last 16,000
# characters verbatim.
def _fallback_research_memory(raw_history: str) -> str:
    handles = list(dict.fromkeys(re.findall(r"\bc[a-z0-9]{3,}-\d+\b", raw_history, flags=re.I)))
    urls = list(dict.fromkeys(re.findall(r"https?://[^\s\"'<>]+", raw_history)))
    tail = raw_history[-16_000:]
    return (
        "## Research Goal\nPreserved in the active request.\n\n"
        "## Plan State\nCompression fallback retained the newest research observations.\n\n"
        "## Verified Claims\nReview the preserved observations below.\n\n"
        "## Contradictions\nNot semantically classified by the fallback.\n\n"
        "## Open Evidence Gaps\nReassess against the active plan.\n\n"
        "## Source Index\n"
        + "\n".join([*(f"- [{handle}]" for handle in handles), *(f"- {url}" for url in urls)])
        + "\n\n## Sandbox Findings\nSee preserved observations.\n\n"
        "## Next Queries\nRe-evaluate the next unresolved evidence gap.\n\n"
        "## Preserved Observations\n"
        + tail
    )


# Build the compaction callback for the orchestrator. It asks the selected model to summarize
# completed research turns into durable memory and logs the compression events. When the model
# returns nothing it falls back to the mechanical memory.
def _research_compactor(
    runtime: dict[str, Any],
    event_logger: ResearchEventLogger | None = None,
):
    engine = str(runtime["engine"])
    model = str(runtime["model"])

    # Summarize the history entries and return the research memory text. The provider format is
    # only recorded in the logged event.
    def compact(entries: list[dict[str, Any]], *, provider_format: str = "standard") -> str:
        # Announce the compression, then stream the model's summary of the serialized history.
        if event_logger is not None:
            event_logger.emit(
                "compression_started",
                phase="compression",
                data={"provider_format": provider_format, "entry_count": len(entries)},
            )
        raw_history = json.dumps(entries, ensure_ascii=False, default=str)
        compression_result = llm_api.generate(
            engine,
            model,
            [
                {"role": "system", "content": RESEARCH_COMPRESSION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": "Compress these completed research turns:\n\n" + raw_history,
                },
            ],
            stream=True,
        )
        summary, _sources = _collect_generation(
            compression_result,
            event_logger=event_logger,
            phase="compression",
        )

        # Use the fallback memory when the model produced no summary, and log the outcome.
        fallback_used = not bool(summary)
        compressed = summary or _fallback_research_memory(raw_history)
        if event_logger is not None:
            event_logger.emit(
                "compression_completed",
                phase="compression",
                data={
                    "fallback_used": fallback_used,
                    "input_chars": len(raw_history),
                    "output_chars": len(compressed),
                },
            )
            event_logger.phase_iterations["compression"] = (
                event_logger.current_iteration("compression") + 1
            )
        return compressed

    return compact


# Entry point

# Run the deterministic v2 state machine with the caller's selected model. When the model cannot
# be resolved, mark the stored session as failed (if the arguments carry a session id) and
# re-raise the error.
def run_deep_research(
    arguments: Mapping[str, Any],
    context: Mapping[str, Any] | None = None,
    *,
    metadata_path: Path = RUNTIME_METADATA_FILE,
    logs_dir: Path = LOGS_DIR,
) -> dict[str, Any]:
    try:
        runtime = resolve_runtime_model(context, metadata_path=metadata_path)
    except Exception as exc:
        # Record the failure in the session so the UI stops waiting, then re-raise.
        session_id = str(arguments.get("session_id") or "").strip()
        if session_id:
            try:
                research_control.update_state(
                    session_id,
                    status="failed",
                    phase="failed",
                    latest_action="The selected research model could not be initialized",
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                    can_approve=False,
                    can_edit=False,
                    can_stop=False,
                )
            except (OSError, ValueError):
                # Recording the failure is best-effort; the original error is re-raised anyway.
                pass
        raise

    # Hand over to the orchestrator with the resolved model.
    return run_deep_research_v2(
        arguments,
        context,
        runtime=runtime,
        generation_options=_generation_options(runtime),
        logs_dir=logs_dir,
    )
