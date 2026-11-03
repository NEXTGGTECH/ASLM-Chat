# Copyright NEXTGGTECH. Elastic License 2.0.

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock


from Tools.deep_research import orchestrator


# Build the chunk stream of a fake model reply that delivers the whole text as one assistant message.
def _generation(text: str):
    return iter([{"message": {"role": "assistant", "content": text}}])


# Build a fake model stream that raises the given exception as soon as it is first iterated.
def _failed_generation(exc: Exception):
    # Generator that raises on first iteration; the unreachable yield makes it a generator function.
    def stream():
        raise exc
        yield  # pragma: no cover - keeps this a lazy generator

    return stream()


# Build the structured result of a fake search or read tool call that returns one primary source.
def _source_result() -> dict[str, Any]:
    source = {
        "id": "c1000-1",
        # Deliberately not HTTP(S): the tests exercise search without triggering
        # the optional read-page branch.
        "url": "urn:test:primary-source",
        "title": "Primary test source",
    }
    return {
        "_tool_result_structured": {
            "model_context": "Primary evidence [c1000-1]",
            "sources": [source],
        }
    }


# Build the six scripted model replies of a successful run: plan, first reflection (gap found), query
# selection (replaceable through the selection argument), second reflection (complete), final audit and report.
def _successful_outputs(*, selection: str | None = None) -> list[str]:
    # JSON replies carry a one-step plan for a single primary-source search; the last item is the final report.
    return [
        (
            '{"summary":"Verify the claim","steps":'
            '[{"id":"s1","title":"Find primary evidence"}],'
            '"candidates":[{"text":"claim primary evidence","vertical":"web",'
            '"purpose":"find the primary source"}]}'
        ),
        (
            '{"assessment":"Primary evidence is still missing.",'
            '"gaps":["primary source"],"updates":[],"complete":false,'
            '"candidates":[{"text":"claim primary evidence","vertical":"web",'
            '"purpose":"find the primary source"}]}'
        ),
        selection
        or (
            '{"queries":[{"text":"claim primary evidence","vertical":"web",'
            '"purpose":"highest-value gap"}]}'
        ),
        (
            '{"assessment":"The claim now has primary evidence.","gaps":[],'
            '"updates":[{"id":"s1","status":"done"}],"complete":true,'
            '"candidates":[]}'
        ),
        (
            '{"assessment":"Final audit confirms the primary evidence.","gaps":[],'
            '"updates":[{"id":"s1","status":"done"}],"complete":true,'
            '"candidates":[]}'
        ),
        "Verified final report [c1000-1]",
    ]


# Cover the deep research v2 orchestrator: planning, approval, search, resume, persistence and failure handling.
class ResearchV2Tests(unittest.TestCase):
    # Create temporary control and log directories, redirect the control state paths to them and replace the
    # tool registry build and scope cleanup with mocks (exposed as build_tools and clear_scope).
    def setUp(self) -> None:
        # Temporary directories removed after each test.
        self._control_directory = tempfile.TemporaryDirectory()
        self._log_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self._log_directory.cleanup)
        self.addCleanup(self._control_directory.cleanup)

        # Keep session state and commands inside the temporary directory.
        control_root = Path(self._control_directory.name)
        self.enterContext(
            mock.patch.object(orchestrator.control, "STATE_ROOT", control_root / "state")
        )
        self.enterContext(
            mock.patch.object(orchestrator.control, "COMMAND_ROOT", control_root / "commands")
        )

        # Offer only the web search and read page tools, and skip real runtime scope cleanup.
        self.build_tools = self.enterContext(
            mock.patch.object(
                orchestrator.tool_registry,
                "build_ollama_tools",
                return_value=(
                    [],
                    {
                        "web_search__web_search": {},
                        "web_search__read_page": {},
                    },
                ),
            )
        )
        self.clear_scope = self.enterContext(
            mock.patch.object(orchestrator.tool_registry, "clear_tool_runtime_scope")
        )

    # Run the v2 orchestrator on a fixed test topic with a stub runtime and return its result; approval is
    # automatic only when auto_approve is set.
    def _run(
        self,
        session_id: str,
        *,
        auto_approve: bool = False,
        max_rounds: int = 2,
    ) -> dict[str, Any]:
        return orchestrator.run_deep_research_v2(
            {
                "topic": "Investigate the test claim",
                "session_id": session_id,
                "max_rounds": max_rounds,
                "approval_timeout_s": 5,
            },
            {"auto_approve": auto_approve},
            runtime={"engine": "test-engine", "model": "test-model"},
            generation_options={"stream": True, "think": True, "think_level": "high"},
            logs_dir=Path(self._log_directory.name),
        )

    # Start a run in a background thread and return the thread with an outcome dict that receives either
    # "result" or "error"; the thread is cancelled on cleanup if it is still alive.
    def _start_run(self, session_id: str) -> tuple[threading.Thread, dict[str, Any]]:
        outcome: dict[str, Any] = {}

        # Thread body: store the run result, or the exception so the owning test thread can re-raise it.
        def target() -> None:
            try:
                outcome["result"] = self._run(session_id)
            except BaseException as exc:
                # Surfaced in the owning test thread through the outcome dict.
                outcome["error"] = exc

        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        self.addCleanup(self._stop_thread, thread, session_id)
        return thread, outcome

    # Cancel a research thread that is still running and wait up to two seconds for it to exit.
    @staticmethod
    def _stop_thread(thread: threading.Thread, session_id: str) -> None:
        if not thread.is_alive():
            return
        try:
            orchestrator.control.submit_command(session_id, "cancel")
        except (FileNotFoundError, ValueError):
            # The session is gone or already finished, so there is nothing left to cancel.
            pass
        thread.join(timeout=2)

    # Poll the session state until the predicate accepts it and return that snapshot; fail the test with the
    # latest snapshot when the timeout expires.
    def _wait_for_state(
        self,
        session_id: str,
        predicate,
        *,
        timeout: float = 3,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        latest: dict[str, Any] = {}
        while time.monotonic() < deadline:
            latest = orchestrator.control.read_state(session_id)
            if latest and predicate(latest):
                return latest
            time.sleep(0.01)
        self.fail(f"Timed out waiting for research state; latest snapshot: {latest!r}")

    # Wait for a background run to finish, re-raise the error it stored, and return its result.
    def _join_successfully(
        self,
        thread: threading.Thread,
        outcome: dict[str, Any],
    ) -> dict[str, Any]:
        thread.join(timeout=5)
        self.assertFalse(thread.is_alive(), "research worker did not stop")
        if "error" in outcome:
            raise outcome["error"]
        self.assertIn("result", outcome)
        return outcome["result"]

    # Check that resuming a session that was interrupted in the given phase (search or reading_completed)
    # after iteration 1 had spent its full query batch does not replay that batch: only one new search runs,
    # as iteration 2, within the per-iteration query limit and the doubled total budget, and the model is
    # not asked to plan again.
    def _assert_counted_phase_resume_advances(
        self,
        *,
        session_id: str,
        phase: str,
    ) -> None:
        # Arrange: a persisted running session positioned in the given phase of iteration 1.
        orchestrator.control.create_session(session_id, topic="Investigate the test claim")
        orchestrator.control.update_state(
            session_id,
            status="running",
            phase=phase,
            plan="Verify the claim",
            plan_version=1,
            checklist=[{"id": "s1", "title": "Verify the claim", "status": "active"}],
            iteration=1,
            # Iteration 1 already charged its maximum batch before entering either
            # tested phase.  Recovery must not charge another batch as iteration 1.
            queries_used=orchestrator.MAX_QUERIES_PER_ITERATION,
            query_budget=orchestrator.MAX_QUERIES_PER_ITERATION * 2,
            seen_queries=["first counted query", "second counted query"],
            active_queries=[
                {"text": "first counted query", "vertical": "web"},
                {"text": "second counted query", "vertical": "web"},
            ] if phase == "search" else [],
            initial_candidates=[],
        )

        # Scripted model replies for the resumed run: reflection with two new candidates, selection of both,
        # a completing reflection and the final report.
        outputs = iter(
            [
                (
                    '{"assessment":"More evidence is required.","gaps":["primary source"],'
                    '"updates":[],"complete":false,"candidates":['
                    '{"text":"third query","vertical":"web","purpose":"verify"},'
                    '{"text":"fourth query","vertical":"web","purpose":"corroborate"}]}'
                ),
                (
                    '{"queries":['
                    '{"text":"third query","vertical":"web","purpose":"verify"},'
                    '{"text":"fourth query","vertical":"web","purpose":"corroborate"}]}'
                ),
                (
                    '{"assessment":"The claim is now verified.","gaps":[],'
                    '"updates":[{"id":"s1","status":"done"}],'
                    '"complete":true,"candidates":[]}'
                ),
                "Verified final report [c1000-1]",
            ]
        )

        # Act: resume the session with the model and the search tool mocked.
        with (
            mock.patch.object(
                orchestrator.llm_api,
                "generate",
                side_effect=lambda *_args, **_kwargs: _generation(next(outputs)),
            ) as generate,
            mock.patch.object(
                orchestrator.tool_registry,
                "call_ollama_tool",
                return_value=_source_result(),
            ) as call_tool,
        ):
            result = self._run(session_id, auto_approve=False, max_rounds=2)

        # Assert: a single search of at most MAX_QUERIES_PER_ITERATION queries, charged to iteration 2, and
        # the first model call is not the planning prompt.
        web_calls = [
            call
            for call in call_tool.call_args_list
            if call.args[1] == "web_search__web_search"
        ]
        self.assertEqual(len(web_calls), 1)
        self.assertTrue(
            all(
                len(
                    call.args[2]["query"]
                    if isinstance(call.args[2]["query"], list)
                    else [call.args[2]["query"]]
                ) <= orchestrator.MAX_QUERIES_PER_ITERATION
                for call in web_calls
            )
        )
        self.assertEqual(
            result["ui"]["queries_used"],
            orchestrator.MAX_QUERIES_PER_ITERATION * 2,
        )
        self.assertEqual(
            [
                event["iteration"]
                for event in result["ui"]["events"]
                if event.get("type") == "search_started"
            ],
            [2],
        )
        self.assertNotEqual(generate.call_args_list[0].args[2][0]["content"], orchestrator.PLAN_PROMPT)

    # Verify that resuming a session interrupted during the search phase does not replay the already charged
    # query batch within the same iteration.
    def test_resume_from_inflight_search_does_not_replay_batch_in_same_iteration(self) -> None:
        self._assert_counted_phase_resume_advances(
            session_id="deep-research:10000000000000000000000000000031",
            phase="search",
        )

    # Verify that resuming a session whose reading phase had completed continues with the next iteration
    # instead of charging another batch to the same one.
    def test_resume_from_reading_completed_starts_the_next_iteration(self) -> None:
        self._assert_counted_phase_resume_advances(
            session_id="deep-research:10000000000000000000000000000032",
            phase="reading_completed",
        )

    # Verify that a plan JSON object is extracted from a fenced block surrounded by prose, and that a plain
    # text plan without JSON becomes a checklist of pending items.
    def test_tolerant_json_and_plain_text_plan_fallback(self) -> None:
        # Fenced JSON between a preface and trailing commentary.
        parsed = orchestrator._extract_json_object(
            "Model preface that should be ignored.\n"
            "```json\n"
            '{"summary":"Focused plan","steps":[{"title":"Verify primary evidence"}]}\n'
            "```\nTrailing commentary."
        )
        self.assertEqual(parsed["summary"], "Focused plan")
        self.assertEqual(parsed["steps"][0]["title"], "Verify primary evidence")

        # A numbered and a bulleted line become two pending checklist items.
        fallback = orchestrator._normalize_checklist(
            None,
            "1. Verify the primary record\n- Compare an independent source",
        )
        self.assertEqual(
            [item["title"] for item in fallback],
            ["Verify the primary record", "Compare an independent source"],
        )
        self.assertTrue(all(item["status"] == "pending" for item in fallback))

    # Verify that a reflection update cannot move a done checklist item back to pending (its note is kept),
    # that other items follow their updates, and that the current item is the last active one.
    def test_completed_checklist_item_never_regresses_during_later_reflection(self) -> None:
        # Arrange: one done, one active and one pending item.
        checklist = [
            {
                "id": "s1",
                "title": "Find primary evidence",
                "status": "done",
                "note": "Primary record verified",
            },
            {"id": "s2", "title": "Verify an independent source", "status": "active"},
            {"id": "s3", "title": "Resolve contradictions", "status": "pending"},
        ]

        # Act: apply stale updates that try to reset s1 and activate both s2 and s3.
        updated = orchestrator._apply_updates(
            checklist,
            [
                {"id": "s1", "status": "pending", "note": "stale partial audit"},
                {"id": "s2", "status": "active"},
                {"id": "s3", "status": "active"},
            ],
        )

        # Assert: s1 stays done with its note, s2 falls back to pending, and s3 is the current item.
        self.assertEqual(updated[0]["status"], "done")
        self.assertEqual(updated[0]["note"], "Primary record verified")
        self.assertEqual(updated[1]["status"], "pending")
        self.assertEqual(updated[2]["status"], "active")
        self.assertEqual(orchestrator._current_checklist_item_id(updated), "s3")

    # Verify that merging a reloaded plan keeps the done status and note of an item whose title is unchanged,
    # and resets an item whose title was changed to pending.
    def test_unchanged_completed_item_survives_plan_reload_but_renamed_item_does_not(self) -> None:
        previous = [
            {
                "id": "s1",
                "title": "Find primary evidence",
                "status": "done",
                "note": "Primary record found",
            }
        ]

        recovered = orchestrator._merge_checklist_progress(
            previous,
            [{"id": "s1", "title": "Find primary evidence", "status": "pending"}],
        )
        revised = orchestrator._merge_checklist_progress(
            previous,
            [{"id": "s1", "title": "Find contradictory evidence", "status": "pending"}],
        )

        self.assertEqual(recovered[0]["status"], "done")
        self.assertEqual(recovered[0]["note"], "Primary record found")
        self.assertEqual(revised[0]["status"], "pending")

    # Verify that writing a checkpoint with an older checklist after a worker restart cannot move a done
    # item back to pending, while the other item still takes the new status.
    def test_durable_checkpoint_cannot_reset_completed_item_after_worker_restart(self) -> None:
        session_id = "deep-research:10000000000000000000000000000021"
        orchestrator.control.create_session(session_id, topic="Investigate the test claim")
        orchestrator.control.update_state(
            session_id,
            checklist=[
                {"id": "s1", "title": "Find primary evidence", "status": "done"},
                {"id": "s2", "title": "Compare sources", "status": "pending"},
            ],
        )

        recovered = orchestrator.control.update_state(
            session_id,
            checklist=[
                {"id": "s1", "title": "Find primary evidence", "status": "pending"},
                {"id": "s2", "title": "Compare sources", "status": "active"},
            ],
        )

        self.assertEqual(recovered["checklist"][0]["status"], "done")
        self.assertEqual(recovered["checklist"][1]["status"], "active")

    # Verify that a model that refuses the connection fails the run during planning, with no sources and no
    # search tool call.
    def test_unavailable_model_fails_before_search_without_fallback(self) -> None:
        session_id = "deep-research:10000000000000000000000000000008"
        self.enterContext(
            mock.patch.object(
                orchestrator.llm_api,
                "generate",
                side_effect=ConnectionRefusedError("connection refused"),
            )
        )
        call_tool = self.enterContext(
            mock.patch.object(orchestrator.tool_registry, "call_ollama_tool")
        )

        result = self._run(session_id, auto_approve=True)

        self.assertEqual(result["ui"]["status"], "failed")
        self.assertIn("model 'test-model' failed during planning", result["model_context"])
        self.assertEqual(result["sources"], [])
        call_tool.assert_not_called()

    # Verify that a provider error raised only when the reply stream is iterated also fails the run before
    # any search, and the error text appears in the model context.
    def test_lazy_provider_error_also_fails_before_search(self) -> None:
        session_id = "deep-research:10000000000000000000000000000018"
        self.enterContext(
            mock.patch.object(
                orchestrator.llm_api,
                "generate",
                return_value=_failed_generation(RuntimeError("Connection error.")),
            )
        )
        call_tool = self.enterContext(
            mock.patch.object(orchestrator.tool_registry, "call_ollama_tool")
        )

        result = self._run(session_id, auto_approve=True)

        self.assertEqual(result["ui"]["status"], "failed")
        self.assertIn("RuntimeError: Connection error.", result["model_context"])
        self.assertEqual(result["sources"], [])
        call_tool.assert_not_called()

    # Verify that selected queries become web_search calls with exactly the query, vertical, effort and
    # description arguments, one call per vertical.
    def test_search_calls_follow_simple_web_search_contract(self) -> None:
        calls = orchestrator._tool_search_calls(
            [
                {
                    "text": 'site:docs.python.org "free-threaded mode" -forum',
                    "vertical": "web",
                    "purpose": "Find official documentation",
                },
                {
                    "text": "transformer attention complexity",
                    "vertical": "academic",
                    "purpose": "Find papers",
                },
            ],
            1,
        )

        # One call per vertical: a call carries a single vertical under the
        # simple web_search contract.
        self.assertEqual(
            [call["vertical"] for call in calls],
            ["web", "academic"],
        )

        # The first call keeps the raw query text (including operators), and its description is the purpose.
        web_call = calls[0]
        self.assertEqual(
            set(web_call),
            {"query", "vertical", "effort", "description"},
        )
        self.assertEqual(web_call["query"], 'site:docs.python.org "free-threaded mode" -forum')
        self.assertEqual(web_call["effort"], "medium")
        self.assertEqual(web_call["description"], "Find official documentation")

        # A single query with an empty purpose still produces one call with the same query.
        single = orchestrator._tool_search_calls(
            [{"text": "python asyncio tutorial", "vertical": "web", "purpose": ""}],
            2,
        )
        self.assertEqual(single[0]["query"], "python asyncio tutorial")

    # Verify that a run without auto-approval stops in the awaiting_approval state after planning only, makes
    # no tool call until an approve command arrives, and then completes with a stored report and sources.
    def test_research_waits_for_explicit_approval_before_any_search(self) -> None:
        # Arrange: scripted model replies and a mocked search tool.
        session_id = "deep-research:10000000000000000000000000000001"
        outputs = iter(_successful_outputs())
        generate = self.enterContext(
            mock.patch.object(
                orchestrator.llm_api,
                "generate",
                side_effect=lambda *_args, **_kwargs: _generation(next(outputs)),
            )
        )
        call_tool = self.enterContext(
            mock.patch.object(
                orchestrator.tool_registry,
                "call_ollama_tool",
                return_value=_source_result(),
            )
        )

        # Act and assert before approval: the run waits with one model call (planning) and no tool call.
        thread, outcome = self._start_run(session_id)
        awaiting = self._wait_for_state(
            session_id,
            lambda state: state.get("status") == "awaiting_approval",
        )

        self.assertEqual(awaiting["plan_version"], 1)
        self.assertEqual(generate.call_count, 1, "only planning may run before approval")
        call_tool.assert_not_called()

        # Approve plan version 1 and assert the run completes with the report persisted in the final state.
        accepted = orchestrator.control.submit_command(
            session_id,
            "approve",
            expected_plan_version=1,
        )
        self.assertTrue(accepted["accepted"])
        result = self._join_successfully(thread, outcome)

        self.assertEqual(result["ui"]["status"], "completed")
        self.assertGreaterEqual(call_tool.call_count, 1)
        terminal_state = orchestrator.control.read_state(session_id)
        self.assertEqual(terminal_state["report"], result["report"])
        self.assertTrue(terminal_state["sources"])

    # Verify that a revise command while awaiting approval replaces the plan, rebuilds the checklist and
    # increments the plan version, and that approving the new version completes the run with the revised plan.
    def test_plan_revision_increments_version_before_approval(self) -> None:
        # Arrange: scripted model replies and a mocked search tool.
        session_id = "deep-research:10000000000000000000000000000002"
        outputs = iter(_successful_outputs())
        self.enterContext(
            mock.patch.object(
                orchestrator.llm_api,
                "generate",
                side_effect=lambda *_args, **_kwargs: _generation(next(outputs)),
            )
        )
        self.enterContext(
            mock.patch.object(
                orchestrator.tool_registry,
                "call_ollama_tool",
                return_value=_source_result(),
            )
        )

        # Act: wait for the first plan, then submit a revision of it.
        thread, outcome = self._start_run(session_id)
        self._wait_for_state(
            session_id,
            lambda state: state.get("status") == "awaiting_approval"
            and state.get("plan_version") == 1,
        )
        revised_plan = (
            "- [ ] Verify the official primary record\n"
            "- [ ] Compare it with independent evidence"
        )
        orchestrator.control.submit_command(
            session_id,
            "revise",
            payload={"plan": revised_plan},
            expected_plan_version=1,
        )

        # Assert: the state shows plan version 2 with a two-item checklist taken from the revised plan.
        revised = self._wait_for_state(
            session_id,
            lambda state: state.get("plan_version") == 2,
        )
        self.assertEqual(revised["plan"], revised_plan)
        self.assertEqual(len(revised["checklist"]), 2)
        self.assertEqual(
            [item["title"] for item in revised["checklist"]],
            ["Verify the official primary record", "Compare it with independent evidence"],
        )

        # Approve version 2 and check that the finished run carries the revised plan.
        orchestrator.control.submit_command(
            session_id,
            "approve",
            expected_plan_version=2,
        )
        result = self._join_successfully(thread, outcome)

        self.assertEqual(result["ui"]["plan_version"], 2)
        self.assertEqual(result["plan"], revised_plan)

    # Verify that two handles for the same page (different host prefix and trailing slash) merge into one
    # source that keeps the second handle as a citation alias, so a citation using it still links to the page.
    def test_duplicate_source_handles_remain_linkable(self) -> None:
        sources_by_key: dict[str, dict[str, Any]] = {}
        orchestrator._merge_sources(
            sources_by_key,
            [{"id": "c0001-1", "url": "https://www.example.com/page", "title": "First"}],
        )
        orchestrator._merge_sources(
            sources_by_key,
            [{"id": "c0002-7", "url": "https://example.com/page/", "title": "Duplicate"}],
        )

        sources = list(sources_by_key.values())
        linked = orchestrator._citation_links("Claim [c0002-7]", sources)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["citation_aliases"], ["c0002-7"])
        self.assertIn("[First](https://www.example.com/page)", linked)

    # Verify that a resumed session keeps its stored evidence and stable source ids, and that a new worker's
    # raw handle "c0000-1" (which collides with the old source's alias) is mapped to a new stable id instead
    # of being reused; a repeated run after completion replays the stored result without any model or tool call.
    def test_restart_restores_evidence_and_never_reuses_raw_worker_citations(self) -> None:
        # Arrange: a persisted session at a checkpoint with one source, one evidence entry and a missing gap.
        session_id = "deep-research:deadbeef000000000000000000000001"
        stable_one = "cdeadbeef-1"
        stable_two = "cdeadbeef-2"
        checklist = [{"id": "s1", "title": "Verify both sources", "status": "active"}]
        old_source = {
            "id": stable_one,
            "citation_aliases": ["c0000-1"],
            "title": "Original source",
            "url": "https://example.com/original",
        }
        orchestrator.control.create_session(session_id, topic="Investigate the test claim")
        orchestrator.control.update_state(
            session_id,
            status="running",
            phase="checkpoint",
            plan="Verify both sources",
            plan_version=1,
            checklist=checklist,
            queries_used=1,
            query_budget=4,
            iteration=1,
            sources=[old_source],
            evidence_entries=[
                {
                    "iteration": 1,
                    "kind": "search",
                    "model_context": f"Original evidence [{stable_one}]",
                }
            ],
            seen_queries=["original evidence"],
            seen_urls=["https://example.com/original"],
            citation_next=2,
            initial_candidates=[],
            latest_assessment="The second source is still missing.",
            latest_gaps=["second source"],
            last_sequence=10,
        )

        # Scripted model replies for the resumed run: reflection, query selection, completing reflection and a
        # report that cites both the old and the new stable handles.
        outputs = iter(
            [
                (
                    '{"assessment":"The second source is still missing.",'
                    '"gaps":["second source"],"updates":[],"complete":false,'
                    '"candidates":[{"text":"second source evidence","vertical":"web",'
                    '"purpose":"fill the remaining gap"}]}'
                ),
                (
                    '{"queries":[{"text":"second source evidence","vertical":"web",'
                    '"purpose":"fill the remaining gap"}]}'
                ),
                (
                    '{"assessment":"Both sources are verified.","gaps":[],'
                    '"updates":[{"id":"s1","status":"done"}],"complete":true,'
                    '"candidates":[]}'
                ),
                f"Original claim [{stable_one}]. New claim [{stable_two}].",
            ]
        )
        generate = self.enterContext(
            mock.patch.object(
                orchestrator.llm_api,
                "generate",
                side_effect=lambda *_args, **_kwargs: _generation(next(outputs)),
            )
        )

        new_source = {
            # Simulate a restarted web-search process: its local counter begins at
            # c0000-1 again, but this URL is not the old source.
            "id": "c0000-1",
            "title": "New source",
            "url": "https://example.org/new",
        }

        # Fake tool call: search returns the colliding new source, any other tool returns its page text.
        def fake_tool(_lookup, alias, _arguments, context=None):
            del context
            if alias == "web_search__web_search":
                return {
                    "_tool_result_structured": {
                        "model_context": "New evidence [c0000-1]",
                        "sources": [new_source],
                    }
                }
            return {
                "_tool_result_structured": {
                    "model_context": "Full new-source page.",
                    "sources": [{"title": "New source", "url": "https://example.org/new"}],
                }
            }

        call_tool = self.enterContext(
            mock.patch.object(
                orchestrator.tool_registry,
                "call_ollama_tool",
                side_effect=fake_tool,
            )
        )

        # Act and assert the result: the run resumes with a reflection (not planning), both sources get
        # stable ids with the raw handle stored only as an alias, and the report links both sources.
        result = self._run(session_id, auto_approve=False, max_rounds=2)

        self.assertEqual(
            generate.call_args_list[0].args[2][0]["content"],
            orchestrator.REFLECTION_PROMPT,
        )
        self.assertNotIn(orchestrator.PLAN_PROMPT, [
            call.args[2][0]["content"] for call in generate.call_args_list
        ])
        self.assertEqual([source["id"] for source in result["sources"]], [stable_one, stable_two])
        self.assertEqual(result["sources"][1]["citation_aliases"], ["c0000-1"])
        self.assertIn("[Original source](https://example.com/original)", result["report"])
        self.assertIn("[New source](https://example.org/new)", result["report"])
        self.assertNotIn(f"[{stable_one}]", result["report"])
        self.assertEqual([item["id"] for item in result["citations"]], [stable_one, stable_two])

        # Assert the persisted terminal state: the citation counter advanced, three evidence entries exist
        # and they reference the stable handles.
        terminal_state = orchestrator.control.read_state(session_id)
        self.assertEqual(terminal_state["citation_next"], 3)
        self.assertEqual(len(terminal_state["evidence_entries"]), 3)
        self.assertIn(stable_one, terminal_state["evidence_entries"][0]["model_context"])
        self.assertIn(stable_two, terminal_state["evidence_entries"][1]["model_context"])
        self.assertIn("[New source](https://example.org/new)", terminal_state["report"])

        # A retry after the worker died post-completion replays the durable result and
        # must not invoke planning, search, or synthesis again.
        generate.reset_mock()
        call_tool.reset_mock()
        replayed = self._run(session_id, auto_approve=False, max_rounds=2)
        self.assertEqual(replayed["report"], result["report"])
        self.assertEqual(replayed["citations"], result["citations"])
        generate.assert_not_called()
        call_tool.assert_not_called()

    # Verify that running a session whose stored completed snapshot uses a legacy raw handle migrates the
    # stored state to a stable source id and a linked report, and returns it without any model or tool call.
    def test_legacy_terminal_snapshot_is_atomically_migrated_for_ui_polling(self) -> None:
        # Arrange: a completed legacy session whose report still cites the raw handle c0000-1.
        session_id = "deep-research:facefeed000000000000000000000001"
        legacy_source = {
            "id": "c0000-1",
            "title": "Legacy primary source",
            "url": "https://legacy.example/report",
        }
        orchestrator.control.create_session(session_id, topic="Investigate the test claim")
        orchestrator.control.update_state(
            session_id,
            status="completed",
            phase="completed",
            plan="Verify the legacy source",
            plan_version=1,
            checklist=[{"id": "s1", "title": "Verify source", "status": "done"}],
            report="Verified legacy claim [c0000-1].",
            model_context="Verified legacy claim [c0000-1].",
            sources=[legacy_source],
            source_count=1,
            queries_used=1,
            query_budget=4,
            iteration=1,
        )
        generate = self.enterContext(mock.patch.object(orchestrator.llm_api, "generate"))
        call_tool = self.enterContext(
            mock.patch.object(orchestrator.tool_registry, "call_ollama_tool")
        )

        # Act: run the session again and read back the migrated state.
        result = self._run(session_id, auto_approve=False, max_rounds=2)
        migrated = orchestrator.control.read_state(session_id)

        # Assert: the returned and stored report are linked, the source has a stable id with the legacy
        # handle as alias, and nothing was generated or searched.
        expected_report = (
            "Verified legacy claim "
            "[Legacy primary source](https://legacy.example/report)."
        )
        self.assertEqual(result["report"], expected_report)
        self.assertEqual(migrated["status"], "completed")
        self.assertEqual(migrated["phase"], "completed")
        self.assertEqual(migrated["report"], expected_report)
        self.assertEqual(migrated["model_context"], expected_report)
        self.assertEqual(migrated["sources"][0]["id"], "cfacefeed-1")
        self.assertEqual(migrated["sources"][0]["citation_aliases"], ["c0000-1"])
        self.assertEqual(migrated["citations"][0]["id"], "cfacefeed-1")
        self.assertEqual(migrated["citation_next"], 2)
        generate.assert_not_called()
        call_tool.assert_not_called()

    # Verify that a cancel command sent while the run awaits approval ends it as cancelled after the single
    # planning call, without any tool call.
    def test_cancel_command_stops_a_run_waiting_for_approval(self) -> None:
        # Arrange: the model returns only a plan and the search tool is mocked.
        session_id = "deep-research:10000000000000000000000000000003"
        generate = self.enterContext(
            mock.patch.object(
                orchestrator.llm_api,
                "generate",
                return_value=_generation(_successful_outputs()[0]),
            )
        )
        call_tool = self.enterContext(
            mock.patch.object(orchestrator.tool_registry, "call_ollama_tool")
        )

        # Act: wait for the approval prompt, then cancel.
        thread, outcome = self._start_run(session_id)
        self._wait_for_state(
            session_id,
            lambda state: state.get("status") == "awaiting_approval",
        )
        orchestrator.control.submit_command(
            session_id,
            "cancel",
            expected_plan_version=1,
        )
        result = self._join_successfully(thread, outcome)

        # Assert: both the result and the stored state are cancelled, with no further model or tool calls.
        self.assertEqual(result["ui"]["status"], "cancelled")
        self.assertEqual(
            orchestrator.control.read_state(session_id)["status"],
            "cancelled",
        )
        self.assertEqual(generate.call_count, 1)
        call_tool.assert_not_called()

    # Verify that when a revise and an approve command both target plan version 1, handling the commands
    # applies the revision (version 2), does not approve the outdated version and records a rejected command.
    def test_approval_for_previous_revision_is_not_replayed(self) -> None:
        # Arrange: a session awaiting approval of version 1 with a revise and an approve for version 1 queued.
        session_id = "deep-research:10000000000000000000000000000007"
        checklist = [{"id": "s1", "title": "Original goal", "status": "pending"}]
        orchestrator.control.create_session(
            session_id,
            topic="Investigate the test claim",
            status="awaiting_approval",
            extra={"plan_version": 1},
        )
        orchestrator.control.submit_command(
            session_id,
            "revise",
            payload={"plan": "- [ ] Revised goal"},
            expected_plan_version=1,
        )
        orchestrator.control.submit_command(
            session_id,
            "approve",
            expected_plan_version=1,
        )
        events = orchestrator.SemanticEventStream(
            session_id,
            callback=None,
            logs_dir=Path(self._log_directory.name),
        )
        self.addCleanup(events.close)

        # Act: process the queued commands.
        _plan, _checklist, version, approved = orchestrator._handle_commands(
            orchestrator.CommandInbox(session_id),
            topic="Investigate the test claim",
            plan="- [ ] Original goal",
            checklist=checklist,
            plan_version=1,
            events=events,
            iteration=0,
        )

        self.assertEqual(version, 2)
        self.assertFalse(approved)
        self.assertEqual(events.events[-1]["type"], "command_rejected")

    # Verify that when every model reply is malformed or empty, the run ends partial with a report and a
    # five-item checklist, makes seven model calls, selects no queries and performs no search.
    def test_malformed_model_outputs_do_not_trigger_fallback_search(self) -> None:
        # Arrange: seven malformed or empty replies and a search tool that must never be called.
        session_id = "deep-research:10000000000000000000000000000004"
        malformed_outputs = iter(
            [
                "{{ not a plan",
                "<malformed reflection>",
                "{not a selection",
                "still not a reflection",
                "[] is not the requested object",
                "",
                "",
            ]
        )
        generate = self.enterContext(
            mock.patch.object(
                orchestrator.llm_api,
                "generate",
                side_effect=lambda *_args, **_kwargs: _generation(next(malformed_outputs)),
            )
        )
        call_tool = self.enterContext(
            mock.patch.object(
                orchestrator.tool_registry,
                "call_ollama_tool",
                return_value="malformed non-JSON tool output",
            )
        )

        # Act and assert: a partial result, and no query was selected in any event.
        result = self._run(session_id, auto_approve=True)

        self.assertEqual(result["ui"]["status"], "partial")
        self.assertEqual(result["ui"]["source_count"], 0)
        self.assertTrue(result["report"])
        self.assertEqual(len(result["checklist"]), 5)
        self.assertEqual(generate.call_count, 7)
        call_tool.assert_not_called()
        selected_queries = [
            query
            for event in result["ui"]["events"]
            if event.get("type") == "queries_selected"
            for query in event.get("data", {}).get("queries", [])
        ]
        self.assertFalse(selected_queries)

    # Verify that a four-query selection over three verticals is capped to its first two queries, searched
    # with one call per vertical, that the final model call is the report synthesis, and that it receives
    # no tool arguments.
    def test_query_selection_is_capped_and_synthesis_is_separate_no_tools_call(self) -> None:
        # Arrange: a selection reply with four queries over three verticals.
        session_id = "deep-research:10000000000000000000000000000005"
        selection = (
            '{"queries":['
            '{"text":"query one","vertical":"web","purpose":"first"},'
            '{"text":"query two","vertical":"academic","purpose":"second"},'
            '{"text":"query three","vertical":"web","purpose":"third"},'
            '{"text":"query four","vertical":"shopping","purpose":"fourth"}'
            "]}"
        )
        outputs = iter(_successful_outputs(selection=selection))

        # Ordered log of model phases (by system prompt) and tool calls, used to check the call order.
        timeline: list[str] = []

        # Fake model call: record the phase inferred from the system prompt, then return the next scripted reply.
        def fake_generate(_engine, _model, messages, **_kwargs):
            system_prompt = messages[0]["content"]
            if system_prompt == orchestrator.REPORT_PROMPT:
                timeline.append("model:synthesis")
            elif system_prompt == orchestrator.QUERY_SELECTION_PROMPT:
                timeline.append("model:query_selection")
            elif system_prompt == orchestrator.REFLECTION_PROMPT:
                timeline.append("model:reflection")
            else:
                timeline.append("model:planning")
            return _generation(next(outputs))

        generate = self.enterContext(
            mock.patch.object(orchestrator.llm_api, "generate", side_effect=fake_generate)
        )

        search_calls: list[dict] = []

        # Fake tool call: record the call in the timeline, check the search argument shape and collect
        # the search arguments.
        def fake_tool(_lookup, alias, arguments, context=None):
            del context
            timeline.append(f"tool:{alias}")
            if alias == "web_search__web_search":
                # Mixed web/academic selection: one simple-shape call per vertical.
                self.assertEqual(
                    set(arguments),
                    {"query", "vertical", "effort", "description"},
                )
                search_calls.append(arguments)
            return _source_result()

        call_tool = self.enterContext(
            mock.patch.object(
                orchestrator.tool_registry,
                "call_ollama_tool",
                side_effect=fake_tool,
            )
        )

        # Act and assert: two queries were used, synthesis is the last model call and comes after the search.
        result = self._run(session_id, auto_approve=True)

        self.assertEqual(result["ui"]["queries_used"], 2)
        self.assertEqual(len(search_calls), 2)
        self.assertEqual(
            {(call["vertical"], call["query"]) for call in search_calls},
            {("web", "query one"), ("academic", "query two")},
        )
        self.assertEqual(timeline[-1], "model:synthesis")
        self.assertIn("tool:web_search__web_search", timeline[:-1])

        # The synthesis call uses the report prompt and none of the tool-related arguments.
        final_call = generate.call_args_list[-1]
        self.assertEqual(final_call.args[2][0]["content"], orchestrator.REPORT_PROMPT)
        for forbidden_tool_argument in (
            "tools",
            "tool_server_ids",
            "allowed_tool_aliases",
            "required_tool_aliases",
            "forced_tool_name",
        ):
            self.assertNotIn(forbidden_tool_argument, final_call.kwargs)
        self.assertEqual(result["report"], "Verified final report [c1000-1]")


# Allow running the tests without pytest.
if __name__ == "__main__":
    unittest.main()
