"""Event-log composition tests for integrated flows (F15).

Cross-cutting by design: these tests assert full event-log sequences for
flows composed through the container, rather than mirroring a single
source file. The QA service's unit shape is covered by
test_qa_agent_service.py; this file covers the composition concern — one
container, real event store, sequences read back via events_for_run.

The graph-mock helpers below are duplicated inline (not imported across
test files): the bare `_mock_graph` mirrors test_container_auth_wiring.py
for a graph seam that is filled but never streamed, and the configured
`_make_graph_yielding` mirrors test_qa_agent_service.py for a graph that
is exercised. Cross-test-file imports would couple unrelated test files;
the duplication is two helpers and trivial.
"""
from typing import Any
from unittest.mock import MagicMock

from langgraph.graph.state import CompiledStateGraph

from application.auth_agent_service import AuthAgentService
from application.container import initialise
from application.qa_agent_service import QAAgentService
from domain.events.auth_events import LoginAttempted, LoginSucceeded
from domain.events.qa_events import (
    AnswerGenerated,
    ContextRetrieved,
    QuestionReceived,
)
from domain.qa_exchange import QAExchange

from _helpers.scripted_input_provider import ScriptedInputProvider


def _mock_graph() -> MagicMock:
    return MagicMock(spec=CompiledStateGraph)


def _make_graph_yielding(*chunks: tuple[str, Any]) -> MagicMock:
    """Build a mocked CompiledStateGraph whose `.stream(...)` yields the
    given (mode, chunk) tuples."""
    graph = MagicMock(spec=CompiledStateGraph)
    graph.stream.return_value = iter(chunks)
    return graph


class TestContainerQAFlowEndToEnd:

    def test_qa_flow_appends_expected_events(self) -> None:
        """Proves QA composition through the container: a configured QA
        graph mock drives the happy path, and the run's event log carries
        the full three-event QA sequence read back via events_for_run. The
        auth slot is filled with an empty scripted provider that is never
        invoked — auth does not run in this test."""
        context_exchange = QAExchange(
            question="What is the capital?",
            context="Paris is the capital of France.",
        )
        answer_exchange = QAExchange(
            question="What is the capital?",
            context="Paris is the capital of France.",
            answer="Paris.",
        )
        qa_graph = _make_graph_yielding(
            ("updates", {"ContextNode": {"exchange": context_exchange}}),
            ("updates", {"QANode": {"exchange": answer_exchange}}),
            ("values", {"exchange": answer_exchange}),
        )
        app = initialise(
            qa_graph=qa_graph,
            input_provider=ScriptedInputProvider([]),
            use_console_consumer=False,
        )

        result = app.qa.run("What is the capital?")

        events = app.event_store.events_for_run(result.run_id)
        assert [type(e).__name__ for e in events] == [
            "QuestionReceived",
            "ContextRetrieved",
            "AnswerGenerated",
        ]
        assert isinstance(events[0], QuestionReceived)
        assert isinstance(events[1], ContextRetrieved)
        assert isinstance(events[2], AnswerGenerated)
        assert events[0].question == "What is the capital?"
        assert events[1].context == "Paris is the capital of France."
        assert events[2].answer == "Paris."


class TestContainerEventLogComposition:

    def test_auth_then_qa_runs_produce_independent_run_scoped_sequences(
        self,
    ) -> None:
        """The F15 headline: one container, an auth run and a QA run, two
        independent run_id-scoped sequences in the shared store. The auth
        side runs the real graph (happy path, two-event sequence); the QA
        side runs the configured mock (three-event sequence). Each run's log
        is read back by its own run_id and asserted independently; the
        run_ids differ (per-run, not per-service); and the singleton store is
        the same instance both services hold — the F17 contract restated at
        the composition read-path level."""
        context_exchange = QAExchange(
            question="What is the capital?",
            context="Paris is the capital of France.",
        )
        answer_exchange = QAExchange(
            question="What is the capital?",
            context="Paris is the capital of France.",
            answer="Paris.",
        )
        qa_graph = _make_graph_yielding(
            ("updates", {"ContextNode": {"exchange": context_exchange}}),
            ("updates", {"QANode": {"exchange": answer_exchange}}),
            ("values", {"exchange": answer_exchange}),
        )
        app = initialise(
            qa_graph=qa_graph,
            input_provider=ScriptedInputProvider(["test_user", "secure_password"]),
            use_console_consumer=False,
        )

        auth_result = app.auth.run()
        qa_result = app.qa.run("What is the capital?")

        auth_events = app.event_store.events_for_run(auth_result.run_id)
        assert [type(e).__name__ for e in auth_events] == [
            "LoginAttempted",
            "LoginSucceeded",
        ]
        assert isinstance(auth_events[0], LoginAttempted)
        assert isinstance(auth_events[1], LoginSucceeded)
        assert auth_events[0].username == "test_user"

        qa_events = app.event_store.events_for_run(qa_result.run_id)
        assert [type(e).__name__ for e in qa_events] == [
            "QuestionReceived",
            "ContextRetrieved",
            "AnswerGenerated",
        ]
        assert isinstance(qa_events[0], QuestionReceived)
        assert isinstance(qa_events[1], ContextRetrieved)
        assert isinstance(qa_events[2], AnswerGenerated)
        assert qa_events[0].question == "What is the capital?"

        assert auth_result.run_id != qa_result.run_id
        assert isinstance(app.qa, QAAgentService)
        assert isinstance(app.auth, AuthAgentService)
        assert app.event_store is app.qa._event_store
        assert app.event_store is app.auth._event_store
        assert app.qa._event_store is app.auth._event_store
