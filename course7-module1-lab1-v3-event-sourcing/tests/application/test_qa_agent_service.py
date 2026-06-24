from datetime import datetime, timezone
from typing import Any, Callable
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from langgraph.graph.state import CompiledStateGraph

from application.qa_agent_service import QAAgentService
from application.interfaces.qa_agent_service_interface import (
    QAAgentServiceInterface,
)
from domain.error_info import ErrorInfo
from domain.events.qa_events import (
    AnswerGenerated,
    ContextRetrieved,
    ModelInvocationFailed,
    QuestionReceived,
)
from domain.qa_exchange import QAExchange
from interfaces.agent_event_store_interface import AgentEventStoreInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


_FIXED_TIME = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)


def _fixed_clock() -> Callable[[], datetime]:
    return lambda: _FIXED_TIME


def _make_graph_yielding(*chunks: tuple[str, Any]) -> MagicMock:
    """Build a mocked CompiledStateGraph whose `.stream(...)` yields the
    given (mode, chunk) tuples."""
    graph = MagicMock(spec=CompiledStateGraph)
    graph.stream.return_value = iter(chunks)
    return graph


def _make_service(
    graph: MagicMock | None = None,
    event_store: AgentEventStoreInterface | None = None,
    inner_consumer: StreamConsumerInterface | None = None,
    clock: Callable[[], datetime] | None = None,
) -> QAAgentService:
    return QAAgentService(
        graph=graph or _make_graph_yielding(),
        event_store=event_store or MagicMock(spec=AgentEventStoreInterface),
        inner_consumer=inner_consumer or MagicMock(spec=StreamConsumerInterface),
        clock=clock or _fixed_clock(),
    )


def _appended_events(store: MagicMock) -> list[Any]:
    return [call.args[0] for call in store.append.call_args_list]


def _accepts_qa_service(service: QAAgentServiceInterface) -> None:
    """Type-guard helper."""


class TestQAAgentServiceInterfaceSatisfaction:

    def test_satisfies_qa_agent_service_interface(self) -> None:
        _accepts_qa_service(_make_service())


class TestQAAgentServiceQuestionReceived:

    def test_fires_question_received_before_graph_stream(self) -> None:
        """Pinned: QuestionReceived is appended to the store before
        graph.stream() runs. The run-lifecycle event is the first thing
        in the log, not interleaved with node updates."""
        store = MagicMock(spec=AgentEventStoreInterface)
        exchange = QAExchange(question="Q", answer="A")
        graph = _make_graph_yielding(("values", {"exchange": exchange}))

        service = _make_service(graph=graph, event_store=store)
        service.run("Q")

        events = _appended_events(store)
        assert len(events) >= 1
        assert isinstance(events[0], QuestionReceived)
        assert events[0].question == "Q"
        assert events[0].occurred_at == _FIXED_TIME

    def test_question_received_carries_the_run_id(self) -> None:
        """Pinned: QuestionReceived's aggregate_id is the same run_id
        the translating consumer uses for its events. All events for
        one run share the aggregate_id."""
        store = MagicMock(spec=AgentEventStoreInterface)
        exchange = QAExchange(question="Q", answer="A")
        graph = _make_graph_yielding(
            ("updates", {"QANode": {"exchange": exchange}}),
            ("values", {"exchange": exchange}),
        )

        service = _make_service(graph=graph, event_store=store)
        service.run("Q")

        events = _appended_events(store)
        assert isinstance(events[0], QuestionReceived)
        assert isinstance(events[1], AnswerGenerated)
        assert events[0].aggregate_id == events[1].aggregate_id


class TestQAAgentServiceRunIdReturn:

    def test_returned_run_id_matches_event_aggregate_id(self) -> None:
        """Pinned: the run_id on the returned RunResult is the same one
        used to tag every event for the run. Callers can use it for
        replay-by-run and projection summarising."""
        store = MagicMock(spec=AgentEventStoreInterface)
        exchange = QAExchange(question="Q", answer="A")
        graph = _make_graph_yielding(("values", {"exchange": exchange}))

        service = _make_service(graph=graph, event_store=store)
        result = service.run("Q")

        events = _appended_events(store)
        assert isinstance(events[0], QuestionReceived)
        assert result.run_id == events[0].aggregate_id

    def test_returned_run_id_is_a_uuid(self) -> None:
        exchange = QAExchange(question="Q", answer="A")
        graph = _make_graph_yielding(("values", {"exchange": exchange}))

        result = _make_service(graph=graph).run("Q")
        assert isinstance(result.run_id, UUID)


class TestQAAgentServiceRunIdGeneration:

    def test_consecutive_runs_have_different_run_ids(self) -> None:
        """Pinned: each .run() generates a fresh run_id. The translating
        consumer is per-run; a service instance can be reused across many
        runs without leaking state between them."""
        store = MagicMock(spec=AgentEventStoreInterface)
        exchange = QAExchange(question="Q", answer="A")

        graph = MagicMock(spec=CompiledStateGraph)
        graph.stream.side_effect = [
            iter([("values", {"exchange": exchange})]),
            iter([("values", {"exchange": exchange})]),
        ]

        service = _make_service(graph=graph, event_store=store)
        first = service.run("Q1")
        second = service.run("Q2")

        assert first.run_id != second.run_id


class TestQAAgentServiceUpdateDispatch:

    def test_node_updates_produce_translated_events_in_store(self) -> None:
        """Pinned: end-to-end through the translating consumer. ContextNode
        update produces ContextRetrieved; QANode update produces
        AnswerGenerated. All events ordered as the stream yielded them,
        preceded by QuestionReceived."""
        store = MagicMock(spec=AgentEventStoreInterface)
        context_exchange = QAExchange(question="Q", context="C")
        answer_exchange = QAExchange(question="Q", context="C", answer="A")
        graph = _make_graph_yielding(
            ("updates", {"ContextNode": {"exchange": context_exchange}}),
            ("updates", {"QANode": {"exchange": answer_exchange}}),
            ("values", {"exchange": answer_exchange}),
        )

        service = _make_service(graph=graph, event_store=store)
        service.run("Q")

        events = _appended_events(store)
        assert len(events) == 3
        assert isinstance(events[0], QuestionReceived)
        assert isinstance(events[1], ContextRetrieved)
        assert isinstance(events[2], AnswerGenerated)
        assert events[1].context == "C"
        assert events[2].answer == "A"

    def test_qa_node_with_error_info_produces_model_invocation_failed(self) -> None:
        store = MagicMock(spec=AgentEventStoreInterface)
        info = ErrorInfo(exception_type="X", exception_message="Y")
        failed_exchange = QAExchange(
            question="Q",
            context="C",
            answer="user-safe message",
            error_info=info,
        )
        graph = _make_graph_yielding(
            ("updates", {"QANode": {"exchange": failed_exchange}}),
            ("values", {"exchange": failed_exchange}),
        )

        service = _make_service(graph=graph, event_store=store)
        service.run("Q")

        events = _appended_events(store)
        assert isinstance(events[1], ModelInvocationFailed)
        assert events[1].error_info == info

    def test_inner_consumer_receives_every_node_update(self) -> None:
        """Pinned: the translating consumer wraps the inner; the inner
        sees every node update unchanged. Dev visibility is preserved."""
        inner = MagicMock(spec=StreamConsumerInterface)
        context_exchange = QAExchange(question="Q", context="C")
        answer_exchange = QAExchange(question="Q", context="C", answer="A")
        graph = _make_graph_yielding(
            ("updates", {"ContextNode": {"exchange": context_exchange}}),
            ("updates", {"QANode": {"exchange": answer_exchange}}),
            ("values", {"exchange": answer_exchange}),
        )

        service = _make_service(graph=graph, inner_consumer=inner)
        service.run("Q")

        assert inner.on_update.call_count == 2
        first_call_args = inner.on_update.call_args_list[0].args
        second_call_args = inner.on_update.call_args_list[1].args
        assert first_call_args[0] == "ContextNode"
        assert second_call_args[0] == "QANode"


class TestQAAgentServiceReturn:

    def test_returns_run_result_with_final_exchange(self) -> None:
        final = QAExchange(question="Q", context="C", answer="A")
        graph = _make_graph_yielding(("values", {"exchange": final}))

        service = _make_service(graph=graph)
        result = service.run("Q")

        assert result.exchange == final

    def test_raises_when_no_values_chunk_yields_exchange(self) -> None:
        """Pinned: graph topology that completes without producing an
        exchange is a bug, not a runtime condition. Raises loudly."""
        graph = _make_graph_yielding(
            ("updates", {"ContextNode": {"exchange": QAExchange(question="Q")}}),
        )

        service = _make_service(graph=graph)
        with pytest.raises(RuntimeError, match="graph topology is broken"):
            service.run("Q")