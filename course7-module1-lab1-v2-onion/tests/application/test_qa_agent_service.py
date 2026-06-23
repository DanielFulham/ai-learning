from unittest.mock import MagicMock

import pytest
from langgraph.graph.state import CompiledStateGraph

from application.qa_agent_service import QAAgentService
from domain.qa_exchange import QAExchange
from interfaces.stream_consumer_interface import StreamConsumerInterface


def _make_graph(stream_chunks: list[tuple[str, dict]]) -> MagicMock:
    graph = MagicMock(spec=CompiledStateGraph)
    graph.stream.return_value = iter(stream_chunks)
    return graph


def test_run_constructs_exchange_from_question_and_returns_final() -> None:
    final_exchange = QAExchange(
        question="What is LangGraph?",
        context="some context",
        answer="A state machine framework.",
    )
    graph = _make_graph([("values", {"exchange": final_exchange})])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = QAAgentService(graph, consumer)

    result = service.run("What is LangGraph?")

    assert result is final_exchange
    passed_state = graph.stream.call_args.args[0]
    assert passed_state["exchange"].question == "What is LangGraph?"


def test_run_raises_on_whitespace_question_via_domain_invariant() -> None:
    """The domain layer rejects whitespace-only questions before the graph
    ever runs. Pinned here so the bypass-the-domain temptation surfaces
    as a test failure."""
    graph = _make_graph([])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = QAAgentService(graph, consumer)

    with pytest.raises(ValueError, match="non-empty"):
        service.run("   ")


def test_run_forwards_updates_to_consumer() -> None:
    graph = _make_graph([
        ("updates", {"ContextNode": {"exchange": QAExchange(question="X")}}),
        ("values", {"exchange": QAExchange(question="X", answer="A")}),
    ])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = QAAgentService(graph, consumer)

    service.run("X")

    consumer.on_update.assert_called_once()
    assert consumer.on_update.call_args[0][0] == "ContextNode"


def test_run_raises_when_final_state_has_no_exchange() -> None:
    graph = _make_graph([("values", {})])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = QAAgentService(graph, consumer)

    with pytest.raises(RuntimeError, match="graph topology is broken"):
        service.run("anything")


def test_run_uses_dual_mode_streaming() -> None:
    """Pins finding from V2 lab note — services subscribe to both updates
    (for the consumer) and values (for final-state extraction)."""
    graph = _make_graph([("values", {"exchange": QAExchange(question="x", answer="a")})])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = QAAgentService(graph, consumer)

    service.run("any question")

    assert graph.stream.call_args.kwargs["stream_mode"] == ["updates", "values"]