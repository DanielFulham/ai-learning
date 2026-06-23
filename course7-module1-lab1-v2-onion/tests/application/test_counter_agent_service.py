from unittest.mock import MagicMock

import pytest
from langgraph.graph.state import CompiledStateGraph

from application.counter_agent_service import CounterAgentService
from domain.counter_tick import CounterTick
from interfaces.stream_consumer_interface import StreamConsumerInterface


def _make_graph(stream_chunks: list[tuple[str, dict]]) -> MagicMock:
    graph = MagicMock(spec=CompiledStateGraph)
    graph.stream.return_value = iter(stream_chunks)
    return graph


def test_run_returns_final_tick() -> None:
    final_tick = CounterTick(n=13, letter="z")
    graph = _make_graph([("values", {"tick": final_tick})])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = CounterAgentService(graph, consumer)

    result = service.run()

    assert result is final_tick


def test_run_passes_empty_initial_state() -> None:
    graph = _make_graph([("values", {"tick": CounterTick(n=13, letter="a")})])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = CounterAgentService(graph, consumer)

    service.run()

    assert graph.stream.call_args.args[0] == {}


def test_run_forwards_all_tick_updates_to_consumer() -> None:
    """Counter loops AddNode 13 times — the consumer sees 13 update entries."""
    chunks: list[tuple[str, dict]] = [
        ("updates", {"AddNode": {"tick": CounterTick(n=i, letter="a")}})
        for i in range(1, 14)
    ]
    chunks.append(("values", {"tick": CounterTick(n=13, letter="a")}))
    graph = _make_graph(chunks)
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = CounterAgentService(graph, consumer)

    service.run()

    assert consumer.on_update.call_count == 13


def test_run_raises_when_final_state_has_no_tick() -> None:
    graph = _make_graph([("values", {})])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = CounterAgentService(graph, consumer)

    with pytest.raises(RuntimeError, match="graph topology is broken"):
        service.run()


def test_run_uses_dual_mode_streaming() -> None:
    """Pins finding from V2 lab note — services subscribe to both updates
    (for the consumer) and values (for final-state extraction)."""
    graph = _make_graph([("values", {"tick": CounterTick(n=13, letter="a")})])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = CounterAgentService(graph, consumer)

    service.run()

    assert graph.stream.call_args.kwargs["stream_mode"] == ["updates", "values"]