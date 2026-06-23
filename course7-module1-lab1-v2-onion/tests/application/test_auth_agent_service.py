from unittest.mock import MagicMock

import pytest
from langgraph.graph.state import CompiledStateGraph

from application.auth_agent_service import AuthAgentService
from domain.auth_credentials import AuthCredentials
from interfaces.stream_consumer_interface import StreamConsumerInterface


def _make_graph(stream_chunks: list[tuple[str, dict]]) -> MagicMock:
    graph = MagicMock(spec=CompiledStateGraph)
    graph.stream.return_value = iter(stream_chunks)
    return graph


def test_run_returns_credentials_from_final_values_chunk() -> None:
    final_creds = AuthCredentials(
        username="test_user", is_authenticated=True, message="Welcome."
    )
    graph = _make_graph([
        ("updates", {"InputNode": {"credentials": AuthCredentials(username="test_user")}}),
        ("values", {"credentials": final_creds}),
    ])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = AuthAgentService(graph, consumer)

    result = service.run()

    assert result is final_creds


def test_run_forwards_updates_to_stream_consumer() -> None:
    graph = _make_graph([
        ("updates", {"InputNode": {"credentials": AuthCredentials(username="test_user")}}),
        ("updates", {"ValidateCredential": {"credentials": AuthCredentials(username="test_user", is_authenticated=True)}}),
        ("values", {"credentials": AuthCredentials(username="test_user", is_authenticated=True)}),
    ])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = AuthAgentService(graph, consumer)

    service.run()

    assert consumer.on_update.call_count == 2
    assert consumer.on_update.call_args_list[0][0][0] == "InputNode"
    assert consumer.on_update.call_args_list[1][0][0] == "ValidateCredential"


def test_run_passes_initial_state_to_stream() -> None:
    initial = AuthCredentials(username="test_user")
    graph = _make_graph([("values", {"credentials": initial})])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = AuthAgentService(graph, consumer)

    service.run({"credentials": initial})

    call_kwargs = graph.stream.call_args.kwargs
    call_args = graph.stream.call_args.args
    passed_state = call_args[0]
    assert passed_state == {"credentials": initial}
    assert call_kwargs["stream_mode"] == ["updates", "values"]


def test_run_raises_when_final_state_has_no_credentials() -> None:
    """Defensive — the graph topology should always produce credentials.
    Pinning that violation surfaces as a RuntimeError, not a silent None."""
    graph = _make_graph([("values", {})])
    consumer = MagicMock(spec=StreamConsumerInterface)
    service = AuthAgentService(graph, consumer)

    with pytest.raises(RuntimeError, match="graph topology is broken"):
        service.run()