from datetime import datetime, timezone
from typing import Any, Callable
from unittest.mock import MagicMock
from uuid import UUID

import pytest
from langgraph.graph.state import CompiledStateGraph

from application.auth_agent_service import AuthAgentService
from application.interfaces.auth_agent_service_interface import (
    AuthAgentServiceInterface,
)
from domain.auth_credentials import AuthCredentials
from domain.auth_result import AuthResult
from domain.events.auth_events import LoginAttempted, LoginFailed, LoginSucceeded
from interfaces.agent_event_store_interface import AgentEventStoreInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


_FIXED_TIME = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)

_VALID_USERNAME = "test_user"


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
) -> AuthAgentService:
    return AuthAgentService(
        graph=graph or _make_graph_yielding(),
        event_store=event_store or MagicMock(spec=AgentEventStoreInterface),
        inner_consumer=inner_consumer or MagicMock(spec=StreamConsumerInterface),
        clock=clock or _fixed_clock(),
    )


def _appended_events(store: MagicMock) -> list[Any]:
    return [call.args[0] for call in store.append.call_args_list]


def _accepts_auth_service(service: AuthAgentServiceInterface) -> None:
    """Type-guard helper."""


def _input_update(username: str = _VALID_USERNAME) -> tuple[str, Any]:
    return (
        "updates",
        {"InputNode": {"credentials": AuthCredentials(username=username)}},
    )


def _validate_update(
    is_authenticated: bool, username: str = _VALID_USERNAME
) -> tuple[str, Any]:
    return (
        "updates",
        {
            "ValidateCredentialsNode": {
                "credentials": AuthCredentials(
                    username=username, is_authenticated=is_authenticated
                )
            }
        },
    )


def _final_values(credentials: AuthCredentials) -> tuple[str, Any]:
    return ("values", {"credentials": credentials})


def _success_credentials() -> AuthCredentials:
    return AuthCredentials(
        username=_VALID_USERNAME, is_authenticated=True, message="ok"
    )


class TestAuthAgentServiceInterfaceSatisfaction:

    def test_satisfies_auth_agent_service_interface(self) -> None:
        _accepts_auth_service(_make_service())


class TestAuthAgentServiceInitialState:

    def test_seeds_initial_state_with_empty_credentials(self) -> None:
        """Pinned: the service seeds {"credentials": AuthCredentials()} as
        the initial state before streaming. input_node requires the carrier
        present; without this seed it raises in production."""
        graph = _make_graph_yielding(_final_values(_success_credentials()))

        service = _make_service(graph=graph)
        service.run()

        graph.stream.assert_called_once()
        initial_state = graph.stream.call_args.args[0]
        assert initial_state == {"credentials": AuthCredentials()}


class TestAuthAgentServiceUpdateDispatch:

    def test_input_then_success_produces_attempted_then_succeeded(self) -> None:
        """Pinned: end-to-end through the translating consumer. InputNode
        update produces LoginAttempted; ValidateCredentialsNode (verdict
        True) produces LoginSucceeded. No direct lifecycle event precedes
        them — auth's first event is translator-driven."""
        store = MagicMock(spec=AgentEventStoreInterface)
        graph = _make_graph_yielding(
            _input_update(),
            _validate_update(True),
            _final_values(_success_credentials()),
        )

        service = _make_service(graph=graph, event_store=store)
        service.run()

        events = _appended_events(store)
        assert len(events) == 2
        assert isinstance(events[0], LoginAttempted)
        assert isinstance(events[1], LoginSucceeded)
        assert events[0].username == _VALID_USERNAME

    def test_validate_failure_produces_login_failed(self) -> None:
        """Pinned: the translator's failure branch through the service —
        ValidateCredentialsNode with verdict False produces LoginFailed."""
        store = MagicMock(spec=AgentEventStoreInterface)
        failed = AuthCredentials(
            username=_VALID_USERNAME, is_authenticated=False, message="no"
        )
        graph = _make_graph_yielding(
            _input_update(),
            _validate_update(False),
            _final_values(failed),
        )

        service = _make_service(graph=graph, event_store=store)
        service.run()

        events = _appended_events(store)
        assert isinstance(events[-1], LoginFailed)


class TestAuthAgentServiceInnerConsumer:

    def test_inner_consumer_receives_every_node_update(self) -> None:
        """Pinned: the translating consumer wraps the inner; the inner sees
        every node update unchanged. Dev visibility is preserved."""
        inner = MagicMock(spec=StreamConsumerInterface)
        graph = _make_graph_yielding(
            _input_update(),
            _validate_update(True),
            _final_values(_success_credentials()),
        )

        service = _make_service(graph=graph, inner_consumer=inner)
        service.run()

        assert inner.on_update.call_count == 2
        assert inner.on_update.call_args_list[0].args[0] == "InputNode"
        assert (
            inner.on_update.call_args_list[1].args[0] == "ValidateCredentialsNode"
        )


class TestAuthAgentServiceClock:

    def test_translated_event_uses_injected_clock(self) -> None:
        """Pinned: the clock reaches events via the per-run consumer/
        translator — the service appends nothing directly, so occurred_at
        on a translated event is the injected clock's value."""
        store = MagicMock(spec=AgentEventStoreInterface)
        graph = _make_graph_yielding(
            _input_update(),
            _final_values(_success_credentials()),
        )

        service = _make_service(graph=graph, event_store=store, clock=_fixed_clock())
        service.run()

        events = _appended_events(store)
        assert events[0].occurred_at == _FIXED_TIME


class TestAuthAgentServiceRunId:

    def test_returned_run_id_matches_event_aggregate_id(self) -> None:
        """Pinned: the run_id on the returned AuthResult is the same one
        used to tag every event for the run."""
        store = MagicMock(spec=AgentEventStoreInterface)
        graph = _make_graph_yielding(
            _input_update(),
            _final_values(_success_credentials()),
        )

        service = _make_service(graph=graph, event_store=store)
        result = service.run()

        events = _appended_events(store)
        assert isinstance(result.run_id, UUID)
        assert events[0].aggregate_id == result.run_id

    def test_consecutive_runs_have_different_run_ids(self) -> None:
        """Pinned: each .run() generates a fresh run_id. The service can be
        reused across runs without leaking state."""
        graph = MagicMock(spec=CompiledStateGraph)
        graph.stream.side_effect = [
            iter([_final_values(_success_credentials())]),
            iter([_final_values(_success_credentials())]),
        ]

        service = _make_service(graph=graph)
        first = service.run()
        second = service.run()

        assert first.run_id != second.run_id


class TestAuthAgentServiceReturn:

    def test_returns_auth_result_with_final_credentials(self) -> None:
        final = _success_credentials()
        graph = _make_graph_yielding(_final_values(final))

        service = _make_service(graph=graph)
        result = service.run()

        assert isinstance(result, AuthResult)
        assert result.credentials == final

    def test_raises_when_no_values_chunk_yields_credentials(self) -> None:
        """Pinned: a graph that completes without producing a credentials
        values-chunk is a broken-topology bug, not a runtime condition.
        Raises loudly. Mirrors QA's no-exchange guard."""
        graph = _make_graph_yielding(_input_update())

        service = _make_service(graph=graph)
        with pytest.raises(RuntimeError, match="graph topology is broken"):
            service.run()
