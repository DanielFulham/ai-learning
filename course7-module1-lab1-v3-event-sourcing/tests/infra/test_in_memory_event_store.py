from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from domain.events.qa_events import (
    AnswerGenerated,
    ContextRetrieved,
    QuestionReceived,
)
from infra.in_memory_event_store import InMemoryEventStore
from interfaces.agent_event_store_interface import AgentEventStoreInterface


def _common_kwargs(aggregate_id: UUID) -> dict[str, Any]:
    return {
        "event_id": uuid4(),
        "aggregate_id": aggregate_id,
        "occurred_at": datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc),
    }


def _accepts_event_store(store: AgentEventStoreInterface) -> None:
    """Type-guard helper. Pyright fails the call site if the argument
    doesn't satisfy the Protocol — the runtime check is a no-op."""


class TestInMemoryEventStoreInterfaceSatisfaction:

    def test_satisfies_agent_event_store_interface(self) -> None:
        """Pinned: InMemoryEventStore satisfies AgentEventStoreInterface
        structurally. If the interface gains a method or the concrete loses
        one, pyright fails this call site at static-check time."""
        store = InMemoryEventStore()
        _accepts_event_store(store)


class TestInMemoryEventStoreAppendAndRead:

    def test_append_then_replay_single_event(self) -> None:
        store = InMemoryEventStore()
        run_id = uuid4()
        event = QuestionReceived(**_common_kwargs(run_id), question="What?")

        store.append(event)
        result = store.events_for_run(run_id)

        assert result == [event]

    def test_replay_preserves_append_order(self) -> None:
        """Pinned: append order is the canonical order on replay. Projections
        and composition assertions rely on this — 'the first event was a
        QuestionReceived' is meaningless if order isn't preserved."""
        store = InMemoryEventStore()
        run_id = uuid4()
        first = QuestionReceived(**_common_kwargs(run_id), question="Q")
        second = ContextRetrieved(**_common_kwargs(run_id), context="C")
        third = AnswerGenerated(**_common_kwargs(run_id), answer="A")

        store.append(first)
        store.append(second)
        store.append(third)

        result = store.events_for_run(run_id)
        assert result == [first, second, third]

    def test_replay_of_unknown_run_returns_empty_list(self) -> None:
        """Pinned: unknown run_id returns empty list, not None and not
        an exception. Callers can iterate the result without a presence
        check; projections degrade to empty output for unknown runs."""
        store = InMemoryEventStore()
        result = store.events_for_run(uuid4())
        assert result == []

    def test_runs_are_isolated_by_aggregate_id(self) -> None:
        """Pinned: events for one run never leak into another's replay."""
        store = InMemoryEventStore()
        run_a = uuid4()
        run_b = uuid4()
        event_a = QuestionReceived(**_common_kwargs(run_a), question="QA")
        event_b = QuestionReceived(**_common_kwargs(run_b), question="QB")

        store.append(event_a)
        store.append(event_b)

        assert store.events_for_run(run_a) == [event_a]
        assert store.events_for_run(run_b) == [event_b]


class TestInMemoryEventStoreDefensiveCopy:

    def test_returned_list_is_fresh_per_call(self) -> None:
        """Pinned: events_for_run returns a fresh list each call. Mutating
        the returned list does not affect the store, and the store does
        not return the same list reference on repeat calls. Combined with
        frozen events, callers see immutable replay output."""
        store = InMemoryEventStore()
        run_id = uuid4()
        event = QuestionReceived(**_common_kwargs(run_id), question="Q")
        store.append(event)

        first_result = store.events_for_run(run_id)
        second_result = store.events_for_run(run_id)

        # Different list objects, same contents
        assert first_result is not second_result
        assert first_result == second_result

    def test_mutating_returned_list_does_not_affect_store(self) -> None:
        store = InMemoryEventStore()
        run_id = uuid4()
        event = QuestionReceived(**_common_kwargs(run_id), question="Q")
        store.append(event)

        result = store.events_for_run(run_id)
        result.clear()

        # Store still has the event
        assert store.events_for_run(run_id) == [event]