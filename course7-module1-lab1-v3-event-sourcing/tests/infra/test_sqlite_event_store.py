from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest

from domain.error_info import ErrorInfo
from domain.events.base import BaseAgentEvent
from domain.events.qa_events import (
    AnswerGenerated,
    ContextRetrieved,
    ModelInvocationFailed,
    QuestionReceived,
)
from infra.sqlite_event_store import SqliteEventStore
from interfaces.agent_event_store_interface import AgentEventStoreInterface


_QA_EVENT_TYPES: list[type[BaseAgentEvent]] = [
    QuestionReceived,
    ContextRetrieved,
    AnswerGenerated,
    ModelInvocationFailed,
]


def _common_kwargs(aggregate_id: UUID) -> dict[str, Any]:
    return {
        "event_id": uuid4(),
        "aggregate_id": aggregate_id,
        "occurred_at": datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc),
    }


def _make_store(tmp_path: Path) -> SqliteEventStore:
    return SqliteEventStore(tmp_path / "events.db", _QA_EVENT_TYPES)


def _accepts_event_store(store: AgentEventStoreInterface) -> None:
    """Type-guard helper. Pyright fails the call site if structural
    typing breaks."""


class TestSqliteEventStoreInterfaceSatisfaction:

    def test_satisfies_agent_event_store_interface(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        _accepts_event_store(store)


class TestSqliteEventStoreSchema:

    def test_schema_initialised_on_construction(self, tmp_path: Path) -> None:
        """Pinned: schema is created when the store is constructed.
        Subsequent appends do not need to check for schema existence;
        the table is guaranteed present."""
        db_path = tmp_path / "events.db"
        SqliteEventStore(db_path, _QA_EVENT_TYPES)
        assert db_path.exists()

    def test_construction_is_idempotent(self, tmp_path: Path) -> None:
        """Pinned: constructing the store twice against the same file
        does not error. CREATE TABLE IF NOT EXISTS handles re-init."""
        db_path = tmp_path / "events.db"
        SqliteEventStore(db_path, _QA_EVENT_TYPES)
        SqliteEventStore(db_path, _QA_EVENT_TYPES)


class TestSqliteEventStoreRoundTrip:
    """One round-trip per event type. Each event has a distinct payload
    shape — primitive string, nullable string, nested dataclass — that
    exercises a different branch of the serialisation logic.
    """

    def test_question_received_roundtrip(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        run_id = uuid4()
        event = QuestionReceived(
            **_common_kwargs(run_id), question="What is event sourcing?"
        )

        store.append(event)
        result = store.events_for_run(run_id)

        assert result == [event]

    def test_context_retrieved_with_string_roundtrip(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        run_id = uuid4()
        event = ContextRetrieved(
            **_common_kwargs(run_id), context="Some retrieved context."
        )

        store.append(event)
        result = store.events_for_run(run_id)

        assert result == [event]

    def test_context_retrieved_with_none_roundtrip(self, tmp_path: Path) -> None:
        """Pinned: None survives the JSON round-trip as None, not as missing
        field or empty string. ContextRetrieved(context=None) records a
        keyword-match miss explicitly; the projection layer relies on this."""
        store = _make_store(tmp_path)
        run_id = uuid4()
        event = ContextRetrieved(**_common_kwargs(run_id), context=None)

        store.append(event)
        result = store.events_for_run(run_id)

        assert result == [event]
        assert isinstance(result[0], ContextRetrieved)
        assert result[0].context is None

    def test_answer_generated_roundtrip(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        run_id = uuid4()
        event = AnswerGenerated(**_common_kwargs(run_id), answer="An answer.")

        store.append(event)
        result = store.events_for_run(run_id)

        assert result == [event]

    def test_model_invocation_failed_with_nested_error_info_roundtrip(
        self, tmp_path: Path
    ) -> None:
        """Pinned: nested frozen dataclass fields reconstruct correctly.
        ErrorInfo is the V3a test case; the recursive serialisation logic
        generalises to any future event with nested dataclass payload."""
        store = _make_store(tmp_path)
        run_id = uuid4()
        info = ErrorInfo(
            exception_type="OllamaConnectionError",
            exception_message="Connection refused",
        )
        event = ModelInvocationFailed(**_common_kwargs(run_id), error_info=info)

        store.append(event)
        result = store.events_for_run(run_id)

        assert result == [event]
        assert isinstance(result[0], ModelInvocationFailed)
        assert result[0].error_info == info


class TestSqliteEventStoreOrderingAndIsolation:

    def test_replay_preserves_append_order(self, tmp_path: Path) -> None:
        """Pinned: events_for_run returns events in the order they were
        appended. ORDER BY rowid is the mechanism — projections and
        composition assertions assume this contract."""
        store = _make_store(tmp_path)
        run_id = uuid4()
        first = QuestionReceived(**_common_kwargs(run_id), question="Q")
        second = ContextRetrieved(**_common_kwargs(run_id), context="C")
        third = AnswerGenerated(**_common_kwargs(run_id), answer="A")

        store.append(first)
        store.append(second)
        store.append(third)

        assert store.events_for_run(run_id) == [first, second, third]

    def test_replay_of_unknown_run_returns_empty_list(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        assert store.events_for_run(uuid4()) == []

    def test_runs_are_isolated_by_aggregate_id(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path)
        run_a = uuid4()
        run_b = uuid4()
        event_a = QuestionReceived(**_common_kwargs(run_a), question="QA")
        event_b = QuestionReceived(**_common_kwargs(run_b), question="QB")

        store.append(event_a)
        store.append(event_b)

        assert store.events_for_run(run_a) == [event_a]
        assert store.events_for_run(run_b) == [event_b]


class TestSqliteEventStorePersistence:

    def test_append_survives_store_reconstruction(self, tmp_path: Path) -> None:
        """Pinned: data persists across store instances on the same file.
        This is what makes it the production concrete — InMemoryEventStore
        cannot satisfy this test."""
        db_path = tmp_path / "events.db"
        run_id = uuid4()
        event = QuestionReceived(**_common_kwargs(run_id), question="Q")

        first_store = SqliteEventStore(db_path, _QA_EVENT_TYPES)
        first_store.append(event)

        second_store = SqliteEventStore(db_path, _QA_EVENT_TYPES)
        assert second_store.events_for_run(run_id) == [event]


class TestSqliteEventStoreRegistry:

    def test_unregistered_event_type_on_read_raises(self, tmp_path: Path) -> None:
        """Pinned: reading an event whose type is not in the registry
        raises ValueError. The registry is the schema-evolution boundary —
        a stored event for a type the store has forgotten about should
        fail loud, not silently drop or guess."""
        db_path = tmp_path / "events.db"
        run_id = uuid4()
        event = QuestionReceived(**_common_kwargs(run_id), question="Q")

        full_store = SqliteEventStore(db_path, _QA_EVENT_TYPES)
        full_store.append(event)

        # Reconstruct with empty registry
        narrowed_store = SqliteEventStore(db_path, [])
        with pytest.raises(ValueError, match="Unknown event type"):
            narrowed_store.events_for_run(run_id)


class TestSqliteEventStoreAuthEventRoundTrip:

    def test_login_attempted_survives_store_reconstruction(
        self, tmp_path: Path
    ) -> None:
        """F01 integration proof: a LoginAttempted event appended to the
        store can be read back from a fresh store instance on the same
        file with the union registry. This is the read-side of F01 —
        V3a's QA-only registry would have accepted the write (no
        write-side registry check) but raised ValueError at read time.

        The registry is built directly from imports here; the test must
        not import _ALL_EVENT_TYPES from application.container — the
        store-level test should not depend on container symbols."""
        from domain.events.auth_events import (
            LoginAttempted,
            LoginFailed,
            LoginSucceeded,
        )

        union_registry: list[type[BaseAgentEvent]] = [
            QuestionReceived,
            ContextRetrieved,
            AnswerGenerated,
            ModelInvocationFailed,
            LoginAttempted,
            LoginSucceeded,
            LoginFailed,
        ]

        db_path = tmp_path / "test.db"
        run_id = uuid4()
        event = LoginAttempted(
            **_common_kwargs(run_id),
            username="test_user",
        )

        first_store = SqliteEventStore(db_path, union_registry)
        first_store.append(event)

        second_store = SqliteEventStore(db_path, union_registry)
        result = second_store.events_for_run(run_id)

        assert len(result) == 1
        assert isinstance(result[0], LoginAttempted)
        assert result[0].username == event.username
        assert result[0].event_id == event.event_id
        assert result[0].aggregate_id == event.aggregate_id
        assert result[0].occurred_at == event.occurred_at