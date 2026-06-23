from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langgraph.graph.state import CompiledStateGraph

from application.container import initialise
from application.lab_app import LabApp
from application.qa_agent_service import QAAgentService
from infra.console_stream_consumer import ConsoleStreamConsumer
from infra.in_memory_event_store import InMemoryEventStore
from infra.null_stream_consumer import NullStreamConsumer
from infra.sqlite_event_store import SqliteEventStore
from interfaces.agent_event_store_interface import AgentEventStoreInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


def _mock_graph() -> MagicMock:
    return MagicMock(spec=CompiledStateGraph)


class TestContainerReturnShape:

    def test_returns_lab_app_with_qa_service(self) -> None:
        app = initialise(qa_graph=_mock_graph())
        assert isinstance(app, LabApp)
        assert isinstance(app.qa, QAAgentService)


class TestContainerDefaultWiring:

    def test_default_event_store_is_in_memory(self) -> None:
        """Pinned: default V3a wiring is in-memory. Tests, dev iteration,
        and the smallest demonstration use InMemoryEventStore; SQLite is
        the opt-in for persistence demos."""
        app = initialise(qa_graph=_mock_graph())
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert isinstance(qa_service._event_store, InMemoryEventStore)

    def test_default_inner_consumer_is_console(self) -> None:
        """Pinned: default inner consumer is ConsoleStreamConsumer.
        Dev observation channel is on by default; tests opt out."""
        app = initialise(qa_graph=_mock_graph())
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert isinstance(qa_service._inner_consumer, ConsoleStreamConsumer)

    def test_default_clock_is_callable_returning_utc_datetime(self) -> None:
        app = initialise(qa_graph=_mock_graph())
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        now = qa_service._clock()
        assert isinstance(now, datetime)
        assert now.tzinfo == timezone.utc


class TestContainerSqliteWiring:

    def test_use_sqlite_persistence_wires_sqlite_event_store(
        self, tmp_path: Path
    ) -> None:
        db_path = tmp_path / "events.db"
        app = initialise(
            qa_graph=_mock_graph(),
            use_sqlite_persistence=True,
            db_path=db_path,
        )
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert isinstance(qa_service._event_store, SqliteEventStore)

    def test_use_sqlite_persistence_without_db_path_raises(self) -> None:
        """Pinned: SQLite requires a path. The container fails loud at
        construction rather than crashing the first append."""
        with pytest.raises(ValueError, match="db_path is required"):
            initialise(qa_graph=_mock_graph(), use_sqlite_persistence=True)


class TestContainerNullConsumerWiring:

    def test_use_console_consumer_false_wires_null_consumer(self) -> None:
        app = initialise(qa_graph=_mock_graph(), use_console_consumer=False)
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert isinstance(qa_service._inner_consumer, NullStreamConsumer)


class TestContainerExplicitInjection:

    def test_explicit_event_store_overrides_sqlite_flag(self) -> None:
        """Pinned: explicit injection wins. The booleans only select
        defaults when no instance is passed."""
        injected = MagicMock(spec=AgentEventStoreInterface)
        app = initialise(
            qa_graph=_mock_graph(),
            event_store=injected,
            use_sqlite_persistence=True,  # ignored
            db_path="/dev/null",  # ignored
        )
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert qa_service._event_store is injected

    def test_explicit_inner_consumer_overrides_console_flag(self) -> None:
        injected = MagicMock(spec=StreamConsumerInterface)
        app = initialise(
            qa_graph=_mock_graph(),
            inner_consumer=injected,
            use_console_consumer=True,  # ignored
        )
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert qa_service._inner_consumer is injected

    def test_explicit_clock_overrides_default(self) -> None:
        fixed_time = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)
        injected_clock = lambda: fixed_time
        app = initialise(qa_graph=_mock_graph(), clock=injected_clock)
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert qa_service._clock() == fixed_time


class TestContainerSingleton:

    def test_event_store_reference_is_held_by_qa_service(self) -> None:
        """Pinned: the container constructs one event_store and the QA
        service holds the same reference. V3b and V3c's services will
        receive the same instance — the singleton contract is what makes
        ThreadHistoryProjection's cross-aggregate joins meaningful."""
        injected_store = MagicMock(spec=AgentEventStoreInterface)
        app = initialise(qa_graph=_mock_graph(), event_store=injected_store)
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert qa_service._event_store is injected_store


class TestContainerStateless:

    def test_two_calls_return_different_instances(self) -> None:
        """Pinned: container is stateless. Two initialise() calls return
        independent LabApp instances with independent dependencies. The
        entry point owns any singleton behaviour."""
        first = initialise(qa_graph=_mock_graph())
        second = initialise(qa_graph=_mock_graph())
        assert first is not second
        assert first.qa is not second.qa