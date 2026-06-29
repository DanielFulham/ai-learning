import os
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.language_models import BaseChatModel
from langgraph.graph.state import CompiledStateGraph

from application.container import initialise
from application.lab_app import LabApp
from application.qa_agent_service import QAAgentService
from infra.console_stream_consumer import ConsoleStreamConsumer
from infra.in_memory_event_store import InMemoryEventStore
from infra.null_stream_consumer import NullStreamConsumer
from infra.ollama_chat_model_provider import OllamaChatModelProvider
from infra.openai_chat_model_provider import OpenAIChatModelProvider
from infra.sqlite_event_store import SqliteEventStore
from interfaces.agent_checkpointer_interface import AgentCheckpointerInterface
from interfaces.agent_event_store_interface import AgentEventStoreInterface
from interfaces.chat_model_provider_interface import ChatModelProviderInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


def _mock_graph() -> MagicMock:
    return MagicMock(spec=CompiledStateGraph)


def _mock_provider() -> MagicMock:
    """Mock provider returning a mock BaseChatModel. Lets the container's
    graph-construction path run without instantiating a real LLM client."""
    provider = MagicMock(spec=ChatModelProviderInterface)
    provider.create.return_value = MagicMock(spec=BaseChatModel)
    return provider


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


class TestContainerGraphConstruction:

    def test_default_builds_qa_graph_from_ollama_provider(self) -> None:
        """Pinned: default path constructs the QA graph via Ollama provider.
        ChatOllama construction does not network; the graph is built
        without Ollama running."""
        app = initialise(chat_model_provider=_mock_provider())
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert isinstance(qa_service._graph, CompiledStateGraph)

    def test_use_openai_selects_openai_provider(self) -> None:
        """Pinned: use_openai=True selects OpenAIChatModelProvider when
        no explicit provider is injected."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "dummy-key-for-test"}):
            mock_provider = _mock_provider()
            # Patch the OpenAI provider construction so we can verify
            # use_openai=True wired it without needing real OpenAI.
            with patch(
                "application.container.OpenAIChatModelProvider",
                return_value=mock_provider,
            ) as openai_ctor:
                initialise(use_openai=True)
                openai_ctor.assert_called_once()

    def test_default_provider_when_not_openai_is_ollama(self) -> None:
        with patch(
            "application.container.OllamaChatModelProvider",
            return_value=_mock_provider(),
        ) as ollama_ctor:
            initialise()
            ollama_ctor.assert_called_once()

    def test_explicit_chat_model_provider_overrides_use_openai(self) -> None:
        """Pinned: explicit injection wins over the boolean."""
        injected = _mock_provider()
        initialise(chat_model_provider=injected, use_openai=True)
        injected.create.assert_called_once()

    def test_explicit_qa_graph_bypasses_provider_entirely(self) -> None:
        """Pinned: when qa_graph is injected, the provider is never called.
        Tests pass a mock graph and skip all LLM-provider plumbing."""
        provider = _mock_provider()
        initialise(qa_graph=_mock_graph(), chat_model_provider=provider)
        provider.create.assert_not_called()


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

    def test_explicit_injections_override_sqlite_flag(self) -> None:
        """Pinned: explicit injection wins for both persistence concretes.
        The booleans only select defaults when no instance is passed.
        Injecting event_store and checkpointer bypasses both SQLite
        construction paths, so use_sqlite_persistence and db_path are
        ignored — the /dev/null sentinel is never opened."""
        injected_store = MagicMock(spec=AgentEventStoreInterface)
        injected_checkpointer = MagicMock(spec=AgentCheckpointerInterface)
        app = initialise(
            qa_graph=_mock_graph(),
            event_store=injected_store,
            checkpointer=injected_checkpointer,
            use_sqlite_persistence=True,
            db_path="/dev/null",
        )
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert qa_service._event_store is injected_store
        assert app.checkpointer is injected_checkpointer

    def test_explicit_inner_consumer_overrides_console_flag(self) -> None:
        injected = MagicMock(spec=StreamConsumerInterface)
        app = initialise(
            qa_graph=_mock_graph(),
            inner_consumer=injected,
            use_console_consumer=True,
        )
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert qa_service._inner_consumer is injected

    def test_explicit_clock_overrides_default(self) -> None:
        fixed_time = datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc)

        def injected_clock() -> datetime:
            return fixed_time

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

    def test_event_store_on_lab_app_matches_service_reference(self) -> None:
        """Pinned: the same event_store reference is exposed on LabApp and
        held by the service. The demo's RunSummaryProjection call reads
        from the same store the service wrote to — that's the read-side
        of the singleton contract."""
        app = initialise(qa_graph=_mock_graph())
        qa_service = app.qa
        assert isinstance(qa_service, QAAgentService)
        assert app.event_store is qa_service._event_store


class TestContainerStateless:

    def test_two_calls_return_different_instances(self) -> None:
        """Pinned: container is stateless. Two initialise() calls return
        independent LabApp instances with independent dependencies."""
        first = initialise(qa_graph=_mock_graph())
        second = initialise(qa_graph=_mock_graph())
        assert first is not second
        assert first.qa is not second.qa


class TestContainerRegistryUnion:

    def test_sqlite_event_store_receives_union_registry(
        self, tmp_path: Path
    ) -> None:
        """Pinned: when use_sqlite_persistence=True, the SqliteEventStore is
        constructed with a registry list containing all four QA event types
        AND all three Auth event types. The composition (_ALL_EVENT_TYPES)
        must be passed at the construction site — not the QA-only constant.

        Patched at the import site in application.container to intercept
        the constructor call and inspect the registry argument."""
        from domain.events.auth_events import LoginAttempted, LoginFailed, LoginSucceeded
        from domain.events.qa_events import (
            AnswerGenerated,
            ContextRetrieved,
            ModelInvocationFailed,
            QuestionReceived,
        )

        expected_registry = [
            QuestionReceived,
            ContextRetrieved,
            AnswerGenerated,
            ModelInvocationFailed,
            LoginAttempted,
            LoginSucceeded,
            LoginFailed,
        ]

        with patch(
            "application.container.SqliteEventStore",
            spec=SqliteEventStore,
        ) as mock_ctor:
            mock_ctor.return_value = MagicMock(spec=SqliteEventStore)
            initialise(
                qa_graph=_mock_graph(),
                use_sqlite_persistence=True,
                db_path=tmp_path / "events.db",
            )
            mock_ctor.assert_called_once()
            _, call_kwargs = mock_ctor.call_args
            # SqliteEventStore(db_path, registry) — positional args
            call_args, _ = mock_ctor.call_args
            actual_registry = call_args[1]
            assert actual_registry == expected_registry

    def test_in_memory_path_does_not_call_sqlite_event_store(self) -> None:
        """Pinned: the in-memory path must not construct SqliteEventStore.
        This test must still pass after the registry union change."""
        with patch(
            "application.container.SqliteEventStore",
            spec=SqliteEventStore,
        ) as mock_ctor:
            initialise(qa_graph=_mock_graph())
            mock_ctor.assert_not_called()