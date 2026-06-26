from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from langgraph.graph.state import CompiledStateGraph

from application.graph_builders import build_qa_graph
from application.lab_app import LabApp
from application.qa_agent_service import QAAgentService
from domain.events.auth_events import (
    LoginAttempted,
    LoginFailed,
    LoginSucceeded,
)
from domain.events.base import BaseAgentEvent
from domain.events.qa_events import (
    AnswerGenerated,
    ContextRetrieved,
    ModelInvocationFailed,
    QuestionReceived,
)
from infra.console_stream_consumer import ConsoleStreamConsumer
from infra.in_memory_event_store import InMemoryEventStore
from infra.null_stream_consumer import NullStreamConsumer
from infra.ollama_chat_model_provider import OllamaChatModelProvider
from infra.openai_chat_model_provider import OpenAIChatModelProvider
from infra.sqlite_event_store import SqliteEventStore
from interfaces.agent_event_store_interface import AgentEventStoreInterface
from interfaces.chat_model_provider_interface import ChatModelProviderInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


_QA_EVENT_TYPES: list[type[BaseAgentEvent]] = [
    QuestionReceived,
    ContextRetrieved,
    AnswerGenerated,
    ModelInvocationFailed,
]

_AUTH_EVENT_TYPES: list[type[BaseAgentEvent]] = [
    LoginAttempted,
    LoginSucceeded,
    LoginFailed,
]

_ALL_EVENT_TYPES: list[type[BaseAgentEvent]] = _QA_EVENT_TYPES + _AUTH_EVENT_TYPES


def _production_clock() -> datetime:
    return datetime.now(timezone.utc)


def initialise(
    chat_model_provider: ChatModelProviderInterface | None = None,
    qa_graph: CompiledStateGraph | None = None,
    event_store: AgentEventStoreInterface | None = None,
    inner_consumer: StreamConsumerInterface | None = None,
    clock: Callable[[], datetime] | None = None,
    use_openai: bool = False,
    use_sqlite_persistence: bool = False,
    db_path: str | Path | None = None,
    use_console_consumer: bool = True,
) -> LabApp:
    """V3a composition root.

    Wires V3a's event-sourced substrate around V2's QA graph. Stateless
    — each call constructs fresh instances; the entry point caches the
    LabApp if it wants singleton behaviour.

    Explicit injection wins over booleans. The booleans select between
    default concretes when no instance is passed:
    - `use_openai`: OpenAI provider vs Ollama (default)
    - `use_sqlite_persistence`: SqliteEventStore vs InMemoryEventStore;
      SQLite requires `db_path` and raises if missing
    - `use_console_consumer`: ConsoleStreamConsumer vs NullStreamConsumer

    `qa_graph` is optional. When omitted, the container constructs it
    from the chat_model_provider via `build_qa_graph(model)`. When
    `qa_graph` is explicitly injected, the chat_model_provider is
    bypassed entirely — tests pass a mock graph and no provider.

    The same event_store instance is held by the QA service. When V3b
    and V3c land, Auth and Counter services receive the same reference
    — the singleton contract for the event store is what makes
    cross-aggregate projections meaningful in V3c.
    """
    if event_store is None:
        if use_sqlite_persistence:
            if db_path is None:
                raise ValueError(
                    "db_path is required when use_sqlite_persistence=True"
                )
            event_store = SqliteEventStore(db_path, _ALL_EVENT_TYPES)
        else:
            event_store = InMemoryEventStore()

    if inner_consumer is None:
        if use_console_consumer:
            inner_consumer = ConsoleStreamConsumer()
        else:
            inner_consumer = NullStreamConsumer()

    if clock is None:
        clock = _production_clock

    if qa_graph is None:
        if chat_model_provider is None:
            if use_openai:
                chat_model_provider = OpenAIChatModelProvider()
            else:
                chat_model_provider = OllamaChatModelProvider()
        model = chat_model_provider.create()
        qa_graph = build_qa_graph(model)

    return LabApp(
        qa=QAAgentService(
            graph=qa_graph,
            event_store=event_store,
            inner_consumer=inner_consumer,
            clock=clock,
        ),
        event_store=event_store,
    )