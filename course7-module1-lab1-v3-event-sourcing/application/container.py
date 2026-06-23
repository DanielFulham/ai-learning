from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from langgraph.graph.state import CompiledStateGraph

from application.lab_app import LabApp
from application.qa_agent_service import QAAgentService
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
from infra.sqlite_event_store import SqliteEventStore
from interfaces.agent_event_store_interface import AgentEventStoreInterface
from interfaces.stream_consumer_interface import StreamConsumerInterface


_QA_EVENT_TYPES: list[type[BaseAgentEvent]] = [
    QuestionReceived,
    ContextRetrieved,
    AnswerGenerated,
    ModelInvocationFailed,
]


def _production_clock() -> datetime:
    return datetime.now(timezone.utc)


def initialise(
    qa_graph: CompiledStateGraph,
    event_store: AgentEventStoreInterface | None = None,
    inner_consumer: StreamConsumerInterface | None = None,
    clock: Callable[[], datetime] | None = None,
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
    - `use_sqlite_persistence` chooses between InMemoryEventStore and
      SqliteEventStore. SQLite requires `db_path`; raises if missing.
    - `use_console_consumer` chooses between ConsoleStreamConsumer (the
      dev observation channel) and NullStreamConsumer (silent default
      for tests and headless production).

    `qa_graph` is a required positional argument in V3a — the container
    does not yet lift V2's graph builders and node factories. The
    caller (demo.py, tests) constructs the graph and hands it in. The
    next commit lifts the V2 application files and moves graph
    construction inside.

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
            event_store = SqliteEventStore(db_path, _QA_EVENT_TYPES)
        else:
            event_store = InMemoryEventStore()

    if inner_consumer is None:
        if use_console_consumer:
            inner_consumer = ConsoleStreamConsumer()
        else:
            inner_consumer = NullStreamConsumer()

    if clock is None:
        clock = _production_clock

    return LabApp(
        qa=QAAgentService(
            graph=qa_graph,
            event_store=event_store,
            inner_consumer=inner_consumer,
            clock=clock,
        ),
    )