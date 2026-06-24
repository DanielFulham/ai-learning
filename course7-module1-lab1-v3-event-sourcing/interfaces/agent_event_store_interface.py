from typing import Protocol
from uuid import UUID

from domain.events.base import BaseAgentEvent


class AgentEventStoreInterface(Protocol):
    """Append-only persistent log of agent events.

    Two operations: `append` on write, `events_for_run` on read. The store
    is keyed on `aggregate_id` (the run UUID). Replay-by-run is the only
    read path V3a needs — `RunSummaryProjection` consumes it, and the
    event-log composition assertions read it to verify a full run's
    semantics. Time-window and cross-run queries are deferred until a
    consumer demands them.

    Concretes:
    - `InMemoryEventStore` (V3a) — list-backed, test default
    - `SqliteEventStore` (V3a) — single append-only table indexed on
      aggregate_id; production concrete
    """

    def append(self, event: BaseAgentEvent) -> None: ...

    def events_for_run(self, run_id: UUID) -> list[BaseAgentEvent]: ...