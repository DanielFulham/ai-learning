from uuid import UUID

from domain.events.base import BaseAgentEvent


class InMemoryEventStore:
    """List-backed event store, the test default and smallest demonstration.

    Storage is a single list of events; reads filter by `aggregate_id`. The
    returned list is a fresh copy on each call — callers mutating it cannot
    affect the store. Events themselves are frozen, so the full graph is
    effectively immutable from the caller's perspective.

    Append order is preserved on replay. No thread-safety; V3a's services
    are single-shot and the in-memory store is for tests and the smallest
    demonstration, where concurrent appends are not a concern.

    Satisfies `AgentEventStoreInterface` structurally — no explicit
    inheritance because Protocol is structural typing.
    """

    def __init__(self) -> None:
        self._events: list[BaseAgentEvent] = []

    def append(self, event: BaseAgentEvent) -> None:
        self._events.append(event)

    def events_for_run(self, run_id: UUID) -> list[BaseAgentEvent]:
        return [e for e in self._events if e.aggregate_id == run_id]