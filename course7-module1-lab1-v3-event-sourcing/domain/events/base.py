from dataclasses import dataclass
from datetime import datetime
from uuid import UUID


@dataclass(frozen=True, kw_only=True)
class BaseAgentEvent:
    """Shared shape for every event written to the event store.

    `event_id` and `aggregate_id` carry no defaults — the translator is
    the canonical source of event IDs (uuid4 at translation time), and
    the aggregate_id is the run_id threaded through the per-run streaming
    consumer. Defaulting them here would make accidental construction
    silently violate the "translator is the source" pin.

    `occurred_at` carries no default for the same reason — time is an
    injected concern at the translator boundary, not a hidden import-time
    side effect. Tests construct deterministic timestamps explicitly.

    `schema_version` defaults to 1. New version → bump the default and
    teach storage to handle older versions on read. New event type → new
    dataclass under the per-service union, not a version bump.

    `kw_only=True` lets subclasses add non-default fields after the base's
    defaulted `schema_version` — without it, dataclass inheritance forbids
    that ordering.
    """

    event_id: UUID
    aggregate_id: UUID
    occurred_at: datetime
    schema_version: int = 1