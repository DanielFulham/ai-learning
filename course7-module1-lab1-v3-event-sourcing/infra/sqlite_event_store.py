import json
import sqlite3
import types
from dataclasses import fields, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Union, get_args, get_origin, get_type_hints
from uuid import UUID

from domain.events.base import BaseAgentEvent


_COMMON_FIELD_NAMES = frozenset(
    {"event_id", "aggregate_id", "occurred_at", "schema_version"}
)


def _to_json_compatible(value: Any) -> Any:
    """Recursively convert a value graph to JSON-serialisable primitives.

    Handles UUIDs (→ string), datetimes (→ ISO 8601), and nested frozen
    dataclasses (→ dict by field name). Other primitives pass through.
    """
    if value is None:
        return None
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _to_json_compatible(getattr(value, f.name))
            for f in fields(value)
        }
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    return value


def _from_json_compatible(field_type: Any, value: Any) -> Any:
    """Reconstruct a typed value from a JSON-deserialised primitive.

    Walks the field's resolved type annotation:
    - None for any nullable field stays None
    - Union types are unwrapped to their non-None member (only T | None
      shapes are supported; T | U with two non-None members is V3a YAGNI)
    - Nested dataclass types are reconstructed recursively
    - UUID and datetime are parsed from their string forms
    - Everything else passes through
    """
    if value is None:
        return None

    origin = get_origin(field_type)
    if origin is Union or origin is types.UnionType:
        non_none_args = [a for a in get_args(field_type) if a is not type(None)]
        if len(non_none_args) == 1:
            field_type = non_none_args[0]

    if is_dataclass(field_type) and isinstance(field_type, type):
        nested_hints = get_type_hints(field_type)
        return field_type(
            **{
                f.name: _from_json_compatible(nested_hints[f.name], value[f.name])
                for f in fields(field_type)
            }
        )
    if field_type is UUID:
        return UUID(value)
    if field_type is datetime:
        return datetime.fromisoformat(value)
    return value


class SqliteEventStore:
    """SQLite-backed event store, the production concrete.

    Single append-only `events` table:
    - `event_id` PK
    - `aggregate_id` indexed (events_for_run is the only read path)
    - `event_type` discriminator string (the concrete dataclass name)
    - `schema_version` promoted out of the JSON payload for read-side
      migration logic when schema versions diverge across stored events
    - `occurred_at` as ISO 8601
    - `payload` as a JSON blob of the event-type-specific fields

    Append order is preserved on replay via `ORDER BY rowid` — SQLite's
    rowid is monotonically incrementing for append-only tables.

    Event types must be registered at construction time. Reading an
    event type not in the registry raises `ValueError` — the registry
    is the schema-evolution boundary, not silent failure.

    ## Concurrency boundary (V3a scope)

    Each `append` and `events_for_run` opens and closes its own
    connection. Concurrency is the OS file lock; V3a's single-shot
    services do not need anything stronger.

    Concurrent writes from multiple workers will fail under contention
    with `sqlite3.OperationalError: database is locked`. The production
    migration path is: enable WAL mode (`PRAGMA journal_mode=WAL`),
    set `PRAGMA busy_timeout` to a value matching expected write
    latency, and add a bounded retry loop on the locked-error class
    around append calls. Connection pooling becomes worthwhile when
    the per-call connect cost dominates query latency — that's a
    Postgres-tier concern, not a SQLite one.

    For V3a (single-shot lab demo), this is a documented scope
    boundary, not a defect.
    """

    def __init__(
        self,
        db_path: str | Path,
        event_types: list[type[BaseAgentEvent]],
    ) -> None:
        self._db_path = str(db_path)
        self._registry: dict[str, type[BaseAgentEvent]] = {
            cls.__name__: cls for cls in event_types
        }
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    aggregate_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_events_aggregate "
                "ON events(aggregate_id)"
            )

    def append(self, event: BaseAgentEvent) -> None:
        event_type = type(event).__name__
        payload = {
            f.name: _to_json_compatible(getattr(event, f.name))
            for f in fields(event)
            if f.name not in _COMMON_FIELD_NAMES
        }
        payload_json = json.dumps(payload)

        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO events
                    (event_id, aggregate_id, event_type,
                     schema_version, occurred_at, payload)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    str(event.event_id),
                    str(event.aggregate_id),
                    event_type,
                    event.schema_version,
                    event.occurred_at.isoformat(),
                    payload_json,
                ),
            )

    def events_for_run(self, run_id: UUID) -> list[BaseAgentEvent]:
        with sqlite3.connect(self._db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """
                SELECT event_id, aggregate_id, event_type,
                       schema_version, occurred_at, payload
                FROM events
                WHERE aggregate_id = ?
                ORDER BY rowid
                """,
                (str(run_id),),
            )
            return [self._row_to_event(row) for row in cursor]

    def _row_to_event(self, row: sqlite3.Row) -> BaseAgentEvent:
        event_type_str = row["event_type"]
        cls = self._registry.get(event_type_str)
        if cls is None:
            raise ValueError(
                f"Unknown event type '{event_type_str}' — not in registry. "
                f"Registered types: {sorted(self._registry.keys())}"
            )
        payload = json.loads(row["payload"])
        type_hints = get_type_hints(cls)
        payload_kwargs = {
            f.name: _from_json_compatible(type_hints[f.name], payload[f.name])
            for f in fields(cls)
            if f.name not in _COMMON_FIELD_NAMES
        }
        return cls(
            event_id=UUID(row["event_id"]),
            aggregate_id=UUID(row["aggregate_id"]),
            occurred_at=datetime.fromisoformat(row["occurred_at"]),
            schema_version=row["schema_version"],
            **payload_kwargs,
        )