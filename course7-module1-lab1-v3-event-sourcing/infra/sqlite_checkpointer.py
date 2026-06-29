import json
import sqlite3
from pathlib import Path
from typing import Any
from uuid import UUID

from domain.agent_checkpoint import AgentCheckpoint


class SqliteCheckpointer:
    """SQLite-backed checkpointer, the production concrete.

    Single `checkpoints` table keyed by `run_id`:
    - `run_id` PK (the run the checkpoint belongs to, UUID stringified)
    - `state` as a JSON blob of the LangGraph checkpoint state dict

    `save` is last-write-wins via INSERT OR REPLACE — a second save for the
    same run_id replaces the prior checkpoint, matching both LangGraph's
    checkpointer semantics (each save replaces the state-at-that-point) and
    `InMemoryCheckpointer`'s dict overwrite.

    `load` returns `None` for unknown run_ids rather than raising — callers
    branch on presence to choose between resuming and starting fresh.

    Shares the SQLite database file with `SqliteEventStore` when persistence
    is enabled: one DB, two tables (`events` + `checkpoints`). The container
    selects one `db_path` and passes it to both concretes.

    ## Concurrency boundary (V3 scope)

    Each `save` and `load` opens and closes its own connection; concurrency
    is the OS file lock. WAL mode, `busy_timeout`, and retry-on-locked are
    the production migration path (F14), out of V3 scope — the same boundary
    `SqliteEventStore` documents. Satisfies `AgentCheckpointerInterface`
    structurally.
    """

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = str(db_path)
        self._init_schema()

    def _init_schema(self) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL
                )
                """
            )

    def save(self, checkpoint: AgentCheckpoint) -> None:
        with sqlite3.connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO checkpoints (run_id, state)
                VALUES (?, ?)
                """,
                (str(checkpoint.run_id), json.dumps(checkpoint.state)),
            )

    def load(self, run_id: UUID) -> AgentCheckpoint | None:
        with sqlite3.connect(self._db_path) as conn:
            cursor = conn.execute(
                "SELECT state FROM checkpoints WHERE run_id = ?",
                (str(run_id),),
            )
            row = cursor.fetchone()

        if row is None:
            return None

        state: dict[str, Any] = json.loads(row[0])
        return AgentCheckpoint(run_id=run_id, state=state)
