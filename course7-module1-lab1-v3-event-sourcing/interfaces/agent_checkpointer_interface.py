from typing import Protocol
from uuid import UUID

from domain.agent_checkpoint import AgentCheckpoint


class AgentCheckpointerInterface(Protocol):
    """Persistent store for resumable run state.

    Two operations: `save` on write, `load` on read keyed by run_id. The
    load contract returns `None` when no checkpoint exists for the run_id,
    rather than raising — callers branch on presence to choose between
    resuming an interrupted run and starting fresh.

    Checkpoints are opaque to the application. The state dict matches
    LangGraph's checkpointer surface; the application doesn't reason
    about its contents. Persisting and restoring it is the only operation
    the interface exposes.

    Concretes:
    - `InMemoryCheckpointer` — dict-backed, test default
    - `SqliteCheckpointer` — same SQLite database as the event store
      when persistence is enabled

    Implementations that serialise `state` to a string format (e.g.
    `SqliteCheckpointer` using `json.dumps`) require `state` to contain
    only JSON-serialisable values. Non-serialisable values raise the
    underlying serialisation library's error at write time.
    """

    def save(self, checkpoint: AgentCheckpoint) -> None: ...

    def load(self, run_id: UUID) -> AgentCheckpoint | None: ...