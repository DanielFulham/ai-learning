from uuid import UUID

from domain.agent_checkpoint import AgentCheckpoint


class InMemoryCheckpointer:
    """Dict-backed checkpointer, the test default and smallest demonstration.

    Storage is a single dict keyed by run_id. `save` is last-write-wins —
    a second save for the same run_id replaces the prior checkpoint
    (matches LangGraph's checkpointer semantics: each save replaces the
    state-at-that-point).

    `load` returns `None` for unknown run_ids rather than raising — callers
    branch on presence to choose between resuming and starting fresh.

    No thread-safety; V3a's services are single-shot. Satisfies
    `AgentCheckpointerInterface` structurally.
    """

    def __init__(self) -> None:
        self._checkpoints: dict[UUID, AgentCheckpoint] = {}

    def save(self, checkpoint: AgentCheckpoint) -> None:
        self._checkpoints[checkpoint.run_id] = checkpoint

    def load(self, run_id: UUID) -> AgentCheckpoint | None:
        return self._checkpoints.get(run_id)