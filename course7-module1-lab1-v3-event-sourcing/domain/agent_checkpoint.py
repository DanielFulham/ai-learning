from dataclasses import dataclass
from typing import Any
from uuid import UUID


@dataclass(frozen=True)
class AgentCheckpoint:
    """Opaque snapshot of agent state at a point in execution.

    `state` is `dict[str, Any]` matching LangGraph's checkpointer surface.
    The application does not reason about its contents — checkpoints are
    a persistence concern around LangGraph's execution state, not a
    domain aggregate. Lives in `domain/` because it crosses the
    checkpointer interface boundary, but carries no invariants or
    application semantics.

    The frozen wrapper prevents accidental run_id reassignment after
    construction. The state dict itself is referentially mutable; callers
    that hold the original dict reference and mutate it will see those
    mutations reflected in the checkpoint. Treat the checkpoint as the
    canonical reference after construction.
    """

    run_id: UUID
    state: dict[str, Any]