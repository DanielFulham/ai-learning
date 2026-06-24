from dataclasses import FrozenInstanceError
from typing import Any
from uuid import uuid4

import pytest

from domain.agent_checkpoint import AgentCheckpoint


def _make_checkpoint(
    state: dict[str, Any] | None = None,
) -> AgentCheckpoint:
    return AgentCheckpoint(
        run_id=uuid4(),
        state=state if state is not None else {"messages": []},
    )


class TestAgentCheckpoint:

    def test_fields_accessible(self) -> None:
        run_id = uuid4()
        state = {"messages": [], "step": 3}
        checkpoint = AgentCheckpoint(run_id=run_id, state=state)
        assert checkpoint.run_id == run_id
        assert checkpoint.state == {"messages": [], "step": 3}

    def test_is_frozen(self) -> None:
        """Pinned: run_id can't be reassigned after construction. Prevents a
        checkpoint instance from silently representing the wrong run."""
        checkpoint = _make_checkpoint()
        with pytest.raises(FrozenInstanceError):
            setattr(checkpoint, "run_id", uuid4())

    def test_structural_equality(self) -> None:
        """Pinned: frozen dataclass equality is structural across both fields.
        Two checkpoints with the same run_id and the same state dict compare
        equal — lets tests assert on checkpoint contents without holding the
        original reference."""
        run_id = uuid4()
        a = AgentCheckpoint(run_id=run_id, state={"k": "v"})
        b = AgentCheckpoint(run_id=run_id, state={"k": "v"})
        assert a == b

    def test_different_run_id_not_equal(self) -> None:
        state = {"k": "v"}
        a = AgentCheckpoint(run_id=uuid4(), state=state)
        b = AgentCheckpoint(run_id=uuid4(), state=state)
        assert a != b

    def test_different_state_not_equal(self) -> None:
        run_id = uuid4()
        a = AgentCheckpoint(run_id=run_id, state={"k": "v1"})
        b = AgentCheckpoint(run_id=run_id, state={"k": "v2"})
        assert a != b