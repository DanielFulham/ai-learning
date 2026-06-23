from uuid import uuid4

from domain.agent_checkpoint import AgentCheckpoint
from infra.in_memory_checkpointer import InMemoryCheckpointer
from interfaces.agent_checkpointer_interface import AgentCheckpointerInterface


def _accepts_checkpointer(checkpointer: AgentCheckpointerInterface) -> None:
    """Type-guard helper. Pyright fails the call site if the argument
    doesn't satisfy the Protocol."""


class TestInMemoryCheckpointerInterfaceSatisfaction:

    def test_satisfies_agent_checkpointer_interface(self) -> None:
        """Pinned: InMemoryCheckpointer satisfies AgentCheckpointerInterface
        structurally."""
        checkpointer = InMemoryCheckpointer()
        _accepts_checkpointer(checkpointer)


class TestInMemoryCheckpointerSaveLoad:

    def test_save_then_load_roundtrip(self) -> None:
        checkpointer = InMemoryCheckpointer()
        run_id = uuid4()
        checkpoint = AgentCheckpoint(run_id=run_id, state={"step": 3})

        checkpointer.save(checkpoint)
        loaded = checkpointer.load(run_id)

        assert loaded == checkpoint

    def test_load_unknown_run_returns_none(self) -> None:
        """Pinned: unknown run_id returns None, not an exception. Callers
        branch on presence to decide between resume and fresh start."""
        checkpointer = InMemoryCheckpointer()
        assert checkpointer.load(uuid4()) is None


class TestInMemoryCheckpointerLastWriteWins:

    def test_second_save_overwrites_first_for_same_run(self) -> None:
        """Pinned: save is last-write-wins. Each save replaces the prior
        checkpoint for that run_id — matches LangGraph's checkpointer
        semantics, where each save replaces the state-at-that-point."""
        checkpointer = InMemoryCheckpointer()
        run_id = uuid4()
        first = AgentCheckpoint(run_id=run_id, state={"step": 1})
        second = AgentCheckpoint(run_id=run_id, state={"step": 2})

        checkpointer.save(first)
        checkpointer.save(second)

        loaded = checkpointer.load(run_id)
        assert loaded == second

    def test_saves_for_different_runs_are_independent(self) -> None:
        checkpointer = InMemoryCheckpointer()
        run_a = uuid4()
        run_b = uuid4()
        checkpoint_a = AgentCheckpoint(run_id=run_a, state={"k": "A"})
        checkpoint_b = AgentCheckpoint(run_id=run_b, state={"k": "B"})

        checkpointer.save(checkpoint_a)
        checkpointer.save(checkpoint_b)

        assert checkpointer.load(run_a) == checkpoint_a
        assert checkpointer.load(run_b) == checkpoint_b