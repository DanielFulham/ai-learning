from pathlib import Path
from uuid import uuid4

from domain.agent_checkpoint import AgentCheckpoint
from infra.sqlite_checkpointer import SqliteCheckpointer
from interfaces.agent_checkpointer_interface import AgentCheckpointerInterface


def _make_checkpointer(tmp_path: Path) -> SqliteCheckpointer:
    return SqliteCheckpointer(tmp_path / "events.db")


def _accepts_checkpointer(checkpointer: AgentCheckpointerInterface) -> None:
    """Type-guard helper. Pyright fails the call site if structural typing
    breaks."""


class TestSqliteCheckpointerInterfaceSatisfaction:

    def test_satisfies_agent_checkpointer_interface(self, tmp_path: Path) -> None:
        checkpointer = _make_checkpointer(tmp_path)
        _accepts_checkpointer(checkpointer)


class TestSqliteCheckpointerSchema:

    def test_construction_is_idempotent(self, tmp_path: Path) -> None:
        """Pinned: constructing the checkpointer twice against the same file
        does not error. CREATE TABLE IF NOT EXISTS handles re-init."""
        db_path = tmp_path / "events.db"
        SqliteCheckpointer(db_path)
        SqliteCheckpointer(db_path)


class TestSqliteCheckpointerSaveLoad:

    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        """Pinned: a saved checkpoint loads back with equal run_id and
        state. The state dict carries a non-trivial shape to exercise the
        JSON round-trip."""
        checkpointer = _make_checkpointer(tmp_path)
        run_id = uuid4()
        checkpoint = AgentCheckpoint(
            run_id=run_id, state={"step": 3, "messages": ["a", "b"]}
        )

        checkpointer.save(checkpoint)
        loaded = checkpointer.load(run_id)

        assert loaded == checkpoint

    def test_load_unknown_run_returns_none(self, tmp_path: Path) -> None:
        """Pinned: unknown run_id returns None, not an exception — matches
        AgentCheckpointerInterface and InMemoryCheckpointer. Callers branch
        on presence to decide between resume and fresh start."""
        checkpointer = _make_checkpointer(tmp_path)
        assert checkpointer.load(uuid4()) is None

    def test_second_save_overwrites_first_for_same_run(self, tmp_path: Path) -> None:
        """Pinned: save is last-write-wins. A second save for the same run_id
        replaces the prior checkpoint — matches InMemoryCheckpointer and
        LangGraph's checkpointer semantics (each save replaces the
        state-at-that-point)."""
        checkpointer = _make_checkpointer(tmp_path)
        run_id = uuid4()
        first = AgentCheckpoint(run_id=run_id, state={"step": 1})
        second = AgentCheckpoint(run_id=run_id, state={"step": 2})

        checkpointer.save(first)
        checkpointer.save(second)

        assert checkpointer.load(run_id) == second


class TestSqliteCheckpointerPersistence:

    def test_save_survives_checkpointer_reconstruction(self, tmp_path: Path) -> None:
        """Pinned: data persists across checkpointer instances on the same
        file. This is what makes it the production concrete —
        InMemoryCheckpointer cannot satisfy this test. It is also the
        on-disk basis for the container-level resume contract."""
        db_path = tmp_path / "events.db"
        run_id = uuid4()
        checkpoint = AgentCheckpoint(run_id=run_id, state={"step": 7})

        first = SqliteCheckpointer(db_path)
        first.save(checkpoint)

        second = SqliteCheckpointer(db_path)
        assert second.load(run_id) == checkpoint
