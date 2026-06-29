import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from langgraph.graph.state import CompiledStateGraph

from application.container import initialise
from domain.agent_checkpoint import AgentCheckpoint
from domain.events.qa_events import QuestionReceived
from infra.in_memory_checkpointer import InMemoryCheckpointer
from infra.sqlite_checkpointer import SqliteCheckpointer
from infra.sqlite_event_store import SqliteEventStore
from interfaces.agent_checkpointer_interface import AgentCheckpointerInterface


def _mock_graph() -> MagicMock:
    return MagicMock(spec=CompiledStateGraph)


def _make_qa_event(run_id) -> QuestionReceived:
    return QuestionReceived(
        event_id=uuid4(),
        aggregate_id=run_id,
        occurred_at=datetime(2026, 6, 23, 12, 0, tzinfo=timezone.utc),
        question="Q",
    )


class TestContainerSharedDbPath:

    def test_same_db_path_passed_to_both_concretes(self, tmp_path: Path) -> None:
        """Pinned: when use_sqlite_persistence=True and neither concrete is
        injected, the event store and checkpointer are both constructed from
        the one db_path the container selected. This is the structural fix
        F07 names — a single source of truth for persistence location."""
        db_path = tmp_path / "events.db"
        app = initialise(
            qa_graph=_mock_graph(),
            use_sqlite_persistence=True,
            db_path=db_path,
        )

        assert isinstance(app.event_store, SqliteEventStore)
        assert isinstance(app.checkpointer, SqliteCheckpointer)
        assert app.event_store._db_path == str(db_path)
        assert app.checkpointer._db_path == str(db_path)

    def test_non_sqlite_branch_produces_in_memory_checkpointer(self) -> None:
        """Pinned: the non-SQLite default wires InMemoryCheckpointer. Pins
        that the checkpointer follows use_sqlite_persistence in lockstep
        with the event store — prevents a future bug where only one of the
        two concretes honours the flag."""
        app = initialise(qa_graph=_mock_graph(), use_sqlite_persistence=False)

        assert isinstance(app.checkpointer, InMemoryCheckpointer)

    def test_explicit_checkpointer_injection_overrides_sqlite_flag(
        self, tmp_path: Path
    ) -> None:
        """Pinned: an injected checkpointer wins over the SQLite flag — the
        injected instance is used directly and no SqliteCheckpointer is
        constructed. Mirrors the event-store override and exercises the D6
        injection seam."""
        injected = MagicMock(spec=AgentCheckpointerInterface)

        with patch(
            "application.container.SqliteCheckpointer",
            spec=SqliteCheckpointer,
        ) as mock_ctor:
            app = initialise(
                qa_graph=_mock_graph(),
                checkpointer=injected,
                use_sqlite_persistence=True,
                db_path=tmp_path / "events.db",
            )

        assert app.checkpointer is injected
        mock_ctor.assert_not_called()


class TestContainerOneDbTwoTables:

    def test_one_db_file_on_disk_after_round_trip(self, tmp_path: Path) -> None:
        """Pinned: event append and checkpoint save land in one SQLite file,
        not two. The shared db_path means a single file with both tables —
        no second file with a different name or suffix appears."""
        db_path = tmp_path / "events.db"
        app = initialise(
            qa_graph=_mock_graph(),
            use_sqlite_persistence=True,
            db_path=db_path,
        )
        run_id = uuid4()

        app.event_store.append(_make_qa_event(run_id))
        app.checkpointer.save(AgentCheckpoint(run_id=run_id, state={"step": 1}))

        db_files = list(tmp_path.glob("*.db"))
        assert len(db_files) == 1
        assert db_files[0].name == "events.db"

    def test_two_tables_exist_in_the_one_db_file(self, tmp_path: Path) -> None:
        """Pinned: the one DB file holds both the events and checkpoints
        tables. Asserts the one-DB-two-tables claim structurally by reading
        sqlite_master directly, not just behaviourally."""
        db_path = tmp_path / "events.db"
        app = initialise(
            qa_graph=_mock_graph(),
            use_sqlite_persistence=True,
            db_path=db_path,
        )
        run_id = uuid4()

        app.event_store.append(_make_qa_event(run_id))
        app.checkpointer.save(AgentCheckpoint(run_id=run_id, state={"step": 1}))

        with sqlite3.connect(str(db_path)) as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        table_names = {row[0] for row in rows}

        assert "events" in table_names
        assert "checkpoints" in table_names


class TestContainerResume:

    def test_checkpoint_resume_across_container_instances(
        self, tmp_path: Path
    ) -> None:
        """F07 release gate: a checkpoint saved via container instance N is
        loaded by a fresh instance N+1 built from the same db_path. The
        container is rebuilt between save and load — no shared Python
        reference — so this proves on-disk resume the way a restarted
        process sees it."""
        db_path = tmp_path / "events.db"
        run_id = uuid4()
        checkpoint = AgentCheckpoint(run_id=run_id, state={"step": 5})

        app_n = initialise(
            qa_graph=_mock_graph(),
            use_sqlite_persistence=True,
            db_path=db_path,
        )
        app_n.checkpointer.save(checkpoint)
        del app_n

        app_n1 = initialise(
            qa_graph=_mock_graph(),
            use_sqlite_persistence=True,
            db_path=db_path,
        )
        loaded = app_n1.checkpointer.load(run_id)

        assert loaded == checkpoint
