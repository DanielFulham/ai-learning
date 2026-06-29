from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

import demo
from application.interfaces.auth_agent_service_interface import (
    AuthAgentServiceInterface,
)
from application.lab_app import LabApp
from domain.qa_exchange import QAExchange
from domain.run_result import RunResult
from interfaces.agent_checkpointer_interface import AgentCheckpointerInterface
from interfaces.agent_event_store_interface import AgentEventStoreInterface


def _patch_initialise(mock_app: LabApp):
    return patch("demo.initialise", return_value=mock_app)


def _make_mock_app() -> tuple[LabApp, MagicMock]:
    """Returns the LabApp and a reference to the qa-service mock so
    tests can assert on .run.call_count etc. without pyright flagging
    MagicMock attribute access through the Protocol type."""
    qa_service = MagicMock()
    qa_service.run.side_effect = [
        RunResult(
            exchange=QAExchange(question=q, answer="A"),
            run_id=uuid4(),
        )
        for q in demo._CANNED_QUESTIONS
    ]
    event_store = MagicMock(spec=AgentEventStoreInterface)
    event_store.events_for_run.return_value = []
    return (
        LabApp(
            qa=qa_service,
            event_store=event_store,
            checkpointer=MagicMock(spec=AgentCheckpointerInterface),
            auth=MagicMock(spec=AuthAgentServiceInterface),
        ),
        qa_service,
    )


class TestParseArgs:

    def test_defaults(self) -> None:
        args = demo._parse_args([])
        assert args.provider == "ollama"
        assert args.persistence == "memory"
        assert args.db_path is None
        assert args.quiet is False

    def test_provider_openai(self) -> None:
        args = demo._parse_args(["--provider", "openai"])
        assert args.provider == "openai"

    def test_persistence_sqlite_with_db_path(self) -> None:
        args = demo._parse_args(
            ["--persistence", "sqlite", "--db-path", "runs.db"]
        )
        assert args.persistence == "sqlite"
        assert args.db_path == Path("runs.db")

    def test_quiet_flag(self) -> None:
        args = demo._parse_args(["--quiet"])
        assert args.quiet is True

    def test_invalid_provider_rejected(self) -> None:
        with pytest.raises(SystemExit):
            demo._parse_args(["--provider", "anthropic"])

    def test_invalid_persistence_rejected(self) -> None:
        with pytest.raises(SystemExit):
            demo._parse_args(["--persistence", "redis"])


class TestMain:

    def test_main_returns_zero_on_success(self) -> None:
        app, _ = _make_mock_app()
        with _patch_initialise(app):
            assert demo.main([]) == 0

    def test_main_runs_all_canned_questions(self) -> None:
        """Pinned: the demo runs every canned question, in order."""
        app, qa_mock = _make_mock_app()
        with _patch_initialise(app):
            demo.main([])

        assert qa_mock.run.call_count == len(demo._CANNED_QUESTIONS)
        for i, expected in enumerate(demo._CANNED_QUESTIONS):
            call_args = qa_mock.run.call_args_list[i].args
            assert call_args[0] == expected

    def test_sqlite_persistence_defaults_db_path_to_events_db(self) -> None:
        """Pinned: --persistence sqlite without --db-path defaults to
        ./events.db rather than raising."""
        app, _ = _make_mock_app()
        with patch("demo.initialise") as mock_init:
            mock_init.return_value = app
            demo.main(["--persistence", "sqlite"])

            init_kwargs = mock_init.call_args.kwargs
            assert init_kwargs["use_sqlite_persistence"] is True
            assert init_kwargs["db_path"] == Path("events.db")

    def test_quiet_flag_disables_console_consumer(self) -> None:
        app, _ = _make_mock_app()
        with patch("demo.initialise") as mock_init:
            mock_init.return_value = app
            demo.main(["--quiet"])

            init_kwargs = mock_init.call_args.kwargs
            assert init_kwargs["use_console_consumer"] is False

    def test_openai_provider_flag_maps_to_use_openai(self) -> None:
        app, _ = _make_mock_app()
        with patch("demo.initialise") as mock_init:
            mock_init.return_value = app
            demo.main(["--provider", "openai"])

            init_kwargs = mock_init.call_args.kwargs
            assert init_kwargs["use_openai"] is True