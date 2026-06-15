from unittest.mock import MagicMock, patch

from langchain_core.language_models import BaseChatModel

from application.container import initialise
from application.sql_agent import SqlAgent
from interfaces.chat_model_provider_interface import ChatModelProviderInterface
from interfaces.sql_database_interface import SqlDatabaseInterface


def _make_provider() -> MagicMock:
    provider = MagicMock(spec=ChatModelProviderInterface)
    provider.create.return_value = MagicMock(spec=BaseChatModel)
    return provider


def _make_database(dialect: str = "sqlite") -> MagicMock:
    db = MagicMock(spec=SqlDatabaseInterface)
    db.dialect = dialect
    db.get_table_names.return_value = ["Album", "Artist"]
    return db


def test_initialise_returns_a_sql_agent():
    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        agent = initialise(
            database_uri="ignored",
            chat_model_provider=_make_provider(),
            sql_database=_make_database(),
        )

    assert isinstance(agent, SqlAgent)


def test_initialise_calls_provider_create_exactly_once():
    provider = _make_provider()

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        initialise(
            database_uri="ignored",
            chat_model_provider=provider,
            sql_database=_make_database(),
        )

    provider.create.assert_called_once()


def test_initialise_passes_database_dialect_through_to_agent():
    db = _make_database(dialect="postgresql")

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        initialise(
            database_uri="ignored",
            chat_model_provider=_make_provider(),
            sql_database=db,
        )

    system_prompt = mock_create.call_args.kwargs["system_prompt"]
    assert "postgresql" in system_prompt


def test_initialise_wires_four_tools():
    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        initialise(
            database_uri="ignored",
            chat_model_provider=_make_provider(),
            sql_database=_make_database(),
        )

    tools = mock_create.call_args.args[1]
    tool_names = sorted(t.name for t in tools)
    assert tool_names == sorted([
        "sql_db_list_tables",
        "sql_db_schema",
        "sql_db_query",
        "sql_db_query_checker",
    ])


def test_initialise_returns_a_fresh_agent_per_call():
    provider = _make_provider()
    db = _make_database()

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        first = initialise(
            database_uri="ignored",
            chat_model_provider=provider,
            sql_database=db,
        )
        second = initialise(
            database_uri="ignored",
            chat_model_provider=provider,
            sql_database=db,
        )

    assert first is not second


def test_initialise_returns_a_fresh_trace_per_call():
    provider = _make_provider()
    db = _make_database()

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        first = initialise(
            database_uri="ignored",
            chat_model_provider=provider,
            sql_database=db,
        )
        second = initialise(
            database_uri="ignored",
            chat_model_provider=provider,
            sql_database=db,
        )

    assert first.trace is not second.trace


def test_initialise_constructs_default_provider_when_none_given(monkeypatch):
    construct_calls = []

    class _FakeProvider:
        def __init__(self):
            construct_calls.append(self)

        def create(self):
            return MagicMock(spec=BaseChatModel)

    monkeypatch.setattr(
        "application.container.OpenAIChatModelProvider", _FakeProvider
    )

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        initialise(
            database_uri="ignored",
            sql_database=_make_database(),
        )

    assert len(construct_calls) == 1


def test_initialise_constructs_default_database_when_none_given(monkeypatch):
    construct_calls = []

    class _FakeDatabase:
        def __init__(self, uri):
            construct_calls.append(uri)
            self.dialect = "sqlite"

        def get_table_names(self):
            return ["Album"]

    monkeypatch.setattr(
        "application.container.SqlAlchemyDatabase", _FakeDatabase
    )

    with patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        initialise(
            database_uri="sqlite:///test.db",
            chat_model_provider=_make_provider(),
        )

    assert construct_calls == ["sqlite:///test.db"]


def test_initialise_uses_injected_database_over_uri():
    db = _make_database()

    with patch("application.container.SqlAlchemyDatabase") as mock_db_cls, \
         patch("application.sql_agent.create_agent") as mock_create:
        mock_create.return_value = MagicMock()
        initialise(
            database_uri="sqlite:///would-fail.db",
            chat_model_provider=_make_provider(),
            sql_database=db,
        )

    mock_db_cls.assert_not_called()