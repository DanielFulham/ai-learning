from unittest.mock import MagicMock

from langchain_core.tools import BaseTool

from application.tools.run_query import make_run_query
from interfaces.sql_database_interface import SqlDatabaseInterface


def test_make_run_query_returns_a_base_tool():
    db = MagicMock(spec=SqlDatabaseInterface)
    tool = make_run_query(db)
    assert isinstance(tool, BaseTool)


def test_tool_is_named_sql_db_query():
    db = MagicMock(spec=SqlDatabaseInterface)
    tool = make_run_query(db)
    assert tool.name == "sql_db_query"


def test_tool_description_matches_canonical_toolkit():
    db = MagicMock(spec=SqlDatabaseInterface)
    tool = make_run_query(db)
    assert tool.description == (
        "Input to this tool is a detailed and correct SQL query, output is "
        "a result from the database. If the query is not correct, an error "
        "message will be returned. If an error is returned, rewrite the "
        "query, check the query, and try again."
    )


def test_invocation_delegates_to_database_run():
    db = MagicMock(spec=SqlDatabaseInterface)
    db.run.return_value = "[(347,)]"
    tool = make_run_query(db)

    result = tool.invoke({"query": "SELECT COUNT(*) FROM Album"})

    assert result == "[(347,)]"
    db.run.assert_called_once_with("SELECT COUNT(*) FROM Album")


def test_invocation_returns_database_result_unchanged():
    db = MagicMock(spec=SqlDatabaseInterface)
    db.run.return_value = "[('AC/DC', 1), ('Accept', 2)]"
    tool = make_run_query(db)

    result = tool.invoke({"query": "SELECT Name, AlbumId FROM ..."})

    assert result == "[('AC/DC', 1), ('Accept', 2)]"


def test_invocation_returns_error_string_when_database_raises():
    db = MagicMock(spec=SqlDatabaseInterface)
    db.run.side_effect = RuntimeError("no such table: NonExistent")
    tool = make_run_query(db)

    result = tool.invoke({"query": "SELECT * FROM NonExistent"})

    assert result == "Error: no such table: NonExistent"


def test_invocation_catches_arbitrary_exception_types():
    db = MagicMock(spec=SqlDatabaseInterface)
    db.run.side_effect = ValueError("malformed input")
    tool = make_run_query(db)

    result = tool.invoke({"query": "SELECT garbage"})

    assert result.startswith("Error:")
    assert "malformed input" in result


def test_invocation_does_not_swallow_keyboard_interrupt():
    db = MagicMock(spec=SqlDatabaseInterface)
    db.run.side_effect = KeyboardInterrupt()
    tool = make_run_query(db)

    try:
        tool.invoke({"query": "SELECT 1"})
    except KeyboardInterrupt:
        return
    assert False, "expected KeyboardInterrupt to propagate"