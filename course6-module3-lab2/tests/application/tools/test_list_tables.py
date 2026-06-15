from unittest.mock import MagicMock

from langchain_core.tools import BaseTool

from application.tools.list_tables import make_list_tables
from interfaces.sql_database_interface import SqlDatabaseInterface


def test_make_list_tables_returns_a_base_tool():
    db = MagicMock(spec=SqlDatabaseInterface)
    tool = make_list_tables(db)
    assert isinstance(tool, BaseTool)


def test_tool_is_named_sql_db_list_tables():
    db = MagicMock(spec=SqlDatabaseInterface)
    tool = make_list_tables(db)
    assert tool.name == "sql_db_list_tables"


def test_tool_description_matches_canonical_toolkit():
    db = MagicMock(spec=SqlDatabaseInterface)
    tool = make_list_tables(db)
    assert tool.description == (
        "Input is an empty string, output is a comma-separated list of "
        "tables in the database."
    )


def test_invocation_returns_comma_separated_table_names():
    db = MagicMock(spec=SqlDatabaseInterface)
    db.get_table_names.return_value = ["Album", "Artist", "Customer"]
    tool = make_list_tables(db)

    result = tool.invoke({})

    assert result == "Album, Artist, Customer"


def test_invocation_returns_empty_string_when_database_has_no_tables():
    db = MagicMock(spec=SqlDatabaseInterface)
    db.get_table_names.return_value = []
    tool = make_list_tables(db)

    result = tool.invoke({})

    assert result == ""


def test_invocation_preserves_table_name_order_from_database():
    db = MagicMock(spec=SqlDatabaseInterface)
    db.get_table_names.return_value = ["Zebra", "Apple", "Mango"]
    tool = make_list_tables(db)

    result = tool.invoke({})

    assert result == "Zebra, Apple, Mango"


def test_invocation_calls_get_table_names_once_per_invocation():
    db = MagicMock(spec=SqlDatabaseInterface)
    db.get_table_names.return_value = ["Album"]
    tool = make_list_tables(db)

    tool.invoke({})
    tool.invoke({})

    assert db.get_table_names.call_count == 2