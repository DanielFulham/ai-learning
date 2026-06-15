from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool

from application.tools.get_schema import make_get_schema
from interfaces.sql_database_interface import SqlDatabaseInterface


def _make_db_with_tables(table_names: list[str]) -> MagicMock:
    db = MagicMock(spec=SqlDatabaseInterface)
    db.get_table_names.return_value = table_names
    db.get_table_ddl.return_value = "CREATE TABLE Stub (...)"
    db.get_sample_rows.return_value = "/* 3 rows from Stub table: ... */"
    return db


def test_make_get_schema_returns_a_base_tool():
    db = _make_db_with_tables(["Album"])
    tool = make_get_schema(db)
    assert isinstance(tool, BaseTool)


def test_tool_is_named_sql_db_schema():
    db = _make_db_with_tables(["Album"])
    tool = make_get_schema(db)
    assert tool.name == "sql_db_schema"


def test_tool_description_matches_canonical_toolkit():
    db = _make_db_with_tables(["Album"])
    tool = make_get_schema(db)
    assert tool.description == (
        "Input to this tool is a comma-separated list of tables, output is "
        "the schema and sample rows for those tables. Be sure that the "
        "tables actually exist by calling sql_db_list_tables first! "
        "Example Input: table1, table2, table3"
    )
    

def test_invocation_returns_ddl_followed_by_sample_rows():
    db = _make_db_with_tables(["Album"])
    db.get_table_ddl.return_value = "CREATE TABLE Album (...)"
    db.get_sample_rows.return_value = "/* 3 rows from Album table: ... */"
    tool = make_get_schema(db)

    result = tool.invoke({"table_names": "Album"})

    assert result == "CREATE TABLE Album (...)\n\n/* 3 rows from Album table: ... */"


def test_invocation_parses_comma_separated_table_names():
    db = _make_db_with_tables(["Artist", "Album"])
    tool = make_get_schema(db)

    tool.invoke({"table_names": "Artist, Album"})

    db.get_table_ddl.assert_called_once_with(["Artist", "Album"])


def test_invocation_strips_whitespace_from_table_names():
    db = _make_db_with_tables(["Artist", "Album"])
    tool = make_get_schema(db)

    tool.invoke({"table_names": "  Artist  ,  Album  "})

    db.get_table_ddl.assert_called_once_with(["Artist", "Album"])


def test_invocation_ignores_empty_segments_in_comma_separated_input():
    db = _make_db_with_tables(["Artist"])
    tool = make_get_schema(db)

    tool.invoke({"table_names": "Artist,,"})

    db.get_table_ddl.assert_called_once_with(["Artist"])


def test_invocation_returns_error_string_when_table_does_not_exist():
    db = _make_db_with_tables(["Album", "Artist"])
    tool = make_get_schema(db)

    result = tool.invoke({"table_names": "NonExistent"})

    assert result.startswith("Error:")
    assert "NonExistent" in result
    assert "Album" in result
    assert "Artist" in result


def test_invocation_returns_error_when_any_requested_table_is_invalid():
    db = _make_db_with_tables(["Album"])
    tool = make_get_schema(db)

    result = tool.invoke({"table_names": "Album, NonExistent"})

    assert result.startswith("Error:")
    assert "NonExistent" in result


def test_invocation_does_not_call_ddl_when_validation_fails():
    db = _make_db_with_tables(["Album"])
    tool = make_get_schema(db)

    tool.invoke({"table_names": "NonExistent"})

    db.get_table_ddl.assert_not_called()
    db.get_sample_rows.assert_not_called()


def test_invocation_requests_sample_rows_for_every_valid_table():
    db = _make_db_with_tables(["Artist", "Album"])
    tool = make_get_schema(db)

    tool.invoke({"table_names": "Artist, Album"})

    assert db.get_sample_rows.call_count == 2
    sample_call_args = [call.args[0] for call in db.get_sample_rows.call_args_list]
    assert sample_call_args == ["Artist", "Album"]


def test_invocation_requests_three_sample_rows_per_table():
    db = _make_db_with_tables(["Album"])
    tool = make_get_schema(db)

    tool.invoke({"table_names": "Album"})

    db.get_sample_rows.assert_called_once_with("Album", 3)


@pytest.mark.parametrize("table_names_input,expected_parsed", [
    ("Album", ["Album"]),
    ("Album,Artist", ["Album", "Artist"]),
    ("Album, Artist", ["Album", "Artist"]),
    ("  Album  ", ["Album"]),
    ("Album, , Artist", ["Album", "Artist"]),
])
def test_table_name_parsing_handles_various_input_shapes(
    table_names_input, expected_parsed
):
    db = _make_db_with_tables(["Album", "Artist"])
    tool = make_get_schema(db)

    tool.invoke({"table_names": table_names_input})

    db.get_table_ddl.assert_called_once_with(expected_parsed)