from langchain.tools import tool
from langchain_core.tools import BaseTool

from interfaces.sql_database_interface import SqlDatabaseInterface


def make_list_tables(db: SqlDatabaseInterface) -> BaseTool:
    @tool("sql_db_list_tables")
    def sql_db_list_tables() -> str:
        """Input is an empty string, output is a comma-separated list of tables in the database."""
        return ", ".join(db.get_table_names())

    return sql_db_list_tables