from langchain.tools import tool
from langchain_core.tools import BaseTool

from interfaces.sql_database_interface import SqlDatabaseInterface


def make_run_query(db: SqlDatabaseInterface) -> BaseTool:
    @tool("sql_db_query")
    def sql_db_query(query: str) -> str:
        """Input to this tool is a detailed and correct SQL query, output is a result from the database. If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again. If you encounter an issue with Unknown column 'xxxx' in 'field list', use sql_db_schema to query the correct table fields."""
        try:
            return db.run(query)
        except Exception as exc:
            return f"Error: {exc}"

    return sql_db_query