from langchain.tools import tool
from langchain_core.tools import BaseTool

from interfaces.sql_database_interface import SqlDatabaseInterface

_SAMPLE_ROW_LIMIT = 3


def make_get_schema(db: SqlDatabaseInterface) -> BaseTool:
    @tool("sql_db_schema")
    def sql_db_schema(table_names: str) -> str:
        """Input is a comma-separated list of tables, output is the schema and sample rows for those tables. Be sure that the tables actually exist by calling sql_db_list_tables first!"""
        tables = [name.strip() for name in table_names.split(",") if name.strip()]

        valid_tables = set(db.get_table_names())
        invalid = [t for t in tables if t not in valid_tables]
        if invalid:
            return (
                f"Error: table(s) {invalid} not found. "
                f"Available tables: {sorted(valid_tables)}"
            )

        ddl = db.get_table_ddl(tables)
        samples = "\n\n".join(
            db.get_sample_rows(table, _SAMPLE_ROW_LIMIT) for table in tables
        )
        return f"{ddl}\n\n{samples}"

    return sql_db_schema