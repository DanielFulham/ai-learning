from sqlalchemy import MetaData, create_engine, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.schema import CreateTable


class SqlAlchemyDatabase:
    def __init__(self, uri: str) -> None:
        self._engine: Engine = create_engine(uri)
        self._metadata = MetaData()
        self._metadata.reflect(bind=self._engine)

    @property
    def dialect(self) -> str:
        return self._engine.dialect.name

    def get_table_names(self) -> list[str]:
        return inspect(self._engine).get_table_names()

    def get_table_ddl(self, tables: list[str]) -> str:
        ddls: list[str] = []
        for table_name in tables:
            table = self._metadata.tables[table_name]
            ddl = str(CreateTable(table).compile(self._engine)).strip()
            ddls.append(ddl)
        return "\n\n".join(ddls)

    def get_sample_rows(self, table: str, limit: int) -> str:
        with self._engine.connect() as conn:
            result = conn.execute(text(f'SELECT * FROM "{table}" LIMIT {limit}'))
            rows = result.fetchall()
            columns = list(result.keys())

        header = "\t".join(columns)
        body = "\n".join("\t".join(str(c) for c in row) for row in rows)
        return f"/*\n{len(rows)} rows from {table} table:\n{header}\n{body}\n*/"

    def run(self, query: str) -> str:
        with self._engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
        return str([tuple(row) for row in rows])