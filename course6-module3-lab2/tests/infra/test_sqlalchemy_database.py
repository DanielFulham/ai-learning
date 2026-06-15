import pytest

from infra.sqlalchemy_database import SqlAlchemyDatabase


_TEST_SCHEMA_SQL = """
CREATE TABLE Artist (
    ArtistId INTEGER PRIMARY KEY,
    Name TEXT NOT NULL
);

CREATE TABLE Album (
    AlbumId INTEGER PRIMARY KEY,
    Title TEXT NOT NULL,
    ArtistId INTEGER NOT NULL,
    FOREIGN KEY (ArtistId) REFERENCES Artist (ArtistId)
);

INSERT INTO Artist (ArtistId, Name) VALUES (1, 'AC/DC');
INSERT INTO Artist (ArtistId, Name) VALUES (2, 'Accept');
INSERT INTO Album (AlbumId, Title, ArtistId) VALUES (1, 'Back in Black', 1);
INSERT INTO Album (AlbumId, Title, ArtistId) VALUES (2, 'Balls to the Wall', 2);
"""


@pytest.fixture
def db(tmp_path):
    db_path = tmp_path / "test.db"
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript(_TEST_SCHEMA_SQL)
    conn.commit()
    conn.close()
    return SqlAlchemyDatabase(f"sqlite:///{db_path}")


def test_dialect_returns_sqlite(db):
    assert db.dialect == "sqlite"


def test_get_table_names_returns_all_tables(db):
    assert sorted(db.get_table_names()) == ["Album", "Artist"]


def test_get_table_ddl_returns_create_table_statement(db):
    ddl = db.get_table_ddl(["Artist"])
    assert "CREATE TABLE" in ddl
    assert "Artist" in ddl
    assert "ArtistId" in ddl
    assert "Name" in ddl


def test_get_table_ddl_returns_multiple_tables_separated_by_blank_lines(db):
    ddl = db.get_table_ddl(["Artist", "Album"])
    assert "Artist" in ddl
    assert "Album" in ddl
    assert "FOREIGN KEY" in ddl


def test_get_table_ddl_raises_keyerror_for_unknown_table(db):
    with pytest.raises(KeyError):
        db.get_table_ddl(["NonexistentTable"])


def test_get_sample_rows_returns_canonical_format(db):
    samples = db.get_sample_rows("Artist", 2)
    assert samples.startswith("/*")
    assert samples.endswith("*/")
    assert "2 rows from Artist table:" in samples
    assert "ArtistId" in samples
    assert "Name" in samples
    assert "AC/DC" in samples
    assert "Accept" in samples


def test_get_sample_rows_reports_actual_row_count_not_limit(db):
    samples = db.get_sample_rows("Artist", 100)
    assert "2 rows from Artist table:" in samples


def test_get_sample_rows_respects_limit(db):
    samples = db.get_sample_rows("Artist", 1)
    assert "1 rows from Artist table:" in samples
    assert "AC/DC" in samples
    assert "Accept" not in samples


def test_run_returns_stringified_tuple_results(db):
    result = db.run("SELECT COUNT(*) FROM Album")
    assert result == "[(2,)]"


def test_run_returns_multiple_rows_as_list_of_tuples(db):
    result = db.run("SELECT ArtistId, Name FROM Artist ORDER BY ArtistId")
    assert result == "[(1, 'AC/DC'), (2, 'Accept')]"


def test_run_raises_on_invalid_sql(db):
    from sqlalchemy.exc import OperationalError
    with pytest.raises(OperationalError):
        db.run("SELECT * FROM NoSuchTable")
        

@pytest.fixture
def read_only_db(tmp_path):
    db_path = tmp_path / "readonly.db"
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.executescript(_TEST_SCHEMA_SQL)
    conn.commit()
    conn.close()
    return SqlAlchemyDatabase(f"sqlite:///file:{db_path}?mode=ro&uri=true")


def test_read_only_database_permits_select(read_only_db):
    result = read_only_db.run("SELECT COUNT(*) FROM Album")
    assert result == "[(2,)]"


def test_read_only_database_permits_schema_introspection(read_only_db):
    names = read_only_db.get_table_names()
    assert sorted(names) == ["Album", "Artist"]


def test_read_only_database_rejects_insert(read_only_db):
    from sqlalchemy.exc import OperationalError
    with pytest.raises(OperationalError, match="readonly database"):
        read_only_db.run(
            "INSERT INTO Artist (ArtistId, Name) VALUES (99, 'Test')"
        )


def test_read_only_database_rejects_update(read_only_db):
    from sqlalchemy.exc import OperationalError
    with pytest.raises(OperationalError, match="readonly database"):
        read_only_db.run("UPDATE Artist SET Name = 'Changed' WHERE ArtistId = 1")


def test_read_only_database_rejects_delete(read_only_db):
    from sqlalchemy.exc import OperationalError
    with pytest.raises(OperationalError, match="readonly database"):
        read_only_db.run("DELETE FROM Artist WHERE ArtistId = 1")


def test_read_only_database_rejects_drop_table(read_only_db):
    from sqlalchemy.exc import OperationalError
    with pytest.raises(OperationalError, match="readonly database"):
        read_only_db.run("DROP TABLE Artist")