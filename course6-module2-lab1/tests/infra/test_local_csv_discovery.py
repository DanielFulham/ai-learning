"""Tests for LocalCsvDiscovery — filesystem-based CSV discovery."""
from pathlib import Path

from infra.local_csv_discovery import LocalCsvDiscovery


def test_list_datasets_returns_csv_files_in_directory(tmp_path: Path) -> None:
    """Given a directory with CSV files, list_datasets returns their basenames."""
    (tmp_path / "first.csv").write_text("col1,col2\n1,2")
    (tmp_path / "second.csv").write_text("col1,col2\n3,4")

    discovery = LocalCsvDiscovery(data_dir=tmp_path)

    result = discovery.list_datasets()

    assert sorted(result) == ["first.csv", "second.csv"]


def test_list_datasets_ignores_non_csv_files(tmp_path: Path) -> None:
    """Files that aren't CSVs should not appear in the result."""
    (tmp_path / "data.csv").write_text("col1\n1")
    (tmp_path / "notes.txt").write_text("not a csv")
    (tmp_path / "config.json").write_text("{}")

    discovery = LocalCsvDiscovery(data_dir=tmp_path)

    result = discovery.list_datasets()

    assert result == ["data.csv"]


def test_list_datasets_returns_empty_list_when_directory_has_no_csvs(tmp_path: Path) -> None:
    """An empty (or non-CSV-containing) directory returns an empty list, not None."""
    discovery = LocalCsvDiscovery(data_dir=tmp_path)

    result = discovery.list_datasets()

    assert result == []


def test_list_datasets_returns_basenames_not_full_paths(tmp_path: Path) -> None:
    """The contract is to return filenames, not absolute paths."""
    (tmp_path / "data.csv").write_text("col1\n1")

    discovery = LocalCsvDiscovery(data_dir=tmp_path)

    result = discovery.list_datasets()

    assert result == ["data.csv"]
    assert "/" not in result[0]
    assert "\\" not in result[0]