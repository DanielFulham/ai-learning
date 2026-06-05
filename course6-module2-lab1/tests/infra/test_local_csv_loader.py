"""Tests for LocalCsvLoader — filesystem-based CSV loading."""
from pathlib import Path

import pandas as pd
import pytest

from infra.local_csv_loader import LocalCsvLoader


def test_load_returns_dataframe_when_file_exists(tmp_path: Path) -> None:
    """A CSV file in the data directory loads into a DataFrame."""
    (tmp_path / "data.csv").write_text("a,b,c\n1,2,3\n4,5,6\n")

    loader = LocalCsvLoader(data_dir=tmp_path)

    result = loader.load("data.csv")

    assert isinstance(result, pd.DataFrame)
    assert result.shape == (2, 3)
    assert result.columns.tolist() == ["a", "b", "c"]


def test_load_preserves_csv_data_correctly(tmp_path: Path) -> None:
    """The loaded DataFrame contains the values from the CSV file."""
    (tmp_path / "data.csv").write_text("name,age\nAlice,30\nBob,25\n")

    loader = LocalCsvLoader(data_dir=tmp_path)

    result = loader.load("data.csv")

    assert result["name"].tolist() == ["Alice", "Bob"]
    assert result["age"].tolist() == [30, 25]


def test_load_raises_file_not_found_when_file_missing(tmp_path: Path) -> None:
    """Loading a non-existent file raises FileNotFoundError with a useful message."""
    loader = LocalCsvLoader(data_dir=tmp_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load("missing.csv")

    assert "missing.csv" in str(exc_info.value)
    assert str(tmp_path) in str(exc_info.value)


def test_load_error_message_uses_our_format_not_pandas_default(tmp_path: Path) -> None:
    """The error message comes from our code, not from pandas' generic error."""
    loader = LocalCsvLoader(data_dir=tmp_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load("missing.csv")

    assert "Dataset" in str(exc_info.value)
    assert "not found in" in str(exc_info.value)


def test_load_traversal_attempt_raises_with_specific_message(tmp_path: Path) -> None:
    """An attempt to escape the data directory raises with a distinct error message."""
    loader = LocalCsvLoader(data_dir=tmp_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        loader.load("../../../etc/passwd")

    assert "outside the allowed data directory" in str(exc_info.value)