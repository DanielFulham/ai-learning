"""Tests for CachedDatasetLoader — the cache decorator.

The decorator's correctness is mostly about behaviour relative to the wrapped
loader, not about the data itself. So most tests here mock the inner loader
and verify call patterns: first call hits the inner; subsequent calls don't.
"""
from unittest.mock import MagicMock

import pandas as pd
import pytest

from infra.cached_dataset_loader import CachedDatasetLoader
from interfaces.dataset_loader_interface import DatasetLoaderInterface


def _make_mock_loader(return_df=None):
    """Build a MagicMock satisfying DatasetLoaderInterface."""
    mock = MagicMock(spec=DatasetLoaderInterface)
    mock.load.return_value = return_df if return_df is not None else pd.DataFrame({"a": [1]})
    return mock


def test_first_load_calls_through_to_inner():
    inner = _make_mock_loader()
    cache = CachedDatasetLoader(inner=inner)

    cache.load("data.csv")

    inner.load.assert_called_once_with("data.csv")


def test_second_load_of_same_file_does_not_call_inner_again():
    inner = _make_mock_loader()
    cache = CachedDatasetLoader(inner=inner)

    cache.load("data.csv")
    cache.load("data.csv")

    assert inner.load.call_count == 1


def test_different_filenames_are_cached_independently():
    inner = _make_mock_loader()
    cache = CachedDatasetLoader(inner=inner)

    cache.load("first.csv")
    cache.load("second.csv")
    cache.load("first.csv")
    cache.load("second.csv")

    assert inner.load.call_count == 2
    inner.load.assert_any_call("first.csv")
    inner.load.assert_any_call("second.csv")


def test_load_returns_same_dataframe_instance_on_cache_hit():
    df = pd.DataFrame({"a": [1, 2, 3]})
    inner = _make_mock_loader(return_df=df)
    cache = CachedDatasetLoader(inner=inner)

    first = cache.load("data.csv")
    second = cache.load("data.csv")

    assert first is second
    assert first is df


def test_load_propagates_exceptions_from_inner_loader():
    inner = MagicMock(spec=DatasetLoaderInterface)
    inner.load.side_effect = FileNotFoundError("Dataset 'missing.csv' not found in /data")
    cache = CachedDatasetLoader(inner=inner)

    with pytest.raises(FileNotFoundError):
        cache.load("missing.csv")

    with pytest.raises(FileNotFoundError):
        cache.load("missing.csv")

    assert inner.load.call_count == 2


def test_cache_is_instance_scoped_not_class_scoped():
    inner_a = _make_mock_loader()
    inner_b = _make_mock_loader()

    cache_a = CachedDatasetLoader(inner=inner_a)
    cache_b = CachedDatasetLoader(inner=inner_b)

    cache_a.load("data.csv")
    cache_b.load("data.csv")

    inner_a.load.assert_called_once_with("data.csv")
    inner_b.load.assert_called_once_with("data.csv")