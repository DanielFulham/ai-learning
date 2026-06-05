"""Tests for SklearnRegressionEvaluator — Random Forest regressor on a DataFrame."""
import pandas as pd
import pytest

from infra.sklearn_regression_evaluator import SklearnRegressionEvaluator
from domain.models import RegressionMetrics


def _make_linear_df(rows=200):
    """Build a DataFrame where target = 2 * feature_1 + 3 * feature_2 + noise.

    Linear, learnable, should produce high R² and low MSE.
    """
    return pd.DataFrame({
        "feature_1": list(range(rows)),
        "feature_2": [i * 0.5 for i in range(rows)],
        "target": [2.0 * i + 3.0 * (i * 0.5) for i in range(rows)],
    })


def test_evaluate_returns_regression_metrics():
    df = _make_linear_df()
    evaluator = SklearnRegressionEvaluator()

    metrics = evaluator.evaluate(df, target_column="target")

    assert isinstance(metrics, RegressionMetrics)
    assert isinstance(metrics.r2, float)
    assert isinstance(metrics.mse, float)


def test_evaluate_mse_is_non_negative():
    """MSE is a squared metric — cannot be negative."""
    df = _make_linear_df()
    evaluator = SklearnRegressionEvaluator()

    metrics = evaluator.evaluate(df, target_column="target")

    assert metrics.mse >= 0.0


def test_evaluate_returns_high_r2_on_linear_data():
    """A near-linear target should produce R² close to 1.0."""
    df = _make_linear_df()
    evaluator = SklearnRegressionEvaluator()

    metrics = evaluator.evaluate(df, target_column="target")

    assert metrics.r2 >= 0.9


def test_evaluate_is_deterministic_with_default_random_state():
    df = _make_linear_df()
    evaluator = SklearnRegressionEvaluator()

    first = evaluator.evaluate(df, target_column="target")
    second = evaluator.evaluate(df, target_column="target")

    assert first.r2 == second.r2
    assert first.mse == second.mse


def test_evaluate_raises_value_error_when_target_column_missing():
    df = _make_linear_df()
    evaluator = SklearnRegressionEvaluator()

    with pytest.raises(ValueError) as exc_info:
        evaluator.evaluate(df, target_column="not_a_column")

    assert "not_a_column" in str(exc_info.value)


def test_evaluate_error_message_lists_available_columns():
    df = _make_linear_df()
    evaluator = SklearnRegressionEvaluator()

    with pytest.raises(ValueError) as exc_info:
        evaluator.evaluate(df, target_column="bad_target")

    assert "feature_1" in str(exc_info.value)
    assert "feature_2" in str(exc_info.value)
    assert "target" in str(exc_info.value)


def test_regression_metrics_is_frozen():
    metrics = RegressionMetrics(r2=0.5, mse=10.0)

    with pytest.raises(Exception):
        setattr(metrics, "r2", 0.99)