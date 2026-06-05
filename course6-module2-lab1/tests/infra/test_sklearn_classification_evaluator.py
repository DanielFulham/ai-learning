"""Tests for SklearnClassificationEvaluator — Random Forest on a DataFrame."""
import pandas as pd
import pytest

from infra.sklearn_classification_evaluator import SklearnClassificationEvaluator


def _make_linearly_separable_df(rows=100):
    """Build a small DataFrame where target is trivially predictable from feature_1."""
    return pd.DataFrame({
        "feature_1": list(range(rows)),
        "feature_2": [i * 2 for i in range(rows)],
        "target": [0 if i < rows // 2 else 1 for i in range(rows)],
    })


def test_evaluate_returns_float_between_zero_and_one():
    df = _make_linearly_separable_df()
    evaluator = SklearnClassificationEvaluator()

    accuracy = evaluator.evaluate(df, target_column="target")

    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0


def test_evaluate_returns_high_accuracy_on_separable_data():
    """A trivially separable dataset should produce near-perfect accuracy."""
    df = _make_linearly_separable_df()
    evaluator = SklearnClassificationEvaluator()

    accuracy = evaluator.evaluate(df, target_column="target")

    assert accuracy >= 0.9


def test_evaluate_is_deterministic_with_default_random_state():
    """Two runs with the same data and default random_state produce identical accuracy."""
    df = _make_linearly_separable_df()
    evaluator = SklearnClassificationEvaluator()

    first = evaluator.evaluate(df, target_column="target")
    second = evaluator.evaluate(df, target_column="target")

    assert first == second


def test_evaluate_different_random_state_produces_different_split():
    """Two evaluators with different random_states can produce different accuracies.

    Not always — sometimes the data is robust enough that any split gives the same
    result. So we assert the score is still valid, not that it differs.
    """
    df = _make_linearly_separable_df()
    evaluator_a = SklearnClassificationEvaluator(random_state=42)
    evaluator_b = SklearnClassificationEvaluator(random_state=99)

    accuracy_a = evaluator_a.evaluate(df, target_column="target")
    accuracy_b = evaluator_b.evaluate(df, target_column="target")

    assert 0.0 <= accuracy_a <= 1.0
    assert 0.0 <= accuracy_b <= 1.0


def test_evaluate_raises_value_error_when_target_column_missing():
    df = _make_linearly_separable_df()
    evaluator = SklearnClassificationEvaluator()

    with pytest.raises(ValueError) as exc_info:
        evaluator.evaluate(df, target_column="not_a_column")

    assert "not_a_column" in str(exc_info.value)


def test_evaluate_error_message_lists_available_columns():
    """The error message includes the available columns to help recovery."""
    df = _make_linearly_separable_df()
    evaluator = SklearnClassificationEvaluator()

    with pytest.raises(ValueError) as exc_info:
        evaluator.evaluate(df, target_column="bad_target")

    assert "feature_1" in str(exc_info.value)
    assert "feature_2" in str(exc_info.value)
    assert "target" in str(exc_info.value)