"""Tests for domain models — Pydantic field validation and contract pinning.

These models cross the LLM tool-call boundary. The validation constraints
(ge/le bounds, Literal status discriminators, required fields) are the runtime
safety net that catches malformed tool returns before they reach the agent.
"""
import pytest
from pydantic import ValidationError

from domain.models import (
    ClassificationResult,
    DataFrameMethodResult,
    DatasetSummariesResult,
    DatasetSummary,
    DiscoveryResult,
    RegressionResult,
    ToolError,
)


# ---------------------------------------------------------------------------
# ToolError
# ---------------------------------------------------------------------------

def test_tool_error_status_defaults_to_error():
    err = ToolError(message="something went wrong")

    assert err.status == "error"


def test_tool_error_requires_message():
    with pytest.raises(ValidationError):
        ToolError()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# DiscoveryResult
# ---------------------------------------------------------------------------

def test_discovery_result_status_defaults_to_ok():
    result = DiscoveryResult(files=["a.csv", "b.csv"])

    assert result.status == "ok"


def test_discovery_result_accepts_empty_file_list():
    result = DiscoveryResult(files=[])

    assert result.files == []


# ---------------------------------------------------------------------------
# DatasetSummary / DatasetSummariesResult
# ---------------------------------------------------------------------------

def test_dataset_summary_rejects_negative_row_count():
    with pytest.raises(ValidationError):
        DatasetSummary(
            file_name="data.csv",
            column_names=["a"],
            data_types={"a": "int64"},
            row_count=-1,
        )


def test_dataset_summary_accepts_zero_row_count():
    summary = DatasetSummary(
        file_name="empty.csv",
        column_names=[],
        data_types={},
        row_count=0,
    )

    assert summary.row_count == 0


def test_dataset_summaries_result_status_defaults_to_ok():
    summary = DatasetSummary(
        file_name="data.csv",
        column_names=["a"],
        data_types={"a": "int64"},
        row_count=10,
    )
    result = DatasetSummariesResult(summaries=[summary])

    assert result.status == "ok"


# ---------------------------------------------------------------------------
# DataFrameMethodResult
# ---------------------------------------------------------------------------

def test_dataframe_method_result_status_defaults_to_ok():
    result = DataFrameMethodResult(
        file_name="data.csv",
        method="head",
        output="...",
    )

    assert result.status == "ok"


# ---------------------------------------------------------------------------
# ClassificationResult
# ---------------------------------------------------------------------------

def test_classification_result_accepts_valid_accuracy():
    result = ClassificationResult(accuracy=0.85)

    assert result.accuracy == 0.85
    assert result.status == "ok"


def test_classification_result_rejects_negative_accuracy():
    with pytest.raises(ValidationError):
        ClassificationResult(accuracy=-0.1)


def test_classification_result_rejects_accuracy_above_one():
    with pytest.raises(ValidationError):
        ClassificationResult(accuracy=1.5)


def test_classification_result_accepts_boundary_values():
    """Accuracy of exactly 0.0 and 1.0 are valid."""
    zero = ClassificationResult(accuracy=0.0)
    one = ClassificationResult(accuracy=1.0)

    assert zero.accuracy == 0.0
    assert one.accuracy == 1.0


# ---------------------------------------------------------------------------
# RegressionResult
# ---------------------------------------------------------------------------

def test_regression_result_accepts_negative_r2():
    """R² can be negative — worse than predicting the mean."""
    result = RegressionResult(r2_score=-0.5, mean_squared_error=10.0)

    assert result.r2_score == -0.5


def test_regression_result_rejects_negative_mse():
    """MSE is squared, must be non-negative."""
    with pytest.raises(ValidationError):
        RegressionResult(r2_score=0.8, mean_squared_error=-1.0)


def test_regression_result_accepts_zero_mse():
    """MSE of exactly 0.0 is valid (perfect prediction)."""
    result = RegressionResult(r2_score=1.0, mean_squared_error=0.0)

    assert result.mean_squared_error == 0.0