from dataclasses import dataclass
from typing import Literal, List, Dict
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Domain transport — frozen dataclass, not Pydantic
#
# RegressionMetrics is the two-field carrier returned by RegressionEvaluatorInterface
# and consumed by the regression tool body. It crosses an interface boundary; that's
# what makes it a domain object. Frozen dataclass is the right shape: no LLM boundary
# crossing, no runtime validation requirement, immutability enforced by the type system.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RegressionMetrics:
    """Two-field carrier for regression evaluation output."""
    r2: float
    mse: float


# ---------------------------------------------------------------------------
# Error model — used by every tool
# ---------------------------------------------------------------------------

class ToolError(BaseModel):
    """Returned by any tool when the operation fails."""
    status: Literal["error"] = "error"
    message: str


# ---------------------------------------------------------------------------
# Discovery tool model
# ---------------------------------------------------------------------------

class DiscoveryResult(BaseModel):
    """Returned by the list_csv_files tool on success."""
    status: Literal["ok"] = "ok"
    files: List[str] = Field(description="Basenames of available CSV datasets.")


# ---------------------------------------------------------------------------
# Inspection tool models
# ---------------------------------------------------------------------------

class DatasetSummary(BaseModel):
    """One dataset's metadata."""
    file_name: str
    column_names: List[str]
    data_types: Dict[str, str] = Field(description="Column name -> dtype as string")
    row_count: int = Field(ge=0)


class DatasetSummariesResult(BaseModel):
    """Returned by the get_dataset_summaries tool on success."""
    status: Literal["ok"] = "ok"
    summaries: List[DatasetSummary]


class DataFrameMethodResult(BaseModel):
    """Returned by the call_dataframe_method tool on success."""
    status: Literal["ok"] = "ok"
    file_name: str
    method: str
    output: str = Field(description="String representation of the DataFrame method's result.")


# ---------------------------------------------------------------------------
# Evaluator tool models
# ---------------------------------------------------------------------------

class ClassificationResult(BaseModel):
    """Returned by the evaluate_classification_dataset tool on success."""
    status: Literal["ok"] = "ok"
    accuracy: float = Field(ge=0.0, le=1.0, description="Accuracy on the test split.")


class RegressionResult(BaseModel):
    """Returned by the evaluate_regression_dataset tool on success."""
    status: Literal["ok"] = "ok"
    r2_score: float = Field(description="Coefficient of determination. Can be negative.")
    mean_squared_error: float = Field(ge=0.0, description="MSE on the test split.")