from typing import Protocol

import pandas as pd

from domain.models import RegressionMetrics


class RegressionEvaluatorInterface(Protocol):
    """Trains and evaluates a regressor on a DataFrame."""

    def evaluate(self, df: pd.DataFrame, target_column: str) -> RegressionMetrics:
        """
        Train on 80% of the data, test on 20%, return R² and MSE.

        Raises ValueError if target_column is not in df.columns.
        """
        ...