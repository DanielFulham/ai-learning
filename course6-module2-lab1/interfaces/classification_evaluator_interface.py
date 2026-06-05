from typing import Protocol

import pandas as pd


class ClassificationEvaluatorInterface(Protocol):
    """Trains and evaluates a classifier on a DataFrame."""

    def evaluate(self, df: pd.DataFrame, target_column: str) -> float:
        """
        Train on 80% of the data, test on 20%, return accuracy.
        
        Raises ValueError if target_column is not in df.columns.
        """
        ...