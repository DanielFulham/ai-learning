from typing import Protocol

import pandas as pd


class DatasetLoaderInterface(Protocol):
    """Loads a dataset by name into a pandas DataFrame."""

    def load(self, name: str) -> pd.DataFrame:
        """Load the dataset. Raises FileNotFoundError if missing."""
        ...