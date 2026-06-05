from pathlib import Path

import pandas as pd

from interfaces.dataset_loader_interface import DatasetLoaderInterface


class LocalCsvLoader(DatasetLoaderInterface):
    """Loads CSV files from a local directory."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def load(self, name: str) -> pd.DataFrame:
        path = (self._data_dir / name).resolve()
        if not path.is_relative_to(self._data_dir.resolve()):
            raise FileNotFoundError(
                f"Dataset '{name}' is outside the allowed data directory {self._data_dir}"
            )
        if not path.exists():
            raise FileNotFoundError(f"Dataset '{name}' not found in {self._data_dir}")
        return pd.read_csv(path)