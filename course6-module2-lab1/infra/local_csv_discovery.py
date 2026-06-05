from pathlib import Path

from interfaces.dataset_discovery_interface import DatasetDiscoveryInterface


class LocalCsvDiscovery(DatasetDiscoveryInterface):
    """Discovers CSV files in a local directory via glob."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir

    def list_datasets(self) -> list[str]:
        return [f.name for f in self._data_dir.glob("*.csv")]