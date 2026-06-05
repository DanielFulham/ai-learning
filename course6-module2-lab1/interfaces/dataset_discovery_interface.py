from typing import Protocol


class DatasetDiscoveryInterface(Protocol):
    """Finds available datasets in some location."""

    def list_datasets(self) -> list[str]:
        """Return basenames of available datasets. Empty list if none."""
        ...