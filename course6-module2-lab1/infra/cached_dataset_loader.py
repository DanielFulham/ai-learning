import pandas as pd

from interfaces.dataset_loader_interface import DatasetLoaderInterface


class CachedDatasetLoader(DatasetLoaderInterface):
    """Decorator over a DatasetLoaderInterface that adds in-memory caching.
    
    Satisfies the same interface as the loader it wraps, so consumers don't know
    or care that caching is happening. To remove caching, remove this from the
    composition root — application code doesn't change.
    
    Not thread-safe. Single-process, single-agent use only. Production version
    would back this with functools.lru_cache(maxsize=N) or a Redis client behind
    the same interface.
    """

    def __init__(self, inner: DatasetLoaderInterface) -> None:
        self._inner = inner
        self._cache: dict[str, pd.DataFrame] = {}

    def load(self, name: str) -> pd.DataFrame:
        if name not in self._cache:
            self._cache[name] = self._inner.load(name)
        return self._cache[name]