from typing import Protocol


class PathOutsideRootError(ValueError):
    """Raised when a requested path resolves outside the store's root."""


class FileStoreInterface(Protocol):
    """File access under one root folder. Every path is relative to that root.

    Every method raises PathOutsideRootError for a path that leaves the root.
    """

    def exists(self, path: str) -> bool: ...

    def read_text(self, path: str) -> str: ...

    def write_text(self, path: str, content: str) -> None: ...
