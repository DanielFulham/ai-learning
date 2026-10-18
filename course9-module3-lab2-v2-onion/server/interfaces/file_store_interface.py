from typing import Literal, Protocol, TypedDict


class PathOutsideRootError(ValueError):
    """Raised when a requested path resolves outside the store's root."""


class RawEntry(TypedDict):
    name: str
    path: str
    is_dir: bool
    size: int


class FileStoreInterface(Protocol):
    """File access under one root folder. Every path is relative to that root.

    Every method raises PathOutsideRootError for a path that leaves the root.
    """

    def kind_of(self, path: str) -> Literal["file", "directory", "missing"]: ...

    def read_text(self, path: str) -> str: ...

    def write_text(self, path: str, content: str) -> None: ...

    def list_entries(self, path: str) -> list[RawEntry]:
        """The entries of a folder, sorted by name, with `path` relative to the root."""
        ...
