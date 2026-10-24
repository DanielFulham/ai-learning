from typing import Protocol


class PathOutsideRootError(Exception):
    """Raised when a path resolves to somewhere outside the store's root folder."""


class FileStoreInterface(Protocol):
    def exists(self, relative_path: str) -> bool: ...

    def read_text(self, relative_path: str) -> str: ...

    def write_text(self, relative_path: str, content: str) -> None: ...

    def delete(self, relative_path: str) -> None: ...
