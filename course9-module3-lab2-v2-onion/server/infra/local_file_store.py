from pathlib import Path
from typing import Literal

from interfaces.file_store_interface import (
    FileStoreInterface,
    PathOutsideRootError,
    RawEntry,
)


class LocalFileStore(FileStoreInterface):
    """Reads and writes text files on local disk, confined to one root folder.

    The check is the lab's: resolve the path, then require it to be inside the root.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise PathOutsideRootError(f"Path is outside the server root: {path}")
        return resolved

    def kind_of(self, path: str) -> Literal["file", "directory", "missing"]:
        resolved = self._resolve(path)
        if resolved.is_file():
            return "file"
        if resolved.is_dir():
            return "directory"
        return "missing"

    def read_text(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")

    def list_entries(self, path: str) -> list[RawEntry]:
        directory = self._resolve(path)
        entries: list[RawEntry] = []
        for item in sorted(directory.iterdir()):
            entries.append(
                {
                    "name": item.name,
                    "path": str(item.relative_to(self._root)),
                    "is_dir": item.is_dir(),
                    "size": item.stat().st_size if item.is_file() else 0,
                }
            )
        return entries
