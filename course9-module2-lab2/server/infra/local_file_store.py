from pathlib import Path
from typing import Literal

from interfaces.file_store_interface import (
    FileStoreInterface,
    PathOutsideRootError,
    RawEntry,
)


class LocalFileStore(FileStoreInterface):
    """Reads and writes files on local disk, confined to one root folder."""

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

    def write_text(self, path: str, content: str, append: bool = False) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with open(resolved, "a" if append else "w", encoding="utf-8") as f:
            f.write(content)

    def delete(self, path: str) -> None:
        resolved = self._resolve(path)
        if (self._root / path).is_symlink():
            raise ValueError(f"Refusing to delete a symbolic link: {path}")
        resolved.unlink()

    def list_entries(self, path: str) -> list[RawEntry]:
        directory = self._resolve(path)
        entries: list[RawEntry] = []
        for item in sorted(directory.iterdir(), key=lambda p: p.name):
            stat = item.stat()
            entries.append(
                {
                    "name": item.name,
                    "path": str(item.relative_to(self._root)),
                    "is_dir": item.is_dir(),
                    "size": stat.st_size,
                    "modified": stat.st_mtime,
                    "created": stat.st_ctime,
                }
            )
        return entries
