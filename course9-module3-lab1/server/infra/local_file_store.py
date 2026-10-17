from pathlib import Path

from interfaces.file_store_interface import FileStoreInterface, PathOutsideRootError


class LocalFileStore(FileStoreInterface):
    """Reads and writes text files on local disk, confined to one root folder."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def _resolve(self, path: str) -> Path:
        resolved = (self._root / path).resolve()
        if not resolved.is_relative_to(self._root):
            raise PathOutsideRootError(f"Path is outside the server root: {path}")
        return resolved

    def exists(self, path: str) -> bool:
        return self._resolve(path).exists()

    def read_text(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def write_text(self, path: str, content: str) -> None:
        file_path = self._resolve(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")
