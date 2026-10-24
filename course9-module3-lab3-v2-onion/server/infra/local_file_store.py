from pathlib import Path

from interfaces.file_store_interface import FileStoreInterface, PathOutsideRootError


def _without_server_path(error: OSError, relative_path: str) -> OSError:
    """The same kind of error, worded with the caller's relative path.

    The operating system's own message carries the absolute path, which would tell any client
    or model where the server keeps its files.
    """
    if isinstance(error, FileNotFoundError):
        return FileNotFoundError(f"File not found: {relative_path}")
    if isinstance(error, PermissionError):
        return PermissionError(f"Permission denied: {relative_path}")
    return OSError(f"Could not access: {relative_path}")


class LocalFileStore(FileStoreInterface):
    """Reads, writes and deletes single text files on local disk, confined to one root folder.

    Every path is resolved first and must land inside the root, so `..`, absolute paths and
    links that point elsewhere are refused. Only files are handled: a folder, including the
    root itself, is never read, written or deleted. Errors name the relative path only.
    """

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, relative_path: str) -> Path:
        resolved = (self._root / relative_path).resolve()
        if not resolved.is_relative_to(self._root):
            raise PathOutsideRootError(f"Path is outside the server root: {relative_path}")
        return resolved

    def _resolve_file(self, relative_path: str) -> Path:
        resolved = self._resolve(relative_path)
        if resolved.is_dir():
            raise IsADirectoryError(f"Path is a folder, not a file: {relative_path}")
        return resolved

    def exists(self, relative_path: str) -> bool:
        return self._resolve(relative_path).is_file()

    def read_text(self, relative_path: str) -> str:
        resolved = self._resolve_file(relative_path)
        try:
            return resolved.read_text(encoding="utf-8")
        except OSError as error:
            raise _without_server_path(error, relative_path) from None

    def write_text(self, relative_path: str, content: str) -> None:
        resolved = self._resolve_file(relative_path)
        try:
            resolved.parent.mkdir(parents=True, exist_ok=True)
            resolved.write_text(content, encoding="utf-8")
        except OSError as error:
            raise _without_server_path(error, relative_path) from None

    def delete(self, relative_path: str) -> None:
        resolved = self._resolve_file(relative_path)
        try:
            resolved.unlink()
        except OSError as error:
            raise _without_server_path(error, relative_path) from None
