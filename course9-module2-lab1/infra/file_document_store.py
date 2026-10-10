from pathlib import Path


class FileDocumentStore:
    """Documents held as files directly under one root directory."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    def read(self, name: str) -> str | None:
        target = (self._root / name).resolve()
        # Containment check: `..` segments or an absolute name must not leave the root.
        if not target.is_relative_to(self._root) or not target.is_file():
            return None
        return target.read_text(encoding="utf-8")
