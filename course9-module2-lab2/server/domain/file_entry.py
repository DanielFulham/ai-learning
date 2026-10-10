from dataclasses import dataclass
from datetime import datetime
from typing import Literal


@dataclass(frozen=True)
class FileEntry:
    """One file or directory under the server's root, with its metadata."""

    name: str
    path: str
    kind: Literal["file", "directory"]
    size: int
    modified: datetime
    created: datetime
