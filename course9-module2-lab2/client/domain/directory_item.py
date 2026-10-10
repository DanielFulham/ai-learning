from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DirectoryItem:
    """One row of the server's directory listing."""

    name: str
    kind: Literal["file", "directory"]
    size: int
    modified: str
