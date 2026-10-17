from dataclasses import dataclass


@dataclass(frozen=True)
class HistoryEntry:
    """One operation the client has done, for the `history` command."""

    operation: str
    name: str
