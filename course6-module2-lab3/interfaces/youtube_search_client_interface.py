from typing import Protocol


class YouTubeSearchClientInterface(Protocol):
    """Searches YouTube for videos matching a query."""

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        ...