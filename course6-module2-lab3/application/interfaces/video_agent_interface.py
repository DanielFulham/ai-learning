from typing import Protocol


class VideoAgentInterface(Protocol):
    """Answers natural-language queries about YouTube videos using tool calls."""

    def run(self, query: str) -> str:
        ...