from typing import Protocol


class TranscriptClientInterface(Protocol):
    """Fetches transcripts for YouTube videos. Returns transcript text or raises."""

    def fetch(self, video_id: str, language: str = "en") -> str:
        ...