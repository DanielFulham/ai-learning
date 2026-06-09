from typing import Protocol

from domain.video_metadata import MetadataResult


class YouTubeMetadataClientInterface(Protocol):
    """Provides video metadata operations via yt-dlp."""

    def get_metadata(self, url: str) -> MetadataResult:
        ...

    def get_thumbnails(self, url: str) -> list[dict]:
        ...

    def get_trending(self, region_code: str, max_results: int = 25) -> list[dict]:
        ...