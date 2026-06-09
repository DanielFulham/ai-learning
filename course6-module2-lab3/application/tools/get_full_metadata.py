from langchain.tools import BaseTool, tool

from domain.video_metadata import MetadataResult
from interfaces.youtube_metadata_client_interface import YouTubeMetadataClientInterface


def make_get_full_metadata(client: YouTubeMetadataClientInterface) -> BaseTool:
    """Factory for the get_full_metadata tool."""

    @tool
    def get_full_metadata(url: str) -> MetadataResult:
        """
        Extract metadata for a YouTube URL: title, views, duration, channel, likes, comments, chapters.

        Args:
            url: A YouTube video URL.

        Returns:
            Dict with keys: title, views, duration, channel, likes, comments, chapters.
            Any individual value may be None — YouTube returns null for restricted
            videos, age-gated content, creator-disabled metrics (likes/comments
            often hidden), and live streams without final counts. Always check
            for None before using a value in a sentence.

            On extraction failure, returns a dict with a single 'error' key
            instead of the metadata fields. Check for 'error' first.
        """
        try:
            return client.get_metadata(url)
        except Exception as e:
            return {"error": str(e)}

    return get_full_metadata  # type: ignore[return-value]  # @tool wraps the return type as BaseTool