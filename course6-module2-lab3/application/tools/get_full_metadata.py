from langchain.tools import BaseTool, tool

from interfaces.youtube_metadata_client_interface import YouTubeMetadataClientInterface


def make_get_full_metadata(client: YouTubeMetadataClientInterface) -> BaseTool:
    """Factory for the get_full_metadata tool."""

    @tool
    def get_full_metadata(url: str) -> dict:
        """
        Extract metadata for a YouTube URL: title, views, duration, channel, likes, comments, chapters.

        Args:
            url: A YouTube video URL.

        Returns:
            Dict with keys: title, views, duration, channel, likes, comments, chapters.
            Returns a dict with an 'error' key if extraction fails.
        """
        try:
            return client.get_metadata(url)
        except Exception as e:
            return {"error": str(e)}

    return get_full_metadata  # type: ignore[return-value]