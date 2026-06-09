from langchain.tools import BaseTool, tool

from interfaces.youtube_metadata_client_interface import YouTubeMetadataClientInterface


def make_get_thumbnails(client: YouTubeMetadataClientInterface) -> BaseTool:
    """Factory for the get_thumbnails tool."""

    @tool
    def get_thumbnails(url: str) -> list[dict]:
        """
        Get available thumbnails for a YouTube video using its URL.

        Args:
            url: YouTube video URL (any format).

        Returns:
            List of dicts with thumbnail url, width, height, and resolution.
            Returns a list with a single error dict if extraction fails.
        """
        try:
            return client.get_thumbnails(url)
        except Exception as e:
            return [{"error": f"Failed to get thumbnails: {str(e)}"}]

    return get_thumbnails  # type: ignore[return-value]  # @tool returns StructuredTool, a BaseTool subtype