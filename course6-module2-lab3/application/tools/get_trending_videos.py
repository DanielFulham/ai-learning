from langchain.tools import BaseTool, tool

from interfaces.youtube_metadata_client_interface import YouTubeMetadataClientInterface


def make_get_trending_videos(client: YouTubeMetadataClientInterface) -> BaseTool:
    """Factory for the get_trending_videos tool."""

    @tool
    def get_trending_videos(region_code: str, max_results: int = 25) -> list[dict]:
        """
        Fetches currently trending YouTube videos for a specific region.

        NOTE: YouTube removed the global trending feed in 2024+. This tool
        currently returns an error for most regions. Production use should
        migrate to the YouTube Data API v3 (videos.list with chart=mostPopular).

        Args:
            region_code: 2-letter country code (e.g., "US", "IN", "GB").
            max_results: Maximum results to return. Clamped to [1, 50] by the client. Default 25.

        Returns:
            List of dicts with video details: title, video_id, url, channel, duration, view_count.
            Returns a list with a single error dict if extraction fails.
        """
        try:
            return client.get_trending(region_code, max_results)
        except Exception as e:
            return [{"error": f"Failed to fetch trending videos: {str(e)}"}]

    return get_trending_videos  # type: ignore[return-value]  # @tool returns StructuredTool, a BaseTool subtype