from langchain.tools import BaseTool, tool

from interfaces.youtube_search_client_interface import YouTubeSearchClientInterface


def make_search_youtube(client: YouTubeSearchClientInterface) -> BaseTool:
    """Factory for the search_youtube tool."""

    @tool
    def search_youtube(query: str, max_results: int = 5) -> list[dict]:
        """
        Search YouTube for videos matching the query.

        Args:
            query: The search term to look for on YouTube.
            max_results: Maximum results to return. Clamped to [1, 20] by the client. Default 5.

        Returns:
            List of dicts with video titles, IDs, and URLs:
            [{'title': 'Video Title', 'video_id': 'abc123', 'url': 'https://youtu.be/abc123'}, ...]
            On error, returns a list with a single error dict: [{'error': '...'}].
        """
        try:
            return client.search(query, max_results)
        except Exception as e:
            return [{"error": f"Search failed: {str(e)}"}]

    return search_youtube