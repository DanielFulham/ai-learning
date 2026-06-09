from typing import Union

from langchain.tools import BaseTool, tool

from interfaces.youtube_search_client_interface import YouTubeSearchClientInterface


def make_search_youtube(client: YouTubeSearchClientInterface) -> BaseTool:
    """Factory for the search_youtube tool."""

    @tool
    def search_youtube(query: str, max_results: int = 5) -> Union[list[dict], str]:
        """
        Search YouTube for videos matching the query.

        Args:
            query: The search term to look for on YouTube.
            max_results: Maximum results to return. Clamped to [1, 20] by the client. Default 5.

        Returns:
            List of dicts with video titles, IDs, and URLs:
            [{'title': 'Video Title', 'video_id': 'abc123', 'url': 'https://youtu.be/abc123'}, ...]
            Returns an error string starting with 'Error:' if search fails.
        """
        try:
            return client.search(query, max_results)
        except Exception as e:
            return f"Error: {str(e)}"

    return search_youtube  # type: ignore[return-value]