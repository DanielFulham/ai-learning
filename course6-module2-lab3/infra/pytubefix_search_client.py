from pytubefix import Search

from interfaces.youtube_search_client_interface import YouTubeSearchClientInterface


class PytubefixSearchClient(YouTubeSearchClientInterface):
    """Concrete client for pytubefix Search.

    Returns a list of dicts shaped {title, video_id, url}. Clamps
    max_results to [1, 20] at the boundary so the LLM cannot accidentally
    request a runaway result set.
    """

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        max_results = max(1, min(max_results, 20))
        s = Search(query)
        return [
            {
                "title": yt.title,
                "video_id": yt.video_id,
                "url": f"https://youtu.be/{yt.video_id}",
            }
            for yt in s.videos[:max_results]
        ]