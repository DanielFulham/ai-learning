from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from pytubefix import Search

from interfaces.youtube_search_client_interface import YouTubeSearchClientInterface


class PytubefixSearchClient(YouTubeSearchClientInterface):
    """Concrete client for pytubefix Search.

    Returns a list of dicts shaped {title, video_id, url}. Clamps
    max_results to [1, 20] at the boundary so the LLM cannot accidentally
    request a runaway result set.

    Enforces a wall-clock timeout via concurrent.futures because the
    pytubefix version in use does not expose a timeout parameter
    directly. A hung HTTP call will raise TimeoutError after
    _TIMEOUT_SECONDS.
    """

    _TIMEOUT_SECONDS = 10

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        max_results = max(1, min(max_results, 20))
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._search_blocking, query, max_results)
            try:
                return future.result(timeout=self._TIMEOUT_SECONDS)
            except FuturesTimeoutError as e:
                raise TimeoutError(
                    f"YouTube search for query={query!r} exceeded {self._TIMEOUT_SECONDS}s"
                ) from e

    def _search_blocking(self, query: str, max_results: int) -> list[dict]:
        s = Search(query)
        return [
            {
                "title": yt.title,
                "video_id": yt.video_id,
                "url": f"https://youtu.be/{yt.video_id}",
            }
            for yt in s.videos[:max_results]
        ]