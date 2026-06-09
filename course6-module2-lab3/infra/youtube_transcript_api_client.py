from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from youtube_transcript_api import YouTubeTranscriptApi

from interfaces.transcript_client_interface import TranscriptClientInterface


class YouTubeTranscriptApiClient(TranscriptClientInterface):
    """Concrete client for youtube-transcript-api.

    Returns the transcript as a single space-joined string. Raises whatever
    the underlying library raises (TranscriptsDisabled, NoTranscriptFound,
    VideoUnavailable, etc.) — the application layer decides how to map
    those into tool-call error messages.

    Enforces a wall-clock timeout via concurrent.futures because the
    underlying library does not expose a timeout parameter directly.
    A hung HTTP call will raise TimeoutError after _TIMEOUT_SECONDS.
    """

    _TIMEOUT_SECONDS = 10

    def fetch(self, video_id: str, language: str = "en") -> str:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._fetch_blocking, video_id, language)
            try:
                return future.result(timeout=self._TIMEOUT_SECONDS)
            except FuturesTimeoutError as e:
                raise TimeoutError(
                    f"Transcript fetch for video_id={video_id} exceeded {self._TIMEOUT_SECONDS}s"
                ) from e

    def _fetch_blocking(self, video_id: str, language: str) -> str:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=[language])
        return " ".join(snippet.text for snippet in fetched.snippets)