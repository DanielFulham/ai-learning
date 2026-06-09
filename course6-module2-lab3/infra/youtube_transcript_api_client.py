from youtube_transcript_api import YouTubeTranscriptApi

from interfaces.transcript_client_interface import TranscriptClientInterface


class YouTubeTranscriptApiClient(TranscriptClientInterface):
    """Concrete client for youtube-transcript-api.

    Returns the transcript as a single space-joined string. Raises whatever
    the underlying library raises (TranscriptsDisabled, NoTranscriptFound,
    VideoUnavailable, etc.) — the application layer decides how to map
    those into tool-call error messages.
    """

    def fetch(self, video_id: str, language: str = "en") -> str:
        api = YouTubeTranscriptApi()
        fetched = api.fetch(video_id, languages=[language])
        return " ".join(snippet.text for snippet in fetched.snippets)