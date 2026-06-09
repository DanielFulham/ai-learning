from langchain.tools import BaseTool, tool

from interfaces.transcript_client_interface import TranscriptClientInterface


def make_fetch_transcript(client: TranscriptClientInterface) -> BaseTool:
    """Factory for the fetch_transcript tool. The transcript client is captured in
    the tool's closure, so the LLM only sees (video_id, language) as arguments.
    """

    @tool
    def fetch_transcript(video_id: str, language: str = "en") -> str:
        """
        Fetches the transcript of a YouTube video.

        Args:
            video_id: The YouTube video ID (e.g., "dQw4w9WgXcQ").
            language: Language code for the transcript (e.g., "en", "es").

        Returns:
            The transcript text or an error message starting with 'Error:'.
        """
        try:
            return client.fetch(video_id, language)
        except Exception as e:
            return f"Error: {str(e)}"

    return fetch_transcript  # type: ignore[return-value]