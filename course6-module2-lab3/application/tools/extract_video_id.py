import re

from langchain.tools import BaseTool, tool


def make_extract_video_id() -> BaseTool:
    """Factory for the extract_video_id tool. No infra dependency — pure regex.

    Returned as a factory for consistency with the other tool factories so the
    container builds them all the same way.
    """

    @tool
    def extract_video_id(url: str) -> str:
        """
        Extracts the 11-character YouTube video ID from a URL.

        Supports watch URLs, youtu.be short links, embeds, shorts, and live URLs.
        Query strings and timestamps are ignored.

        Args:
            url: A YouTube URL containing a video ID.

        Returns:
            Extracted video ID, or 'Error: Invalid YouTube URL' if parsing fails.
        """
        pattern = (
            r"(?:youtube\.com/(?:watch\?(?:[^&]*&)*v=|embed/|shorts/|live/|v/)"
            r"|youtu\.be/)"
            r"([a-zA-Z0-9_-]{11})"
        )
        match = re.search(pattern, url)
        return match.group(1) if match else "Error: Invalid YouTube URL"

    return extract_video_id