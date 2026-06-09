from unittest.mock import MagicMock

from application.tools.fetch_transcript import make_fetch_transcript
from interfaces.transcript_client_interface import TranscriptClientInterface


def test_returns_transcript_text_on_success() -> None:
    client = MagicMock(spec=TranscriptClientInterface)
    client.fetch.return_value = "the transcript text"

    tool = make_fetch_transcript(client)
    result = tool.invoke({"video_id": "abc123"})

    assert result == "the transcript text"
    client.fetch.assert_called_once_with("abc123", "en")


def test_passes_language_through() -> None:
    client = MagicMock(spec=TranscriptClientInterface)
    client.fetch.return_value = "el texto"

    tool = make_fetch_transcript(client)
    tool.invoke({"video_id": "abc123", "language": "es"})

    client.fetch.assert_called_once_with("abc123", "es")


def test_returns_error_string_on_exception() -> None:
    client = MagicMock(spec=TranscriptClientInterface)
    client.fetch.side_effect = ValueError("Subtitles disabled")

    tool = make_fetch_transcript(client)
    result = tool.invoke({"video_id": "abc123"})

    assert result == "Error: Subtitles disabled"


def test_tool_name() -> None:
    client = MagicMock(spec=TranscriptClientInterface)
    tool = make_fetch_transcript(client)
    assert tool.name == "fetch_transcript"