from unittest.mock import MagicMock, patch

import pytest

from infra.youtube_transcript_api_client import YouTubeTranscriptApiClient


def test_fetch_joins_snippets_into_single_string() -> None:
    snippet1 = MagicMock()
    snippet1.text = "first"
    snippet2 = MagicMock()
    snippet2.text = "second"
    fetched = MagicMock()
    fetched.snippets = [snippet1, snippet2]

    with patch("infra.youtube_transcript_api_client.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.fetch.return_value = fetched
        mock_api.return_value = mock_instance

        client = YouTubeTranscriptApiClient()
        result = client.fetch("abc123")

        assert result == "first second"
        mock_instance.fetch.assert_called_once_with("abc123", languages=["en"])


def test_fetch_passes_language_through() -> None:
    fetched = MagicMock()
    fetched.snippets = []

    with patch("infra.youtube_transcript_api_client.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.fetch.return_value = fetched
        mock_api.return_value = mock_instance

        client = YouTubeTranscriptApiClient()
        client.fetch("abc123", language="es")

        mock_instance.fetch.assert_called_once_with("abc123", languages=["es"])


def test_fetch_raises_when_underlying_raises() -> None:
    with patch("infra.youtube_transcript_api_client.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.fetch.side_effect = ValueError("transcripts disabled")
        mock_api.return_value = mock_instance

        client = YouTubeTranscriptApiClient()
        with pytest.raises(ValueError, match="transcripts disabled"):
            client.fetch("abc123")


@pytest.mark.slow
def test_fetch_raises_timeout_error_when_hung(blocking_barrier) -> None:
    """A hung network call must raise TimeoutError, not block the thread indefinitely."""
    def slow_fetch(*args, **kwargs):
        blocking_barrier.wait(timeout=30)
        return MagicMock(snippets=[])

    with patch("infra.youtube_transcript_api_client.YouTubeTranscriptApi") as mock_api:
        mock_instance = MagicMock()
        mock_instance.fetch.side_effect = slow_fetch
        mock_api.return_value = mock_instance

        client = YouTubeTranscriptApiClient()
        client._TIMEOUT_SECONDS = 0.05  # type: ignore[attr-defined]

        with pytest.raises(TimeoutError, match="exceeded"):
            client.fetch("abc123")