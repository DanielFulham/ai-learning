import pytest
from unittest.mock import MagicMock, patch

from infra.pytubefix_search_client import PytubefixSearchClient


def _make_video(title: str, video_id: str) -> MagicMock:
    video = MagicMock()
    video.title = title
    video.video_id = video_id
    return video


def test_search_returns_shaped_dicts() -> None:
    with patch("infra.pytubefix_search_client.Search") as mock_search:
        mock_search.return_value.videos = [
            _make_video("Video 1", "id1"),
            _make_video("Video 2", "id2"),
        ]

        client = PytubefixSearchClient()
        result = client.search("test query")

        assert result == [
            {"title": "Video 1", "video_id": "id1", "url": "https://youtu.be/id1"},
            {"title": "Video 2", "video_id": "id2", "url": "https://youtu.be/id2"},
        ]


def test_search_clamps_max_results_high() -> None:
    with patch("infra.pytubefix_search_client.Search") as mock_search:
        mock_search.return_value.videos = [_make_video(f"V{i}", f"id{i}") for i in range(30)]

        client = PytubefixSearchClient()
        result = client.search("test", max_results=999)

        assert len(result) == 20


def test_search_clamps_max_results_low() -> None:
    with patch("infra.pytubefix_search_client.Search") as mock_search:
        mock_search.return_value.videos = [_make_video(f"V{i}", f"id{i}") for i in range(5)]

        client = PytubefixSearchClient()
        result = client.search("test", max_results=0)

        assert len(result) == 1


def test_search_default_max_results_is_five() -> None:
    with patch("infra.pytubefix_search_client.Search") as mock_search:
        mock_search.return_value.videos = [_make_video(f"V{i}", f"id{i}") for i in range(10)]

        client = PytubefixSearchClient()
        result = client.search("test")

        assert len(result) == 5


@pytest.mark.slow
def test_search_raises_timeout_error_when_hung(blocking_barrier) -> None:
    """A hung network call must raise TimeoutError, not block the thread indefinitely."""
    def slow_search(query):
        blocking_barrier.wait(timeout=30)
        return MagicMock(videos=[])

    with patch("infra.pytubefix_search_client.Search", side_effect=slow_search):
        client = PytubefixSearchClient()
        client._TIMEOUT_SECONDS = 0.05  # type: ignore[attr-defined]

        with pytest.raises(TimeoutError, match="exceeded"):
            client.search("test")