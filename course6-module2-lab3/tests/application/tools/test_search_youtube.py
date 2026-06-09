from unittest.mock import MagicMock

from application.tools.search_youtube import make_search_youtube
from interfaces.youtube_search_client_interface import YouTubeSearchClientInterface


def test_returns_search_results_on_success() -> None:
    client = MagicMock(spec=YouTubeSearchClientInterface)
    client.search.return_value = [
        {"title": "Test Video", "video_id": "abc", "url": "https://youtu.be/abc"},
    ]

    tool = make_search_youtube(client)
    result = tool.invoke({"query": "test query"})

    assert result == [
        {"title": "Test Video", "video_id": "abc", "url": "https://youtu.be/abc"},
    ]
    client.search.assert_called_once_with("test query", 5)


def test_passes_max_results_through() -> None:
    client = MagicMock(spec=YouTubeSearchClientInterface)
    client.search.return_value = []

    tool = make_search_youtube(client)
    tool.invoke({"query": "test", "max_results": 10})

    client.search.assert_called_once_with("test", 10)


def test_returns_error_string_on_exception() -> None:
    client = MagicMock(spec=YouTubeSearchClientInterface)
    client.search.side_effect = RuntimeError("rate limited")

    tool = make_search_youtube(client)
    result = tool.invoke({"query": "test"})

    assert result == "Error: rate limited"


def test_tool_name() -> None:
    client = MagicMock(spec=YouTubeSearchClientInterface)
    tool = make_search_youtube(client)
    assert tool.name == "search_youtube"