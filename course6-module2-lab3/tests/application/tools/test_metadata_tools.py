from unittest.mock import MagicMock

from application.tools.get_full_metadata import make_get_full_metadata
from application.tools.get_thumbnails import make_get_thumbnails
from application.tools.get_trending_videos import make_get_trending_videos
from interfaces.youtube_metadata_client_interface import YouTubeMetadataClientInterface


# --- get_full_metadata ---


def test_metadata_returns_dict_on_success() -> None:
    client = MagicMock(spec=YouTubeMetadataClientInterface)
    client.get_metadata.return_value = {"title": "Test Title", "views": 100}

    tool = make_get_full_metadata(client)
    result = tool.invoke({"url": "https://youtu.be/abc"})

    assert result == {"title": "Test Title", "views": 100}
    client.get_metadata.assert_called_once_with("https://youtu.be/abc")


def test_metadata_returns_error_dict_on_exception() -> None:
    client = MagicMock(spec=YouTubeMetadataClientInterface)
    client.get_metadata.side_effect = RuntimeError("video unavailable")

    tool = make_get_full_metadata(client)
    result = tool.invoke({"url": "https://youtu.be/abc"})

    assert result == {"error": "video unavailable"}


# --- get_thumbnails ---


def test_thumbnails_returns_list_on_success() -> None:
    client = MagicMock(spec=YouTubeMetadataClientInterface)
    client.get_thumbnails.return_value = [{"url": "https://i.ytimg.com/x.jpg"}]

    tool = make_get_thumbnails(client)
    result = tool.invoke({"url": "https://youtu.be/abc"})

    assert result == [{"url": "https://i.ytimg.com/x.jpg"}]


def test_thumbnails_returns_error_list_on_exception() -> None:
    client = MagicMock(spec=YouTubeMetadataClientInterface)
    client.get_thumbnails.side_effect = RuntimeError("fetch failed")

    tool = make_get_thumbnails(client)
    result = tool.invoke({"url": "https://youtu.be/abc"})

    assert result == [{"error": "Failed to get thumbnails: fetch failed"}]


# --- get_trending_videos ---


def test_trending_returns_list_on_success() -> None:
    client = MagicMock(spec=YouTubeMetadataClientInterface)
    client.get_trending.return_value = [{"title": "Trending Video", "video_id": "x"}]

    tool = make_get_trending_videos(client)
    result = tool.invoke({"region_code": "US"})

    assert result == [{"title": "Trending Video", "video_id": "x"}]
    client.get_trending.assert_called_once_with("US", 25)


def test_trending_returns_error_list_on_exception() -> None:
    client = MagicMock(spec=YouTubeMetadataClientInterface)
    client.get_trending.side_effect = RuntimeError("trending feed gone")

    tool = make_get_trending_videos(client)
    result = tool.invoke({"region_code": "US"})

    assert result == [{"error": "Failed to fetch trending videos: trending feed gone"}]

def test_metadata_handles_none_values_in_fields() -> None:
    """The contract is that any field may be None — likes/comments often disabled."""
    client = MagicMock(spec=YouTubeMetadataClientInterface)
    client.get_metadata.return_value = {
        "title": "Test Video",
        "views": 1000,
        "duration": 600,
        "channel": "Test Channel",
        "likes": None,  # creator disabled likes
        "comments": None,  # creator disabled comments
        "chapters": [],
    }

    tool = make_get_full_metadata(client)
    result = tool.invoke({"url": "https://youtu.be/abc"})

    assert result["likes"] is None
    assert result["comments"] is None
    # The tool must not crash on None values, must not invent defaults