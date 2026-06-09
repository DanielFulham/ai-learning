from unittest.mock import MagicMock, patch

from infra.yt_dlp_metadata_client import YtDlpMetadataClient


def _patched_ydl(info: dict | None):
    """Helper: patch yt_dlp.YoutubeDL so extract_info returns the given info."""
    ydl_instance = MagicMock()
    ydl_instance.extract_info.return_value = info
    ydl_instance.__enter__ = MagicMock(return_value=ydl_instance)
    ydl_instance.__exit__ = MagicMock(return_value=False)

    patcher = patch("infra.yt_dlp_metadata_client.yt_dlp.YoutubeDL", return_value=ydl_instance)
    return patcher, ydl_instance


# --- get_metadata ---


def test_metadata_extracts_expected_fields() -> None:
    info = {
        "title": "Test Title",
        "view_count": 1000,
        "duration": 600,
        "uploader": "Test Channel",
        "like_count": 50,
        "comment_count": 10,
        "chapters": None,
    }
    patcher, _ = _patched_ydl(info)
    with patcher:
        client = YtDlpMetadataClient()
        result = client.get_metadata("https://youtu.be/abc")

    assert result == {
        "title": "Test Title",
        "views": 1000,
        "duration": 600,
        "channel": "Test Channel",
        "likes": 50,
        "comments": 10,
        "chapters": [],
    }


def test_metadata_handles_none_info() -> None:
    patcher, _ = _patched_ydl(None)
    with patcher:
        client = YtDlpMetadataClient()
        result = client.get_metadata("https://youtu.be/abc")
    assert result == {"error": "Could not extract video info"}


def test_metadata_handles_none_chapters() -> None:
    patcher, _ = _patched_ydl({"chapters": None})
    with patcher:
        client = YtDlpMetadataClient()
        result = client.get_metadata("https://youtu.be/abc")
    assert result["chapters"] == []


# --- get_thumbnails ---


def test_thumbnails_extracts_with_resolution() -> None:
    info = {
        "thumbnails": [
            {"url": "https://t.jpg", "width": 1280, "height": 720},
            {"url": "https://t2.jpg", "width": 640, "height": 360},
        ]
    }
    patcher, _ = _patched_ydl(info)
    with patcher:
        client = YtDlpMetadataClient()
        result = client.get_thumbnails("https://youtu.be/abc")

    assert result == [
        {"url": "https://t.jpg", "width": 1280, "height": 720, "resolution": "1280x720"},
        {"url": "https://t2.jpg", "width": 640, "height": 360, "resolution": "640x360"},
    ]


def test_thumbnails_filters_entries_without_url() -> None:
    info = {"thumbnails": [{"width": 1280}, {"url": "https://t.jpg"}]}
    patcher, _ = _patched_ydl(info)
    with patcher:
        client = YtDlpMetadataClient()
        result = client.get_thumbnails("https://youtu.be/abc")
    assert len(result) == 1
    assert result[0]["url"] == "https://t.jpg"


# --- get_trending ---


def test_trending_returns_entries() -> None:
    info = {
        "entries": [
            {
                "title": "Video 1",
                "id": "id1",
                "url": "https://youtu.be/id1",
                "uploader": "Channel A",
                "duration": 300,
                "view_count": 1000,
            },
        ]
    }
    patcher, _ = _patched_ydl(info)
    with patcher:
        client = YtDlpMetadataClient()
        result = client.get_trending("US")

    assert result == [
        {
            "title": "Video 1",
            "video_id": "id1",
            "url": "https://youtu.be/id1",
            "channel": "Channel A",
            "duration": 300,
            "view_count": 1000,
        }
    ]


def test_trending_handles_empty_entries() -> None:
    patcher, _ = _patched_ydl({"entries": []})
    with patcher:
        client = YtDlpMetadataClient()
        result = client.get_trending("US")
    assert result == [{"error": "No trending videos returned"}]


def test_trending_clamps_max_results() -> None:
    info = {"entries": [{"title": f"V{i}", "id": f"id{i}"} for i in range(100)]}
    patcher, _ = _patched_ydl(info)
    with patcher:
        client = YtDlpMetadataClient()
        result = client.get_trending("US", max_results=999)
    assert len(result) == 50