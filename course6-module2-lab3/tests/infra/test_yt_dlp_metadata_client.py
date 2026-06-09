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


# --- configuration ---


def test_socket_timeout_included_in_ydl_opts() -> None:
    """yt-dlp must be constructed with socket_timeout so hung connections don't block forever."""
    with patch("infra.yt_dlp_metadata_client.yt_dlp.YoutubeDL") as mock_ydl_cls:  # type: ignore[attr-defined]
        mock_instance = MagicMock()
        mock_instance.__enter__ = MagicMock(return_value=mock_instance)
        mock_instance.__exit__ = MagicMock(return_value=False)
        mock_instance.extract_info.return_value = None
        mock_ydl_cls.return_value = mock_instance

        client = YtDlpMetadataClient()
        client.get_metadata("https://youtu.be/abc")

        call_kwargs = mock_ydl_cls.call_args[0][0]
        assert "socket_timeout" in call_kwargs
        assert call_kwargs["socket_timeout"] > 0


def test_suppress_logs_true_sets_yt_dlp_log_level_to_error() -> None:
    """Default suppress_logs=True must raise yt_dlp logger to ERROR."""
    import logging

    logging.getLogger("yt_dlp").setLevel(logging.DEBUG)  # start permissive
    YtDlpMetadataClient(suppress_logs=True)
    assert logging.getLogger("yt_dlp").level == logging.ERROR


def test_suppress_logs_false_leaves_log_level_untouched() -> None:
    """suppress_logs=False must not mutate global yt-dlp logger state."""
    import logging

    logging.getLogger("yt_dlp").setLevel(logging.DEBUG)
    YtDlpMetadataClient(suppress_logs=False)
    assert logging.getLogger("yt_dlp").level == logging.DEBUG