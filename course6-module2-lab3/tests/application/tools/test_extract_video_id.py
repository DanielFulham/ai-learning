import pytest

from application.tools.extract_video_id import make_extract_video_id


tool = make_extract_video_id()


@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ?t=42", "dQw4w9WgXcQ"),
        ("https://youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/watch?feature=share&v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtube.com/v/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ],
)
def test_extracts_video_id_from_valid_urls(url: str, expected: str) -> None:
    assert tool.invoke(url) == expected


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/be/dQw4w9WgXcQ",
        "not a url at all",
        "https://youtube.com/watch",
        "",
    ],
)
def test_returns_error_for_invalid_urls(url: str) -> None:
    assert tool.invoke(url) == "Error: Invalid YouTube URL"


def test_tool_name() -> None:
    assert tool.name == "extract_video_id"


def test_tool_has_description() -> None:
    assert tool.description and len(tool.description) > 0