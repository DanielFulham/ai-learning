from application.formatters import format_directory_listing, format_progress
from domain.directory_item import DirectoryItem


def test_progress_with_total_shows_percentage() -> None:
    assert format_progress(5, 20, "TEST_MSG") == "Progress: 25.0% - TEST_MSG"


def test_progress_with_zero_total_does_not_divide() -> None:
    assert format_progress(0, 0, "Write complete") == "Progress: 0 - Write complete"


def test_progress_without_total_shows_raw_value() -> None:
    assert format_progress(5, None, None) == "Progress: 5 - "


def test_directory_listing_has_header_and_rows() -> None:
    items = [
        DirectoryItem("a.txt", "file", 3, "TEST_TIME"),
        DirectoryItem("sub", "directory", 0, "TEST_TIME"),
    ]
    text = format_directory_listing(items)
    assert "Directory Listing:" in text
    assert "a.txt" in text
    assert "3 B" in text
    assert "📁" in text
    assert "📄" in text
