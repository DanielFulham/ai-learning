from datetime import datetime

from application.mappers import to_file_entry, to_listing_item
from domain.file_entry import FileEntry
from interfaces.file_store_interface import RawEntry


def _make_raw(is_dir: bool = False) -> RawEntry:
    return {
        "name": "TEST_NAME",
        "path": "TEST_DIR/TEST_NAME",
        "is_dir": is_dir,
        "size": 42,
        "modified": 1_700_000_000.0,
        "created": 1_600_000_000.0,
    }


def test_to_file_entry_copies_name_path_and_size() -> None:
    entry = to_file_entry(_make_raw())
    assert (entry.name, entry.path, entry.size) == ("TEST_NAME", "TEST_DIR/TEST_NAME", 42)


def test_to_file_entry_names_the_kind_from_is_dir() -> None:
    assert to_file_entry(_make_raw(is_dir=False)).kind == "file"
    assert to_file_entry(_make_raw(is_dir=True)).kind == "directory"


def test_to_file_entry_turns_timestamps_into_local_datetimes() -> None:
    entry = to_file_entry(_make_raw())
    assert entry.modified == datetime.fromtimestamp(1_700_000_000.0)
    assert entry.created == datetime.fromtimestamp(1_600_000_000.0)


def test_to_listing_item_keeps_the_labs_key_names_and_iso_times() -> None:
    entry = FileEntry(
        name="TEST_NAME",
        path="TEST_DIR/TEST_NAME",
        kind="directory",
        size=7,
        modified=datetime(2026, 1, 2, 3, 4, 5),
        created=datetime(2025, 6, 7, 8, 9, 10),
    )
    assert to_listing_item(entry) == {
        "name": "TEST_NAME",
        "path": "TEST_DIR/TEST_NAME",
        "type": "directory",
        "size": 7,
        "modified": "2026-01-02T03:04:05",
        "created": "2025-06-07T08:09:10",
    }
