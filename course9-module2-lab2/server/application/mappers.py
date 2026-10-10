from datetime import datetime

from domain.file_entry import FileEntry
from interfaces.file_store_interface import RawEntry


def to_file_entry(raw: RawEntry) -> FileEntry:
    """Turn the file store's plain values into a FileEntry."""
    return FileEntry(
        name=raw["name"],
        path=raw["path"],
        kind="directory" if raw["is_dir"] else "file",
        size=raw["size"],
        modified=datetime.fromtimestamp(raw["modified"]),
        created=datetime.fromtimestamp(raw["created"]),
    )


def to_listing_item(entry: FileEntry) -> dict[str, str | int]:
    """Turn a FileEntry into the item shape the directory resource has always returned."""
    return {
        "name": entry.name,
        "path": entry.path,
        "type": entry.kind,
        "size": entry.size,
        "modified": entry.modified.isoformat(),
        "created": entry.created.isoformat(),
    }
