from domain.directory_item import DirectoryItem


def format_progress(progress: float, total: float | None, message: str | None) -> str:
    if total is not None and total != 0:
        return f"Progress: {progress / total * 100:.1f}% - {message or ''}"
    return f"Progress: {progress} - {message or ''}"


def format_directory_listing(items: list[DirectoryItem]) -> str:
    lines = [
        "\nDirectory Listing:\n",
        f"{'Type':<10} {'Size':>10} {'Modified':<25} {'Name'}",
        "-" * 70,
    ]
    for item in items:
        icon = "📁" if item.kind == "directory" else "📄"
        size = f"{item.size} B"
        lines.append(f"{icon:<2} {item.kind:<8} {size:>10}  {item.modified:<25} {item.name}")
    return "\n".join(lines)
