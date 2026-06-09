from typing import TypedDict


class MetadataResult(TypedDict, total=False):
    """Metadata fields returned for a YouTube video.

    total=False because every field is genuinely optional at runtime —
    YouTube returns None for restricted videos, age-gated content,
    creator-disabled metrics, and live streams without final counts.
    The 'error' key is present only when extraction failed; consumers
    should check for it first.
    """

    title: str | None
    views: int | None
    duration: int | None
    channel: str | None
    likes: int | None
    comments: int | None
    chapters: list[dict]
    error: str