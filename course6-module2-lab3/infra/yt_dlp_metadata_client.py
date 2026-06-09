import logging
from typing import cast

import yt_dlp


from interfaces.youtube_metadata_client_interface import YouTubeMetadataClientInterface


class _YtDlpLoggerAdapter:
    """Adapts stdlib logging.Logger to yt-dlp's expected logger protocol.

    yt-dlp's stubs declare a logger protocol that stdlib Logger doesn't satisfy.
    This adapter bridges the two cleanly, keeping the type-ignore footprint
    contained to one file at the boundary.
    """

    def __init__(self, ydl: "yt_dlp.YoutubeDL | None" = None) -> None:
        # ydl parameter accepted to match yt-dlp logger protocol signature; unused here
        self._logger = logging.getLogger("yt_dlp")

    def debug(self, message: str) -> None:
        self._logger.debug(message)

    def info(self, message: str) -> None:
        self._logger.info(message)

    def warning(self, message: str) -> None:
        self._logger.warning(message)

    def error(self, message: str) -> None:
        self._logger.error(message)

class YtDlpMetadataClient(YouTubeMetadataClientInterface):
    """Concrete client for yt-dlp metadata operations.

    Shared across get_metadata, get_thumbnails, and get_trending because all
    three use the same underlying YoutubeDL machinery and the same logger
    adapter. Raises whatever yt-dlp raises; the application layer decides
    how to map those into tool-call error messages.
    """

    def __init__(self) -> None:
        self._logger = _YtDlpLoggerAdapter()
        logging.getLogger("yt_dlp").setLevel(logging.ERROR)

    def _extract(self, url: str, extra_opts: dict | None = None) -> dict | None:
        opts = {"quiet": True, "logger": self._logger}
        if extra_opts:
            opts.update(extra_opts)
        with yt_dlp.YoutubeDL(opts) as ydl:  # type: ignore[arg-type, no-untyped-call]
            info = ydl.extract_info(url, download=False)
            return cast(dict | None, info)

    def get_metadata(self, url: str) -> dict:
        info = self._extract(url)
        if info is None:
            return {"error": "Could not extract video info"}
        return {
            "title": info.get("title"),
            "views": info.get("view_count"),
            "duration": info.get("duration"),
            "channel": info.get("uploader"),
            "likes": info.get("like_count"),
            "comments": info.get("comment_count"),
            "chapters": info.get("chapters") or [],
        }

    def get_thumbnails(self, url: str) -> list[dict]:
        info = self._extract(url)
        if info is None:
            return [{"error": "Could not extract video info"}]
        thumbnails = info.get("thumbnails") or []
        return [
            {
                "url": t["url"],
                "width": t.get("width"),
                "height": t.get("height"),
                "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip("x"),
            }
            for t in thumbnails
            if "url" in t
        ]

    def get_trending(self, region_code: str, max_results: int = 25) -> list[dict]:
        max_results = max(1, min(max_results, 50))
        extra = {
            "geo_bypass_country": region_code.upper(),
            "extract_flat": True,
            "force_generic_extractor": True,
        }
        info = self._extract("https://www.youtube.com/feed/trending", extra_opts=extra)
        if info is None:
            return [{"error": "No trending videos returned"}]
        entries = info.get("entries")
        if not entries:
            return [{"error": "No trending videos returned"}]
        return [
            {
                "title": entry.get("title"),
                "video_id": entry.get("id"),
                "url": entry.get("url"),
                "channel": entry.get("uploader"),
                "duration": entry.get("duration"),
                "view_count": entry.get("view_count"),
            }
            for entry in entries[:max_results]
        ]