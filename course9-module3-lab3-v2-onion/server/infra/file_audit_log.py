import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from interfaces.audit_log_interface import AuditLogInterface


def _utc_now() -> datetime:
    return datetime.now(UTC)


class FileAuditLog(AuditLogInterface):
    """Appends one JSON object per line to a text file.

    JSON escapes newlines inside a value, so a tool name, argument or detail containing a line
    break still produces exactly one line and cannot add a forged entry.
    """

    def __init__(self, path: Path, clock: Callable[[], datetime] = _utc_now) -> None:
        self._path = path
        self._clock = clock

    def append(self, tool: str, arguments_json: str, event: str, detail: str) -> None:
        entry = {
            "time": self._clock().isoformat(),
            "tool": tool,
            "arguments": arguments_json,
            "event": event,
            "detail": detail,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")

    def read_text(self) -> str:
        if not self._path.exists():
            return ""
        return self._path.read_text(encoding="utf-8")
