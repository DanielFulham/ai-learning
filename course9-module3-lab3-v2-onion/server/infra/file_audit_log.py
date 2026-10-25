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
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(entry) + "\n")
        except OSError as error:
            raise self._without_path(error, "written to") from None

    def read_text(self) -> str:
        try:
            return self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return ""
        except OSError as error:
            raise self._without_path(error, "read") from None

    def _without_path(self, error: OSError, action: str) -> OSError:
        """An error naming the log file and the kind of failure, but not where the file lives.

        The operating system's own message carries the absolute path, which would tell any
        client where the server keeps its state.
        """
        name = self._path.name
        return OSError(f"The audit log {name} could not be {action} ({type(error).__name__})")
