from typing import Protocol


class AuditLogInterface(Protocol):
    def append(self, tool: str, arguments_json: str, event: str, detail: str) -> None: ...

    def read_text(self) -> str: ...
