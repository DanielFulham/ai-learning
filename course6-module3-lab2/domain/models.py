from dataclasses import dataclass
from typing import Mapping, TypeAlias


JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)

ToolCallArgs: TypeAlias = Mapping[str, JsonValue]


@dataclass(frozen=True)
class ToolCallRecord:
    tool_call_id: str
    tool_name: str
    args: ToolCallArgs
    result: str
    duration_ms: float


class AgentTrace:
    def __init__(self) -> None:
        self._records: list[ToolCallRecord] = []

    def append(self, record: ToolCallRecord) -> None:
        self._records.append(record)

    @property
    def tool_calls(self) -> tuple[ToolCallRecord, ...]:
        return tuple(self._records)

    def __len__(self) -> int:
        return len(self._records)