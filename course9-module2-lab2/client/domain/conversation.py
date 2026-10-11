from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(frozen=True)
class ToolSpec:
    """A tool the server offers, as the model sees it."""

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """The model's request to run one tool."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    """The outcome of one tool call, sent back to the model."""

    tool_call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True)
class Message:
    """One turn of the conversation with the model."""

    role: Literal["user", "assistant"]
    text: str = ""
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_results: tuple[ToolResult, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelReply:
    """The model's answer: text, any tool calls to run, and whether it was cut off."""

    text: str
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    truncated: bool = False
