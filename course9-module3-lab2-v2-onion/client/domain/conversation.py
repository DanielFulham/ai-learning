from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ToolCall:
    """The model's request to run one tool. `arguments_json` is the text the model wrote."""

    id: str
    name: str
    arguments_json: str


@dataclass(frozen=True)
class Message:
    """One turn of the conversation with the model.

    An assistant turn may carry tool calls; a tool turn carries the result of one call.
    """

    role: Literal["user", "assistant", "tool"]
    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
    tool_call_id: str | None = None


@dataclass(frozen=True)
class ToolActivity:
    """One tool the model had run for an answer: its name, the arguments it wrote, the result."""

    name: str
    arguments_json: str
    result: str


@dataclass(frozen=True)
class ChatReply:
    """What the host gives back for a message: the answer, and the tools run to get it."""

    text: str
    tool_activity: tuple[ToolActivity, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ModelReply:
    """The model's answer: text (none if it only asked for tools) and any tool calls."""

    text: str | None
    tool_calls: tuple[ToolCall, ...] = field(default_factory=tuple)
