from dataclasses import dataclass


@dataclass(frozen=True)
class ToolInfo:
    """A tool the server offers."""

    name: str
    description: str


@dataclass(frozen=True)
class ToolCallResult:
    """What a tool call returned: the text parts, and whether the server flagged an error."""

    texts: tuple[str, ...]
    is_error: bool
