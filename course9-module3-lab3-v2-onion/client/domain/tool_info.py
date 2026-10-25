from dataclasses import dataclass, field


@dataclass(frozen=True)
class ToolInfo:
    """A tool the server offers: its name, description and JSON Schema for the arguments."""

    name: str
    description: str
    input_schema: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallResult:
    """What a tool call returned: the text parts, and whether the server flagged an error."""

    texts: tuple[str, ...]
    is_error: bool
