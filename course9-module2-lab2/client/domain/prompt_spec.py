from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptArgument:
    """One argument a server prompt template takes."""

    name: str
    required: bool


@dataclass(frozen=True)
class PromptSpec:
    """A prompt template the server offers."""

    name: str
    arguments: tuple[PromptArgument, ...] = field(default_factory=tuple)
