from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptInfo:
    """A prompt template the server offers."""

    name: str
    description: str
    argument_names: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class PromptMessage:
    """One message of a rendered prompt."""

    role: str
    text: str


@dataclass(frozen=True)
class RenderedPrompt:
    """A prompt template filled in with arguments."""

    description: str
    messages: tuple[PromptMessage, ...] = field(default_factory=tuple)
