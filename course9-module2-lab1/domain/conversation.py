from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationTurn:
    """One message in an agent run, reduced to plain values.

    `tool_names` lists the tools an AI message asked to call, so a caller can
    check from the turns alone that a tool was actually used.
    """

    role: str
    text: str
    tool_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentReply:
    """The full conversation from one agent run."""

    turns: tuple[ConversationTurn, ...]

    @property
    def text(self) -> str:
        """The final turn's text, or an empty string if there were no turns."""
        return self.turns[-1].text if self.turns else ""
