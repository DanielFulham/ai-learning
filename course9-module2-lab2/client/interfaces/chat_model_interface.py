from collections.abc import Sequence
from typing import Protocol

from domain.conversation import Message, ModelReply, ToolSpec


class ChatModelInterface(Protocol):
    """A chat model that can ask for tools to be run."""

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolSpec]
    ) -> ModelReply: ...
