from collections.abc import Sequence
from typing import Protocol

from domain.conversation import Message, ModelReply
from domain.tool_info import ToolInfo


class LlmChatModelInterface(Protocol):
    """A chat model that can ask for tools to be run."""

    async def complete(
        self, messages: Sequence[Message], tools: Sequence[ToolInfo]
    ) -> ModelReply | None:
        """Answer the conversation, or return None if the model gave no answer at all.

        With no tools offered, the model can only answer in text.
        """
        ...
