from collections.abc import Sequence
from typing import Protocol

from domain.conversation import ChatReply, Message


class HostServiceInterface(Protocol):
    """What the host entry point needs from the client application: a chat that takes the
    conversation so far and returns it, longer, with the answer."""

    async def chat(self, user_message: str, history: Sequence[Message] = ()) -> ChatReply: ...
