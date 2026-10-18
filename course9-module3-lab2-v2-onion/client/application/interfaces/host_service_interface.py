from typing import Protocol

from domain.conversation import ChatReply


class HostServiceInterface(Protocol):
    """What the host entry point needs from the client application: a chat that can be reset."""

    async def chat(self, user_message: str) -> ChatReply: ...

    def reset(self) -> None: ...
