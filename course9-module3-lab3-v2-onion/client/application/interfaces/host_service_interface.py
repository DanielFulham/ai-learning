from collections.abc import Sequence
from typing import Protocol

from application.interfaces.approval_desk_interface import ApprovalDeskInterface
from domain.conversation import ChatReply, Message


class HostServiceInterface(Protocol):
    """What the host entry point needs from the client application: a chat that takes the
    conversation so far and the page session's approval desk, and returns the conversation,
    longer, with the answer."""

    async def chat(
        self, user_message: str, history: Sequence[Message], desk: ApprovalDeskInterface
    ) -> ChatReply: ...
