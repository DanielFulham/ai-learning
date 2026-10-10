from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage

from domain.conversation import AgentReply, ConversationTurn


def to_agent_reply(messages: Sequence[BaseMessage]) -> AgentReply:
    """Map framework messages to a domain AgentReply."""
    turns = tuple(
        ConversationTurn(
            role=message.type,
            text=message.text,
            tool_names=(
                tuple(call["name"] for call in message.tool_calls)
                if isinstance(message, AIMessage)
                else ()
            ),
        )
        for message in messages
    )
    return AgentReply(turns=turns)
