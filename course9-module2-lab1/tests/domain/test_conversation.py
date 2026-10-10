import dataclasses

import pytest

from domain.conversation import AgentReply, ConversationTurn


def test_turn_defaults_to_no_tool_names() -> None:
    turn = ConversationTurn(role="human", text="TEST_PROMPT")
    assert turn.tool_names == ()


def test_turn_is_immutable() -> None:
    turn = ConversationTurn(role="human", text="TEST_PROMPT")
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(turn, "text", "changed")


def test_reply_text_is_the_final_turn() -> None:
    reply = AgentReply(
        turns=(
            ConversationTurn(role="human", text="TEST_PROMPT"),
            ConversationTurn(role="ai", text="TEST_ANSWER"),
        )
    )
    assert reply.text == "TEST_ANSWER"


def test_reply_text_is_empty_when_there_are_no_turns() -> None:
    assert AgentReply(turns=()).text == ""
