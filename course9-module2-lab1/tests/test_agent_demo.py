from unittest.mock import AsyncMock, patch

import pytest

import agent_demo
from domain.conversation import AgentReply, ConversationTurn


def test_print_reply_shows_each_turn_and_names_tool_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    reply = AgentReply(
        turns=(
            ConversationTurn(role="human", text="TEST_PROMPT"),
            ConversationTurn(role="ai", text="", tool_names=("TEST_TOOL_A", "TEST_TOOL_B")),
            ConversationTurn(role="tool", text="TEST_RESULT"),
            ConversationTurn(role="ai", text="TEST_ANSWER"),
        )
    )

    agent_demo.print_reply(reply)

    assert capsys.readouterr().out.splitlines() == [
        "[HUMAN] TEST_PROMPT",
        "[AI] tool call: TEST_TOOL_A, TEST_TOOL_B",
        "[TOOL] TEST_RESULT",
        "[AI] TEST_ANSWER",
    ]


@pytest.mark.parametrize("transport", ["http", "stdio", "multi"])
def test_main_runs_the_chosen_transport(transport: str) -> None:
    with patch("agent_demo.run", new=AsyncMock()) as run:
        assert agent_demo.main([transport]) == 0
        run.assert_awaited_once_with(transport)


def test_main_rejects_an_unknown_transport() -> None:
    with pytest.raises(SystemExit):
        agent_demo.main(["TEST_UNKNOWN"])


def test_every_transport_has_a_prompt() -> None:
    assert set(agent_demo.PROMPTS) == {"http", "stdio", "multi"}
