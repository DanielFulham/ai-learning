from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from application.mappers import to_agent_reply


def test_maps_roles_and_text_in_order() -> None:
    reply = to_agent_reply(
        [
            HumanMessage(content="TEST_PROMPT"),
            AIMessage(content="TEST_ANSWER"),
        ]
    )
    assert [(turn.role, turn.text) for turn in reply.turns] == [
        ("human", "TEST_PROMPT"),
        ("ai", "TEST_ANSWER"),
    ]


def test_ai_tool_calls_become_tool_names() -> None:
    reply = to_agent_reply(
        [
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "TEST_TOOL_A", "args": {}, "id": "1"},
                    {"name": "TEST_TOOL_B", "args": {}, "id": "2"},
                ],
            )
        ]
    )
    assert reply.turns[0].tool_names == ("TEST_TOOL_A", "TEST_TOOL_B")


def test_non_ai_messages_have_no_tool_names() -> None:
    reply = to_agent_reply(
        [
            HumanMessage(content="TEST_PROMPT"),
            ToolMessage(content="TEST_RESULT", tool_call_id="1"),
        ]
    )
    assert all(turn.tool_names == () for turn in reply.turns)


def test_text_of_a_tool_result_is_kept() -> None:
    reply = to_agent_reply([ToolMessage(content="TEST_RESULT", tool_call_id="1")])
    assert reply.turns[0].role == "tool"
    assert reply.turns[0].text == "TEST_RESULT"


def test_empty_conversation_maps_to_empty_reply() -> None:
    assert to_agent_reply([]).turns == ()
