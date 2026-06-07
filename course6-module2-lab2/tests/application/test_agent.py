from unittest.mock import MagicMock

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool

from application.agent import ToolCallingAgent


@tool
def echo(text: str) -> str:
    """Echo back the input."""
    return f"echoed: {text}"


def _make_llm_returning(*responses) -> MagicMock:
    """Build a mock LLM whose bind_tools().invoke() returns each response in turn."""
    bound = MagicMock()
    bound.invoke.side_effect = list(responses)
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = bound
    return llm


def test_no_tool_call_returns_direct_response():
    llm = _make_llm_returning(AIMessage(content="Hi there"))
    agent = ToolCallingAgent(llm, tools=[echo])

    assert agent.run("hello") == "Hi there"


def test_single_tool_call_dispatches_and_summarises():
    tool_call = {"name": "echo", "args": {"text": "hi"}, "id": "call_1", "type": "tool_call"}
    llm = _make_llm_returning(
        AIMessage(content="", tool_calls=[tool_call]),
        AIMessage(content="The echo was: echoed: hi"),
    )
    agent = ToolCallingAgent(llm, tools=[echo])

    result = agent.run("echo hi")

    assert result == "The echo was: echoed: hi"

    # Second invoke received the AIMessage + ToolMessage with the right correlation id
    second_call_args = llm.bind_tools.return_value.invoke.call_args_list[1][0][0]
    tool_msg = next(m for m in second_call_args if isinstance(m, ToolMessage))
    assert tool_msg.tool_call_id == "call_1"
    assert tool_msg.content == "echoed: hi"


def test_parallel_tool_calls_all_dispatched():
    calls = [
        {"name": "echo", "args": {"text": "a"}, "id": "call_1", "type": "tool_call"},
        {"name": "echo", "args": {"text": "b"}, "id": "call_2", "type": "tool_call"},
    ]
    llm = _make_llm_returning(
        AIMessage(content="", tool_calls=calls),
        AIMessage(content="done"),
    )