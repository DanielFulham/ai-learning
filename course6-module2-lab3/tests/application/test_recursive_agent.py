from unittest.mock import MagicMock

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from application.recursive_agent import RecursiveAgent


def _make_tool(name: str, return_value: str):
    @tool
    def _t(arg: str) -> str:
        """Test tool."""
        return f"{name}({arg})={return_value}"

    _t.name = name  # type: ignore[misc]
    return _t


def _make_llm(*responses: AIMessage) -> MagicMock:
    bound = MagicMock()
    bound.invoke.side_effect = list(responses)
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = bound
    return llm


def test_returns_final_content_when_no_tool_calls() -> None:
    llm = _make_llm(AIMessage(content="direct answer"))
    agent = RecursiveAgent(llm, [_make_tool("noop", "result")])

    assert agent.run("hello") == "direct answer"


def test_recurses_for_two_sequential_tool_calls() -> None:
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "step1", "args": {"arg": "in"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "step2", "args": {"arg": "in"}, "id": "call_2", "type": "tool_call"}],
        ),
        AIMessage(content="done"),
    )
    agent = RecursiveAgent(llm, [_make_tool("step1", "a"), _make_tool("step2", "b")])

    assert agent.run("multi-step") == "done"


def test_recurses_for_n_sequential_tool_calls() -> None:
    llm = _make_llm(
        *[
            AIMessage(
                content="",
                tool_calls=[{"name": "step", "args": {"arg": "in"}, "id": f"call_{i}", "type": "tool_call"}],
            )
            for i in range(5)
        ],
        AIMessage(content="five-step done"),
    )
    agent = RecursiveAgent(llm, [_make_tool("step", "x")])

    assert agent.run("query") == "five-step done"


def test_parallel_tool_calls_in_one_turn() -> None:
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "a", "args": {"arg": "in"}, "id": "call_1", "type": "tool_call"},
                {"name": "b", "args": {"arg": "in"}, "id": "call_2", "type": "tool_call"},
            ],
        ),
        AIMessage(content="parallel done"),
    )
    agent = RecursiveAgent(llm, [_make_tool("a", "x"), _make_tool("b", "y")])

    assert agent.run("do both") == "parallel done"