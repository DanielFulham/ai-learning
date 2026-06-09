from unittest.mock import MagicMock

from langchain.tools import tool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from application.manual_loop_agent import ManualLoopAgent


def _make_tool(name: str, return_value: str):
    @tool
    def _t(arg: str) -> str:
        """Test tool."""
        return f"{name}({arg})={return_value}"

    _t.name = name  # type: ignore[misc]
    return _t


def _make_llm(*responses: AIMessage) -> MagicMock:
    """Build a mocked LLM whose bind_tools().invoke() returns the given AIMessages in order."""
    bound = MagicMock()
    bound.invoke.side_effect = list(responses)
    llm = MagicMock(spec=BaseChatModel)
    llm.bind_tools.return_value = bound
    return llm


def test_returns_final_content_when_no_tool_calls() -> None:
    llm = _make_llm(AIMessage(content="direct answer"))
    agent = ManualLoopAgent(llm, [_make_tool("noop", "result")])

    result = agent.run("hello")

    assert result == "direct answer"


def test_dispatches_single_tool_call_then_summarises() -> None:
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "extract_id", "args": {"arg": "url"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="final summary"),
    )
    agent = ManualLoopAgent(llm, [_make_tool("extract_id", "abc123")])

    result = agent.run("get the id")

    assert result == "final summary"


def test_dispatches_two_sequential_tool_calls() -> None:
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "step1", "args": {"arg": "input"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "step2", "args": {"arg": "input"}, "id": "call_2", "type": "tool_call"}],
        ),
        AIMessage(content="done"),
    )
    agent = ManualLoopAgent(llm, [_make_tool("step1", "a"), _make_tool("step2", "b")])

    result = agent.run("multi-step query")

    assert result == "done"


def test_dispatches_parallel_tool_calls_in_one_turn() -> None:
    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "step1", "args": {"arg": "input"}, "id": "call_1", "type": "tool_call"},
                {"name": "step2", "args": {"arg": "input"}, "id": "call_2", "type": "tool_call"},
            ],
        ),
        AIMessage(content="parallel done"),
    )
    agent = ManualLoopAgent(llm, [_make_tool("step1", "a"), _make_tool("step2", "b")])

    result = agent.run("do both")

    assert result == "parallel done"


def test_tool_exception_returns_error_message_to_llm() -> None:
    failing_tool = _make_tool("failing", "irrelevant")
    failing_tool.func = MagicMock(side_effect=RuntimeError("boom"))  # type: ignore[attr-defined]

    llm = _make_llm(
        AIMessage(
            content="",
            tool_calls=[{"name": "failing", "args": {"arg": "input"}, "id": "call_1", "type": "tool_call"}],
        ),
        AIMessage(content="recovered"),
    )
    agent = ManualLoopAgent(llm, [failing_tool])

    result = agent.run("query")

    assert result == "recovered"